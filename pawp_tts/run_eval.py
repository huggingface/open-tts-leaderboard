"""
TTS synthesis for the Open TTS Leaderboard (PAWP-TTS folded ONNX backend, stage 1).

PAWP-TTS (https://github.com/Firojpaudel/PAWP_TTS) is a compact bilingual English-Nepali
streaming TTS family (ConvNeXt1D + iSTFT, 24 kHz, all learned inference parameters counted
including the waveform decoder). Two fixed-voice English folded ONNX checkpoints are supported:

  * Firoj112/pawp-tts-1.5m   1,490,390 params (5.98 MB FP32, 5.16 MB folded EN ONNX, n_fft=1024)
  * Firoj112/pawp-tts-300k   281,078 unique / 328,158 materialized params (1.12 MB FP32,
                             1.32 MB folded EN ONNX, n_fft=512)

Inference runs on CPU via ONNX Runtime (`CPUExecutionProvider`) with zero PyTorch dependency,
using the repo's pure-Python deterministic English eSpeak-NG frontend and an inlined NumPy
iSTFT + bounded-halo streaming runner (60 ms first chunk for `--ttfa_probe`).
Stage 2 (`transformers/transcribe.py`) transcribes with Qwen3-ASR, UTMOS22 scores naturalness,
and stage 4 computes WER + RTFx (no stage 3 SIM: fixed single-speaker voice).
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import yaml
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from run_eval_utils import (
    add_common_args,
    load_done_entries,
    load_tts_dataset,
    manifest_entry,
    open_manifest,
    output_paths,
    pending_batches,
    print_next_steps,
    set_seed,
    warm_up,
    wav_rel_path,
    write_entry,
)

PINNED_REVISIONS: dict[str, str] = {
    "Firoj112/pawp-tts-1.5m": "a1f53764ba614f7a91639e07d47caaf11969db50",
    "Firoj112/pawp-tts-300k": "e7f3c94053ea88e9c17df1264475f4a41d90529e",
}


def _hann_istft_numpy(
    real_spec: np.ndarray,
    imag_spec: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: np.ndarray,
) -> np.ndarray:
    """Fast NumPy inverse STFT matching `torch.istft(..., center=True)`."""
    n_frames = real_spec.shape[1]
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    complex_spec = real_spec.astype(np.float32) + 1j * imag_spec.astype(np.float32)
    frames = np.fft.irfft(complex_spec, n=n_fft, axis=0).astype(np.float32)
    frames *= window[:, None]

    raw_len = (n_frames - 1) * hop_length + n_fft
    audio = np.zeros(raw_len, dtype=np.float32)
    win_env = np.zeros(raw_len, dtype=np.float32)
    win_sq = window * window

    for t in range(n_frames):
        offset = t * hop_length
        audio[offset : offset + n_fft] += frames[:, t]
        win_env[offset : offset + n_fft] += win_sq

    nonzero = win_env > 1e-8
    audio[nonzero] /= win_env[nonzero]
    pad = n_fft // 2
    if raw_len <= 2 * pad:
        return np.zeros(0, dtype=np.float32)
    return audio[pad : raw_len - pad].astype(np.float32)


def _expand_frames_numpy(
    enc_h: np.ndarray,
    durations: np.ndarray,
    prog_weight: np.ndarray,
    prog_bias: np.ndarray,
) -> np.ndarray:
    int_durs = np.maximum(np.round(durations[0]).astype(np.int64), 1)
    expanded = np.repeat(enc_h[0], int_durs, axis=0)
    prog_chunks = []
    for d in int_durs:
        steps = np.arange(d, dtype=np.float32)
        denom = float(max(d, 1))
        prog_chunks.append(np.stack([steps / denom, (d - 1 - steps) / denom], axis=-1))
    prog = np.concatenate(prog_chunks, axis=0)
    return (expanded + (prog @ prog_weight.T + prog_bias))[None, :, :].astype(np.float32)


class _PureOnnxStreamingRunner:
    """Zero-PyTorch CPU ONNX Runtime chunked streaming synthesizer."""

    def __init__(
        self,
        enc_path: Path,
        dec_path: Path,
        prog_path: Path,
        n_fft: int,
        hop_length: int,
        intra_op_threads: int = 4,
    ) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._enc_sess = ort.InferenceSession(
            str(enc_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._dec_sess = ort.InferenceSession(
            str(dec_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        prog_data = np.load(prog_path)
        self._prog_w = prog_data["weight"].astype(np.float32)
        self._prog_b = prog_data["bias"].astype(np.float32)
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._hann = np.hanning(self._n_fft + 1)[:-1].astype(np.float32)

    def stream_pcm(
        self,
        phoneme_ids: np.ndarray,
        lang_ids: np.ndarray,
        stress_ids: np.ndarray,
        boundary_flags: np.ndarray,
        first_chunk_frames: int = 6,
        chunk_frames: int = 16,
        left_halo: int = 24,
        right_halo: int = 11,
    ) -> Iterator[np.ndarray]:
        enc_h, pred_dur = self._enc_sess.run(
            None,
            {
                "phoneme_ids": phoneme_ids.astype(np.int64),
                "lang_ids": lang_ids.astype(np.int64),
                "stress_ids": stress_ids.astype(np.int64),
                "boundary_flags": boundary_flags.astype(np.float32),
            },
        )
        frame_h = _expand_frames_numpy(enc_h, pred_dur, self._prog_w, self._prog_b)
        total_frames = frame_h.shape[1]

        cursor = 0
        step = 0
        while cursor < total_frames:
            cur_size = first_chunk_frames if step == 0 else chunk_frames
            end = min(cursor + cur_size, total_frames)
            w_start = max(0, cursor - left_halo)
            w_end = min(total_frames, end + right_halo)

            _, real_s, imag_s = self._dec_sess.run(
                None, {"frame_h": frame_h[:, w_start:w_end, :].astype(np.float32)}
            )
            real_win, imag_win = real_s[0], imag_s[0]

            spec_pad_l = min(2, cursor - w_start)
            spec_pad_r = min(2, w_end - end)
            rel_l = cursor - w_start
            rel_r = end - w_start

            audio_win = _hann_istft_numpy(
                real_win[:, rel_l - spec_pad_l : rel_r + spec_pad_r],
                imag_win[:, rel_l - spec_pad_l : rel_r + spec_pad_r],
                n_fft=self._n_fft,
                hop_length=self._hop_length,
                window=self._hann,
            )

            trim_l = spec_pad_l * self._hop_length
            target_samples = (
                (end - cursor) * self._hop_length
                if end < total_frames
                else max((end - cursor - 1) * self._hop_length, 0)
            )
            if len(audio_win) >= trim_l + target_samples and target_samples > 0:
                yield audio_win[trim_l : trim_l + target_samples]
            elif len(audio_win) > trim_l:
                yield audio_win[trim_l:]
            cursor = end
            step += 1


def _load_pawp_runner(model_id: str, revision: str | None, threads: int):
    # Pure-Python frontend imports only (zero torch / dotenv / safetensors dependencies)
    from pawp_tts.application.frontend_service import PhonemizeUtteranceUseCase
    from pawp_tts.infrastructure.frontend.english_adapter import EnglishEspeakG2PAdapter
    from pawp_tts.infrastructure.frontend.nepali_adapter import NepaliRuleG2PAdapter

    rev = revision or PINNED_REVISIONS.get(model_id)
    enc_p = Path(hf_hub_download(model_id, "onnx/pawp_en_encoder_dur.onnx", revision=rev))
    dec_p = Path(hf_hub_download(model_id, "onnx/pawp_en_frame_decoder.onnx", revision=rev))
    prog_p = Path(hf_hub_download(model_id, "onnx/pawp_en_progress.npz", revision=rev))
    cfg_p = Path(hf_hub_download(model_id, "config.yaml", revision=rev))
    cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))

    sample_rate = int(cfg.get("sample_rate", 24000))
    hop_length = int(cfg.get("hop_length", 240))
    n_fft = int(cfg.get("n_fft", 1024))

    runner = _PureOnnxStreamingRunner(
        enc_path=enc_p,
        dec_path=dec_p,
        prog_path=prog_p,
        n_fft=n_fft,
        hop_length=hop_length,
        intra_op_threads=threads,
    )
    phonemize = PhonemizeUtteranceUseCase(EnglishEspeakG2PAdapter(), NepaliRuleG2PAdapter())
    return runner, phonemize, sample_rate, rev


def _frontend_arrays(phonemize, text: str):
    from pawp_tts.domain.entities.models import LanguageCode

    seq = phonemize.execute(text.strip())
    tokens = seq.tokens
    lang_map = {
        LanguageCode.SHARED: 0,
        LanguageCode.EN: 1,
        LanguageCode.NE: 2,
        LanguageCode.MIXED: 3,
    }
    p_ids = np.array([[tok.token_id for tok in tokens]], dtype=np.int64)
    l_ids = np.array([[lang_map.get(tok.language, 0) for tok in tokens]], dtype=np.int64)
    s_ids = np.array([[min(tok.stress, 3) for tok in tokens]], dtype=np.int64)
    b_flags = np.array(
        [
            [
                [
                    1.0 if tok.is_word_start else 0.0,
                    1.0 if tok.is_word_end else 0.0,
                    1.0 if tok.is_syllable_start else 0.0,
                    1.0 if tok.is_phrase_break else 0.0,
                ]
                for tok in tokens
            ]
        ],
        dtype=np.float32,
    )
    return p_ids, l_ids, s_ids, b_flags


def main(args: argparse.Namespace) -> None:
    if args.device != "cpu":
        raise ValueError("pawp_tts folded ONNX backend runs on CPU (--device=cpu).")

    runner, phonemize, sample_rate, rev = _load_pawp_runner(
        args.model_id, args.revision, args.threads
    )

    def synth_one(text: str) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        p_ids, l_ids, s_ids, b_flags = _frontend_arrays(phonemize, text)
        chunks = list(
            runner.stream_pcm(
                p_ids, l_ids, s_ids, b_flags, first_chunk_frames=6, chunk_frames=32
            )
        )
        audio = np.concatenate(chunks) if chunks else np.zeros(2400, dtype=np.float32)
        elapsed = time.perf_counter() - t0
        return audio, elapsed

    def synth_one_streaming(text: str, early_stop: bool) -> tuple[np.ndarray, float, int]:
        p_ids, l_ids, s_ids, b_flags = _frontend_arrays(phonemize, text)
        parts: list[np.ndarray] = []
        first: float | None = None
        for part in runner.stream_pcm(
            p_ids, l_ids, s_ids, b_flags, first_chunk_frames=6, chunk_frames=16
        ):
            parts.append(part)
            if first is None:
                first = time.perf_counter()
                if early_stop:
                    break
        if not parts or first is None:
            raise RuntimeError(f"no audio generated for {text[:80]!r}")
        return np.concatenate(parts), first, len(parts)

    paths = output_paths(args, "")
    model_dir, dataset_dir_name, manifest_path = (
        paths.model_dir,
        paths.dataset_dir_name,
        paths.manifest_path,
    )
    dataset = load_tts_dataset(args)

    # TTFA probe (streaming latency sidecar)
    if args.ttfa_probe != 0:
        from ttfa_probe import run_probe, sample_indices

        chunk_counts: list[int] = []

        def _ttfa_gen(job, early_stop):
            audio, first, n_chunks = synth_one_streaming(job["text"], early_stop)
            chunk_counts.append(n_chunks)
            return audio, sample_rate, first

        run_probe(
            _ttfa_gen,
            [
                {"text": dataset[i][args.text_column]}
                for i in sample_indices(args.ttfa_probe, len(dataset))
            ],
            model_id=args.model_id,
            dataset_path=args.dataset_path,
            dataset_config=args.dataset,
            split=args.split,
            out_dir=model_dir,
            mode_suffix="",
            n=args.ttfa_probe,
            extra={
                "device": "cpu",
                "runtime": "onnxruntime (CPUExecutionProvider)",
                "revision": rev,
                "threads": args.threads,
                "first_chunk_ms": 60.0,
                "chunks_per_utterance": chunk_counts,
            },
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(
        lambda batch: [synth_one(t) for t in batch[args.text_column]],
        dataset,
        starts,
        1,
        args.warmup_steps,
    )
    set_seed(args.seed)

    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        out_path = os.path.join(model_dir, rel_path)

        audio, elapsed = synth_one(text)
        sf.write(out_path, audio, sample_rate)

        entry = manifest_entry(rel_path, len(audio) / sample_rate, elapsed, text)
        write_entry(manifest_file, entry)
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="Firoj112/pawp-tts-1.5m",
        help="HF model repo (Firoj112/pawp-tts-1.5m or Firoj112/pawp-tts-300k).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF repo commit revision (defaults to pinned v3 release).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="ONNX Runtime intra-op CPU threads (default 4).",
    )
    add_common_args(parser)
    parser.set_defaults(device="cpu")
    parsed = parser.parse_args()
    main(parsed)
