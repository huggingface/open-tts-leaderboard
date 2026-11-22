"""
TTS synthesis for the Open TTS Leaderboard (PAWP-TTS folded ONNX backend, stage 1).

PAWP-TTS (https://github.com/Firojpaudel/PAWP_TTS) is a compact bilingual English-Nepali
streaming TTS family (ConvNeXt1D + iSTFT, 24 kHz, all learned inference parameters counted
including the waveform decoder). Two fixed-voice English folded ONNX checkpoints are supported:

  * Firoj112/pawp-tts-1.5m   1,490,390 params (5.98 MB FP32, 5.16 MB folded EN ONNX, n_fft=1024)
  * Firoj112/pawp-tts-300k   281,078 unique / 328,158 materialized params (1.12 MB FP32,
                             1.32 MB folded EN ONNX, n_fft=512)

Inference runs on CPU via ONNX Runtime (`CPUExecutionProvider`) with the repo's deterministic
English eSpeak-NG frontend and `OnnxStreamingRunner` (60 ms first chunk for `--ttfa_probe`).
Stage 2 (`transformers/transcribe.py`) transcribes with Qwen3-ASR, UTMOS22 scores naturalness,
and stage 4 computes WER + RTFx (no stage 3 SIM: fixed single-speaker voice).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
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


def _load_pawp_runner(model_id: str, revision: str | None, threads: int):
    from pawp_tts.infrastructure.inference.streaming import ExportedOnnxPaths, OnnxStreamingRunner
    from pawp_tts.interfaces.container import build_container

    rev = revision or PINNED_REVISIONS.get(model_id)
    enc_p = Path(hf_hub_download(model_id, "onnx/pawp_en_encoder_dur.onnx", revision=rev))
    dec_p = Path(hf_hub_download(model_id, "onnx/pawp_en_frame_decoder.onnx", revision=rev))
    prog_p = Path(hf_hub_download(model_id, "onnx/pawp_en_progress.npz", revision=rev))
    cfg_p = Path(hf_hub_download(model_id, "config.yaml", revision=rev))
    cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))

    sample_rate = int(cfg.get("sample_rate", 24000))
    hop_length = int(cfg.get("hop_length", 240))
    n_fft = int(cfg.get("n_fft", 1024))

    paths = ExportedOnnxPaths(
        encoder_dur_path=enc_p,
        frame_decoder_path=dec_p,
        progress_weights_path=prog_p,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    runner = OnnxStreamingRunner(paths, intra_op_threads=threads)
    phonemize = build_container().phonemize
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
