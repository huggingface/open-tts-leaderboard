"""
TTS synthesis for the Open TTS Leaderboard (suxin-light backend, stage 1).

suxin-light (https://huggingface.co/TEQUILA123/suxin-light) is a 28.65 M-parameter Mandarin–English
VITS student distilled from Fun-CosyVoice3-0.5B, with the SuxinVoc decoder, at 22.05 kHz mono. One
fixed voice; no voice cloning. The repo ships two ONNX graphs of the **same** weights:

  * ``zh_en-suxin-light.onnx`` (~88 MB) — FP32 source graph. Default on ``--device=cuda``.
  * ``zh_en-suxin-light.int8-excldec.onnx`` (~67 MB) — QDQ INT8 with decoder + duration path left
    in float (the on-device graph). Default on ``--device=cpu``.

There is no FP16 graph. INT8 is dynamic-range QDQ (hundreds of ``QuantizeLinear`` nodes); those
ops typically stay on the CPU even under the CUDA EP, so GPU eval uses the FP32 graph. Quality:
FP32 is the unquantised source, so it should match or slightly beat advertised INT8 (the card's
x86 single-thread RTF is 0.034 FP32 vs 0.026 INT8; duration-predictor numerics also differ
slightly, so wav lengths are not bit-identical). Override with ``--graph fp32|int8``.

The front end is mixed ``phoneme_type`` (stock piper-tts cannot load it): Chinese via pypinyin,
English via piper-tts's ``EspeakPhonemizer`` (espeak-ng data, GPL-3.0). G2P always runs on CPU.
PAD is inserted only after syllable/word ends (``pad_mode=group``). In a Chinese context, Arabic
digits are read as Chinese numerals inside the frontend (``请到3号窗口`` → 三).

Text handling is the CLI default (``suxin-tts`` without ``--no-split``): ``synthesize_long``
splits on sentence then clause punctuation, synthesises each piece, and stitches with a 50 ms
gap plus a short crossfade. A text with nothing to pronounce is scored as a tenth of a second
of silence.

Device: ``cuda`` (default; ``CUDAExecutionProvider`` + ``disable_fallback``) or ``cpu``.
``--threads`` caps onnxruntime intra-op threads (default 1).

Batching. Both graphs declare a dynamic ``batch_size`` on ``input`` / ``input_lengths`` / the
audio output, and a padded batch **does** run. The output is padded to the longest item's audio
length, and the pad is **synthesised speech** (PAD phonemes get duration), not zeros. There is no
duration head to crop with. Using that tail would inflate ``duration`` and pollute ASR, so this
backend stays at batch size 1 (one ONNX call per sentence piece).

Streaming / TTFA. The ONNX graph is a full-sequence VITS forward (phonemes → wav); there is no
duration head or chunked vocoder, so intra-model frame streaming is not possible. The package
already synthesises incrementally: ``split_sentences`` then one ``synthesize`` (one ONNX run)
per sentence/clause. ``synthesize_long`` / ``synthesize_stream`` **hold** the first piece until
the second exists so they can crossfade — that makes first audio equal last audio. The TTFA
path uses the same split and the same per-piece ONNX calls, but emits the first piece as soon
as it returns (Paradee / Woof / Kokoro sentence streaming). A one-sentence prompt has nothing
to stream (``n_chunks == 1``). Quality eval stays on ``synthesize_long`` so WER/CER do not move.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, warm_up, wav_rel_path, write_entry,
)

SAMPLE_RATE = 22050
# Model card: 28.65 M VITS student. ONNX initializer counts are ~20.6 M (fused / const-folded).
N_PARAMS = 28_650_000

GRAPHS = {
    "fp32": "zh_en-suxin-light.onnx",
    "int8": "zh_en-suxin-light.int8-excldec.onnx",
}

# Pinned model repo revision (TEQUILA123/suxin-light main on 2026-10-13).
MODEL_REVISION = "5ee2da546f3635da263602842d238d8624cb7a81"


def resolve_graph(device, graph):
    """``auto``: FP32 on CUDA (QDQ INT8 is a poor CUDA-EP fit), INT8 on CPU (the advertised graph)."""
    if graph == "auto":
        return "fp32" if device.startswith("cuda") else "int8"
    if graph not in GRAPHS:
        raise SystemExit(f"--graph must be one of auto, {', '.join(GRAPHS)}; got {graph!r}")
    return graph


def _configure_session(tts, onnx_path, device, threads, conv_algo_search):
    """Rebuild the ORT session with explicit providers. G2P/frontend is left on CPU.

    The package hardcodes ``CPUExecutionProvider``. On CUDA we load the CUDA EP first, refuse a
    silent CPU-only session, and disable ORT's per-run CPU fallback so a cuDNN miss doesn't get
    timed as GPU. ``HEURISTIC`` conv search (not EXHAUSTIVE) so new input lengths are not
    re-benchmarked inside the timed loop (see paradee-v1.3).
    """
    import onnxruntime as ort

    # Wheel-shipped CUDA/cuDNN (nvidia-cudnn-cu12) are not on LD_LIBRARY_PATH in the job image.
    # Skip on CPU: the CPU onnxruntime wheel warns if preload_dlls looks for CUDA 12.x.
    if device.startswith("cuda") and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode(0)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if device.startswith("cuda"):
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"--device={device} but this onnxruntime has no CUDA EP "
                f"(available={available}). Install onnxruntime-gpu "
                "(see suxin-light/job_image.sh)."
            )
        device_id = int(device.split(":")[1]) if ":" in device else 0
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": device_id,
                "cudnn_conv_algo_search": conv_algo_search,
            }),
            "CPUExecutionProvider",
        ]
    elif device.startswith("cpu"):
        providers = ["CPUExecutionProvider"]
    else:
        raise SystemExit(f"--device must be cpu or cuda[:N]; got {device!r}")

    tts.session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
    loaded = tts.session.get_providers()
    if device.startswith("cuda"):
        if "CUDAExecutionProvider" not in loaded:
            raise RuntimeError(
                f"--device={device} but the CUDA EP did not load: {loaded}. "
                "Need onnxruntime-gpu plus a matching CUDA/cuDNN (see suxin-light/job_image.sh)."
            )
        if hasattr(tts.session, "disable_fallback"):
            tts.session.disable_fallback()
    return loaded


# Inter-sentence gap used by SuxinTTS.synthesize_long (50 ms). The TTFA path inserts the same
# silence between pieces so audio_s is in the same ballpark; it does not apply the crossfade hold.
_STREAM_SILENCE_S = 0.05


def synth_streaming(tts, text, early_stop=False):
    """Sentence/clause streaming for TTFA. Return ``(audio, first_ts, n_chunks)``.

    Same ``split_sentences`` + ``synthesize`` loop as ``synthesize_long``, but the first non-empty
    piece is handed back as soon as that ONNX call returns instead of being buffered for a
    crossfade. ``first_ts`` is ``time.perf_counter()`` at that moment, or None if nothing spoke.
    ``early_stop`` aborts after the first piece (probe tail). A one-piece prompt has nothing to
    stream: first audio is last audio.
    """
    sample_rate = int(tts.sample_rate)
    silence = np.zeros(int(sample_rate * _STREAM_SILENCE_S), dtype=np.float32)
    first_ts = None
    chunks = []
    for sent in tts.split_sentences(text):
        audio = tts.synthesize(sent)
        if audio is None:
            continue
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            continue
        if first_ts is None:
            first_ts = time.perf_counter()
            chunks.append(audio)
            if early_stop:
                break
        else:
            chunks.append(np.concatenate([silence, audio]))
    if not chunks:
        return np.zeros(1, dtype=np.float32), None, 0
    return np.concatenate(chunks), first_ts, len(chunks)


def main(args):
    if args.batch_size != 1:
        raise SystemExit(
            f"--batch_size={args.batch_size} is not supported. Both ONNX graphs have a dynamic "
            "batch dim, but the audio pad is synthesised speech (PAD phonemes get duration), not "
            "silence, and there is no duration output to crop. Use --batch_size=1."
        )

    graph_key = resolve_graph(args.device, args.graph)
    onnx_name = GRAPHS[graph_key]

    from huggingface_hub import snapshot_download
    if args.checkpoint_path:
        repo_dir = args.checkpoint_path
    else:
        repo_dir = snapshot_download(
            args.model_id, revision=args.revision,
            local_dir=os.path.join(args.local_dir, args.model_id.replace("/", "-")),
            allow_patterns=["suxin_tts/**", onnx_name, onnx_name + ".json"],
        )
    print(f"Checkpoint: {repo_dir} (revision {args.revision})")

    onnx_path = Path(repo_dir) / onnx_name
    if not onnx_path.is_file():
        raise SystemExit(f"missing {onnx_path}")

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from suxin_tts import SuxinTTS  # noqa: E402

    tts = SuxinTTS.from_pretrained(
        repo_dir,
        model_file=onnx_name,
        intra_op_num_threads=args.threads,
        inter_op_num_threads=1,
        noise_scale=0.0,
        noise_w=0.0,
        pad_mode="group",
        use_lang_tags=False,
        normalize_audio=False,
    )
    providers = _configure_session(
        tts, onnx_path, args.device, args.threads, args.cudnn_conv_algo_search)

    if graph_key == "int8" and args.device.startswith("cuda"):
        print("WARNING: INT8 QDQ graph on CUDA EP — QuantizeLinear nodes often run on CPU. "
              "Default GPU eval uses --graph fp32.")

    sample_rate = int(tts.sample_rate)
    if sample_rate != SAMPLE_RATE:
        print(f"WARNING: frontend sample_rate={sample_rate}, expected {SAMPLE_RATE}")
    print(f"Loaded suxin-light | graph={graph_key} ({onnx_name}) | providers={providers} | "
          f"{sample_rate} Hz | threads={args.threads} | batch_size=1 | "
          f"TTS model size: {N_PARAMS / 1e9:.4f}B parameters ({N_PARAMS:,}).")

    failures = []

    def synth_one(text):
        """Return (audio float32, elapsed_s). Empty / unspeakable text is scored as silence.

        Times ``synthesize_long`` only (G2P + ONNX + stitch). ORT CUDA EP ``run`` is synchronous
        (outputs are on host when it returns), so no extra device sync.
        """
        start = time.perf_counter()
        try:
            audio = tts.synthesize_long(text)
            if audio is None:
                audio = np.zeros(0, dtype=np.float32)
            else:
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        except (ValueError, RuntimeError) as exc:
            failures.append(str(exc))
            print(f"WARNING: refused ({exc}); writing silence for: {text[:80]!r}")
            audio = np.zeros(0, dtype=np.float32)
        if audio.size == 0:
            audio = np.zeros(sample_rate // 10, dtype=np.float32)
        return audio, time.perf_counter() - start

    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    extra = {
        "device": args.device, "threads": args.threads, "revision": args.revision,
        "graph": graph_key, "onnx": onnx_name, "providers": list(providers),
        "streaming_api": "split_sentences + synthesize (first piece immediate)", "batch_size": 1,
    }

    if args.ttfa_probe != 0:
        from ttfa_probe import run_probe, sample_indices
        chunk_counts = []

        def _ttfa_gen(job, early_stop):
            audio, first, n_chunks = synth_streaming(tts, job["text"], early_stop)
            chunk_counts.append(n_chunks)
            return audio, sample_rate, first

        run_probe(
            _ttfa_gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            extra={**extra, "segments_per_utterance": chunk_counts},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)
    failures.clear()

    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        audio, elapsed = synth_one(text)
        sf.write(os.path.join(model_dir, rel_path), audio, sample_rate)
        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / sample_rate, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    print(f"Refused texts (scored as silence): {len(failures)} of {len(starts)}")
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="TEQUILA123/suxin-light",
                        help="suxin-light model repo (MIT; English G2P via piper-tts/espeak-ng, GPL-3.0).")
    parser.add_argument("--revision", type=str, default=MODEL_REVISION,
                        help="Model repo revision (default: main on 2026-10-13).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local copy of the repo. Default: download --model_id under --local_dir.")
    parser.add_argument("--local_dir", type=str, default="/tmp/suxin_light_models",
                        help="Where to materialise the model repo as real files.")
    parser.add_argument("--threads", type=int, default=1,
                        help="onnxruntime intra-op threads (default 1).")
    parser.add_argument("--graph", type=str, default="auto", choices=["auto", "fp32", "int8"],
                        help="ONNX graph: auto = fp32 on CUDA / int8 on CPU; or force fp32 / int8.")
    parser.add_argument("--cudnn_conv_algo_search", type=str, default="HEURISTIC",
                        help="CUDA EP conv algo search (HEURISTIC avoids EXHAUSTIVE re-bench per length).")
    add_common_args(parser, batch_size=1)
    parser.set_defaults(seed=0, device="cuda")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} device={args.device} graph={args.graph} on "
          f"{args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
