"""
TTS synthesis for the Open TTS Leaderboard (Scylla's Band backend, stage 1).

Scylla's Band (spybyscript/scyllasband, runtime: https://github.com/lowkeytea/scyllasband) is a
~131M-parameter phone-duration / pitch / rectified-flow TTS with a ~30M neural G2P, fourteen built-in
voices and six languages (en, es, it, fr, de, vi). There is no voice cloning, so each split is spoken
by one fixed voice (`--voice`, default `tuesday`, which speaks British English; harness `en` selects
each voice's own English dialect).

Inference runs through the `scyllasband` package on a bundle baked into the image:
  - `--device cuda`: ONNX Runtime with the CUDA execution provider on the WebGPU bundle (8-bit
    weight-only MatMuls, which stay on the GPU; the ONNX bundle's dynamic INT8 would fall back to CPU).
  - `--device cpu`: the LiteRT bundle on XNNPACK, the runtime's default CPU path, with one thread per CPU
    of the job's quota (`--threads`, else OMP_NUM_THREADS).

There is no batched API: rows are synthesized one at a time. Every graph is loaded before the
untimed warm-up, so per-sample `time` covers synthesis only (long text is planned into passages of
about 1.4-9 s internally). The sampling seed is fixed (`--seed`), so a rerun gives the same audio.

`--ttfa_probe` uses the streaming API (`synthesize_stream`): the first passage is short, so the
timestamp of its `audio_chunk` is a real time to first audio.

Stage 2 (`transformers/transcribe.py`) fills `pred_text`; stage 4 scores WER + RTFx. Fixed voice,
so no stage 3 (SIM).
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

SAMPLING_RATE = 24_000
# Harness split / ASR language -> runtime language. "en" is the voice's own English dialect.
LANGUAGES = {"en": "en", "es": "es", "it": "it", "fr": "fr", "de": "de", "vi": "vi"}
BUNDLES = {"cuda": "/app/models/webgpu", "cpu": "/app/models/litert"}


def cpu_threads(requested):
    """Threads per graph on CPU: --threads, else the job's CPU quota (OMP_NUM_THREADS, which the TTFA probe sets
    from cgroup cpu.max), else the CPUs this process may run on. LiteRT left to choose runs on one thread."""
    if requested:
        return requested
    value = os.environ.get("OMP_NUM_THREADS", "")
    if value.isdigit() and int(value) > 0:
        return int(value)
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def main(args):
    from scyllasband import ScyllasBandRuntime, SynthesisRequest
    from scyllasband.planner import records_from_text

    set_seed(args.seed)
    language = args.language or LANGUAGES.get(args.split)
    if language is None:
        raise SystemExit(f"Scylla's Band has no language for split {args.split!r}; supported: {', '.join(LANGUAGES)}")

    on_gpu = (args.device or "cuda").startswith("cuda")
    bundle = args.bundle_dir or BUNDLES["cuda" if on_gpu else "cpu"]
    if on_gpu:
        rt = ScyllasBandRuntime.from_bundle(bundle, backend="onnx", compute_units="gpu")
    else:
        rt = ScyllasBandRuntime.from_bundle(bundle, threads=cpu_threads(args.threads))
    # Load every graph (all length buckets) now, so no session is created inside a timed call.
    for name in rt.engine.manifest["components"]:
        rt.engine.session(name)
    threads = "" if on_gpu else f", threads={cpu_threads(args.threads)}"
    print(f"Loaded Scylla's Band ({bundle}, backend={rt.backend}{threads}, voice={args.voice}, language={language})")
    print("TTS model size: 0.16B parameters (0.13B without the G2P)")

    def synth_one(text):
        """Synthesize one text. Return (audio_1d_numpy, elapsed_s)."""
        start = time.perf_counter()
        try:
            result = rt.synthesize(SynthesisRequest(text=text, voice_id=args.voice, language=language, seed=args.seed))
            audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
        except ValueError as error:   # e.g. a row with nothing speakable; scored as a miss, not a crash
            print(f"Synthesis failed ({error}); writing silence for: {text!r}")
            audio = np.zeros(SAMPLING_RATE // 10, dtype=np.float32)
        return audio, time.perf_counter() - start

    def synth_one_streaming(text, early_stop=False):
        """TTFA probe only: return (audio_1d_numpy, sample_rate, first_chunk_ts)."""
        first, chunks = None, []
        stream = rt.synthesize_stream(records_from_text(text, voice=args.voice, language=language), seed=args.seed)
        for event in stream:
            if event.type != "audio_chunk" or event.audio is None:
                continue
            if first is None:
                first = time.perf_counter()
            chunks.append(np.asarray(event.audio, dtype=np.float32).reshape(-1))
            if early_stop:
                break
        getattr(stream, "close", lambda: None)()   # stop the remaining passages after an early stop
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        return audio, SAMPLING_RATE, first

    # Layout: results/<model_safe>/{MODEL_<safe>_DATASET_<dsid>.jsonl, <dsid>/output_<id>.wav}.
    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    if args.ttfa_probe != 0:
        from ttfa_probe import run_probe, sample_indices
        run_probe(
            lambda job, early_stop: synth_one_streaming(job["text"], early_stop),
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            extra={"device": args.device, "backend": rt.backend, "voice": args.voice},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    # Serial synthesis: one sample per "batch"; warm-up runs untimed on the first pending samples.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)
    set_seed(args.seed)

    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        path = os.path.join(model_dir, rel_path)

        audio, elapsed = synth_one(text)
        sf.write(path, audio, SAMPLING_RATE)

        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLING_RATE, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="spybyscript/scyllasband", help="Model repo id (names the results).")
    parser.add_argument("--voice", type=str, default="tuesday", help="Built-in voice (python -m scyllasband list-voices).")
    parser.add_argument("--language", type=str, default=None,
                        help="Runtime language; default: from --split (en is the voice's own English dialect).")
    parser.add_argument("--bundle_dir", type=str, default=None,
                        help="Bundle directory; default: the WebGPU bundle on CUDA, the LiteRT bundle on CPU.")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads per graph (LiteRT on CPU); default: the job's CPU quota (OMP_NUM_THREADS).")
    add_common_args(parser)
    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
