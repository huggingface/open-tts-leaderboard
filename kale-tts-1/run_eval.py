"""
TTS synthesis for the Open TTS Leaderboard (kale-tts-1 backend, stage 1).

kale-tts-1 (https://huggingface.co/KaleComputer/kale-tts-1-nano, https://huggingface.co/KaleComputer/kale-tts-1-tiny) is a
small English TTS for CPU and on-device use, built from a Kokoro-82M teacher fine-tuned on its voices (nano is distilled
from it; tiny keeps the teacher's timing-and-melody planner, frozen). Ten synthetic voices designed with
Maya-1 (text descriptions only, no cloning). Each model is one ONNX graph (text side + duration/pitch heads + a harmonic-plus-noise
iSTFT decoder, 24 kHz, no separate vocoder):
  * ``KaleComputer/kale-tts-1-nano``  7.5M parameters, 14.8 MB (int8 decoder)
  * ``KaleComputer/kale-tts-1-tiny``  45.7M parameters (16.8M of them a frozen Ettin word encoder), 79.9 MB (int8)

One leaderboard row per model, with the default voice ``kenzie``. Another voice is a variant row (manifest suffix
``_<voice>``, publish with ``--variant <voice>``).

Text handling is exactly what the shipped package does (``python/kale_tts.py`` in the model repo, ``KaleTTS.speak``): the release
front end (number / date / abbreviation rules, CMUdict-based lexicon, a small G2P model for unknown words; run in QuickJS, no
espeak-ng) splits the text into sentences, each sentence is one call of the graph (the tiny's Ettin encoder also sees the previous
sentence), and the sentences are joined with 120 ms of silence. No peak limiting or loudness change. A text with nothing to
pronounce is written as one tenth of a second of silence.

Device: CPU only (ONNX Runtime, ``CPUExecutionProvider``). ``--threads`` sets the intra-op thread count (default 1: a fixed count
keeps RTFx independent of the host's core count, matching the sanotts / tiny-voca convention).
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, warm_up, wav_rel_path, write_entry,
)

SAMPLE_RATE = 24000

# Pinned model repo revisions (main of each public repo on the release day, 19 Nov 2026).
MODEL_REVISIONS = {
    "KaleComputer/kale-tts-1-nano": "b6ce1887268c3667994fc3b0b2c82887c069915c",
    "KaleComputer/kale-tts-1-tiny": "69011009e3380f1cc8fe990cbb55b799bde8455a",
}
# Parameter counts from the model cards (an ONNX graph has no nn.Module to count).
MODEL_PARAMS = {
    "KaleComputer/kale-tts-1-nano": 7_500_000,
    "KaleComputer/kale-tts-1-tiny": 45_700_000,
}
DEFAULT_VOICE = "kenzie"
# Inference files only: the int8 graph at the repo root (``<repo name>.onnx``, named in voices.json "graphs"), voices,
# tokenizers and the shipped Python package + front end. Skips the fp32 graph and the PyTorch weights under ``pytorch/``.
ALLOW_PATTERNS = [
    "voices.json", "voices/*", "config.json", "tokenizer.json", "tokenizer_config.json", "ettin-tokenizer.json",
    "python/kale_tts.py", "python/kale_frontend.js", "frontend/words/**",
]


def allow_patterns(model_id):
    return [f"{model_id.split('/')[-1]}.onnx", *ALLOW_PATTERNS]


def main(args):
    if not args.device.startswith("cpu"):
        raise SystemExit("kale-tts-1 runs on CPU only here; pass --device=cpu.")

    if args.checkpoint_path:
        repo_dir = args.checkpoint_path
    else:
        from huggingface_hub import snapshot_download
        repo_dir = snapshot_download(
            args.model_id, revision=args.revision or MODEL_REVISIONS.get(args.model_id, "main"),
            local_dir=os.path.join(args.local_dir, args.model_id.replace("/", "-")), allow_patterns=allow_patterns(args.model_id),
        )
    print(f"Checkpoint: {repo_dir} (revision {args.revision or MODEL_REVISIONS.get(args.model_id, 'main')})")

    # The model repo ships its own Python package next to the weights.
    sys.path.insert(0, os.path.join(repo_dir, "python"))
    from kale_tts import KaleTTS, QuickJSFrontEnd   # noqa: E402

    tts = KaleTTS(repo_dir, threads=args.threads, frontend=QuickJSFrontEnd(os.path.join(repo_dir, "frontend")))
    voice = tts.resolve_voice(args.voice)
    n_params = MODEL_PARAMS.get(args.model_id)
    print(f"Loaded {args.model_id} voice={args.voice} ({voice}) | "
          f"{(n_params or 0) / 1e9:.4f}B parameters | onnxruntime threads={args.threads}")

    failures = []

    def synth_one(text):
        """Return (audio float32, elapsed_s). A text the front end refuses is scored as silence."""
        start = time.perf_counter()
        try:
            audio = tts.speak(text, voice=voice)
        except (ValueError, RuntimeError) as exc:
            failures.append(str(exc))
            print(f"WARNING: refused ({exc}); writing silence for: {text[:80]!r}")
            audio = np.zeros(0, np.float32)
        if audio.size == 0:
            audio = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
        return audio.astype(np.float32, copy=False), time.perf_counter() - start

    mode_suffix = "" if voice == tts.resolve_voice(DEFAULT_VOICE) else f"_{args.voice.lower()}"
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # First audio = the first sentence, as KaleTTS.stream yields sentence by sentence.
    if args.ttfa_probe != 0:
        from ttfa_probe import run_probe, sample_indices

        def gen(job):
            first, parts = None, []
            for chunk in tts.stream(job["text"], voice=voice):
                if first is None:
                    first = time.perf_counter()
                parts.append(chunk)
            n = sum(len(p) for p in parts) + int(0.12 * SAMPLE_RATE) * len(parts)
            return n / SAMPLE_RATE, SAMPLE_RATE, first

        run_probe(
            gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={"device": args.device, "voice": voice, "threads": args.threads,
                   "revision": args.revision or MODEL_REVISIONS.get(args.model_id, "main")},
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
        sf.write(os.path.join(model_dir, rel_path), audio, SAMPLE_RATE)
        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLE_RATE, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    print(f"Refused texts (scored as silence): {len(failures)} of {len(starts)}")
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="KaleComputer/kale-tts-1-nano",
                        help="kale-tts-1 model repo (apache-2.0): KaleComputer/kale-tts-1-nano or KaleComputer/kale-tts-1-tiny.")
    parser.add_argument("--revision", type=str, default=None,
                        help="Model repo revision (default: the pinned release commit in MODEL_REVISIONS).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local copy of the repo. Default: download --model_id under --local_dir.")
    parser.add_argument("--local_dir", type=str, default="/tmp/kale_tts_models",
                        help="Where to materialise the model repo as real files.")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE,
                        help=f"Voice name or id from the repo's voices.json. {DEFAULT_VOICE} is the default row.")
    parser.add_argument("--threads", type=int, default=1,
                        help="ONNX Runtime intra-op threads (default 1 for RTFx independent of host cores).")
    add_common_args(parser)
    # ``--device`` comes from ``add_common_args``; override the default to cpu (``main`` refuses a GPU device).
    parser.set_defaults(seed=0, device="cpu")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} voice={args.voice} on "
          f"{args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
