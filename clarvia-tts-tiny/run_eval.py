"""
TTS synthesis for the Open TTS Leaderboard (clarvia-tts-tiny backend, stage 1).

clarvia-tts-tiny (https://huggingface.co/wite-tech/clarvia-tts-tiny) is a 4,536,907-parameter
Turkish TTS trained from scratch at 24 kHz with the tiny-voca architecture (a FastSpeech2-style
acoustic model plus a small source-filter waveform generator, no neural vocoder). It is the Turkish
port of tiny-voca (https://huggingface.co/n4ze3m/tiny-voca); the English phoneme inventory does not
carry over, so the weights were trained from scratch on Turkish. CPU-only (~25x real time on one
thread).

Inference is the ONNX graph shipped in the repo (``onnx/model.onnx``) driven by the repo's own
``speak.ClarviaTiny`` wrapper. The Turkish front end is the pure-Python ``clarvia_phonemizer``
package (https://huggingface.co/wite-tech/clarvia-phonemizer), which reads numbers, dates, money,
phone numbers and addresses from context (the work a separate text-normalisation step usually does).

Text handling is exactly what ``ClarviaTiny.speak`` does: sentence split, each sentence phonemised
and synthesised as one ONNX call, long sentences cut at commas then between words (the model's
``max_tokens`` budget), and the audio joined peak-limited to 0.99. A sentence with nothing to
pronounce is skipped; a text with nothing pronounceable is scored as a tenth of a second of silence.

Each voice is its own leaderboard row:
  * ``female_1``  default row (no manifest suffix).
  * ``male_1``    variant row, manifest suffix ``_male1`` (publish with ``--variant male1``).

Device: CPU only. ``--threads`` caps onnxruntime intra-op threads (default 1: a fixed count keeps
RTFx independent of the host's core count, matching the tiny-voca/sanotts convention).
"""

import argparse
import importlib.util
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

# Pinned model repo revision (wite-tech/clarvia-tts-tiny main on 2026-11-22).
MODEL_REVISION = "51023684a0a39742455c838e9d1d8714d3ae7289"

VOICES = {
    "female_1": "",        # default row
    "male_1": "_male1",    # variant row (publish with --variant male1)
}
DEFAULT_VOICE = "female_1"


def _load_speak_module(repo_dir):
    """Import the repo's own ``speak.py`` (``ClarviaTiny``) from the downloaded snapshot."""
    path = os.path.join(repo_dir, "speak.py")
    spec = importlib.util.spec_from_file_location("clarvia_speak", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["clarvia_speak"] = module
    spec.loader.exec_module(module)
    return module


def main(args):
    if not args.device.startswith("cpu"):
        raise SystemExit("clarvia-tts-tiny runs on CPU only; pass --device=cpu.")
    if args.voice not in VOICES:
        raise SystemExit(f"--voice must be one of {list(VOICES)}")

    from huggingface_hub import snapshot_download
    if args.checkpoint_path:
        repo_dir = args.checkpoint_path
    else:
        repo_dir = snapshot_download(
            args.model_id, revision=args.revision,
            local_dir=os.path.join(args.local_dir, args.model_id.replace("/", "-")),
            allow_patterns=["onnx/**", "config.json", "speak.py"],
        )
    print(f"Checkpoint: {repo_dir} (revision {args.revision})")

    speak_mod = _load_speak_module(repo_dir)
    tts = speak_mod.ClarviaTiny(
        model=os.path.join(repo_dir, "onnx", "model.onnx"),
        config=os.path.join(repo_dir, "config.json"),
        threads=args.threads,
    )
    sample_rate = int(tts.config["sample_rate"])
    print(f"Loaded clarvia-tts-tiny voice={args.voice} | onnxruntime, {sample_rate} Hz, threads={args.threads}.")

    failures = []

    def synth_one(text):
        """Return (audio float32, elapsed_s). Text with nothing pronounceable is scored as silence."""
        start = time.perf_counter()
        try:
            wave = tts.speak(text, voice=args.voice)
            audio = np.asarray(wave, dtype=np.float32)
        except ValueError as exc:   # ClarviaTiny raises ValueError when nothing can be read
            failures.append(str(exc))
            print(f"WARNING: refused ({exc}); writing silence for: {text[:80]!r}")
            audio = np.zeros(sample_rate // 10, dtype=np.float32)
        return audio, time.perf_counter() - start

    mode_suffix = VOICES[args.voice]
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # No streaming API, so `first` is None and the probe records the whole-utterance time.
    if args.ttfa_probe != 0:
        from ttfa_probe import run_probe, sample_indices
        run_probe(
            lambda job: (synth_one(job["text"])[0], sample_rate, None),
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={"device": args.device, "voice": args.voice, "threads": args.threads, "revision": args.revision},
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
    parser.add_argument("--model_id", type=str, default="wite-tech/clarvia-tts-tiny",
                        help="clarvia-tts-tiny model repo (apache-2.0).")
    parser.add_argument("--revision", type=str, default=MODEL_REVISION,
                        help="Model repo revision (default: main on 2026-11-22).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local copy of the repo. Default: download --model_id under --local_dir.")
    parser.add_argument("--local_dir", type=str, default="/tmp/clarvia_models",
                        help="Where to materialise the model repo as real files.")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE,
                        help=f"One of {list(VOICES)}. {DEFAULT_VOICE} is the default row.")
    parser.add_argument("--threads", type=int, default=1,
                        help="onnxruntime intra-op threads (default 1 for RTFx independent of host cores).")
    add_common_args(parser)
    # ``--device`` comes from ``add_common_args``; override the default to cpu since clarvia is
    # CPU-only (``main`` refuses a GPU device).
    parser.set_defaults(seed=0, device="cpu")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} voice={args.voice} on "
          f"{args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
