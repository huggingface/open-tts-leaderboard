"""
TTS synthesis for the Open TTS Leaderboard (tiny-voca-hindi backend, stage 1).

tiny-voca-hindi (https://huggingface.co/RL10x/tiny-voca-hindi) is a ~4.5M-parameter Hindi TTS trained
from scratch with the tiny-voca architecture (acoustic model + source-filter waveform generator, no
neural vocoder). CPU-only. Licence CC-BY-4.0.

Inference is the PyTorch checkpoint (``tiny_voca_hindi.pt``) with the repo's own code: the normaliser
``hindi_text.normalize_hi`` (numbers, dates, money, units and common English words -> Hindi words) and
the tiny-voca inference package shipped in the repo (``tiny_voca.infer``). The front end is
self-contained (Devanagari is largely phonetic — no eSpeak, no external dictionary).

Text handling mirrors the repo's ``speak.py``: the text is normalised, "।" mapped to ".", then read
sentence by sentence by ``speak_text`` (which splits long text and joins the audio), and the result
peak-limited with ``limit_peak``. A text with nothing to pronounce is scored as a tenth of a second
of silence.

The model exposes delivery *styles* (neutral, conv, happy, …), not speaker identities. The leaderboard
row uses the fixed ``neutral`` style.

Device: CPU only. ``--threads`` caps torch intra-op threads (default 1: a fixed count keeps RTFx
independent of the host's core count, matching the tiny-voca/sanotts convention).
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, warm_up, wav_rel_path, write_entry,
)

# Pinned model repo revision (RL10x/tiny-voca-hindi main on 2026-11-22).
MODEL_REVISION = "b76940f5edbc9c4ca2c7c8cf29f20897ff3339fe"

# The model's fixed delivery style (not a speaker identity). One leaderboard row.
STYLE = "neutral"


def main(args):
    if not args.device.startswith("cpu"):
        raise SystemExit("tiny-voca-hindi runs on CPU only; pass --device=cpu.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    from huggingface_hub import snapshot_download
    if args.checkpoint_path:
        repo_dir = args.checkpoint_path
    else:
        repo_dir = snapshot_download(
            args.model_id, revision=args.revision,
            local_dir=os.path.join(args.local_dir, args.model_id.replace("/", "-")),
            allow_patterns=["tiny_voca_hindi.pt", "config.json", "tiny_voca/**", "hindi_text/**"],
        )
    print(f"Checkpoint: {repo_dir} (revision {args.revision})")

    # The repo's code imports as top-level packages (tiny_voca, hindi_text); put the snapshot first.
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from hindi_text import normalize_hi                               # noqa: E402
    from tiny_voca.infer import limit_peak, load_model, speak_text    # noqa: E402

    model = load_model(os.path.join(repo_dir, "tiny_voca_hindi.pt"), "cpu")
    sample_rate = int(model.config.sample_rate)
    print(f"Loaded tiny-voca-hindi style={STYLE} | PyTorch, {sample_rate} Hz, threads={args.threads}.")

    failures = []

    def synth_one(text):
        """Return (audio float32, elapsed_s). Text with nothing pronounceable is scored as silence."""
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                spoken = normalize_hi(text, english=True).replace("।", ".")
                wave, _ = speak_text(model, spoken, STYLE, speed=1.0, peak=None)
                wave = limit_peak(wave)
            audio = wave.detach().cpu().float().numpy().astype(np.float32, copy=False)
        except (ValueError, RuntimeError) as exc:
            failures.append(str(exc))
            print(f"WARNING: refused ({exc}); writing silence for: {text[:80]!r}")
            audio = np.zeros(sample_rate // 10, dtype=np.float32)
        if audio.size == 0:
            audio = np.zeros(sample_rate // 10, dtype=np.float32)
        return audio, time.perf_counter() - start

    mode_suffix = ""   # single fixed style -> default row
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
            extra={"device": args.device, "style": STYLE, "threads": args.threads, "revision": args.revision},
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
    parser.add_argument("--model_id", type=str, default="RL10x/tiny-voca-hindi",
                        help="tiny-voca-hindi model repo (cc-by-4.0).")
    parser.add_argument("--revision", type=str, default=MODEL_REVISION,
                        help="Model repo revision (default: main on 2026-11-22).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local copy of the repo. Default: download --model_id under --local_dir.")
    parser.add_argument("--local_dir", type=str, default="/tmp/tiny_voca_hindi_models",
                        help="Where to materialise the model repo as real files.")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch intra-op threads (default 1 for RTFx independent of host cores).")
    add_common_args(parser)
    # ``--device`` comes from ``add_common_args``; override the default to cpu since this model is
    # CPU-only (``main`` refuses a GPU device).
    parser.set_defaults(seed=0, device="cpu")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} style={STYLE} on "
          f"{args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
