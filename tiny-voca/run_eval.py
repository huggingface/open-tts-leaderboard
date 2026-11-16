"""
TTS synthesis for the Open TTS Leaderboard (tiny-voca backend, stage 1).

tiny-voca (https://huggingface.co/n4ze3m/tiny-voca, code
https://github.com/n4ze3m/tiny-voca) is a 4,537,291-parameter English TTS trained from scratch at
24 kHz with two voices (F2 and M2). Its waveform generator is a source-filter synthesizer (no
neural vocoder) so the whole model is one small torch graph. CPU-only (the model card reports
~70x real-time on four CPU threads).

Each voice is evaluated as its own leaderboard row:
  * ``F2``  default row (no manifest suffix).
  * ``M2``  variant row, manifest suffix ``_m2`` (publish with ``--variant m2``).

Weights come from ``n4ze3m/tiny-voca``'s ``tiny-voca.pt`` at a pinned revision. The phonemizer
is ``dizione`` (https://github.com/n4ze3m/dizione), whose data files live in the same repo under
``phonemizer/``; this script points it at the downloaded copy via the ``DIZIONE_DATA`` env var
before importing ``tiny_voca.infer``.

Text handling is exactly what ``tiny_voca.infer.speak_text`` does: sentence split (abbreviation-
and initial-aware), an added full stop if the text has no closing mark, each sentence synthesized
as one call of the model, long sentences cut at commas then between words, and the audio joined
peak-limited to 0.99. A text with nothing to pronounce (another script, signs only) is left as
one tenth of a second of silence.

Device: CPU only. ``--threads`` caps torch intra-op threads (default 1: a fixed count keeps RTFx
independent of the host's core count, matching the sanotts convention).
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, warm_up, wav_rel_path, write_entry,
)

SAMPLE_RATE = 24000

# Pinned model repo revision (n4ze3m/tiny-voca main on 2026-11-16).
MODEL_REVISION = "eac048b7a4392584725af9792460a0211fa1cac7"

VOICES = {
    "F2": "",       # default row
    "M2": "_m2",    # variant row (publish with --variant m2)
}
DEFAULT_VOICE = "F2"


def main(args):
    if not args.device.startswith("cpu"):
        raise SystemExit("tiny-voca runs on CPU only; pass --device=cpu.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    from huggingface_hub import snapshot_download
    if args.checkpoint_path:
        repo_dir = args.checkpoint_path
    else:
        repo_dir = snapshot_download(
            args.model_id, revision=args.revision,
            local_dir=os.path.join(args.local_dir, args.model_id.replace("/", "-")),
            allow_patterns=["tiny-voca.pt", "config.json", "phonemizer/**"],
        )
    print(f"Checkpoint: {repo_dir} (revision {args.revision})")

    # dizione reads its lexicon/G2P tables from this folder. Must be set BEFORE importing tiny_voca.
    os.environ["DIZIONE_DATA"] = os.path.join(repo_dir, "phonemizer")
    from tiny_voca.infer import load_model, speak_text   # noqa: E402

    if args.voice not in VOICES:
        raise SystemExit(f"--voice must be one of {list(VOICES)}")

    model = load_model(os.path.join(repo_dir, "tiny-voca.pt"), device="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded tiny-voca voice={args.voice} | {n_params / 1e9:.4f}B parameters ({n_params:,}).")

    failures = []

    def synth_one(text):
        """Return (audio float32, elapsed_s). A text with nothing pronounceable or one word that
        exceeds the frame budget is scored as silence."""
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                wave, _ = speak_text(model, text, args.voice, seed=args.seed)
            audio = wave.detach().cpu().float().numpy().astype(np.float32, copy=False)
        except (ValueError, RuntimeError) as exc:
            failures.append(str(exc))
            print(f"WARNING: refused ({exc}); writing silence for: {text[:80]!r}")
            audio = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
        return audio, time.perf_counter() - start

    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # No streaming API, so `first` is None and the probe records the whole-utterance time.
    if args.ttfa_probe != 0:
        from ttfa_probe import run_probe, sample_indices
        run_probe(
            lambda job: (synth_one(job["text"])[0], SAMPLE_RATE, None),
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            extra={"device": args.device, "voice": args.voice, "seed": args.seed,
                   "threads": args.threads, "revision": args.revision},
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
    parser.add_argument("--model_id", type=str, default="n4ze3m/tiny-voca",
                        help="tiny-voca model repo (apache-2.0).")
    parser.add_argument("--revision", type=str, default=MODEL_REVISION,
                        help="Model repo revision (default: main on 2026-11-16).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local copy of the repo. Default: download --model_id under --local_dir.")
    parser.add_argument("--local_dir", type=str, default="/tmp/tiny_voca_models",
                        help="Where to materialise the model repo as real files.")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE,
                        help=f"One of {list(VOICES)}. {DEFAULT_VOICE} is the default row.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="tiny-voca is CPU-only; this must be 'cpu'.")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch intra-op threads (default 1 for RTFx independent of host cores).")
    add_common_args(parser)
    parser.set_defaults(seed=0)   # speak_text's default seed

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} voice={args.voice} on "
          f"{args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
