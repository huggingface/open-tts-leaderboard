"""
TTS synthesis for the Open TTS Leaderboard (alpa-es backend, stage 1).

alpa-es (https://huggingface.co/FranciscoCarlosErra/alpa-es) is a ~4.5M-parameter Rioplatense
Spanish TTS built on the tiny-voca architecture (acoustic model + source-filter waveform generator,
no neural vocoder), with two voices (female, male) at 24 kHz. CPU-only. Licence Apache-2.0. It is a
proof of concept trained for only 3000 steps on OpenSLR 61.

Inference is the PyTorch checkpoint (``alpa-es.pt``) with the repo's own ``alpa.infer`` code. The
front end (``alpa.text``) is a rule-based Spanish grapheme-to-phoneme converter with Rioplatense
pronunciation (seseo, yeísmo, /x/ for j and soft g, trilled rr, silent h, v≈b); stress from the
written accent or the default rule. Note: the front end does NOT expand digits, so numbers in the
source text are read as written.

Text handling: the text is read sentence by sentence (``alpa.infer.speak`` synthesises one utterance
per call), and the audio joined peak-limited to 0.99. A sentence that cannot be phonemised is skipped;
a text with nothing pronounceable is scored as a tenth of a second of silence.

Each voice is its own leaderboard row:
  * ``female``  default row (no manifest suffix).
  * ``male``    variant row, manifest suffix ``_male`` (publish with ``--variant male``).

Device: CPU only. ``--threads`` caps torch intra-op threads (default 1: a fixed count keeps RTFx
independent of the host's core count, matching the tiny-voca/sanotts convention).
"""

import argparse
import os
import re
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

PEAK = 0.99

# Pinned model repo revision (FranciscoCarlosErra/alpa-es main on 2026-11-22).
MODEL_REVISION = "de2bf894bdd9eb768ae03fc6b385176a0ff505d1"

VOICES = {
    "female": "",        # default row
    "male": "_male",     # variant row (publish with --variant male)
}
DEFAULT_VOICE = "female"

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [part for part in _SENTENCE_END.split(text) if re.search(r"\w", part)]


def main(args):
    if not args.device.startswith("cpu"):
        raise SystemExit("alpa-es runs on CPU only; pass --device=cpu.")
    if args.voice not in VOICES:
        raise SystemExit(f"--voice must be one of {list(VOICES)}")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    from huggingface_hub import snapshot_download
    if args.checkpoint_path:
        repo_dir = args.checkpoint_path
    else:
        repo_dir = snapshot_download(
            args.model_id, revision=args.revision,
            local_dir=os.path.join(args.local_dir, args.model_id.replace("/", "-")),
            allow_patterns=["alpa-es.pt", "alpa/**"],
        )
    print(f"Checkpoint: {repo_dir} (revision {args.revision})")

    # The repo's code imports as the top-level ``alpa`` package; put the snapshot first.
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from alpa.infer import load_model, speak   # noqa: E402

    model = load_model(os.path.join(repo_dir, "alpa-es.pt"), "cpu")
    sample_rate = int(model.config.sample_rate)
    print(f"Loaded alpa-es voice={args.voice} | PyTorch, {sample_rate} Hz, threads={args.threads}.")

    failures = []

    def synth_one(text):
        """Return (audio float32, elapsed_s). Text with nothing pronounceable is scored as silence."""
        start = time.perf_counter()
        chunks = []
        with torch.inference_mode():
            for sentence in _sentences(text):
                try:
                    wave, _, _ = speak(model, sentence, args.voice)
                    chunks.append(wave.detach().cpu().float().numpy().astype(np.float32, copy=False))
                except (ValueError, RuntimeError) as exc:
                    failures.append(str(exc))
                    print(f"WARNING: refused ({exc}); skipping sentence: {sentence[:80]!r}")
        if not chunks:
            audio = np.zeros(sample_rate // 10, dtype=np.float32)
        else:
            wave = np.concatenate(chunks)
            loudest = float(np.abs(wave).max())
            audio = (wave * (PEAK / loudest) if loudest > PEAK else wave).astype(np.float32)
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
    print(f"Refused sentences (skipped): {len(failures)}")
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="FranciscoCarlosErra/alpa-es",
                        help="alpa-es model repo (apache-2.0).")
    parser.add_argument("--revision", type=str, default=MODEL_REVISION,
                        help="Model repo revision (default: main on 2026-11-22).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local copy of the repo. Default: download --model_id under --local_dir.")
    parser.add_argument("--local_dir", type=str, default="/tmp/alpa_es_models",
                        help="Where to materialise the model repo as real files.")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE,
                        help=f"One of {list(VOICES)}. {DEFAULT_VOICE} is the default row.")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch intra-op threads (default 1 for RTFx independent of host cores).")
    add_common_args(parser)
    # ``--device`` comes from ``add_common_args``; override the default to cpu since alpa-es is
    # CPU-only (``main`` refuses a GPU device).
    parser.set_defaults(seed=0, device="cpu")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} voice={args.voice} on "
          f"{args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
