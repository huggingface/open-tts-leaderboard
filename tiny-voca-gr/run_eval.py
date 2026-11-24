"""
TTS synthesis for the Open TTS Leaderboard (tiny-voca-gr backend, stage 1).

tiny-voca-gr (https://huggingface.co/ComputationalLinguisticsPlatform/tiny-voca-gr) is a 4.53M-param
Modern Greek TTS trained from scratch with the tiny-voca architecture (acoustic model + source-filter
waveform generator, no neural vocoder), with two fixed voices (M, F) at 24 kHz. CPU-only. Licence
CC-BY-4.0.

Inference uses the repo's Greek adapter package (``tiny_voca_gr``, pip-installed from the repo) on top
of the upstream ``tiny_voca`` package. The front end is eSpeak NG's Greek phonemiser; the adapter
checks the eSpeak version stored in the checkpoint, so the job installs eSpeak NG (see job_image.sh).
The released checkpoint is ``inference.pt`` (104,000 steps).

Text handling follows the repo's usage: ``normalize_text`` prepares the text, then ``infer.speak_text``
(with the adapter's sentence splitter patched in) reads it sentence by sentence and joins the audio.
A text with nothing to pronounce is scored as a tenth of a second of silence.

Each voice is its own leaderboard row:
  * ``F``  default row (no manifest suffix).
  * ``M``  variant row, manifest suffix ``_m`` (publish with ``--variant m``).

Device: CPU only. ``--threads`` caps torch intra-op threads (default 1: a fixed count keeps RTFx
independent of the host's core count, matching the tiny-voca/sanotts convention).
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

# Pinned model repo revision (ComputationalLinguisticsPlatform/tiny-voca-gr main on 2026-11-22).
MODEL_REVISION = "927ea15cf4edcb8695b75328c9c90c5c53928554"

VOICES = {
    "F": "",      # default row
    "M": "_m",    # variant row (publish with --variant m)
}
DEFAULT_VOICE = "F"


def main(args):
    if not args.device.startswith("cpu"):
        raise SystemExit("tiny-voca-gr runs on CPU only; pass --device=cpu.")
    if args.voice not in VOICES:
        raise SystemExit(f"--voice must be one of {list(VOICES)}")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    from huggingface_hub import hf_hub_download
    if args.checkpoint_path:
        checkpoint = args.checkpoint_path
    else:
        checkpoint = hf_hub_download(
            repo_id=args.model_id, filename="inference.pt", revision=args.revision)
    print(f"Checkpoint: {checkpoint} (revision {args.revision})")

    # The Greek adapter registers its front end and backend on the upstream tiny_voca package, then
    # patches in its own sentence splitter — exactly the sequence from the model card.
    from tiny_voca_gr.frontend import install, normalize_text   # noqa: E402
    from tiny_voca_gr.backend import install as install_backend  # noqa: E402
    from tiny_voca_gr.speak import sentences                      # noqa: E402
    install()
    install_backend()
    from tiny_voca import infer                                   # noqa: E402
    infer.sentences = sentences

    # The adapter's fingerprint bundles the full eSpeak version banner (training used eSpeak NG 1.51
    # at /usr/lib/x86_64-linux-gnu/espeak-ng-data). job_image.sh pins the Debian 12 (bookworm) base,
    # which provides exactly that eSpeak 1.51 + data path, so the check passes natively — no bypass.
    # (The Greek eSpeak G2P is byte-identical between 1.51 and 1.52, verified across all eval
    # sentences, so this is a provenance/reproducibility fix and leaves the audio unchanged.)
    model = infer.load_model(checkpoint, device="cpu")
    sample_rate = int(model.config.sample_rate)
    print(f"Loaded tiny-voca-gr voice={args.voice} | PyTorch + eSpeak NG, {sample_rate} Hz, "
          f"threads={args.threads}.")

    failures = []

    def synth_one(text):
        """Return (audio float32, elapsed_s). Text with nothing pronounceable is scored as silence."""
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                wave, _ = infer.speak_text(model, normalize_text(text), args.voice)
            audio = wave.detach().cpu().float().numpy().astype(np.float32, copy=False)
        except (ValueError, RuntimeError) as exc:
            failures.append(str(exc))
            print(f"WARNING: refused ({exc}); writing silence for: {text[:80]!r}")
            audio = np.zeros(sample_rate // 10, dtype=np.float32)
        if audio.size == 0:
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
    parser.add_argument("--model_id", type=str, default="ComputationalLinguisticsPlatform/tiny-voca-gr",
                        help="tiny-voca-gr model repo (cc-by-4.0).")
    parser.add_argument("--revision", type=str, default=MODEL_REVISION,
                        help="Model repo revision (default: main on 2026-11-22).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local inference.pt. Default: download from --model_id.")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE,
                        help=f"One of {list(VOICES)}. {DEFAULT_VOICE} is the default row.")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch intra-op threads (default 1 for RTFx independent of host cores).")
    add_common_args(parser)
    # ``--device`` comes from ``add_common_args``; override the default to cpu since tiny-voca-gr is
    # CPU-only (``main`` refuses a GPU device).
    parser.set_defaults(seed=0, device="cpu")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} voice={args.voice} on "
          f"{args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
