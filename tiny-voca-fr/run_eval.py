"""
TTS synthesis for the Open TTS Leaderboard (tiny-voca-fr backend, stage 1).

tiny-voca-fr (https://huggingface.co/DanielSuncost/tiny-voca-fr, code
https://github.com/DanielSuncost/tiny-voca-fr) is a 4.5M-parameter standard French TTS trained from
scratch at 24 kHz with two fixed voices (FR_F1 female, FR_M1 male). It is the French port of tiny-voca
(https://huggingface.co/n4ze3m/tiny-voca): the same compact acoustic model and source-filter waveform
generator (no neural vocoder), trained from scratch on French. CPU-only. Licence apache-2.0.

Inference uses the model's own fork of the ``tiny_voca`` package (installed from its GitHub repo), with
``TINY_VOCA_LANGUAGE=fr`` selecting the French front end: eSpeak NG's ``fr-fr`` phonemiser into the
model's French phone inventory, with a stress feature. ``infer.speak_text`` does sentence splitting,
chunking to the 512-token budget, joining and peak-limiting to 0.99 internally, so one call handles any
text. A text with nothing to pronounce is scored as a tenth of a second of silence.

The checkpoint records the eSpeak front-end fingerprint (training used eSpeak NG 1.51); job_image.sh
pins the Debian 12 (bookworm) base so eSpeak is 1.51 at the training data path and the fingerprint
passes natively (the check stays ON — no bypass). The French eSpeak fr-fr G2P is byte-identical between
1.51 and 1.52 (verified on all eval sentences), so the base pin is for the fingerprint, not the audio.

Each voice is its own leaderboard row:
  * ``FR_F1``  default row (no manifest suffix).
  * ``FR_M1``  variant row, manifest suffix ``_m1`` (publish with ``--variant m1``).

Device: CPU only. ``--threads`` caps torch intra-op threads (default 1: a fixed count keeps RTFx
independent of the host's core count, matching the tiny-voca/sanotts convention).
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, warm_up, wav_rel_path, write_entry,
)

# Pinned model repo revision (DanielSuncost/tiny-voca-fr main on 2026-11-22).
MODEL_REVISION = "73f7700f514e70eef3b459aedcc49829d41945d4"

VOICES = {
    "FR_F1": "",      # default row
    "FR_M1": "_m1",   # variant row (publish with --variant m1)
}
DEFAULT_VOICE = "FR_F1"


def main(args):
    if not args.device.startswith("cpu"):
        raise SystemExit("tiny-voca-fr runs on CPU only; pass --device=cpu.")
    if args.voice not in VOICES:
        raise SystemExit(f"--voice must be one of {list(VOICES)}")
    mode_suffix = VOICES[args.voice]

    # The French front end is selected by an env var read at import time, so set it before importing the
    # package's text/infer modules.
    os.environ["TINY_VOCA_LANGUAGE"] = "fr"
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    from huggingface_hub import hf_hub_download
    if args.checkpoint_path:
        checkpoint = args.checkpoint_path
    else:
        checkpoint = hf_hub_download(args.model_id, "tiny-voca-fr.pt", revision=args.revision)
    print(f"Checkpoint: {checkpoint} (revision {args.revision})")

    from tiny_voca import infer
    from tiny_voca.text import NothingToPronounce
    # check_frontend stays ON: the bookworm base provides eSpeak 1.51, matching the checkpoint.
    model = infer.load_model(checkpoint, device="cpu")
    sample_rate = int(model.config.sample_rate)
    print(f"Loaded tiny-voca-fr voice={args.voice} | PyTorch + eSpeak NG fr-fr, {sample_rate} Hz, "
          f"threads={args.threads}.")

    failures = []

    def synth_one(text):
        """Return (audio float32, elapsed_s). Text with nothing pronounceable is scored as silence."""
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                wave, _ = infer.speak_text(model, text, args.voice)   # peak-limited to 0.99 inside
            audio = wave.detach().cpu().float().numpy().astype(np.float32, copy=False)
        except (ValueError, RuntimeError, NothingToPronounce) as exc:
            failures.append(str(exc))
            print(f"WARNING: refused ({exc}); writing silence for: {text[:80]!r}")
            audio = np.zeros(sample_rate // 10, dtype=np.float32)
        if audio.size == 0:
            audio = np.zeros(sample_rate // 10, dtype=np.float32)
        return audio, time.perf_counter() - start

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
    parser.add_argument("--model_id", type=str, default="DanielSuncost/tiny-voca-fr",
                        help="tiny-voca-fr model repo (apache-2.0).")
    parser.add_argument("--revision", type=str, default=MODEL_REVISION,
                        help="Model repo revision (default: main on 2026-11-22).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local tiny-voca-fr.pt. Default: download from --model_id.")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE,
                        help=f"One of {list(VOICES)}. {DEFAULT_VOICE} is the default row.")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch intra-op threads (default 1 for RTFx independent of host cores).")
    add_common_args(parser)
    # ``--device`` comes from ``add_common_args``; override the default to cpu since tiny-voca-fr is
    # CPU-only (``main`` refuses a GPU device).
    parser.set_defaults(seed=0, device="cpu")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} voice={args.voice} on "
          f"{args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
