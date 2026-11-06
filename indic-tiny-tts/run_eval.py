"""
TTS synthesis for the Open TTS Leaderboard (indic-tiny-tts Hindi backend, stage 1).

indic-tiny-tts-hindi (sulabhkatiyar/indic-tiny-tts-hindi) is a 22.5M-parameter single-language model
(Matcha-style flow-matching acoustic model + Vocos vocoder, 24 kHz, character input, no eSpeak). It is
not on PyPI and not loadable with `transformers`: the HF repo itself ships the runtime package
`indic_tiny_tts/`. This script downloads the repo at a pinned revision, puts the download directory on
`sys.path`, and loads `TinyTTS` in-process.

The model has two fixed voices (`female`, `male`) and no voice cloning, so only the default-voice
("no cloning") table applies: one voice per run (`--voice`, default `male`). There is no batch API, so
rows are processed serially (batch size 1). The raw dataset text goes straight to the model (its bundled
text normaliser spells out digits etc.); the leaderboard normaliser is applied at scoring time only.

Stage 2 (`transformers/transcribe.py`) fills `pred_text` with Qwen3-ASR; stage 4
(`normalizer.eval_utils.score_results`, run locally) computes WER + RTFx. Fixed voice, so no SIM stage.
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

# The released model's native sample rate (config.json "sample_rate"; checked against tts.sample_rate below).
SAMPLING_RATE = 24_000
# Only the files the runtime needs: skip the ONNX graphs, sample wavs and card images.
IGNORE_PATTERNS = ["onnx/*", "samples/*", "assets/*"]


def load_model(args):
    """Download the pinned repo revision, import its bundled package from there, and build TinyTTS."""
    local_dir = snapshot_download(args.model_id, revision=args.revision, ignore_patterns=IGNORE_PATTERNS)
    # snapshot dirs are named after the resolved commit sha
    print(f"Model snapshot: {local_dir} (requested revision: {args.revision or 'default branch'})")
    if args.revision and len(args.revision) == 40 and os.path.basename(local_dir) != args.revision:
        raise RuntimeError(f"snapshot {os.path.basename(local_dir)} does not match requested revision {args.revision}")
    sys.path.insert(0, local_dir)
    from indic_tiny_tts import TinyTTS

    tts = TinyTTS.from_pretrained(local_dir, device=args.device)
    if tts.sample_rate != SAMPLING_RATE:
        raise RuntimeError(f"model sample rate {tts.sample_rate} != {SAMPLING_RATE}")
    if args.voice not in tts.voices:
        raise ValueError(f"unknown voice {args.voice!r}; choose from {sorted(tts.voices)}")
    return tts


def main(args):
    if args.ttfa_probe != 0:
        raise SystemExit("--ttfa_probe is not implemented for this backend (non-streaming model).")
    set_seed(args.seed)

    tts = load_model(args)
    print(f"Loaded indic-tiny-tts ({args.model_id}, lang={tts.lang}, voice={args.voice}, device={tts.device})")
    print_model_size(tts.acoustic, tts.vocoder)

    is_cuda = tts.device.type == "cuda"

    def synth_one(text, seed):
        """Synthesize one text with the default pipeline (chunking + pauses included).
        Return (audio_1d_numpy, elapsed_s); elapsed covers the whole synthesize() call."""
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        audio = tts.synthesize(text, voice=args.voice, speed=args.speed, seed=seed)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if audio.size == 0:  # e.g. a text with nothing speakable (Latin-only); keep the row, as the other backends do
            print(f"WARNING: empty audio for text {text[:60]!r}")
            audio = np.zeros(1, dtype=np.float32)
        return audio.astype(np.float32, copy=False), elapsed

    # Layout: results/<model_safe>/{MODEL_<safe>_DATASET_<dsid>.jsonl, <dsid>/output_<id>.wav}.
    # Manifest paths are relative to model_dir so later stages resolve them wherever it is copied.
    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    # Serial synthesis: one sample per "batch"; warm-up runs untimed on the first pending samples.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t, args.seed) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)
    set_seed(args.seed)  # so warm-up does not shift the RNG stream of the timed run

    # ── Main loop: synthesize one sample at a time → write JSONL ─────────────
    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]

        # Store the path relative to model_dir (the manifest's dir); write to the full path.
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        path = os.path.join(model_dir, rel_path)

        # Per-call seed (base seed + dataset row index): deterministic and independent of --resume.
        audio, elapsed = synth_one(text, args.seed + i)
        sf.write(path, audio, SAMPLING_RATE)

        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLING_RATE, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="sulabhkatiyar/indic-tiny-tts-hindi", help="HF repo id of the model.")
    parser.add_argument("--revision", type=str, default=None,
                        help="Model repo revision (full commit sha) to pin; submit_jobs.sh always passes one.")
    parser.add_argument("--voice", type=str, default="male", help="Built-in voice: 'female' or 'male'.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier.")
    add_common_args(parser)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
