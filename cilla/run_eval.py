"""
TTS synthesis for the Open TTS Leaderboard (Cilla Persian backend, stage 1).

Cilla Mini / Cilla Lite (mehdi-hf/cilla-mini, mehdi-hf/cilla-lite) are small Persian models in the Pocket TTS
architecture (flow-matching LM over Mimi latents, 24 kHz output; Mini 18.2M parameters), loaded in-process with the
released `pocket-tts` package (>= 4.0.1). The HF repo ships the model config, the weights, three voice prompts and
the Persian text front-end (`normalize_fa.py` + `speak_fa.py`: text normalisation, one piece per sentence, silence
trimming). This script downloads the repo at a pinned revision, imports that front-end from the download and
calls its `speak()` for every row, so the leaderboard hears what the model's own scripts produce.

The weights are gated (CC BY-NC 4.0, access is granted automatically): the job's HF_TOKEN must belong to an account
that has accepted the terms at https://huggingface.co/mehdi-hf/cilla-mini.

The models use fixed voices (`voice1` a woman, `voice2` a man, `parisa`), each a 5-10 s prompt clip in the repo's
`prompts/`. Only the default-voice ("no cloning") table applies: one voice per run (`--voice`, default `voice1`).
Persian only: the front-end drops Latin text, so the splits of other languages must not be run.

Stage 2 (`transformers/transcribe.py`) fills `pred_text` with Qwen3-ASR; stage 4 scores it. Fixed voice: no SIM stage.
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
    add_common_args, count_parameters, load_done_entries, load_tts_dataset, manifest_entry, open_manifest,
    output_paths, pending_batches, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

HF_PREFIX = "hf://{repo}/"  # how the repo's config.yaml names its own files


def load_model(args):
    """Download the repo at the pinned revision, read its Persian front-end from there and build the model."""
    local_dir = snapshot_download(
        args.model_id, revision=args.revision,
        allow_patterns=["config.yaml", "model.safetensors", "tokenizer.model", "normalize_fa.py", "speak_fa.py",
                        f"prompts/{args.voice}.wav"],
    )
    print(f"Model snapshot: {local_dir} (requested revision: {args.revision or 'default branch'})")
    if args.revision and len(args.revision) == 40 and os.path.basename(local_dir) != args.revision:
        raise RuntimeError(f"snapshot {os.path.basename(local_dir)} does not match requested revision {args.revision}")

    # The shipped config points at hf://<repo>/..., which always resolves the default branch: point it at the
    # pinned snapshot instead.
    config = open(os.path.join(local_dir, "config.yaml"), encoding="utf-8").read()
    prefix = HF_PREFIX.format(repo=args.model_id)
    if prefix not in config:
        raise RuntimeError(f"config.yaml of {args.model_id} does not refer to {prefix}")
    local_config = os.path.join(local_dir, "config.pinned.yaml")
    with open(local_config, "w", encoding="utf-8") as f:
        f.write(config.replace(prefix, local_dir + "/"))

    sys.path.insert(0, local_dir)
    from pocket_tts import TTSModel

    model = TTSModel.load_model(config=local_config, temp=args.temperature, eos_threshold=args.eos_threshold)
    model.to(args.device)
    return model, os.path.join(local_dir, "prompts", f"{args.voice}.wav")


def main(args):
    if args.ttfa_probe != 0:
        raise SystemExit("--ttfa_probe is not implemented for this backend (sentence-by-sentence synthesis).")
    set_seed(args.seed)

    model, prompt_path = load_model(args)
    from speak_fa import speak  # the repo's own front-end, imported from the snapshot (see load_model)

    sampling_rate = int(model.mimi.sample_rate)
    print(f"Loaded Cilla ({args.model_id}, voice={args.voice}, sr={sampling_rate}, temp={model.temp}, device={model.device})")
    print(f"TTS model size: {count_parameters(model) / 1e6:.1f}M parameters (the file also holds the Mimi encoder)")
    is_cuda = torch.device(args.device).type == "cuda"

    # One voice for the whole run: encoded once, outside the timed region. `speak` copies the state for every piece.
    state = model.get_state_for_audio_prompt(prompt_path)
    print(f"Using fixed voice {args.voice!r} (no cloning, SIM stage skipped).")

    def synth_one(text):
        """Synthesize one text; return (audio_1d_numpy, elapsed_seconds). Normalisation, sentence splitting and
        silence trimming are inside the timed region: they are part of generating the text."""
        if is_cuda:
            torch.cuda.synchronize(device=args.device)
        start = time.perf_counter()
        audio, sr = speak(model, state, text)
        if is_cuda:
            torch.cuda.synchronize(device=args.device)
        elapsed = time.perf_counter() - start
        if sr != sampling_rate:
            raise RuntimeError(f"speak() returned {sr} Hz, expected {sampling_rate}")
        if audio.size == 0:
            print(f"WARNING: empty audio for text {text[:60]!r}")
            audio = np.zeros(1, dtype=np.float32)
        return np.asarray(audio, dtype=np.float32).reshape(-1), elapsed

    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path
    dataset = load_tts_dataset(args)

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    # Serial synthesis: warm-up runs untimed on the first pending rows.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda b: synth_one(b[args.text_column][0]), dataset, starts, 1, args.warmup_steps)

    n_written = 0
    for i in tqdm(starts, desc="Generating"):
        row = dataset[i]
        rel_path = wav_rel_path(dataset_dir_name, row["id"])
        path = os.path.join(model_dir, rel_path)

        # Re-seed per sample so generation is reproducible wherever a resume picked up.
        torch.manual_seed(args.seed)
        audio, elapsed = synth_one(row[args.text_column])
        sf.write(path, audio, sampling_rate)

        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / sampling_rate, elapsed, row[args.text_column]))
        n_written += 1
        if n_written % args.batch_size == 0:
            manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="mehdi-hf/cilla-mini",
                        help="Hub repo of the model (also used for manifest naming): mehdi-hf/cilla-mini or mehdi-hf/cilla-lite.")
    parser.add_argument("--revision", type=str, default="",
                        help="Commit sha of the model repo (submit_jobs.sh pins one). Empty = default branch.")
    # --batch_size is the manifest-flush granularity ONLY: generation is one sample at a time.
    add_common_args(parser, batch_size=32, voice_clone=False)
    parser.add_argument("--voice", type=str, default="voice1", choices=["voice1", "voice2", "parisa"],
                        help="Fixed voice: the repo's prompts/<voice>.wav (voice1 a woman, voice2 a man).")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (the repo's own default).")
    parser.add_argument("--eos_threshold", type=float, default=-2.0, help="End-of-sequence threshold (the repo's own default).")
    args = parser.parse_args()
    if args.voice_clone:
        raise SystemExit("Cilla is evaluated with its fixed voices only: run without --voice_clone.")

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
