"""
TTS synthesis for the Open TTS Leaderboard (Echo-TTS backend, stage 1).

Echo-TTS (jordand/echo-tts-base) is a diffusion TTS with speaker-reference conditioning, run with
the reference code from https://github.com/jordandare/echo-tts (on PYTHONPATH in the image).
With `--voice_clone` each sample is conditioned on its `prompt_audio`; by default
(`--no-voice_clone`) the speaker latent is zeroed, so the model picks its own speaker (no SIM).

`sample_pipeline()` takes a single text, so rows are processed serially.
"""

import argparse
import os
from functools import partial

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from inference import (
    load_fish_ae_from_hf, load_model_from_hf, load_pca_state_from_hf, sample_euler_cfg_independent_guidances,
    sample_pipeline,
)

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

# Echo-TTS outputs 44.1 kHz audio (the Fish S1 DAC autoencoder rate).
SAMPLING_RATE = 44_100


def main(args):
    set_seed(args.seed)

    model = load_model_from_hf(args.model_id, device=args.device, delete_blockwise_modules=True)
    fish_ae = load_fish_ae_from_hf(device=args.device)
    pca_state = load_pca_state_from_hf(args.model_id, device=args.device)
    print(f"Loaded Echo-TTS ({args.model_id}, num_steps={args.num_steps})")
    print_model_size(model)

    # Model-card defaults.
    sample_fn = partial(
        sample_euler_cfg_independent_guidances,
        num_steps=args.num_steps,
        cfg_scale_text=args.cfg_scale_text,
        cfg_scale_speaker=args.cfg_scale_speaker,
        cfg_min_t=0.5,
        cfg_max_t=1.0,
        truncation_factor=None,
        rescale_k=None,
        rescale_sigma=None,
        speaker_kv_scale=None,
        speaker_kv_max_layers=None,
        speaker_kv_min_t=None,
        sequence_length=640,  # ~30 s; shorter texts are padded by the model and cropped
    )

    def synth_one(text, prompt_audio):
        """Synthesize one text (optionally cloning `prompt_audio`). Return (audio_1d_numpy, elapsed_s)."""
        speaker_audio = None
        if prompt_audio is not None:
            # Same preprocessing as inference.load_audio: mono, 44.1 kHz, peak <= 1.
            speaker_audio = torch.tensor(prompt_audio["array"], dtype=torch.float32).reshape(1, -1)
            speaker_audio = torchaudio.functional.resample(speaker_audio, prompt_audio["sampling_rate"], SAMPLING_RATE)
            speaker_audio = (speaker_audio / max(speaker_audio.abs().max().item(), 1.0)).to(args.device)

        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        audio, _ = sample_pipeline(
            model=model, fish_ae=fish_ae, pca_state=pca_state, sample_fn=sample_fn,
            text_prompt=text, speaker_audio=speaker_audio, rng_seed=args.seed,
        )
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        return audio[0].reshape(-1).float().cpu().numpy(), start_event.elapsed_time(end_event) / 1000.0

    # Keep the two modes' audio separate so voice-clone and auto-voice runs don't overwrite.
    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir
    if args.voice_clone:
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)

    dataset = load_tts_dataset(args, ("prompt_audio",) if args.voice_clone else ())

    manifest_path = paths.manifest_path
    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    # Serial synthesis: one sample per "batch"; warm-up runs untimed on the first pending samples.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: synth_one(batch[args.text_column][0], batch["prompt_audio"][0] if args.voice_clone else None),
            dataset, starts, 1, args.warmup_steps)

    # ── Main loop: synthesize one sample at a time → write JSONL ─────────────
    for i in tqdm(starts, desc="Generating"):
        row = dataset[i]
        text = row[args.text_column]

        prompt_rel = None
        if args.voice_clone:
            # Written for stage 3 (SIM); not timed.
            prompt_rel = os.path.join(dataset_dir_name, "prompts", f"prompt_{row['id']}.wav")
            pa = row["prompt_audio"]
            sf.write(os.path.join(model_dir, prompt_rel), np.asarray(pa["array"], dtype=np.float32), pa["sampling_rate"])

        audio, elapsed = synth_one(text, row["prompt_audio"] if args.voice_clone else None)
        rel_path = wav_rel_path(dataset_dir_name, row["id"])
        sf.write(os.path.join(model_dir, rel_path), audio, SAMPLING_RATE)

        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLING_RATE, elapsed, text, prompt_rel))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="jordand/echo-tts-base", help="Echo-TTS repo id.")
    parser.add_argument("--num_steps", type=int, default=40, help="Diffusion steps (model-card default 40).")
    parser.add_argument("--cfg_scale_text", type=float, default=3.0, help="Text CFG scale (model-card default 3.0).")
    parser.add_argument("--cfg_scale_speaker", type=float, default=8.0, help="Speaker CFG scale (model-card default 8.0).")
    add_common_args(parser, voice_clone=False)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
