"""
TTS synthesis for the Open TTS Leaderboard (Reverb-TTS backend, stage 1).

Reverb-TTS (https://huggingface.co/software-mansion/Reverb-TTS) is a 297M masked-diffusion TTS derived from OmniVoice,
run in-process through the `reverb-tts` package (usage: https://github.com/software-mansion-labs/Reverb-TTS). It
generates in batches, so texts are synthesized in minibatches (the batch call is timed for RTFx). The package reads
numbers, dates and abbreviations out as words before generating, as it does for users.

`--voice_clone` clones each sample's `prompt_audio`/`prompt_text` reference and adds a `_voice_clone` suffix to the
outputs; encoding the reference is timed with the generation. Without it, every sample uses the preset voice for the
split's language (`VOICES`), loaded once before timing (no SIM).

Weights in fp32, computed in the package's default dtype for the device: bfloat16 on a GPU, float32 on a CPU.

`--ttfa_probe N` measures time to first audio instead (submit_ttfa_jobs.sh): Reverb doesn't stream, so it times whole
utterances at batch size 1, through the same generate_tts as the eval, and writes only a JSON sidecar.

Stage 2 (`transformers/transcribe.py`) fills `pred_text` with Qwen3-ASR; stage 3 (`transformers/score_similarity.py`,
voice-clone only) fills `sim`; stage 4 (`normalizer.eval_utils.score_results`) computes WER + RTFx (+ mean SIM).
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from reverb_tts import TTS

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, wav_rel_path, warm_up, write_entry,
)

torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_cudnn_sdp(False)  # no cuDNN attention: with it off, Reverb measured 203x real time on the H200

# The preset voice per language when not cloning: the best of Reverb's presets on MiniMax (WER under Qwen3-ASR and
# Whisper, then UTMOS), one voice per language so every split hears a native voice.
VOICES = {
    "en": "en_female_narrator",
    "de": "de_male_standard",
    "es": "es_male_standard_1",
    "fr": "fr_female_standard",
    "it": "it_male_expressive",
    "pl": "pl_male_standard",
    "pt": "pt_female_standard",
}


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device=device)


def main(args):
    set_seed(args.seed)
    tts = TTS.from_pretrained(args.model_id, variant="fp32", device=args.device)  # weights at the package's pinned release
    sampling_rate = tts.sample_rate  # 24000
    print(f"Loaded Reverb-TTS ({args.model_id}, fp32 as {next(tts.network.parameters()).dtype}, sr={sampling_rate})")
    print_model_size(tts.network)

    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir
    if args.voice_clone:
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)
        preset = None
    else:
        preset = tts.preset_voice(VOICES[args.language])
        print(f"Preset voice: {VOICES[args.language]}")

    def generate_tts(texts, ids, prompt_audios, prompt_texts):
        """Synthesizes a batch, timed for RTFx; returns manifest-ready paths, lengths and per-sample times."""
        n = len(texts)
        prompt_rel_paths = [""] * n
        if args.voice_clone:  # the prompts are written for stage 3 before timing: I/O is not generation
            prompt_rel_paths = []
            for sample_id, prompt_audio in zip(ids, prompt_audios):
                rel = os.path.join(dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
                full = os.path.join(model_dir, rel)
                if not os.path.exists(full):
                    sf.write(full, np.asarray(prompt_audio["array"], dtype=np.float32), prompt_audio["sampling_rate"])
                prompt_rel_paths.append(rel)

        synchronize(args.device)
        start = time.perf_counter()
        if args.voice_clone:
            voices = [tts.voice(np.asarray(a["array"], dtype=np.float32), t, args.language, sample_rate=a["sampling_rate"])
                      for a, t in zip(prompt_audios, prompt_texts)]
        else:
            voices = [preset] * n
        audios = tts.generate(list(texts), voices, args.language)
        synchronize(args.device)
        per_sample_time = (time.perf_counter() - start) / n

        gen_paths, audio_length_s = [], []
        for audio, sample_id in zip(audios, ids):
            rel = wav_rel_path(dataset_dir_name, sample_id)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            sf.write(os.path.join(model_dir, rel), audio, sampling_rate)
            gen_paths.append(rel)
            audio_length_s.append(len(audio) / sampling_rate)
        return gen_paths, prompt_rel_paths, audio_length_s, n * [per_sample_time]

    dataset = load_tts_dataset(args, ("prompt_text", "prompt_audio") if args.voice_clone else ())

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # Reverb doesn't stream, so time to first audio is the time to the whole utterance (first audio = None). Driven
    # at batch size 1 through generate_tts, so it times the eval's code path. model_dir and dataset_dir_name are
    # rebound to a temp folder (generate_tts reads them when called), so no results are touched.
    if args.ttfa_probe != 0:
        import tempfile
        from ttfa_probe import run_probe, sample_indices  # injected only by submit_ttfa_jobs.sh

        probe_out_dir = model_dir  # the real results folder, for the sidecar
        model_dir, dataset_dir_name = tempfile.mkdtemp(prefix="ttfa_out_"), "ttfa"
        os.makedirs(os.path.join(model_dir, dataset_dir_name, "prompts"), exist_ok=True)

        def one(row):
            _, _, lengths, _ = generate_tts(
                [row[args.text_column]], [row["id"]],
                [row["prompt_audio"] if args.voice_clone else None],
                [row["prompt_text"] if args.voice_clone else None],
            )
            return lengths[0], None  # seconds of audio; None: no streaming timestamp

        run_probe(
            one, [dataset[i] for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path, dataset_config=args.dataset, split=args.split,
            out_dir=probe_out_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={"device": args.device, "note": "whole utterances at batch size 1; Reverb doesn't stream"},
        )
        return

    manifest_path = paths.manifest_path
    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    def run_batch(batch):
        n = len(batch["id"])
        prompt_audios = batch["prompt_audio"] if args.voice_clone else [None] * n
        prompt_texts = list(batch["prompt_text"]) if args.voice_clone else [None] * n
        return generate_tts(list(batch[args.text_column]), batch["id"], prompt_audios, prompt_texts)

    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(run_batch, dataset, starts, args.batch_size, args.warmup_steps)
    for batch_start in tqdm(starts, desc="Generating"):
        batch = dataset[batch_start: batch_start + args.batch_size]
        gen_paths, prompt_paths, lengths, times = run_batch(batch)
        for afp, ppath, alen, gtime, ref in zip(gen_paths, prompt_paths, lengths, times, batch[args.text_column]):
            write_entry(manifest_file, manifest_entry(afp, alen, gtime, ref, ppath if args.voice_clone else None))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="software-mansion/Reverb-TTS", help="Reverb-TTS model repo.")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code: en, de, fr, es, it, pl or pt.")
    add_common_args(parser, batch_size=32, voice_clone=False)
    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)
    main(args)
