"""
TTS synthesis for the Open TTS Leaderboard (Ming-3.0-tts backend, stage 1).

Ming-3.0-tts-0.5B (inclusionAI/Ming-3.0-tts-0.5B) is a dense Qwen2.5-0.5B backbone with a
flow-matching (DiT) head: each autoregressive step samples a patch of continuous audio latents,
which a VAE audio tokenizer decodes into 44.1 kHz audio. The Hub repo only holds the checkpoint;
inference uses the `ming_tts.MingTTS` API of the pinned GitHub repo
(https://github.com/inclusionAI/Ming-3.0-tts), installed in the Docker image.

Two modes, both through `MingTTS.synthesize` with `raw_text=True` (the text in one piece, without
text normalization or splitting — how the model's published benchmarks were run):
- `--voice_clone`: the `TTS` task cloning each sample's `prompt_audio`/`prompt_text`, with the
  CAM++ speaker embedding (`use_spk_emb`). The reference's trailing silence is trimmed first (Ming's
  own `trim_trailing_silence`): the tail of the reference seeds the first decode step, and the
  MiniMax references end in ~1 s of silence, which made the model stop at once (French: 0.3 s
  outputs, 99% WER on 8 rows; 3.5% trimmed).
- default (`--no-voice_clone`): the `IP` task with a built-in voice (no SIM; see VOICES). The
  documented built-in voices are English and Chinese; the other five languages use the English
  voice with `language_id` (cross-lingual; on 8 MiniMax rows each: de 0%, es 3.4%, fr 3.5% WER,
  ko 1.0%, ja 8.6% CER).

Both modes cover Chinese, English, German, Spanish, French, Korean and Japanese; the last five pass
`language_id`, which the model requires for them.

Generation is batch-1 only (the model's own code assumes it), so rows are processed serially. The
timed region covers what `synthesize` does per row: reference loading + encoding (voice cloning),
autoregressive sampling and decoding.

Stage 2 (`transformers/transcribe.py`) fills `pred_text` with Qwen3-ASR; stage 3
(`transformers/score_similarity.py`, voice-clone only) fills `sim`; stage 4
(`normalizer.eval_utils.score_results`) computes WER + RTFx (+ mean SIM).
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
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

# Languages that need an explicit `language_id` (tts_tasks.MULTILINGUAL_LANGUAGE_IDS); Chinese and
# English work without one.
MULTILINGUAL = {"ja", "ko", "de", "fr", "es"}
LANGUAGES = {"en", "zh"} | MULTILINGUAL
# Built-in voices (README "Recommended voices"): a female voice per language. Only English and
# Chinese voices exist, so the other languages speak with the English one (cross-lingual).
VOICES = {"en": "US_Emma", "zh": "灵小柔", **{lang: "US_Emma" for lang in MULTILINGUAL}}


def main(args):
    sys.path.insert(0, args.ming_dir)
    import tts_tasks
    from ming_tts import SPEAKER_EMBEDDING_DIM, MingTTS, trim_trailing_silence

    if args.split not in LANGUAGES:
        raise ValueError(f"Ming-3.0-tts does not speak split '{args.split}'.")

    set_seed(args.seed)
    # cache_prompts=False: every row encodes its own reference inside the timed region, like the
    # other voice-cloning backends (each Seed-TTS / CV3 row has its own prompt anyway).
    tts = MingTTS(args.model_id, device=args.device, text_normalization=False, cache_prompts=False)
    sampling_rate = int(tts.sample_rate)  # 44100
    print(f"Loaded Ming-3.0-tts ({args.model_id}, sr={sampling_rate})")
    print_model_size(tts.model)

    language_id = args.split if args.split in MULTILINGUAL else None
    sampling = dict(cfg=args.cfg, sigma=args.sigma, temperature=args.temperature)

    def request(text, prompt_path=None, prompt_text=None):
        """`synthesize` kwargs for one row in the selected mode."""
        if args.voice_clone:
            return dict(task="TTS", text=text, language_id=language_id, prompt_wav_path=prompt_path,
                        prompt_text=prompt_text, use_spk_emb=True, **sampling)
        return dict(task="IP", text=text, language_id=language_id,
                    controls={"ip": args.voice or VOICES[args.split]}, use_zero_spk_emb=True, **sampling)

    def synth_one(text, prompt_path=None, prompt_text=None):
        """Synthesize one text. Return (audio_1d_numpy, elapsed_s)."""
        torch.cuda.synchronize(args.device)
        start = time.perf_counter()
        # synthesize() returns a CPU tensor, so the clock stops once the audio is materialised.
        audio = tts.synthesize(raw_text=True, max_decode_steps=args.max_decode_steps,
                               **request(text, prompt_path, prompt_text))
        elapsed = time.perf_counter() - start
        return audio.reshape(-1).numpy().astype(np.float32), elapsed

    def synth_one_streaming(text, early_stop=False):
        """TTFA probe only (built-in voice): return (audio_1d_numpy, first_audio_ts).

        Mirrors synthesize() for the IP task, but drives the model's step generator directly and
        feeds each sampled latent patch to the VAE decoder's streaming path (`use_cache` +
        `stream_state`); the first time that returns samples is the first playable audio. The
        whole utterance is then decoded in one pass, as synthesize() does (chunked decoding gives
        slightly different audio at the chunk boundaries).
        """
        req = tts_tasks.build_task_generation_kwargs(**request(text))
        model = tts.model
        latents, first, chunk = [], None, None
        cache, state = None, (None, None, None)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for latent, _ in model.sample(
                prompt=req["prompt"], text=req["text"], spk_emb=[torch.zeros(1, SPEAKER_EMBEDDING_DIM)],
                instruction=req["instruction"], max_decode_steps=args.max_decode_steps,
                cfg=req["cfg"], sigma=req["sigma"], temperature=req["temperature"],
                use_zero_spk_emb=True,
            ):
                latents.append(latent)
                if first is None:
                    chunk, state, cache = model.audio.decode(
                        latent, past_key_values=cache, use_cache=True, stream_state=state, last_chunk=False)
                    if chunk.numel():
                        chunk = chunk.float().cpu().numpy().reshape(-1)  # .cpu() syncs
                        first = time.perf_counter()
                        if early_stop:
                            return chunk, first
            speech = model.audio.decode(torch.cat(latents, dim=1), past_key_values=None,
                                        use_cache=False, last_chunk=True)[0]
        audio = trim_trailing_silence(speech.cpu().float()[0], sampling_rate).reshape(-1).numpy()
        return audio, first or time.perf_counter()

    # Layout: results/<model_safe>/{manifest.jsonl, <dsid>/output_<id>.wav}; manifest paths are
    # relative to model_dir. The two modes' audio is kept apart so they don't overwrite each other.
    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path
    if args.voice_clone:
        os.makedirs(os.path.join(paths.output_dir, "prompts"), exist_ok=True)

    dataset = load_tts_dataset(args, ("prompt_text", "prompt_audio") if args.voice_clone else ())

    def write_prompt(sample_id, prompt_audio):
        """Write a cloning reference to a wav (MingTTS reads it from a path; stage 3 scores against it),
        with its trailing silence trimmed (see the module docstring).

        Returns (path relative to model_dir, full path). Done outside the timed region.
        """
        rel = os.path.join(dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
        full = os.path.join(model_dir, rel)
        if not os.path.exists(full):
            audio = torch.from_numpy(np.asarray(prompt_audio["array"], dtype=np.float32)).reshape(1, -1)
            audio = trim_trailing_silence(audio, prompt_audio["sampling_rate"])
            sf.write(full, audio[0].numpy(), prompt_audio["sampling_rate"])
        return rel, full

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # Built-in voice only (the leaderboard's TTFA runs without voice cloning).
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        if args.voice_clone:
            raise ValueError("The TTFA probe runs with a built-in voice; drop --voice_clone.")
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices

        def _ttfa_gen(job, early_stop):
            audio, first = synth_one_streaming(job["text"], early_stop)
            return audio, sampling_rate, first

        # The probe discards its first rows as warm-up but does not run one: compile the kernels
        # here so the first probed rows are not dominated by CUDA initialisation.
        for i in range(min(args.warmup_steps, len(dataset))):
            synth_one(dataset[i][args.text_column])
        set_seed(args.seed)

        run_probe(
            _ttfa_gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            extra={"device": args.device, "voice": args.voice or VOICES[args.split]},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    def run_row(row):
        """Synthesize one dataset row. Return (audio, elapsed, prompt path relative to model_dir)."""
        if not args.voice_clone:
            return (*synth_one(row[args.text_column]), None)
        prompt_rel, prompt_full = write_prompt(row["id"], row["prompt_audio"])
        return (*synth_one(row[args.text_column], prompt_full, row["prompt_text"]), prompt_rel)

    # Serial synthesis: one sample per "batch"; warm-up runs untimed on the first pending samples.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: run_row({k: v[0] for k, v in batch.items()}), dataset, starts, 1, args.warmup_steps)
    set_seed(args.seed)  # so warm-up does not shift the RNG stream of the timed run

    # ── Main loop: synthesize one sample at a time → write JSONL ─────────────
    for i in tqdm(starts, desc="Generating"):
        row = dataset[i]
        audio, elapsed, prompt_rel = run_row(row)

        # Store the path relative to model_dir (the manifest's dir); write to the full path.
        rel_path = wav_rel_path(dataset_dir_name, row["id"])
        sf.write(os.path.join(model_dir, rel_path), audio, sampling_rate)

        entry = manifest_entry(rel_path, len(audio) / sampling_rate, elapsed, row[args.text_column], prompt_rel)
        write_entry(manifest_file, entry)
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="inclusionAI/Ming-3.0-tts-0.5B", help="Ming-3.0-tts repo id.")
    parser.add_argument("--ming_dir", type=str, default=os.environ.get("MING_DIR", "/opt/Ming-3.0-tts"),
                        help="Checkout of https://github.com/inclusionAI/Ming-3.0-tts (provides ming_tts).")
    parser.add_argument("--voice", type=str, default=None,
                        help="Built-in voice (`ip`) without --voice_clone; default: VOICES[split].")
    # Sampling defaults from the repo's infer.py / MingTTS.synthesize.
    parser.add_argument("--cfg", type=float, default=2.0, help="Classifier-free guidance scale.")
    parser.add_argument("--sigma", type=float, default=0.25, help="Flow-matching sampling sigma.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Flow-matching sampling temperature.")
    parser.add_argument("--max_decode_steps", type=int, default=1000, help="Max autoregressive steps per sample.")
    add_common_args(parser, voice_clone=False)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
