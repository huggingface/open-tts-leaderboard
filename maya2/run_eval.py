"""
TTS synthesis for the Open TTS Leaderboard (Maya 2 Native backend, stage 1).

Maya 2 Native (maya-research/maya2-native) is a Qwen3-1.7B backbone that predicts the first Mimi
codebook of each 80 ms frame, with a depth decoder filling the other 31; Mimi decodes the frames
into 24 kHz audio. It targets 11 Indian languages and English. The Hub repo only holds the weights;
inference uses the `maya3` engine of the pinned GitHub repo (https://github.com/MayaResearch/maya2-native),
installed in the Docker image. Model card (usage): https://huggingface.co/maya-research/maya2-native

The engine does continuous batching: requests submitted together share up to `max_seats` decoding
seats. A minibatch of `--batch_size` rows is therefore submitted at once (`max_seats` =
`--batch_size`) and timed from submission until its last request ends. The engine splits long
text itself (~20 s pieces) and seeds sampling with its fixed default seed.

Text goes through the engine's bundled normalizer first (`normalize=True`, which the model card
requires for the Python API: digits, currency, dates, ...).

- `--voice_clone`: each row's `prompt_audio`/`prompt_text` becomes a one-off voice (`icl_spk`: the
  reference clip's Mimi codes in context + its ECAPA speaker embedding), built inside the timed
  region. The reference is in the target language in every benchmark here.
- default (`--no-voice_clone`): a built-in voice (`--voice`, default Ahana, trained on English and
  Hindi among others); no SIM.

Stage 2 (`transformers/transcribe.py`) fills `pred_text` with Qwen3-ASR; stage 3
(`transformers/score_similarity.py`, voice-clone only) fills `sim`; stage 4
(`normalizer.eval_utils.score_results`) computes WER + RTFx (+ mean SIM).
"""

import argparse
import os
import threading
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

SAMPLING_RATE = 24_000  # Mimi
# The model's languages (model card). Only en and hi overlap with the leaderboard's benchmarks.
LANGUAGES = {"as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"}


def main(args):
    from maya3 import Maya3TTS
    from maya3.text import normalise_text

    if args.split not in LANGUAGES:
        raise ValueError(f"Maya 2 Native does not speak split '{args.split}'.")

    set_seed(args.seed)
    tts = Maya3TTS.from_pretrained(args.model_id, device=args.device, max_seats=args.batch_size)
    engine = tts.engine
    print(f"Loaded Maya 2 Native ({args.model_id}, max_seats={args.batch_size}, voice="
          f"{'clone' if args.voice_clone else args.voice})")
    print_model_size(*(m for m in (getattr(engine, "model", None), getattr(engine, "mimi", None))
                       if isinstance(m, torch.nn.Module)))
    if not args.voice_clone and args.voice not in tts.voices:
        raise ValueError(f"Unknown voice {args.voice!r}; built-in voices: {sorted(tts.voices)}")

    def run_requests(texts, voices):
        """Submit every text at once (continuous batching); wait for all. Return per-row float32 audio."""
        chunks = [[] for _ in texts]
        done = [threading.Event() for _ in texts]
        errors = []

        def on_event(row):
            def handle(kind, payload):
                if kind == "audio":
                    chunks[row].append(payload)
                else:  # 'end' or 'error'
                    if kind == "error":
                        errors.append(payload)
                    done[row].set()
            return handle

        for row, (text, voice) in enumerate(zip(texts, voices)):
            engine.submit(normalise_text(text, args.split), args.split, voice, on_event=on_event(row))
        for event in done:
            if not event.wait(args.request_timeout):
                raise TimeoutError(f"a request did not finish within {args.request_timeout} s")
        if errors:
            raise RuntimeError(f"generation failed: {errors[0]}")
        return [np.concatenate(c).astype(np.float32) / 32767.0 if c else np.zeros(1, np.float32) for c in chunks]

    def generate_tts(texts, prompt_paths=None, prompt_texts=None):
        """Synthesize a batch; time voice building (cloning) + generation. Return (audios, per-sample s)."""
        torch.cuda.synchronize(args.device)
        start = time.perf_counter()
        if args.voice_clone:
            voices = [tts.make_voice(f"ref-{i}", ref_audio=path, ref_text=ref_text, lang=args.split)
                      for i, (path, ref_text) in enumerate(zip(prompt_paths, prompt_texts))]
        else:
            voices = [engine.registry.get(args.voice)] * len(texts)
        audios = run_requests(texts, voices)  # PCM arrives on the CPU: the audio is in hand
        elapsed = time.perf_counter() - start
        return audios, elapsed / len(texts)

    # Layout: results/<model_safe>/{manifest.jsonl, <dsid>/output_<id>.wav}; manifest paths are
    # relative to model_dir. The two modes' audio is kept apart so they don't overwrite each other.
    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path
    if args.voice_clone:
        os.makedirs(os.path.join(paths.output_dir, "prompts"), exist_ok=True)

    dataset = load_tts_dataset(args, ("prompt_text", "prompt_audio") if args.voice_clone else ())

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # Batch size 1, built-in voice; the engine streams PCM per frame, so the first chunk is the
    # first playable audio.
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        if args.voice_clone:
            raise ValueError("The TTFA probe runs with a built-in voice; drop --voice_clone.")
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices

        def _ttfa_gen(job, early_stop):
            chunks, first = [], None
            # stream() cancels the request when the generator is closed (early stop).
            for chunk in tts.stream(job["text"], lang=args.split, voice=args.voice, normalize=True):
                if first is None:
                    first = time.perf_counter()
                    if early_stop:
                        return chunk, SAMPLING_RATE, first
                chunks.append(chunk)
            return np.concatenate(chunks) if chunks else np.zeros(1), SAMPLING_RATE, first

        # The probe discards its first rows as warm-up but does not run one: compile the kernels
        # here so the first probed rows are not dominated by CUDA initialisation.
        for i in range(min(args.warmup_steps, len(dataset))):
            generate_tts([dataset[i][args.text_column]])

        run_probe(
            _ttfa_gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            extra={"device": args.device, "voice": args.voice},
        )
        tts.close()
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    def write_prompt(sample_id, prompt_audio):
        """Write a cloning reference to a wav (the engine reads it from a path; stage 3 scores against it).

        Returns (path relative to model_dir, full path). Done outside the timed region.
        """
        rel = os.path.join(dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
        full = os.path.join(model_dir, rel)
        if not os.path.exists(full):
            sf.write(full, np.asarray(prompt_audio["array"], dtype=np.float32), prompt_audio["sampling_rate"])
        return rel, full

    def run_batch(batch):
        """Synthesize one minibatch. Return (audios, per-sample time, prompt paths relative to model_dir)."""
        texts = list(batch[args.text_column])
        if not args.voice_clone:
            return (*generate_tts(texts), [None] * len(texts))
        prompts = [write_prompt(sid, audio) for sid, audio in zip(batch["id"], batch["prompt_audio"])]
        audios, per_sample = generate_tts(texts, [full for _, full in prompts], list(batch["prompt_text"]))
        return audios, per_sample, [rel for rel, _ in prompts]

    # ── Main loop: TTS → write JSONL, one batch at a time ───────────────────
    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(run_batch, dataset, starts, args.batch_size, args.warmup_steps)

    for batch_start in tqdm(starts, desc="Generating"):
        batch = dataset[batch_start : batch_start + args.batch_size]
        audios, per_sample_time, prompt_rels = run_batch(batch)

        for sample_id, text, audio, prompt_rel in zip(batch["id"], batch[args.text_column], audios, prompt_rels):
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            sf.write(os.path.join(model_dir, rel_path), audio, SAMPLING_RATE)
            write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLING_RATE, per_sample_time, text, prompt_rel))
        manifest_file.flush()

    manifest_file.close()
    tts.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="maya-research/maya2-native", help="Maya 2 Native repo id.")
    parser.add_argument("--voice", type=str, default="Ahana",
                        help="Built-in voice without --voice_clone (model card 'Voices'; Ahana is trained on English and Hindi).")
    parser.add_argument("--request_timeout", type=float, default=600, help="Seconds to wait for one batch's requests.")
    add_common_args(parser, batch_size=32, voice_clone=False)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
