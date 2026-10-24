"""
TTS synthesis for the Open TTS Leaderboard (OmniVoice backend, stage 1).

OmniVoice (k2-fsa/OmniVoice) is a ~0.6B diffusion-LM TTS loaded in-process via the
`omnivoice` package. It supports native batched generation, so texts are synthesized in
minibatches (the batch call is timed for RTFx).

`--voice_clone` clones each sample's `prompt_audio`/`prompt_text` reference and adds a
`_voice_clone` suffix to outputs; by default (`--no-voice_clone`) OmniVoice picks its own speaker (no SIM).

Stage 2 (`transformers/transcribe.py`) fills `pred_text` with Qwen3-ASR; stage 3
(`transformers/score_similarity.py`, voice-clone only) fills `sim`; stage 4
(`normalizer.eval_utils.score_results`) computes WER + RTFx (+ mean SIM).
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from omnivoice import OmniVoice

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, wav_rel_path, warm_up, write_entry,
)

torch.set_float32_matmul_precision("high")

# Dataset codes OmniVoice knows under another id: its language list has no bare 'ar', only Arabic
# varieties, and Modern Standard Arabic is 'arb' (k2-fsa/OmniVoice docs/languages.md).
LANGUAGE_ID_OVERRIDES = {"ar": "arb"}


def main(args):
    # Set seed for reproducibility (diffusion sampling).
    set_seed(args.seed)

    torch_dtype = getattr(torch, args.dtype)
    model = OmniVoice.from_pretrained(args.model_id, device_map=args.device, dtype=torch_dtype)
    sampling_rate = int(model.sampling_rate)  # 24000
    print(f"Loaded OmniVoice ({args.model_id}, sr={sampling_rate})")
    if isinstance(model, torch.nn.Module):
        print_model_size(model)

    # Layout: results/<model_safe>/{manifest.jsonl, <dsid>/output_<id>.wav}; manifest paths are
    # relative to model_dir.
    # Keep the two modes' audio separate so voice-clone and auto-voice runs don't overwrite.
    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir

    # Zero-shot voice cloning references (prompt_audio) are written here as wavs so
    # `generate()` can take them as file paths and so stage 3 can score similarity.
    if args.voice_clone:
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)

    def generate_tts(texts, ids, prompt_audios, prompt_texts):
        """Synthesize a batch of texts (optionally from cloning refs); time it for RTFx.

        Returns manifest-ready paths (relative to model_dir) for the generated wavs and the
        cloning prompts; the full on-disk paths are only used for I/O and for OmniVoice's
        `ref_audio`, which needs a real readable path.
        """
        minibatch_size = len(texts)

        # Write the cloning prompts to wavs BEFORE timing (I/O is not generation).
        prompt_rel_paths, prompt_full_paths = [], []
        if args.voice_clone:
            for sample_id, prompt_audio in zip(ids, prompt_audios):
                prel = os.path.join(dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
                ppath = os.path.join(model_dir, prel)
                if not os.path.exists(ppath):
                    sf.write(ppath, np.asarray(prompt_audio["array"], dtype=np.float32), prompt_audio["sampling_rate"])
                prompt_rel_paths.append(prel)
                prompt_full_paths.append(ppath)
        else:
            prompt_rel_paths = [""] * minibatch_size
            prompt_full_paths = [""] * minibatch_size

        # START TIMING (TTS batch generation)
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Voice-cloning mode passes per-sample reference audio/text; auto-voice mode lets
        # OmniVoice pick its own speaker. Returns list[np.ndarray] @ 24kHz.
        gen_kwargs = dict(
            text=list(texts),
            language=[LANGUAGE_ID_OVERRIDES.get(args.language, args.language)] * minibatch_size,
            num_step=args.num_step,
        )
        if args.voice_clone:
            gen_kwargs["ref_audio"] = list(prompt_full_paths)
            gen_kwargs["ref_text"] = list(prompt_texts)
        audios = model.generate(**gen_kwargs)

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0
        per_sample_time = runtime / minibatch_size

        gen_paths, audio_length_s = [], []
        for audio, sample_id in zip(audios, ids):
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            audio = np.asarray(audio).reshape(-1)
            sf.write(os.path.join(model_dir, rel_path), audio, sampling_rate)
            gen_paths.append(rel_path)
            audio_length_s.append(len(audio) / sampling_rate)

        return gen_paths, prompt_rel_paths, audio_length_s, minibatch_size * [per_sample_time]

    # Keep `id` + the target text; for voice cloning also keep the reference
    # `prompt_audio`/`prompt_text`. Drop everything else.
    dataset = load_tts_dataset(args, ("prompt_text", "prompt_audio") if args.voice_clone else ())

    probe_out_dir = model_dir  # real results dir, captured before the probe rebinds model_dir
    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # TTFA is per-request, so generate_tts() is driven with one-row batches. model_dir and
    # dataset_dir_name are rebound to a temp dir (generate_tts reads them late-bound), so no
    # results are touched. No streaming API: `first` is None -> whole-utterance fallback.
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        model_dir = tempfile.mkdtemp(prefix="ttfa_out_")
        dataset_dir_name = "ttfa"
        output_dir = os.path.join(model_dir, dataset_dir_name)
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)

        def _gen(job):
            result = generate_tts(
                [job[args.text_column][0]], [job["id"][0]],
                [job["prompt_audio"][0] if args.voice_clone else None],
                [job["prompt_text"][0] if args.voice_clone else None],
            )
            return float(result[-2][0]), None, None

        columns = dataset.column_names
        jobs = [{c: [dataset[i][c]] for c in columns}
                for i in sample_indices(args.ttfa_probe, len(dataset))]

        run_probe(
            _gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=probe_out_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={"device": args.device, "note": "driven at batch size 1"},
        )
        return

    # ── Manifest (written incrementally; resume skips rows already in it) ───
    manifest_path = paths.manifest_path
    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    def run_batch(batch):
        n_batch = len(batch["id"])
        prompt_audios = batch["prompt_audio"] if args.voice_clone else [None] * n_batch
        prompt_texts = list(batch["prompt_text"]) if args.voice_clone else [None] * n_batch
        return generate_tts(list(batch[args.text_column]), batch["id"], prompt_audios, prompt_texts)

    # ── Main loop: TTS → write JSONL, one batch at a time ───────────────────
    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(run_batch, dataset, starts, args.batch_size, args.warmup_steps)
    for batch_start in tqdm(starts, desc="Generating"):
        batch = dataset[batch_start : batch_start + args.batch_size]
        texts = list(batch[args.text_column])
        gen_paths, prompt_paths, audio_length_s, gen_times = run_batch(batch)

        for afp, ppath, alen, gtime, ref in zip(gen_paths, prompt_paths, audio_length_s, gen_times, texts):
            entry = manifest_entry(afp, alen, gtime, ref, ppath if args.voice_clone else None)
            write_entry(manifest_file, entry)
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="k2-fsa/OmniVoice", help="OmniVoice model id.")
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language hint: an ISO id ('en', 'zh') or an English language name ('English', "
        "'Chinese'). NOTE: OmniVoice matches ids case-sensitively and silently falls back to "
        "language-agnostic generation (slightly worse) on an unrecognized value — so use "
        "lowercase ids ('zh', not 'ZH' or 'cn').",
    )
    parser.add_argument("--num_step", type=int, default=32, help="Diffusion steps (16 = faster, 32 = default).")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype, e.g. 'bfloat16'.")
    add_common_args(parser, batch_size=16, voice_clone=False)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
