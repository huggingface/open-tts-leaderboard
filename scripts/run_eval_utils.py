"""Helpers shared by every backend's run_eval.py (stage 1: TTS generation).

Injected into each generation job at /app/scripts/ (see `inject_run_eval` in
scripts/tts_jobs_common.sh), which the job puts on PYTHONPATH, so a backend simply does
`from run_eval_utils import ...`. To run a backend directly from the repo root:

    PYTHONPATH=scripts python kokoro/run_eval.py ...

Must stay importable without torch or `datasets` at module level: some images lack one or the other.
"""

import json
import os
import random
from types import SimpleNamespace

DEFAULT_WARMUP_STEPS = 2


# ── Reproducibility / reporting ──────────────────────────────────────────────
def set_seed(seed=42):
    """Seed Python, NumPy and (if installed) torch, on every CUDA device."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def count_parameters(*objs):
    """Sum unique parameters across the given nn.Modules (or the nn.Module attributes of wrappers)."""
    import torch

    modules = []
    for o in objs:
        if isinstance(o, torch.nn.Module):
            modules.append(o)
        elif o is not None:
            modules += [m for m in vars(o).values() if isinstance(m, torch.nn.Module)]
    seen, total = set(), 0
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


def print_model_size(*objs):
    """Print the parameter count of `objs` (see count_parameters); silent if it finds none."""
    n_params = count_parameters(*objs)
    if n_params:
        print(f"TTS model size: {n_params / 1e9:.2f}B parameters")


# ── Dataset ──────────────────────────────────────────────────────────────────
def load_tts_dataset(args, extra_columns=(), decode_audio=True):
    """Load `args.dataset_path` (config `args.dataset`, split `args.split`), ready for generation.

    Applies `--max_eval_samples`, adds a positional `id` when the dataset has none (it names the
    wavs), and keeps only `id`, `args.text_column` and `extra_columns`, so nothing else triggers
    audio decoding. `decode_audio=False` leaves `prompt_audio` as raw bytes.
    """
    from datasets import Audio, load_dataset

    kwargs = {"split": args.split}
    if args.dataset:
        kwargs["name"] = args.dataset
    dataset = load_dataset(args.dataset_path, **kwargs)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    if "id" not in dataset.column_names:
        dataset = dataset.add_column("id", list(range(len(dataset))))
    keep = {"id", args.text_column, *extra_columns}
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep])
    if not decode_audio and "prompt_audio" in dataset.column_names:
        dataset = dataset.cast_column("prompt_audio", Audio(decode=False))
    return dataset


# ── Output layout + manifest ─────────────────────────────────────────────────
def output_paths(args, mode_suffix=""):
    """Paths for one (model, dataset split): results/<model_safe>/{<manifest>.jsonl, <dataset_dir>/}.

    Manifest `audio_filepath`s are relative to `model_dir`, so later stages resolve them wherever
    the folder is copied. Creates `output_dir`.
    """
    model_safe = args.model_id.replace("/", "-")
    dataset_safe = args.dataset_path.replace("/", "-")
    model_dir = os.path.join("results", model_safe)
    dataset_dir_name = f"{dataset_safe}_{args.dataset}_{args.split}{mode_suffix}"
    output_dir = os.path.join(model_dir, dataset_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(
        model_dir,
        f"MODEL_{model_safe}_DATASET_{dataset_safe}_{args.dataset.replace('/', '-')}_{args.split}{mode_suffix}.jsonl",
    )
    return SimpleNamespace(
        model_safe=model_safe, model_dir=model_dir, dataset_dir_name=dataset_dir_name,
        output_dir=output_dir, manifest_path=manifest_path,
    )


def wav_rel_path(dataset_dir_name, sample_id):
    """Manifest `audio_filepath` of a sample's generated wav (relative to `model_dir`)."""
    return os.path.join(dataset_dir_name, f"output_{sample_id}.wav")


def load_done_entries(manifest_path, resume):
    """{audio_filepath: entry} already in the manifest when resuming, else {}."""
    done = {}
    if resume and os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("audio_filepath"):
                        done[entry["audio_filepath"]] = entry
        print(f"Resuming: {len(done)} samples already in manifest.")
    return done


def open_manifest(manifest_path, done_entries):
    """Open the manifest for appending (resume) or fresh writing."""
    return open(manifest_path, "a" if done_entries else "w", encoding="utf-8")


def manifest_entry(audio_filepath, duration, time, text, prompt_audio_filepath=None):
    """One manifest row. `pred_text` is filled by stage 2 (ASR); `sim` by stage 3 (voice cloning)."""
    entry = {"audio_filepath": audio_filepath, "duration": duration, "time": time, "text": text, "pred_text": ""}
    if prompt_audio_filepath is not None:
        entry["prompt_audio_filepath"] = prompt_audio_filepath
        entry["sim"] = ""
    return entry


def write_entry(manifest_file, entry):
    manifest_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Batching / warm-up ───────────────────────────────────────────────────────
def pending_batches(dataset, batch_size, done_entries, dataset_dir_name):
    """Start indices of the batches with at least one sample not yet in the manifest."""
    ids = dataset["id"]
    return [
        start for start in range(0, len(ids), batch_size)
        if not all(wav_rel_path(dataset_dir_name, sid) in done_entries for sid in ids[start : start + batch_size])
    ]


def warm_up(generate_fn, dataset, batch_starts, batch_size, steps=DEFAULT_WARMUP_STEPS):
    """Run `generate_fn(batch)` untimed on the first `steps` pending batches.

    Only pending batches are used, so anything warm-up writes is regenerated (and timed) by the
    main loop, and resumed rows are never touched.
    """
    from tqdm import tqdm

    for start in tqdm(batch_starts[:steps], desc="Warming up..."):
        generate_fn(dataset[start : start + batch_size])


def add_common_args(parser, batch_size=None, voice_clone=None):
    """Add the flags every backend shares. `batch_size` / `voice_clone` add those flags when given
    (their value is the default). Change another default with `parser.set_defaults(...)`."""
    import argparse

    parser.add_argument("--dataset_path", type=str, default="hf-audio/seed_tts_eval",
                        help="Dataset repo id (scripts/prepare_seed_tts_eval.py or prepare_cv3_eval.py).")
    parser.add_argument("--dataset", type=str, default="tts",
                        help="Dataset config: 'tts' (Seed-TTS) or 'zero_shot' (CV3-Eval).")
    parser.add_argument("--split", type=str, default="en", help="Split / language, e.g. 'en'.")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column holding the target text to synthesize (the WER reference).")
    parser.add_argument("--device", type=str, default="cuda", help="'cuda', 'cpu', or 'cuda:0'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Evaluate only the first N samples (e.g. 8 for testing).")
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_WARMUP_STEPS,
                        help="Untimed warm-up batches before generation (excluded from RTFx).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples already in the manifest (resume after OOM/crash).")
    parser.add_argument("--ttfa_probe", type=int, default=0,
                        help="Measure time-to-first-audio: N evenly spaced samples, or -1 for the whole "
                             "split; 0 disables. Writes ONLY a JSON sidecar — no wavs, no manifest.")
    if batch_size is not None:
        parser.add_argument("--batch_size", type=int, default=batch_size,
                            help="Samples per batch (also the manifest-writing / resume granularity).")
    if voice_clone is not None:
        parser.add_argument("--voice_clone", action=argparse.BooleanOptionalAction, default=voice_clone,
                            help="Clone each sample's prompt_audio/prompt_text speaker (adds SIM scoring "
                                 "and the '_voice_clone' suffix). Default: the model's own voice.")


def print_next_steps(voice_clone):
    """Final message of stage 1, naming the remaining pipeline stages."""
    if voice_clone:
        print("Stage 1 (TTS generation) complete. Run stage 2 (transcribe), stage 3 (SIM), stage 4 (score) next.")
    else:
        print("Stage 1 (TTS generation) complete. Run stage 2 (transcribe) + stage 4 (score) next (no SIM).")
