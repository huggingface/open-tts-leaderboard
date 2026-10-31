"""Audio8 Realistic batch inference; use validate-results for offline artifact checks."""

import argparse
import importlib.metadata
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import sys

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest,
    output_paths, pending_batches, print_model_size, print_next_steps, set_seed,
    wav_rel_path, warm_up, write_entry,
)

MODEL_ID = "Edge0/Audio8-TTS-Realistic-0.6b"
MODEL_REVISION = "1e857568bfd4f02931aca3988ee773a44d2d26d3"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, batch_size=32, voice_clone=False)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION,
                        help="Immutable 40-character Hub commit, including code and bundled codec.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Default: 0.7 for CV3 zero_shot, 0.1 for Seed/MiniMax; explicit override allowed.")
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--retry_max_new_tokens", type=int, default=0,
                        help="One retry of unfinished rows; same sampling, larger token budget; 0 disables.")
    args = parser.parse_args(argv)
    if args.temperature is None:
        args.temperature = 0.7 if args.dataset == "zero_shot" else 0.1
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full immutable Hub commit SHA")
    if args.batch_size < 1 or args.warmup_steps < 0 or args.max_new_tokens < 1:
        parser.error("batch_size/max_new_tokens must be positive; warmup_steps must be nonnegative")
    if args.retry_max_new_tokens and args.retry_max_new_tokens <= args.max_new_tokens:
        parser.error("retry_max_new_tokens must be 0 or larger than max_new_tokens")
    if not 0 < args.temperature < float("inf") or not 0 < args.top_p <= 1 or args.top_k < 1:
        parser.error("invalid sampling parameters")
    if args.ttfa_probe:
        parser.error("TTFA is not implemented for this backend; no streaming result is claimed")
    return args


def take_rows(batch, indexes):
    return {key: [values[i] for i in indexes] for key, values in batch.items()}


def synthesize_batch(batch, processor, model, args, generator, synchronize):
    """Time processor + reference codec + AR + output codec, including one optional retry.

    No file I/O or dataset decoding here. One model.generate call receives the whole
    batch. Failed rows can be retried together, with all retry work counted in RTFx.
    Unfinished/empty/non-finite output fails the job instead of silently scoring truncation.
    """
    import numpy as np

    def attempt(rows, token_limit):
        kwargs = {"text": list(rows[args.text_column]), "return_tensors": "pt"}
        if args.voice_clone:
            kwargs.update(reference_audio=list(rows["prompt_audio"]),
                          reference_text=list(rows["prompt_text"]))
        inputs = processor(**kwargs)
        inputs = {name: value.to(args.device) for name, value in inputs.items()}
        output = model.generate(
            **inputs, max_new_tokens=token_limit, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, do_sample=True, generator=generator,
            return_dict_in_generate=True,
        )
        waveforms, lengths = model.decode_audio(output.codes)
        expected = len(rows["id"])
        if any(len(values) != expected for values in
               (lengths.tolist(), output.finished.tolist(), output.code_lengths.tolist())):
            raise RuntimeError("Model returned inconsistent batch metadata")
        # Include transfers to CPU and all codec work; persistence is outside this timer.
        arrays = [waveforms[i, :int(length)].float().cpu().numpy()
                  for i, length in enumerate(lengths.tolist())]
        return arrays, output.finished.tolist(), output.code_lengths.tolist()

    synchronize()
    start = time.perf_counter()
    arrays, finished, frame_counts = attempt(batch, args.max_new_tokens)
    attempts = [1] * len(arrays)
    retry_indexes = [i for i, ended in enumerate(finished) if not ended]
    if retry_indexes and args.retry_max_new_tokens:
        retry_arrays, retry_finished, retry_frames = attempt(
            take_rows(batch, retry_indexes), args.retry_max_new_tokens,
        )
        for i, audio, ended, frames in zip(retry_indexes, retry_arrays, retry_finished, retry_frames):
            arrays[i], finished[i], frame_counts[i], attempts[i] = audio, ended, frames, 2
    synchronize()
    elapsed = time.perf_counter() - start
    if len(arrays) != len(batch["id"]):
        raise RuntimeError("Model returned a different batch size")
    failures = [str(batch["id"][i]) for i, ended in enumerate(finished)
                if not ended or frame_counts[i] < 1 or arrays[i].size == 0
                or not np.isfinite(arrays[i]).all()]
    if failures:
        raise RuntimeError("Unfinished (NO_EOS), empty or non-finite output; refusing partial "
                           f"scoring for sample ids: {', '.join(failures)}")
    return arrays, elapsed, frame_counts, attempts


def validate_sample_ids(sample_ids):
    # Shared wav_rel_path embeds IDs directly in filenames.
    names = [str(sid) for sid in sample_ids]
    if len(names) != len(set(names)):
        raise ValueError("Dataset IDs must be unique")
    if any(not sid or sid in (".", "..") or "/" in sid or "\\" in sid for sid in names):
        raise ValueError("Dataset IDs must be safe single filename components")


def generation_settings(args):
    return {key: getattr(args, key) for key in (
        "model_id", "revision", "dtype", "device", "batch_size", "voice_clone", "seed",
        "temperature", "top_p", "top_k", "max_new_tokens", "retry_max_new_tokens",
        "dataset_path", "dataset", "split", "text_column", "warmup_steps", "max_eval_samples",
    )}


def main(args):
    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
    from transformers import AutoModel, AutoProcessor

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for leaderboard timing; no silent CPU fallback")
    if device.type == "cpu" and args.dtype != "float32":
        raise ValueError("CPU development runs require --dtype float32; not leaderboard RTFx")
    paths = output_paths(args, "_voice_clone" if args.voice_clone else "")
    metadata_path = Path(paths.manifest_path).with_suffix(".run.json")
    settings = generation_settings(args)
    done = load_done_entries(paths.manifest_path, args.resume)
    if done:
        if not metadata_path.is_file():
            raise ValueError("Resume requires the existing .run.json provenance file")
        prior = json.loads(metadata_path.read_text())
        if prior.get("settings") != settings:
            raise ValueError("Resume configuration differs from the recorded run")
        for entry in done.values():
            for key in ("audio_filepath", "prompt_audio_filepath"):
                if entry.get(key) and not Path(paths.model_dir, entry[key]).is_file():
                    raise FileNotFoundError(f"Resume manifest references a missing file: {entry[key]}")

    # _codec_path() in this revision does not pass revision to cached_file(). Loading
    # a pinned LOCAL snapshot fixes model, remote code, processor AND codec.pth together.
    snapshot = snapshot_download(
        args.model_id, revision=args.revision,
        allow_patterns=["*.json", "*.py", "*.safetensors", "*.pth", "*.model", "*.txt"],
    )
    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True, dtype=getattr(torch, args.dtype),
    ).eval().to(device)
    # Make local codec resolution explicit even if a future loader changes config._name_or_path.
    model.config._name_or_path = snapshot
    codec = model.load_codec(device=device)
    sampling_rate = int(model.config.codec_sample_rate)
    if sampling_rate != 44100:
        raise ValueError(f"Expected 44.1 kHz Realistic codec, got {sampling_rate}")
    print("Main model (release size excludes codec):")
    print_model_size(model)
    print("Bundled codec (reported separately):")
    print_model_size(codec)
    print(f"Audio8: {args.model_id}@{args.revision}; batch={args.batch_size}; sr={sampling_rate}")

    dataset = load_tts_dataset(args, ("prompt_text", "prompt_audio") if args.voice_clone else ())
    if not len(dataset):
        raise ValueError("Evaluation dataset is empty")
    if args.text_column not in dataset.column_names:
        raise ValueError(f"Missing target text column: {args.text_column}")
    validate_sample_ids(dataset["id"])
    required = {"prompt_text", "prompt_audio"} if args.voice_clone else set()
    if not required.issubset(dataset.column_names):
        raise ValueError("Voice cloning requires prompt_text and prompt_audio columns")
    dataset_identity = {
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "rows": len(dataset),
        "ids_sha256": hashlib.sha256(json.dumps(list(dataset["id"]), ensure_ascii=False,
                                                default=str).encode()).hexdigest(),
        "texts_sha256": hashlib.sha256(json.dumps(list(dataset[args.text_column]),
                                                  ensure_ascii=False).encode()).hexdigest(),
    }
    if done and prior.get("dataset_identity") != dataset_identity:
        raise ValueError("Resume dataset differs from the recorded run")
    expected_paths = {wav_rel_path(paths.dataset_dir_name, sid) for sid in dataset["id"]}
    if not set(done).issubset(expected_paths):
        raise ValueError("Resume manifest contains IDs absent from the selected dataset")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def only_pending(batch):
        return take_rows(batch, [i for i, sid in enumerate(batch["id"])
                                 if wav_rel_path(paths.dataset_dir_name, sid) not in done])

    def generate(batch):
        return synthesize_batch(only_pending(batch), processor, model, args, generator, synchronize)

    metadata = {
        "backend": "audio8-transformers", "settings": settings,
        "dataset_identity": dataset_identity, "expected_rows": len(dataset),
        "sample_rate": sampling_rate, "codec_revision": args.revision,
        "timing": "wall time with CUDA sync; processor, reference codec, AR, output codec, "
                  "CPU transfer and retries included; loading, warmup and file I/O excluded",
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "huggingface_hub", "datasets", "numpy", "soundfile")},
        "status": "running", "rows": len(done),
        "retried_rows": sum(entry.get("generation_attempts", 1) > 1 for entry in done.values()),
        "hardware": {"device": str(device), "cuda_version": torch.version.cuda,
                     "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "note": "Resume reseeds the run; stochastic outputs need not match an uninterrupted run.",
    }
    def save_metadata():
        temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(metadata_path)

    save_metadata()
    starts = pending_batches(dataset, args.batch_size, done, paths.dataset_dir_name)
    try:
        with torch.inference_mode():
            warm_up(generate, dataset, starts, args.batch_size, args.warmup_steps)
            # Warmup should not consume the measured run's sampling stream.
            set_seed(args.seed)
            generator.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with open_manifest(paths.manifest_path, done) as manifest:
                for start in tqdm(starts, desc="Generating"):
                    batch = only_pending(dataset[start : start + args.batch_size])
                    prompt_paths = [None] * len(batch["id"])
                    if args.voice_clone:
                        prompt_dir = Path(paths.output_dir, "prompts")
                        prompt_dir.mkdir(parents=True, exist_ok=True)
                        for i, (sid, prompt) in enumerate(zip(batch["id"], batch["prompt_audio"])):
                            rel = os.path.join(paths.dataset_dir_name, "prompts", f"prompt_{sid}.wav")
                            audio = np.asarray(prompt["array"], dtype=np.float32)
                            if audio.ndim == 2:
                                audio = audio.mean(axis=0)  # datasets audio is channels-first
                            sf.write(Path(paths.model_dir, rel), audio, prompt["sampling_rate"], subtype="FLOAT")
                            prompt_paths[i] = rel
                    audios, elapsed, frames, attempts = synthesize_batch(
                        batch, processor, model, args, generator, synchronize,
                    )
                    for i, (sid, audio, text) in enumerate(zip(batch["id"], audios, batch[args.text_column])):
                        rel = wav_rel_path(paths.dataset_dir_name, sid)
                        sf.write(Path(paths.model_dir, rel), audio, sampling_rate, subtype="FLOAT")
                        entry = manifest_entry(rel, len(audio) / sampling_rate, elapsed / len(audios),
                                               text, prompt_paths[i])
                        entry.update(generation_status="OK", code_frames=int(frames[i]),
                                     generation_attempts=attempts[i], actual_batch_size=len(audios),
                                     model_revision=args.revision, sample_rate=sampling_rate)
                        write_entry(manifest, entry)
                    manifest.flush()
                    metadata["rows"] += len(audios)
                    metadata["retried_rows"] += sum(n > 1 for n in attempts)
                    save_metadata()
        if metadata["rows"] != len(dataset):
            raise RuntimeError("Generated row count does not match the selected dataset")
        metadata["status"] = "complete"
        if device.type == "cuda":
            metadata["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["error_type"] = type(exc).__name__
        metadata["error"] = str(exc)
        raise
    finally:
        save_metadata()
    print("Results saved at path:", os.path.abspath(paths.manifest_path))
    print_next_steps(args.voice_clone)


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path.name}")
    names = [row["audio_filepath"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate audio records: {path.name}")
    return rows


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_manifest(directory, name, language, args):
    if Path(name).name != name:
        raise ValueError("Expected manifests must be filenames, not paths")
    manifest = directory / name
    rows = read_rows(manifest)
    provenance = json.loads(manifest.with_suffix(".run.json").read_text(encoding="utf-8"))
    if provenance.get("status") != "complete":
        raise ValueError(f"Generation is not complete: {name}")
    if provenance.get("expected_rows") != len(rows) or provenance.get("rows") != len(rows):
        raise ValueError(f"Generation row count mismatch: {name}")
    settings = provenance["settings"]
    if settings.get("model_id") != args.model_id or settings.get("revision") != args.revision:
        raise ValueError(f"Model identity differs from this run: {name}")
    if settings.get("voice_clone") != args.voice_clone:
        raise ValueError(f"Voice-clone mode differs from this run: {name}")
    if "generate" in args.stages.split() and settings.get("max_eval_samples") != args.max_eval_samples:
        raise ValueError(f"Sample limit differs from this run: {name}")
    for row in rows:
        for key in ("audio_filepath", "prompt_audio_filepath"):
            if key in row and (Path(row[key]).is_absolute() or ".." in Path(row[key]).parts):
                raise ValueError(f"Unsafe {key}: {name}")
        if row.get("generation_status") != "OK" or row.get("model_revision") != args.revision:
            raise ValueError(f"Unverified generation row: {name}")
        if not all(finite_number(row.get(k)) and row[k] > 0 for k in ("duration", "time")):
            raise ValueError(f"Invalid duration or generation time: {name}")
        if row.get("sample_rate") != 44100 or not isinstance(row.get("text"), str):
            raise ValueError(f"Invalid waveform metadata or text: {name}")
        if not isinstance(row.get("pred_text"), str):
            raise ValueError(f"Invalid ASR field: {name}")
        if args.voice_clone and not row.get("prompt_audio_filepath"):
            raise ValueError(f"Missing reference audio record: {name}")
    stages = args.stages.split()
    if args.voice_clone and "sim" in stages:
        sim_path = manifest if args.sim_backend == "xvector" else manifest.with_name(
            f"{manifest.stem}_{args.sim_backend}.jsonl")
        sim_rows = read_rows(sim_path)
        baseline = {row["audio_filepath"]: row for row in rows}
        if {row["audio_filepath"] for row in sim_rows} != set(baseline):
            raise ValueError(f"SIM sample set differs: {name}")
        for row in sim_rows:
            original = baseline[row["audio_filepath"]]
            if any(row.get(k) != original.get(k) for k in
                   ("text", "pred_text", "duration", "time", "model_revision")):
                raise ValueError(f"Stale SIM manifest: {name}")
            if not finite_number(row.get("sim")) or not row.get("sim_model"):
                raise ValueError(f"Missing SIM score: {name}")
    if "utmos" in stages and language in args.utmos_langs.split():
        sidecar = json.loads((directory / f"UTMOS_{manifest.stem}.json").read_text())
        if any(not finite_number(sidecar["scores"].get(row["audio_filepath"])) for row in rows):
            raise ValueError(f"Missing UTMOS score: {name}")
    return {"manifest": name, "language": language, "rows": len(rows),
            "hardware": provenance.get("hardware"), "settings": settings}


def validate_results(args):
    directory = Path(args.directory).resolve()
    records = []
    by_language = {}
    for pair in args.expected:
        language, name = pair.split(":", 1)
        records.append(validate_manifest(directory, name, language, args))
        by_language.setdefault(language, []).append(name)
    scores = {}
    if not args.validate_only and args.stages.split() != ["generate"]:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from normalizer.eval_utils import score_results
        for language, names in by_language.items():
            _, per_dataset = score_results(str(directory), args.model_id, language=language,
                                           manifests=names, sim_backend=args.sim_backend)
            if len(per_dataset) != len(names):
                raise ValueError(f"Missing official scoring results for {language}")
            scores.update(per_dataset)
    summary = {"status": "validated", "model_id": args.model_id, "revision": args.revision,
               "stages": args.stages.split(), "score_computation": "official normalizer.eval_utils.score_results"
               if scores else "not run", "artifacts": records, "scores": scores,
               "limits": "Manifest and sidecar checks only; remote WAV contents are not checked here."}
    output = directory / f"audio8_review{'_voice_clone' if args.voice_clone else ''}.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output)
    print(f"Validated {len(records)} manifests; saved review results to {output}")


def parse_validation_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate Audio8 manifests and compute official scores without model loading.")
    parser.add_argument("--directory", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--expected", action="append", required=True, help="language:manifest.jsonl")
    parser.add_argument("--voice-clone", action="store_true")
    parser.add_argument("--stages", default="generate transcribe sim utmos")
    parser.add_argument("--sim-backend", choices=("wavlm_seed_tts", "xvector"), default="wavlm_seed_tts")
    parser.add_argument("--utmos-langs", default="en")
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument("--validate-only", action="store_true", help="Do not import/run the official scorer")
    return parser.parse_args(argv)


if __name__ == "__main__":
    if sys.argv[1:2] == ["validate-results"]:
        validate_results(parse_validation_args(sys.argv[2:]))
    else:
        main(parse_args())
