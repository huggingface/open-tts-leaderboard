"""Generate API audio locally; use the existing ASR/SIM stages for quality scoring."""

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_eval_utils import (
    load_done_entries,
    load_tts_dataset,
    manifest_entry,
    output_paths,
    wav_rel_path,
)

from api.models import MODELS, get_model
from api.providers import APIError, SynthesisRequest, get_provider


def decode_audio(result):
    import numpy as np
    import soundfile as sf

    rate = result.sample_rate
    if result.encoding in ("pcm_s16le", "pcm_f32le"):
        width = 2 if result.encoding == "pcm_s16le" else 4
        if not result.audio_bytes or len(result.audio_bytes) % width:
            raise APIError("Provider returned incomplete PCM frames")
        if not rate or rate <= 0:
            raise APIError("Provider returned an invalid sample rate")
        dtype = "<i2" if width == 2 else "<f4"
        audio = np.frombuffer(result.audio_bytes, dtype=dtype).astype(np.float32)
        if width == 2:
            audio /= 32768.0
    else:
        try:
            audio, rate = sf.read(io.BytesIO(result.audio_bytes), dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
        except (sf.LibsndfileError, ValueError, RuntimeError) as exc:
            raise APIError("Provider returned undecodable audio") from exc
    if not len(audio) or not np.isfinite(audio).all() or rate <= 0:
        raise APIError("Provider returned empty or non-finite audio")
    return audio, rate


def reference_audio(sample):
    """Normalize the benchmark reference to WAV for providers and the shared SIM stage."""
    import numpy as np
    import soundfile as sf

    source = sample.get("prompt_audio")
    if isinstance(source, dict) and source.get("array") is not None:
        audio, rate = np.asarray(source["array"]), source["sampling_rate"]
    else:
        if isinstance(source, dict):
            source = io.BytesIO(source["bytes"]) if source.get("bytes") else source.get("path")
        source = source or sample.get("prompt_audio_filepath")
        if not source:
            raise ValueError("This model/mode requires prompt_audio or prompt_audio_filepath")
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    if np.ndim(audio) == 2:
        audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Reference audio must be non-empty and finite")
    out = io.BytesIO()
    sf.write(out, audio, rate, format="WAV", subtype="PCM_16")
    return out.getvalue(), sample.get("prompt_text", "")


def synthesize_with_retry(provider, model, request, *, max_retries=3, retry_delay=1):
    """Keep retries in the client latency; never substitute silence or skip errors as successes."""
    start = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            result = provider.synthesize(model, request)
            elapsed = time.perf_counter() - start
            return result, elapsed, attempt + 1
        except APIError as exc:
            if not exc.retryable or attempt == max_retries:
                raise
            delay = exc.retry_after if exc.retry_after is not None else retry_delay * (2 ** attempt)
            time.sleep(min(60, max(0, delay)))
    raise AssertionError("unreachable")


def summary(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    return {"n": len(values), **{
        f"p{pct}": round(values[round((len(values) - 1) * pct / 100)], 5)
        for pct in (50, 90, 95)
    }}


def atomic_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_samples(args, references):
    if args.input_jsonl:
        source = Path(args.input_jsonl).resolve()
        samples = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        for i, sample in enumerate(samples):
            sample.setdefault("id", i)
            if sample.get("prompt_audio_filepath"):
                path = Path(sample["prompt_audio_filepath"])
                sample["prompt_audio_filepath"] = str(path if path.is_absolute() else source.parent / path)
        if args.max_eval_samples and args.max_eval_samples > 0:
            samples = samples[:args.max_eval_samples]
    else:
        samples = load_tts_dataset(
            args, extra_columns=("prompt_audio", "prompt_text") if references else (), decode_audio=False,
        )
    validate_samples(args, samples)
    return samples


def validate_samples(args, samples):
    if not len(samples):
        raise ValueError("Dataset split is empty")
    ids = set()
    for sample in samples:
        sid = str(sample["id"])
        if not re.fullmatch(r"[\w.-]+", sid) or sid in ids:
            raise ValueError("Sample IDs must be unique filename components")
        ids.add(sid)
        if not isinstance(sample.get(args.text_column), str) or not sample[args.text_column].strip():
            raise ValueError(f"Sample {sid} has no synthesis text")


def run(args, provider=None, samples=None):
    import soundfile as sf

    config = get_model(args.model_id)
    if args.split in ("en", "zh", "fr", "es", "ja", "ko", "it", "de", "ru") and args.language != args.split:
        raise ValueError("Benchmark split and --language must match")
    if args.language not in config.languages:
        raise ValueError(f"{args.model_id} does not support benchmark language {args.language}")
    if args.voice_clone and config.reference_mode != "inline":
        raise ValueError("Per-sample cloning is implemented for Fish Audio and Mistral only")
    if args.voice_clone and args.voice:
        raise ValueError("--voice cannot be combined with per-sample --voice_clone")
    provider = provider or get_provider(config.provider, timeout=args.timeout)
    voice = args.voice or config.voice_for_language(args.language)
    needs_reference = args.voice_clone or (config.reference_mode == "inline" and not voice)
    if needs_reference and not provider.supports_reference:
        raise ValueError("Provider does not support inline reference audio")
    samples = samples if samples is not None else load_samples(args, needs_reference)
    validate_samples(args, samples)
    fixed_reference = reference_audio(samples[0]) if needs_reference and not args.voice_clone else None
    mode = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix=mode)
    manifest = Path(paths.manifest_path)
    metadata_path = manifest.with_suffix(".run.json")

    # Hash the text and reference inputs before synthesis, to prevent mixed results after a resume.
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(json.dumps([sample["id"], sample[args.text_column]], ensure_ascii=False).encode())
        if args.voice_clone:
            audio, transcript = reference_audio(sample)
            digest.update(audio)
            digest.update(transcript.encode())
    if fixed_reference:
        digest.update(fixed_reference[0])
        digest.update(fixed_reference[1].encode())
    settings = {
        "model_id": args.model_id, "provider": config.provider, "model": config.model,
        "voice": voice, "language": args.language, "dataset_path": args.dataset_path,
        "dataset": args.dataset, "split": args.split, "voice_clone": args.voice_clone,
        "reference_mode": "per_sample" if args.voice_clone else "fixed" if fixed_reference else "voice",
        "input_sha256": digest.hexdigest(), "sample_rate": args.sample_rate,
        "max_workers": args.max_workers, "timeout": args.timeout, "max_retries": args.max_retries,
        "provider_api_version": getattr(provider, "API_VERSION", None),
    }

    def request_for(sample):
        ref = reference_audio(sample) if args.voice_clone else fixed_reference
        return SynthesisRequest(
            sample[args.text_column], args.language, voice,
            ref[0] if ref else None, ref[1] if ref else "", args.sample_rate,
        )

    if args.ttfa_probe:
        from ttfa_probe import probe_filename, run_probe, sample_indices

        def generate(sample):
            result, _, _ = synthesize_with_retry(
                provider, config.model, request_for(sample), max_retries=args.max_retries,
            )
            audio, rate = decode_audio(result)
            return audio, rate, result.first_audio_at

        payload = run_probe(
            generate, [samples[i] for i in sample_indices(args.ttfa_probe, len(samples))],
            model_id=args.model_id, dataset_path=args.dataset_path, dataset_config=args.dataset,
            split=args.split, out_dir=paths.model_dir, mode_suffix=mode, n=args.ttfa_probe,
            extra={"device": "api", "timing_backend": "api", "settings": settings},
        )
        # An API probe must never be published in the GPU/CPU streaming column groups.
        plain = Path(paths.model_dir) / probe_filename(
            paths.model_safe, args.dataset_path.replace("/", "-"), args.dataset, args.split, mode,
        )
        plain.replace(plain.with_name(plain.stem + "__api.json"))
        return payload

    previous = None
    if manifest.exists():
        if not args.resume and not args.overwrite:
            raise ValueError("Results exist; use --resume or --overwrite")
        if args.resume:
            if not metadata_path.exists():
                raise ValueError("Cannot resume without API run metadata")
            previous = json.loads(metadata_path.read_text())
            if previous.get("settings") != settings:
                raise ValueError("Resume settings or inputs differ; use a separate output tree or --overwrite")
    done = load_done_entries(str(manifest), args.resume)
    for entry in done.values():
        if entry.get("timing_backend") != "api" or entry.get("model_id") != args.model_id:
            raise ValueError("Resume manifest contains non-API or mismatched model rows")
        if not (Path(paths.model_dir) / entry["audio_filepath"]).is_file():
            raise ValueError("Resume manifest contains missing audio; use --overwrite")
    if args.overwrite:
        # SIM forks contain scores for the old audio at the same paths; regeneration invalidates them.
        for fork in manifest.parent.glob(manifest.stem + "_wavlm_seed_tts.jsonl"):
            fork.unlink()
    pending = [s for s in samples if wav_rel_path(paths.dataset_dir_name, s["id"]) not in done]
    if not pending and previous and previous.get("complete"):
        print("API generation already complete:", metadata_path.resolve())
        return previous
    failures = []
    rows = list(done.values())
    metadata = {**settings, "settings": settings, "timing_backend": "api", "n_samples": len(samples),
                "n_completed": len(rows), "n_failed": 0, "max_workers": args.max_workers}
    start = time.perf_counter()
    previous_wall_time = previous.get("wall_time_s", 0) if previous else 0

    def checkpoint():
        metadata.update(
            n_completed=len(rows), n_failed=len(failures), complete=False,
            wall_time_s=previous_wall_time + time.perf_counter() - start,
            failures=failures, updated_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(metadata_path, metadata)

    checkpoint()

    def generate(sample):
        request = request_for(sample)
        call_start = time.perf_counter()
        result, latency, attempts = synthesize_with_retry(
            provider, config.model, request, max_retries=args.max_retries,
        )
        audio, rate = decode_audio(result)
        rel = wav_rel_path(paths.dataset_dir_name, sample["id"])
        output = Path(paths.model_dir) / rel
        temp = output.with_suffix(".tmp.wav")
        sf.write(temp, audio, rate, subtype="PCM_16")
        temp.replace(output)
        prompt_rel = None
        if args.voice_clone:
            prompt_rel = str(Path(paths.dataset_dir_name) / f"prompt_{sample['id']}.wav")
            (Path(paths.model_dir) / prompt_rel).write_bytes(request.reference_audio)
        # time=null intentionally prevents the shared scorer/publisher treating API latency as H200 RTFx.
        entry = manifest_entry(rel, len(audio) / rate, None, sample[args.text_column], prompt_rel)
        # Absence distinguishes an untranscribed row from a legitimate empty ASR prediction.
        entry.pop("pred_text")
        entry.update({
            "model_id": args.model_id, "language": args.language, "timing_backend": "api",
            "audio_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "prompt_audio_sha256": hashlib.sha256(request.reference_audio).hexdigest() if args.voice_clone else None,
            "api_latency_s": latency, "api_attempts": attempts,
            "api_ttfa_ms": (result.first_audio_at - call_start) * 1000 if result.first_audio_at else None,
            "api_request_id": result.request_id,
        })
        return entry

    # A bounded queue limits memory and outstanding work after an authentication failure.
    with (
        manifest.open("a" if args.resume else "w", encoding="utf-8") as handle,
        ThreadPoolExecutor(max_workers=args.max_workers) as executor,
    ):
        iterator = iter(pending)
        futures = {}
        exhausted = False
        fatal = False
        while futures or not exhausted:
            while not exhausted and not fatal and len(futures) < args.max_workers:
                sample = next(iterator, None)
                if sample is None:
                    exhausted = True
                else:
                    futures[executor.submit(generate, sample)] = sample["id"]
            if not futures:
                break
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                sid = futures.pop(future)
                try:
                    entry = future.result()
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    handle.flush()
                    rows.append(entry)
                    print(f"Generated {len(rows)}/{len(samples)}", flush=True)
                except Exception as exc:  # noqa: BLE001 — retain successful rows on provider/decoder failures
                    # Log the exception type, never arbitrary provider messages/credential-bearing URLs.
                    failure = {"sample_id": str(sid), "error_type": type(exc).__name__}
                    if isinstance(exc, APIError):
                        failure["http_status"] = exc.status_code
                        fatal = fatal or exc.status_code in (401, 403)
                    failures.append(failure)
                    print(f"Failed sample {sid}: {type(exc).__name__}", flush=True)
                checkpoint()
            if fatal:
                exhausted = True
    elapsed = time.perf_counter() - start
    # Resumed throughput pools only active generation time, excluding time between invocations.
    wall_time = elapsed + previous_wall_time
    metadata.update({
        "n_completed": len(rows), "n_failed": len(failures), "failures": failures,
        "complete": len(rows) == len(samples) and not failures,
        "wall_time_s": wall_time, "updated_at": datetime.now(timezone.utc).isoformat(),
        "api_throughput_rtfx": sum(r["duration"] for r in rows) / wall_time if wall_time else None,
        "api_latency_s": summary([r["api_latency_s"] for r in rows]),
        "api_ttfa_ms": summary([r.get("api_ttfa_ms") for r in rows]),
    })
    atomic_json(metadata_path, metadata)
    print("API generation metadata:", metadata_path.resolve())
    if not metadata["complete"]:
        raise RuntimeError(f"Incomplete API run: {len(rows)}/{len(samples)} generated; resume to retry failures")
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id")
    parser.add_argument("--list_models", action="store_true")
    parser.add_argument("--dataset_path", default="bezzam/seed_tts_eval")
    parser.add_argument("--dataset", default="tts")
    parser.add_argument("--split", default="en")
    parser.add_argument("--language", default="en")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--input_jsonl", help="Local JSONL with text/id and optional prompt_audio_filepath/prompt_text")
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--voice")
    parser.add_argument("--voice_clone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ttfa_probe", type=int, default=0)
    args = parser.parse_args(argv)
    if args.list_models:
        for model_id, model in MODELS.items():
            print(f"{model_id} | languages={','.join(model.languages)} | reference={model.reference_mode}")
        return args
    if not args.model_id:
        parser.error("--model_id is required unless --list_models is passed")
    if args.max_workers < 1 or args.timeout <= 0 or args.max_retries < 0 or args.sample_rate <= 0:
        parser.error("workers, timeout and sample rate must be positive; retries must be non-negative")
    if args.max_eval_samples == 0 or args.max_eval_samples < -1 or args.ttfa_probe < -1:
        parser.error("Sample limits must be -1 (all) or positive; TTFA accepts zero to disable")
    if not math.isfinite(args.timeout):
        parser.error("Timeout must be finite")
    if args.resume and args.overwrite:
        parser.error("Choose either --resume or --overwrite")
    return args


if __name__ == "__main__":
    # Keep API manifests separate even when the command is launched from the repository root.
    os.chdir(Path(__file__).resolve().parent)
    args = parse_args()
    if not args.list_models:
        run(args)
