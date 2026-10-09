"""Export API quality and network metrics without H200 leaderboard timing rows.

Each manifest is normalized in its recorded language using normalizer.eval_utils.
Generation metadata and samples remain beside the WAVs; this exporter never fabricates
results for registered models that have not been evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

API_DIR = Path(__file__).resolve().parent
REPO_ROOT = API_DIR.parent
SIM_SUFFIX = "_wavlm_seed_tts"
RESULT_FIELDS = (
    "model_id", "provider", "model", "voice", "dataset_path", "dataset", "split", "language", "voice_clone",
    "n_samples", "n_completed", "n_failed", "complete", "coverage_percent", "metric", "wer", "cer_unnormalized",
    "sim", "sim_backend", "api_latency_mean_s", "api_latency_p50_s", "api_latency_p95_s",
    "api_ttfa_mean_ms", "api_ttfa_p50_ms", "api_ttfa_p95_ms", "api_throughput_rtfx",
    "api_attempts_total", "max_workers", "wall_time_s", "timing_backend",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical_path(path: Path) -> Path:
    return path.with_name(path.stem.removesuffix(SIM_SUFFIX) + path.suffix)


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * percent / 100
    lower, upper = math.floor(rank), math.ceil(rank)
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def metric_values(rows: list[dict], field: str) -> list[float]:
    return [float(row[field]) for row in rows if isinstance(row.get(field), (int, float)) and math.isfinite(row[field])]


def checked_rows(path: Path, metadata: dict) -> list[dict]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"No generated samples in {path}")
    if len({row.get("audio_filepath") for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate sample audio paths")
    for row in rows:
        if row.get("timing_backend") != "api" or row.get("time") is not None:
            raise ValueError(f"{path}: expected timing_backend='api' and time=null; refuse GPU/API metric mixing")
        if row.get("language") != metadata.get("language") or row.get("model_id") != metadata.get("model_id"):
            raise ValueError(f"{path}: sample language/model does not match run metadata")
        if "pred_text" not in row:
            raise ValueError(f"{path}: missing ASR predictions; run the transcribe stage first")
    if not metadata.get("complete") or metadata.get("n_failed") != 0:
        raise ValueError(f"{path}: incomplete API generation; resume failed requests before exporting results")
    if metadata.get("n_samples") != len(rows) or metadata.get("n_completed") != len(rows):
        raise ValueError(f"{path}: sample counts do not match complete API run metadata")
    return rows


def checked_similarity(rows: list[dict], sim_backend: str, path: Path):
    for row in rows:
        model = row.get("sim_model", "")
        backend_matches = (isinstance(model, str) and model.startswith("wavlm_large_finetune+ecapa_tdnn (")) if sim_backend == "wavlm_seed_tts" else model == "microsoft/wavlm-base-plus-sv"
        sim = row.get("sim")
        numeric = isinstance(sim, (int, float)) and not isinstance(sim, bool) and math.isfinite(sim)
        excluded_short = "sim" in row and sim is None and str(row.get("sim_note", "")).startswith("clip shorter than ")
        if not row.get("prompt_audio_filepath") or not backend_matches or not (numeric or excluded_short):
            raise ValueError(f"{path}: incomplete {sim_backend} similarity; run the sim stage before exporting cloning results")


def assert_current_sim(canonical_rows: list[dict], scored_rows: list[dict], path: Path):
    fields = ("audio_filepath", "text", "pred_text", "prompt_audio_filepath", "audio_sha256",
              "prompt_audio_sha256", "api_latency_s", "api_attempts", "api_ttfa_ms")
    def identities(rows):
        return {row["audio_filepath"]: tuple(row.get(field) for field in fields) for row in rows}
    if identities(canonical_rows) != identities(scored_rows):
        raise ValueError(f"{path}: SIM fork is stale; rerun the sim stage with --sim_overwrite")


def collect_results(manifests: list[Path], *, model_id: str | None = None,
                    sim_backend: str = "wavlm_seed_tts", evaluator: Callable | None = None) -> list[dict]:
    if evaluator is None:
        sys.path.insert(0, str(REPO_ROOT))
        from normalizer.eval_utils import score_results as evaluator

    if sim_backend not in ("wavlm_seed_tts", "xvector"):
        raise ValueError(f"Unsupported SIM backend {sim_backend!r}")
    output = []
    for manifest in sorted({canonical_path(Path(path).resolve()) for path in manifests}):
        sidecar = manifest.with_suffix(".run.json")
        if not sidecar.is_file():
            raise ValueError(f"Missing API run metadata: {sidecar}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        required = ("model_id", "provider", "model", "language", "dataset_path", "dataset", "split", "voice_clone")
        if any(key not in metadata for key in required):
            raise ValueError(f"{sidecar}: missing API run fields")
        if model_id and metadata["model_id"] != model_id:
            raise ValueError(f"{manifest}: expected model {model_id!r}, found {metadata['model_id']!r}")
        canonical_rows = checked_rows(manifest, metadata)
        scored_manifest = manifest
        clone = metadata["voice_clone"]
        if clone and sim_backend == "wavlm_seed_tts":
            fork = manifest.with_name(manifest.stem + SIM_SUFFIX + ".jsonl")
            if fork.is_file():
                scored_manifest = fork
        rows = checked_rows(scored_manifest, metadata)
        if scored_manifest != manifest:
            assert_current_sim(canonical_rows, rows, scored_manifest)
        if clone:
            checked_similarity(rows, sim_backend, scored_manifest)
        # Fixed voices never consume a cloning SIM fork or publish a SIM score.
        effective_backend = sim_backend if clone else "xvector"
        _, quality_results = evaluator(
            str(manifest.parent), model_id=metadata["model_id"], language=metadata["language"],
            manifests=[manifest.name], sim_backend=effective_backend, csv_only=True,
        )
        if len(quality_results) != 1:
            raise ValueError(f"Expected exactly one quality result for {manifest}, got {len(quality_results)}")
        quality = next(iter(quality_results.values()))
        latencies = metric_values(rows, "api_latency_s")
        ttfa = metric_values(rows, "api_ttfa_ms")
        total = metadata.get("n_samples", len(rows))
        row = {key: metadata.get(key) for key in RESULT_FIELDS}
        row.update(
            n_samples=total, n_completed=len(rows), n_failed=metadata.get("n_failed", max(0, total - len(rows))),
            complete=len(rows) == total and metadata.get("n_failed", 0) == 0 and metadata.get("complete", True),
            coverage_percent=round(100 * len(rows) / total, 2) if total else None,
            metric=quality["metric"], wer=quality["wer"], cer_unnormalized=quality.get("cer_unnormalized"),
            sim=quality.get("sim") if clone else None, sim_backend=sim_backend if clone else None,
            api_latency_mean_s=statistics.mean(latencies) if latencies else None,
            api_latency_p50_s=percentile(latencies, 50), api_latency_p95_s=percentile(latencies, 95),
            api_ttfa_mean_ms=statistics.mean(ttfa) if ttfa else None,
            api_ttfa_p50_ms=percentile(ttfa, 50), api_ttfa_p95_ms=percentile(ttfa, 95),
            api_attempts_total=sum(row.get("api_attempts", 1) for row in rows), timing_backend="api",
        )
        output.append(row)
    if not output:
        raise ValueError("No API manifests selected; run generation and ASR before scoring")
    return output


def result_key(row):
    return tuple(row.get(key) for key in ("model_id", "dataset_path", "dataset", "split", "voice_clone"))


def write_results(rows: list[dict], output_dir: Path):
    """Merge evaluated splits by identity so later models retain earlier results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "API_RESULTS.json"
    previous = json.loads(json_path.read_text(encoding="utf-8")).get("results", []) if json_path.is_file() else []
    merged = {result_key(row): row for row in previous}
    merged.update({result_key(row): row for row in rows})
    values = sorted(merged.values(), key=lambda row: tuple(str(value) for value in result_key(row)))
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"schema_version": 1, "timing_backend": "api", "results": values}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(json_path)
    csv_path = output_dir / "API_RESULTS.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in RESULT_FIELDS} for row in values)
    temporary.replace(csv_path)
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id")
    parser.add_argument("--manifests", nargs="+", type=Path, help="Exact canonical manifests to score")
    parser.add_argument("--results_dir", type=Path, default=API_DIR / "results")
    parser.add_argument("--output_dir", type=Path, default=API_DIR / "results")
    parser.add_argument("--sim_backend", choices=("wavlm_seed_tts", "xvector"), default="wavlm_seed_tts")
    args = parser.parse_args()
    manifests = args.manifests or list(args.results_dir.glob("**/MODEL_*.jsonl"))
    if args.model_id and not args.manifests:
        model_safe = args.model_id.replace("/", "-")
        manifests = [path for path in manifests if path.name.startswith(f"MODEL_{model_safe}_DATASET_")]
    try:
        rows = collect_results(manifests, model_id=args.model_id, sim_backend=args.sim_backend)
        paths = write_results(rows, args.output_dir)
    except (ValueError, OSError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    for row in rows:
        print(f"{row['model_id']} {row['dataset']}/{row['split']}: {row['metric']}={row['wer']}%, "
              f"coverage={row['n_completed']}/{row['n_samples']}, API latency p50={row['api_latency_p50_s']}s")
    print("API results written to " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
