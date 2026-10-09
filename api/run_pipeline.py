"""Local Docker API generation followed by the leaderboard's shared ASR/SIM HF Jobs.

API results use their own bucket and report network timings separately from H200 RTFx.
All subprocess arguments are lists; injected job scripts quote paths and values.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

API_DIR = Path(__file__).resolve().parent
REPO_ROOT = API_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
BENCHMARK_LANGUAGES = ("en", "fr", "es", "zh", "ja", "ko", "de", "it", "ru")
STAGES = ("generate", "transcribe", "sim", "score")
H200_BUCKET = "hf-audio/tts_leaderboard_h200"
DEFAULT_TTS_SPACE = "hf-audio/open-tts-leaderboard-apis"


@dataclass(frozen=True)
class DatasetConfig:
    path: str
    dataset: str
    split: str
    language: str


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() not in ("true", "false", "1", "0"):
        raise ValueError(f"{name} must be true or false, got {value!r}")
    return value.lower() in ("true", "1")


def dataset_configs(args) -> list[DatasetConfig]:
    """Match the existing Seed-TTS and CV3 evaluation configs."""
    configs = []
    if args.datasets in ("seed_tts", "both"):
        configs.extend(DatasetConfig(args.dataset_path, "tts", lang, lang) for lang in ("en", "zh"))
    if args.datasets in ("cv3", "both"):
        configs.extend(DatasetConfig(args.cv3_eval_path, "zero_shot", lang, lang) for lang in BENCHMARK_LANGUAGES)
    return [config for config in configs if not args.only_langs or config.language in args.only_langs]


def manifest_path(model_id: str, config: DatasetConfig, voice_clone: bool = False) -> Path:
    model_safe = model_id.replace("/", "-")
    dataset_safe = config.path.replace("/", "-")
    suffix = "_voice_clone" if voice_clone else ""
    name = f"MODEL_{model_safe}_DATASET_{dataset_safe}_{config.dataset.replace('/', '-')}_{config.split}{suffix}.jsonl"
    return API_DIR / "results" / model_safe / name


def cache_directory() -> Path:
    return Path(os.environ.get("HF_CACHE_DIR") or os.environ.get("HF_HOME") or
                Path.home() / ".cache" / "huggingface").expanduser().resolve()


def fixture_mounts(source: Path) -> list[tuple[Path, Path]]:
    """Resolve host files while preserving reference path aliases inside Docker."""
    mounts = {source: source}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        reference = json.loads(line).get("prompt_audio_filepath")
        if reference:
            path = Path(reference)
            path = path if path.is_absolute() else source.parent / path
            mounts[Path(os.path.abspath(path))] = path.resolve()
    return [(host, destination) for destination, host in sorted(mounts.items())]


def generation_image(args) -> str:
    if args.api_image:
        return args.api_image
    slug = args.tts_space.lower().replace("/", "-").replace("_", "-").replace(".", "-")
    if len(args.tts_space.split("/")) != 2 or len(slug) > 63:
        raise ValueError("Use --tts_space OWNER/SPACE, or --api_image with the exact reference from Run with Docker")
    return f"registry.hf.space/{slug}:latest"


def generation_command(args, model_id: str, config: DatasetConfig) -> list[str]:
    from api.models import get_model
    from api.providers import KEY_ENV

    command = [
        "/app/api/run_eval.py",
        "--model_id", model_id, "--dataset_path", config.path,
        "--dataset", config.dataset, "--split", config.split,
        "--language", config.language, "--max_eval_samples", str(args.max_eval_samples),
        "--max_workers", str(args.max_workers),
    ]
    if args.resume:
        command.append("--resume")
    if args.overwrite:
        command.append("--overwrite")
    if args.voice:
        command.extend(("--voice", args.voice))
    if args.voice_clone:
        command.append("--voice_clone")
    if args.ttfa_probe:
        command.extend(("--ttfa_probe", str(args.ttfa_probe)))
    if args.input_jsonl:
        command.extend(("--input_jsonl", str(Path(args.input_jsonl).resolve())))
    docker = ["docker", "run", "--rm", "--platform", args.docker_platform,
              "--entrypoint", "python", "--workdir", "/app/api", "--user", f"{os.getuid()}:{os.getgid()}",
              "--volume", f"{API_DIR}:/app/api:ro",
              "--volume", f"{REPO_ROOT / 'scripts'}:/app/scripts:ro",
              "--volume", f"{API_DIR / 'results'}:/app/api/results",
              "--volume", f"{cache_directory()}:/hf_cache",
              "--env", "HF_TOKEN", "--env", KEY_ENV[get_model(model_id).provider]]
    if args.input_jsonl:
        for host, destination in fixture_mounts(Path(args.input_jsonl).resolve()):
            docker.extend(("--volume", f"{host}:{destination}:ro"))
    return [*docker, generation_image(args), *command]


def prepare_generation(args):
    if not args.dry_run:
        if not shutil.which("docker"):
            raise ValueError("Docker is required for API generation; install Docker and start its daemon")
        (API_DIR / "results").mkdir(parents=True, exist_ok=True)
        cache_directory().mkdir(parents=True, exist_ok=True)
    if not args.skip_image_pull:
        try:
            run_command(["docker", "pull", "--platform", args.docker_platform, generation_image(args)], dry_run=args.dry_run)
        except subprocess.CalledProcessError:
            if args.api_image:
                # An explicit image/digest must never be replaced with a different build.
                raise
            build_space_environment(args)


def build_space_environment(args):
    """Use the Space's pinned source when its public registry image is unavailable."""
    from huggingface_hub import HfApi

    info = HfApi().space_info(args.tts_space)
    if info.sdk != "docker" or info.private:
        raise ValueError("TTS_SPACE must be a public Docker environment Space")
    source = cache_directory() / "tts_api_spaces" / args.tts_space.replace("/", "--") / info.sha
    print(f"Space registry image unavailable; building {args.tts_space}@{info.sha} from Hub source", flush=True)
    run_command(["hf", "download", args.tts_space, "--repo-type", "space", "--revision", info.sha,
                 "--local-dir", str(source)])
    run_command(["docker", "build", "--platform", args.docker_platform, "--file", str(source / "Dockerfile"),
                 "--tag", generation_image(args), str(source)])


def injected_script(source: Path, target: str) -> str:
    """Portable injection: Python base64 decoding works on Linux and macOS."""
    payload = base64.b64encode(source.read_bytes()).decode("ascii")
    code = (
        "import base64; from pathlib import Path; "
        f"p=Path({target!r}); p.parent.mkdir(parents=True, exist_ok=True); "
        f"p.write_bytes(base64.b64decode({payload!r}))"
    )
    return "python -c " + shlex.quote(code)


def job_command(args, stage: str, manifest: Path, language: str) -> list[str]:
    if stage not in ("transcribe", "sim"):
        raise ValueError(f"Unsupported job stage {stage!r}")
    remote_manifest = "/results/" + manifest.relative_to(API_DIR / "results").as_posix()
    filename = "transcribe.py" if stage == "transcribe" else "score_similarity.py"
    target = "/app/transformers/" + filename
    command = ["python", target, "--manifest_path", remote_manifest, "--device", "cuda:0",
               "--max_audio_seconds", str(args.max_audio_seconds)]
    if stage == "transcribe":
        command.extend(("--asr_language", language, "--asr_batch_size", str(args.asr_batch_size)))
        overwrite = args.asr_overwrite
    else:
        command.extend(("--sim_backend", args.sim_backend, "--batch_size", str(args.sim_batch_size)))
        overwrite = args.sim_overwrite
    # Regenerated WAV filenames are reused, so previous predictions/SIM must be discarded.
    if overwrite or ("generate" in args.stages and not args.resume):
        command.append("--overwrite")
    body = injected_script(REPO_ROOT / "transformers" / filename, target) + " && " + shlex.join(command)
    result = ["hf", "jobs", "run", "--flavor", args.asr_flavor if stage == "transcribe" else args.sim_flavor,
              "--timeout", "4h", "--secrets", "HF_TOKEN"]
    if args.org_name:
        result.extend(("--namespace", args.org_name))
    if stage == "sim":
        result.extend(("--env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"))
    result.extend(("--volume", f"hf://buckets/{args.results_bucket}:/results",
                   f"hf.co/spaces/{args.scorer_space}", "bash", "-c", body))
    return result


def bucket_sync(args, model_id: str, *, upload: bool) -> list[str]:
    folder = model_id.replace("/", "-")
    local = str(API_DIR / "results" / folder)
    remote = f"hf://buckets/{args.results_bucket}/{folder}"
    # No delete flag: downloading manifests must retain the locally generated WAVs.
    return ["hf", "buckets", "sync", local, remote] if upload else ["hf", "buckets", "sync", remote, local, "--exclude", "*.wav"]


def run_command(command: list[str], *, dry_run: bool = False):
    # Avoid printing the injected base64 source (often >100 KiB).
    display = command[:]
    if "bash" in display and display[-2] == "-c":
        display[-1] = "<injected shared scorer>"
    print("$ " + shlex.join(display), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=API_DIR, check=True)


def validate_args(args):
    from api.models import get_model

    if not args.stages or any(stage not in STAGES for stage in args.stages):
        raise ValueError("--stages must be a nonempty subset of: " + " ".join(STAGES))
    if "generate" in args.stages:
        generation_image(args)
    if args.max_workers < 1:
        raise ValueError("--max_workers must be >= 1")
    if args.max_eval_samples == 0 or args.max_eval_samples < -1:
        raise ValueError("--max_eval_samples must be -1 (all) or positive")
    if args.ttfa_probe < -1:
        raise ValueError("--ttfa_probe must be -1 (full split), zero, or positive")
    if args.ttfa_probe and args.stages != ["generate"]:
        raise ValueError("TTFA probes produce a sidecar only; use --stages generate with --ttfa_probe")
    if args.voice_clone and args.voice:
        raise ValueError("--voice cannot be combined with per-sample --voice_clone")
    remote = "transcribe" in args.stages or ("sim" in args.stages and args.voice_clone)
    if remote and not args.results_bucket:
        raise ValueError("Set RESULTS_BUCKET or --results_bucket to a dedicated API results bucket for HF Jobs")
    if remote and args.results_bucket.rstrip("/") == H200_BUCKET:
        raise ValueError("API results must use a separate bucket from the official H200 results")
    if args.only_langs and any(lang not in BENCHMARK_LANGUAGES for lang in args.only_langs):
        raise ValueError("--only_langs must use benchmark language codes: " + " ".join(BENCHMARK_LANGUAGES))
    if args.input_jsonl and (args.datasets != "seed_tts" or args.only_langs != ["en"]):
        raise ValueError("A local --input_jsonl requires --datasets seed_tts --only_langs en to avoid relabelling one file as several datasets")
    for model_id in args.models:
        model = get_model(model_id)
        if args.voice_clone and model.reference_mode != "inline":
            raise ValueError(f"{model_id} has no inline benchmark reference-cloning adapter")
    if args.voice and len(args.models) != 1:
        raise ValueError("--voice is provider-specific; select exactly one model")
    if not dataset_configs(args):
        raise ValueError("No dataset splits selected")


def run_pipeline(args):
    from api.models import get_model

    validate_args(args)
    generation_prepared = False
    for model_id in args.models:
        model = get_model(model_id)
        configs = [config for config in dataset_configs(args) if config.language in model.languages]
        if not configs:
            print(f"Skipping {model_id}: no supported languages selected", flush=True)
            continue
        manifests = [manifest_path(model_id, config, args.voice_clone) for config in configs]
        if "generate" in args.stages:
            if not generation_prepared:
                prepare_generation(args)
                generation_prepared = True
            for config in configs:
                run_command(generation_command(args, model_id, config), dry_run=args.dry_run)
        remote_stages = [stage for stage in ("transcribe", "sim") if stage in args.stages and (stage != "sim" or args.voice_clone)]
        if remote_stages:
            if "generate" not in args.stages:
                # Re-scoring starts from the bucket, rather than uploading stale local predictions.
                run_command(bucket_sync(args, model_id, upload=False), dry_run=args.dry_run)
            if not args.dry_run:
                missing = [str(path) for path in manifests if not path.is_file()]
                if missing:
                    raise FileNotFoundError("Run generation first; missing manifests: " + ", ".join(missing))
            if "generate" in args.stages:
                run_command(bucket_sync(args, model_id, upload=True), dry_run=args.dry_run)
            for config, manifest in zip(configs, manifests):
                for stage in remote_stages:
                    run_command(job_command(args, stage, manifest, config.language), dry_run=args.dry_run)
            run_command(bucket_sync(args, model_id, upload=False), dry_run=args.dry_run)
        if "score" in args.stages:
            command = [sys.executable, str(API_DIR / "score_results.py"),
                       "--model_id", model_id, "--sim_backend", args.sim_backend, "--manifests"]
            command.extend(str(path) for path in manifests)
            run_command(command, dry_run=args.dry_run)


def make_parser():
    from api.models import MODELS

    namespace = os.environ.get("DATASET_NAMESPACE", "bezzam")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS), help="Registered provider/model IDs (default: all)")
    parser.add_argument("--datasets", choices=("seed_tts", "cv3", "both"), default="both")
    parser.add_argument("--dataset_path", default=os.environ.get("DATASET_PATH", f"{namespace}/seed_tts_eval"))
    parser.add_argument("--cv3_eval_path", default=os.environ.get("CV3_EVAL_PATH", f"{namespace}/cv3_eval"))
    parser.add_argument("--only_langs", nargs="+", default=os.environ.get("ONLY_LANGS", "").split())
    parser.add_argument("--stages", nargs="+", default=os.environ.get("STAGES", "generate transcribe sim score").split(), choices=STAGES)
    parser.add_argument("--max_eval_samples", type=int, default=int(os.environ.get("MAX_EVAL_SAMPLES", "-1")))
    parser.add_argument("--max_workers", type=int, default=int(os.environ.get("MAX_WORKERS", "1")))
    parser.add_argument("--voice_clone", action=argparse.BooleanOptionalAction, default=env_bool("VOICE_CLONE"))
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true")
    resume.add_argument("--overwrite", action="store_true", help="Regenerate existing audio and invalidate prior ASR/SIM")
    parser.add_argument("--voice")
    parser.add_argument("--ttfa_probe", type=int, default=0)
    parser.add_argument("--input_jsonl", help="Local en JSONL for a generation smoke test")
    parser.add_argument("--tts_space", default=os.environ.get("TTS_SPACE", DEFAULT_TTS_SPACE),
                        help="Public Docker Space providing the API environment (TTS_SPACE)")
    parser.add_argument("--api_image", default=os.environ.get("API_IMAGE"),
                        help="Exact registry image reference/digest; overrides --tts_space (API_IMAGE)")
    parser.add_argument("--docker_platform", default=os.environ.get("API_DOCKER_PLATFORM", "linux/amd64"),
                        help="Docker platform; Hub Space images use linux/amd64 (API_DOCKER_PLATFORM)")
    parser.add_argument("--skip_image_pull", action="store_true", help="Use the cached Space image without pulling it again")
    parser.add_argument("--results_bucket", default=os.environ.get("RESULTS_BUCKET", ""))
    parser.add_argument("--org_name", default=os.environ.get("ORG_NAME", ""))
    parser.add_argument("--scorer_space", default=os.environ.get("SPACE", "bezzam/evals"))
    parser.add_argument("--asr_flavor", default=os.environ.get("ASR_FLAVOR", "l4x1"))
    parser.add_argument("--sim_flavor", default=os.environ.get("SIM_FLAVOR", "l4x1"))
    parser.add_argument("--asr_batch_size", type=int, default=int(os.environ.get("ASR_BATCH_SIZE", "32")))
    parser.add_argument("--sim_batch_size", type=int, default=int(os.environ.get("SIM_BATCH_SIZE", "32")))
    parser.add_argument("--sim_backend", choices=("wavlm_seed_tts", "xvector"), default=os.environ.get("SIM_BACKEND", "wavlm_seed_tts"))
    parser.add_argument("--asr_overwrite", action="store_true", default=env_bool("ASR_OVERWRITE"))
    parser.add_argument("--sim_overwrite", action="store_true", default=env_bool("SIM_OVERWRITE"))
    parser.add_argument("--max_audio_seconds", type=float, default=float(os.environ.get("MAX_AUDIO_SECONDS", "30")))
    parser.add_argument("--dry_run", action="store_true", help="Print commands without API calls, uploads, or Jobs")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    try:
        run_pipeline(args)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
