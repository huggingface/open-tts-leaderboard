#!/usr/bin/env bash
# Preview bounded generation jobs; --execute submits them serially without scoring.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash fishaudio-drama-3/submit_jobs.sh [--execute]

Default: preview one Seed-TTS en generation job, 8 rows, H200, 30-minute timeout.
Execution requires TTS_SPACE=owner/public-space or an immutable TTS_IMAGE, plus RESULTS_BUCKET.
The image must retain the pinned server checkout, or use the explicit runtime bootstrap.

Environment:
  DATASET_SCOPE=seed_en|cv3_en|full    default seed_en; full means 11 serial jobs
  MAX_EVAL_SAMPLES=8                 -1 explicitly opts into full splits
  JOB_TIMEOUT=30m                   HF Job wall-clock limit per job
  FIRST_JOB_TIMEOUT=30m             optional tighter limit for the first selected job
  VOICE_CLONE=false                  true saves references for later SIM scoring
  TTFA_PROBE=0                      positive N writes only a TTFA sidecar
  BATCH_SIZE=1                      concurrent requests during offline generation
  WARMUP_STEPS=2                    untimed warm-up batches
  RESULTS_PREFIX=fish-drama-3/full   optional subfolder within RESULTS_BUCKET
  INPUTS_BUCKET=owner/private-bucket optional local dataset input mount, read-only
  INPUTS_PREFIX=fish-drama-3/inputs  optional subfolder within INPUTS_BUCKET
  TTS_IMAGE=repository@sha256:...    direct immutable Docker image; excludes TTS_SPACE
  BOOTSTRAP_RUNTIME=false           true installs the pinned runtime in the approved base
  DATASET_NAMESPACE=bezzam           default private eval-dataset namespace
  DATASET_PATH=owner/seed_tts_eval    override Seed-TTS repository
  CV3_EVAL_PATH=owner/cv3_eval        override CV3 repository
  PYTHON_BIN=python                 Python in the image's SGLang environment
  ORG_NAME=owner                    optional HF Jobs billing namespace

This launcher does not start ASR, SIM, scoring, result publication, or automatic retries.
EOF
}

# Keep the Job's bootstrap and exit traps in separate child Bash processes.
drama_job_script() {
    cat <<'DRAMA_JOB_SCRIPT'
#!/usr/bin/env bash
# Runs inside the selected HF Job image after the local launcher injects the backend.
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID=fishaudio/drama-3
MODEL_REVISION=e0a9218aa48888270f06ef1ed9ea67622c0e17d2
SERVER_REVISION=12f65fa60fbb5217ad47f58b2e58b18092c818a1
JOB_FLAVOR="${JOB_FLAVOR:-h200}"
DRAMA_BUCKET_MOUNT="${DRAMA_BUCKET_MOUNT:-/results}"
[[ "$JOB_FLAVOR" == h200 || "$JOB_FLAVOR" == a100-large ]] || { echo 'Unsupported GPU job flavor.' >&2; exit 2; }

# Reject mismatched identity before imports, downloads, or server startup.
CLIENT_ARGS=("$@")
for (( argument_index=0; argument_index < ${#CLIENT_ARGS[@]}; argument_index++ )); do
    argument="${CLIENT_ARGS[$argument_index]}"
    case "$argument" in
        --model_id|--model_revision|--server_revision)
            (( argument_index + 1 < ${#CLIENT_ARGS[@]} )) || { echo 'Missing identity argument value.' >&2; exit 2; }
            identity_value="${CLIENT_ARGS[$((argument_index + 1))]}"
            case "$argument" in
                --model_id) expected_value="$MODEL_ID" ;;
                --model_revision) expected_value="$MODEL_REVISION" ;;
                --server_revision) expected_value="$SERVER_REVISION" ;;
            esac
            [[ "$identity_value" == "$expected_value" ]] || { printf 'Rejected mismatched %s.\n' "$argument" >&2; exit 2; }
            ;;
        --model_id=*|--model_revision=*|--server_revision=*)
            echo 'Use separate identity arguments and values.' >&2; exit 2 ;;
        --host*|--port*|--base_url*)
            echo 'The job uses its verified local server on port 8000.' >&2; exit 2 ;;
    esac
done

PYTHONPATH="$BACKEND_DIR${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON_BIN" - "$DRAMA_BUCKET_MOUNT" "$JOB_FLAVOR" "$@" <<'PY'
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
parser.add_argument("--dataset_path", default="bezzam/seed_tts_eval")
parser.add_argument("--dataset", default="tts")
parser.add_argument("--dataset_local_root")
parser.add_argument("--split", default="en")
parser.add_argument("--voice_clone", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--ttfa_probe", type=int, default=0)
parser.add_argument("--resume", action="store_true")
args, _ = parser.parse_known_args(sys.argv[3:])
if args.resume:
    raise SystemExit("Bucket resume is not supported; review and resume partial results separately.")
if any("/" in value or "\\" in value or value in ("", ".", "..") for value in (args.split,)):
    raise SystemExit("Split must be a filename component.")
bucket = Path(sys.argv[1])
if not bucket.is_dir():
    raise SystemExit("Results bucket mount is unavailable; refusing generation.")
suffix = "_voice_clone" if args.voice_clone else ""
dataset = args.dataset_path.replace("/", "-")
config = args.dataset.replace("/", "-")
stem = f"MODEL_fishaudio-drama-3_DATASET_{dataset}_{config}_{args.split}{suffix}"
if args.ttfa_probe:
    stem = f"TTFA_{stem}"
    candidates = [f"{stem}.json", f"{stem}.run.json",
                  f"{stem}__sglang__{sys.argv[2]}.json", f"{stem}__sglang__{sys.argv[2]}.run.json"]
else:
    candidates = [f"{stem}.jsonl", f"{stem}.run.json"]
for name in candidates:
    if (bucket / "fishaudio-drama-3" / name).exists():
        raise SystemExit("Targeted results already exist in the bucket; refusing a duplicate run or overwrite.")
if args.dataset_local_root:
    from run_eval import resolve_dataset_source
    resolve_dataset_source(args)
    print("Verified selected private input Parquet checksum and canonical split provenance.")
print("Verified that this dataset, clone mode, and probe target have no existing bucket results.")
PY

"$PYTHON_BIN" - "$SERVER_REVISION" <<'PY'
import pathlib
import subprocess
import sys

import sglang_omni

module_root = pathlib.Path(sglang_omni.__file__).resolve().parent
source_root = pathlib.Path(subprocess.check_output(
    ["git", "-C", str(module_root), "rev-parse", "--show-toplevel"], text=True
).strip())
if not (source_root / "sglang_omni").samefile(module_root):
    raise SystemExit("SGLang Omni must be installed from its retained Git checkout.")
revision = subprocess.check_output(
    ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
).strip()
if revision != sys.argv[1]:
    raise SystemExit(f"SGLang Omni revision mismatch: expected {sys.argv[1]}, got {revision}")
modified = subprocess.check_output(
    ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"], text=True
).strip()
if modified:
    raise SystemExit("SGLang Omni checkout has modified tracked files; refusing unpinned code.")
print(f"Verified SGLang Omni revision {revision}")
PY

SNAPSHOT_PATH="$("$PYTHON_BIN" - "$MODEL_ID" "$MODEL_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]))
PY
)"
[[ -d "$SNAPSHOT_PATH" ]] || { echo 'Pinned model snapshot directory was not returned.' >&2; exit 1; }
printf 'Verified model snapshot: %s @ %s\n' "$MODEL_ID" "$MODEL_REVISION"

cd "$BACKEND_DIR"
mkdir -p results
SERVER_PID=""
cleanup() {
    local job_status=$?
    trap - EXIT
    # Retain the single attempt even if inference fails partway through a split.
    # The existing-target guard still prevents an automatic restart/overwrite.
    if [[ -d results/fishaudio-drama-3 ]]; then
        if [[ "$job_status" -ne 0 ]]; then
            "$PYTHON_BIN" - "$job_status" <<'PY' || echo 'Could not mark incomplete receipt; preserving its original bytes for inspection.' >&2
import json
from pathlib import Path
import sys

for path in Path("results/fishaudio-drama-3").glob("MODEL_*.run.json"):
    receipt = json.loads(path.read_text())
    receipt.update(status="failed", execution_exit_status=int(sys.argv[1]))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path)
PY
        fi
        if cp -R results/fishaudio-drama-3 "$DRAMA_BUCKET_MOUNT/"; then
            printf 'Preserved generation outputs in the private results mount (client exit status %s).\n' "$job_status"
        else
            echo 'Failed to preserve generation outputs; manual artifact inspection is required.' >&2
            [[ "$job_status" -ne 0 ]] || job_status=1
        fi
    fi
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit "$job_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$PYTHON_BIN" -m sglang_omni.cli serve --model-path "$SNAPSHOT_PATH" --model-name "$MODEL_ID" \
    --host 127.0.0.1 --port 8000 --allowed-local-media-path "$BACKEND_DIR/results" &
SERVER_PID=$!

"$PYTHON_BIN" run_eval.py "$@" --model_id "$MODEL_ID" \
    --model_revision "$MODEL_REVISION" --server_revision "$SERVER_REVISION" --port 8000

MODEL_OUTPUT=results/fishaudio-drama-3
[[ -d "$MODEL_OUTPUT" ]] || { echo 'Client produced no model results directory.' >&2; exit 1; }
for sidecar in "$MODEL_OUTPUT"/TTFA_MODEL_*.json; do
    [[ -e "$sidecar" ]] || continue
    case "$sidecar" in
        *__sglang__"$JOB_FLAVOR".json) ;;
        *) mv "$sidecar" "${sidecar%.json}__sglang__${JOB_FLAVOR}.json" ;;
    esac
done
echo 'ASR, SIM, and scoring are separate steps; the exit handler preserves this attempt before shutdown.'
DRAMA_JOB_SCRIPT
}

drama_bootstrap_script() {
    cat <<'DRAMA_BOOTSTRAP_SCRIPT'
#!/usr/bin/env bash
# Installs the pinned runtime inside an ephemeral job; never builds or publishes an image.
set -euo pipefail

EXPECTED_IMAGE=lmsysorg/sglang@sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75
SERVER_REVISION=12f65fa60fbb5217ad47f58b2e58b18092c818a1
[[ "${DRAMA_RUNTIME_IMAGE:-}" == "$EXPECTED_IMAGE" ]] || { echo 'Runtime bootstrap requires the approved immutable SGLang base.' >&2; exit 2; }
[[ "${SGLANG_OMNI_AUTO_CLONE:-}" == 0 ]] || { echo 'Moving-main initialization must be disabled before bootstrap.' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-/opt/sglang/bin/python3}"
BOOTSTRAP_ROOT="${DRAMA_BOOTSTRAP_ROOT:-/opt/fish-drama-3-runtime}"
OMNI_ROOT="$BOOTSTRAP_ROOT/sglang-omni"
DRAMA_BUCKET_MOUNT="${DRAMA_BUCKET_MOUNT:-/results}"
[[ -x "$PYTHON_BIN" && -d "$DRAMA_BUCKET_MOUNT" ]] || { echo 'Expected Python interpreter or results mount is unavailable.' >&2; exit 2; }
BOOTSTRAP_PHASE=base-check
BOOTSTRAP_RECEIPT="$DRAMA_BUCKET_MOUNT/fishaudio-drama-3/runtime_bootstrap/$(date -u +%Y%m%dT%H%M%SZ)_$$.json"
write_receipt() {
    "$PYTHON_BIN" - "$BOOTSTRAP_RECEIPT" "$1" "$BOOTSTRAP_PHASE" "$EXPECTED_IMAGE" "$OMNI_ROOT" "$SERVER_REVISION" "${2:-}" <<'PY' || true
import datetime
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys

destination, exit_code, phase, image, source_root, server_revision, status = sys.argv[1:]
packages = {}
for name in ("torch", "torchvision", "torchaudio", "torchcodec", "sglang", "flashinfer-python",
             "flash-attn-4", "transformers", "descript-audiotools", "descript-audio-codec", "protobuf", "datasets"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
try:
    installed_source = subprocess.check_output(
        ["git", "-C", source_root, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    installed_source = None
try:
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"],
                            check=False, capture_output=True, text=True, timeout=60)
    # Public image packages may include direct URLs; strip URL credentials and query strings.
    dependency_freeze = re.sub(r"(https?://)[^/\s@]+@", r"\1<redacted>@", freeze.stdout)
    dependency_freeze = re.sub(r"(https?://[^\s?#]+)[?#][^\s]*", r"\1", dependency_freeze)
    freeze_status = freeze.returncode
except (OSError, subprocess.TimeoutExpired):
    dependency_freeze, freeze_status = "", None
def tool_version(command):
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()[:2000] if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None
payload = {
    "status": status or ("complete" if exit_code == "0" else "failed"), "exit_code": int(exit_code),
    "phase": phase, "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "image": image, "python": sys.version.split()[0], "source_root": source_root,
    "expected_server_revision": server_revision, "installed_server_revision": installed_source,
    "transport_scope": "single node, all Drama 3 stages on GPU 0, CUDA IPC / SHM",
    "ucx_build_performed": False, "ucx_version": tool_version(["ucx_info", "-v"]),
    "sox_version": tool_version(["sox", "--version"]), "packages": packages,
    "dependency_freeze": dependency_freeze.splitlines(), "freeze_exit_code": freeze_status,
    "note": "Single-node bootstrap only; does not establish cross-node RDMA/UCX compatibility or certify synthesis.",
}
target = Path(destination)
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(target)
print(f"Runtime bootstrap receipt saved ({payload['status']}, phase={phase}).")
PY
}
finish_bootstrap() {
    bootstrap_exit=$?
    trap - EXIT
    write_receipt "$bootstrap_exit"
    exit "$bootstrap_exit"
}
trap finish_bootstrap EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
write_receipt 0 running

"$PYTHON_BIN" - <<'PY'
import os
import platform
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit("Bootstrap requires Python 3.12 from the approved base.")
if platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("Bootstrap requires the approved Linux amd64 image.")
if os.environ.get("CUDA_VERSION") != "13.0.3":
    raise SystemExit("Bootstrap requires the approved CUDA 13.0.3 base.")
import importlib.metadata
import torch

for name, expected in (("torch", "2.13.0"), ("sglang", "0.5.21"), ("flashinfer-python", "0.6.18")):
    installed = importlib.metadata.version(name).partition("+")[0]
    if installed != expected:
        raise SystemExit(f"Base dependency mismatch: {name} expected {expected}, got {installed}")
if torch.version.cuda is None or not torch.version.cuda.startswith("13."):
    raise SystemExit("The installed Torch build must target CUDA 13.")
if not torch.cuda.is_available():
    raise SystemExit("The approved GPU runtime is unavailable.")
if torch.cuda.device_count() != 1:
    raise SystemExit("This bootstrap supports one GPU with all model stages on GPU 0.")
print("Verified Python 3.12, CUDA 13.0.3, Torch 2.13.0, SGLang 0.5.21 and FlashInfer 0.6.18.")
PY

for bootstrap_tool in git uv ffmpeg; do
    command -v "$bootstrap_tool" >/dev/null || { printf 'Missing bootstrap tool: %s\n' "$bootstrap_tool" >&2; exit 1; }
done
[[ ! -e "$BOOTSTRAP_ROOT" ]] || { echo 'Bootstrap source directory already exists; refusing to replace it.' >&2; exit 1; }
mkdir -p "$BOOTSTRAP_ROOT"

# The pinned upstream Dockerfile adds the sox executable for audio dependencies.
BOOTSTRAP_PHASE=audio-system-dependencies
write_receipt 0 running
if ! command -v sox >/dev/null; then
    apt-get update
    apt-get install -y --no-install-recommends sox
    rm -rf /var/lib/apt/lists/*
fi

BOOTSTRAP_PHASE=omni-install
write_receipt 0 running
git init -q "$OMNI_ROOT"
git -C "$OMNI_ROOT" fetch -q --depth 1 https://github.com/sgl-project/sglang-omni.git "$SERVER_REVISION"
git -C "$OMNI_ROOT" checkout -q --detach FETCH_HEAD
[[ "$(git -C "$OMNI_ROOT" rev-parse HEAD)" == "$SERVER_REVISION" ]] || { echo 'Omni source revision mismatch.' >&2; exit 1; }
"$PYTHON_BIN" - "$OMNI_ROOT/pyproject.toml" "$BOOTSTRAP_ROOT/omni-overrides.txt" <<'PY'
from pathlib import Path
import sys
import tomllib

with open(sys.argv[1], "rb") as source:
    requirements = tomllib.load(source)["tool"]["uv"]["override-dependencies"]
Path(sys.argv[2]).write_text("\n".join(requirements) + "\n", encoding="utf-8")
PY
(
    cd "$OMNI_ROOT"
    uv pip install --python "$PYTHON_BIN" --prerelease=allow --no-build-isolation \
        --overrides "$BOOTSTRAP_ROOT/omni-overrides.txt" -e '.[fishaudio]' datasets==3.6.0
    "$PYTHON_BIN" -m pip uninstall -y flashinfer-cubin flashinfer-jit-cache
)

BOOTSTRAP_PHASE=verify-installed
write_receipt 0 running
"$PYTHON_BIN" "$OMNI_ROOT/.github/scripts/verify_omni_installed_pins.py" "$PYTHON_BIN" "$OMNI_ROOT"
"$PYTHON_BIN" - "$OMNI_ROOT" <<'PY'
import importlib.metadata
from pathlib import Path
import subprocess
import sys

import audiotools
import dac
import sglang_omni
import torch
from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.comm.router import CommRouter
from sglang_omni.models.fishaudio_drama3.config import Drama3PipelineConfig
from sglang_omni.platforms import current_platform
import sglang_omni.cli
import sglang_omni.relay

root = Path(sys.argv[1]).resolve()
if not (root / "sglang_omni").samefile(Path(sglang_omni.__file__).resolve().parent):
    raise SystemExit("Omni import does not resolve to the retained pinned source checkout.")
if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
    raise SystemExit("Installed Omni checkout has modified tracked files.")
for name, expected in (("descript-audiotools", "0.7.2"), ("descript-audio-codec", "1.0.0")):
    if importlib.metadata.version(name) != expected:
        raise SystemExit(f"Fish codec dependency mismatch: {name}")
if importlib.metadata.version("datasets") != "3.6.0":
    raise SystemExit("The evaluation runner requires datasets==3.6.0 for decoded prompt audio.")
if Drama3PipelineConfig.architecture != "FishOmniModel":
    raise SystemExit("Drama 3 pipeline registration mismatch.")
if torch.version.cuda is None or not torch.version.cuda.startswith("13.") or not torch.cuda.is_available():
    raise SystemExit("Installed runtime no longer has the required CUDA 13 GPU backend.")
if current_platform.get_intra_node_transport() != TransportKind.CUDA_IPC:
    raise SystemExit("This single-node bootstrap requires the pinned CUDA IPC transport.")
if any(stage.gpu != 0 for stage in Drama3PipelineConfig.model_fields["stages"].default):
    raise SystemExit("The pinned Drama 3 placement no longer uses only GPU 0.")
print("Verified pinned clean Omni source and Drama 3 codec imports.")
PY
BOOTSTRAP_PHASE=complete
DRAMA_BOOTSTRAP_SCRIPT
}

inject_script() {
    local target="$1" payload
    payload="$(gzip -9c | base64 -w0)"
    printf "mkdir -p /app/fishaudio-drama-3 && echo '%s' | base64 -d | gzip -dc > %s &&" "$payload" "$target"
}

EXECUTE=false
for option in "$@"; do
    case "$option" in
        --execute) EXECUTE=true ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$option" >&2; usage >&2; exit 2 ;;
    esac
done

MODEL_ID="${MODEL_ID:-fishaudio/drama-3}"
MODEL_REVISION="${MODEL_REVISION:-e0a9218aa48888270f06ef1ed9ea67622c0e17d2}"
SERVER_REVISION="${SERVER_REVISION:-12f65fa60fbb5217ad47f58b2e58b18092c818a1}"
[[ "$MODEL_ID" == fishaudio/drama-3 ]] || { echo 'Only fishaudio/drama-3 is supported.' >&2; exit 2; }
[[ "$MODEL_REVISION" == e0a9218aa48888270f06ef1ed9ea67622c0e17d2 ]] || { echo 'Unexpected Drama 3 model revision.' >&2; exit 2; }
[[ "$SERVER_REVISION" == 12f65fa60fbb5217ad47f58b2e58b18092c818a1 ]] || { echo 'Unexpected SGLang Omni revision.' >&2; exit 2; }

DATASET_SCOPE="${DATASET_SCOPE:-seed_en}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-8}"
JOB_TIMEOUT="${JOB_TIMEOUT:-30m}"
FIRST_JOB_TIMEOUT="${FIRST_JOB_TIMEOUT:-$JOB_TIMEOUT}"
VOICE_CLONE="${VOICE_CLONE:-false}"
TTFA_PROBE="${TTFA_PROBE:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
FLAVOR="${FLAVOR:-h200}"
DATASET_NAMESPACE="${DATASET_NAMESPACE:-bezzam}"
DATASET_PATH="${DATASET_PATH:-${DATASET_NAMESPACE}/seed_tts_eval}"
CV3_EVAL_PATH="${CV3_EVAL_PATH:-${DATASET_NAMESPACE}/cv3_eval}"
ORG_NAME="${ORG_NAME:-}"
TTS_IMAGE="${TTS_IMAGE:-}"
BOOTSTRAP_RUNTIME="${BOOTSTRAP_RUNTIME:-false}"
INPUTS_BUCKET="${INPUTS_BUCKET:-}"
INPUTS_PREFIX="${INPUTS_PREFIX:-}"
[[ "$BOOTSTRAP_RUNTIME" == false || "$BOOTSTRAP_RUNTIME" == true ]] || { echo 'BOOTSTRAP_RUNTIME must be true or false.' >&2; exit 2; }
[[ -z "$TTS_IMAGE" || -z "${TTS_SPACE:-}" ]] || { echo 'TTS_IMAGE and TTS_SPACE are mutually exclusive.' >&2; exit 2; }
if [[ -n "$TTS_IMAGE" && ! "$TTS_IMAGE" =~ ^[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)+@sha256:[0-9a-f]{64}$ ]]; then
    echo 'TTS_IMAGE must be a repository pinned by a lowercase sha256 digest; moving tags are rejected.' >&2
    exit 2
fi
if [[ "$BOOTSTRAP_RUNTIME" == true ]]; then
    [[ "$TTS_IMAGE" == lmsysorg/sglang@sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75 ]] || { echo 'Runtime bootstrap requires the approved immutable SGLang linux/amd64 base.' >&2; exit 2; }
    PYTHON_BIN="${PYTHON_BIN:-/opt/sglang/bin/python3}"
    [[ "$PYTHON_BIN" == /opt/sglang/bin/python3 ]] || { echo 'Runtime bootstrap requires PYTHON_BIN=/opt/sglang/bin/python3.' >&2; exit 2; }
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi
RESULTS_PREFIX="${RESULTS_PREFIX:-}"
if [[ -n "$RESULTS_PREFIX" && ! "$RESULTS_PREFIX" =~ ^[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$ ]]; then
    echo 'RESULTS_PREFIX must contain safe relative path components; no traversal, whitespace, or shell metacharacters.' >&2
    exit 2
fi
RESULTS_SUBPATH="${RESULTS_PREFIX:+/${RESULTS_PREFIX}}"
if [[ -n "$INPUTS_PREFIX" && ! "$INPUTS_PREFIX" =~ ^[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$ ]]; then
    echo 'INPUTS_PREFIX must contain safe relative path components.' >&2
    exit 2
fi
[[ -z "$INPUTS_PREFIX" || -n "$INPUTS_BUCKET" ]] || { echo 'INPUTS_PREFIX requires INPUTS_BUCKET.' >&2; exit 2; }
[[ -z "$INPUTS_BUCKET" || "$INPUTS_BUCKET" =~ ^[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*$ ]] || { echo 'INPUTS_BUCKET must be owner/name.' >&2; exit 2; }
INPUTS_SUBPATH="${INPUTS_PREFIX:+/${INPUTS_PREFIX}}"
[[ "$MAX_EVAL_SAMPLES" =~ ^(-1|[1-9][0-9]*)$ ]] || { echo 'MAX_EVAL_SAMPLES must be -1 or positive.' >&2; exit 2; }
[[ "$TTFA_PROBE" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'TTFA_PROBE must be zero or positive.' >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo 'BATCH_SIZE must be positive.' >&2; exit 2; }
[[ "$WARMUP_STEPS" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'WARMUP_STEPS must be nonnegative.' >&2; exit 2; }
[[ "$JOB_TIMEOUT" =~ ^[1-9][0-9]*[smhd]$ ]] || { echo 'JOB_TIMEOUT must be a positive duration, e.g. 30m or 8h.' >&2; exit 2; }
[[ "$FIRST_JOB_TIMEOUT" =~ ^[1-9][0-9]*[smhd]$ ]] || { echo 'FIRST_JOB_TIMEOUT must be a positive duration.' >&2; exit 2; }
[[ "$VOICE_CLONE" == false || "$VOICE_CLONE" == true ]] || { echo 'VOICE_CLONE must be true or false.' >&2; exit 2; }
if [[ "$TTFA_PROBE" == 0 ]]; then
    [[ "$FLAVOR" == h200 ]] || { echo 'Offline RTFx generation requires FLAVOR=h200.' >&2; exit 2; }
else
    [[ "$FLAVOR" == h200 || "$FLAVOR" == a100-large ]] || { echo 'TTFA supports h200 or a100-large GPU jobs.' >&2; exit 2; }
    [[ "$BATCH_SIZE" == 1 ]] || { echo 'TTFA requires BATCH_SIZE=1.' >&2; exit 2; }
fi

JOB_CONFIGS=()
case "$DATASET_SCOPE" in
    seed_en) JOB_CONFIGS=("$DATASET_PATH tts en") ;;
    cv3_en) JOB_CONFIGS=("$CV3_EVAL_PATH zero_shot en") ;;
    full)
        JOB_CONFIGS=("$DATASET_PATH tts en" "$DATASET_PATH tts zh")
        for language in en zh fr ja ko de es it ru; do
            JOB_CONFIGS+=("$CV3_EVAL_PATH zero_shot $language")
        done
        ;;
    *) echo 'DATASET_SCOPE must be seed_en, cv3_en, or full.' >&2; exit 2 ;;
esac
for dataset_repo in "$DATASET_PATH" "$CV3_EVAL_PATH"; do
    [[ "$dataset_repo" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo 'Dataset IDs must be owner/repository.' >&2; exit 2; }
done
[[ -z "$ORG_NAME" || "$ORG_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid ORG_NAME.' >&2; exit 2; }

printf 'Model: %s @ %s\nSGLang Omni: %s\n' "$MODEL_ID" "$MODEL_REVISION" "$SERVER_REVISION"
printf 'Plan: %s serial job(s), scope=%s, rows per split=%s, clone=%s, TTFA=%s\n' "${#JOB_CONFIGS[@]}" "$DATASET_SCOPE" "$MAX_EVAL_SAMPLES" "$VOICE_CLONE" "$TTFA_PROBE"
printf 'Per job: flavor=%s timeout=%s concurrency=%s warm-up=%s\n' "$FLAVOR" "$JOB_TIMEOUT" "$BATCH_SIZE" "$WARMUP_STEPS"
[[ "$FIRST_JOB_TIMEOUT" == "$JOB_TIMEOUT" ]] || printf 'First job timeout: %s\n' "$FIRST_JOB_TIMEOUT"
SELECTED_IMAGE="${TTS_IMAGE:-${TTS_SPACE:+hf.co/spaces/${TTS_SPACE}}}"
printf 'Image: %s\nBucket: %s\n' "${SELECTED_IMAGE:-<set TTS_SPACE or TTS_IMAGE>}" "${RESULTS_BUCKET:-<set RESULTS_BUCKET>}"
printf 'Prefix: %s\n' "${RESULTS_PREFIX:-<bucket root>}"
printf 'Runtime bootstrap: %s\nInputs: %s\n' "$BOOTSTRAP_RUNTIME" "${INPUTS_BUCKET:-<dataset Hub access>}"
[[ -z "$INPUTS_BUCKET" ]] || printf 'Inputs prefix: %s (read-only at /inputs)\n' "${INPUTS_PREFIX:-<bucket root>}"
printf '  %s\n' "${JOB_CONFIGS[@]}"
if [[ "$EXECUTE" == false ]]; then
    echo 'Preview only. Pass --execute after approval of this bounded run.'
    exit 0
fi

[[ -n "$SELECTED_IMAGE" && -n "${RESULTS_BUCKET:-}" ]] || { echo '--execute requires explicit TTS_SPACE or TTS_IMAGE, and RESULTS_BUCKET.' >&2; exit 2; }
[[ -z "${TTS_SPACE:-}" || "$TTS_SPACE" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo 'TTS_SPACE must be owner/name.' >&2; exit 2; }
[[ "$RESULTS_BUCKET" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo 'RESULTS_BUCKET must be owner/name.' >&2; exit 2; }
command -v hf >/dev/null || { echo 'The hf CLI is required for --execute.' >&2; exit 2; }
command -v python3 >/dev/null || { echo 'Local python3 is required to verify remote job completion.' >&2; exit 2; }

# Shared injectors use GNU base64 -w0; normalize that call on macOS too.
base64() {
    if [[ "${1:-}" == -w0 ]]; then
        shift
        if [[ $# -gt 0 ]]; then command base64 < "$1"; else command base64; fi | tr -d '\r\n'
    else
        command base64 "$@"
    fi
}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

INJECTION="$(inject_cmd_gz "${BACKEND_DIR}/run_eval.py" /app/fishaudio-drama-3/run_eval.py) "
INJECTION+="$(drama_job_script | inject_script /app/fishaudio-drama-3/run_job.sh) "
BOOTSTRAP_COMMAND=""
if [[ "$BOOTSTRAP_RUNTIME" == true ]]; then
    INJECTION+="$(drama_bootstrap_script | inject_script /app/fishaudio-drama-3/bootstrap_runtime.sh) "
    BOOTSTRAP_COMMAND='bash /app/fishaudio-drama-3/bootstrap_runtime.sh &&'
fi
for shared_file in run_eval_utils.py ttfa_probe.py; do
    INJECTION+="$(inject_cmd_gz "${REPO_ROOT}/scripts/${shared_file}" "/app/scripts/${shared_file}") "
done
SOURCE_IDENTITY="$(python3 - "$REPO_ROOT" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted((root / "normalizer").glob("*.py")):
    digest.update(path.name.encode() + b"\0" + path.read_bytes())
def git(*args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
print(json.dumps({
    "checkout_head": git("rev-parse", "HEAD"),
    "committed_tree": git("rev-parse", "HEAD:normalizer"),
    "normalizer_file_sha256": hashlib.sha256((root / "normalizer/normalizer.py").read_bytes()).hexdigest(),
    "python_files_sha256": digest.hexdigest(),
}))
PY
)"

NEXT_JOB_TIMEOUT="$FIRST_JOB_TIMEOUT"
for job_config in "${JOB_CONFIGS[@]}"; do
    ACTIVE_JOB_TIMEOUT="$NEXT_JOB_TIMEOUT"
    NEXT_JOB_TIMEOUT="$JOB_TIMEOUT"
    read -r job_dataset_path job_dataset job_split <<< "$job_config"
    CLIENT_ARGS=(--model_id "$MODEL_ID" --model_revision "$MODEL_REVISION" --server_revision "$SERVER_REVISION"
        --dataset_path "$job_dataset_path" --dataset "$job_dataset" --split "$job_split"
        --max_eval_samples "$MAX_EVAL_SAMPLES" --batch_size "$BATCH_SIZE" --warmup_steps "$WARMUP_STEPS"
        --ttfa_probe "$TTFA_PROBE")
    if [[ "$VOICE_CLONE" == true ]]; then CLIENT_ARGS+=(--voice_clone); else CLIENT_ARGS+=(--no-voice_clone); fi
    [[ -z "$INPUTS_BUCKET" ]] || CLIENT_ARGS+=(--dataset_local_root /inputs)
    printf -v QUOTED_CLIENT_ARGS '%q ' "${CLIENT_ARGS[@]}"
    JOB_ARGS=(jobs run --flavor "$FLAVOR" --timeout "$ACTIVE_JOB_TIMEOUT" --format agent --secrets HF_TOKEN
        --env "JOB_FLAVOR=$FLAVOR" --env "PYTHON_BIN=$PYTHON_BIN"
        --env "OPEN_TTS_SOURCE_IDENTITY=$SOURCE_IDENTITY"
        --volume "hf://buckets/${RESULTS_BUCKET}${RESULTS_SUBPATH}:/results")
    [[ -z "$INPUTS_BUCKET" ]] || JOB_ARGS+=(--volume "hf://buckets/${INPUTS_BUCKET}${INPUTS_SUBPATH}:/inputs:ro")
    if [[ "$BOOTSTRAP_RUNTIME" == true ]]; then
        JOB_ARGS+=(--env SGLANG_OMNI_AUTO_CLONE=0 --env "DRAMA_RUNTIME_IMAGE=$TTS_IMAGE"
            --env LD_LIBRARY_PATH=/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64)
    fi
    [[ -z "$ORG_NAME" ]] || JOB_ARGS+=(--namespace "$ORG_NAME")
    printf 'Submitting generation job: %s / %s / %s\n' "$job_dataset_path" "$job_dataset" "$job_split"
    JOB_LOG="$(mktemp "${TMPDIR:-/tmp}/drama_job_XXXXXX")"
    trap 'rm -f "$JOB_LOG"' EXIT
    hf "${JOB_ARGS[@]}" "$SELECTED_IMAGE" bash -c \
        "set -e; ${INJECTION} ${SCRIPTS_PYTHONPATH} ${BOOTSTRAP_COMMAND} bash /app/fishaudio-drama-3/run_job.sh ${QUOTED_CLIENT_ARGS}" | tee "$JOB_LOG"
    IFS= read -r JOB_START_LINE < "$JOB_LOG"
    START_PATTERN='^id=([A-Za-z0-9_-]+)( name=[^[:space:]]+)? url=https://huggingface.co/jobs/([A-Za-z0-9_.-]+)/([A-Za-z0-9_-]+)$'
    [[ "$JOB_START_LINE" =~ $START_PATTERN ]] || { echo 'Cannot verify the submitted job identity; stopping without retry.' >&2; exit 1; }
    [[ "${BASH_REMATCH[1]}" == "${BASH_REMATCH[4]}" ]] || { echo 'Job ID and URL disagree; stopping.' >&2; exit 1; }
    JOB_REFERENCE="${BASH_REMATCH[3]}/${BASH_REMATCH[1]}"
    # Older hf versions stream logs but return success even when the remote job failed.
    hf jobs inspect "$JOB_REFERENCE" --format json | python3 -c '
import json
import sys

expected_owner, expected_id = sys.argv[1].split("/", 1)
jobs = json.load(sys.stdin)
if not isinstance(jobs, list) or len(jobs) != 1:
    raise SystemExit("Expected exactly one remote job status; stopping without retry.")
job = jobs[0]
if job.get("id") != expected_id or job.get("owner", {}).get("name") != expected_owner:
    raise SystemExit("Inspected job identity does not match the submitted job.")
stage = job.get("status", {}).get("stage")
if stage != "COMPLETED":
    raise SystemExit(f"Remote job is {stage!r}, not COMPLETED; stopping without retry.")
print(f"Verified remote job {expected_owner}/{expected_id} completed.")
' "$JOB_REFERENCE"
    rm -f "$JOB_LOG"
    trap - EXIT
done
