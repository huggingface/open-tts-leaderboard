#!/bin/bash
# API generation runs in local Docker; only ASR/SIM use HF Jobs.
# Compatible with macOS's Bash 3.2 (no GNU base64 or associative arrays).
set -euo pipefail
API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${MODEL:-}" ]]; then
    set -- --models "$MODEL" "$@"
fi
cd "$API_DIR"
exec "$PYTHON_BIN" run_pipeline.py "$@"
