"""Drive the actual shell orchestrator with a local HF Jobs capture (no paid jobs)."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "stages,device,overwrite",
    [
        ("komos", "cpu", False),
        ("generate komos", "cuda:0", True),
    ],
)
def test_komos_job_is_korean_only_and_supports_rescoring(
    tmp_path: Path,
    stages: str,
    device: str,
    overwrite: bool,
) -> None:
    """Only the Korean combo submits an isolated, pinned scoring job."""
    script = tmp_path / "backend" / "submit_jobs.sh"
    script.parent.mkdir()
    capture = tmp_path / "jobs.txt"
    script.write_text("""#!/bin/bash
set -e
# The project's injection helper expects GNU base64; make its -w0 form portable for this test.
base64() { command base64 < "$2" | tr -d '\\n'; }
# BSD mktemp does not support the existing Linux template suffix.
mktemp() { command mktemp "${TMPDIR:-/tmp}/komos-stage.XXXXXX"; }
source "$REPO_ROOT/scripts/tts_jobs_common.sh"
TTS_SPACE=example/scorer
MODEL_CONFIGS=("example/model")
DATASET_CONFIGS=("zero_shot en en" "zero_shot ko ko" "minimax ko ko example/minimax")
generate_stage() { :; }
hf() { printf '%s\\n' "$*" >> "$CAPTURE"; }
_wait_and_score() { for pid in "${PIDS[@]}"; do wait "$pid"; done; }
run_pipeline
""")
    env = os.environ | {
        "REPO_ROOT": str(ROOT),
        "CAPTURE": str(capture),
        "STAGES": stages,
        "KOMOS_DEVICE": device,
        "ONLY_LANGS": "",
        "KOMOS_LANGS": "ko",
        "KOMOS_OVERWRITE": "false",
        "MAX_EVAL_SAMPLES": "8",
    }

    subprocess.run(
        ["bash", str(script)], env=env, check=True, capture_output=True, text=True
    )

    commands = capture.read_text()
    assert commands.count("jobs run") == 2  # CV3 and a future MiniMax Korean split.
    assert "_en.jsonl" not in commands
    assert commands.count("--max_rows=8") == 2
    assert commands.count("--device=" + device) == 2
    assert (
        "komos==0.0.2" in commands
        and "uv venv --python python /tmp/komos-env" in commands
    )
    assert ("--overwrite" in commands) is overwrite
