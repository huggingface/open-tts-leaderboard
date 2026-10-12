"""Offline integration tests: real WAV/manifest I/O, fake model, no cloud jobs or weights."""

import base64
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from datasets import Audio, Dataset, Features, Value

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("gibbstts_eval", Path(__file__).with_name("run_eval.py"))
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def make_dataset(language):
    return Dataset.from_dict({
        "id": [str(i) for i in range(3)],
        "text": ["Hello." if language == "en" else "你好。"] * 3,
        "prompt_text": ["Reference." if language == "en" else "参考声音。"] * 3,
        "prompt_audio": [{"array": np.ones(1600) * 0.125, "sampling_rate": 16000}] * 3,
    }, features=Features({"id": Value("string"), "text": Value("string"),
                          "prompt_text": Value("string"), "prompt_audio": Audio()}))


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Match the torch methods used by shared seeding and the CPU/CUDA timer.
    events = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        Tensor=type("Tensor", (), {}),
        Generator=type("Generator", (), {}),
        nn=SimpleNamespace(Module=type("Module", (), {})),
        manual_seed=lambda seed: None,
        cuda=SimpleNamespace(manual_seed_all=lambda seed: None,
                             synchronize=lambda device: events.append(("sync", device))),
    ))
    return events


@pytest.mark.parametrize("language", ["en", "zh"])
def test_batch_timing_padding_and_resume(setup, monkeypatch, language):
    monkeypatch.setattr(backend, "load_tts_dataset", lambda *args, **kwargs: make_dataset(language))
    calls = []
    def generate(rows, info):
        for row in rows:
            audio, sr = sf.read(row["prompt_path"])
            assert sr == 16000
            np.testing.assert_allclose(audio, 0.125, atol=1e-7)
        calls.append([r["id"] for r in rows])
        info.update(real_count=len(rows), execution_batch_size=2, dummy_count=2-len(rows))
        return [np.ones(2400, dtype=np.float32)*0.25 for _ in rows]
    monkeypatch.setattr(backend, "configure_inference", lambda *args: (generate, None))
    args = backend.build_parser().parse_args(["--split", language, "--device", "cpu",
        "--precision", "fp32", "--no-compile", "--batch_size", "2"])
    clock = iter([0,10,20,30,40,42,50,54])
    monkeypatch.setattr(backend.time, "perf_counter", lambda: next(clock))
    backend.evaluate(args, None)
    assert calls == [["0","1"],["0","1"],["0","1"],["2"]]
    paths = backend.output_paths(args, "_voice_clone")
    rows = [json.loads(line) for line in Path(paths.manifest_path).read_text().splitlines()]
    assert len(rows)==3 and [r["time"] for r in rows]==[1,1,4]
    assert sum(r["time"] for r in rows)==6  # warm-up excluded; all tail work charged to real rows
    assert [r["dummy_count"] for r in rows]==[0,0,1]
    assert [r["actual_batch_size"] for r in rows]==[2,2,1]
    for row in rows:
        assert row["duration"] == pytest.approx(.1)
        assert row["pred_text"] == row["sim"] == ""
        assert Path(paths.model_dir, row["prompt_audio_filepath"]).is_file()
        audio, sr = sf.read(Path(paths.model_dir, row["audio_filepath"]))
        assert sr==24000 and len(audio)==2400
    args.resume=True
    backend.evaluate(args, None)
    assert len(calls)==4
    Path(paths.model_dir, rows[1]["audio_filepath"]).unlink()
    clock=iter([100,110,120,130,140,142])
    backend.evaluate(args, None)
    assert calls[-3:]==[["0","1"]]*3  # repair the whole batch, preserving original RNG shape
    repaired=[json.loads(line) for line in Path(paths.manifest_path).read_text().splitlines()]
    assert [r["sample_id"] for r in repaired]==["0","1","2"]
    args.steps=16
    with pytest.raises(ValueError, match="different generation settings"):
        backend.evaluate(args, None)


def test_cuda_timer_and_invalid_audio(setup, monkeypatch):
    args=backend.build_parser().parse_args(["--device", "cuda:0"])
    clock=iter([10,12.5])
    monkeypatch.setattr(backend.time, "perf_counter", lambda: next(clock))
    def generate(rows, info):
        setup.append(("generate", len(rows)))
        return [np.ones(2400)]
    _, elapsed, _=backend.generate_batch(generate,args,[{}])
    assert elapsed==2.5
    assert setup==[("sync","cuda:0"),("generate",1),("sync","cuda:0")]
    args.device="cpu"
    monkeypatch.setattr(backend.time,"perf_counter",lambda:1.)
    with pytest.raises(ValueError,match="nonfinite"):
        backend.generate_batch(lambda rows,info:[np.array([np.nan])],args,[{}])
    with pytest.raises(ValueError,match="one output"):
        backend.generate_batch(lambda rows,info:[],args,[{}])


@pytest.mark.parametrize("flags", [["--no-voice_clone"], ["--batch_size","0"],
    ["--split","ja"],["--ttfa_probe","8"],["--steps","0"],["--device","cuda:1"],
    ["--warmup_steps","0"],["--speed","nan"]])
def test_reject_unsupported_modes(flags):
    with pytest.raises(ValueError):
        backend.validate_args(backend.build_parser().parse_args(flags))


def test_submit_generates_h200_commands_without_cloud(tmp_path):
    """Run the actual shell backend with a failing fake hf; inspect its real job arguments."""
    hf = tmp_path / "hf"
    hf.write_text('''#!/bin/bash
exec "$TEST_PYTHON" - "$@" <<'PY'
import json, os, pathlib, sys
args = sys.argv[1:]
command = args[-1]
language = "zh" if "--split=zh" in command else "en"
dataset = "zero_shot" if "--dataset=zero_shot" in command else "tts"
pathlib.Path(os.environ["CAPTURE_DIR"], dataset + "_" + language + ".json").write_text(json.dumps(args))
sys.exit(99)
PY
''')
    hf.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
               CAPTURE_DIR=str(tmp_path), TEST_PYTHON=sys.executable, DATASET_NAMESPACE="ydqmkkx",
               RESULTS_BUCKET="ydqmkkx/test-results", TTS_SPACE="ydqmkkx/evals-gibbstts",
               STAGE_RETRIES="0", ONLY_LANGS="", FLAVOR="h200", VOICE_CLONE="true")
    result = subprocess.run(["bash", str(ROOT / "gibbstts/submit_jobs.sh")],
                            env=env, capture_output=True, text=True, timeout=20, cwd=tmp_path)
    assert "One or more job chains failed" in result.stdout
    assert result.returncode != 0
    captures = list(tmp_path.glob("*.json"))
    assert len(captures) == 4
    for capture in captures:
        command = json.loads(capture.read_text())
        assert command[:2] == ["jobs", "run"]
        assert command[command.index("--flavor") + 1] == "h200"
        assert "hf://buckets/ydqmkkx/test-results:/results" in command
        assert "--voice_clone" in command[-1]
        assert "--steps=32" in command[-1]
        assert "--batch_size=32 --precision=bf16 --compile --tail_padding" in command[-1]
        assert "/app/gibbstts/batched_inference.py" in command[-1]
        assert "/app/gibbstts/compiler_backend.py" in command[-1]
        expected_dataset = "cv3_eval" if "zero_shot" in capture.stem else "seed_tts_eval"
        assert f"ydqmkkx/{expected_dataset}" in command[-1]
        # Verify the generated script payload is complete (including macOS base64 handling).
        payload = command[-1].split("echo '", 1)[1].split("'", 1)[0]
        assert base64.b64decode(payload) == (ROOT / "gibbstts/run_eval.py").read_bytes()
    env["VOICE_CLONE"] = "false"
    result = subprocess.run(["bash", str(ROOT / "gibbstts/submit_jobs.sh")], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "requires VOICE_CLONE" in result.stderr


def test_portable_sync_preserves_failure_and_timeout(tmp_path):
    python = tmp_path / "python"
    python.write_text('#!/bin/bash\nexec "$TEST_PYTHON" "$@"\n')
    python.chmod(0o755)
    hf = tmp_path / "hf"
    hf.write_text('#!/bin/bash\nexit 17\n')
    hf.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
               TEST_PYTHON=sys.executable, SYNC_TIMEOUT="5")
    cmd = ['bash', '-c', 'source "$1"; sync_manifests_with_timeout', '_',
           str(ROOT / "scripts/tts_jobs_common.sh")]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 17
    env["SYNC_TIMEOUT"] = "0.1"
    hf.write_text('#!/bin/bash\nexec "$TEST_PYTHON" -c "import time; time.sleep(5)"\n')
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 124 and "timed out" in result.stderr


