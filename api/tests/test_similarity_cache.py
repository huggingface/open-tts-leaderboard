"""Exercise the shared SIM scorer's real cache path without torch or remote Jobs."""

import ast
import io
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from api import run_pipeline, score_results


def run_shared_scorer(current, previous):
    source_path = "/results/model/manifest.jsonl"
    fork_path = "/results/model/manifest_wavlm_seed_tts.jsonl"
    files = {source_path: json.dumps(current) + "\n", fork_path: json.dumps(previous) + "\n"}

    @contextmanager
    def memory_open(path, mode="r", **kwargs):
        stream = io.StringIO(files[str(path)] if "r" in mode else "")
        yield stream
        if "w" in mode:
            files[str(path)] = stream.getvalue()

    # Compile the actual scorer entry point, replacing its model runtime only.
    # This catches regressions in carrying, skipping and rewriting the SIM fork.
    source = Path(__file__).resolve().parents[2] / "transformers" / "score_similarity.py"
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    model_name = "wavlm_large_finetune+ecapa_tdnn (wavlm_large_finetune.pth)"
    embedder = Mock(name=model_name, sampling_rate=16000, min_samples=320, max_batch_size=1)
    embedder.name = model_name
    load_audio = Mock(return_value=[0.0] * 16000)
    namespace = {
        "json": json, "os": os,
        "torch": SimpleNamespace(bfloat16=None, cuda=SimpleNamespace(is_available=lambda: False)),
        "F": SimpleNamespace(cosine_similarity=lambda *args, **kwargs: SimpleNamespace(tolist=lambda: [0.42])),
        "sim_manifest_path": lambda path, backend: fork_path,
        "build_embedder": lambda args, dtype: embedder,
        "LEGACY_SIM_MODEL": "microsoft/wavlm-base-plus-sv",
        "load_audio": load_audio,
    }
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102 — checked-in scorer, mocked runtime
    args = SimpleNamespace(
        dtype="bfloat16", manifest_path=source_path, output_manifest_path=None,
        sim_backend="wavlm_seed_tts", batch_size=1, max_audio_seconds=30,
        overwrite=False, save_every=32,
    )
    with patch("builtins.open", memory_open), patch("os.path.exists", return_value=True):
        namespace["main"](args)
    return json.loads(files[fork_path]), load_audio


def audio_pair():
    current = {
        "audio_filepath": "output_0.wav", "prompt_audio_filepath": "prompt_0.wav",
        "text": "Same text", "pred_text": "Same text", "duration": 1,
        "timing_backend": "api", "audio_sha256": "new-generated-audio",
        "prompt_audio_sha256": "reference-audio", "api_latency_s": 4.0,
    }
    previous = {
        **current, "sim": 0.95,
        "sim_model": "wavlm_large_finetune+ecapa_tdnn (wavlm_large_finetune.pth)",
        "api_latency_s": 1.0,
    }
    return current, previous


@pytest.mark.parametrize("field", ["audio_sha256", "prompt_audio_sha256"])
def test_resume_recomputes_remote_sim_after_audio_pair_changes(field):
    current, previous = audio_pair()
    previous[field] = "old-audio"
    args = run_pipeline.make_parser().parse_args([
        "--models", "fish/s2.1-pro", "--datasets", "seed_tts", "--only_langs", "en",
        "--voice_clone", "--resume", "--results_bucket", "example/api-results",
    ])
    config = run_pipeline.dataset_configs(args)[0]
    manifest = run_pipeline.manifest_path("fish/s2.1-pro", config, True)
    assert "--overwrite" not in run_pipeline.job_command(args, "sim", manifest, "en")[-1]
    scored, loader = run_shared_scorer(current, previous)
    assert scored["sim"] == 0.42 and loader.call_count == 2
    score_results.assert_current_sim([current], [scored], manifest)
    score_results.checked_similarity([scored], "wavlm_seed_tts", manifest)


def test_resume_reuses_scores_for_identical_audio_pair():
    current, previous = audio_pair()
    scored, loader = run_shared_scorer(current, previous)
    assert scored["sim"] == 0.95 and scored["api_latency_s"] == current["api_latency_s"]
    loader.assert_not_called()


@pytest.mark.parametrize("missing_in", ["current", "previous"])
def test_api_scores_without_hashes_are_recomputed(missing_in):
    current, previous = audio_pair()
    for field in ("audio_sha256", "prompt_audio_sha256"):
        (current if missing_in == "current" else previous).pop(field)
    scored, loader = run_shared_scorer(current, previous)
    assert scored["sim"] == 0.42 and loader.call_count == 2


def test_local_backend_resume_remains_compatible_without_api_hashes():
    current, previous = audio_pair()
    for row in (current, previous):
        for field in ("timing_backend", "audio_sha256", "prompt_audio_sha256"):
            row.pop(field)
    scored, loader = run_shared_scorer(current, previous)
    assert scored["sim"] == 0.95
    loader.assert_not_called()


def test_export_rejects_fork_with_an_old_audio_hash():
    current, previous = audio_pair()
    previous["audio_sha256"] = "old-generated-audio"
    previous["api_latency_s"] = current["api_latency_s"]
    with pytest.raises(ValueError, match="stale"):
        score_results.assert_current_sim([current], [previous], Path("manifest_wavlm_seed_tts.jsonl"))
