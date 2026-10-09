"""Exercise real WAV/manifest generation with an offline synthesis provider."""

import hashlib
import io
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests
import soundfile as sf
from requests.structures import CaseInsensitiveDict

from api import run_eval, score_results
from api.providers.base import (
    APIError,
    AudioResponse,
    Provider,
    SynthesisRequest,
    binary_audio,
    retry_after_seconds,
)


class OfflineProvider:
    supports_reference = True

    def __init__(self, failures=None):
        self.requests = []
        self.failures = failures or {}

    def synthesize(self, model, request):
        self.requests.append(request)
        if request.text in self.failures:
            raise self.failures[request.text]
        return AudioResponse(b"\x00\x10" * 240, "pcm_s16le", 24000, time.perf_counter(), "test")


def arguments(*extra):
    return run_eval.parse_args(["--model_id", "cartesia/sonic-3.6-2026-08-27", *extra])


def samples():
    return [{"id": "a", "text": "Hello world."}, {"id": "b", "text": "Second sample."}]


def manifest():
    return next(Path("results").glob("*/*.jsonl"))


def rows():
    return [json.loads(line) for line in manifest().read_text().splitlines()]


def test_generation_writes_portable_wavs_and_separate_timings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provider = OfflineProvider()
    meta = run_eval.run(arguments("--max_workers", "2"), provider, samples())
    assert meta["complete"] and meta["n_completed"] == 2
    assert meta["api_latency_s"]["n"] == 2 and meta["api_throughput_rtfx"] > 0
    for row in rows():
        assert row["time"] is None and row["timing_backend"] == "api"
        assert "pred_text" not in row
        assert row["api_attempts"] == 1 and row["api_ttfa_ms"] >= 0
        audio, rate = sf.read(manifest().parent / row["audio_filepath"])
        assert row["audio_sha256"] == hashlib.sha256((manifest().parent / row["audio_filepath"]).read_bytes()).hexdigest()
        assert len(audio) == 240 and rate == 24000
        assert not Path(row["audio_filepath"]).is_absolute()


def test_complete_resume_is_noop_and_changed_input_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = run_eval.run(arguments(), OfflineProvider(), samples())
    provider = OfflineProvider()
    assert run_eval.run(arguments("--resume"), provider, samples()) == first
    assert not provider.requests
    changed = samples()
    changed[0]["text"] = "Changed target text."
    with pytest.raises(ValueError, match="inputs differ"):
        run_eval.run(arguments("--resume"), provider, changed)


def test_failed_run_resumes_only_missing_audio_and_preserves_wall_time(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provider = OfflineProvider({"Second sample.": APIError("Failure", status_code=400)})
    with pytest.raises(RuntimeError, match="Incomplete API run"):
        run_eval.run(arguments(), provider, samples())
    assert len(rows()) == 1
    old = json.loads(manifest().with_suffix(".run.json").read_text())
    assert old["complete"] is False and old["n_failed"] == 1 and old["wall_time_s"] > 0
    provider = OfflineProvider()
    meta = run_eval.run(arguments("--resume"), provider, samples())
    assert [request.text for request in provider.requests] == ["Second sample."]
    assert meta["complete"] and meta["wall_time_s"] >= old["wall_time_s"]
    assert len(rows()) == 2


def test_auth_failure_stops_queuing_and_failure_details_are_redacted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provider = OfflineProvider({"Hello world.": APIError("secret=DO_NOT_LOG", status_code=401)})
    with pytest.raises(RuntimeError):
        run_eval.run(arguments(), provider, samples())
    assert len(provider.requests) == 1
    meta = manifest().with_suffix(".run.json").read_text()
    assert "DO_NOT_LOG" not in meta
    assert json.loads(meta)["failures"][0]["http_status"] == 401


def test_overwrite_invalidates_old_sim_and_asr(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_eval.run(arguments(), OfflineProvider(), samples())
    fork = manifest().with_name(manifest().stem + "_wavlm_seed_tts.jsonl")
    fork.write_text("stale scores")
    with pytest.raises(ValueError, match="Results exist"):
        run_eval.run(arguments(), OfflineProvider(), samples())
    run_eval.run(arguments("--overwrite"), OfflineProvider(), samples())
    assert not fork.exists()
    assert all("pred_text" not in row for row in rows())


def test_clone_uses_each_reference_and_non_clone_uses_one_fixed_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = [
        {"id": str(i), "text": f"Sample {i}.", "prompt_text": f"Reference {i}.",
         "prompt_audio": {"array": np.full(240, i / 10), "sampling_rate": 24000}}
        for i in (1, 2)
    ]
    provider = OfflineProvider()
    args = arguments("--model_id", "fish/s2.1-pro", "--voice_clone")
    run_eval.run(args, provider, data)
    assert provider.requests[0].reference_audio != provider.requests[1].reference_audio
    for row in rows():
        assert (manifest().parent / row["prompt_audio_filepath"]).is_file()
        assert row["prompt_audio_sha256"] == hashlib.sha256((manifest().parent / row["prompt_audio_filepath"]).read_bytes()).hexdigest()
        assert "sim" in row
    provider = OfflineProvider()
    run_eval.run(arguments("--model_id", "fish/s2.1-pro"), provider, data)
    assert provider.requests[0].reference_audio == provider.requests[1].reference_audio


def test_local_fixture_resolves_reference_and_caps_samples(tmp_path):
    fixture = tmp_path / "samples.jsonl"
    fixture.write_text('\n'.join(json.dumps(row) for row in [
        {"text": "Hello", "prompt_audio_filepath": "reference.wav"}, {"text": "Bye"},
    ]))
    args = arguments("--input_jsonl", str(fixture), "--max_eval_samples", "1")
    loaded = run_eval.load_samples(args, True)
    assert len(loaded) == 1 and loaded[0]["id"] == 0
    assert loaded[0]["prompt_audio_filepath"] == str(tmp_path / "reference.wav")


def test_ttfa_probe_is_api_labeled_and_keeps_generation_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_eval.run(arguments(), OfflineProvider(), samples())
    before = manifest().read_bytes()
    meta_before = manifest().with_suffix(".run.json").read_bytes()
    payload = run_eval.run(arguments("--ttfa_probe", "2"), OfflineProvider(), samples())
    assert payload["timing_backend"] == "api" and payload["device"] == "api"
    assert payload["batch_size"] == 1 and payload["n_samples"] == 2
    probes = list(Path("results").glob("*/TTFA_*.json"))
    assert len(probes) == 1 and probes[0].stem.endswith("__api")
    assert manifest().read_bytes() == before
    assert manifest().with_suffix(".run.json").read_bytes() == meta_before


@pytest.mark.parametrize("data", [
    [{"id": "bad/path", "text": "Hello"}],
    [{"id": "a", "text": "Hello"}, {"id": "a", "text": "Again"}],
    [{"id": "a", "text": "  "}],
])
def test_invalid_input_fails_before_api_call(tmp_path, monkeypatch, data):
    monkeypatch.chdir(tmp_path)
    provider = OfflineProvider()
    with pytest.raises(ValueError):
        run_eval.run(arguments(), provider, data)
    assert not provider.requests


def test_language_and_unsupported_clone_fail_before_api_call():
    provider = OfflineProvider()
    with pytest.raises(ValueError, match="must match"):
        run_eval.run(arguments("--split", "fr"), provider, samples())
    with pytest.raises(ValueError, match="does not support"):
        run_eval.run(arguments("--model_id", "deepgram/aura-2-thalia-en", "--split", "fr", "--language", "fr"), provider, samples())
    with pytest.raises(ValueError, match="Fish Audio and Mistral"):
        run_eval.run(arguments("--voice_clone"), provider, samples())
    assert not provider.requests


def test_retry_only_transient_errors_and_count_attempts():
    provider = Mock()
    response = AudioResponse(b"\x00\x00", "pcm_s16le", 24000)
    provider.synthesize.side_effect = [APIError("limited", retryable=True, retry_after=0.5), response]
    request = SynthesisRequest("Hello", "en", "voice")
    with patch("api.run_eval.time.sleep") as sleep:
        actual, latency, attempts = run_eval.synthesize_with_retry(provider, "model", request)
    assert actual == response and latency >= 0 and attempts == 2
    sleep.assert_called_once_with(0.5)
    provider.synthesize.side_effect = APIError("bad request", status_code=400)
    with patch("api.run_eval.time.sleep") as sleep, pytest.raises(APIError):
        run_eval.synthesize_with_retry(provider, "model", request)
    sleep.assert_not_called()


def test_transport_redacts_credentials_and_closes_rate_limit_response():
    provider = Provider("secret")
    with (
        patch("api.providers.base.requests.post", side_effect=requests.ConnectionError("https://secret")),
        pytest.raises(APIError) as caught,
        provider.post("https://provider"),
    ):
        pass
    assert "secret" not in str(caught.value) and caught.value.__suppress_context__
    response = Mock(status_code=429, headers={"Retry-After": "2"})
    with (
        patch("api.providers.base.requests.post", return_value=response),
        pytest.raises(APIError) as caught,
        provider.post("https://provider"),
    ):
        pass
    assert caught.value.retryable and caught.value.retry_after == 2
    response.close.assert_called_once()
    assert retry_after_seconds("NaN") is None


@pytest.mark.parametrize("chunks,content_type", [
    ([b"{\"error\":true}"], "application/json"),
    ([b"\x00"], "audio/pcm"),
    ([b"RIFF1234"], "audio/wav"),
    ([b"R", b"IFF1234"], "application/octet-stream"),
    ([], "audio/pcm"),
])
def test_binary_transport_rejects_errors_and_partial_audio(chunks, content_type):
    response = Mock(headers=CaseInsensitiveDict({"Content-Type": content_type}))
    response.iter_content.return_value = iter(chunks)
    with pytest.raises(APIError):
        binary_audio(response, "pcm_s16le", 24000)


def test_decode_container_keeps_actual_rate_and_downmixes():
    out = io.BytesIO()
    sf.write(out, np.full((80, 2), 0.25), 8000, format="WAV")
    audio, rate = run_eval.decode_audio(AudioResponse(out.getvalue(), "wav", 24000))
    assert rate == 8000 and audio.shape == (80,)
    with pytest.raises(APIError):
        run_eval.decode_audio(AudioResponse(b"not audio", "wav", 24000))


@pytest.mark.parametrize("language,text,prediction,metric,expected", [
    ("en", "Hello world", "Hello world", "WER", 0),
    ("zh", "你好世界", "你好", "CER", 50),
])
def test_real_shared_normalizer_scores_without_gpu_rtfx(tmp_path, monkeypatch, language, text, prediction, metric, expected):
    monkeypatch.chdir(tmp_path)
    args = arguments("--split", language, "--language", language)
    run_eval.run(args, OfflineProvider(), [{"id": "a", "text": text}])
    row = rows()[0]
    row["pred_text"] = prediction
    manifest().write_text(json.dumps(row, ensure_ascii=False) + "\n")
    exported = score_results.collect_results([manifest()])[0]
    assert exported["metric"] == metric and exported["wer"] == expected
    assert "rtfx" not in exported
    assert exported["timing_backend"] == "api"
