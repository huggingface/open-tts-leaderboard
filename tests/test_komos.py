"""KOMOS resume and audio handling tests; no checkpoint download required."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray

from score_komos import MODEL_REVISION, ScoreOptions, score_manifest


@dataclass(slots=True)
class RecordingModel:
    """Record mutable audio-call history in place of heavyweight neural inference."""

    calls: list[tuple[int, int]] = field(default_factory=list)

    def score_long_form(self, audio: NDArray[np.float32], sr: int) -> float:
        """Record mono waveform length/rate and return a calibrated test score."""
        assert audio.ndim == 1
        self.calls.append((len(audio), sr))
        return 4.25


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    """Create a manifest containing valid, short, and initially missing audio."""
    sf.write(tmp_path / "good.wav", np.zeros((8000, 2)), 8000)
    sf.write(tmp_path / "short.wav", np.zeros(100), 8000)
    path = tmp_path / "MODEL_test-model_DATASET_hf-audio_cv3_eval_zero_shot_ko.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"audio_filepath": name})
            for name in ["good.wav", "short.wav", "missing.wav"]
        ),
        encoding="utf-8",
    )
    return path


def test_resume_retries_missing_audio_but_preserves_completed_scores(
    manifest: Path,
) -> None:
    """A resumed run scores newly available audio without recomputing prior rows."""
    model = RecordingModel()
    original = manifest.read_bytes()
    output = score_manifest(model, manifest, ScoreOptions(save_every=1))
    sf.write(manifest.parent / "missing.wav", np.zeros(8000), 8000)

    score_manifest(model, manifest, ScoreOptions(save_every=1))

    data = json.loads(output.read_text())
    assert model.calls == [(8000, 8000), (8000, 8000)]
    assert data["scores"] == {"good.wav": 4.25, "short.wav": None, "missing.wav": 4.25}
    assert "missing.wav" not in data["errors"]
    assert "short.wav" in data["errors"]
    assert data["komos_model"]["revision"] == MODEL_REVISION
    assert manifest.read_bytes() == original


def test_overwrite_recomputes_valid_audio(manifest: Path) -> None:
    """Explicit overwrite invalidates cached scores even when filenames match."""
    model = RecordingModel()
    score_manifest(model, manifest, ScoreOptions())

    score_manifest(model, manifest, ScoreOptions(overwrite=True))

    assert len(model.calls) == 2


def test_resume_rejects_changed_preprocessing(manifest: Path) -> None:
    """Changing the clip cap requires overwrite instead of mixing protocols."""
    model = RecordingModel()
    score_manifest(model, manifest, ScoreOptions())

    with pytest.raises(ValueError, match="provenance changed"):
        score_manifest(model, manifest, ScoreOptions(max_audio_seconds=10))


@pytest.mark.parametrize("seconds,expected", [(10, 80000), (0, 88000)])
def test_audio_cap_and_mono_conversion(
    tmp_path: Path, seconds: int, expected: int
) -> None:
    """The requested prefix is passed as mono audio with its original rate."""
    sf.write(tmp_path / "audio.wav", np.zeros((88000, 2)), 8000)
    manifest = tmp_path / "ko.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": "audio.wav"}))
    model = RecordingModel()

    score_manifest(model, manifest, ScoreOptions(max_audio_seconds=seconds))

    assert model.calls == [(expected, 8000)]


def test_max_rows_and_output_directory(manifest: Path, tmp_path: Path) -> None:
    """A limited pilot writes only its selected rows to the requested directory."""
    model = RecordingModel()

    output = score_manifest(
        model, manifest, ScoreOptions(output_dir=tmp_path / "out", max_rows=1)
    )

    assert output.parent == tmp_path / "out"
    assert json.loads(output.read_text())["scores"] == {"good.wav": 4.25}


@pytest.mark.parametrize("cap", [-1, 61, float("nan")])
def test_invalid_audio_caps_are_rejected(cap: float) -> None:
    """Reject invalid preprocessing before loading a model or touching output."""
    with pytest.raises(ValueError, match="max_audio_seconds"):
        ScoreOptions(max_audio_seconds=cap)


def test_invalid_save_interval_is_rejected() -> None:
    """A zero checkpoint interval is a CLI error, not a modulo crash."""
    with pytest.raises(ValueError, match="save_every"):
        ScoreOptions(save_every=0)


def test_inference_failure_preserves_completed_checkpoint(tmp_path: Path) -> None:
    """A backend failure must propagate while completed rows remain resumable."""

    class FailingModel(RecordingModel):
        """Simulate failure on the second call after a valid first inference."""

        def score_long_form(self, audio: NDArray[np.float32], sr: int) -> float:
            """Return one score, then fail like an unavailable CUDA allocation."""
            if self.calls:
                message = "CUDA out of memory"
                raise RuntimeError(message)
            return super().score_long_form(audio, sr)

    for name in ["a.wav", "b.wav"]:
        sf.write(tmp_path / name, np.zeros(8000), 8000)
    manifest = tmp_path / "ko.jsonl"
    manifest.write_text(
        "\n".join(json.dumps({"audio_filepath": name}) for name in ["a.wav", "b.wav"])
    )

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        score_manifest(FailingModel(), manifest, ScoreOptions(save_every=1))

    data = json.loads((tmp_path / "KOMOS_ko.json").read_text())
    assert data["scores"] == {"a.wav": 4.25}
