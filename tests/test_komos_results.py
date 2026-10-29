"""Exercise Korean sidecar aggregation through the actual CSV publication helpers."""

import json
from pathlib import Path

import pytest

from normalizer.eval_utils import read_komos, score_results
from open_results_pr import (
    TARGETS,
    Published,
    add_komos_columns,
    collect_values,
    header_languages,
    recompute_averages,
    render,
    upsert,
)


@pytest.fixture
def results(tmp_path: Path) -> Path:
    """Provide Korean ASR/generation data and score provenance without model weights."""
    path = tmp_path / "MODEL_test-model_DATASET_hf-audio_cv3_eval_zero_shot_ko.jsonl"
    rows = [
        {
            "audio_filepath": name,
            "text": "안녕하세요",
            "pred_text": "안녕하세요",
            "duration": 1.0,
            "time": 0.5,
        }
        for name in ["a.wav", "b.wav", "missing.wav"]
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    (tmp_path / f"KOMOS_{path.stem}.json").write_text(
        json.dumps(
            {
                "komos_model": {
                    "model_id": "upskyy/komos",
                    "revision": "pinned",
                    "method": "score_long_form",
                },
                "scores": {"a.wav": 3.5, "b.wav": 4.5, "other.wav": 1.0},
            }
        )
    )
    return path


def test_sidecar_lookup_handles_sim_forks(results: Path) -> None:
    """SIM-fork manifests use their canonical generation manifest's sidecar."""
    fork = results.with_stem(results.stem + "_wavlm_seed_tts")
    rows = [json.loads(line) for line in results.read_text().splitlines()]

    value = read_komos(str(fork), rows)

    assert value.score == 4.0 and value.count == 2
    assert json.loads(value.model)["revision"] == "pinned"


@pytest.mark.parametrize("score", [None, "4.0", True, float("nan"), 9])
def test_sidecar_ignores_invalid_scores(
    results: Path, score: float | str | None
) -> None:
    """Only finite MOS values in the calibrated range contribute to the mean."""
    sidecar = results.parent / f"KOMOS_{results.stem}.json"
    data = json.loads(sidecar.read_text())
    data["scores"]["a.wav"] = score
    sidecar.write_text(json.dumps(data))

    value = read_komos(
        str(results), [{"audio_filepath": "a.wav"}, {"audio_filepath": "b.wav"}]
    )

    assert value.score == 4.5 and value.count == 1


def test_missing_sidecar_has_no_metric(results: Path) -> None:
    """Old runs without KOMOS keep their existing results."""
    value = read_komos(str(results.with_stem("other")), [])

    assert value.score is None and value.count == 0


def test_korean_csv_publication_preserves_existing_metrics_and_averages(
    results: Path,
) -> None:
    """Publish score, coverage and provenance without changing any ranking inputs."""
    _, scored = score_results(str(results.parent), language="ko", csv_only=True)
    result = next(iter(scored.values()))
    published = Published(
        "model,en WER,ko WER,en RTFx,ko RTFx,en UTMOS,avg WER\n"
        "test/model,2,3,4,5,4.2,2.5\n"
    )
    languages = header_languages(published.header)
    dataset = results.stem.split("_DATASET_", 1)[1]

    values, unknown = collect_values(
        TARGETS["cv3"], languages, {results.stem: "ko"}, {dataset: result}, {}
    )
    extended = add_komos_columns(published, languages)
    _, index, row = upsert(extended, "test/model", values, {})
    recompute_averages(extended.header, row, languages)
    final = Published(render(extended, index, row, False))

    record = dict(zip(final.header, final.rows[0]))
    assert not unknown
    assert record["ko KOMOS"] == "4" and record["ko KOMOS samples"] == "2"
    assert json.loads(record["ko KOMOS model"])["revision"] == "pinned"
    assert record["en UTMOS"] == "4.2" and record["en WER"] == "2"
    assert "avg KOMOS" not in final.header and "en KOMOS" not in final.header
    assert add_komos_columns(final, languages).text == final.text


def test_non_korean_scoring_does_not_read_komos(results: Path) -> None:
    """A sidecar beside a non-Korean manifest cannot add a Korean metric."""
    _, scored = score_results(str(results.parent), language="en", csv_only=True)

    assert next(iter(scored.values()))["komos"] is None


def test_english_utmos_remains_unchanged(results: Path) -> None:
    """English retains UTMOS without publishing Korean diagnostic columns."""
    path = results.with_stem(results.stem.removesuffix("_ko") + "_en")
    rows = [
        {
            "audio_filepath": "en.wav",
            "text": "hello",
            "pred_text": "hello",
            "duration": 1,
            "time": 0.5,
        }
    ]
    path.write_text(json.dumps(rows[0]))
    path.with_name(f"UTMOS_{path.stem}.json").write_text(
        json.dumps({"scores": {"en.wav": 4.125}})
    )

    _, scored = score_results(
        str(path.parent), language="en", manifests=[path.name], csv_only=True
    )
    value = next(iter(scored.values()))

    assert value["utmos"] == 4.125 and value["komos"] is None


def test_publication_cli_dry_run_uses_komos_sidecar(
    results: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run publication with a local stand-in for the private published CSV download."""
    import sys
    import open_results_pr

    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    for path in [results, results.parent / f"KOMOS_{results.stem}.json"]:
        path.rename(model_dir / path.name)
    template = tmp_path / "cv3-template.csv"
    template.write_text(
        "model,en WER,ko WER,en RTFx,ko RTFx,en UTMOS,avg WER\n"
        "other/model,2,3,4,5,4.2,2.5\n"
    )
    # The official results repository is private; stub only the external download boundary.
    monkeypatch.setattr(
        open_results_pr, "hf_hub_download", lambda *args, **kwargs: str(template)
    )
    out = tmp_path / "published"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "open_results_pr.py",
            "--model_id",
            "test/model",
            "--local_dir",
            str(tmp_path),
            "--skip_sync",
            "--skip_model_info",
            "--targets",
            "cv3",
            "--out_dir",
            str(out),
        ],
    )

    open_results_pr.main()

    published = Published((out / "cv3.csv").read_text())
    row = dict(zip(published.header, published.rows[published.index_of("test/model")]))
    assert row["ko KOMOS"] == "4" and row["ko KOMOS samples"] == "2"
    assert published.rows[0][:7] == ["other/model", "2", "3", "4", "5", "4.2", "2.5"]
    assert "avg KOMOS" not in published.header
