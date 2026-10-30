from unittest.mock import patch
import json
import sys

from huggingface_hub.errors import EntryNotFoundError
import pytest

from scripts.open_results_pr import TARGETS, collect_values, fetch_csv, header_languages, select_manifests
from scripts import open_results_pr


def test_validation_and_clone_modes_do_not_mix(tmp_path):
    prefix = "MODEL_test-model_DATASET_user-neyshekar_eval_"
    names = ["tts_fa", "tts_fa_voice_clone", "tts_validation_fa", "tts_validation_fa_voice_clone"]
    for name in names:
        (tmp_path / f"{prefix}{name}.jsonl").touch()
    for target, suffix in [("neyshekar", "tts_fa"), ("neyshekar_voice_clone", "tts_fa_voice_clone")]:
        selected, _ = select_manifests(str(tmp_path), "test-model", TARGETS[target], "")
        assert selected == {prefix + suffix: "fa"}


@pytest.mark.parametrize("target", ["neyshekar", "neyshekar_voice_clone"])
def test_first_results_file_can_be_created(target):
    with patch("scripts.open_results_pr.hf_hub_download", side_effect=EntryNotFoundError("missing")):
        published = fetch_csv(TARGETS[target].filename, None)
    assert header_languages(published.header) == ["fa"]
    stem = "MODEL_test-model_DATASET_user-neyshekar_eval_tts_fa"
    values, unknown = collect_values(TARGETS[target], ["fa"], {stem: "fa"}, {
        "user-neyshekar_eval_tts_fa": {"wer": 12.5, "rtfx": 10, "sim": None},
    }, {})
    assert values == {"fa WER": "12.5", "fa RTFx": "10"}
    assert not unknown


def test_other_missing_csvs_still_fail():
    with patch("scripts.open_results_pr.hf_hub_download", side_effect=EntryNotFoundError("missing")):
        with pytest.raises(EntryNotFoundError):
            fetch_csv("cv3.csv", None)


def test_publisher_dry_run_scores_test_only(tmp_path, monkeypatch):
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    for config, prediction in [("tts", "من میروم و دوازده کتاب دارم"), ("tts_validation", "خطای عمدی")]:
        stem = f"MODEL_test-model_DATASET_user-neyshekar_eval_{config}_fa_voice_clone"
        row = {"text": "من مي\u200cروم و ۱۲ كتاب دارم", "pred_text": prediction, "time": 1, "duration": 5}
        (model_dir / f"{stem}.jsonl").write_text(json.dumps(row), encoding="utf-8")
    out = tmp_path / "csvs"
    monkeypatch.setattr(sys, "argv", [
        "open_results_pr.py", "--model_id", "test/model", "--local_dir", str(tmp_path),
        "--targets", "neyshekar_voice_clone", "--skip_sync", "--skip_model_info", "--out_dir", str(out),
    ])
    with patch("scripts.open_results_pr.hf_hub_download", side_effect=EntryNotFoundError("missing")):
        open_results_pr.main()
    assert (out / "neyshekar_voice_clone.csv").read_text() == "model,fa WER,fa RTFx,fa SIM\ntest/model,0,5,\n"
