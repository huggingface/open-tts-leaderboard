import json

import pytest

from normalizer.data_utils import ml_normalizer
from normalizer.eval_utils import score_results
from normalizer.persian import PersianTextNormalizer


@pytest.mark.parametrize("raw,expected", [
    ("سَلام، كِتاب يار!", "سلام کتاب یار"),
    ("عَلِى", "علی"),
    ("مي\u200cروم", "می روم"),
    ("سلام\u200f دنیا\u200e", "سلام دنیا"),
    ("ســلام\u200d", "سلام"),
    ("خانهٔ دوست", "خانه دوست"),
    ("خانۀ دوست", "خانه دوست"),
    ("مسئول رؤیا آسمان", "مسئول رؤیا آسمان"),
    ("۱۲۳۴", "هزار و دویست و سی و چهار"),
    ("١٢٣٤", "هزار و دویست و سی و چهار"),
    ("1,234,567", "یک میلیون و دویست و سی و چهار هزار و پانصد و شصت و هفت"),
    ("۱٬۲۳۴", "هزار و دویست و سی و چهار"),
    ("۱۲٫۵", "دوازده ممیز پنج"),
    ("3.05", "سه ممیز صفر پنج"),
    ("−۱۲", "منفی دوازده"),
    ("-0.5", "منفی صفر ممیز پنج"),
    ("[laugh] سلام (دنیا)", "سلام دنیا"),
    ("OpenAI 42", "openai چهل و دو"),
])
def test_persian_normalizer(raw, expected):
    normalize = PersianTextNormalizer()
    assert normalize(raw) == expected
    assert ml_normalizer(raw, lang="fa") == expected
    assert normalize(expected) == expected


def test_meaningful_spelling_distinctions_survive():
    normalize = PersianTextNormalizer()
    for left, right in [("آب", "اب"), ("مؤثر", "موثر"), ("قلب", "کلب"), ("می روم", "می روی"), ("می روم", "میرم")]:
        assert normalize(left) != normalize(right)


def test_existing_language_normalizers():
    assert ml_normalizer("Hello, World!", lang="en") == "hello world"
    assert ml_normalizer("كِتاب", lang="ar") == "كِتاب"
    assert ml_normalizer("12", lang="de") == "zwölf"


def test_manifest_scoring_uses_persian_wer(tmp_path):
    name = "MODEL_test-model_DATASET_user-neyshekar_eval_tts_fa.jsonl"
    rows = [
        {"text": "من مي\u200cروم و ۱۲ كتاب دارم.", "pred_text": "من میروم و دوازده کتاب دارم", "time": 1, "duration": 5},
        {"text": "سلام دنیا", "pred_text": "سلام دوست", "time": 1, "duration": 5},
    ]
    (tmp_path / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    _, results = score_results(str(tmp_path), language="fa", csv_only=True)
    result = next(iter(results.values()))
    assert result["metric"] == "WER"
    assert result["wer"] > 0
    assert result["sub"] == 1
    assert result["del"] == result["ins"] == 0
    assert result["rtfx"] == 5
    assert result["utmos"] is None
