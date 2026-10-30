import io

from datasets import Audio, load_from_disk
import numpy as np
import pytest
import soundfile as sf

from scripts.prepare_neyshekar_eval import build_dataset, check_disjoint, pair_records, text_key


def clip(ident, text, duration=5.0, speaker="s1"):
    return {"id": ident, "text": text, "duration": duration, "speaker_id": speaker, "audio": {"bytes": b"wav", "path": None}}


def test_deterministic_same_speaker_pairs_without_prompt_text_leakage():
    rows = [
        clip(1, "سلام دنیا"), clip(2, "من مي\u200cروم", 4), clip(3, "من میروم", 6),
        clip(4, "متن دیگر", 7), clip(5, "سلام دنيا!", 4, "s2"),
        clip(6, "درود بر شما", 5, "s2"), clip(7, "تنها جمله", 5, "s3"),
        clip(8, "خیلی کوتاه", 0.5), clip(9, "", 5), clip(10, "بیش از حد", 31),
    ]
    records, report = pair_records(rows, "test")
    assert records == pair_records(list(reversed(rows)), "test")[0]
    assert [r["source_id"] for r in records] == [2, 4]
    assert all(r["prompt_id"] == 1 and r["speaker_id"] == "s1" for r in records)
    assert report["duplicate_target_text"] == 1
    assert report["invalid_or_out_of_duration_range"] == 3
    assert report["speakers_without_prompt_or_distinct_target"] == 1
    assert all(text_key(r["text"]) != text_key(r["prompt_text"]) for r in records)
    assert all(r["text"] == next(row["text"] for row in rows if row["id"] == r["source_id"]) for r in records)


def test_split_boundaries():
    check_disjoint({"test": [clip(1, "الف")], "validation": [clip(2, "ب", speaker="s2")]})
    with pytest.raises(ValueError, match="overlap"):
        check_disjoint({"test": [clip(1, "الف")], "validation": [clip(2, "ب")]})
    with pytest.raises(ValueError, match="overlap"):
        check_disjoint({"test": [clip(1, "الف")], "validation": [clip(1, "ب", speaker="s2")]})
    with pytest.raises(ValueError, match="Only predefined"):
        pair_records([], "train")


def test_bad_sources_fail():
    with pytest.raises(ValueError, match="Duplicate source"):
        pair_records([clip(1, "الف"), clip(1, "ب")], "test")
    with pytest.raises(ValueError, match="No usable"):
        pair_records([clip(1, "الف")], "test")
    # No 3–10 second prompt, even though both clips are usable as target texts.
    with pytest.raises(ValueError, match="No usable"):
        pair_records([clip(1, "الف", 2), clip(2, "ب", 20)], "test")


def test_schema_and_wav_bytes_survive_disk_roundtrip(tmp_path):
    wav = io.BytesIO()
    sf.write(wav, np.zeros(80_000), 16_000, format="WAV", subtype="PCM_16")
    rows = [clip(1, "متن راهنما"), clip(2, "متن هدف", 6)]
    rows[0]["audio"]["bytes"] = wav.getvalue()
    records, _ = pair_records(rows, "validation")
    dataset = build_dataset(records)
    dataset.save_to_disk(str(tmp_path / "dataset"))
    restored = load_from_disk(str(tmp_path / "dataset")).cast_column("prompt_audio", Audio(decode=False))
    assert restored[0]["prompt_audio"]["bytes"] == wav.getvalue()
    assert restored[0]["source_split"] == "validation"
    assert restored[0]["text"] == "متن هدف"
    assert restored[0]["prompt_text"] == "متن راهنما"


def test_invalid_selected_audio_fails():
    wav = io.BytesIO()
    sf.write(wav, np.zeros(80_000), 8000, format="WAV")
    rows = [clip(1, "متن راهنما"), clip(2, "متن هدف", 6)]
    rows[0]["audio"]["bytes"] = wav.getvalue()
    records, _ = pair_records(rows, "test")
    with pytest.raises(ValueError, match="expected mono 16 kHz"):
        build_dataset(records)
