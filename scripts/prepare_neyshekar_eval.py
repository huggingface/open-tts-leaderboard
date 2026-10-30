"""Prepare Persian Neyshekar evaluation data in the leaderboard's prompt/text schema.

Only the predefined test/validation splits are downloaded (never train). Test becomes
`tts/fa`; validation becomes `tts_validation/fa`, reserved for development. Pair clips
within each split because the source splits are speaker-disjoint.

Examples:
    python scripts/prepare_neyshekar_eval.py --dry-run --output-dir /tmp/neyshekar_eval
    python scripts/prepare_neyshekar_eval.py YOUR_USERNAME/neyshekar_eval
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import sys

from datasets import Audio, Dataset, Features, Value, load_dataset
from huggingface_hub import DatasetCard, HfApi, snapshot_download
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from normalizer.persian import PersianTextNormalizer  # noqa: E402


SOURCE_REPO_ID = "shekar-ai/Neyshekar"
# V6.0: 2,149 test and 1,886 validation clips, with stable speaker IDs.
SOURCE_REVISION = "30696ada746a66bb3047c984744b5ecfd60fe4e9"
CONFIGS = {"test": "tts", "validation": "tts_validation"}
_NORMALIZE = PersianTextNormalizer()


def text_key(text):
    """Detect repeated prompt/target sentences despite punctuation and half-space variants."""
    return _NORMALIZE(text).replace(" ", "")


def check_disjoint(splits):
    """Fail rather than leak speakers or recordings between development and final test."""
    seen_speakers, seen_ids = set(), set()
    for name, rows in splits.items():
        speakers = {row["speaker_id"] for row in rows}
        ids = {row["id"] for row in rows}
        if seen_speakers & speakers or seen_ids & ids:
            raise ValueError(f"Speaker or recording overlap in source split {name!r}")
        seen_speakers.update(speakers)
        seen_ids.update(ids)


def pair_records(rows, source_split):
    """One fixed 3–10 s prompt per speaker; all other eligible, distinct texts are targets.

    Choose the prompt closest to 5 s (tie: lowest numeric source ID), without using ASR
    or model scores. Targets must be 1–30 s. Exclude every selected prompt's ID and
    normalized text from ALL targets in this split. Deduplicate target texts per speaker.
    Speakers without a usable prompt and a distinct target are omitted and reported.
    """
    if source_split not in CONFIGS:
        raise ValueError("Only predefined test and validation splits are allowed")
    counts = Counter()
    grouped = defaultdict(list)
    ids = set()
    for row in rows:
        counts["source_rows"] += 1
        if row["id"] in ids:
            raise ValueError(f"Duplicate source id {row['id']}")
        ids.add(row["id"])
        duration = row["duration"]
        if (not isinstance(row["text"], str) or not row["text"].strip()
                or not row["speaker_id"] or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or not 1 <= duration <= 30):
            counts["invalid_or_out_of_duration_range"] += 1
            continue
        key = text_key(row["text"])
        if not key:
            counts["empty_normalized_text"] += 1
            continue
        grouped[row["speaker_id"]].append((row, key))

    prompts = {}
    for speaker, clips in sorted(grouped.items()):
        candidates = [r for r, _ in clips if 3 <= r["duration"] <= 10]
        if not candidates or len({key for _, key in clips}) < 2:
            counts["speakers_without_prompt_or_distinct_target"] += 1
            continue
        prompts[speaker] = min(candidates, key=lambda r: (abs(r["duration"] - 5), r["id"]))

    prompt_ids = {p["id"] for p in prompts.values()}
    prompt_texts = {text_key(p["text"]) for p in prompts.values()}
    records = []
    for speaker, clips in sorted(grouped.items()):
        if speaker not in prompts:
            counts["rows_without_prompt"] += len(clips)
            continue
        prompt = prompts[speaker]
        seen_texts = set()
        for row, key in sorted(clips, key=lambda item: item[0]["id"]):
            if row["id"] in prompt_ids or key in prompt_texts:
                counts["prompt_id_or_text_excluded"] += 1
                continue
            if key in seen_texts:
                counts["duplicate_target_text"] += 1
                continue
            seen_texts.add(key)
            records.append({
                "id": f"{source_split}_{row['id']}",
                "text": row["text"],
                "prompt_text": prompt["text"],
                "prompt_audio": prompt["audio"],
                "speaker_id": speaker,
                "source_id": row["id"],
                "prompt_id": prompt["id"],
                "source_split": source_split,
            })
    if not records:
        raise ValueError(f"No usable prompt/target pairs in {source_split}")
    counts["output_rows"] = len(records)
    counts["output_speakers"] = len({r["speaker_id"] for r in records})
    return records, dict(counts)


def build_dataset(records):
    """Embed source WAV bytes unchanged and validate each distinct selected prompt."""
    for prompt_id in sorted({r["prompt_id"] for r in records}):
        audio = next(r["prompt_audio"] for r in records if r["prompt_id"] == prompt_id)
        source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio["path"]
        info = sf.info(source)
        if info.channels != 1 or info.samplerate != 16_000 or not 3 <= info.duration <= 10:
            raise ValueError(f"Prompt {prompt_id}: expected mono 16 kHz audio of 3–10 s, got {info}")
    features = Features({
        "id": Value("string"), "text": Value("string"), "prompt_text": Value("string"),
        "prompt_audio": Audio(sampling_rate=16_000), "speaker_id": Value("string"),
        "source_id": Value("int64"), "prompt_id": Value("int64"), "source_split": Value("string"),
    })
    return Dataset.from_dict({k: [r[k] for r in records] for k in features}, features=features)


def dataset_card(revision, reports):
    return f"""# Neyshekar Persian TTS evaluation

Reformatted from [{SOURCE_REPO_ID}](https://huggingface.co/datasets/{SOURCE_REPO_ID}),
revision `{revision}`, under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
Source authors: Ahmad Amirivojdan, Farzad Nadiri, Abolfazl Alizadeh and Shaghayegh Yaraghi.
See [Neyshekar (2026)](https://arxiv.org/abs/2609.14542). Do not attempt to identify speakers.

When requested, `tts/fa` uses the predefined **test** split and `tts_validation/fa` uses
**validation**, for development only. No training recordings are used. Requested source
splits are checked for disjoint speaker IDs. This does not establish disjointness from any
evaluated model's training data; model authors must disclose Neyshekar training overlap.

Within each split, select one 3–10 second prompt per speaker, closest to 5 seconds,
ties broken by ascending source ID. Targets are the remaining 1–30 second recordings'
transcripts, deduplicated per speaker. All selected prompt IDs and normalized prompt
texts are excluded from targets across that split. Speakers without a prompt and a
distinct target are omitted. Pairing uses no ASR/TTS quality scores. Rows are ordered
by speaker ID, then numeric target ID. Speaker distribution follows the source and
is not balanced. This is an automatically constructed evaluation set, not a manually
audited TTS benchmark.

**Changes:** prompt/target pairing, filtering, and conversion to the leaderboard's
`id`, `text`, `prompt_text`, `prompt_audio` schema, with source/prompt IDs and split
provenance. Original transcripts and prompt WAV bytes are unchanged. Target recordings
are not copied into the evaluation set. Preparation script:
[`prepare_neyshekar_eval.py`](https://github.com/huggingface/open-tts-leaderboard/blob/main/scripts/prepare_neyshekar_eval.py).

Scoring: Qwen3-ASR (`fa`), Persian normalized WER, RTFx, and WavLM SIM for cloning.
UTMOS22 is English-only and is not reported here. Normalization is applied only at
scoring time, never to the synthesis input. See the repository's Persian evaluation docs.

## Preparation report

The SHA-256 covers ordered `[target_id, prompt_id]` pairs, encoded as compact JSON.

```json
{json.dumps(reports, ensure_ascii=False, indent=2)}
```
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", nargs="?", help="HF destination, e.g. YOUR_USERNAME/neyshekar_eval")
    parser.add_argument("--revision", default=SOURCE_REVISION, help="Source revision (default: pinned V6.0 commit)")
    parser.add_argument("--splits", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--dry-run", action="store_true", help="Prepare and validate without uploading")
    parser.add_argument("--output-dir", type=Path, help="Also save datasets, card and provenance locally")
    parser.add_argument("--private", action="store_true", help="Create a private destination (default: public)")
    args = parser.parse_args()
    if not args.dry_run and not args.repo_id:
        parser.error("repo_id is required unless --dry-run is passed")
    # load_dataset(repo, split=...) can still prepare/download every split. Restrict
    # the actual source files so the 10 GB training set is never downloaded.
    revision = HfApi().dataset_info(SOURCE_REPO_ID, revision=args.revision).sha
    source_dir = Path(snapshot_download(
        SOURCE_REPO_ID, repo_type="dataset", revision=revision,
        allow_patterns=[f"data/{name}-*.parquet" for name in args.splits],
    ))
    splits = {
        name: load_dataset("parquet", data_files={name: sorted(str(p) for p in (source_dir / "data").glob(f"{name}-*.parquet"))},
                           split=name).cast_column("audio", Audio(decode=False))
        for name in dict.fromkeys(args.splits)
    }
    check_disjoint(splits)
    prepared, reports = {}, {}
    for source_split, source in splits.items():
        records, report = pair_records(source, source_split)
        config = CONFIGS[source_split]
        pairs = [[r["source_id"], r["prompt_id"]] for r in records]
        report["pairs_sha256"] = hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()
        prepared[config] = build_dataset(records)
        reports[config] = {"source_split": source_split, **report}
        print(f"{config}/fa: {json.dumps(report)}")
    card_text = dataset_card(revision, reports)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for config, dataset in prepared.items():
            dataset.save_to_disk(str(args.output_dir / config / "fa"))
        (args.output_dir / "README.md").write_text(card_text, encoding="utf-8")
        (args.output_dir / "provenance.json").write_text(json.dumps({
            "source_repo": SOURCE_REPO_ID, "source_revision": revision, "configs": reports,
        }, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print("Dry run — nothing uploaded.")
        return
    for config, dataset in prepared.items():
        dataset.push_to_hub(args.repo_id, config_name=config, split="fa", private=args.private)
    card = DatasetCard.load(args.repo_id, repo_type="dataset")
    card.data.license = "cc0-1.0"
    card.data.language = ["fa"]
    card.data.task_categories = ["text-to-speech"]
    card.data.source_datasets = [SOURCE_REPO_ID]
    card.text = card_text
    card.push_to_hub(args.repo_id, repo_type="dataset", commit_message="Document Neyshekar Persian evaluation provenance")


if __name__ == "__main__":
    main()
