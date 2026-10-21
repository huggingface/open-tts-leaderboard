"""
Script to prepare the CV3-Eval benchmark data for the HF hub.

CV3-Eval is the in-the-wild zero-shot TTS benchmark released with CosyVoice 3.
GitHub: https://github.com/FunAudioLLM/CosyVoice (see the CV3-Eval repo)
Paper:  https://arxiv.org/pdf/2505.17589

Clone the benchmark repo first; this script reads its `data/` tree in place:
```
git clone git@github.com:QwenAudio/CV3-Eval.git
python scripts/prepare_cv3_eval.py hf-audio/cv3_eval
```

The data is organized Kaldi-style: each leaf task dir holds three parallel files keyed by
utterance id (`<uttid> <value>`, split on the first space) plus a `waveform/` dir:

    text            <uttid> text to synthesize            (the WER/CER reference)
    prompt_text     <uttid> transcript of the reference clip
    prompt_wav.scp  <uttid> path to the reference clip     (relative to the CV3-Eval root)
    gt_wav.scp      <uttid> path to the ground-truth clip  (continuation tasks only)

Rows are joined on uttid, NOT on line order. Some parent dirs (e.g. `emotion_zeroshot/en`)
carry a concatenation of their children's meta files; this script reads the leaves only, so
the extra per-leaf label (emotion / style) is preserved as a column.

Each task becomes a config of the dataset; each language becomes a split within it:

  zero_shot               11 splits (de en es fr it ja ko ru zh hard_en hard_zh)   5124 rows
  cross_lingual_zeroshot   6 splits (to_{zh,en,ja,ko,hard_zh,hard_en})             2524 rows
  emotion_zeroshot         2 splits (en zh), `emotion` column: angry/happy/sad       300 rows
  subjective_zeroshot      1 split  (test)                                          194 rows
  subjective_continue      2 splits (en zh), `style` column + ground_truth_audio      90 rows

`zero_shot_clone_10speak` is NOT uploaded by default (pass it via --configs). It has no
scoring recipe upstream, and it adds no new content: its target texts are byte-identical to
`zero_shot`'s and its 10 prompt clips are the `subjective_zeroshot` ones, reused for all 2124
texts per speaker. Embedding audio per row would duplicate ~4.5 GB of identical bytes, so it
is emitted WITHOUT `prompt_audio` — join on the `speaker` column against `subjective_zeroshot`.

Usage:
```
python scripts/prepare_cv3_eval.py <hf_dataset_id> [--cv3-root CV3-Eval]
python scripts/prepare_cv3_eval.py <hf_dataset_id> --configs zero_shot emotion_zeroshot
python scripts/prepare_cv3_eval.py <hf_dataset_id> --dry-run   # parse + validate, no upload
python scripts/prepare_cv3_eval.py <hf_dataset_id>_mini --max-samples-per-leaf 2   # small test copy
```

The copy is always pushed **private**: the benchmark is redistributed audio, and its license does
not allow republishing it.
"""

import argparse
from pathlib import Path

from datasets import Audio, Dataset, Features, Value


# Several `subjective_continue` scp files point at `data/subjective_zeroshot_continue/...`, a
# directory that does not exist in the release — the audio lives under `data/subjective_continue/`.
# Applied as a prefix rewrite when the literal path is missing.
PATH_FIXUPS = {"data/subjective_zeroshot_continue/": "data/subjective_continue/"}

# Config layout. Each entry describes one Hub config:
#   dir:      task dir under `data/`
#   splits:   {split name -> [leaf dirs relative to `dir`]}. More than one leaf per split means
#             the leaves are concatenated and `label_column` records which leaf a row came from
#             (uttids repeat across leaves, so ids get prefixed with the label to stay unique).
#   label_column:      name of that per-leaf column, if any
#   labels:            {leaf dir -> label value}; defaults to the leaf dir name
#   ground_truth:      read `gt_wav.scp` into a `ground_truth_audio` Audio() column
#   prompt_audio:      embed `prompt_wav.scp` as an Audio() column (default True)
#   speaker_from_dir:  take a `speaker` column from the leaf's PARENT dir name (10speak only)
CONFIGS = {
    "zero_shot": {
        "dir": "zero_shot",
        "splits": {lang: [lang] for lang in
                   ["en", "zh", "hard_en", "hard_zh", "ja", "ko", "de", "es", "fr", "it", "ru"]},
    },
    "cross_lingual_zeroshot": {
        "dir": "cross_lingual_zeroshot",
        "splits": {t: [t] for t in ["to_en", "to_zh", "to_hard_en", "to_hard_zh", "to_ja", "to_ko"]},
    },
    "emotion_zeroshot": {
        "dir": "emotion_zeroshot",
        "splits": {"en": ["en/angry", "en/happy", "en/sad"], "zh": ["zh/angry", "zh/happy", "zh/sad"]},
        "label_column": "emotion",
        "labels": {f"{lang}/{emo}": emo for lang in ("en", "zh") for emo in ("angry", "happy", "sad")},
    },
    "subjective_zeroshot": {
        "dir": "subjective_zeroshot",
        "splits": {"test": ["."]},
    },
    "subjective_continue": {
        "dir": "subjective_continue",
        # `rhyme` names its language dirs en_rhyme / zh_rhyme; the rest use plain en / zh.
        "splits": {
            "en": ["emotion/en", "rhyme/en_rhyme", "speed/en", "volume/en"],
            "zh": ["emotion/zh", "rhyme/zh_rhyme", "speed/zh", "volume/zh"],
        },
        "label_column": "style",
        "labels": {f"{style}/{lang}": style
                   for style in ("emotion", "speed", "volume") for lang in ("en", "zh")}
        | {f"rhyme/{lang}_rhyme": "rhyme" for lang in ("en", "zh")},
        "ground_truth": True,
    },
    # Opt-in. Same texts as `zero_shot`, one fixed prompt per speaker → no prompt_audio (see docstring).
    "zero_shot_clone_10speak": {
        "dir": "zero_shot_clone_10speak",
        "splits": None,        # filled in by _expand_10speak(): {lang: [<speaker>/<lang>, ...]}
        "label_column": "speaker",
        "prompt_audio": False,
        "speaker_from_dir": True,
    },
}

DEFAULT_CONFIGS = [c for c in CONFIGS if c != "zero_shot_clone_10speak"]


def _expand_10speak(task_dir: Path) -> dict[str, list[str]]:
    """Build {language -> [<speaker>/<language>, ...]} from the speaker dirs present on disk."""
    speakers = sorted(p.name for p in task_dir.iterdir() if p.is_dir())
    splits: dict[str, list[str]] = {}
    for speaker in speakers:
        for lang_dir in sorted(p.name for p in (task_dir / speaker).iterdir() if p.is_dir()):
            splits.setdefault(lang_dir, []).append(f"{speaker}/{lang_dir}")
    return splits


def read_keyed_file(path: Path) -> dict[str, str]:
    """Read a `<uttid> <value>` file into {uttid: value}. Values may contain spaces."""
    entries = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            uttid, _, value = line.partition(" ")
            if not value:
                raise ValueError(f"{path}:{lineno} expected '<uttid> <value>', got {line!r}")
            if uttid in entries:
                raise ValueError(f"{path}:{lineno} duplicate uttid {uttid!r}")
            entries[uttid] = value
    if not entries:
        raise ValueError(f"{path} is empty")
    return entries


def resolve_audio(rel_path: str, cv3_root: Path) -> str:
    """Resolve an scp path (relative to the CV3-Eval root) to an existing file, applying fixups."""
    candidates = [rel_path]
    for stale, current in PATH_FIXUPS.items():
        if rel_path.startswith(stale):
            candidates.append(current + rel_path[len(stale):])
    for candidate in candidates:
        resolved = cv3_root / candidate
        if resolved.is_file():
            return str(resolved)
    raise FileNotFoundError(f"audio file not found for {rel_path!r} (tried: {candidates})")


def parse_leaf(leaf_dir: Path, cv3_root: Path, *, with_prompt_audio: bool, with_ground_truth: bool) -> list[dict]:
    """Parse one leaf task dir into records, joining `text` / `prompt_text` / scp files on uttid."""
    texts = read_keyed_file(leaf_dir / "text")
    prompt_texts = read_keyed_file(leaf_dir / "prompt_text")
    prompt_wavs = read_keyed_file(leaf_dir / "prompt_wav.scp") if with_prompt_audio else {}
    gt_wavs = read_keyed_file(leaf_dir / "gt_wav.scp") if with_ground_truth else {}

    records = []
    for uttid, text in texts.items():
        for name, table in (("prompt_text", prompt_texts), ("prompt_wav.scp", prompt_wavs),
                            ("gt_wav.scp", gt_wavs)):
            if table and uttid not in table:
                raise KeyError(f"{leaf_dir}: uttid {uttid!r} present in `text` but missing from `{name}`")

        record = {"id": uttid, "prompt_text": prompt_texts[uttid], "text": text}
        if with_prompt_audio:
            record["prompt_audio"] = resolve_audio(prompt_wavs[uttid], cv3_root)
        if with_ground_truth:
            record["ground_truth_audio"] = resolve_audio(gt_wavs[uttid], cv3_root)
        records.append(record)
    return records


def build_split(config: dict, leaves: list[str], task_dir: Path, cv3_root: Path,
                max_per_leaf: int | None = None) -> Dataset:
    """Parse every leaf of a split, tag rows with the per-leaf label, and build the Dataset.

    `max_per_leaf` caps rows per LEAF, not per split, so a truncated `emotion_zeroshot` still
    carries all three emotions (and a truncated 10speak split all ten speakers).
    """
    with_prompt_audio = config.get("prompt_audio", True)
    with_ground_truth = config.get("ground_truth", False)
    label_column = config.get("label_column")
    labels = config.get("labels", {})

    records = []
    for leaf in leaves:
        leaf_dir = task_dir / leaf
        rows = parse_leaf(leaf_dir, cv3_root, with_prompt_audio=with_prompt_audio,
                          with_ground_truth=with_ground_truth)
        if max_per_leaf is not None:
            rows = rows[:max_per_leaf]
        if label_column:
            # `<speaker>/<lang>` → speaker; otherwise the configured label (or the leaf name).
            label = leaf.split("/")[0] if config.get("speaker_from_dir") else labels.get(leaf, Path(leaf).name)
            for row in rows:
                row[label_column] = label
                # uttids restart at uttid_1 in every leaf — prefix so ids stay unique in the split.
                row["id"] = f"{label}_{row['id']}"
        records.extend(rows)

    feature_dict = {"id": Value("string")}
    if label_column:
        feature_dict[label_column] = Value("string")
    feature_dict["prompt_text"] = Value("string")
    if with_prompt_audio:
        feature_dict["prompt_audio"] = Audio()
    feature_dict["text"] = Value("string")
    if with_ground_truth:
        feature_dict["ground_truth_audio"] = Audio()

    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate ids after prefixing in {task_dir} ({leaves})")

    columns = {key: [r[key] for r in records] for key in feature_dict}
    return Dataset.from_dict(columns, features=Features(feature_dict))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="HF dataset id to push to, e.g. 'hf-audio/cv3_eval'.")
    parser.add_argument(
        "--cv3-root",
        default="CV3-Eval",
        help="Path to the CV3-Eval checkout (the dir containing `data/`; scp paths are relative to it).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        choices=list(CONFIGS),
        help=f"Configs to upload (default: {' '.join(DEFAULT_CONFIGS)}).",
    )
    parser.add_argument(
        "--max-samples-per-leaf",
        type=int,
        default=None,
        help="Keep only the first N rows of each leaf task dir — for pushing a small test copy "
             "to a scratch repo id. Capping per leaf keeps every emotion/style/speaker represented.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate only; do not upload.")
    args = parser.parse_args()

    cv3_root = Path(args.cv3_root)
    data_dir = cv3_root / "data"
    if not data_dir.is_dir():
        raise SystemExit(f"{data_dir} not found — pass --cv3-root pointing at the CV3-Eval checkout.")

    total = 0
    for config_name in args.configs:
        config = CONFIGS[config_name]
        task_dir = data_dir / config["dir"]
        if not task_dir.is_dir():
            print(f"Skipping config '{config_name}': {task_dir} not found.")
            continue

        splits = config["splits"] or _expand_10speak(task_dir)
        for split, leaves in splits.items():
            missing = [leaf for leaf in leaves if not (task_dir / leaf / "text").is_file()]
            if missing:
                print(f"Skipping '{config_name}/{split}': no `text` file in {missing}.")
                continue

            print(f"Parsing '{config_name}/{split}' from {len(leaves)} leaf dir(s) ...")
            dataset = build_split(config, leaves, task_dir, cv3_root, args.max_samples_per_leaf)
            print(f"  {len(dataset)} examples, columns: {', '.join(dataset.column_names)}")
            total += len(dataset)

            if args.dry_run:
                continue
            print(f"Pushing config '{config_name}' split '{split}' to {args.repo_id} ...")
            dataset.push_to_hub(
                args.repo_id,
                config_name=config_name,
                # Always private — the license does not permit redistributing this benchmark.
                private=True,
                split=split,
            )
            print(f"  done ({config_name}/{split}).")

    print(f"\n{total} examples across {len(args.configs)} config(s).")
    if args.dry_run:
        print("Dry run — nothing uploaded.")
    else:
        print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
