"""
Script to prepare ZTTS2-Eval Mini, the coreset panels of ZTTS2-Eval, for the HF hub.

Source: a ZTTS2-Eval checkout, https://github.com/Zyphra/ZTTS2-Eval. Its prompt wavs come from the
private `Zyphra/ZTTS2-Eval` audio repo, fetched into the checkout by ZTTS2-Eval's
`python scripts/prep/download_data.py`.

Layout (Kaldi-style, as CV3-Eval; scp paths are relative to the checkout root):
  - data/clean_mini/<lang>/{text, prompt_text, prompt_wav.scp, waveform/}         FLEURS-R, 35 languages
  - data/in-the-wild_mini/<lang>/{text, prompt_text, prompt_wav.scp, waveform/}   VoxBlink2, 20 languages
  - data/{clean_mini,in-the-wild_mini}/panel.json   {"langs": {<lang>: {<uttid>: weight}}, ...}

Each panel becomes a config (`clean`, `in_the_wild`), each language a split named by its ZTTS2-Eval code. Every
row carries its panel weight (`weight` column), so ZTTS2-Eval's weighted panel estimate can be reproduced from the
dataset; the leaderboard's scoring does not use it.
The In-The-Wild held-out target recordings are not included.

Pushed **public** by default: FLEURS-R is CC-BY-4.0 and VoxBlink2 CC-BY-NC-SA-4.0, both of which allow
redistribution with attribution (see CARD_TEXT).

Prerequisite — get the checkout and its prompt wavs:
```
# 1. Clone the benchmark repo somewhere OUTSIDE this one (path becomes --ztts2-root).
git clone https://github.com/Zyphra/ZTTS2-Eval.git

# 2. Fetch the prompt wavs (gated `Zyphra/ZTTS2-Eval` audio repo — accept its terms on the Hub
#    and `huggingface-cli login` first). This populates data/clean_mini/ and data/in-the-wild_mini/.
cd ZTTS2-Eval && python scripts/prep/download_data.py && cd -
```

Usage:
```
python scripts/prepare_ztts2_eval_mini.py [repo_id] --ztts2-root <PATH>/ZTTS2-Eval [--configs clean in_the_wild] [--dry-run] [--private]
```
"""

import argparse
import json
from pathlib import Path

from datasets import Audio, Dataset, Features, Value
from huggingface_hub import DatasetCard

from prepare_cv3_eval import parse_leaf


# Split names of the two panels, in ZTTS2-Eval's order (utils/benchmark.py CLEAN_LANGS / ITW_LANGS).
PANEL_LANGS = {
    "clean": ("en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru", "pt", "ar", "hi", "tr", "pl", "id", "th",
              "fil", "nl", "vi", "cs", "sv", "da", "fi", "el", "hu", "ro", "uk", "nb", "he", "bg", "hr", "sk",
              "ca", "ms", "fa"),
    "in_the_wild": ("en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru", "pt", "ar", "hi", "id", "tr", "tl",
                    "pl", "th", "nl", "uk", "vi"),
}

DEFAULT_REPO_ID = "hf-audio/ztts2_eval_mini"
# Config -> panel dir under the checkout's data/.
PANEL_DIRS = {"clean": "clean_mini", "in_the_wild": "in-the-wild_mini"}
FEATURES = Features({
    "id": Value("string"),
    "prompt_text": Value("string"),
    "prompt_audio": Audio(),
    "text": Value("string"),
    "weight": Value("float64"),
})
LICENSE_LINK = "https://github.com/Zyphra/ZTTS2-Eval/blob/main/DATA_LICENSE.md"
CARD_TEXT = f"""\
# ZTTS2-Eval Mini (Open TTS Leaderboard copy)

The coreset panels of [ZTTS2-Eval](https://github.com/Zyphra/ZTTS2-Eval), a multilingual zero-shot TTS benchmark:
each row is a sentence to synthesize, in the voice of a prompt clip whose transcript is given.

| Config | Source | License | Languages | Items |
|---|---|---|---|---|
| `clean` | [FLEURS-R](https://huggingface.co/datasets/google/fleurs-r) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | 35 | 3,500 |
| `in_the_wild` | [VoxBlink2](https://voxblink2.github.io/) (YouTube) | [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), **non-commercial** | 20 | 1,919 |

Each language is a split, named by its ZTTS2-Eval code (Tagalog is `fil` in `clean` and `tl` in `in_the_wild`).
See [DATA_LICENSE.md]({LICENSE_LINK}) for the terms of each source.

**Panel weights.** `weight` is the item's panel weight; the weights sum to 1 per split. A language's estimate of the
full ZTTS2-Eval score is the weighted mean, over its items, of the per-utterance WER clipped at 100% (CER for `zh`,
`ja`, `ko`, `th`), as ZTTS2-Eval's `scripts/coreset/score_panel.py` computes it. The Open TTS Leaderboard does
not use the weights: it scores each split like its other datasets, with every item counted equally.

**Changes:** reorganized from the ZTTS2-Eval checkout's `data/clean_mini/` and `data/in-the-wild_mini/` by
[`scripts/prepare_ztts2_eval_mini.py`](https://github.com/huggingface/open-tts-leaderboard/blob/main/scripts/prepare_ztts2_eval_mini.py).
Text and audio are unchanged. The In-The-Wild held-out target recordings are not included.

Used by the [Open TTS Leaderboard](https://github.com/huggingface/open-tts-leaderboard).

## Citation

```bibtex
@misc{{ztts2eval,
  title={{{{ZTTS2-Eval}}: A Multilingual, In-the-Wild Zero-Shot TTS Benchmark}},
  author={{Zyphra}},
  year={{2026}},
  howpublished={{\\url{{https://github.com/Zyphra/ZTTS2-Eval}}}},
}}

@article{{ma2024fleursr,
  title={{{{FLEURS-R}}: A Restored Multilingual Speech Corpus for Generation Tasks}},
  author={{Ma, Min and Koizumi, Yuma and Karita, Shigeki and Zen, Heiga and Riesa, Jason and Bapna, Ankur and Rosenberg, Andrew}},
  journal={{arXiv preprint arXiv:2408.06227}},
  year={{2024}}
}}

@inproceedings{{lin2024voxblink2,
  title={{{{VoxBlink2}}: A 100K+ Speaker Recognition Corpus and the Open-Set Speaker-Identification Benchmark}},
  author={{Lin, Yuke and others}},
  booktitle={{Interspeech}},
  year={{2024}}
}}
```
"""


def read_weights(root: Path) -> dict:
    """Read both panels' weights and check their languages. Returns {config: {split: {item id: weight}}}."""
    weights = {}
    for config, panel_dir in PANEL_DIRS.items():
        with open(root / "data" / panel_dir / "panel.json", encoding="utf-8") as f:
            panel = json.load(f)
        diff = set(panel["langs"]) ^ set(PANEL_LANGS[config])
        if diff:
            raise SystemExit(f"{panel_dir}/panel.json languages differ from PANEL_LANGS['{config}']: {sorted(diff)}")
        weights[config] = panel["langs"]
    return weights


def push_card(repo_id: str):
    """Set the license and put CARD_TEXT at the top of the card. Runs after every split is pushed, since each
    push_to_hub rewrites the card's metadata; the `configs` / `dataset_info` it wrote are kept."""
    card = DatasetCard.load(repo_id, repo_type="dataset")
    card.data.license = "other"
    card.data.license_name = "ztts2-eval-mixed"
    card.data.license_link = LICENSE_LINK
    card.data.task_categories = ["text-to-speech"]
    card.text = CARD_TEXT
    card.push_to_hub(repo_id, repo_type="dataset", commit_message="Add dataset card (licenses and attribution)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=DEFAULT_REPO_ID,
        help=f"HF dataset id to push to (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--ztts2-root",
        default="ZTTS2-Eval",
        help="Path to the ZTTS2-Eval checkout (the dir containing `data/`; scp paths are relative to it).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(PANEL_DIRS),
        choices=list(PANEL_DIRS),
        help="Configs to upload (default: both).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate only; do not upload.")
    parser.add_argument("--private", action="store_true", help="Push a private copy (default: public).")
    args = parser.parse_args()

    root = Path(args.ztts2_root)
    if not (root / "data").is_dir():
        raise SystemExit(f"{root / 'data'} not found — pass --ztts2-root pointing at the ZTTS2-Eval checkout.")

    # Always written (it is a local file): scoring needs it whether or not anything is pushed.
    weights = read_weights(root)

    totals = {}
    for config in args.configs:
        for split in PANEL_LANGS[config]:
            split_weights = weights[config][split]
            records = parse_leaf(root / "data" / PANEL_DIRS[config] / split, root,
                                 with_prompt_audio=True, with_ground_truth=False)
            ids = {r["id"] for r in records}
            if ids != set(split_weights):
                raise SystemExit(f"{config}/{split}: {len(ids - set(split_weights))} row(s) without a weight, "
                                 f"{len(set(split_weights) - ids)} weight(s) without a row")
            for r in records:
                r["weight"] = split_weights[r["id"]]
            dataset = Dataset.from_dict({key: [r[key] for r in records] for key in FEATURES}, features=FEATURES)
            print(f"  {config}/{split}: {len(dataset)} examples")
            totals[config] = totals.get(config, 0) + len(dataset)
            if not args.dry_run:
                # `private` only applies when the repo is created; an existing repo keeps its visibility.
                dataset.push_to_hub(args.repo_id, config_name=config, split=split, private=args.private)

    for config, total in totals.items():
        print(f"{config}: {total} examples in {len(PANEL_LANGS[config])} splits")
    if args.dry_run:
        print("Dry run — nothing uploaded.")
        return
    print("Writing the dataset card ...")
    push_card(args.repo_id)
    print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
