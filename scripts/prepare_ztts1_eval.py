"""
Script to prepare the ZTTS1-Eval benchmark data for the HF hub.

ZTTS1-Eval is Zyphra's multilingual zero-shot TTS benchmark, released with ZONOS2.
GitHub: https://github.com/Zyphra/ZTTS1-Eval

Clone the benchmark repo first; this script reads its `data/` tree in place:
```
git clone https://github.com/Zyphra/ZTTS1-Eval.git
python scripts/prepare_ztts1_eval.py Zyphra/ztts1_eval
```

The data uses CV3-Eval's Kaldi-style layout (see prepare_cv3_eval.py): each language dir holds
`text`, `prompt_text` and `prompt_wav.scp`, keyed by utterance id, plus a `waveform/` dir.

Each set becomes a config of the dataset; each language becomes a split within it:

  zero_shot            11 splits (en zh hard_en hard_zh de es fr it ja ko ru)            4624 rows
  zero_shot_voxblink   17 splits (en zh de es fr it ja ko ru pt ar hi tr tl pl id th)   1618 rows

`zero_shot` is the Clean set (FLEURS-R, CC-BY-4.0); `zero_shot_voxblink` is the In-The-Wild set
(VoxBlink2, CC-BY-NC-SA-4.0, non-commercial). Both licenses allow redistribution, so the copy is
pushed **public**.

Usage:
```
python scripts/prepare_ztts1_eval.py <hf_dataset_id> [--ztts1-root ZTTS1-Eval] [--configs zero_shot]
python scripts/prepare_ztts1_eval.py <hf_dataset_id> --dry-run   # parse + validate, no upload
```
"""

import argparse
from pathlib import Path

from datasets import Audio, Dataset, Features, Value

# Same layout as CV3-Eval. Resolves because `python scripts/<name>.py` puts scripts/ on sys.path.
from prepare_cv3_eval import parse_leaf


# Hub config (= dir under `data/`) -> splits (= language dirs within it).
CONFIGS = {
    "zero_shot": ["en", "zh", "hard_en", "hard_zh", "de", "es", "fr", "it", "ja", "ko", "ru"],
    "zero_shot_voxblink": ["en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru",
                           "pt", "ar", "hi", "tr", "tl", "pl", "id", "th"],
}

FEATURES = Features({
    "id": Value("string"),
    "prompt_text": Value("string"),
    "prompt_audio": Audio(),
    "text": Value("string"),
})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="HF dataset id to push to, e.g. 'Zyphra/ztts1_eval'.")
    parser.add_argument(
        "--ztts1-root",
        default="ZTTS1-Eval",
        help="Path to the ZTTS1-Eval checkout (the dir containing `data/`; scp paths are relative to it).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(CONFIGS),
        choices=list(CONFIGS),
        help=f"Configs to upload (default: {' '.join(CONFIGS)}).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate only; do not upload.")
    args = parser.parse_args()

    root = Path(args.ztts1_root)
    data_dir = root / "data"
    if not data_dir.is_dir():
        raise SystemExit(f"{data_dir} not found — pass --ztts1-root pointing at the ZTTS1-Eval checkout.")

    total = 0
    for config_name in args.configs:
        for split in CONFIGS[config_name]:
            print(f"Parsing '{config_name}/{split}' ...")
            records = parse_leaf(data_dir / config_name / split, root,
                                 with_prompt_audio=True, with_ground_truth=False)
            columns = {key: [r[key] for r in records] for key in FEATURES}
            dataset = Dataset.from_dict(columns, features=FEATURES)
            print(f"  {len(dataset)} examples")
            total += len(dataset)

            if args.dry_run:
                continue
            print(f"Pushing config '{config_name}' split '{split}' to {args.repo_id} ...")
            dataset.push_to_hub(
                args.repo_id,
                config_name=config_name,
                # Public — both data licenses allow redistribution (see docstring).
                private=False,
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
