"""
Script to prepare the Gradium TTS hard sentences set for the HF hub.

Hub: https://huggingface.co/datasets/gradium/tts-hard-sentences-202610

Text only (no prompt audio), so there is no voice-clone mode. The source is a single `test` split
with a `language` column; everything goes under the `tts` config with one split per language.

Usage:
```
python scripts/prepare_gradium_eval.py <hf_dataset_id> [--langs en fr ...]
```
"""

import argparse

from datasets import load_dataset


SOURCE = "gradium/tts-hard-sentences-202610"
CONFIG = "tts"
LANGUAGES = ["en", "fr", "es", "pt", "de"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="HF dataset id to push to, e.g. 'username/gradium_eval'")
    parser.add_argument(
        "--langs",
        nargs="+",
        default=LANGUAGES,
        choices=LANGUAGES,
        help="Language codes to upload (default: all).",
    )
    args = parser.parse_args()

    source = load_dataset(SOURCE, split="test")
    for lang in args.langs:
        print(f"Parsing '{CONFIG}/{lang}' ...")
        dataset = source.filter(lambda r: r["language"] == lang.upper())
        dataset = dataset.select_columns(["item_id", "category", "sentence"])
        dataset = dataset.rename_columns({"item_id": "id", "sentence": "text"})
        print(f"  {len(dataset)} examples")

        print(f"Pushing config '{CONFIG}' split '{lang}' to {args.repo_id} ...")
        dataset.push_to_hub(args.repo_id, config_name=CONFIG, private=True, split=lang)
        print(f"  done ({CONFIG}/{lang}).")

    print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
