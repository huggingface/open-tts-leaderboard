"""
Script to prepare the MiniMax multilingual TTS test set for the HF hub.

Hub: https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set

Download via CLI:
```
hf download MiniMaxAI/TTS-Multilingual-Test-Set --repo-type dataset --local-dir minimax_testset
```

Layout (24 languages, one female and one male prompt speaker each):
  - speaker/<language>/<language>_<gender>/<clip>.mp3   prompt audio
  - speaker/prompt_text.txt                            <clip>.mp3|prompt text
  - text/<language>.txt                                <language>_<gender>|text to synthesize

Everything goes under the `tts` config; each language is a split (named by its language code).

Usage:
```
python scripts/prepare_minimax_eval.py <hf_dataset_id> [--data-dir minimax_testset] [--langs ar yue ...]
```
"""

import argparse
from pathlib import Path

from datasets import Audio, Dataset, Features, Value


CONFIG = "tts"
# Language code (split name) -> language name used in the file layout
LANGUAGES = {
    "ar": "arabic", "yue": "cantonese", "zh": "chinese", "cs": "czech", "nl": "dutch", "en": "english",
    "fi": "finnish", "fr": "french", "de": "german", "el": "greek", "hi": "hindi", "id": "indonesian",
    "it": "italian", "ja": "japanese", "ko": "korean", "pl": "polish", "pt": "portuguese", "ro": "romanian",
    "ru": "russian", "es": "spanish", "th": "thai", "tr": "turkish", "uk": "ukrainian", "vi": "vietnamese",
}


def parse_language(data_dir: Path, name: str, prompt_texts: dict[str, str]) -> list[dict]:
    """Pair each line of text/<name>.txt with its speaker's prompt clip and transcript."""
    records = []
    counts = {}
    with open(data_dir / "text" / f"{name}.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            speaker, text = line.split("|", 1)
            (clip,) = (data_dir / "speaker" / name / speaker).glob("*.mp3")
            counts[speaker] = counts.get(speaker, 0) + 1
            records.append({
                "id": f"{speaker}_{counts[speaker]:03d}",
                "prompt_text": prompt_texts[clip.name],
                "prompt_audio": str(clip),
                "text": text,
            })
    return records


def build_dataset(records: list[dict]) -> Dataset:
    feature_dict = {
        "id": Value("string"),
        "prompt_text": Value("string"),
        "prompt_audio": Audio(),
        "text": Value("string"),
    }
    columns = {key: [r[key] for r in records] for key in feature_dict}
    return Dataset.from_dict(columns, features=Features(feature_dict))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="HF dataset id to push to, e.g. 'username/minimax_eval'")
    parser.add_argument(
        "--data-dir",
        default="minimax_testset",
        help="Path to the downloaded test set (default: minimax_testset).",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=list(LANGUAGES),
        choices=list(LANGUAGES),
        help="Language codes to upload (default: all).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    with open(data_dir / "speaker" / "prompt_text.txt", encoding="utf-8") as f:
        prompt_texts = dict(line.strip().split("|", 1) for line in f if line.strip())

    for lang in args.langs:
        print(f"Parsing '{CONFIG}/{lang}' ({LANGUAGES[lang]}) ...")
        records = parse_language(data_dir, LANGUAGES[lang], prompt_texts)
        dataset = build_dataset(records)
        print(f"  {len(dataset)} examples")

        print(f"Pushing config '{CONFIG}' split '{lang}' to {args.repo_id} ...")
        
        dataset.push_to_hub(args.repo_id, config_name=CONFIG, private=True, split=lang)
        print(f"  done ({CONFIG}/{lang}).")

    print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
