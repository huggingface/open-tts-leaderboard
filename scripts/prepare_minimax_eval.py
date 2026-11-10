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

Unlike the Seed-TTS-Eval and CV3-Eval copies, this one is pushed **public** by default: the test set
is CC BY-SA 4.0, which allows redistribution with attribution under the same license. The script
writes a dataset card that does both (see CARD_TEXT).

Usage:
```
python scripts/prepare_minimax_eval.py [hf_dataset_id] [--data-dir minimax_testset] [--langs ar yue ...] [--private]
```
"""

import argparse
from pathlib import Path

from datasets import Audio, Dataset, Features, Value
from huggingface_hub import DatasetCard


CONFIG = "tts"
DEFAULT_REPO_ID = "hf-audio/minimax_eval"
SOURCE_REPO_ID = "MiniMaxAI/TTS-Multilingual-Test-Set"
LICENSE = "cc-by-sa-4.0"
# Top of the dataset card: the attribution, license and change notice CC BY-SA 4.0 requires.
CARD_TEXT = f"""\
# MiniMax TTS Multilingual Test Set (Open TTS Leaderboard copy)

This is a reformatted copy of [{SOURCE_REPO_ID}](https://huggingface.co/datasets/{SOURCE_REPO_ID}), created by
MiniMax and licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This copy is
distributed under the same license.

**Changes:** reorganized into a single `tts` config with one split per language (named by language code), each row
pairing a test sentence with its speaker's prompt clip and that clip's transcript, by
[`scripts/prepare_minimax_eval.py`](https://github.com/huggingface/open-tts-leaderboard/blob/main/scripts/prepare_minimax_eval.py).
The text and audio themselves are unchanged. Per the original card, the prompt clips come from Mozilla Common Voice.

Used by the [Open TTS Leaderboard](https://github.com/huggingface/open-tts-leaderboard): 100 sentences in each of
24 languages, synthesized in each prompt speaker's voice.

## Citation

```bibtex
@misc{{minimax2025minimaxspeechintrinsiczeroshottexttospeech,
      title={{MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder}},
      author={{Bowen Zhang, Congchao Guo, Geng Yang, Hang Yu, Haozhe Zhang, Heidi Lei, Jialong Mai, Junjie Yan, Kaiyue Yang, Mingqi Yang, Peikai Huang, Ruiyang Jin, Sitan Jiang, Weihua Cheng, Yawei Li, Yichen Xiao, Yiying Zhou, Yongmao Zhang, Yuan Lu, Yucen He}},
      year={{2025}},
      eprint={{2505.07916}},
      archivePrefix={{arXiv}},
      primaryClass={{eess.AS}},
      url={{https://arxiv.org/abs/2505.07916}},
}}
```
"""
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


def push_card(repo_id: str):
    """Set the license and put CARD_TEXT at the top of the card. Runs after every split is pushed, since each
    push_to_hub rewrites the card's metadata; the `configs` / `dataset_info` it wrote are kept."""
    card = DatasetCard.load(repo_id, repo_type="dataset")
    card.data.license = LICENSE
    card.data.task_categories = ["text-to-speech"]
    card.data.source_datasets = [SOURCE_REPO_ID]
    card.text = CARD_TEXT
    card.push_to_hub(repo_id, repo_type="dataset", commit_message="Add dataset card (CC BY-SA 4.0 attribution)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=DEFAULT_REPO_ID,
        help=f"HF dataset id to push to (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument("--private", action="store_true", help="Push a private copy (default: public).")
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
        # `private` only applies when the repo is created; an existing repo keeps its visibility.
        dataset.push_to_hub(args.repo_id, config_name=CONFIG, private=args.private, split=lang)
        print(f"  done ({CONFIG}/{lang}).")

    print("Writing the dataset card ...")
    push_card(args.repo_id)
    print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
