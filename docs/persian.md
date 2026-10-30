# Persian evaluation with Neyshekar

This recipe uses [shekar-ai/Neyshekar](https://huggingface.co/datasets/shekar-ai/Neyshekar),
not ParsVoice. The pinned V6.0 source revision is
`30696ada746a66bb3047c984744b5ecfd60fe4e9` (CC0 1.0): 2,149 test clips from 30 speakers
and 1,886 validation clips from 26 speakers. Source WAVs are mono, 16 kHz. See the
[dataset card](https://huggingface.co/datasets/shekar-ai/Neyshekar/blob/30696ada746a66bb3047c984744b5ecfd60fe4e9/README.md)
and [paper](https://arxiv.org/abs/2609.14542) for collection details and citation.
Do not attempt to identify speakers.

## Prepare

Install the repository's `requirements.txt`, then inspect the output locally:

```bash
python scripts/prepare_neyshekar_eval.py --dry-run --output-dir /tmp/neyshekar_eval
```

Only test/validation Parquet shards are downloaded. The much larger training split is
neither downloaded nor used. A normal `load_dataset(..., split="test")` can still prepare
all source splits, so this script restricts the actual downloaded files.

After inspection, upload a copy to a namespace you control:

```bash
python scripts/prepare_neyshekar_eval.py YOUR_USERNAME/neyshekar_eval
# Optional: --private; --splits test; --revision <source-commit>
```

Keep the repository name `neyshekar_eval`: the results publisher recognizes that marker
in manifests, just as it recognizes `seed_tts_eval`, `cv3_eval` and `minimax_eval`.
The generated card preserves the source, license, revision, pairing recipe, exclusion
counts and hashes. `--output-dir` also saves datasets with `save_to_disk`, plus a local
`provenance.json`. The command writes only after all requested splits pass validation.

## Pairing protocol

The official splits are assigned by speaker, so validation audio cannot serve as a
same-speaker prompt for test targets. Each split is paired independently:

1. Retain nonempty, normalizable transcripts with speaker IDs and finite 1–30 second durations.
2. For each speaker, select one fixed 3–10 second prompt, closest to 5 seconds; break ties
   by ascending numeric source ID. Require another distinct normalized sentence.
3. Exclude every selected prompt's recording ID and normalized sentence from all targets
   in that split, including copies recorded by other speakers. Deduplicate remaining
   target sentences per speaker, retaining the lowest source ID.
4. Emit targets ordered by speaker ID and numeric source ID. Validate the selected
   prompt WAVs as mono 16 kHz, 3–10 seconds. Preserve their bytes and both original texts.

Transcript equivalence for exclusion uses the Persian normalizer with whitespace removed,
so half-spaces, punctuation, digit scripts and letter variants cannot bypass it. No
model predictions or quality scores influence selection. Output rows retain `source_id`,
`prompt_id`, `speaker_id`, and `source_split`; revision lives in the card/provenance file.
Speaker/recording overlap between the requested evaluation splits fails preparation.

The pinned source produces:

| Output | Purpose | Targets | Speakers | Prompt ID/text exclusions | Duplicate targets |
|---|---|---:|---:|---:|---:|
| `tts/fa` | Final test | 2,098 | 30 | 31 | 20 |
| `tts_validation/fa` | Development only | 1,851 | 26 | 28 | 7 |

No rows fail the text/duration filters at this revision. Ordered `[target_id, prompt_id]`
pair hashes (SHA-256 of compact JSON):

- Test: `b170ed14adc3718ed9fb2d958456c203c3127c0a528e4d0ed7b120bd49414bfe`
- Validation: `24ce4e740a466a478421bd90a1bbabf8627fb8e12d1ee1cfa55937a49796c757`

This is an automatically constructed read-speech evaluation, not a manually audited TTS
benchmark. Speakers are not balanced; scores follow the source's transcript distribution.
The source card describes train/evaluation speaker disjointness; this script does not
audit the training split. Sentence overlap across source splits is not ruled out.
Models trained on Neyshekar must disclose that exposure; the predefined splits do not
prove independence from a model's training data. Use validation for development and
freeze the final test selection before comparing models.

## Generate and score

OmniVoice's existing backend accepts Persian; opt in after preparing a Hub copy:

```bash
# Submits paid HF Jobs, using the same H200/L4 hardware as other languages.
NEYSHEKAR_EVAL_PATH=YOUR_USERNAME/neyshekar_eval \
RESULTS_BUCKET=YOUR_USERNAME/your_bucket \
ONLY_LANGS=fa ONLY_DATASETS=neyshekar_eval MAX_EVAL_SAMPLES=8 \
VOICE_CLONE=true bash omnivoice/submit_jobs.sh
```

Remove `MAX_EVAL_SAMPLES=8` for the complete test set. To add another Persian-capable
backend, add `"tts fa fa ${NEYSHEKAR_EVAL_PATH}"` to its `DATASET_CONFIGS`, with the same
opt-in guard as OmniVoice. For development, use config `tts_validation`, split `fa`;
those manifests are deliberately excluded from the final results publisher.

The shared transcription stage already supports `--asr_language=fa` through
[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf). It does not require a new ASR model.
Score an existing test manifest locally without submitting jobs:

```bash
python scripts/score_model.py --model_id k2-fsa/OmniVoice --language fa --no_sync \
  --manifests MODEL_k2-fsa-OmniVoice_DATASET_YOUR_USERNAME-neyshekar_eval_tts_fa_voice_clone.jsonl
```

Persian uses **WER**, with the same multilingual compound-boundary policy as the other
space-delimited languages. `normalizer/persian.py` is applied to both reference and ASR text:

- NFKC and lowercase; fold Arabic yeh/alef maksura/kaf to Persian yeh/kaf.
- Remove short vowels, shadda/sukun, tatweel, directional formatting and explicit
  non-speech tags. Normalize optional written ezafe on heh. Preserve hamza/madda and
  lexical distinctions; do not stem, transliterate, or equate formal/colloquial words.
- Convert half-spaces to spaces. The existing shared alignment handles joined/split
  spellings (`می‌روم`, `می روم`, `میروم`).
- Fold Persian/Arabic-Indic digits to ASCII and expand integers with `num2words`' Persian
  rules. Handle comma/Arabic thousands groups and `.`/`٫` decimals before punctuation;
  decimals use `ممیز` and retain leading fractional zeros (`۳٫۰۵` → `سه ممیز صفر پنج`).
  This is a fixed notation policy, not a general Persian text verbalizer: dates, phone
  numbers, digit-by-digit readings, percentages, currencies and alternative decimal
  readings are not semantically reconciled. Numbers outside the supported range remain
  numeric. These limits should be considered when interpreting ASR-based WER.
- Replace punctuation/symbols with spaces and collapse whitespace.

RTFx and WavLM SIM reuse existing scorers; SIM is available only for voice cloning.
UTMOS22 remains English-only, so no Persian UTMOS is claimed. WER reflects both TTS and
ASR errors; no Persian TTS/ASR model evaluation is included in this code contribution.

## Publish results

The publisher supports separate `neyshekar.csv` and `neyshekar_voice_clone.csv` files
with `fa WER`, `fa RTFx`, and (for cloning) `fa SIM`. The first submission bootstraps their
headers if absent; validation cannot populate them. As usual, the default is a dry run:

```bash
python scripts/open_results_pr.py --model_id k2-fsa/OmniVoice \
  --bucket YOUR_USERNAME/your_bucket --targets neyshekar_voice_clone
# Add --open_pr only after a full run and inspection.
```

Maintainers still need to publish an official prepared dataset, run model evaluations,
merge result CSVs, and configure the hosted Space to display the new benchmark. This
repository PR alone does not make Persian scores live on the leaderboard.

## Local checks

```bash
pip install pytest
python -m pytest -q
bash -n scripts/tts_jobs_common.sh omnivoice/submit_jobs.sh
```
