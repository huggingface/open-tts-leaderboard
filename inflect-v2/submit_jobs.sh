#!/bin/bash
# HF Jobs TTS eval — Inflect v2 backend (fixed voice → 2 stages: generate → transcribe → score).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["owensong/Inflect-Micro-v2" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
#
# Serves the Inflect **v2** family (Micro-v2, Nano-v2), which share one codebase. Inflect-Nano-**v1**
# is a different architecture and has its own backend under inflect-nano/.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-inflect-v2}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-inflect-v2/blob/main/Dockerfile)
DEFAULT_STAGES="generate transcribe utmos"                # no SIM (fixed voice, no reference to compare)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Languages this backend can select ────────────────────────────────────────
# English-only (model cards: `language: [en]`); non-en splits are skipped rather than synthesized
# with English phonemes, which would give a meaningless (>100%) CER.
INFLECT_V2_LANGUAGES="en"
supports_language() { [[ " ${INFLECT_V2_LANGUAGES} " == *" $1 "* ]]; }

# ── Stage 1: TTS generation (one sample at a time — no batched API) ──────────
# The model repo carries the inference code as well as the ~40 MB checkpoint, so run_eval.py
# snapshot_downloads it at job start; that is what lets one image serve both v2 sizes.
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/inflect-v2 && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" ────────────────────
# The optional 4th field points one line at a different dataset repo (default: Seed-TTS).
# English-only model, so only the `en` splits.
DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
)

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
# Nano-v2 is the same code path and image — uncomment to evaluate both in one submission.
MODEL_CONFIGS=(
    "owensong/Inflect-Micro-v2"
    # "owensong/Inflect-Nano-v2"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
