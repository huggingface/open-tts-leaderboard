#!/bin/bash
# HF Jobs TTS eval — Inflect-Nano backend (fixed voice → 2 stages: generate → transcribe → score).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["owensong/Inflect-Nano-v1" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-inflect}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-inflect/blob/main/Dockerfile)
DEFAULT_STAGES="generate transcribe utmos"             # no SIM (fixed voice)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Stage 1: TTS generation (Inflect-Nano synthesizes one sample at a time — no batch size) ──
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/inflect-nano && python run_eval.py \
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

# ── Languages this backend can select ────────────────────────────────────────
# English-only (model card: `language: [en]`); non-en splits are skipped rather than synthesized
# with English phonemes, which would give a meaningless (>100%) CER.
INFLECT_NANO_LANGUAGES="en"
supports_language() { [[ " ${INFLECT_NANO_LANGUAGES} " == *" $1 "* ]]; }

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
)

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=("owensong/Inflect-Nano-v1")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
