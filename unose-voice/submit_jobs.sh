#!/bin/bash
# HF Jobs TTS eval — Unose-Voice backend (fixed voice → 2 stages: generate → transcribe → score).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["igunosalabs/Unose-Voice-v1.0" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-igunosalabs/evals-unose-voice}"   # stage-1 image
DEFAULT_STAGES="generate transcribe utmos"      # no SIM (fixed voice)
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Supported languages ──────────────────────────────────────────────────────
# Unose-Voice-v1.0 is an English model trained with Nigerian English / West African cadence.
supports_language() { [[ "$1" == "en" ]]; }

# ── Stage 1: TTS generation ──────────────────────────────────────────────────
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/unose-voice && python run_eval.py \
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

# ── Model & dataset configurations ───────────────────────────────────────────
MODEL_CONFIGS=(
    "igunosalabs/Unose-Voice-v1.0"
)

# Standard evaluation splits (English)
DATASET_CONFIGS=(
    "hf-audio/minimax_eval tts en"
    "hf-audio/cv3_eval zero_shot en"
)

run_pipeline "$@"
