#!/bin/bash
# HF Jobs TTS eval — PAWP-TTS folded ONNX backend (fixed voice -> generate -> transcribe -> utmos -> score; no SIM).
# Usage: HF_TOKEN=hf_... bash pawp_tts/submit_jobs.sh ["Firoj112/pawp-tts-1.5m" ...]
# Smoke test: MAX_EVAL_SAMPLES=8 ONLY_LANGS=en bash pawp_tts/submit_jobs.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

source "${BACKEND_DIR}/job_image.sh"
PAWP_THREADS="${PAWP_THREADS:-4}"
GEN_TIMEOUT="${GEN_TIMEOUT:-2h}"
DEFAULT_STAGES="generate transcribe utmos"
RUN_EVAL_INJECT="$(inject_run_eval)"

resolve_mode() {
    CLONE="false"
    VOICE_CLONE_FLAG=""
    MODE_SUFFIX=""
    true
}

SUPPORTED_LANGUAGES="en"

generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${GEN_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            $(gen_setup cpu)
            cd /app/pawp_tts && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cpu \
                --threads=${PAWP_THREADS} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

MODEL_CONFIGS=("Firoj112/pawp-tts-1.5m" "Firoj112/pawp-tts-300k")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
