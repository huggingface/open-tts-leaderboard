#!/bin/bash
# HF Jobs TTS eval — Ming-3.0-tts backend (generate → transcribe → SIM* → score; *voice-clone only).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["inclusionAI/Ming-3.0-tts-0.5B" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-ming-tts}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-ming-tts/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# VOICE_CLONE=true: zero-shot voice cloning + SIM; false (default) = a built-in voice. Both modes cover
# the model's seven languages (built-in voices exist for English and Chinese; the other five use the
# English voice cross-lingually, see run_eval.py).
SUPPORTS_VOICE_CLONE=true
SUPPORTED_LANGUAGES="en zh ja ko de fr es"

# ── Stage 1: TTS generation. MODEL_CFG = "model_id" (batch 1: the model's code assumes it). ──
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/ming-tts && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${VOICE_CLONE_FLAG} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=(
    "inclusionAI/Ming-3.0-tts-0.5B"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
