#!/bin/bash
# HF Jobs TTS eval — LFM3-Audio backend (fixed voice → generate → transcribe → UTMOS → score).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["LiquidAI/LFM3-Audio-2B 32" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-lfm3-audio}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-lfm3-audio/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"                 # no SIM (fixed voice)
RUN_EVAL_INJECT="$(inject_run_eval)"

# Two built-in voices (female/male), picked by the system prompt; no voice cloning.
LFM_VOICE="${LFM_VOICE:-female}"

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size". ──
generate_stage() {
    local _ BATCH_SIZE
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/lfm3-audio && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${LFM_VOICE} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --batch_size=${BATCH_SIZE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# The 18 languages of the model card (LANGUAGES in run_eval.py).
SUPPORTED_LANGUAGES="en zh ja ko fr de es it pt nl pl el fi tr ar da he sv"

# ── Models: "model_id batch_size" (CLI args override) ────────────────────────
MODEL_CONFIGS=(
    "LiquidAI/LFM3-Audio-2B 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
