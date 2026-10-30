#!/bin/bash
# HF Jobs TTS eval — Maya 2 Native backend (generate → transcribe → SIM* → score; *voice-clone only).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["maya-research/maya2-native 32" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-maya2}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-maya2/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# VOICE_CLONE=true: zero-shot voice cloning + SIM; false (default) = a built-in voice (MAYA_VOICE).
SUPPORTS_VOICE_CLONE=true
MAYA_VOICE="${MAYA_VOICE:-Ahana}"

# 11 Indian languages + English (model card); of the benchmarks' languages only en and hi.
supports_language() { [[ "$1" == en || "$1" == hi ]]; }

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size" (batch = the engine's max_seats). ──
generate_stage() {
    local _ BATCH_SIZE
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/maya2 && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${MAYA_VOICE} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --batch_size=${BATCH_SIZE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${VOICE_CLONE_FLAG} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
# Without a 4th field the line uses DEFAULT_DATASET_PATH (hf-audio/seed_tts_eval).
DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH}"
    "tts en en ${MINIMAX_EVAL_PATH}"
    "tts hi hi ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id batch_size" (CLI args override) ────────────────────────
MODEL_CONFIGS=(
    "maya-research/maya2-native 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
