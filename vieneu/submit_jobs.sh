#!/bin/bash
# HF Jobs TTS eval — VieNeu-TTS v3 Turbo backend (fixed default voice → generate → transcribe → score; no SIM).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["pnnbao-ump/VieNeu-TTS-v3-Turbo" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-vieneu}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-vieneu/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe"           # no SIM (default voice); no UTMOS (English only)
RUN_EVAL_INJECT="$(inject_run_eval)"

# Vietnamese only for now (the model is Vietnamese/English).
supports_language() { [[ "$1" == "vi" ]]; }

# ── Stage 1: TTS generation (batched on GPU) ─────────────────────────────────
generate_stage() {
    local _ BATCH_SIZE MODE="v3turbo" ORT_SETUP=""
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    # v3 Nano is ONNX: swap the image's CPU onnxruntime for the GPU build so it runs on the CUDA EP
    # (1.23.x is the last line built for CUDA 12 + cuDNN 9, matching the image's torch 2.8 cu128).
    if [[ "${MODEL_ID}" == *-Nano ]]; then
        MODE="v3nano"
        ORT_SETUP="pip uninstall -y -q onnxruntime && pip install -q --no-cache-dir 'onnxruntime-gpu==1.23.2' &&"
    fi
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT:-8h}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            ${ORT_SETUP}
            cd /app/vieneu && python run_eval.py \
                --model_id=${MODEL_ID} \
                --mode=${MODE} \
                --batch_size=${BATCH_SIZE:-32} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" ─────────────────────
# MiniMax multilingual: 100 sentences per language, public at hf-audio/minimax_eval.
DATASET_CONFIGS=(
    "tts vi vi ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id batch_size" (CLI args override) ────────────────────────
MODEL_CONFIGS=(
    "pnnbao-ump/VieNeu-TTS-v3-Turbo 32"
    "pnnbao-ump/VieNeu-TTS-v3-Nano 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
