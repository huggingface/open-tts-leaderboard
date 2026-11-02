#!/bin/bash
# HF Jobs TTS eval — NeuTTS Nano backend (generate → transcribe → SIM* → score; *voice-clone only, VOICE_CLONE=true).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["neuphonic/neutts-nano 32" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-neutts}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-neutts/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Voice cloning: VOICE_CLONE=true appends `_voice_clone` + enables SIM.
# Either way run_eval.py synthesizes from the per-sample reference (NeuTTS always needs one). ──
SUPPORTS_VOICE_CLONE=true

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size" (batch_size optional). ──
generate_stage() {
    local _ BATCH_SIZE BATCH_SIZE_ARG=""
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    [[ -n "${BATCH_SIZE}" ]] && BATCH_SIZE_ARG="--batch_size=${BATCH_SIZE}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/neutts-nano && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${VOICE_CLONE_FLAG} \
                ${BATCH_SIZE_ARG} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment to pick) ──
DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    # MiniMax multilingual (100 sentences/language): the languages the model supports. No `uk`: Qwen3-ASR has no Ukrainian.
    "tts en en ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id batch_size" (batch_size optional; CLI args override) ────
MODEL_CONFIGS=(
    "neuphonic/neutts-nano 32"
    "neuphonic/neutts-nano-q8-gguf 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
