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

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
# The languages of each benchmark that the model speaks (model card: 18 languages; see LANGUAGES
# in run_eval.py). Without a 4th field the line uses DEFAULT_DATASET_PATH (hf-audio/seed_tts_eval).
DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH}"
    "zero_shot zh zh ${CV3_EVAL_PATH}"
    "zero_shot fr fr ${CV3_EVAL_PATH}"
    "zero_shot es es ${CV3_EVAL_PATH}"
    "zero_shot ja ja ${CV3_EVAL_PATH}"
    "zero_shot ko ko ${CV3_EVAL_PATH}"
    "zero_shot de de ${CV3_EVAL_PATH}"
    "zero_shot it it ${CV3_EVAL_PATH}"
    "tts en en ${MINIMAX_EVAL_PATH}"
    "tts zh zh ${MINIMAX_EVAL_PATH}"
    "tts ja ja ${MINIMAX_EVAL_PATH}"
    "tts ko ko ${MINIMAX_EVAL_PATH}"
    "tts fr fr ${MINIMAX_EVAL_PATH}"
    "tts de de ${MINIMAX_EVAL_PATH}"
    "tts es es ${MINIMAX_EVAL_PATH}"
    "tts it it ${MINIMAX_EVAL_PATH}"
    "tts pt pt ${MINIMAX_EVAL_PATH}"
    "tts nl nl ${MINIMAX_EVAL_PATH}"
    "tts pl pl ${MINIMAX_EVAL_PATH}"
    "tts el el ${MINIMAX_EVAL_PATH}"
    "tts fi fi ${MINIMAX_EVAL_PATH}"
    "tts tr tr ${MINIMAX_EVAL_PATH}"
    "tts ar ar ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id batch_size" (CLI args override) ────────────────────────
MODEL_CONFIGS=(
    "LiquidAI/LFM3-Audio-2B 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
