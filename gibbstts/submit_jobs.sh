#!/bin/bash
# HF Jobs TTS eval — GibbsTTS backend (generate → transcribe → SIM → score; voice cloning only).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["ydqmkkx/GibbsTTS 32" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-gibbstts}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-gibbstts/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# GibbsTTS always clones a reference prompt, so every run is in voice_clone mode regardless of VOICE_CLONE.
SUPPORTS_VOICE_CLONE=true
resolve_mode() { VOICE_CLONE=true voice_clone_mode; }

# The base checkpoint covers English and Chinese only.
supports_language() { [[ "$1" == en || "$1" == zh ]]; }

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size". ──
# Sampling settings (steps, CFG, temperature, ...) and BF16/compile/tail padding are run_eval.py's defaults.
generate_stage() {
    local _ BATCH_SIZE
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/gibbstts && python run_eval.py \
                --model_id=${MODEL_ID} \
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
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH}"
    "zero_shot zh zh ${CV3_EVAL_PATH}"
    # MiniMax multilingual (100 sentences/language): the languages the model supports.
    "tts en en ${MINIMAX_EVAL_PATH}"
    "tts zh zh ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id batch_size" (CLI args override) ────────────────────────
MODEL_CONFIGS=(
    "ydqmkkx/GibbsTTS 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
