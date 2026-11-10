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
supports_language() { [[ " en zh ja ko de fr es " == *" $1 "* ]]; }

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

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
# supports_language() skips the splits the selected mode can't do.
# Without a 4th field the line uses DEFAULT_DATASET_PATH (hf-audio/seed_tts_eval).
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
    "tts en en ${MINIMAX_EVAL_PATH}"
    "tts zh zh ${MINIMAX_EVAL_PATH}"
    "tts ja ja ${MINIMAX_EVAL_PATH}"
    "tts ko ko ${MINIMAX_EVAL_PATH}"
    "tts fr fr ${MINIMAX_EVAL_PATH}"
    "tts de de ${MINIMAX_EVAL_PATH}"
    "tts es es ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=(
    "inclusionAI/Ming-3.0-tts-0.5B"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
