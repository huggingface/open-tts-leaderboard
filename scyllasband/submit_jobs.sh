#!/bin/bash
# HF Jobs TTS eval — Scylla's Band backend (fixed voice → generate → transcribe → UTMOS → score).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["spybyscript/scyllasband" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 image: the scyllasband runtime with ONNX Runtime for CUDA, and the WebGPU (CUDA) and LiteRT
# (CPU) bundles baked in (Dockerfile: https://huggingface.co/spaces/spybyscript/evals-scyllasband/blob/main/Dockerfile).
TTS_SPACE="${TTS_SPACE:-spybyscript/evals-scyllasband}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"      # no SIM (fixed voice, no cloning)
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Languages ────────────────────────────────────────────────────────────────
# Six languages, every voice speaks all of them; `en` is the voice's own English dialect.
SCYLLASBAND_LANGUAGES="en es it fr de vi"
supports_language() { [[ " ${SCYLLASBAND_LANGUAGES} " == *" $1 "* ]]; }
# One fixed built-in voice for every split (tuesday: British English).
SCYLLASBAND_VOICE="${SCYLLASBAND_VOICE:-tuesday}"

# ── Stage 1: TTS generation (one sample at a time — no batched API) ──────────
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/scyllasband && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${SCYLLASBAND_VOICE} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" ────────────────────
DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot fr fr ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot de de ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot it it ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    # MiniMax multilingual (100 sentences/language): the languages the model supports.
    "tts en en ${MINIMAX_EVAL_PATH}"
    "tts es es ${MINIMAX_EVAL_PATH}"
    "tts fr fr ${MINIMAX_EVAL_PATH}"
    "tts de de ${MINIMAX_EVAL_PATH}"
    "tts it it ${MINIMAX_EVAL_PATH}"
    "tts vi vi ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=("spybyscript/scyllasband")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
