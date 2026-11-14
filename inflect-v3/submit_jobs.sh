#!/bin/bash
# HF Jobs TTS eval — Inflect v3 backend (fixed voice: generate → transcribe → UTMOS → score).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["owensong/Inflect-Mini-v3" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
#
# run_eval.py downloads the repo at job start; no weights are in the image.
# Inflect v1 (inflect-nano/) and v2 (inflect-v2/) are different architectures with their own backends.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-inflect-v3}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-inflect-v3/blob/main/Dockerfile)
DEFAULT_STAGES="generate transcribe utmos"           # no SIM (fixed voice, no reference to compare)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# English-only (model card: `language: [en]`); other splits are skipped rather than read with English G2P.
supports_language() { [[ "$1" == "en" ]]; }

# ── Stage 1: TTS generation (one sample at a time — no batched API) ──────────
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/inflect-v3 && python run_eval.py \
                --model_id=${MODEL_ID} \
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
    "tts en en ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=(
    "owensong/Inflect-Mini-v3"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
