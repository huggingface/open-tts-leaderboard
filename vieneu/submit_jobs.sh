#!/bin/bash
# HF Jobs TTS eval — VieNeu-TTS v3 Turbo backend (fixed default voice → generate → transcribe → score; no SIM).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["pnnbao-ump/VieNeu-TTS-v3-Turbo" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage 1 runs on a public PyTorch image and pip-installs the SDK at job start (no Docker Space needed).
# Install time is before the timed loop, so RTFx is unaffected. To use a Space instead, build one that
# installs the same packages and set VIENEU_IMAGE=hf.co/spaces/<id>.
VIENEU_IMAGE="${VIENEU_IMAGE:-pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime}"
VIENEU_PIP="${VIENEU_PIP:-vieneu[cuda]==3.8.3}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe"           # no SIM (default voice); no UTMOS (English only)
RUN_EVAL_INJECT="$(inject_run_eval)"

# Vietnamese only for now (the model is Vietnamese/English).
supports_language() { [[ "$1" == "vi" ]]; }

# ── Stage 1: TTS generation (batched on GPU) ─────────────────────────────────
generate_stage() {
    local _ BATCH_SIZE
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT:-8h}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${VIENEU_IMAGE}" \
        bash -c "
            pip install --no-cache-dir '${VIENEU_PIP}' 'transformers==4.57.6' soundfile 'datasets<4.0' tqdm numpy &&
            ${RUN_EVAL_INJECT}
            cd /app/vieneu && python run_eval.py \
                --model_id=${MODEL_ID} \
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
MODEL_CONFIGS=("pnnbao-ump/VieNeu-TTS-v3-Turbo 32")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
