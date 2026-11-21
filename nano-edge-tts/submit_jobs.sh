#!/bin/bash
# HF Jobs TTS eval — nano-edge-tts-standalone backend (fixed voice → generate → transcribe → utmos → score; no SIM).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash nano-edge-tts/submit_jobs.sh ["drmhse/nano-edge-tts-standalone af_heart" ...]
# Smoke test: MAX_EVAL_SAMPLES=8 ONLY_LANGS=en bash nano-edge-tts/submit_jobs.sh
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Kokoro's image: the model uses Kokoro's front end, so only safetensors is added at job start.
TTS_SPACE="${TTS_SPACE:-bezzam/evals-kokoro}"
NANO_THREADS="${NANO_THREADS:-1}"                # one CPU thread on the h200 host, as the other CPU backends
GEN_TIMEOUT="${GEN_TIMEOUT:-2h}"
DEFAULT_STAGES="generate transcribe utmos"       # no SIM (fixed voice)
RUN_EVAL_INJECT="$(inject_run_eval)"

SUPPORTED_LANGUAGES="en"

# ── Stage 1: TTS generation. MODEL_CFG = "model_id voice" (no batched API — one sample at a time) ──
generate_stage() {
    local _ VOICE
    read -r _ VOICE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            pip install --no-cache-dir -q safetensors &&
            cd /app/nano-edge-tts && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${VOICE:-af_heart} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cpu \
                --threads=${NANO_THREADS} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Models: "model_id voice" (CLI args override) ─────────────────────────────
MODEL_CONFIGS=("drmhse/nano-edge-tts-standalone af_heart")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
