#!/bin/bash
# HF Jobs TTS eval — tiny-voca-hindi backend (fixed style → generate → transcribe → UTMOS → score;
# no SIM). Hindi. Runs locally; submits HF Jobs.
# Usage: HF_TOKEN=hf_... bash tiny-voca-hindi/submit_jobs.sh ["RL10x/tiny-voca-hindi" ...]
# Smoke test: MAX_EVAL_SAMPLES=8 bash tiny-voca-hindi/submit_jobs.sh
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 environment: no Docker Space. GEN_IMAGE and gen_setup live in job_image.sh.
source "${BACKEND_DIR}/job_image.sh"
# torch intra-op threads (run_eval.py's --threads). 1: tiny-voca-hindi's matrices are small, and a
# fixed count keeps RTFx independent of the h200 host's core count.
TINY_VOCA_HINDI_THREADS="${TINY_VOCA_HINDI_THREADS:-1}"
GEN_TIMEOUT="${GEN_TIMEOUT:-2h}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"       # no SIM (fixed voice); UTMOS auto-skips non-en
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Mode: single fixed style → default row (no suffix) ────────────────────────
resolve_mode() {
    CLONE="false"; VOICE_CLONE_FLAG=""; MODE_SUFFIX=""
    true
}

# Hindi only (Devanagari front end).
SUPPORTED_LANGUAGES="hi"

# ── Stage 1: TTS generation ─────────────────────────────────────────────────
# CPU on the host CPU of the h200 flavor; --device=cpu is enforced inside run_eval.py.
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${GEN_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            $(gen_setup cpu)
            cd /app/tiny-voca-hindi && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cpu \
                --threads=${TINY_VOCA_HINDI_THREADS} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Models: "model_id" (CLI args override) ──────────────────────────────────
MODEL_CONFIGS=(
    "RL10x/tiny-voca-hindi"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
