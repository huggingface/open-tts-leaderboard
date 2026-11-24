#!/bin/bash
# HF Jobs TTS eval — tiny-voca-arabic backend (fixed voice → generate → transcribe → UTMOS → score;
# no SIM). Modern Standard Arabic. Runs locally; submits HF Jobs.
# Usage: HF_TOKEN=hf_... bash tiny-voca-arabic/submit_jobs.sh ["oddadmix/tiny-voca-arabic" ...]
# Smoke test: MAX_EVAL_SAMPLES=8 bash tiny-voca-arabic/submit_jobs.sh
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 environment: no Docker Space. GEN_IMAGE and gen_setup live in job_image.sh.
source "${BACKEND_DIR}/job_image.sh"
# onnxruntime intra-op threads (run_eval.py's --threads), for both the TTS model and the CATT
# diacritiser. 1: the matrices are small, and a fixed count keeps RTFx independent of the host's cores.
TINY_VOCA_ARABIC_THREADS="${TINY_VOCA_ARABIC_THREADS:-1}"
GEN_TIMEOUT="${GEN_TIMEOUT:-2h}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"       # no SIM (fixed voice); UTMOS auto-skips non-en
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Mode: single synthetic voice → default row (no suffix) ────────────────────
resolve_mode() {
    CLONE="false"; VOICE_CLONE_FLAG=""; MODE_SUFFIX=""
    true
}

# Arabic only (Modern Standard Arabic front end: CATT diacritiser + rule-based MSA phonemes).
SUPPORTED_LANGUAGES="ar"

# ── Stage 1: TTS generation ─────────────────────────────────────────────────
# CPU on the host CPU of the h200 flavor; --device=cpu is enforced inside run_eval.py.
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${GEN_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            $(gen_setup cpu)
            cd /app/tiny-voca-arabic && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cpu \
                --threads=${TINY_VOCA_ARABIC_THREADS} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Models: "model_id" (CLI args override) ──────────────────────────────────
MODEL_CONFIGS=(
    "oddadmix/tiny-voca-arabic"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
