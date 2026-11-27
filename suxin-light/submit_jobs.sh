#!/bin/bash
# HF Jobs TTS eval — suxin-light backend (fixed voice → generate → transcribe → UTMOS → score; no
# SIM). Mandarin + English. Runs locally; submits HF Jobs.
# Usage: HF_TOKEN=hf_... bash suxin-light/submit_jobs.sh ["TEQUILA123/suxin-light" ...]
# Smoke test: MAX_EVAL_SAMPLES=8 ONLY_LANGS="en zh" bash suxin-light/submit_jobs.sh
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 environment: no Docker Space. GEN_IMAGE and gen_setup live in job_image.sh; shared with
# submit_ttfa_jobs.sh so the TTFA probe runs the same install.
source "${BACKEND_DIR}/job_image.sh"
# onnxruntime intra-op threads (run_eval.py's --threads). 1: a fixed count keeps RTFx independent
# of the h200 host's core count, matching the tiny-voca / tiny-broca / apptor convention. The
# model card's on-device recommendation is 4 threads on an 8-core A55; that is a different machine.
SUXIN_LIGHT_THREADS="${SUXIN_LIGHT_THREADS:-1}"
GEN_TIMEOUT="${GEN_TIMEOUT:-2h}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"       # no SIM (fixed voice); UTMOS auto-skips non-en
RUN_EVAL_INJECT="$(inject_run_eval)"

# Mandarin + English (pypinyin + espeak-ng mixed frontend). MiniMax yue is skipped.
SUPPORTED_LANGUAGES="zh en"

# ── Stage 1: TTS generation ─────────────────────────────────────────────────
# CPU on the host CPU of the h200 flavor; --device=cpu is enforced inside run_eval.py.
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${GEN_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            $(gen_setup cpu)
            cd /app/suxin-light && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cpu \
                --threads=${SUXIN_LIGHT_THREADS} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=("TEQUILA123/suxin-light")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
