#!/bin/bash
# HF Jobs TTS eval — tiny-voca-pt-pt backend (fixed voice → generate → transcribe → UTMOS → score; no
# SIM). European Portuguese. Runs locally; submits HF Jobs.
# Usage: HF_TOKEN=hf_... bash tiny-voca-pt-pt/submit_jobs.sh ["joserui/tiny-voca-pt-pt F3" ...]
# Smoke test: MAX_EVAL_SAMPLES=8 bash tiny-voca-pt-pt/submit_jobs.sh
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 environment: no Docker Space. GEN_IMAGE and gen_setup live in job_image.sh.
source "${BACKEND_DIR}/job_image.sh"
# torch intra-op threads (run_eval.py's --threads). 1: tiny-voca-pt-pt's matrices are small, and a fixed
# count keeps RTFx independent of the h200 host's core count.
TINY_VOCA_PTPT_THREADS="${TINY_VOCA_PTPT_THREADS:-1}"
GEN_TIMEOUT="${GEN_TIMEOUT:-2h}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"       # no SIM (fixed voice); UTMOS auto-skips non-en
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Voice → output suffix ────────────────────────────────────────────────────
# Both voices share one repo, so a non-default voice is a leaderboard variant: MODE_SUFFIX must match
# the suffix run_eval.py writes (publish with `open_results_pr.py --variant m1`).
resolve_mode() {
    local _ VOICE
    read -r _ VOICE <<< "${MODEL_CFG}"
    CLONE="false"; VOICE_CLONE_FLAG=""
    case "${VOICE:-F3}" in
        F3) MODE_SUFFIX="" ;;
        M1) MODE_SUFFIX="_m1" ;;
        *) echo "tiny-voca-pt-pt: unknown voice '${VOICE}' (F3, M1)" >&2; exit 1 ;;
    esac
    true
}

# Portuguese only (pronuncia-pt European Portuguese front end).
SUPPORTED_LANGUAGES="pt"

# ── Stage 1: TTS generation ─────────────────────────────────────────────────
# CPU on the host CPU of the h200 flavor; --device=cpu is enforced inside run_eval.py.
generate_stage() {
    local _ VOICE
    read -r _ VOICE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${GEN_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            $(gen_setup cpu)
            cd /app/tiny-voca-pt-pt && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${VOICE:-F3} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cpu \
                --threads=${TINY_VOCA_PTPT_THREADS} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Models: "model_id voice" (CLI args override) ────────────────────────────
MODEL_CONFIGS=(
    "joserui/tiny-voca-pt-pt F3"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
