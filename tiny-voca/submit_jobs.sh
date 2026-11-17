#!/bin/bash
# HF Jobs TTS eval — tiny-voca backend (fixed voice → generate → transcribe → UTMOS → score; no SIM).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash tiny-voca/submit_jobs.sh ["n4ze3m/tiny-voca F2" ...]
# Smoke test: MAX_EVAL_SAMPLES=8 bash tiny-voca/submit_jobs.sh
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 environment: no Docker Space. GEN_IMAGE and gen_setup live in job_image.sh; shared with
# submit_ttfa_jobs.sh so the TTFA probe runs the same install.
source "${BACKEND_DIR}/job_image.sh"
# torch intra-op threads (run_eval.py's --threads). 1: tiny-voca's matrices are small, and a fixed
# count keeps RTFx independent of the h200 host's core count. The TTFA targets pass the same value.
TINY_VOCA_THREADS="${TINY_VOCA_THREADS:-1}"
GEN_TIMEOUT="${GEN_TIMEOUT:-2h}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"       # no SIM (fixed voice)
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Voice → output suffix ────────────────────────────────────────────────────
# Both voices share one repo, so a non-default voice is a leaderboard variant: MODE_SUFFIX must match
# the suffix run_eval.py writes (publish with `open_results_pr.py --variant m2`).
resolve_mode() {
    local _ VOICE
    read -r _ VOICE <<< "${MODEL_CFG}"
    CLONE="false"; VOICE_CLONE_FLAG=""
    case "${VOICE:-F2}" in
        F2) MODE_SUFFIX="" ;;
        M2) MODE_SUFFIX="_m2" ;;
        *) echo "tiny-voca: unknown voice '${VOICE}' (F2, M2)" >&2; exit 1 ;;
    esac
    true
}

# English only (American English phonemizer).
supports_language() { [[ "$1" == "en" ]]; }

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
            cd /app/tiny-voca && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${VOICE:-F2} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cpu \
                --threads=${TINY_VOCA_THREADS} \
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

# ── Models: "model_id voice" (CLI args override) ────────────────────────────
MODEL_CONFIGS=(
    "n4ze3m/tiny-voca F2"
    "n4ze3m/tiny-voca M2"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
