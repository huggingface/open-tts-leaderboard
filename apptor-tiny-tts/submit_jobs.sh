#!/bin/bash
# HF Jobs TTS eval — apptor-tiny-tts backend (fixed voice → generate → transcribe → UTMOS → score;
# no SIM). German. Runs locally; submits HF Jobs.
# Usage: HF_TOKEN=hf_... bash apptor-tiny-tts/submit_jobs.sh ["apptor/apptor-tiny-tts F2" ...]
# Smoke test: MAX_EVAL_SAMPLES=8 bash apptor-tiny-tts/submit_jobs.sh
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 environment: no Docker Space. GEN_IMAGE and gen_setup live in job_image.sh.
source "${BACKEND_DIR}/job_image.sh"
# onnxruntime intra-op threads (run_eval.py's --threads). 1: apptor's matrices are small, and a fixed
# count keeps RTFx independent of the h200 host's core count.
APPTOR_THREADS="${APPTOR_THREADS:-1}"
GEN_TIMEOUT="${GEN_TIMEOUT:-2h}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"       # no SIM (fixed voice); UTMOS auto-skips non-en
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
        *) echo "apptor-tiny-tts: unknown voice '${VOICE}' (F2, M2)" >&2; exit 1 ;;
    esac
    true
}

# German only (German g2p front end).
SUPPORTED_LANGUAGES="de"

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
            cd /app/apptor-tiny-tts && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${VOICE:-F2} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cpu \
                --threads=${APPTOR_THREADS} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Models: "model_id voice" (CLI args override) ────────────────────────────
MODEL_CONFIGS=(
    "apptor/apptor-tiny-tts F2"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
