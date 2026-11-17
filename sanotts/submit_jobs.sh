#!/bin/bash
# HF Jobs TTS eval — sanoTTS backend (fixed voice → generate → transcribe → utmos → score; no SIM).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash sanotts/submit_jobs.sh ["ampixa/sanoTTS heart" ...]
# Smoke test: MAX_EVAL_SAMPLES=8 ONLY_LANGS=en bash sanotts/submit_jobs.sh
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 environment: no Docker Space needed. GEN_IMAGE (public python:3.12-slim) and gen_setup (the
# pip install at job start) live in job_image.sh, shared with submit_ttfa_jobs.sh so the TTFA probe
# runs the same install.
source "${BACKEND_DIR}/job_image.sh"
# NumPy BLAS threads (threadpoolctl in run_eval.py). 1: the matrices are tiny, and a fixed count keeps
# RTFx independent of the h200 host's core count. The TTFA targets pass the same value.
SANOTTS_THREADS="${SANOTTS_THREADS:-1}"
GEN_TIMEOUT="${GEN_TIMEOUT:-2h}"                 # a 0.3-2.3M model on the largest split takes minutes; a hang must not burn H200 hours
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"       # no SIM (fixed voice)
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Voice → output suffix ────────────────────────────────────────────────────
# All voices live in one model repo, so a non-default voice is a variant: MODE_SUFFIX must match the
# suffix run_eval.py writes (publish with `open_results_pr.py --variant amy` / `--variant heartnano`).
resolve_mode() {
    local _ VOICE
    read -r _ VOICE <<< "${MODEL_CFG}"
    CLONE="false"; VOICE_CLONE_FLAG=""
    case "${VOICE:-heart}" in
        heart) MODE_SUFFIX="" ;;
        amy) MODE_SUFFIX="_amy" ;;
        heart-nano) MODE_SUFFIX="_heartnano" ;;
        *) echo "sanotts: unknown voice '${VOICE}' (heart, amy, heart-nano)" >&2; exit 1 ;;
    esac
    true
}

# English only: the three evaluated voices are English, with the package's English front end.
supports_language() { [[ "$1" == "en" ]]; }

# ── Stage 1: TTS generation. MODEL_CFG = "model_id voice" (no batched API — one sample at a time) ──
# NumPy on the host CPU of the h200 flavor; --device=cpu (run_eval.py refuses a GPU device).
generate_stage() {
    local _ VOICE
    read -r _ VOICE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${GEN_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            $(gen_setup cpu)
            cd /app/sanotts && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${VOICE:-heart} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cpu \
                --threads=${SANOTTS_THREADS} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "tts en en ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id voice" (CLI args override) ─────────────────────────────
# heart is the default row; amy and heart-nano are the two other English voices (variants).
MODEL_CONFIGS=("ampixa/sanoTTS heart" "ampixa/sanoTTS amy" "ampixa/sanoTTS heart-nano")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
