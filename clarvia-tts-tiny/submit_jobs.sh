#!/bin/bash
# HF Jobs TTS eval — clarvia-tts-tiny backend (fixed voice → generate → transcribe → UTMOS → score;
# no SIM). Turkish. Runs locally; submits HF Jobs.
# Usage: HF_TOKEN=hf_... bash clarvia-tts-tiny/submit_jobs.sh ["wite-tech/clarvia-tts-tiny female_1" ...]
# Smoke test: MAX_EVAL_SAMPLES=8 bash clarvia-tts-tiny/submit_jobs.sh
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 environment: no Docker Space. GEN_IMAGE and gen_setup live in job_image.sh.
source "${BACKEND_DIR}/job_image.sh"
# onnxruntime intra-op threads (run_eval.py's --threads). 1: clarvia's matrices are small, and a
# fixed count keeps RTFx independent of the h200 host's core count.
CLARVIA_THREADS="${CLARVIA_THREADS:-1}"
GEN_TIMEOUT="${GEN_TIMEOUT:-2h}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"       # no SIM (fixed voice); UTMOS auto-skips non-en
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Voice → output suffix ────────────────────────────────────────────────────
# All voices share one repo, so a non-default voice is a leaderboard variant: MODE_SUFFIX must match
# the suffix run_eval.py writes (publish with `open_results_pr.py --variant male1`).
resolve_mode() {
    local _ VOICE
    read -r _ VOICE <<< "${MODEL_CFG}"
    CLONE="false"; VOICE_CLONE_FLAG=""
    case "${VOICE:-female_1}" in
        female_1) MODE_SUFFIX="" ;;
        male_1)   MODE_SUFFIX="_male1" ;;
        *) echo "clarvia-tts-tiny: unknown voice '${VOICE}' (female_1, male_1)" >&2; exit 1 ;;
    esac
    true
}

# Turkish only (Turkish phonemizer).
SUPPORTED_LANGUAGES="tr"

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
            cd /app/clarvia-tts-tiny && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${VOICE:-female_1} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cpu \
                --threads=${CLARVIA_THREADS} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Models: "model_id voice" (CLI args override) ────────────────────────────
MODEL_CONFIGS=(
    "wite-tech/clarvia-tts-tiny female_1"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
