#!/bin/bash
# HF Jobs TTS eval — kale-tts-1 backend (fixed voice → generate → transcribe → UTMOS → score; no SIM).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash kale-tts-1/submit_jobs.sh ["KaleComputer/kale-tts-1-nano kenzie" ...]
# Smoke test: MAX_EVAL_SAMPLES=8 bash kale-tts-1/submit_jobs.sh
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 environment: no Docker Space. GEN_IMAGE and gen_setup live in job_image.sh; shared with
# submit_ttfa_jobs.sh so the TTFA probe runs the same install.
source "${BACKEND_DIR}/job_image.sh"
# ONNX Runtime intra-op threads (run_eval.py's --threads). 1: a fixed count keeps RTFx independent
# of the h200 host's core count (same convention as sanotts / tiny-voca / pivo).
KALE_TTS_THREADS="${KALE_TTS_THREADS:-1}"
GEN_TIMEOUT="${GEN_TIMEOUT:-2h}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"       # no SIM (fixed voice)
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Voice → output suffix ────────────────────────────────────────────────────
# One leaderboard row per model repo, with the model's default voice (Kenzie). Other voices are
# variants (MODE_SUFFIX _<voice>), not run by default.
resolve_mode() {
    local _ VOICE
    read -r _ VOICE <<< "${MODEL_CFG}"
    CLONE="false"; VOICE_CLONE_FLAG=""
    VOICE="${VOICE:-kenzie}"
    if [[ "${VOICE,,}" == "kenzie" ]]; then MODE_SUFFIX=""; else MODE_SUFFIX="_${VOICE,,}"; fi
    true
}

# English only (US and British English front end).
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
            cd /app/kale-tts-1 && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${VOICE:-kenzie} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cpu \
                --threads=${KALE_TTS_THREADS} \
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
    "KaleComputer/kale-tts-1-nano kenzie"
    "KaleComputer/kale-tts-1-tiny kenzie"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
