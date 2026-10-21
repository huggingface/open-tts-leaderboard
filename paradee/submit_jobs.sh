#!/bin/bash
# HF Jobs TTS eval — Paradee backend (fixed voice → 2 stages: generate → transcribe → score).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["sahilmahendrakar/Paradee-8M-v1.0" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-paradee}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-paradee/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"       # no SIM (fixed voice)
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Device / threads ─────────────────────────────────────────────────────────
# The package ships a 1-thread CPU session; run_eval.py swaps in the CUDA EP, which is ~3x faster
# (see paradee/run_eval.py). PARADEE_DEVICE=cpu benchmarks the package's own CPU session instead.
PARADEE_DEVICE="${PARADEE_DEVICE:-cuda:0}"
PARADEE_THREADS="${PARADEE_THREADS:-1}"

# English only (one voice, misaki American English G2P).
supports_language() { [[ "$1" == "en" ]]; }

# ── Stage 1: TTS generation (no batched API — one sample at a time) ──
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/paradee && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=${PARADEE_DEVICE} \
                --threads=${PARADEE_THREADS} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
)

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=("sahilmahendrakar/Paradee-8M-v1.0")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
