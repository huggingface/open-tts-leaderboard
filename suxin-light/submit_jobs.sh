#!/bin/bash
# HF Jobs TTS eval — suxin-light backend (fixed voice → generate → transcribe → UTMOS → score; no
# SIM). Mandarin + English. Runs locally; submits HF Jobs.
# Usage: HF_TOKEN=hf_... bash suxin-light/submit_jobs.sh ["TEQUILA123/suxin-light" ...]
# Smoke test: MAX_EVAL_SAMPLES=8 ONLY_LANGS="en zh" bash suxin-light/submit_jobs.sh
# CPU fallback: SUXIN_DEVICE=cpu bash suxin-light/submit_jobs.sh
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 environment: no Docker Space. GEN_IMAGE and gen_setup live in job_image.sh; shared with
# submit_ttfa_jobs.sh so the TTFA probe runs the same install.
source "${BACKEND_DIR}/job_image.sh"
# onnxruntime intra-op threads (run_eval.py's --threads). 1 keeps CPU-side ORT work independent of
# the h200 host's core count. On CUDA the GPU EP does the convs; this still caps leftover CPU nodes.
SUXIN_LIGHT_THREADS="${SUXIN_LIGHT_THREADS:-1}"
# Leaderboard RTFx is measured on the h200; default to that GPU. SUXIN_DEVICE=cpu keeps the CPU path
# (INT8 graph + CPU EP). gen FLAVOR (h200) is set centrally — don't override.
SUXIN_DEVICE="${SUXIN_DEVICE:-cuda:0}"
# auto = fp32 ONNX on CUDA, int8-excldec on CPU. Override with fp32 or int8.
SUXIN_GRAPH="${SUXIN_GRAPH:-auto}"
GEN_TIMEOUT="${GEN_TIMEOUT:-2h}"
DEFAULT_STAGES="generate transcribe utmos"       # no SIM (fixed voice); UTMOS auto-skips non-en
RUN_EVAL_INJECT="$(inject_run_eval)"

# Mandarin + English (pypinyin + espeak-ng mixed frontend). MiniMax yue is skipped.
SUPPORTED_LANGUAGES="zh en"

echo "suxin-light: device=${SUXIN_DEVICE} graph=${SUXIN_GRAPH} threads=${SUXIN_LIGHT_THREADS} image=${GEN_IMAGE}"

# ── Stage 1: TTS generation ─────────────────────────────────────────────────
# Default: CUDA EP on the h200, FP32 graph (see run_eval.py). --batch_size=1 is required: the
# graph's batch dim pads every row to the longest audio, and the pad is synthesised speech, not
# silence (no duration output to trim).
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${GEN_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            $(gen_setup "${SUXIN_DEVICE}")
            cd /app/suxin-light && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=${SUXIN_DEVICE} \
                --graph=${SUXIN_GRAPH} \
                --threads=${SUXIN_LIGHT_THREADS} \
                --batch_size=1 \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=("TEQUILA123/suxin-light")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
