#!/bin/bash
# HF Jobs TTS eval: indic-tiny-tts backend (Hindi, fixed voice -> 2 stages: generate -> transcribe -> score).
# Runs locally; submits HF Jobs. Usage:
#   HF_TOKEN=hf_... RESULTS_BUCKET=YOUR_USERNAME/bucket [MODEL_REVISION=<full 40-char sha>] \
#     bash indic-tiny-tts/submit_jobs.sh ["sulabhkatiyar/indic-tiny-tts-hindi"]
# Smoke test: add  MAX_EVAL_SAMPLES=8 ONLY_LANGS=hi ONLY_DATASETS=minimax_eval
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
# Works on Linux and macOS bash 3.2 (no associative arrays).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 environment: no Docker Space needed. The job runs on the public PyTorch image below and
# pip-installs the few eval-loop deps at job start (seconds). indic_tiny_tts itself is NOT pip-installed:
# run_eval.py downloads the pinned model repo and imports the package it ships (its deps: torch, numpy,
# safetensors, huggingface_hub).
GEN_IMAGE="${GEN_IMAGE:-pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime}"
GEN_PIP_DEPS="${GEN_PIP_DEPS:-datasets soundfile tqdm huggingface_hub safetensors numpy}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability: don't override.
GEN_TIMEOUT="${GEN_TIMEOUT:-30m}"                # 100 sentences of a 22M model take minutes; a hang must not burn H200 hours
DEFAULT_STAGES="generate transcribe"             # fixed voice, Hindi: no SIM (no cloning), no UTMOS (English only)
RUN_EVAL_INJECT="$(inject_run_eval)"

# The model revision is pinned: a full 40-char commit sha of the HF model repo. MODEL_REVISION=main only
# with ALLOW_UNPINNED_REVISION=1 (smoke tests); the sha is recorded in the PR.
# Submitted revision: package version 0.1.0 of sulabhkatiyar/indic-tiny-tts-hindi (the public release).
MODEL_REVISION="${MODEL_REVISION:-e15ac436560eb24d115d47c0783f62f10a02d30a}"
if [[ ! "${MODEL_REVISION}" =~ ^[0-9a-f]{40}$ ]] && [[ "${ALLOW_UNPINNED_REVISION:-0}" != "1" ]]; then
    echo "indic-tiny-tts: set MODEL_REVISION to the full 40-char commit sha of the model repo (or ALLOW_UNPINNED_REVISION=1 for a throwaway run)." >&2
    exit 1
fi
# Built-in voice: 'female' or 'male'. One voice per run, one leaderboard row.
VOICE="${VOICE:-male}"
echo "indic-tiny-tts: revision=${MODEL_REVISION} voice=${VOICE} results bucket=${RESULTS_BUCKET}"

# Single-language (Hindi) model: skip every other split instead of burning a GPU job on garbage audio.
supports_language() { [[ "$1" == "hi" ]]; }

# ── Stage 1: TTS generation (the model synthesizes one sample at a time: no batch size) ──
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${GEN_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            pip install --no-cache-dir -q ${GEN_PIP_DEPS} &&
            cd /app/indic-tiny-tts && python run_eval.py \
                --model_id=${MODEL_ID} \
                --revision=${MODEL_REVISION} \
                --voice=${VOICE} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]". Hindi exists only in MiniMax (100 sentences). ──
DATASET_CONFIGS=(
    "tts hi hi ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=("sulabhkatiyar/indic-tiny-tts-hindi")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
