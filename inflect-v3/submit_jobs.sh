#!/bin/bash
# HF Jobs TTS eval — Inflect v3 backend (fixed voice: generate → transcribe → UTMOS → score).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["owensong/Inflect-Mini-v3" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
#
# Covers all three Inflect v3 releases (Mini, Micro, Atom); run_eval.py detects the variant from
# the repo layout. Atom is CPU-only — its runtime sets CUDA_VISIBLE_DEVICES='' — so the job still
# runs on the h200 flavor (uniform host for RTFx) but with --device=cpu.
# Inflect v1 (inflect-nano/) and v2 (inflect-v2/) are different architectures with their own backends.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-inflect-v3}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-inflect-v3/blob/main/Dockerfile)
DEFAULT_STAGES="generate transcribe utmos"           # no SIM (fixed voice, no reference to compare)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# English-only (model card: `language: [en]`); other splits are skipped rather than read with English G2P.
supports_language() { [[ "$1" == "en" ]]; }

# ── Per-model device: Atom is CPU-only; Mini and Micro run on the h200 GPU ──
inflect_v3_device() {
    case "$1" in
        *Inflect-Atom-v3*) echo "cpu" ;;
        *)                 echo "cuda:0" ;;
    esac
}

# ── Per-model pip install at job start — Micro needs num2words/regex/addict,
# Atom needs nothing beyond the image (dsp_backend='torch' avoids numba/llvmlite). The image
# already has misaki/spaCy/espeakng-loader/torch/transformers/soundfile pinned for Mini.
inflect_v3_extra_deps() {
    case "$1" in
        *Inflect-Micro-v3*) echo "num2words==0.5.14 regex==2026.9.29 addict==2.4.0" ;;
        *)                  echo "" ;;
    esac
}

# ── Stage 1: TTS generation (one sample at a time — no batched API) ──────────
generate_stage() {
    local run_device extra
    run_device="$(inflect_v3_device "${MODEL_ID}")"
    extra="$(inflect_v3_extra_deps "${MODEL_ID}")"
    local extra_install=""
    [[ -n "${extra}" ]] && extra_install="pip install --no-cache-dir ${extra} &&"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            ${extra_install}
            cd /app/inflect-v3 && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=${run_device} \
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

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
# Mini-v3 is already published (see tts_leaderboard_results PR #133); default to the two newer
# variants that still need to be measured. Pass "owensong/Inflect-Mini-v3" explicitly to re-run it.
MODEL_CONFIGS=(
    "owensong/Inflect-Micro-v3"
    "owensong/Inflect-Atom-v3"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
