#!/bin/bash
# HF Jobs TTS eval — Underdog Woof backend (fixed voice → generate → transcribe → utmos → score; no SIM).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["ConwayResearch/Underdog-Woof juno" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-woof}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-woof/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"    # no SIM (fixed voice)
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Threads ──────────────────────────────────────────────────────────────────
# Woof is numpy-only CPU inference, so on h200 it runs on the host CPU. BLAS threads cap: RTFx
# saturates at ~8 (AMD EPYC 7742, Seed-TTS en: 1 → 3.5x, 4 → 6.1x, 8 → 6.7x, 16 → 7.3x), and 8 matches
# the cpu-upgrade flavor the CPU TTFA runs on. The model card benchmarks 1 thread (WOOF_THREADS=1).
WOOF_THREADS="${WOOF_THREADS:-8}"

# ── Voice → output suffix ────────────────────────────────────────────────────
# Both voices share the model id, so a non-default voice is a variant: MODE_SUFFIX must match the
# `_<voice>` suffix run_eval.py writes (publish with `open_results_pr.py --variant _mira`).
resolve_mode() {
    local _ VOICE
    read -r _ VOICE <<< "${MODEL_CFG}"
    CLONE="false"; VOICE_CLONE_FLAG=""
    MODE_SUFFIX=""; [[ -n "${VOICE}" && "${VOICE}" != juno ]] && MODE_SUFFIX="_${VOICE}"
    true
}

# English only (two English voices, English text normalizer).
supports_language() { [[ "$1" == "en" ]]; }

# ── Stage 1: TTS generation. MODEL_CFG = "model_id voice" (no batched API — one sample at a time) ──
generate_stage() {
    local _ VOICE
    read -r _ VOICE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/woof && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${VOICE:-juno} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cpu \
                --threads=${WOOF_THREADS} \
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
# Juno is the package default (and the faster voice); Mira is the model card's best-WER voice.
MODEL_CONFIGS=("ConwayResearch/Underdog-Woof juno")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
