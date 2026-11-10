#!/bin/bash
# HF Jobs TTS eval — PiVo-1.4M backend (fixed voice → generate → transcribe → utmos → score; no SIM).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash pivo/submit_jobs.sh ["empero-ai/PiVo-1.4M Pi" ...]
# Smoke test: MAX_EVAL_SAMPLES=8 ONLY_DATASETS=cv3_eval bash pivo/submit_jobs.sh
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-pivo}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-pivo/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"    # no SIM (fixed voice)
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Threads ──────────────────────────────────────────────────────────────────
# PiVo is ONNX Runtime CPU inference, so on h200 it runs on the host CPU. One intra-op thread is the
# model card's setting and also the fastest: the vocoder runs one tiny step per 2.5 ms subframe, so
# more threads only add overhead (AMD EPYC 7742, MiniMax en: 1 → 11.9x, 2/4/8 → 10.9x).
PIVO_THREADS="${PIVO_THREADS:-1}"

# ── Voice → output suffix ────────────────────────────────────────────────────
# Both voices share the model id, so a non-default voice is a variant: MODE_SUFFIX must match the
# `_<voice>` suffix run_eval.py writes (publish with `open_results_pr.py --variant _vo`).
resolve_mode() {
    local _ VOICE
    read -r _ VOICE <<< "${MODEL_CFG}"
    CLONE="false"; VOICE_CLONE_FLAG=""
    MODE_SUFFIX=""; [[ "${VOICE}" == Vo ]] && MODE_SUFFIX="_vo"
    true
}

# English only (two English voices, CMU-dictionary frontend).
supports_language() { [[ "$1" == "en" ]]; }

# ── Stage 1: TTS generation. MODEL_CFG = "model_id voice" (no batched API — one sample at a time) ──
# The image bakes the model at the revision pinned in run_eval.py (PIVO_MODEL_DIR=/opt/pivo).
generate_stage() {
    local _ VOICE
    read -r _ VOICE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 2h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/pivo && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${VOICE:-Pi} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cpu \
                --threads=${PIVO_THREADS} \
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
# Pi is the package default and the leaderboard row; Vo is the second built-in voice.
MODEL_CONFIGS=("empero-ai/PiVo-1.4M Pi")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
