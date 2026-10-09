#!/bin/bash
# HF Jobs TTS eval — LFM2.5-Audio backend (fixed voice → 2 stages: generate → transcribe → score).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["LiquidAI/LFM2.5-Audio-1.5B" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-lfm2-audio}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-lfm2-audio/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"                 # no SIM (fixed voice)
RUN_EVAL_INJECT="$(inject_run_eval)"

# English only (model card), with four built-in voices: us_female, us_male, uk_female, uk_male.
supports_language() { [[ "$1" == "en" ]]; }
LFM_VOICE="${LFM_VOICE:-us_female}"

# ── Stage 1: TTS generation (generate_sequential is batch-1 only — no batch size) ──
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/lfm2-audio && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${LFM_VOICE} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
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
MODEL_CONFIGS=("LiquidAI/LFM2.5-Audio-1.5B")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
