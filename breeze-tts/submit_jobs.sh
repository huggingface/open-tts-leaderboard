#!/bin/bash
# HF Jobs TTS eval — Breeze TTS 2 backend (generate → transcribe → SIM* → score; *voice-clone only).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["BreezeBlue/Breeze-TTS-2" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-breeze}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-breeze/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# CUDA-graph fast path (vendor-supported; warmup excluded from timings). FAST_ALL=false for eager.
FAST_ALL="${FAST_ALL:-true}"

# ── Languages this backend can select ────────────────────────────────────────
# Breeze TTS 2 is English + Chinese only; other CV3-Eval languages are skipped.
BREEZE_LANGUAGES="en zh"
supports_language() { [[ " ${BREEZE_LANGUAGES} " == *" $1 "* ]]; }

# ── Voice cloning ────────────────────────────────────────────────────────────
# Breeze has no default voice: VOICE_CLONE=false uses reference-free voice design (no SIM).
# Standard clone mode + the fast-path flag.
resolve_mode() {
    voice_clone_mode
    if [[ "${FAST_ALL}" == "true" ]]; then FAST_FLAG="--fast_all"; else FAST_FLAG="--no-fast_all"; fi
}

# ── Stage 1: TTS generation. MODEL_CFG = "model_id" (Breeze synthesizes one sample at a time). ──
# The weights are not baked into the image (licence), so run_eval.py downloads them at job start.
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/breeze-tts && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --language=${ASR_LANG} \
                --device=cuda:0 \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${VOICE_CLONE_FLAG} \
                ${FAST_FLAG} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot zh zh ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
)

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=("BreezeBlue/Breeze-TTS-2")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
