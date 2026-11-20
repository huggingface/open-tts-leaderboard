#!/bin/bash
# HF Jobs TTS eval: Reverb-TTS backend (generate -> transcribe -> SIM (voice clone only) -> UTMOS -> score).
# Runs locally; submits HF Jobs. Usage:
#   HF_TOKEN=hf_... RESULTS_BUCKET=YOUR_NAMESPACE/bucket DATASET_NAMESPACE=YOUR_NAMESPACE [VOICE_CLONE=true] \
#     bash reverb/submit_jobs.sh ["software-mansion/Reverb-TTS 32"]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 environment: no Docker Space. job_image.sh (shared with submit_ttfa_jobs.sh) gives the public PyTorch image
# and the reverb-tts install run at job start.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/job_image.sh"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability: don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# VOICE_CLONE=true: zero-shot voice cloning + SIM; false (default) = the preset voice run_eval.py picks for each
# split's language.
SUPPORTS_VOICE_CLONE=true

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size". ──
generate_stage() {
    local _ BATCH_SIZE
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${GEN_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            $(gen_setup cuda)
            cd /app/reverb && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --language=${ASR_LANG} \
                --device=cuda:0 \
                --batch_size=${BATCH_SIZE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${VOICE_CLONE_FLAG} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Languages Reverb speaks (model card) ──
# run_eval.py passes the split language to Reverb and picks that language's preset voice.
SUPPORTED_LANGUAGES="en de fr es it pl pt"

# ── Models: "model_id batch_size" (CLI args override) ────────────────────────
MODEL_CONFIGS=(
    "software-mansion/Reverb-TTS 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
