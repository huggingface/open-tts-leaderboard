#!/bin/bash
# HF Jobs TTS eval — Audio8 Realistic backend (generate → transcribe → SIM* → score; *voice-clone only).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["Edge0/Audio8-TTS-Realistic-0.6b 32" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-audio8}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-audio8/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# VOICE_CLONE=true: zero-shot voice cloning + SIM; false (default) = model's default voice.
SUPPORTS_VOICE_CLONE=true

# Model, remote code and codec are pinned to this Hub commit (run_eval.py rejects anything but a full SHA).
MODEL_REVISION="${MODEL_REVISION:-1e857568bfd4f02931aca3988ee773a44d2d26d3}"

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size". ──
# Sampling defaults live in run_eval.py: temperature 0.7 for CV3 zero_shot, 0.1 otherwise; top_p 0.8, top_k 50.
generate_stage() {
    local _ BATCH_SIZE
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 12h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/audio8 && python run_eval.py \
                --model_id=${MODEL_ID} \
                --revision=${MODEL_REVISION} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --dtype=bfloat16 \
                --batch_size=${BATCH_SIZE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${VOICE_CLONE_FLAG} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Languages Audio8 Realistic covers (model card; no cs, el, ro, uk) ──
SUPPORTED_LANGUAGES="zh en ja ko de fr es it ru ar fi hi id ms nl nb pl pt sv th tr vi yue"

# ── Models: "model_id batch_size" (CLI args override) ────────────────────────
MODEL_CONFIGS=(
    "Edge0/Audio8-TTS-Realistic-0.6b 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
