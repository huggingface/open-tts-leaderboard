#!/bin/bash
# HF Jobs TTS eval — Supertonic backend (fixed voice: generate → transcribe → score; no SIM).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["Supertone/supertonic-3 32 M1" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-super}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-super/blob/main/Dockerfile)
DEFAULT_STAGES="generate transcribe utmos"           # no SIM (fixed voice)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size voice". ──
generate_stage() {
    local _ BATCH_SIZE VOICE
    read -r _ BATCH_SIZE VOICE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/supertonic && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${VOICE} \
                --lang=${ASR_LANG} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --batch_size=${BATCH_SIZE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Languages this backend can select ────────────────────────────────────────
# The language codes from the model card's front-matter. No Chinese, so zh splits are skipped
# (run_eval.py's --lang 'na' = language-agnostic is a fallback, not Chinese support).
SUPERTONIC_LANGUAGES="en ko ja ar bg cs da de el es et fi fr hi hr hu id it lt lv nl pl pt ro ru sk sl sv tr uk vi"
supports_language() { [[ " ${SUPERTONIC_LANGUAGES} " == *" $1 "* ]]; }

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot ja ja ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot ko ko ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot de de ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot fr fr ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot it it ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot ru ru ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
)

# ── Models: "model_id batch_size voice" (CLI args override) ──────────────────
MODEL_CONFIGS=("Supertone/supertonic-3 32 M1")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
