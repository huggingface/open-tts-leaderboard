#!/bin/bash
# HF Jobs TTS eval — Magpie TTS backend (fixed voice → 2 stages: generate → transcribe → score).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["nvidia/magpie_tts_multilingual_357m 32 Sofia" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-magpie}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-magpie/blob/main/Dockerfile)
DEFAULT_STAGES="generate transcribe utmos"      # no SIM (fixed baked voice, no zero-shot cloning)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size speaker". ──
# Generation memory is quadratic in text length, so:
#   * MAX_ATTN_COST — run_eval.py sub-batches when n*max_text_tokens^2 exceeds the budget.
#     0 = calibrate from the first OOM; set it to the value the log printed to skip that probe.
#   * expandable_segments — allocation sizes change every batch, which otherwise fragments memory.
MAX_ATTN_COST="${MAX_ATTN_COST:-0}"
generate_stage() {
    local _ BATCH_SIZE SPEAKER
    read -r _ BATCH_SIZE SPEAKER <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/magpie-tts && python run_eval.py \
                --model_id=${MODEL_ID} \
                --speaker=${SPEAKER} \
                --language=${ASR_LANG} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --batch_size=${BATCH_SIZE} \
                --max_attn_cost=${MAX_ATTN_COST} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Languages this backend can actually select ───────────────────────────────
# An unmapped language silently falls back to the first tokenizer in run_eval.py, so gate the combos.
# These are the 9 languages the checkpoint ships a tokenizer for (no Korean).
MAGPIE_LANGUAGES="en de es fr it vi zh ja hi"
supports_language() { [[ " ${MAGPIE_LANGUAGES} " == *" $1 "* ]]; }

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
# asr_language is the language hint for both the TTS model and Qwen3-ASR, so it must be a real
# language code (`zh_hard` pairs with `zh`).
DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot zh zh ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot ja ja ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot de de ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot fr fr ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot it it ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
)

# ── Models: "model_id batch_size speaker" (CLI args override) ────────────────
# Baked speakers: John, Sofia, Aria, Jason, Leo (no zero-shot cloning).
MODEL_CONFIGS=("nvidia/magpie_tts_multilingual_357m 32 Sofia")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
