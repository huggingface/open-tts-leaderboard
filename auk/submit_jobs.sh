#!/bin/bash
# HF Jobs TTS eval — AuK backend (generate → transcribe → SIM* → score; *voice-clone only).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["tencent/AuK 16" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-auk}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-auk/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Voice mode ───────────────────────────────────────────────────────────────
# A reference audio is used either way; what the toggle changes is WHICH reference:
#   VOICE_CLONE=true   each sample clones its own prompt_audio -> SIM stage runs.
#   VOICE_CLONE=false  (default) ONE fixed reference for the whole split (the split's sample 0,
#                      language-matched); SIM is skipped.
#   INSTRUCT=true      (with VOICE_CLONE=false) diagnostic reference-free Instruct TTS; SIM is skipped.
INSTRUCT="${INSTRUCT:-false}"
VOICE_DESCRIPTION="${VOICE_DESCRIPTION:-}"   # INSTRUCT only; "" = run_eval.py's neutral default
FIXED_PROMPT_INDEX="${FIXED_PROMPT_INDEX:-}" # fixed-voice only; "" = run_eval.py's default (0)
resolve_mode() {
    voice_clone_mode
    if [[ "${CLONE}" != "true" && "${INSTRUCT}" == "true" ]]; then
        MODE_SUFFIX="_instruct"; VOICE_CLONE_FLAG="--no-voice_clone --instruct"
    fi
}

# ── Languages AuK is documented for ─────────────────────────────────────────
AUK_LANGUAGES="${AUK_LANGUAGES:-en zh}"
supports_language() { [[ " ${AUK_LANGUAGES} " == *" $1 "* ]]; }

# ── Stage 1: TTS generation. MODEL_CFG = "model_id [batch_size] [duration_scale]". ──
MAX_ATTN_COST="${MAX_ATTN_COST:-0}"
generate_stage() {
    local _ BATCH_SIZE DURATION_SCALE DESC_ARG="" FIXED_ARG=""
    read -r _ BATCH_SIZE DURATION_SCALE <<< "${MODEL_CFG}"
    BATCH_SIZE="${BATCH_SIZE:-16}"          # tolerate a bare "model_id" with no extra fields
    DURATION_SCALE="${DURATION_SCALE:-1.0}"
    # Each flag is rejected outright by run_eval.py in the wrong mode, so only pass it in its own.
    [[ "${INSTRUCT}" == "true" && -n "${VOICE_DESCRIPTION}" ]] && DESC_ARG=" --voice_description='${VOICE_DESCRIPTION}'"
    [[ "${VOICE_CLONE}" != "true" && "${INSTRUCT}" != "true" && -n "${FIXED_PROMPT_INDEX}" ]] \
        && FIXED_ARG=" --fixed_prompt_index=${FIXED_PROMPT_INDEX}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN \
        --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/auk && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --batch_size=${BATCH_SIZE} \
                --max_attn_cost=${MAX_ATTN_COST} \
                --duration_scale=${DURATION_SCALE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${VOICE_CLONE_FLAG}${DESC_ARG}${FIXED_ARG} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
# Seed-TTS zero-shot TTS splits: en (1088), zh (2020), zh_hard (400).
DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot zh zh ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    # MiniMax multilingual (100 sentences/language): the languages the model supports. No `uk`: Qwen3-ASR has no Ukrainian.
    "tts en en ${MINIMAX_EVAL_PATH}"
    "tts zh zh ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id [batch_size] [duration_scale]" (CLI args override) ─────
MODEL_CONFIGS=(
    "tencent/AuK 16 1.0"
    "tencent/AuK-Flash 16 1.0"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
