#!/bin/bash
# HF Jobs TTS eval — Transformers backend (generate → transcribe → SIM* → score; *voice-clone only).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["microsoft/speecht5_tts 32 " ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-${SPACE}}"
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Voice cloning (dia / higgs / vibevoice families): clone the reference speaker + score SIM ──
# VOICE_CLONE=true requests cloning; it is only applied to clone-capable models.
VOICE_CLONE_MODELS=("dia" "higgs" "vibevoice")   # keep in sync with run_eval.py's VOICE_CLONE_FAMILIES
model_supports_voice_clone() {
    local id_lc="${1,,}" pat
    for pat in "${VOICE_CLONE_MODELS[@]}"; do [[ "$id_lc" == *"$pat"* ]] && return 0; done
    return 1
}
# Effective clone mode = requested AND supported (mirrors run_eval.py). run_pipeline uses CLONE to gate SIM.
resolve_mode() {
    if [[ "${VOICE_CLONE}" == "true" ]] && ! model_supports_voice_clone "${MODEL_ID}"; then
        echo "Warning: VOICE_CLONE=true but ${MODEL_ID} does not support cloning; using default voice (skips SIM)."
        VOICE_CLONE=false voice_clone_mode
    else
        voice_clone_mode
    fi
}

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size max_new_tokens" (max_new_tokens optional). ──
generate_stage() {
    local _ BATCH_SIZE MAX_NEW_TOKENS MAX_NEW_TOKENS_ARG=""
    read -r _ BATCH_SIZE MAX_NEW_TOKENS <<< "${MODEL_CFG}"
    [[ -n "${MAX_NEW_TOKENS}" ]] && MAX_NEW_TOKENS_ARG="--max_new_tokens=${MAX_NEW_TOKENS}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/transformers && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --language=${ASR_LANG} \
                --device=cuda:0 \
                --batch_size=${BATCH_SIZE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${VOICE_CLONE_FLAG} \
                ${MAX_NEW_TOKENS_ARG} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Languages each model can speak (by model id). run_pipeline skips every DATASET_CONFIGS combo whose
# language the model being run does not support, so the one list below serves every model.
TRANSFORMERS_DEFAULT_LANGUAGES="en"   # any model not listed
declare -A TRANSFORMERS_LANGUAGES=(
    ["vibevoice/VibeVoice-1.5B-hf"]="en zh"
    ["vibevoice/VibeVoice-7B-hf"]="en zh"
    ["bosonai/higgs-tts-2-3b-base"]="en zh de ko"
    # The 10 speech-output languages of the model card.
    ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]="en zh fr de ru it es pt ja ko"
    # run_eval.py's SEAMLESS_LANGUAGE_CODES (speech output) that some dataset below has.
    ["facebook/seamless-m4t-v2-large"]="en zh ja ko fr de es it pt ru nl pl cs ro fi tr ar hi id th vi"
)
supports_language() {
    [[ " ${TRANSFORMERS_LANGUAGES[${MODEL_ID}]:-${TRANSFORMERS_DEFAULT_LANGUAGES}} " == *" $1 "* ]]
}

# ── Datasets: "config split asr_language [dataset_path]" — every split; supports_language picks per model ──
DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH}"
    "zero_shot zh zh ${CV3_EVAL_PATH}"
    "zero_shot ja ja ${CV3_EVAL_PATH}"
    "zero_shot ko ko ${CV3_EVAL_PATH}"
    "zero_shot fr fr ${CV3_EVAL_PATH}"
    "zero_shot de de ${CV3_EVAL_PATH}"
    "zero_shot es es ${CV3_EVAL_PATH}"
    "zero_shot it it ${CV3_EVAL_PATH}"
    "zero_shot ru ru ${CV3_EVAL_PATH}"
    # MiniMax multilingual (100 sentences/language). No `uk`: Qwen3-ASR has no Ukrainian.
    "tts en en ${MINIMAX_EVAL_PATH}"
    "tts zh zh ${MINIMAX_EVAL_PATH}"
    "tts yue yue ${MINIMAX_EVAL_PATH}"
    "tts ja ja ${MINIMAX_EVAL_PATH}"
    "tts ko ko ${MINIMAX_EVAL_PATH}"
    "tts fr fr ${MINIMAX_EVAL_PATH}"
    "tts de de ${MINIMAX_EVAL_PATH}"
    "tts es es ${MINIMAX_EVAL_PATH}"
    "tts it it ${MINIMAX_EVAL_PATH}"
    "tts pt pt ${MINIMAX_EVAL_PATH}"
    "tts ru ru ${MINIMAX_EVAL_PATH}"
    "tts nl nl ${MINIMAX_EVAL_PATH}"
    "tts pl pl ${MINIMAX_EVAL_PATH}"
    "tts cs cs ${MINIMAX_EVAL_PATH}"
    "tts ro ro ${MINIMAX_EVAL_PATH}"
    "tts el el ${MINIMAX_EVAL_PATH}"
    "tts fi fi ${MINIMAX_EVAL_PATH}"
    "tts tr tr ${MINIMAX_EVAL_PATH}"
    "tts ar ar ${MINIMAX_EVAL_PATH}"
    "tts hi hi ${MINIMAX_EVAL_PATH}"
    "tts id id ${MINIMAX_EVAL_PATH}"
    "tts th th ${MINIMAX_EVAL_PATH}"
    "tts vi vi ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id batch_size max_new_tokens" (max_new_tokens optional; CLI args override) ──
MODEL_CONFIGS=(
    # "microsoft/speecht5_tts 32 "
    # "Qwen/Qwen3-Omni-30B-A3B-Instruct 32 "
    # "Qwen/Qwen2.5-Omni-3B 32 "
    # "Qwen/Qwen2.5-Omni-7B 32 "
    # "sesame/csm-1b 32 2048"                   # 2048 is predefined max length
    # "suno/bark 32 "
    # "suno/bark-small 32 "
    # "facebook/seamless-m4t-v2-large 32 "     # batched; rows are trimmed with waveform_lengths (pad units buzz)
    # "facebook/mms-tts-eng 32 "
    # "espnet/fastspeech2_conformer_with_hifigan 32 "
    ## -- SUPPORT VOICE CLONING (set VOICE_CLONE=true)
    # "bosonai/higgs-tts-2-3b-base 32 "
    # "nari-labs/Dia-1.6B-0626 32 3072"         # 3072 is predefined max length
    # # VibeVoice: 512 caps audio to ~30s to prevent runaway generation
    # "vibevoice/VibeVoice-1.5B-hf 32 512"
    # "vibevoice/VibeVoice-7B-hf 32 512"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
