#!/bin/bash
# HF Jobs TTS eval — Qwen3-TTS backend (generate → transcribe → SIM* → score; *voice-clone only).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["Qwen/...-Base 32 -" ...]
# Base models always run in voice-clone mode (no built-in speaker); CustomVoice models never do.
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-qwentts}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-qwentts/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Voice cloning (Base variants): clone the reference speaker + score SIM ──
# Base checkpoints only support cloning, CustomVoice only built-in speakers, so the mode follows
# the model; VOICE_CLONE only triggers a note/warning when it disagrees.
VOICE_CLONE_MODELS=("base")                  # Base checkpoints clone; CustomVoice ones don't
model_supports_voice_clone() {
    local id_lc="${1,,}" pat
    for pat in "${VOICE_CLONE_MODELS[@]}"; do [[ "$id_lc" == *"$pat"* ]] && return 0; done
    return 1
}
# Sets RESOLVED_MODE (passed as --mode so bash and python agree on the suffix), CLONE (gates
# the SIM stage) and MODE_SUFFIX.
resolve_mode() {
    if model_supports_voice_clone "${MODEL_ID}"; then
        [[ "${VOICE_CLONE}" != "true" ]] && echo "Note: ${MODEL_ID} is a Base model and only supports voice cloning; running with voice_clone mode regardless of VOICE_CLONE."
        VOICE_CLONE=true voice_clone_mode; RESOLVED_MODE="voice_clone"
    else
        [[ "${VOICE_CLONE}" == "true" ]] && echo "Warning: VOICE_CLONE=true but ${MODEL_ID} does not support cloning; using a built-in speaker (skips SIM)."
        VOICE_CLONE=false voice_clone_mode; RESOLVED_MODE="custom_voice"
    fi
}

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size speaker". ──
generate_stage() {
    local _ BATCH_SIZE SPEAKER SPEAKER_ARG=""
    read -r _ BATCH_SIZE SPEAKER <<< "${MODEL_CFG}"
    # Base configs carry speaker "-": don't pass --speaker for those.
    [[ -n "${SPEAKER}" && "${SPEAKER}" != "-" ]] && SPEAKER_ARG="--speaker=${SPEAKER}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/qwen3tts && python run_eval.py \
                --model_id=${MODEL_ID} \
                --mode=${RESOLVED_MODE} \
                ${SPEAKER_ARG} \
                --language=${ASR_LANG} \
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
# Must match run_eval.py's LANGUAGE_CODE_TO_NAME: unmapped codes would reach the model verbatim.
QWEN_TTS_LANGUAGES="zh en ja ko de fr ru pt es it"
supports_language() { [[ " ${QWEN_TTS_LANGUAGES} " == *" $1 "* ]]; }

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment to pick) ────────
DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot zh zh ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ja ja ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ko ko ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot de de ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot fr fr ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot it it ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ru ru ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
)


# ── Models: "model_id batch_size speaker" (CLI args override) ─────────────────
# speaker is the built-in voice for CustomVoice models; "-" for Base models (which clone).
MODEL_CONFIGS=(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice 32 Aiden"
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base 32 -"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
