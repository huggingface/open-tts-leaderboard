#!/bin/bash
# HF Jobs TTS eval — IndexTTS-2.5 backend (generate → transcribe → SIM* → score; *voice-clone only).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["IndexTeam/IndexTTS-2.5 32" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-indextts}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-indextts/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Voice mode ───────────────────────────────────────────────────────────────
# A reference audio is mandatory either way (no built-in speaker); the toggle picks WHICH one:
#   VOICE_CLONE=true   each sample clones its own prompt_audio -> SIM stage runs.
#   VOICE_CLONE=false  (default) ONE fixed reference for the whole split (the split's sample 0,
#                      language-matched); SIM is skipped.
SUPPORTS_VOICE_CLONE=true

# ── Languages IndexTTS-2.5 covers (model card: zh, en, ja, es, ar) ───────────
# An unsupported `lang` does NOT raise upstream, it just produces wrong-language audio, so gate
# the combos here. Matched against asr_language (so `zh_hard` / `hard_zh` match `zh`).
INDEXTTS_LANGUAGES=" zh en ja es ar "
supports_language() { [[ "${INDEXTTS_LANGUAGES}" == *" $1 "* ]]; }

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size". ──
# No batched API; --batch_size only chunks manifest writing / resume. See run_eval.py.
generate_stage() {
    local _ BATCH_SIZE
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/index-tts && python run_eval.py \
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

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
# asr_language doubles as the TTS `lang` tag (--language above).
DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot zh zh ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ja ja ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
)

# ── Models: "model_id batch_size" (CLI args override) ────────────────────────
MODEL_CONFIGS=(
    "IndexTeam/IndexTTS-2.5 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
