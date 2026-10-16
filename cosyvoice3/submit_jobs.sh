#!/bin/bash
# HF Jobs TTS eval — Fun-CosyVoice3 backend (generate → transcribe → SIM* → score; *voice-clone only).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["FunAudioLLM/Fun-CosyVoice3-0.5B-2512 32" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-cosy}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-cosy/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Voice mode ───────────────────────────────────────────────────────────────
# A reference audio is mandatory either way (no built-in speaker); the toggle picks WHICH one:
#   VOICE_CLONE=true   each sample clones its own prompt_audio -> SIM stage runs.
#   VOICE_CLONE=false  (default) ONE fixed reference for the whole split (the split's sample 0,
#                      language-matched); SIM is skipped.

# The base and RL LMs share one HF model id, so the RL run gets its own manifest suffix (matching
# run_eval.py). MODEL_CFG's optional 3rd field selects the checkpoint.
resolve_mode() {
    local _ _bs LLM VARIANT
    read -r _ _bs LLM <<< "${MODEL_CFG}"
    LLM="${LLM:-llm.pt}"
    VARIANT=""; [[ "${LLM}" != "llm.pt" ]] && VARIANT="_rl"
    voice_clone_mode "${VARIANT}"
}

# ── Languages Fun-CosyVoice3 covers (model card: 9 languages) ────────────────
# Matched against asr_language (so `zh_hard` / `hard_zh` match `zh`).
COSYVOICE3_LANGUAGES=" zh en ja ko de es fr it ru "
supports_language() { [[ "${COSYVOICE3_LANGUAGES}" == *" $1 "* ]]; }

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size [llm_checkpoint]". ──
# No batched API (inference_zero_shot takes one string); --batch_size only chunks manifest
# writing / resume.
generate_stage() {
    local _ BATCH_SIZE LLM
    read -r _ BATCH_SIZE LLM <<< "${MODEL_CFG}"
    LLM="${LLM:-llm.pt}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/cosyvoice3 && python run_eval.py \
                --model_id=${MODEL_ID} \
                --llm_checkpoint=${LLM} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --batch_size=${BATCH_SIZE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${VOICE_CLONE_FLAG} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot de de ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ko ko ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ru ru ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ja ja ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot en en ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot zh zh ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot fr fr ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot it it ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
)

# ── Models: "model_id batch_size [llm_checkpoint]" (CLI args override) ───────
# The base (llm.pt) and RL (llm.rl.pt) checkpoints write separate manifests, so both can coexist.
MODEL_CONFIGS=(
    # "FunAudioLLM/Fun-CosyVoice3-0.5B-2512 32"
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512 32 llm.rl.pt"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
