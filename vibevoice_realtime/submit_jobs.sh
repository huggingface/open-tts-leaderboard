#!/bin/bash
# HF Jobs TTS eval — VibeVoice-Realtime backend (fixed voice: generate → transcribe → score).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["microsoft/VibeVoice-Realtime-0.5B 32" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-vibevoice}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-vibevoice/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"               # no SIM (fixed voice)
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Language → voice preset ──────────────────────────────────────────────────
declare -A VIBEVOICE_VOICES=( [en]=en-Carter_man [de]=de-Spk0_man  [fr]=fr-Spk0_man [it]=it-Spk1_man \
                              [ja]=jp-Spk0_man  [ko]=kr-Spk1_man  [nl]=nl-Spk0_man [pl]=pl-Spk0_man \
                              [pt]=pt-Spk1_man  [es]=sp-Spk1_man )
supports_language() { [[ -n "${VIBEVOICE_VOICES[$1]:-}" ]]; }
# Setting VIBEVOICE_SPEAKER forces one preset for every combo — only sensible for a single-language run.
VIBEVOICE_SPEAKER="${VIBEVOICE_SPEAKER:-}"

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size". ──
# The streaming API asserts batch_size == 1; --batch_size only chunks manifest writing / resume.
generate_stage() {
    local _ BATCH_SIZE SPEAKER
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    SPEAKER="${VIBEVOICE_SPEAKER:-${VIBEVOICE_VOICES[${ASR_LANG}]}}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/vibevoice_realtime && python run_eval.py \
                --model_id=${MODEL_ID} \
                --speaker=${SPEAKER} \
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

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot fr fr ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot ja ja ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot ko ko ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot de de ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot it it ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
)

# ── Models: "model_id batch_size" (CLI args override; speaker comes from VIBEVOICE_VOICES) ──
MODEL_CONFIGS=(
    "microsoft/VibeVoice-Realtime-0.5B 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
