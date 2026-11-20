#!/bin/bash
# HF Jobs TTS eval — Fish Audio S2-Pro backend (generate → transcribe → SIM* → score; *voice-clone only).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["fishaudio/s2-pro" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-fish}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-fish/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"
SUPPORTS_VOICE_CLONE=true   # VOICE_CLONE=true clones each sample's reference speaker + scores SIM

# ── Stage 1: TTS generation. MODEL_CFG = "model_id" (S2-Pro synthesizes one sample at a time). ──
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/fishaudio-s2-pro && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${VOICE_CLONE_FLAG} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# Fish Audio S2 Pro covers every ZTTS2-Eval Mini language (model card): every benchmark language but Cantonese (yue).
SUPPORTED_LANGUAGES="en zh ja ko fr de es it pt ru nl pl cs ro el fi tr ar hi id th vi fa sv da hu uk nb he bg hr sk ca ms fil"

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=("fishaudio/s2-pro")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
