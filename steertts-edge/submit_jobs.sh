#!/bin/bash
# HF Jobs TTS eval — SteerTTS edge backend (generate → transcribe → utmos → score).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash steertts-edge/submit_jobs.sh ["WhissleAI/steer-tts-edge" ...]
# English only: SteerTTS has an English (ARPAbet) front end, so only the `en` splits are configured.
# Model card / usage: https://huggingface.co/WhissleAI/steer-tts-edge (12 MB int4 build of SteerTTS; same image as steertts/)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-steertts}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-steertts/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"
SUPPORTS_VOICE_CLONE=false
STEERTTS_REVISION="${STEERTTS_REVISION:-v2.0.0}"     # pinned release of the model repo
RUN_EVAL_INJECT="$(inject_run_eval)"
supports_language() { [[ "$1" == "en" ]]; }

# ── Stage 1: TTS generation (one sample at a time — no batched API) ──────────
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/steertts-edge && python run_eval.py \
                --model_id=${MODEL_ID} \
                --revision=${STEERTTS_REVISION} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" ─────────────────────
DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH}"
    # MiniMax multilingual (100 sentences/language): English only for SteerTTS.
    "tts en en ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=("WhissleAI/steer-tts-edge")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
