#!/bin/bash
# HF Jobs TTS eval — Voxtral TTS backend (fixed voice: generate → transcribe → score; no SIM).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["mistralai/Voxtral-4B-TTS-2603 32" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
#
# The Voxtral image is vLLM-Omni: stage 1 serves the model with `vllm serve <model> --omni` inside
# the job and run_eval.py is an HTTP client of its /v1/audio/speech endpoint (it waits for /health
# before generating). Voice cloning from prompt_audio is unsupported on the open checkpoint, so this
# backend is fixed-voice (casual_male preset) → no SIM stage.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-voxtral}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-voxtral/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"             # no SIM (fixed voice)
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Stage 1: serve with vLLM-Omni, then run the HTTP client. MODEL_CFG = "model_id batch_size". ──
generate_stage() {
    local _ BATCH_SIZE
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/voxtral-tts
            # The h200 driver is not older than the image's CUDA, so the forward-compat libcuda must
            # stay off (with it, every engine stage dies at init with 'Error 803').
            unset VLLM_ENABLE_CUDA_COMPATIBILITY
            vllm serve ${MODEL_ID} --omni --port 8000 > /app/voxtral-tts/vllm_serve.log 2>&1 &
            VLLM_PID=\$!
            # --server_pid lets run_eval.py abort as soon as the server dies. On failure dump the
            # whole serve log: the engine-core traceback is what explains the crash.
            python run_eval.py \
                --model_id=${MODEL_ID} \
                --base_url=http://localhost:8000 \
                --server_pid=\${VLLM_PID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --batch_size=${BATCH_SIZE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                || { echo '--- vllm serve log (full) ---'; cat /app/voxtral-tts/vllm_serve.log; exit 1; }
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot fr fr ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot it it ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot de de ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
)

# ── Models: "model_id batch_size" (CLI args override) ────────────────────────
MODEL_CONFIGS=(
    "mistralai/Voxtral-4B-TTS-2603 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
