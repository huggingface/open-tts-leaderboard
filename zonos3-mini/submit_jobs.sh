#!/bin/bash
# HF Jobs TTS eval — ZONOS3-Mini backend (generate → transcribe → SIM* → score; *voice-clone only).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["Zyphra/ZONOS3-Mini 32" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-zonos3-mini}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-zonos3-mini/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"
DEFAULT_STAGES="generate transcribe sim utmos komos"   # + KOMOS (Korean MOS): runs on the ko splits only
# Voice clone runs through the runtime's server (stage 1 starts it in the job; run_eval.py is its HTTP client):
# tokenizer processes beside the scheduler (text normalization is CPU work), prefill tokens per scheduler step (a
# batch of 32 is admitted at once). The fixed voice keeps the offline TTSLLM path.
ZONOS3_PORT=1919   # serve.sh's port
ZONOS3_TOKENIZERS="${ZONOS3_TOKENIZERS:-16}"
ZONOS3_PREFILL_TOKENS="${ZONOS3_PREFILL_TOKENS:-8192}"

# VOICE_CLONE=true: zero-shot voice cloning + SIM; false (default) = fixed default voice.
SUPPORTS_VOICE_CLONE=true

# Languages ZONOS3-Mini covers (model card: 43; `no` is nb, `tl` is fil here). The runtime normalizes text for
# en, zh, ja, ko, de, es, fr, it and pt; the others are sent without it, numbers spelled out by run_eval.py.
SUPPORTED_LANGUAGES="en fr es de it pt nl pl ru sv tr uk cs da fi el hu nb ro sk bg zh ja ko ar he vi id th fil ms ka hi bn ta te mr gu kn ml pa or ur"
# The fixed voice is a native default voice of each language (VOICES in run_eval.py); the model has none in
# Japanese or in the languages without text normalization, and a foreign clip derails the speech, so those run in
# voice-clone mode only.
supports_language() {
    language_supported "$1" &&
        [[ "${CLONE}" == "true" || " en zh fr de es it ko pt " == *" $1 "* ]]
}

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size". ──
# Voice clone: batch_size = the client's concurrent requests per batch and the server's running-request cap (and
# CUDA-graph batch), so a batch runs as one. The server's log tail is printed if the job fails.
generate_stage() {
    local _ BATCH_SIZE SERVER="" SERVER_ARGS=""
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    if [[ "${CLONE}" == "true" ]]; then
        SERVER="
            uv pip install -q --python /opt/ZONOS3-Mini/.venv/bin/python num2words
            (cd /opt/ZONOS3-Mini && SERVE_MODEL=hf://${MODEL_ID}/lm/zonos3_mini exec servers/sgl/scripts/serve.sh \
                --port ${ZONOS3_PORT} --max-running-requests ${BATCH_SIZE} --cuda-graph-max-bs ${BATCH_SIZE} \
                --max-prefill-length ${ZONOS3_PREFILL_TOKENS} --num-tokenizer ${ZONOS3_TOKENIZERS}) > /tmp/zonos3_server.log 2>&1 &
            SERVER_PID=\$!
            trap 'rc=\$?; kill \$SERVER_PID 2>/dev/null || true; if [ \$rc -ne 0 ]; then tail -n 200 /tmp/zonos3_server.log; fi; exit \$rc' EXIT"
        SERVER_ARGS="--server_url=http://127.0.0.1:${ZONOS3_PORT} --server_pid=\$SERVER_PID --server_log=/tmp/zonos3_server.log"
    fi
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            set -euo pipefail
            source /opt/ZONOS3-Mini/servers/sgl/scripts/env.sh
            ${SERVER}
            cd /app/zonos3-mini && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --language=${ASR_LANG} \
                --device=cuda:0 \
                --batch_size=${BATCH_SIZE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${SERVER_ARGS} \
                ${VOICE_CLONE_FLAG}
            mkdir -p /results
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Models: "model_id batch_size" (CLI args override) ────────────────────────
MODEL_CONFIGS=(
    "Zyphra/ZONOS3-Mini 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
