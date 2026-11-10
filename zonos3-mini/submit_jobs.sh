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

# The fixed voice is a native default voice of each language (VOICES in run_eval.py); the model has none in
# Japanese or in the languages without text normalization, and a foreign clip derails the speech, so those run in
# voice-clone mode only.
supports_language() {
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

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
# Without a 4th field the line uses DEFAULT_DATASET_PATH (hf-audio/seed_tts_eval).
# The languages with ZONOS3-Mini text normalization: en, zh, ja, ko, de, es, fr, it, pt; the others are sent without
# it, numbers spelled out by run_eval.py. MiniMax: every language but uk (Qwen3-ASR has no Ukrainian) and yue.
DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot fr fr ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot zh zh ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ja ja ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ko ko ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot de de ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot it it ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ru ru ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    # MiniMax multilingual (100 sentences/language).
    "tts en en ${MINIMAX_EVAL_PATH}"
    "tts zh zh ${MINIMAX_EVAL_PATH}"
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

# ── Models: "model_id batch_size" (CLI args override) ────────────────────────
MODEL_CONFIGS=(
    "Zyphra/ZONOS3-Mini 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
