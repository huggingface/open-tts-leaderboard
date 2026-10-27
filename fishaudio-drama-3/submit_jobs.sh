#!/bin/bash
# HF Jobs TTS eval — Fish Audio Drama 3 backend (generate → transcribe → SIM* → score; *voice-clone only, VOICE_CLONE=true).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["fishaudio/drama-3 8" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-drama3}"   # stage-1 image (pinned SGLang Omni, Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-drama3/blob/main/Dockerfile)
# Pinned weights: sglang_omni serve has no --revision, so the job serves this snapshot's local path.
# run_eval.py only accepts this revision (it records it in the run receipt).
MODEL_REVISION="e0a9218aa48888270f06ef1ed9ea67622c0e17d2"
SGLANG_PORT="${SGLANG_PORT:-8000}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"   # untimed warm-up batches
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
# run_eval.py is large: gzip it, since a job body is one `bash -c` argument (128 KiB cap).
RUN_EVAL_INJECT="$(inject_cmd_gz "${BACKEND_DIR}/run_eval.py" /app/fishaudio-drama-3/run_eval.py) $(inject_cmd_gz "${REPO_ROOT}/scripts/run_eval_utils.py" /app/scripts/run_eval_utils.py) ${SCRIPTS_PYTHONPATH}"

# Voice cloning (VOICE_CLONE=true): clone the per-sample reference + score SIM; else built-in voice.
SUPPORTS_VOICE_CLONE=true

# ── Stage 1: launch the SGLang Omni server, then generate via the HTTP client. MODEL_CFG = "model_id concurrency". ──
# The server batches concurrent requests (up to 16 by default); 8 matches the upstream cookbook's
# non-streaming H200 benchmark and stays clear of vocoder OOMs. --allowed-local-media-path lets the
# server read the voice-clone reference wavs that run_eval.py writes under results/.
generate_stage() {
    local _ BATCH_SIZE
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            set -e
            mkdir -p /app/fishaudio-drama-3/results
            SNAPSHOT=\$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download(\"${MODEL_ID}\", revision=\"${MODEL_REVISION}\"))')
            python -m sglang_omni.cli serve --model-path \$SNAPSHOT --model-name ${MODEL_ID} \
                --host 127.0.0.1 --port ${SGLANG_PORT} \
                --allowed-local-media-path /app/fishaudio-drama-3/results \
                > /app/fishaudio-drama-3/results/sglang_server_${SPLIT}.log 2>&1 &
            SERVER_PID=\$!
            trap 'kill \$SERVER_PID 2>/dev/null || true' EXIT
            cd /app/fishaudio-drama-3 && python run_eval.py \
                --model_id=${MODEL_ID} \
                --model_revision=${MODEL_REVISION} \
                --host=127.0.0.1 \
                --port=${SGLANG_PORT} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --batch_size=${BATCH_SIZE} \
                --warmup_steps=${WARMUP_STEPS} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${VOICE_CLONE_FLAG}
            mkdir -p /results
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment to pick) ────────
DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot zh zh ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot fr fr ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ja ja ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ko ko ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot de de ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot it it ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ru ru ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    # MiniMax multilingual (100 sentences/language): the languages the model supports. No `uk`: Qwen3-ASR has no Ukrainian.
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

# ── Models: "model_id concurrency" (CLI args override; only fishaudio/drama-3 is supported) ──
MODEL_CONFIGS=(
    "fishaudio/drama-3 8"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
