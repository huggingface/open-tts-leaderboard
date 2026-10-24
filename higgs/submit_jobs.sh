#!/bin/bash
# HF Jobs TTS eval — Higgs backend (generate → transcribe → SIM* → score; *voice-clone only, VOICE_CLONE=true).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["bosonai/higgs-tts-3-4b 16" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-higgs}"   # stage-1 image (SGLang-Omni, Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-higgs/blob/main/Dockerfile)
SGLANG_PORT="${SGLANG_PORT:-8000}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
# run_eval.py imports no `normalizer`, so no NORMALIZER_INJECT / PYTHONPATH is needed here.
RUN_EVAL_INJECT="$(inject_run_eval)"

# Voice cloning (VOICE_CLONE=true): clone the per-sample reference + score SIM; else built-in voice.
SUPPORTS_VOICE_CLONE=true

# ── Stage 1: launch the SGLang server, then generate via the HTTP client. MODEL_CFG = "model_id concurrency". ──
# Higgs TTS is served (not loaded in-process): sgl-omni serve exposes an OpenAI-compatible
# /v1/audio/speech endpoint; run_eval.py polls /health, then sends synthesis requests. It is an
# HTTP client, so it takes --host/--port (and NO --device). --allowed-local-media-path lets the
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
            mkdir -p /app/higgs/results
            /opt/omni/bin/sgl-omni serve --model-path ${MODEL_ID} --host 127.0.0.1 --port ${SGLANG_PORT} \
                --allowed-local-media-path /app/higgs/results \
                > /app/higgs/results/sglang_server_${SPLIT}.log 2>&1 &
            SERVER_PID=\$!
            trap 'kill \$SERVER_PID 2>/dev/null || true' EXIT
            cd /app/higgs && python run_eval.py \
                --model_id=${MODEL_ID} \
                --host=127.0.0.1 \
                --port=${SGLANG_PORT} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --batch_size=${BATCH_SIZE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${VOICE_CLONE_FLAG}
            mkdir -p /results
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment to pick) ────────
# Higgs TTS 3 is multilingual and infers the language from the text (no --language flag, no
# `supports_language` gate), so asr_language is only the stage-2 ASR hint.
DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot fr fr ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot zh zh ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot ja ja ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot ko ko ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot de de ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot it it ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot ru ru ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    # ZTTS1-Eval: Clean set (zero_shot; hard_* scored as en/zh) + In-The-Wild set (zero_shot_voxblink).
    "zero_shot en en ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot zh zh ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot hard_en en ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot hard_zh zh ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot de de ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot es es ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot fr fr ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot it it ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot ja ja ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot ko ko ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot ru ru ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink en en ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink zh zh ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink de de ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink es es ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink fr fr ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink it it ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink ja ja ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink ko ko ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink ru ru ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink pt pt ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink ar ar ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink hi hi ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink tr tr ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    # Qwen3-ASR's code for Tagalog is `fil`; it rejects `tl`.
    "zero_shot_voxblink tl fil ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink pl pl ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink id id ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
    "zero_shot_voxblink th th ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
)

# ── Models: "model_id concurrency" (CLI args override) ───────────────────────
MODEL_CONFIGS=(
    "bosonai/higgs-tts-3-4b 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
