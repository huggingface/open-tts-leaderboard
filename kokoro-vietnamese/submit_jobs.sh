#!/bin/bash
# HF Jobs TTS eval — Kokoro-Vietnamese backend (fixed voice → generate → transcribe → score; no SIM).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["contextboxai/Kokoro-Vietnamese" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage 1 runs on a public PyTorch image and pip-installs the model package at job start (no Docker Space
# needed). Install time is before the timed loop, so RTFx is unaffected. To use a Space instead, build one
# that installs the same packages and set KVI_IMAGE=hf.co/spaces/<id>.
KVI_IMAGE="${KVI_IMAGE:-pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime}"
KVI_PIP="${KVI_PIP:-https://github.com/iamdinhthuan/Kokoro-Vietnamese/archive/a249afe5555aec6c435165c2f61ec0f71284812f.zip}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe"   # no SIM (fixed voice); no UTMOS (English only)
RUN_EVAL_INJECT="$(inject_run_eval)"

# Vietnamese only. Built-in voices: diem_trinh, hung_thinh, mai_linh, mai_loan, manh_dung, my_yen, ...
KOKORO_VI_VOICE="${KOKORO_VI_VOICE:-diem_trinh}"
supports_language() { [[ "$1" == "vi" ]]; }

# ── Stage 1: TTS generation (serial synthesis, one sample at a time) ─────────
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT:-8h}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${KVI_IMAGE}" \
        bash -c "
            pip install --no-cache-dir '${KVI_PIP}' soundfile 'datasets<4.0' tqdm numpy &&
            ${RUN_EVAL_INJECT}
            cd /app/kokoro-vietnamese && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${KOKORO_VI_VOICE} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" ─────────────────────
# MiniMax multilingual: 100 sentences per language, public at hf-audio/minimax_eval.
DATASET_CONFIGS=(
    "tts vi vi ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=("contextboxai/Kokoro-Vietnamese")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
