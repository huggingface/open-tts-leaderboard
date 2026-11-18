#!/bin/bash
# HF Jobs TTS eval: Cilla backend (Persian, fixed voice -> generate -> transcribe -> score).
# Runs locally; submits HF Jobs. Usage:
#   HF_TOKEN=hf_... RESULTS_BUCKET=YOUR_USERNAME/bucket bash cilla/submit_jobs.sh ["mehdi-hf/cilla-mini" ...]
# The HF_TOKEN must belong to an account with access to the gated model repos (access is granted automatically after
# accepting the terms on https://huggingface.co/mehdi-hf/cilla-mini and /cilla-lite).
# Smoke test: add  MAX_EVAL_SAMPLES=8 ONLY_LANGS=fa
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 environment: no Docker Space needed. The job runs on the public PyTorch image below and pip-installs the
# eval-loop deps and the released `pocket-tts` package at job start. The model repo's own Persian front-end
# (speak_fa.py, normalize_fa.py) is downloaded by run_eval.py at the pinned revision.
GEN_IMAGE="${GEN_IMAGE:-pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime}"
GEN_PIP_DEPS="${GEN_PIP_DEPS:-'pocket-tts>=4.0.1' datasets soundfile tqdm huggingface_hub scipy numpy sentencepiece}"
# Exception to the shared h200 gen flavor: the models are CPU-first (18M parameters, no tested GPU path), so an h200
# would sit idle. Its RTFx is therefore a CPU number, not comparable with the GPU models (same as the Phonon backend).
FLAVOR=cpu-upgrade
GEN_DEVICE="${GEN_DEVICE:-cpu}"
GEN_TIMEOUT="${GEN_TIMEOUT:-8h}"
DEFAULT_STAGES="generate transcribe"             # fixed voice, Persian: no SIM (no cloning), no UTMOS (English only)
RUN_EVAL_INJECT="$(inject_run_eval)"

# Pinned commits of the model repos (the released versions). Override with MODEL_REVISION=<40-char sha> for a new release.
revision_for() {
    case "$1" in
        mehdi-hf/cilla-mini) echo "8720ebea8b22dc9e36392f0df632514df40c263d" ;;
        mehdi-hf/cilla-lite) echo "e12e50f108dff1443012e0b540f095821d1e5245" ;;
        *) echo "" ;;
    esac
}
# Built-in voice: voice1 (a woman), voice2 (a man) or parisa. One voice per run, one leaderboard row.
VOICE="${VOICE:-voice1}"

# Persian-only models: skip every other split instead of burning a job on garbage audio.
supports_language() { [[ "$1" == "fa" ]]; }

# ── Stage 1: TTS generation (one sample at a time: no batch size) ──
generate_stage() {
    local revision="${MODEL_REVISION:-$(revision_for "${MODEL_ID}")}"
    if [[ ! "${revision}" =~ ^[0-9a-f]{40}$ ]]; then
        echo "cilla: no pinned revision for ${MODEL_ID}: set MODEL_REVISION to the full 40-char commit sha." >&2
        return 1
    fi
    echo "cilla: ${MODEL_ID} revision=${revision} voice=${VOICE} results bucket=${RESULTS_BUCKET}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${GEN_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            pip install --no-cache-dir -q ${GEN_PIP_DEPS} &&
            cd /app/cilla && python run_eval.py \
                --model_id=${MODEL_ID} \
                --revision=${revision} \
                --voice=${VOICE} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=${GEN_DEVICE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]". Persian exists only in Neyshekar (2,098 test rows, CER). ──
DATASET_CONFIGS=(
    "tts fa fa ${NEYSHEKAR_EVAL_PATH}"
)

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=("mehdi-hf/cilla-mini" "mehdi-hf/cilla-lite")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
