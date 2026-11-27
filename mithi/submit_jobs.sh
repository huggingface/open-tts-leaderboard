#!/bin/bash
# HF Jobs TTS eval: Mithi backend (fixed voice -> generate -> transcribe -> UTMOS (en) -> score; no SIM).
# Runs locally; submits HF Jobs. Usage:
#   HF_TOKEN=hf_... RESULTS_BUCKET=YOUR_USERNAME/bucket bash mithi/submit_jobs.sh ["Oomni/mithi-v1-edge <sha> 32" ...]
# Smoke test: add  MAX_EVAL_SAMPLES=8 ONLY_LANGS=en
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# No Docker Space: the public PyTorch image plus a pip install. The model repo ships a self-contained run.py, which
# run_eval.py imports after downloading the pinned revision.
GEN_IMAGE="${GEN_IMAGE:-pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime}"
GEN_PIP_DEPS="${GEN_PIP_DEPS:-'datasets<4.0' soundfile tqdm huggingface_hub safetensors numpy sentencepiece 'transformers>=5.0' \
'misaki[en]>=0.9.4' 'espeakng-loader>=0.2.4' 'phonemizer-fork>=3.3.2' 'spacy>=3.8' 'num2words>=0.5'}"
SPACY_MODEL="${SPACY_MODEL:-https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability: don't override.
GEN_TIMEOUT="${GEN_TIMEOUT:-1h}"
RUN_EVAL_INJECT="$(inject_run_eval)"

# Built-in voice (one voice per run, one leaderboard row): charvi, ahana, vaidehi, malini, jahnavi (female);
# satvik, anirban, nakul, vasant, siddhant (male).
VOICE="${VOICE:-charvi}"

# English + ten Indian languages (model card); of the benchmarks' languages only en and hi.
SUPPORTED_LANGUAGES="en hi"

# ── Stage 1: TTS generation. MODEL_CFG = "model_id revision batch_size" ──────
generate_stage() {
    local _ REVISION BATCH_SIZE
    read -r _ REVISION BATCH_SIZE <<< "${MODEL_CFG}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${GEN_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            pip install --no-cache-dir -q ${GEN_PIP_DEPS} ${SPACY_MODEL} &&
            cd /app/mithi && python run_eval.py \
                --model_id=${MODEL_ID} \
                --revision=${REVISION} \
                --voice=${VOICE} \
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

# ── Models: "model_id revision batch_size" (CLI args override) ───────────────
MODEL_CONFIGS=(
    "Oomni/mithi-v1-edge 7daa03162a35644293950c8db367d9690e9552ee 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
