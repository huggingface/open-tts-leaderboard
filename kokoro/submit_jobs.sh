#!/bin/bash
# HF Jobs TTS eval — Kokoro backend (fixed voice → 2 stages: generate → transcribe → score).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["hexgrad/Kokoro-82M" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-kokoro}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-kokoro/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"      # no SIM (fixed voice)
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Language → (Kokoro lang_code, voice pack) ────────────────────────────────
# Kokoro picks its G2P from a one-character lang_code, and a voice's first letter MUST equal it,
# so each language needs a matching voice. Voices in hexgrad/Kokoro-82M:
#   a American English (20)  b British English (8)  e Spanish (3)  f French (1)  h Hindi (4)
#   i Italian (2)  j Japanese (5)  p Portuguese-BR (3)  z Mandarin (8)
# No German, Korean or Russian voice, so CV3-Eval's de/ko/ru splits cannot run here.
declare -A KOKORO_LANG_CODE=( [en]=a [es]=e [fr]=f [it]=i [ja]=j [zh]=z [pt]=p [hi]=h )
declare -A KOKORO_VOICES=( [en]=af_heart [es]=ef_dora [fr]=ff_siwis [it]=if_sara \
                           [ja]=jf_alpha [zh]=zf_xiaobei [pt]=pf_dora [hi]=hf_alpha )
supports_language() { [[ -n "${KOKORO_LANG_CODE[$1]:-}" ]]; }
# Setting KOKORO_VOICE forces one voice for every combo — only safe for a single-language run,
# since a voice whose prefix does not match the lang_code is rejected by Kokoro.
KOKORO_VOICE="${KOKORO_VOICE:-}"

# ── Stage 1: TTS generation (Kokoro synthesizes one sample at a time — no batch size) ──
generate_stage() {
    local LANG_CODE VOICE
    LANG_CODE="${KOKORO_LANG_CODE[${ASR_LANG}]}"
    VOICE="${KOKORO_VOICE:-${KOKORO_VOICES[${ASR_LANG}]}}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/kokoro && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${VOICE} \
                --lang_code=${LANG_CODE} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=cuda:0 \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot fr fr ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot it it ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot ja ja ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
    "zero_shot zh zh ${CV3_EVAL_PATH:-bezzam/cv3_eval}"
)

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=("hexgrad/Kokoro-82M")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
