#!/bin/bash
# HF Jobs TTS eval — Paradee 1.3 backend (fixed voice per language → 2 stages: generate → transcribe → score).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["sahilmahendrakar/Paradee-8M-v1.3" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Separate from paradee/ (v1.0, image bezzam/evals-paradee, which pins the 1.0 package): the 1.3 package
# changed its API (eight voices, a voice argument, no REPO/REVISION constants), so 1.3 gets its own folder.
# Stage-1 environment: no Docker Space needed. GEN_IMAGE (a public PyTorch image) and gen_setup (the
# pip install at job start) live in job_image.sh, shared with submit_ttfa_jobs.sh so the TTFA probe
# runs the same install.
source "${BACKEND_DIR}/job_image.sh"
GEN_TIMEOUT="${GEN_TIMEOUT:-2h}"                 # an 8M model on the largest split takes minutes; a hang must not burn H200 hours
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
DEFAULT_STAGES="generate transcribe utmos"       # no SIM (fixed voice)
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Device / threads ─────────────────────────────────────────────────────────
# The package ships a 1-thread CPU session; run_eval.py swaps in the CUDA EP, which is ~1.5x faster on the
# h200 with HEURISTIC conv search (see paradee-v1.3/run_eval.py). PARADEE_DEVICE=cpu benchmarks the package's own CPU session instead.
PARADEE_DEVICE="${PARADEE_DEVICE:-cuda:0}"
PARADEE_THREADS="${PARADEE_THREADS:-1}"
# cuDNN conv algorithm search: HEURISTIC, not ONNX Runtime's default EXHAUSTIVE, which re-benchmarks every
# conv for each new input length inside the timed loop (see --cudnn_conv_algo_search in run_eval.py).
PARADEE_CONV_ALGO_SEARCH="${PARADEE_CONV_ALGO_SEARCH:-HEURISTIC}"

# ── Language → voice ─────────────────────────────────────────────────────────
# v1.3 speaks English, Hindi, French and Spanish (eight of Kokoro's voices in one model; a voice's
# first letter is its language). One voice per language, one leaderboard row per model, as in the
# Kokoro backend: the same voices as kokoro/submit_jobs.sh where Paradee has them, so the two rows
# compare directly. Kokoro uses ef_dora for Spanish; Paradee's Spanish voice is em_alex.
# Paradee's other voices: am_puck (en), bf_isabella and bm_fable (en, British), hm_omega (hi).
# A case, not an associative array, so it runs on macOS bash 3.2.
paradee_voice() {
    case "$1" in
        en) echo af_heart ;;
        hi) echo hf_alpha ;;
        fr) echo ff_siwis ;;
        es) echo em_alex ;;
    esac
}
supports_language() { [[ -n "$(paradee_voice "$1")" ]]; }
# Setting PARADEE_VOICE forces one voice for every combo; only for a single-language run (e.g.
# ONLY_LANGS=en PARADEE_VOICE=bf_isabella). run_eval.py refuses a voice of another language.
# The results land in the same model folder, so use a separate RESULTS_BUCKET for such runs.
PARADEE_VOICE="${PARADEE_VOICE:-}"

# ── Stage 1: TTS generation (no batched API — one sample at a time) ──
generate_stage() {
    local VOICE="${PARADEE_VOICE:-$(paradee_voice "${ASR_LANG}")}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${GEN_TIMEOUT}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${GEN_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            $(gen_setup "${PARADEE_DEVICE}")
            cd /app/paradee-v1.3 && python run_eval.py \
                --model_id=${MODEL_ID} \
                --voice=${VOICE} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --device=${PARADEE_DEVICE} \
                --threads=${PARADEE_THREADS} \
                --cudnn_conv_algo_search=${PARADEE_CONV_ALGO_SEARCH} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment) ──
DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot fr fr ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    # MiniMax multilingual (100 sentences/language): the languages the model supports. No `uk`: Qwen3-ASR has no Ukrainian.
    "tts en en ${MINIMAX_EVAL_PATH}"
    "tts fr fr ${MINIMAX_EVAL_PATH}"
    "tts es es ${MINIMAX_EVAL_PATH}"
    "tts hi hi ${MINIMAX_EVAL_PATH}"
)

# ── Models: "model_id" (CLI args override) ───────────────────────────────────
MODEL_CONFIGS=("sahilmahendrakar/Paradee-8M-v1.3")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
