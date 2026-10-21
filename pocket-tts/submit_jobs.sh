#!/bin/bash
# HF Jobs TTS eval — Pocket TTS backend (generate → transcribe → SIM* → score; *voice-clone only, VOICE_CLONE=true).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["kyutai/pocket-tts" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
#
# kyutai/pocket-tts is gated on the Hub; if its download fails, pocket_tts silently substitutes the
# no-voice-cloning weights, which run_eval.py detects (has_voice_cloning) and fails fast on.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-pocket}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-pocket/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
# (Pocket TTS is CPU-first with no reported GPU speedup, so its RTFx here is a same-hardware number.)
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Checkpoint variant ───────────────────────────────────────────────────────
# pocket_tts ships one checkpoint per language; VARIANT selects a non-default one for the same
# language by suffixing the config name (run_eval.py validates it against the installed configs):
#   ""        -> english / german / spanish / italian / portuguese  (distilled production models)
#   24l       -> german_24l, italian_24l, ...  bigger 24-layer preview models, not distilled
#   2026-01 / 2026-04 -> the dated English checkpoints (english_2026-04 == the `english` default).
# It is language-independent because resolve_mode() runs once per MODEL, before the dataset loop.
VARIANT="${VARIANT:-}"

# ── Voice cloning (off by default): VOICE_CLONE=true clones the per-sample prompt + scores SIM ──
# VOICE_CLONE=false uses a predefined voice (SPEAKER, or the language default) and skips SIM;
# that path also works without the gated weights.
SPEAKER="${SPEAKER:-}"   # predefined voice, used only when VOICE_CLONE=false ("" = language default)
resolve_mode() {
    # A non-empty VARIANT gets its own suffix (matching run_eval.py), since every checkpoint
    # shares the `kyutai/pocket-tts` model id.
    voice_clone_mode "${VARIANT:+_${VARIANT}}"
}

# ── Languages Pocket TTS has a checkpoint for ────────────────────────────────
# One checkpoint per language and no multilingual fallback (run_eval.py hard-errors on anything
# else), so unsupported combos are skipped here. Every language listed also has a 24l variant.
POCKET_LANGUAGES="en de es it pt fr"
supports_language() { [[ " ${POCKET_LANGUAGES} " == *" $1 "* ]]; }

# ── Stage 1: TTS generation. MODEL_CFG = "model_id batch_size". ──
# NOTE: batch_size only sets the manifest-flush granularity — Pocket TTS has no batched API.
generate_stage() {
    local _ BATCH_SIZE
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    BATCH_SIZE="${BATCH_SIZE:-32}"   # tolerate a bare "model_id" with no second field
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/pocket-tts && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --language=${ASR_LANG} \
                --variant='${VARIANT}' \
                --speaker='${SPEAKER}' \
                --device=cuda:0 \
                --batch_size=${BATCH_SIZE} \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${VOICE_CLONE_FLAG} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment to pick) ──
# Seed-TTS zh and CV3-Eval ja/ko/ru/zh are omitted: Pocket TTS has no checkpoint for them.
DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot de de ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot fr fr ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot it it ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
)

# ── Models: "model_id batch_size" (CLI args override) ────────────────────────
# One Hub repo holds every language checkpoint; the language comes from the dataset combo and the
# checkpoint choice from VARIANT.
MODEL_CONFIGS=(
    "kyutai/pocket-tts 32"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
