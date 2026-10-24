#!/bin/bash
# HF Jobs TTS eval — Chatterbox backend (generate → transcribe → SIM* → score; *voice-clone only, VOICE_CLONE=true).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["ResembleAI/chatterbox multilingual en,zh" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
TTS_SPACE="${TTS_SPACE:-bezzam/evals-chatterbox}"   # stage-1 image (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-chatterbox/blob/main/Dockerfile)
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
RUN_EVAL_INJECT="$(inject_run_eval)"

# ── Voice cloning: VOICE_CLONE=true clones the per-sample prompt + scores SIM;
# false uses the built-in default voice. ──
resolve_mode() {
    # The multilingual checkpoint shares the English-only one's model id, so it gets its own
    # manifest suffix (matching run_eval.py).
    local _ VARIANT
    read -r _ VARIANT _ <<< "${MODEL_CFG}"
    if [[ "${VARIANT}" == "multilingual" ]]; then
        MULTILINGUAL_FLAG="--multilingual"; voice_clone_mode "_multilingual"
    else
        MULTILINGUAL_FLAG="--no-multilingual"; voice_clone_mode
    fi
}

# Skip (model, dataset) combos the checkpoint can't synthesize — the third MODEL_CONFIGS field is a
# comma-separated list of supported language codes.
supports_language() {
    local want="$1" _ _v LANGS
    read -r _ _v LANGS <<< "${MODEL_CFG}"
    [[ ",${LANGS}," == *",${want},"* ]]
}

# ── Stage 1: TTS generation. Chatterbox synthesizes one sample at a time — no batch size. ──
generate_stage() {
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/chatterbox && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} \
                --split=${SPLIT} \
                --language=${ASR_LANG} \
                --device=cuda:0 \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                ${MULTILINGUAL_FLAG} \
                ${VOICE_CLONE_FLAG} &&
            mkdir -p /results &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment to pick) ────────
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
    "zero_shot_voxblink pl pl ${ZTTS1_EVAL_PATH:-Zyphra/ztts1_eval}"
)

# ── Models: "model_id variant supported_langs" (CLI args override) ───────────
# variant: `english` -> ChatterboxTTS (English-only checkpoint, no language argument)
#          `multilingual` -> ChatterboxMultilingualTTS (needs language_id; `_multilingual` suffix)
# supported_langs: comma-separated codes; combos outside the list are skipped (supports_language).
# The two rows are different checkpoints in the same repo, separated by the manifest suffix.
MODEL_CONFIGS=(
    "ResembleAI/chatterbox english en"
    "ResembleAI/chatterbox multilingual ar,da,de,el,en,es,fi,fr,he,hi,it,ja,ko,ms,nl,no,pl,pt,ru,sv,sw,tr,zh"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
