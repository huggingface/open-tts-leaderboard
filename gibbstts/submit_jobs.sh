#!/bin/bash
# GibbsTTS: compiled native batch32, BF16 DiT autocast, padded tail.
set -e -o pipefail

# Never accidentally read the maintainer's private datasets or write their official bucket.
: "${DATASET_NAMESPACE:?Set DATASET_NAMESPACE to the owner of your private benchmark copies}"
: "${RESULTS_BUCKET:?Set RESULTS_BUCKET to your evaluation bucket}"
TTS_SPACE="${TTS_SPACE:-bezzam/evals}"
if [[ "$TTS_SPACE" == bezzam/evals ]]; then
    GIBBSTTS_BOOTSTRAP="${GIBBSTTS_BOOTSTRAP:-true}"
fi
VOICE_CLONE="${VOICE_CLONE:-true}"
[[ "$VOICE_CLONE" == true ]] || { echo "GibbsTTS requires VOICE_CLONE=true." >&2; exit 1; }
[[ "${FLAVOR:-h200}" == h200 ]] || { echo "Leaderboard generation must use h200." >&2; exit 1; }

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"
SUPPORTS_VOICE_CLONE=true
RUN_EVAL_INJECT="$(inject_run_eval)"
RUN_EVAL_INJECT+=" $(inject_cmd "${BACKEND_DIR}/batched_inference.py" /app/gibbstts/batched_inference.py)"
RUN_EVAL_INJECT+=" $(inject_cmd "${BACKEND_DIR}/compiler_backend.py" /app/gibbstts/compiler_backend.py)"
BOOTSTRAP_INJECT=""
if [[ "${GIBBSTTS_BOOTSTRAP:-false}" == true ]]; then
    BOOTSTRAP_INJECT="$(inject_cmd "${BACKEND_DIR}/bootstrap.sh" /app/gibbstts/bootstrap.sh) source /app/gibbstts/bootstrap.sh &&"
fi

supports_language() { [[ "$1" == en || "$1" == zh ]]; }

# Fixed upstream defaults; separate variants should use separate result buckets.
generate_stage() {
    local _ batch_size gen_timeout
    read -r _ batch_size <<< "$MODEL_CFG"
    [[ "$batch_size" == 32 ]] || { echo "This submitted configuration uses batch_size=32." >&2; return 1; }
    gen_timeout="${GEN_TIMEOUT:-35m}"
    if [[ "${GIBBSTTS_FULL_TIMEOUTS:-false}" == true ]]; then
        case "${DATASET}/${SPLIT}" in
            tts/en|tts/zh) gen_timeout=35m ;;
            zero_shot/en|zero_shot/zh) gen_timeout=20m ;;
        esac
    fi
    hf jobs run \
        --flavor "$FLAVOR" --timeout "$gen_timeout" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "$TTS_IMAGE" \
        bash -c "
            set -e
            ${RUN_EVAL_INJECT}
            ${BOOTSTRAP_INJECT}
            cd /app/gibbstts && python run_eval.py \
                --model_id=${MODEL_ID} \
                --dataset_path=${DATASET_PATH} \
                --dataset=${DATASET} --split=${SPLIT} \
                --device=cuda:0 --batch_size=32 --precision=bf16 --compile --tail_padding \
                --max_eval_samples=${MAX_EVAL_SAMPLES} \
                --steps=32 --cfg=2.5 --rescale_cfg=0.75 \
                --temperature=0.6 --top_p=1.0 --speed=1.0 \
                ${VOICE_CLONE_FLAG} &&
            cp -r results/${MODEL_SAFE} /results/
        "
}

DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH}"
    "zero_shot zh zh ${CV3_EVAL_PATH}"
)
case "${GIBBSTTS_EVAL_SET:-all}" in
    all) ;;
    seed) DATASET_CONFIGS=("tts en en" "tts zh zh") ;;
    cv3) DATASET_CONFIGS=("zero_shot en en ${CV3_EVAL_PATH}" "zero_shot zh zh ${CV3_EVAL_PATH}") ;;
    *) echo 'GIBBSTTS_EVAL_SET must be all, seed, or cv3.' >&2; exit 1 ;;
esac
MODEL_CONFIGS=("ydqmkkx/GibbsTTS 32")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

# Check every supplied model config before submitting the first paid job.
for cfg in "${MODEL_CONFIGS[@]}"; do
    read -r model batch extra <<< "$cfg"
    [[ "$model" == ydqmkkx/GibbsTTS && "$batch" == 32 && -z "$extra" ]] || {
        echo 'Expected model config: "ydqmkkx/GibbsTTS 32".' >&2; exit 1;
    }
done
run_pipeline
