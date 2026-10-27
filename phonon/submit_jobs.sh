#!/bin/bash
# English, fixed voice. Usage: bash phonon/submit_jobs.sh [model_id ...]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

TTS_SPACE="${PHONON_SPACE:-Steveeeeeeen/evals-phonon-040}"
RUN_EVAL_INJECT="$(inject_run_eval)"
DEFAULT_STAGES="generate transcribe utmos"
SUPPORTS_VOICE_CLONE=false
STAGE_RETRIES=0
# Exception to the shared h200 gen flavor: the Q8 ptts runtime is CPU-only (no GPU path), so an
# h200 would sit idle. Its RTFx is therefore a CPU number, not directly comparable with the h200 models.
FLAVOR=cpu-upgrade
PHONON_THREADS="${PHONON_THREADS:-1}"
supports_language() { [[ "$1" == en ]]; }

generate_stage() {
    local revision="${PHONON_SEED_DATASET_REVISION:-}" runner command
    [[ "$DATASET" != zero_shot ]] || revision="${PHONON_CV3_DATASET_REVISION:-}"
    local args=(--model_id "$MODEL_ID" --dataset_path "$DATASET_PATH"
        --dataset "$DATASET" --split "$SPLIT" --device cpu --threads "$PHONON_THREADS"
        --warmup_steps 2 --max_eval_samples "$MAX_EVAL_SAMPLES")
    [[ -z "$revision" ]] || args+=(--dataset_revision "$revision")
    printf -v runner '%q ' python3 /app/phonon/run_eval.py "${args[@]}"
    command="set -euo pipefail
        ${RUN_EVAL_INJECT} true || exit 1
        umask 077
        for existing in '${BUCKET_MANIFEST}' '${BUCKET_MANIFEST%.jsonl}.run.json' \
            '/results/${MODEL_FOLDER}/${DATASET_SAFE}_${DATASET}_${SPLIT}'; do
            test ! -e \"\$existing\" || { echo 'Quality artifacts already exist; inspect before rerunning.' >&2; exit 1; }
        done
        work_dir=\$(mktemp -d /tmp/phonon-quality.XXXXXXXX)
        ${runner} --output_root \"\$work_dir\"
        mkdir -p '/results/${MODEL_FOLDER}'
        source_model=\"\$work_dir/results/${MODEL_SAFE}\"
        cp -r \"\$source_model/${DATASET_SAFE}_${DATASET}_${SPLIT}\" '/results/${MODEL_FOLDER}/'
        cp \"\$source_model/$(basename "${BUCKET_MANIFEST%.jsonl}.run.json")\" '${BUCKET_MANIFEST%.jsonl}.run.json'
        cp \"\$source_model/$(basename "$BUCKET_MANIFEST")\" '${BUCKET_MANIFEST}'
    "
    local -a job=(hf jobs run --flavor "$FLAVOR" --timeout 2h --secrets HF_TOKEN)
    [[ -z "$ORG_NAME" ]] || job+=(--namespace "$ORG_NAME")
    "${job[@]}" --volume "hf://buckets/${RESULTS_BUCKET}:/results" "$TTS_IMAGE" bash -c "$command"
}

DATASET_CONFIGS=(
    "tts en en"
    "zero_shot en en ${CV3_EVAL_PATH}"
    # MiniMax multilingual (100 sentences/language): the languages the model supports. No `uk`: Qwen3-ASR has no Ukrainian.
    "tts en en ${MINIMAX_EVAL_PATH}"
)
MODEL_CONFIGS=(
    "gradium/phonon-31m-en"
    "gradium/phonon-40m-en"
)
[[ $# -eq 0 ]] || MODEL_CONFIGS=("$@")

fail() { echo "Phonon: $*" >&2; exit 1; }
for value in "$TTS_SPACE" "$RESULTS_BUCKET" "$DEFAULT_DATASET_PATH" "$CV3_EVAL_PATH"; do
    [[ "$value" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || fail "Use namespace/name resources."
done
[[ "$PHONON_THREADS" =~ ^[1-8]$ ]] || fail "Use 1–8 CPU threads."
[[ "$VOICE_CLONE" == false ]] || fail "Voice cloning is unsupported."
for model in "${MODEL_CONFIGS[@]}"; do
    [[ "$model" == gradium/phonon-31m-en || "$model" == gradium/phonon-40m-en ]] || fail "Select a released English checkpoint."
done
case "${STAGES:-$DEFAULT_STAGES}" in
    generate|"generate transcribe"|"generate transcribe utmos"|transcribe|utmos|"transcribe utmos") ;;
    *) fail "Supported stages: generate, 'generate transcribe', 'generate transcribe utmos', transcribe, utmos, or 'transcribe utmos'." ;;
esac

run_pipeline
status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
exit "$status"
