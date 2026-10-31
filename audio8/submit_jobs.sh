#!/usr/bin/env bash
# Audio8 Realistic -> official shared transcription / SIM / UTMOS / scoring.
# This script submits PAID HF Jobs unless DRY_RUN=true.
if [[ "${1:-}" == "--help" ]]; then
    cat <<'HELP'
Required for HF Jobs: TTS_SPACE=EXECUTOR_NAMESPACE/evals-audio8
          (executing party's public, built Docker Space)
          RESULTS_BUCKET=YOUR_NAMESPACE/tts-eval-results (writable HF bucket)
Set DATASET_NAMESPACE or DATASET_PATH/CV3_EVAL_PATH to accessible licensed copies.
DRY_RUN=true previews the selected matrix without HF calls or uploads.
VOICE_CLONE=true enables prompt conditioning and SIM (default false).
ONLY_LANGS / ONLY_DATASETS / MAX_EVAL_SAMPLES are official pipeline filters.
The executing party supplies its own runtime Space, datasets and result bucket.
This backend does not create or publish those resources.
Stage 1 defaults to the common H200 flavor and real batch size 32.
Local result checks: PYTHONPATH=scripts python audio8/run_eval.py validate-results --help
HELP
    exit 0
fi
: "${TTS_SPACE:?Set TTS_SPACE to a public, built Docker Space supplied by the executing party}"
: "${RESULTS_BUCKET:?Set RESULTS_BUCKET explicitly to a writable HF bucket}"

# Temporary compatibility wrapper for upstream's GNU-only `base64 -w0` injection.
# BSD/macOS and GNU base64 both accept stdin. This is local to this shell, never exported.
base64() {
    if [[ "${1:-}" == "-w0" ]]; then
        shift
        if [[ $# -eq 0 ]]; then command base64 | tr -d '\r\n';
        else command base64 < "$1" | tr -d '\r\n'; fi
    else
        command base64 "$@"
    fi
}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"
AUDIO8_PYTHON="${AUDIO8_PYTHON:-python3}"
RUN_EVAL_INJECT="$(inject_run_eval)"
SUPPORTS_VOICE_CLONE=true
MODEL_REVISION="${MODEL_REVISION:-1e857568bfd4f02931aca3988ee773a44d2d26d3}"
TEMPERATURE="${TEMPERATURE:-}"
CV3_TEMPERATURE="${CV3_TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
RETRY_MAX_NEW_TOKENS="${RETRY_MAX_NEW_TOKENS:-0}"

supports_language() {
    case "$1" in
        ar|yue|zh|nl|en|fi|fr|de|hi|id|it|ja|ko|ms|no|pl|pt|ru|es|sv|th|tr|vi) return 0 ;;
        *) return 1 ;;
    esac
}

generate_stage() {
    local _ BATCH_SIZE EVAL_TEMPERATURE EVAL_COMMAND COPY_SOURCE
    EVAL_TEMPERATURE="${TEMPERATURE:-0.1}"
    if [[ -z "${TEMPERATURE}" && "${DATASET}" == "zero_shot" ]]; then
        EVAL_TEMPERATURE="${CV3_TEMPERATURE}"
    fi
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    printf -v EVAL_COMMAND '%q ' python run_eval.py \
        "--model_id=${MODEL_ID}" "--revision=${MODEL_REVISION}" \
        "--dataset_path=${DATASET_PATH}" "--dataset=${DATASET}" "--split=${SPLIT}" \
        --device=cuda:0 --dtype=bfloat16 "--batch_size=${BATCH_SIZE}" \
        "--temperature=${EVAL_TEMPERATURE}" "--top_p=${TOP_P}" "--top_k=${TOP_K}" \
        "--max_new_tokens=${MAX_NEW_TOKENS}" "--retry_max_new_tokens=${RETRY_MAX_NEW_TOKENS}" \
        "--max_eval_samples=${MAX_EVAL_SAMPLES}" "${VOICE_CLONE_FLAG}"
    printf -v COPY_SOURCE '%q' "results/${MODEL_SAFE}"
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 12h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            ${RUN_EVAL_INJECT}
            cd /app/audio8 && ${EVAL_COMMAND} &&
            mkdir -p /results && cp -r ${COPY_SOURCE} /results/
        "
}

DATASET_CONFIGS=("tts en en" "tts zh zh")
for lang in en zh ja ko de fr es it ru; do
    DATASET_CONFIGS+=("zero_shot ${lang} ${lang} ${CV3_EVAL_PATH}")
done
# MiniMax's 24 evaluated languages intersect Realistic's 23 supported languages in 20.
# Exclude cs/el/ro/uk; ms/no/sv are supported but absent from this benchmark.
for lang in ar yue zh nl en fi fr de hi id it ja ko pl pt ru es th tr vi; do
    DATASET_CONFIGS+=("tts ${lang} ${lang} ${MINIMAX_EVAL_PATH}")
done
MODEL_CONFIGS=("Edge0/Audio8-TTS-Realistic-0.6b 32")
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

# Validate inputs locally before a paid job can be submitted. The shared scorer's
# shell snippets assume plain Hub identifiers; reject shell metacharacters here.
for identifier in "${TTS_SPACE}" "${RESULTS_BUCKET}" "${DEFAULT_DATASET_PATH}" \
    "${CV3_EVAL_PATH}" "${MINIMAX_EVAL_PATH}" "${SPACE}"; do
    if [[ ! "${identifier}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
        echo "ERROR: invalid Hub identifier: ${identifier}" >&2; exit 2
    fi
done
if [[ "${VOICE_CLONE}" != true && "${VOICE_CLONE}" != false ]]; then
    echo "ERROR: VOICE_CLONE must be true or false" >&2; exit 2
fi
if [[ "${SIM_BACKEND}" != wavlm_seed_tts && "${SIM_BACKEND}" != xvector ]]; then
    echo "ERROR: SIM_BACKEND must be wavlm_seed_tts or xvector" >&2; exit 2
fi
for stage in ${STAGES:-generate transcribe sim utmos}; do
    case "${stage}" in generate|transcribe|sim|utmos) ;; *) echo "ERROR: unknown stage ${stage}" >&2; exit 2;; esac
done
for cfg in "${MODEL_CONFIGS[@]}"; do
    read -r model batch extra <<< "${cfg}"
    if [[ ! "${model}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ || ! "${batch}" =~ ^[1-9][0-9]*$ || -n "${extra}" ]]; then
        echo "ERROR: model configuration must be 'namespace/model positive_batch_size'" >&2; exit 2
    fi
    for temperature in "${TEMPERATURE:-0.1}" "${TEMPERATURE:-${CV3_TEMPERATURE}}"; do
        PYTHONPATH="${COMMON_DIR}:${BACKEND_DIR}" "${AUDIO8_PYTHON}" -c \
            'from run_eval import parse_args; parse_args()' \
            "--model_id=${model}" "--revision=${MODEL_REVISION}" "--batch_size=${batch}" \
            "--temperature=${temperature}" "--top_p=${TOP_P}" "--top_k=${TOP_K}" \
            "--max_new_tokens=${MAX_NEW_TOKENS}" "--retry_max_new_tokens=${RETRY_MAX_NEW_TOKENS}" \
            "--max_eval_samples=${MAX_EVAL_SAMPLES}" || exit 2
    done
done
for snippet in "${RUN_EVAL_INJECT}" "${TRANSCRIBE_INJECT}" "${SIM_INJECT}" "${UTMOS_INJECT}"; do
    if [[ -z "${snippet}" || "${snippet}" == *"echo ''"* ]]; then
        echo "ERROR: an injected script is empty" >&2; exit 1
    fi
done

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf 'DRY RUN; no jobs submitted. Space=%s bucket=%s GPU=%s clone=%s\n' \
        "${TTS_SPACE}" "${RESULTS_BUCKET}" "${FLAVOR}" "${VOICE_CLONE}"
    printf 'Revision=%s temperature=%s top_p=%s top_k=%s max_tokens=%s retry_tokens=%s\n' \
        "${MODEL_REVISION}" "${TEMPERATURE:-Seed/MiniMax=0.1,CV3=${CV3_TEMPERATURE}}" "${TOP_P}" "${TOP_K}" "${MAX_NEW_TOKENS}" "${RETRY_MAX_NEW_TOKENS}"
    selected=0
    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r dataset split lang repo <<< "$cfg"
        repo="${repo:-${DEFAULT_DATASET_PATH}}"
        [[ -n "${ONLY_LANGS}" && " ${ONLY_LANGS} " != *" ${lang} "* ]] && continue
        [[ -n "${ONLY_DATASETS}" && " ${ONLY_DATASETS} " != *" ${repo##*/} "* ]] && continue
        supports_language "${lang}" || continue
        selected=$((selected + 1))
        printf '%s %s/%s (ASR=%s)\n' "${repo}" "${dataset}" "${split}" "${lang}"
    done
    if [[ "${selected}" -eq 0 ]]; then echo "ERROR: filters selected no evaluation datasets" >&2; exit 2; fi
    printf 'Generation, transcription, SIM and UTMOS script injection prepared.\n'
    printf 'Model configs: %s\n' "${MODEL_CONFIGS[*]}"
    exit 0
fi
# Shared synchronization/scoring uses GNU timeout/find; a Linux launcher is preferred.
if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "Use a Linux launcher for the full pipeline (shared scoring requires GNU timeout/find)." >&2
    exit 2
fi
# Keep the shared orchestration and scoring intact, but make failures visible to
# automation. Its conditional shell chains can otherwise advance after a failed
# job, and its top-level loop deliberately ignores _wait_and_score's status.
eval "$(declare -f run_stage | sed '1s/run_stage/audio8_shared_run_stage/')"
run_stage() {
    local rc
    if audio8_shared_run_stage "$@"; then return 0; else rc=$?; fi
    echo "ERROR: stopping this dataset chain after a failed stage" >&2
    exit "${rc}"
}
eval "$(declare -f wait_for_bucket | sed '1s/wait_for_bucket/audio8_shared_wait_for_bucket/')"
wait_for_bucket() {
    if audio8_shared_wait_for_bucket; then return 0; fi
    exit 1
}
eval "$(declare -f _wait_and_score | sed '1s/_wait_and_score/audio8_shared_wait_and_score/')"
AUDIO8_PIPELINE_FAILED=0
_wait_and_score() {
    local i
    local -a verify_args=(--directory "${REPO_ROOT}/results/${MODEL_FOLDER}"
        --model-id "${MODEL_ID}" --revision "${MODEL_REVISION}" --stages "${STAGES}"
        --sim-backend "${SIM_BACKEND}" --utmos-langs "${UTMOS_LANGS}"
        --max-eval-samples "${MAX_EVAL_SAMPLES}")
    if [[ ${#PIDS[@]} -eq 0 ]]; then
        echo "ERROR: filters selected no evaluation datasets" >&2
        AUDIO8_PIPELINE_FAILED=1; return 1
    fi
    if ! audio8_shared_wait_and_score; then AUDIO8_PIPELINE_FAILED=1; return 1; fi
    [[ "${CLONE}" == true ]] && verify_args+=(--voice-clone)
    for i in "${!MANIFEST_NAMES[@]}"; do
        verify_args+=(--expected "${LANGS[$i]}:${MANIFEST_NAMES[$i]}")
    done
    if ! PYTHONPATH="${COMMON_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${AUDIO8_PYTHON}" "${BACKEND_DIR}/run_eval.py" validate-results "${verify_args[@]}"; then
        AUDIO8_PIPELINE_FAILED=1; return 1
    fi
}
cd "${REPO_ROOT}" || exit 2
run_pipeline
[[ "${AUDIO8_PIPELINE_FAILED}" -eq 0 ]]
