#!/bin/bash
# Shared orchestration for the per-backend TTS-eval submit_jobs.sh scripts (HF Jobs + buckets).
#
# A backend script sources this file, then:
#   1. sets TTS_SPACE (+ any backend-specific env) and RUN_EVAL_INJECT
#   2. defines generate_stage()  — the stage-1 (TTS generation) job for one (model, dataset)
#   3. optionally defines resolve_mode() — sets CLONE / MODE_SUFFIX per model (default: no clone)
#   4. sets MODEL_CONFIGS (applying the CLI override) + optionally DATASET_CONFIGS
#   5. calls run_pipeline
#
# The pipeline for each (model, dataset) is: stage 1 (generate, backend-specific) → stage 2
# (transcribe, shared) → stage 3 (SIM, shared; voice-clone models only) → local scoring. Stages
# run as sequential HF Jobs sharing one bucket; a model's combos run in parallel, models one after
# another. All eval scripts are base64-injected into the jobs at runtime (the Space images only
# carry the environment).
#
# Contract — before calling generate_stage()/transcribe_stage()/sim_stage(), run_pipeline sets:
#   MODEL_ID MODEL_SAFE MODEL_FOLDER MODEL_CFG   (MODEL_CFG = raw config line; parse extras from it)
#   DATASET SPLIT ASR_LANG   DATASET_PATH DATASET_SAFE   MODE_SUFFIX CLONE   BUCKET_MANIFEST
#   TTS_IMAGE NAMESPACE_ARG
# plus every shared config var below. Backends reference these in their generate_stage().
# NOTE: DATASET_PATH is per-combo (a DATASET_CONFIGS line may name its own dataset repo), so read
# it inside generate_stage() — never cache it at source time.

# ── Locations ────────────────────────────────────────────────────────────────
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${COMMON_DIR}/.." && pwd)"
BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"   # the sourcing backend script's dir

# ── Shared config (all env-overridable) ─────────────────────────────────────
SPACE="${SPACE:-bezzam/evals}"                                   # shared scorer image (stages 2 & 3)
RESULTS_BUCKET="${RESULTS_BUCKET:-hf-audio/tts_leaderboard_h200}"   # shared HF bucket (per-model folders)
# HF namespace holding your PRIVATE copies of the eval datasets (their licenses forbid
# redistribution; push them with scripts/prepare_{seed_tts,cv3,minimax}_eval.py). Only supplies the
# defaults below — DATASET_PATH / CV3_EVAL_PATH / MINIMAX_EVAL_PATH (or a DATASET_CONFIGS line's 4th
# field) can name any repo.
DATASET_NAMESPACE="${DATASET_NAMESPACE:-bezzam}"
# Default dataset repo, used for every DATASET_CONFIGS line that does not name its own (4th field).
DEFAULT_DATASET_PATH="${DATASET_PATH:-${DATASET_NAMESPACE}/seed_tts_eval}"
# The other eval sets — named explicitly by the CV3 / MiniMax lines of a backend's DATASET_CONFIGS.
CV3_EVAL_PATH="${CV3_EVAL_PATH:-${DATASET_NAMESPACE}/cv3_eval}"
MINIMAX_EVAL_PATH="${MINIMAX_EVAL_PATH:-${DATASET_NAMESPACE}/minimax_eval}"
ORG_NAME="${ORG_NAME:-}"
# Stage-1 (generation) flavor. IMPORTANT: keep this the SAME (h200) for EVERY backend — RTFx is
# measured on the generation stage, so a uniform GPU is required for RTFx to be comparable across
# models. Backends should NOT override it; only change it deliberately for a one-off experiment.
FLAVOR="${FLAVOR:-h200}"
# Stage 2/3 flavors are not timed (WER/SIM are hardware-independent), so a cheaper GPU is fine.
ASR_FLAVOR="${ASR_FLAVOR:-l4x1}"             # stage 2 flavor
SIM_FLAVOR="${SIM_FLAVOR:-l4x1}"             # stage 3 flavor
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"   # -1 = all; set e.g. 8 to smoke-test
# Restrict a run to some of the backend's DATASET_CONFIGS: space-separated asr_language codes
# (field 3 of a config line), e.g. ONLY_LANGS="de". Empty (default) runs every configured combo.
ONLY_LANGS="${ONLY_LANGS:-}"
ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-32}"       # ASR minibatch (lower if it OOMs)
SIM_BATCH_SIZE="${SIM_BATCH_SIZE:-32}"       # SIM minibatch (ignored by the wavlm_seed_tts backend)
# Speaker-similarity embedder. 'xvector': WavLMForXVector (wavlm-base-plus-sv) — cheap but poorly
# discriminative (decent cloning models all land in 0.93-0.97). 'wavlm_seed_tts': seed-tts-eval's
# WavLM-Large + ECAPA-TDNN — comparable with published SIM. IMPORTANT: the two scales are NOT
# interchangeable, so keep this the SAME for every model whose SIM you compare (score_similarity.py
# records the embedder per row as `sim_model` and re-scores rows scored by the other one).
SIM_BACKEND="${SIM_BACKEND:-wavlm_seed_tts}"
# Force every row's SIM to be recomputed. REQUIRED after re-running `generate`: score_similarity.py
# carries prior sims over keyed on `audio_filepath` (to resume crashed runs), and regeneration reuses
# the same wav filenames, so without this the old SIM would silently be kept.
SIM_OVERWRITE="${SIM_OVERWRITE:-false}"
# Same opt-out for stage 2: transcribe.py skips rows that already have a `pred_text`. Set
# ASR_OVERWRITE=true with STAGES="transcribe" to re-transcribe the bucket's existing wavs.
ASR_OVERWRITE="${ASR_OVERWRITE:-false}"
MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-30}" # clip cap for ASR + SIM (0 = no cap)
# Clone each sample's prompt speaker (+ SIM stage) on backends that support it; false = the model's
# own fixed/default voice. Ignored by fixed-voice backends.
VOICE_CLONE="${VOICE_CLONE:-false}"
# Which stages to run (subset of: generate transcribe sim). Skip generate to re-score the bucket's
# existing wavs, e.g. STAGES="transcribe sim" or STAGES="sim". Precedence (resolved in
# run_pipeline): user env STAGES > backend DEFAULT_STAGES > "generate transcribe sim".

# HF Jobs references a Space image as "hf.co/spaces/<id>" and pulls it. NOTE: the Space must be
# PUBLIC (the jobs backend 500s on a private Space image). Images carry only the environment; all
# code is injected at runtime, so there are no secrets to keep private.
SCORER_IMAGE="hf.co/spaces/${SPACE}"

# Datasets: "config split asr_language [dataset_path]". Backends may override before run_pipeline.
# asr_language is the language hint for Qwen3-ASR (stage 2) and, for backends that take one,
# for the TTS model too — so it must be a real language code, not a split name.
# The optional 4th field points a single line at a different dataset repo (default:
# DEFAULT_DATASET_PATH), so one list can mix e.g. Seed-TTS and CV3-Eval combos. It only has to
# expose the same columns the backends read: `text` + `prompt_text`/`prompt_audio` when cloning.
DATASET_CONFIGS=("tts en en")

stage_enabled() { [[ " ${STAGES} " == *" $1 "* ]]; }

# ── Injection helpers ────────────────────────────────────────────────────────
# inject_cmd <local_file> <container_dest> → `... &&` snippet recreating the file in the job.
inject_cmd() {
    local b64
    b64=$(base64 -w0 "$1")
    echo "mkdir -p $(dirname "$2") && echo '${b64}' | base64 -d > $2 &&"
}
# inject_cmd_gz <local_file> <container_path> → same, but gzipped first, for large payloads: a job
# body is ONE `bash -c` argument and Linux caps a single argument at 128 KiB (MAX_ARG_STRLEN).
inject_cmd_gz() {
    local b64
    b64=$(gzip -9c "$1" | base64 -w0)
    echo "mkdir -p $(dirname "$2") && echo '${b64}' | base64 -d | gzip -dc > $2 &&"
}
# inject_dir <dir_relative_to_repo_root> <container_parent> → `... &&` snippet tar+extracting a dir.
inject_dir() {
    local b64
    b64=$(tar --exclude='__pycache__' --exclude='*.pyc' -czf - -C "${REPO_ROOT}" "$1" | base64 -w0)
    echo "mkdir -p $2 && echo '${b64}' | base64 -d | tar -xzf - -C $2 &&"
}

# Snippet putting /app/scripts (shared run_eval helpers, ttfa_probe.py) on the job's PYTHONPATH,
# keeping any PYTHONPATH the image already sets.
SCRIPTS_PYTHONPATH='export PYTHONPATH=/app/scripts${PYTHONPATH:+:${PYTHONPATH}} &&'

# inject_run_eval → `... &&` snippet recreating the sourcing backend's run_eval.py at
# /app/<backend>/ plus the shared scripts/run_eval_utils.py it imports, on PYTHONPATH. Backends set
# RUN_EVAL_INJECT="$(inject_run_eval)".
inject_run_eval() {
    echo "$(inject_cmd "${BACKEND_DIR}/run_eval.py" "/app/$(basename "${BACKEND_DIR}")/run_eval.py")" \
         "$(inject_cmd "${REPO_ROOT}/scripts/run_eval_utils.py" /app/scripts/run_eval_utils.py)" \
         "${SCRIPTS_PYTHONPATH}"
}

# Shared scorer-stage injects (transcribe.py / score_similarity.py live in transformers/).
TRANSCRIBE_INJECT="$(inject_cmd "${REPO_ROOT}/transformers/transcribe.py" /app/transformers/transcribe.py)"
SIM_INJECT="$(inject_cmd "${REPO_ROOT}/transformers/score_similarity.py" /app/transformers/score_similarity.py)"

# ── Default hooks (backends override as needed) ─────────────────────────────
# voice_clone_mode [suffix_prefix] — sets CLONE / MODE_SUFFIX / VOICE_CLONE_FLAG from VOICE_CLONE.
# A backend variant that shares a model id passes its own prefix (e.g. "_rl"), so MODE_SUFFIX
# becomes "_rl" or "_rl_voice_clone" (must match the suffix its run_eval.py writes).
voice_clone_mode() {
    if [[ "${VOICE_CLONE}" == "true" ]]; then
        CLONE="true"; MODE_SUFFIX="${1:-}_voice_clone"; VOICE_CLONE_FLAG="--voice_clone"
    else
        CLONE="false"; MODE_SUFFIX="${1:-}"; VOICE_CLONE_FLAG="--no-voice_clone"
    fi
}

# Default: backends that set SUPPORTS_VOICE_CLONE=true follow VOICE_CLONE; fixed-voice backends have
# no SIM stage. Backends with per-model logic override this (calling voice_clone_mode as needed).
resolve_mode() {
    if [[ "${SUPPORTS_VOICE_CLONE:-false}" == "true" ]]; then
        voice_clone_mode
    else
        CLONE="false"; MODE_SUFFIX=""; VOICE_CLONE_FLAG=""
    fi
}

# Whether the CURRENT model (MODEL_CFG/MODEL_ID, already resolved) can synthesize $1, the dataset's
# language code. Default: assume yes. Backends whose model list mixes languages — e.g. an
# English-only checkpoint alongside a multilingual one — override this so incompatible
# (model, dataset) combos are skipped instead of burning a GPU job on garbage audio.
supports_language() { return 0; }

# ── Transient-infra retry around a stage's `hf jobs run` ────────────────────
# Parallel chains advance in lockstep, so N jobs pull the SAME Space image within seconds and the
# registry can rate-limit the burst (429 → ErrImagePull) before any of our code runs. Retry ONLY
# that class of failure: a job that actually started and exited non-zero must not be re-submitted.
STAGE_RETRIES="${STAGE_RETRIES:-4}"           # attempts after the first, per stage
STAGE_RETRY_SLEEP="${STAGE_RETRY_SLEEP:-90}"  # seconds before the 1st retry; doubles each time

_is_transient_job_failure() {
    grep -qEi '429 Too Many Requests|ErrImagePull|ImagePullBackOff|Back-off pulling image|image specification error' "$1"
}

# run_stage <stage_fn> — run a stage, retrying transient registry failures with exponential backoff.
# Output is tee'd, not buffered, so the chain log still streams `hf jobs run`'s live job logs.
run_stage() {
    local stage_fn="$1" attempt=1 delay="${STAGE_RETRY_SLEEP}" rc out
    out="$(mktemp "${TMPDIR:-/tmp}/tts_stage_XXXXXX.log")"
    while :; do
        "${stage_fn}" 2>&1 | tee "${out}"
        rc=${PIPESTATUS[0]}
        if [[ "${rc}" -eq 0 ]]; then
            rm -f "${out}"
            return 0
        fi
        if ! _is_transient_job_failure "${out}" || [[ "${attempt}" -gt "${STAGE_RETRIES}" ]]; then
            _is_transient_job_failure "${out}" &&
                echo "── ${stage_fn}: still failing to pull the image after ${STAGE_RETRIES} retries; giving up."
            rm -f "${out}"
            return "${rc}"
        fi
        echo "── ${stage_fn}: transient registry/image-pull failure (attempt ${attempt}/${STAGE_RETRIES}); retrying in ${delay}s."
        sleep "${delay}"
        attempt=$((attempt + 1))
        delay=$((delay * 2))
        : > "${out}"   # only the latest attempt's output should be classified
    done
}

# ── Shared stages 2 & 3 (identical for every backend) ───────────────────────
transcribe_stage() {
    local ASR_OVERWRITE_FLAG=""
    [[ "${ASR_OVERWRITE}" == "true" ]] && ASR_OVERWRITE_FLAG=" --overwrite"
    hf jobs run \
        --flavor "${ASR_FLAVOR}" --timeout 4h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${SCORER_IMAGE}" \
        bash -c "
            ${TRANSCRIBE_INJECT}
            python /app/transformers/transcribe.py --manifest_path=${BUCKET_MANIFEST} --asr_language=${ASR_LANG} --asr_batch_size=${ASR_BATCH_SIZE} --max_audio_seconds=${MAX_AUDIO_SECONDS} --device=cuda:0${ASR_OVERWRITE_FLAG}
        "
}
sim_stage() {
    local SIM_OVERWRITE_FLAG=""
    [[ "${SIM_OVERWRITE}" == "true" ]] && SIM_OVERWRITE_FLAG=" --overwrite"
    hf jobs run \
        --flavor "${SIM_FLAVOR}" --timeout 4h --secrets HF_TOKEN \
        --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${SCORER_IMAGE}" \
        bash -c "
            ${SIM_INJECT}
            python /app/transformers/score_similarity.py --manifest_path=${BUCKET_MANIFEST} --sim_backend=${SIM_BACKEND} --batch_size=${SIM_BATCH_SIZE} --max_audio_seconds=${MAX_AUDIO_SECONDS} --device=cuda:0${SIM_OVERWRITE_FLAG}
        "
}

# ── Wait for a model's chains, report failures, then sync + score locally ────
# Uses PIDS/TAGS/CHAIN_LOGS set by run_pipeline. Returns non-zero (and skips scoring) on failure.
_wait_and_score() {
    local i CHAIN_FAILED=0
    # Every combo was skipped (e.g. an English-only model against a zh-only dataset list) — there is
    # nothing to wait for, and scoring would only re-read a stale local copy.
    if [[ ${#PIDS[@]} -eq 0 ]]; then
        echo "No compatible (model, dataset) combos for ${MODEL_ID}; nothing submitted, skipping scoring."
        return 0
    fi
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            CHAIN_FAILED=1
            echo "ERROR: pipeline failed for ${MODEL_ID} on ${TAGS[$i]} — last lines of ${CHAIN_LOGS[$i]}:"
            tail -n 25 "${CHAIN_LOGS[$i]}" | sed 's/^/    /'
        fi
    done
    if [[ "${CHAIN_FAILED}" -eq 1 ]]; then
        echo "One or more job chains failed for ${MODEL_ID}; skipping scoring. See logs in ${LOG_DIR}."
        return 1
    fi
    echo "All jobs finished for ${MODEL_ID}."
    # Clear any stale local copy first: score_results globs **/*.jsonl recursively and keys by
    # (model, dataset), so a leftover manifest would silently overwrite this run's result.
    rm -rf "./results/${MODEL_FOLDER}"
    mkdir -p "./results/${MODEL_FOLDER}"
    # Only the manifests are needed for scoring; the wavs stay in the bucket. A job's last writes can
    # take a moment to show up in the bucket, so retry until this run's manifests are all present.
    # `timeout`: a sync can deadlock on a server-closed connection and would otherwise hang forever.
    local attempt sync_out name
    local -a missing=()
    for attempt in 1 2 3 4 5; do
        sleep 15
        sync_out=$(timeout "${SYNC_TIMEOUT:-600}" hf buckets sync "hf://buckets/${RESULTS_BUCKET}/${MODEL_FOLDER}" "./results/${MODEL_FOLDER}" --exclude "*.wav" 2>&1) ||
            echo "WARNING: bucket sync failed (attempt ${attempt}/5):"$'\n'"${sync_out}"
        missing=()
        for name in "${MANIFEST_NAMES[@]}"; do
            # -s, not -f: an interrupted download leaves a 0-byte file behind.
            [[ -s "./results/${MODEL_FOLDER}/${name}" ]] || missing+=("${name}")
        done
        [[ ${#missing[@]} -eq 0 ]] && break
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "WARNING: ${#missing[@]} manifest(s) of this run not found in the bucket:"
        printf '    %s\n' "${missing[@]}"
    else
        echo "All ${#MANIFEST_NAMES[@]} manifest(s) present."
    fi

    # Score ONE LANGUAGE PER CALL: score_results applies `language` to every manifest it reads and
    # only auto-detects CJK, so a single call would normalize e.g. French with the English normalizer.
    local lang py_list
    local -a seen_langs=()
    for i in "${!LANGS[@]}"; do
        [[ " ${seen_langs[*]} " == *" ${LANGS[$i]} "* ]] || seen_langs+=("${LANGS[$i]}")
    done
    for lang in "${seen_langs[@]}"; do
        py_list=""
        for i in "${!LANGS[@]}"; do
            [[ "${LANGS[$i]}" == "${lang}" ]] || continue
            [[ -s "./results/${MODEL_FOLDER}/${MANIFEST_NAMES[$i]}" ]] || continue
            py_list+="'${MANIFEST_NAMES[$i]}',"
        done
        if [[ -z "${py_list}" ]]; then
            echo "No manifest present for language '${lang}'; skipping its scoring."
            continue
        fi
        echo "── Scoring language '${lang}'"
        PYTHONPATH="${REPO_ROOT}" python -c "
from normalizer.eval_utils import score_results
score_results('$(pwd)/results/${MODEL_FOLDER}', '${MODEL_ID}', language='${lang}', manifests=[${py_list}], sim_backend='${SIM_BACKEND}')
" || echo "WARNING: scoring failed for language '${lang}'."
    done

    # Scoring is scoped to THIS run's combos (where each manifest's language is known); list any
    # other manifests in the folder so they don't silently disappear from the summary.
    local unscored
    unscored=$(find "./results/${MODEL_FOLDER}" -maxdepth 1 -name "MODEL_*.jsonl" -printf "%f\n" 2>/dev/null \
        | grep -vxF "$(printf '%s\n' "${MANIFEST_NAMES[@]}")" || true)
    if [[ -n "${unscored}" ]]; then
        echo "Not scored (not part of this run — use scripts/score_model.py --language <lang>):"
        sed 's/^/    /' <<< "${unscored}"
    fi
}

# ── The orchestrator ─────────────────────────────────────────────────────────
run_pipeline() {
    STAGES="${STAGES:-${DEFAULT_STAGES:-generate transcribe sim}}"   # user env > backend > global
    TTS_IMAGE="hf.co/spaces/${TTS_SPACE}"
    LOG_DIR="${BACKEND_DIR}/job_logs"; mkdir -p "${LOG_DIR}"
    NAMESPACE_ARG=""; [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

    local model_cfg cfg
    for model_cfg in "${MODEL_CONFIGS[@]}"; do
        MODEL_CFG="$model_cfg"                       # full line; generate_stage parses extras
        read -r MODEL_ID _ <<< "$model_cfg"          # MODEL_ID = first whitespace token
        MODEL_SAFE="${MODEL_ID//\//-}"; MODEL_FOLDER="${MODEL_SAFE}"
        resolve_mode                                 # sets CLONE / MODE_SUFFIX (per model)

        # Report the stages that will actually run: SIM is gated on CLONE below, so a fixed-voice
        # model never submits it however STAGES is set — don't advertise it as part of the run.
        local EFFECTIVE_STAGES="" s
        for s in ${STAGES}; do
            [[ "$s" == "sim" && "${CLONE}" != "true" ]] && continue
            EFFECTIVE_STAGES+="${EFFECTIVE_STAGES:+ }$s"
        done

        echo "████████████████████████████████████████████████████████████████████████████████"
        echo "  Evaluating: ${MODEL_ID}  (clone=${CLONE}, stages=${EFFECTIVE_STAGES:-none})"
        echo "████████████████████████████████████████████████████████████████████████████████"

        # LANGS + MANIFEST_NAMES are parallel to PIDS: _wait_and_score groups the finished combos
        # by language so each is scored with its own normalizer.
        PIDS=(); TAGS=(); CHAIN_LOGS=(); LANGS=(); MANIFEST_NAMES=()
        for cfg in "${DATASET_CONFIGS[@]}"; do
            local CFG_DATASET_PATH
            read -r DATASET SPLIT ASR_LANG CFG_DATASET_PATH <<< "$cfg"
            # Per-combo dataset repo (4th field, else the default). DATASET_SAFE feeds the manifest
            # name, so two datasets that share a config/split name still land in distinct manifests.
            DATASET_PATH="${CFG_DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
            DATASET_SAFE="${DATASET_PATH//\//-}"
            # Caller-requested subset (ONLY_LANGS), checked before supports_language so the
            # "not supported" message is never printed for a combo the caller simply skipped.
            if [[ -n "${ONLY_LANGS}" && " ${ONLY_LANGS} " != *" ${ASR_LANG} "* ]]; then
                echo "Skipping ${DATASET}/${SPLIT}: not in ONLY_LANGS='${ONLY_LANGS}'."
                continue
            fi
            # Skip combos the model can't do at all — synthesizing Chinese with an English-only
            # checkpoint produces English-phoneme gibberish and a meaningless (>100%) CER.
            if ! supports_language "${ASR_LANG}"; then
                echo "Skipping ${DATASET}/${SPLIT}: ${MODEL_ID} does not support language '${ASR_LANG}'."
                continue
            fi
            BUCKET_MANIFEST="/results/${MODEL_FOLDER}/MODEL_${MODEL_SAFE}_DATASET_${DATASET_SAFE}_${DATASET}_${SPLIT}${MODE_SUFFIX}.jsonl"
            local CHAIN_LOG="${LOG_DIR}/${MODEL_SAFE}_${DATASET}_${SPLIT}${MODE_SUFFIX}.log"
            echo "Submitting pipeline: model=${MODEL_ID} dataset=${DATASET} split=${SPLIT}  (log: ${CHAIN_LOG})"

            # Sequential stages in a backgrounded subshell (combos run in parallel). `set -e`
            # aborts the chain on the first failed stage; the waiter reports it from the log.
            (
                set -e
                stage_enabled generate   && run_stage generate_stage
                stage_enabled transcribe && run_stage transcribe_stage
                stage_enabled sim && [[ "${CLONE}" == "true" ]] && run_stage sim_stage
                true   # ensure subshell exit status is 0 when the last `&&` chain is a skipped stage
            ) > "${CHAIN_LOG}" 2>&1 &
            PIDS+=("$!"); TAGS+=("${DATASET}/${SPLIT}"); CHAIN_LOGS+=("${CHAIN_LOG}")
            LANGS+=("${ASR_LANG}"); MANIFEST_NAMES+=("$(basename "${BUCKET_MANIFEST}")")
        done

        if [ -n "$ORG_NAME" ]; then
            echo "For live status see: https://huggingface.co/organizations/${ORG_NAME}/settings/jobs"
        else
            echo "For live status see: https://huggingface.co/settings/jobs"
        fi

        _wait_and_score || true   # keep going to the next model on failure
    done
}
