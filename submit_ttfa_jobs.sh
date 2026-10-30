#!/bin/bash
# Time-to-first-audio (TTFA) probes for every TTFA_TARGETS entry, as one HF Job per target.
#
# Usage:  HF_TOKEN=hf_... bash submit_ttfa_jobs.sh [label|model_id|backend ...]
#         HF_TOKEN=hf_... bash submit_ttfa_jobs.sh breeze-tts kokoro      # just these
#         TTFA_SAMPLES=50 VOICE_CLONE=true bash submit_ttfa_jobs.sh       # quick run (default: whole split)
#
# Separate from the per-backend submit_jobs.sh scripts because TTFA is measured on different terms:
#   * It is a PER-REQUEST latency, so it is measured at BATCH SIZE 1 (the eval measures RTFx at each
#     backend's real batch size; the two can differ by an order of magnitude).
#   * It probes ONE split (the whole of it by default, for a stable p95), not a full eval.
#   * It writes ONLY a JSON sidecar (no wavs, no manifest), so it is safe to run against models
#     whose results are already in the bucket.
#
# Each job runs the backend's own run_eval.py with --ttfa_probe, so the probe measures exactly the
# code path the real eval uses.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/tts_jobs_common.sh"

# ── Probe configuration ──────────────────────────────────────────────────────
# -1 = the whole split (500 for CV3-Eval zero_shot/en); a positive N probes N evenly spaced rows.
# The median converges quickly; the full split is what gives a trustworthy p90/p95 tail.
TTFA_SAMPLES="${TTFA_SAMPLES:--1}"
# One dataset for every model, so the numbers are comparable: CV3-Eval English.
TTFA_DATASET_PATH="${TTFA_DATASET_PATH:-${CV3_EVAL_PATH}}"   # CV3_EVAL_PATH: set by tts_jobs_common.sh
TTFA_DATASET="${TTFA_DATASET:-zero_shot}"
TTFA_SPLIT="${TTFA_SPLIT:-en}"
TTFA_LANG="${TTFA_LANG:-en}"
# Probe every model in the SAME voice mode, or the numbers are not comparable: a cloning run pays
# reference encoding that a fixed-voice run does not. Default false = each model's own non-cloning
# voice. Backends with no clone mode ignore this.
VOICE_CLONE="${VOICE_CLONE:-false}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
VLLM_PORT="${VLLM_PORT:-8000}"

# ── Hardware ────────────────────────────────────────────────────────────────
# Latency is hardware-bound, so a TTFA is only meaningful next to others measured on the SAME
# hardware. GPU probes default to TTFA_GPU_FLAVOR, NOT the eval's h200 (FLAVOR): at batch size 1
TTFA_GPU_FLAVOR="a100-large"
TTFA_CPU_FLAVOR="cpu-upgrade"
#
# TTFA_DEVICE=cpu picks a CPU flavor and passes --device=cpu to the backend. The sidecar filename
# is suffixed with the flavor, so CPU and GPU results coexist per model.
# NOT every backend runs on CPU (e.g. breeze-tts's fast path, and the voxtral-tts/higgs servers).
TTFA_DEVICE="${TTFA_DEVICE:-cuda}"

# TTFA_DEVICE is a TORCH device ("cuda", "cpu", "cuda:0") — TTFA_FLAVOR is the hardware. Fail fast
# if a flavor is passed here by mistake.
case "${TTFA_DEVICE}" in
    cpu|cuda|cuda:[0-9]*) ;;
    *)
        echo "ERROR: TTFA_DEVICE='${TTFA_DEVICE}' is not a torch device." >&2
        echo "       Expected: cpu, cuda, or cuda:N." >&2
        echo "       To choose HARDWARE use TTFA_FLAVOR instead, e.g." >&2
        echo "         TTFA_FLAVOR=${TTFA_GPU_FLAVOR} bash submit_ttfa_jobs.sh ${*:-<backend>}" >&2
        echo "         TTFA_DEVICE=cpu  bash submit_ttfa_jobs.sh ${*:-<backend>}   # picks a cpu-* flavor" >&2
        exit 1 ;;
esac

# TTFA_FLAVOR overrides either default; its sidecars are then not picked up by open_results_pr.py.
if [[ "${TTFA_DEVICE}" == "cpu" ]]; then
    FLAVOR="${TTFA_FLAVOR:-${TTFA_CPU_FLAVOR}}"
else
    FLAVOR="${TTFA_FLAVOR:-${TTFA_GPU_FLAVOR}}"
fi

# The sidecar suffix is the FLAVOR, which must imply the device: CPU inference on a GPU flavor
# would be filed under __<gpu flavor> and read as a GPU result. Refuse rather than mislabel.
case "${TTFA_DEVICE}:${FLAVOR}" in
    cpu:cpu-*) ;;
    cpu:*)
        echo "ERROR: TTFA_DEVICE=cpu with a non-CPU flavor '${FLAVOR}'." >&2
        echo "       The sidecar would be tagged __${FLAVOR} and misread as a GPU measurement." >&2
        echo "       Use a cpu-* flavor, or set TTFA_FLAVOR explicitly to something CPU-named." >&2
        exit 1 ;;
    cuda*:cpu-*)
        echo "ERROR: TTFA_DEVICE=${TTFA_DEVICE} on CPU flavor '${FLAVOR}' — there is no GPU there." >&2
        exit 1 ;;
esac

# Index-less "cuda" is rejected by some backends (breeze-tts calls torch.cuda.set_device); every
# backend's own submit_jobs.sh passes cuda:0, so match that.
[[ "${TTFA_DEVICE}" == "cuda" ]] && TTFA_DEVICE="cuda:0"

# Phonon is CPU-only and opt-in; keep it out of the default GPU target list.
PHONON_SELECTED=false
for selected in "$@"; do
    case "$selected" in
        phonon|phonon-31m-en|phonon-40m-en|gradium/phonon-31m-en|gradium/phonon-40m-en) PHONON_SELECTED=true ;;
    esac
done
PHONON_TTFA_MODE="${PHONON_TTFA_MODE:-streaming}"
if ${PHONON_SELECTED}; then
    [[ "$TTFA_DEVICE:$FLAVOR:$TTFA_DATASET:$TTFA_SPLIT:$TTFA_LANG:$VOICE_CLONE" == cpu:cpu-upgrade:zero_shot:en:en:false \
        && ( "$TTFA_SAMPLES" == -1 || "$TTFA_SAMPLES" == 500 ) ]] || {
        echo "Phonon requires CPU/cpu-upgrade, all 500 CV3 English samples, and no voice cloning." >&2; exit 1;
    }
fi

LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ttfa_job_logs"
mkdir -p "${LOG_DIR}"
NAMESPACE_ARG=""; [ -n "$ORG_NAME" ] && NAMESPACE_ARG="--namespace ${ORG_NAME}"

# run_eval.py imports scripts/run_eval_utils.py and, for the probe, scripts/ttfa_probe.py. Gzipped
# because one job body stacks those and run_eval.py, and must fit in a single 128 KiB `bash -c`
# argument.
TTFA_INJECT="$(inject_cmd_gz "${REPO_ROOT}/scripts/ttfa_probe.py" /app/scripts/ttfa_probe.py) $(inject_cmd_gz "${REPO_ROOT}/scripts/run_eval_utils.py" /app/scripts/run_eval_utils.py) ${SCRIPTS_PYTHONPATH}"

# ── Targets: "label|backend_dir|space|model_id|extra flags|engine_tag" ──────
# `label` is a free-form name you select on; `backend_dir` MUST be the real directory under the
# repo root (several labels can share one backend, e.g. the two inflect-v2 models).
#
# `engine_tag` (field 6, optional) marks the SAME model measured through a DIFFERENT inference
# engine; it is inserted before the flavor suffix so the sidecars do not overwrite each other
# (TTFA_MODEL_fishaudio-s2-pro_..._en__sglang__a100-large.json).
#
# Select by label, model id, or backend dir: `bash submit_ttfa_jobs.sh inflect-micro-v2`.
# An engine-tagged target is NOT selectable by its backend dir (it borrows another backend's
# client); selecting by model id runs every engine for it.
# Extra flags mirror the default MODEL_CONFIGS entry of the backend's own submit_jobs.sh. Batch
# sizes are absent: the probe forces batch 1.
TTFA_TARGETS=(
    "breeze-tts|breeze-tts|bezzam/evals-breeze|BreezeBlue/Breeze-TTS-2|--language=${TTFA_LANG} --fast_all"
    "chatterbox|chatterbox|bezzam/evals-chatterbox|ResembleAI/chatterbox|--language=${TTFA_LANG}"
    "cosyvoice3|cosyvoice3|bezzam/evals-cosy|FunAudioLLM/Fun-CosyVoice3-0.5B-2512|--llm_checkpoint=llm.rl.pt"
    "fishaudio-s2-pro|fishaudio-s2-pro|bezzam/evals-fish|fishaudio/s2-pro|"
    # Drama 3 under SGLang-Omni (GPU only): submit_one starts the server on the pinned snapshot.
    "fishaudio-drama-3|fishaudio-drama-3|${DRAMA_SPACE:-bezzam/evals-drama3}|fishaudio/drama-3|--host=127.0.0.1 --port=${SGLANG_PORT}|sglang"
    # S2-Pro under SGLang-Omni, its streaming engine (the in-process fish-speech backend above
    # yields once per utterance, so it only gives a whole-utterance clock). Reuses higgs/run_eval.py
    # (a plain client of sgl-omni's /v1/audio/speech), the higgs) case below and the higgs image.
    # Flags are S2-Pro's documented defaults; --top_k=30 is required (S2-Pro rejects >30).
    "s2pro-sglang|higgs|${S2PRO_SGL_SPACE:-bezzam/evals-higgs}|fishaudio/s2-pro|--host=127.0.0.1 --port=${SGLANG_PORT} --max_new_tokens=2048 --temperature=0.8 --top_p=0.8 --top_k=30|sglang"
    "index-tts|index-tts|bezzam/evals-indextts|IndexTeam/IndexTTS-2.5|--language=${TTFA_LANG}"
    "inflect-nano-v1|inflect-nano|bezzam/evals-inflect|owensong/Inflect-Nano-v1|"
    "inflect-micro-v2|inflect-v2|bezzam/evals-inflect-v2|owensong/Inflect-Micro-v2|"
    "inflect-nano-v2|inflect-v2|bezzam/evals-inflect-v2|owensong/Inflect-Nano-v2|"
    "kokoro|kokoro|bezzam/evals-kokoro|hexgrad/Kokoro-82M|"
    # GPU only: liquid-audio hard-codes its detokenizer to CUDA (audiocpp-lfm2-audio below covers CPU).
    "lfm2-audio|lfm2-audio|bezzam/evals-lfm2-audio|LiquidAI/LFM2.5-Audio-1.5B|--voice=us_female"
    # Streams: the codec decodes the first speech frame as soon as it is sampled (lfm3-audio/run_eval.py).
    "lfm3-audio|lfm3-audio|bezzam/evals-lfm3-audio|LiquidAI/LFM3-Audio-2B|--voice=female"
    "magpie-tts|magpie-tts|bezzam/evals-magpie|nvidia/magpie_tts_multilingual_357m|--speaker=Sofia --language=${TTFA_LANG}"
    # GPU only (the engine's fused kernels are Triton). Streams: the engine emits PCM per 80 ms frame.
    "maya2|maya2|bezzam/evals-maya2|maya-research/maya2-native|--voice=Ahana"
    # GPU only (the model's code hard-codes CUDA autocast). Streams: the VAE decoder's streaming path
    # decodes the first latent patch as soon as it is sampled (ming-tts/run_eval.py).
    "ming-tts|ming-tts|bezzam/evals-ming-tts|inclusionAI/Ming-3.0-tts-0.5B|"
    "moss-tts|moss-tts|bezzam/evals-moss|OpenMOSS-Team/MOSS-TTS|"
    "neutts-nano|neutts-nano|bezzam/evals-neutts|neuphonic/neutts-nano|"
    # Same image, quantized backbone. NeuTTS only streams on the gguf path; the torch target above
    # is whole-utterance and is what the leaderboard's WER/RTFx are measured on.
    "neutts-nano-gguf|neutts-nano|bezzam/evals-neutts|neuphonic/neutts-nano-q8-gguf|"
    "omnivoice|omnivoice|bezzam/evals-omnivoice|k2-fsa/OmniVoice|--language=${TTFA_LANG}"
    # ONNX Runtime: the CUDA EP on the GPU flavor; TTFA_DEVICE=cpu runs the package's own 1-thread CPU session.
    "paradee|paradee|bezzam/evals-paradee|sahilmahendrakar/Paradee-8M-v1.0|"
    "pocket-tts|pocket-tts|bezzam/evals-pocket|kyutai/pocket-tts|--language=${TTFA_LANG}"
    "qwen3tts|qwen3tts|bezzam/evals-qwentts|Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice|--speaker=Aiden --mode=custom_voice --language=${TTFA_LANG}"
    # The same checkpoint under andimarafioti/faster-qwen3-tts (CUDA graphs, incremental audio).
    # Qwen's own qwen-tts has no streaming API, so the target above is whole-utterance; keep both.
    # Its own directory and Space because the two pin incompatible transformers versions.
    # --chunk_size is pinned to the library default: it is the floor under TTFA, so numbers at a
    # different chunk_size are not comparable (it is recorded in the sidecar).
    "qwen3tts-fast|qwen3tts-fast|${QWEN3TTS_FAST_SPACE:-bezzam/evals-faster-qwentts}|Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice|--speaker=Aiden --mode=custom_voice --language=${TTFA_LANG} --chunk_size=12|fasterq3"
    # faster-qwen3-tts's ggml runtime (qwentts.cpp), with its own CPU image. The only qwen3tts
    # target that can serve TTFA_DEVICE=cpu:  TTFA_DEVICE=cpu bash submit_ttfa_jobs.sh qwen3tts-ggml
    # --quant=BF16 is the lossless GGUF, so this measures the same model as the targets above;
    # Q8_0/Q4_K_M would be a DIFFERENT model.
    "qwen3tts-ggml|qwen3tts-fast|${QWEN3TTS_GGML_SPACE:-bezzam/evals-ggml-qwentts}|Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice|--speaker=Aiden --mode=custom_voice --language=${TTFA_LANG} --chunk_size=12 --backend=ggml --quant=BF16|ggml"
    # Control for the row above: the same qwentts.cpp runtime on the GPU flavor, so runtime and hardware
    # effects can be separated. Its image differs from the CPU one only in the base image and wheels
    # (+cu128 vs +cpu); run_eval.py refuses a wheel that does not match the device.
    "qwen3tts-ggml-gpu|qwen3tts-fast|${QWEN3TTS_GGML_GPU_SPACE:-bezzam/evals-ggml-gpu-qwentts}|Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice|--speaker=Aiden --mode=custom_voice --language=${TTFA_LANG} --chunk_size=12 --backend=ggml --quant=BF16|ggmlgpu"
    # Qwen3-TTS and Kokoro through Moondream's Photon runtime (see photon/run_eval.py), sharing one
    # image. Photon's first chunk is the first AUDIBLE audio (leading silence is dropped). Flags
    # mirror the reference targets so only the engine differs. Kokoro's number includes G2P, as the
    # kokoro target's does. photon-kokoro also runs on CPU.
    "photon-qwen3tts|photon|${PHOTON_SPACE:-bezzam/evals-photon}|Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice|--voice=Aiden --language=${TTFA_LANG}|photon"
    "photon-qwen3tts-0.6b|photon|${PHOTON_SPACE:-bezzam/evals-photon}|Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice|--voice=Aiden --language=${TTFA_LANG}|photon"
    "photon-kokoro|photon|${PHOTON_SPACE:-bezzam/evals-photon}|hexgrad/Kokoro-82M|--voice=af_heart --lang_code=a|photon"
    # The same models through 0xShug0/audio.cpp (ggml C++), served by audiocpp_server's SSE
    # streaming endpoint (see audiocpp/run_eval.py).
    # The Space's image tag is per device:  TTFA_DEVICE=cpu AUDIOCPP_SPACE=bezzam/evals-audiocpp-cpu ...
    "audiocpp-qwen3tts|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice|--language=${TTFA_LANG}|audiocpp"
    "audiocpp-voxcpm2|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|openbmb/VoxCPM2||audiocpp"
    "audiocpp-supertonic|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|Supertone/supertonic-3|--language=${TTFA_LANG}|audiocpp"
    "audiocpp-pocket-tts|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|kyutai/pocket-tts||audiocpp"
    "audiocpp-breeze-tts|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|BreezeBlue/Breeze-TTS-2|--language=${TTFA_LANG}|audiocpp"
    "audiocpp-omnivoice|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|k2-fsa/OmniVoice|--language=${TTFA_LANG}|audiocpp"
    "audiocpp-fish-s2-pro|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|fishaudio/s2-pro||audiocpp"
    "audiocpp-vibevoice|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|microsoft/VibeVoice-1.5B||audiocpp"
    "audiocpp-voxtral-tts|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|mistralai/Voxtral-4B-TTS-2603||audiocpp"
    # Streaming models that are only on the TTFA tab (no WER/RTFx backend).
    "audiocpp-kugelaudio|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|kugelaudio/kugelaudio-0-open||audiocpp"
    "audiocpp-neutts-2e|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|neuphonic/neutts-2e||audiocpp"
    "audiocpp-audio8|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|Edge0/Audio8-TTS-Preview-0.6b||audiocpp"
    "audiocpp-lfm2-audio|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|LiquidAI/LFM2.5-Audio-1.5B||audiocpp"
    "audiocpp-soprano|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|ekwek/Soprano-1.1-80M||audiocpp"
    "audiocpp-voxcpm1|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|openbmb/VoxCPM-0.5B||audiocpp"
    "audiocpp-dots-soar|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|dots-studio/dots.tts-soar||audiocpp"
    "audiocpp-dots-mf|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|dots-studio/dots.tts-mf||audiocpp"
    # Clone-only: always probed with the row's prompt (sidecar suffixed _voice_clone).
    "audiocpp-confucius4|audiocpp|${AUDIOCPP_SPACE:-bezzam/evals-audiocpp}|netease-youdao/Confucius4-TTS|--language=${TTFA_LANG}|audiocpp"
    "supertonic|supertonic|bezzam/evals-super|Supertone/supertonic-3|--voice=M1 --lang=${TTFA_LANG}"
    "transformers|transformers|bezzam/evals|microsoft/speecht5_tts|--language=${TTFA_LANG}"
    # Second model on the shared transformers backend; not streaming (whole-utterance clock).
    # --language is required: it sets both src_lang and tgt_lang (see _seamless_lang).
    "seamless-m4t-v2|transformers|bezzam/evals|facebook/seamless-m4t-v2-large|--language=${TTFA_LANG}"
    "vibevoice_realtime|vibevoice_realtime|bezzam/evals-vibevoice|microsoft/VibeVoice-Realtime-0.5B|"
    "voxcpm2|voxcpm2|bezzam/evals-voxcpm|openbmb/VoxCPM2|"
    # HTTP CLIENTS of an inference server the job starts itself (see the preamble in submit_one);
    # --server_pid makes a dead server fail the wait immediately.
    "voxtral-tts|voxtral-tts|bezzam/evals-voxtral|mistralai/Voxtral-4B-TTS-2603|--base_url=http://localhost:${VLLM_PORT} --server_pid=\${SERVER_PID}"
    "higgs|higgs|bezzam/evals-higgs|bosonai/higgs-tts-3-4b|--host=127.0.0.1 --port=${SGLANG_PORT}"
)
if ${PHONON_SELECTED}; then
    TTFA_TARGETS+=(
        "phonon-31m-en|phonon|${PHONON_SPACE:-Steveeeeeeen/evals-phonon-040}|gradium/phonon-31m-en||"
        "phonon-40m-en|phonon|${PHONON_SPACE:-Steveeeeeeen/evals-phonon-040}|gradium/phonon-40m-en||"
    )
fi

# Backends whose run_eval.py has no --voice_clone flag at all; passing one would be an error.
NO_CLONE_FLAG="inflect-nano inflect-v2 kokoro lfm2-audio lfm3-audio magpie-tts paradee phonon photon qwen3tts qwen3tts-fast supertonic vibevoice_realtime voxtral-tts"

# run_eval.py writes to its own output roots (outside Git, mode 0700) and names the sidecar itself,
# so Phonon skips the generic run/rename/copy below; `preamble` is submit_one's CPU thread cap.
# `model_ids` (space-separated) are probed one after another in ONE job: cpu-upgrade hosts vary a lot
# between jobs, so only models measured on the same host can be ranked against each other.
submit_phonon() {
    local model_ids="$1" space="$2" inject="$3" preamble="$4" dataset_safe="${TTFA_DATASET_PATH//\//-}"
    local engine="" model_id model_safe filename runner probes="" n_models=0
    [[ "$PHONON_TTFA_MODE" != streaming ]] || engine="__ptts-stream"
    for model_id in ${model_ids}; do
        model_safe="${model_id//\//-}"
        filename="TTFA_MODEL_${model_safe}_DATASET_${dataset_safe}_zero_shot_en${engine}__cpu-upgrade.json"
        local args=(--model_id "$model_id" --dataset_path "$TTFA_DATASET_PATH"
            --dataset zero_shot --split en --device cpu --ttfa_probe 500 --ttfa_flavor cpu-upgrade
            --ttfa_mode "$PHONON_TTFA_MODE" --max_eval_samples -1 --warmup_steps 0)
        [[ -z "${TTFA_DATASET_REVISION:-}" ]] || args+=(--dataset_revision "$TTFA_DATASET_REVISION")
        printf -v runner '%q ' python /app/phonon/run_eval.py "${args[@]}"
        # Copied as soon as each model finishes, so a later model's failure keeps the earlier sidecars.
        probes+="
        echo '=== ${model_id}'
        ${runner} --threads \"\${_cpuq}\" --output_root /tmp/phonon/quality --ttfa_output_root /tmp/phonon/ttfa
        mkdir -p '/results/${model_safe}'
        cp '/tmp/phonon/ttfa/results/${model_safe}/${filename}' '/results/${model_safe}/'"
        n_models=$((n_models + 1))
    done
    local command="set -euo pipefail
        ${inject}
        ${TTFA_INJECT} true || exit 1
        ${preamble}
        ${probes}
    "
    local -a job=(hf jobs run --flavor cpu-upgrade --timeout "${n_models}h" --secrets HF_TOKEN)
    [[ -z "$ORG_NAME" ]] || job+=(--namespace "$ORG_NAME")
    "${job[@]}" --volume "hf://buckets/${RESULTS_BUCKET}:/results" "hf.co/spaces/${space}" bash -c "$command"
}

submit_one() {
    local backend="$1" space="$2" model_id="$3" extra="$4" engine="$5"
    local model_safe="${model_id//\//-}"
    # Drama 3's FP8 experts need native FP8 (sm_89+), so it defaults to h200 instead of the shared
    # A100 default. An explicit TTFA_FLAVOR is kept (and rejected below if it is not h200).
    local FLAVOR="${FLAVOR}"
    if [[ "${backend}" == fishaudio-drama-3 && -z "${TTFA_FLAVOR:-}" && "${TTFA_DEVICE}" != cpu ]]; then
        FLAVOR=h200
        echo "NOTE: ${model_id} runs on h200 (FP8 experts are unsupported on ${TTFA_GPU_FLAVOR})."
    fi
    # Sidecar suffix: engine first (when the target names one), then always the flavor.
    local sidecar_suffix="${engine:+__${engine}}__${FLAVOR}"
    local image="hf.co/spaces/${space}"
    local inject preamble postamble clone_flag=""

    inject="$(inject_cmd_gz "${REPO_ROOT}/${backend}/run_eval.py" "/app/${backend}/run_eval.py")"

    # Every backend takes --device (run_eval_utils.add_common_args); HTTP clients (voxtral-tts,
    # higgs) accept and ignore it, since their GPU belongs to the server process.
    local device_flag="--device=${TTFA_DEVICE}"
    [[ " ${NO_CLONE_FLAG} " == *" ${backend} "* ]] || \
        clone_flag=$([[ "${VOICE_CLONE}" == "true" ]] && echo "--voice_clone" || echo "--no-voice_clone")

    # HTTP-client backends start their own inference server first, as their generate_stage does.
    # On failure the FULL server log is dumped (the engine's traceback explains the crash).
    preamble=""; postamble=""

    # CPU runs: cap every thread pool BEFORE python starts. Nested parallelism (torch intra-op
    # threads each calling into a multi-threaded BLAS) otherwise oversubscribes the vCPUs and
    # inflates TTFA several-fold.
    if [[ "${TTFA_DEVICE}" == "cpu" ]]; then
        # TTFA_CPU_THREADS wins when set (a container can report the HOST's nproc with no cpu.max).
        preamble="
            _cpuq=\"${TTFA_CPU_THREADS:-}\"
            if [ -z \"\${_cpuq}\" ]; then
                _cpuq=\$(awk '{ if (\$1==\"max\") print 0; else printf \"%d\", (\$1/\$2) }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
                [ \"\${_cpuq:-0}\" -gt 0 ] 2>/dev/null || _cpuq=\$(nproc)
            fi
            export OMP_NUM_THREADS=\${_cpuq} MKL_NUM_THREADS=\${_cpuq} OPENBLAS_NUM_THREADS=\${_cpuq}
            echo \"CPU run: nproc=\$(nproc) cgroup_quota=\${_cpuq} -> OMP_NUM_THREADS=\${_cpuq}\"
            # The host CPU can vary between cpu-upgrade jobs and moves TTFA a lot; log it to explain potential outliers.
            echo \"CPU model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')\"
        "
        # Guard against a quoting slip silently emptying the preamble (the probe would run uncapped).
        if [[ "${preamble}" != *OMP_NUM_THREADS* ]]; then
            echo "ERROR: CPU thread-cap preamble is empty or truncated — check its quoting." >&2
            return 1
        fi
    fi

    if [[ "${backend}" == phonon ]]; then
        submit_phonon "${model_id}" "${space}" "${inject}" "${preamble}"
        return
    fi

    case "${backend}" in
        fishaudio-drama-3)
            # Explicit TTFA_FLAVOR / TTFA_DEVICE=cpu: the FP8 experts need sm_89+ (Triton rejects fp8e4nv on A100).
            if [[ "${FLAVOR}" != h200 ]]; then
                echo "ERROR: fishaudio/drama-3 (FP8 experts) does not run on ${FLAVOR}; use TTFA_FLAVOR=h200." >&2
                return 1
            fi
            # Same server start as fishaudio-drama-3/submit_jobs.sh's generate_stage: serve the
            # pinned snapshot (sglang_omni serve has no --revision). The probe writes voice-clone
            # references into a temp dir under results/, the server's allowed media root.
            preamble="${preamble}
            set -e
            mkdir -p /app/fishaudio-drama-3/results
            SNAPSHOT=\$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download(\"${model_id}\", revision=\"e0a9218aa48888270f06ef1ed9ea67622c0e17d2\"))')
            python -m sglang_omni.cli serve --model-path \$SNAPSHOT --model-name ${model_id} --host 127.0.0.1 --port ${SGLANG_PORT} \
                --allowed-local-media-path /app/fishaudio-drama-3/results > /app/fishaudio-drama-3/results/server.log 2>&1 &
            SERVER_PID=\$!
            trap 'kill \$SERVER_PID 2>/dev/null || true' EXIT"
            postamble="|| { echo '--- sglang_omni serve log (full) ---'; cat /app/fishaudio-drama-3/results/server.log; exit 1; }"
            ;;
        higgs)
            # S2-Pro checkpoints carry no `architectures`, so sgl-omni cannot infer the pipeline
            # config: write the two-line --config its cookbook uses (not shipped in the package).
            local serve_config=""
            if [[ "${model_id}" == "fishaudio/s2-pro" ]]; then
                serve_config="--config /tmp/sgl_omni_pipeline.yaml"
                preamble="${preamble}
            echo 'config_cls: S2ProPipelineConfig' >  /tmp/sgl_omni_pipeline.yaml
            echo 'model_path: ${model_id}'         >> /tmp/sgl_omni_pipeline.yaml"
            fi
            preamble="${preamble}
            set -e
            mkdir -p /app/higgs/results
            # /tmp: the probe writes voice-clone references into a tempfile.mkdtemp() dir, and the
            # server refuses to read references outside the allowed root.
            /opt/omni/bin/sgl-omni serve --model-path ${model_id} ${serve_config} --host 127.0.0.1 --port ${SGLANG_PORT} \
                --allowed-local-media-path /tmp > /app/higgs/results/server.log 2>&1 &
            SERVER_PID=\$!
            trap 'kill \$SERVER_PID 2>/dev/null || true' EXIT"
            postamble="|| { echo '--- sgl-omni serve log (full) ---'; cat /app/higgs/results/server.log; exit 1; }"
            ;;
        voxtral-tts)
            # See voxtral-tts/submit_jobs.sh: the forward-compat libcuda must stay unset on h200,
            # or every StageEngineCoreProc dies at init_device() with CUDA error 803.
            preamble="${preamble}
            set -e
            unset VLLM_ENABLE_CUDA_COMPATIBILITY
            vllm serve ${model_id} --omni --port ${VLLM_PORT} > /app/voxtral-tts/vllm_serve.log 2>&1 &
            SERVER_PID=\$!
            trap 'kill \$SERVER_PID 2>/dev/null || true' EXIT"
            postamble="|| { echo '--- vllm serve log (full) ---'; cat /app/voxtral-tts/vllm_serve.log; exit 1; }"
            ;;
    esac

    hf jobs run \
        --flavor "${FLAVOR}" --timeout "${TTFA_TIMEOUT:-8h}" --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${image}" \
        bash -c "
            ${inject}
            ${TTFA_INJECT}
            ${preamble}
            cd /app/${backend} && python run_eval.py \
                --model_id=${model_id} \
                --dataset_path=${TTFA_DATASET_PATH} \
                --dataset=${TTFA_DATASET} \
                --split=${TTFA_SPLIT} \
                --ttfa_probe=${TTFA_SAMPLES} \
                ${device_flag} \
                ${clone_flag} \
                ${extra} ${postamble}
            # Tag the sidecar with engine+flavor so probes that differ only in those do not collide.
            for f in results/${model_safe}/TTFA_MODEL_*.json; do
                [ -e \"\$f\" ] || continue
                case \"\$f\" in *${sidecar_suffix}.json) ;; *) mv \"\$f\" \"\${f%.json}${sidecar_suffix}.json\" ;; esac
            done
            mkdir -p /results
            cp -r results/${model_safe} /results/
        "
}

# ── Run: one job per target, in parallel, one log each ───────────────────────
SELECT=("$@")
PIDS=(); TAGS=(); MIDS=(); LOGS=(); PHONON_MODELS=()
_n_desc=$([[ "${TTFA_SAMPLES}" -lt 0 ]] && echo "the WHOLE split" || echo "${TTFA_SAMPLES} samples")
echo "TTFA probe: ${_n_desc} of ${TTFA_DATASET_PATH}/${TTFA_DATASET}/${TTFA_SPLIT}"
echo "            flavor=${FLAVOR}, device=${TTFA_DEVICE}, batch size 1, voice_clone=${VOICE_CLONE}"
echo "            sidecars are suffixed __${FLAVOR} so results from different hardware coexist"
echo

for target in "${TTFA_TARGETS[@]}"; do
    IFS='|' read -r label backend space model_id extra engine <<< "${target}"
    if [ ${#SELECT[@]} -gt 0 ]; then
        # Backend-dir matching is skipped for engine-tagged targets — see the note above TTFA_TARGETS.
        _sel=(-e "${label}" -e "${model_id}")
        [ -z "${engine}" ] && _sel+=(-e "${backend}")
        printf '%s\n' "${SELECT[@]}" | grep -qxF "${_sel[@]}" || continue
    fi
    if [ ! -f "${REPO_ROOT}/${backend}/run_eval.py" ]; then
        echo "ERROR: target '${label}' names backend dir '${backend}', but" >&2
        echo "       ${REPO_ROOT}/${backend}/run_eval.py does not exist." >&2
        echo "       Field 2 must be the real directory; field 1 is the free-form label." >&2
        exit 1
    fi
    # Phonon: every selected model is probed in one shared job, submitted after this loop.
    if [[ "${backend}" == phonon ]]; then
        PHONON_MODELS+=("${model_id}"); PHONON_JOB_SPACE="${space}"
        continue
    fi
    log="${LOG_DIR}/${label}_${model_id//\//-}.log"
    echo "Submitting TTFA probe: ${model_id}  (label=${label}, backend=${backend}, log: ${log})"
    [[ "${backend}" == fishaudio-drama-3 && -z "${TTFA_FLAVOR:-}" && "${TTFA_DEVICE}" != cpu ]] &&
        echo "  NOTE: ${model_id} runs on h200, not ${FLAVOR}: its FP8 experts need sm_89+."
    ( submit_one "${backend}" "${space}" "${model_id}" "${extra}" "${engine}" ) > "${log}" 2>&1 &
    # Exact log path: two targets can share a model id (one per engine).
    PIDS+=("$!"); TAGS+=("${label}"); MIDS+=("${model_id}"); LOGS+=("${log}")
done

if [ ${#PHONON_MODELS[@]} -gt 0 ]; then
    log="${LOG_DIR}/phonon_${FLAVOR}.log"
    echo "Submitting TTFA probe: ${PHONON_MODELS[*]}  (one job, same host; label=phonon, log: ${log})"
    ( submit_one phonon "${PHONON_JOB_SPACE}" "${PHONON_MODELS[*]}" "" "" ) > "${log}" 2>&1 &
    PIDS+=("$!"); TAGS+=("phonon"); MIDS+=("${PHONON_MODELS[*]}"); LOGS+=("${log}")
fi

if [ ${#PIDS[@]} -eq 0 ]; then
    echo "No targets matched. Known labels:"
    printf '  %s\n' "${TTFA_TARGETS[@]}" | cut -d'|' -f1 | tr -d '"' | sort -u
    exit 1
fi

# _job_id <log> — the HF job id `hf jobs run` printed into <log>, or nothing.
_job_id() {
    grep -oE "id: [0-9a-f]{16,}" "$1" 2>/dev/null | head -1 | awk '{print $2}'
}

# _append_job_log <log> — append the HF job's own log (job id parsed from <log>) to <log>.
_append_job_log() {
    local jid
    jid=$(_job_id "$1")
    [ -n "${jid}" ] && hf jobs logs "${jid}" >> "$1" 2>/dev/null || true
}

# _job_stage <job_id> — the job's stage on HF (COMPLETED, ERROR, RUNNING, ...), or nothing if it
# could not be read.
_job_stage() {
    hf jobs inspect "$1" 2>/dev/null | python3 -c \
        "import sys, json; d = json.load(sys.stdin); d = d[0] if isinstance(d, list) else d; print((d.get('status') or {}).get('stage') or '')" \
        2>/dev/null
}

# _final_stage <log> — when `hf jobs run` exits non-zero, the job may still be fine: the local
# client follows the job's log stream, and a dropped connection (e.g. a DNS blip) kills the client
# but not the job. So ask HF for the job's real stage, polling while it is still running.
# Prints the terminal stage, or nothing if no job id was printed or HF never answered.
_final_stage() {
    local jid stage misses=0
    jid=$(_job_id "$1")
    [ -n "${jid}" ] || return 0
    while :; do
        stage=$(_job_stage "${jid}")
        case "${stage}" in
            COMPLETED|ERROR|CANCELED|CANCELLED|DELETED) echo "${stage}"; return 0 ;;
            "") misses=$((misses+1)); [ "${misses}" -ge 10 ] && return 0 ;;   # ~5 min of no answer
            *) misses=0 ;;
        esac
        sleep 30
    done
}

echo; echo "Waiting on ${#PIDS[@]} probe job(s)..."
FAILED=0
for i in "${!PIDS[@]}"; do
    _log="${LOGS[$i]}"
    _ok=false; _stage=""
    if wait "${PIDS[$i]}"; then
        _ok=true
    else
        _stage=$(_final_stage "${_log}")
        if [ "${_stage}" = "COMPLETED" ]; then
            echo "  (${TAGS[$i]}: the local log stream failed, but the HF job completed — using its log)"
            _ok=true
        fi
    fi
    if ${_ok}; then
        # One summary per model, in run order (a shared Phonon job probes several).
        read -ra _models <<< "${MIDS[$i]}"
        mapfile -t _lines < <(grep -hoE "TTFA SUMMARY .*" "${_log}" 2>/dev/null | tail -n "${#_models[@]}")
        # A short job can finish before `hf jobs run`'s log stream attaches; re-fetch by job id.
        if [ ${#_lines[@]} -lt ${#_models[@]} ]; then
            _append_job_log "${_log}"
            mapfile -t _lines < <(grep -hoE "TTFA SUMMARY .*" "${_log}" 2>/dev/null | tail -n "${#_models[@]}")
        fi
        for j in "${!_models[@]}"; do
            echo "  ✓ ${_models[$j]} [${TAGS[$i]}]: ${_lines[$j]:-completed, but no TTFA line found — see ${_log}}"
        done
        _cpu=$(grep -m1 -oE "CPU model: .*" "${_log}" 2>/dev/null)
        [ -z "${_cpu}" ] || echo "      (${_cpu})"
    else
        echo "  ✗ ${MIDS[$i]} [${TAGS[$i]}]: FAILED (HF job stage: ${_stage:-unknown}) — last lines:"
        # Same race as above: pull the job's own log so a failure is diagnosable.
        _append_job_log "${_log}"
        tail -n 12 "${_log}" 2>/dev/null | sed 's/^/      /'
        FAILED=$((FAILED+1))
    fi
done

echo
echo "RTFx(batch1) above is NOT the leaderboard RTFx: TTFA requires batch size 1, while the eval"
echo "measures RTFx at each backend's real batch size. Do not mix them."
echo
echo "Sidecars land in the bucket as"
echo "  TTFA_MODEL_<model>_DATASET_<dataset>_<config>_<split>[_voice_clone][__<engine>]__${FLAVOR}.json"
[ "${FAILED}" -eq 0 ] || { echo "${FAILED} probe job(s) failed."; exit 1; }
