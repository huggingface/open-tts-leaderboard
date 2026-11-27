# Stage-1 environment for the suxin-light backend, sourced by suxin-light/submit_jobs.sh (eval) and
# submit_ttfa_jobs.sh (TTFA probe), so both measure the same install. Defines GEN_IMAGE and
# gen_setup.
#
# No Docker Space needed. suxin-light is a 28.65 M INT8 ONNX VITS (MIT weights + inference code;
# English G2P goes through piper-tts → espeak-ng, GPL-3.0) that runs on CPU, so the job uses the
# public slim Python image and pip-installs the pinned packages at job start. The INT8 graph and
# the in-repo ``suxin_tts`` package come from the model repo at the pinned revision set in
# run_eval.py.
#
# python:3.12-slim has neither espeak-ng nor libsndfile. MixedFrontend prefers piper-tts's bundled
# espeak-ng-data, then ESPEAK_DATA_PATH / /usr/share/espeak-ng-data; apt-installing espeak-ng covers
# the system fallback and the libespeak-ng1 .so piper may still dlopen. Chinese G2P is pypinyin
# (pure Python, no extra system packages).
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used.
# onnxruntime is pinned to the version the model card measured (1.23.2 on the Cortex-A55 board);
# piper-tts 1.8.0 is the lockfile pin and the first release with EspeakPhonemizer + bundled data.
# The rest match the other CPU ONNX backends (datasets/soundfile/huggingface_hub).
gen_setup() {
    cat <<INSTALL
apt-get update -qq && apt-get install -y --no-install-recommends espeak-ng libsndfile1 >/dev/null &&
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 \
    onnxruntime==1.23.2 pypinyin==0.55.0 piper-tts==1.8.0 &&
INSTALL
}
