# Stage-1 environment for the PAWP-TTS backend, sourced by pawp_tts/submit_jobs.sh (eval) and
# submit_ttfa_jobs.sh (TTFA probe), so both measure the exact same install.
#
# No custom Docker Space needed: PAWP-TTS runs folded ONNX graphs on CPU via onnxruntime,
# installing only espeak-ng (for English G2P) and lightweight Python deps (no PyTorch wheel needed).
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"
PAWP_REPO="${PAWP_REPO:-https://github.com/Firojpaudel/PAWP_TTS.git}"
GEN_PIP_DEPS="${GEN_PIP_DEPS:-onnxruntime==1.21.1 numpy==2.2.4 pyyaml==6.0.2 datasets==3.6.0 \
soundfile==0.13.1 tqdm==4.67.1 huggingface_hub==0.30.2}"

# gen_setup <device> -> `... &&` install snippet for the job body. CPU only.
gen_setup() {
    echo "apt-get update -qq && apt-get install -y -qq --no-install-recommends espeak-ng git libsndfile1 >/dev/null && pip install --no-cache-dir -q ${GEN_PIP_DEPS} && pip install --no-cache-dir -q --no-deps git+${PAWP_REPO} &&"
}
