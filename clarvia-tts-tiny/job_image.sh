# Stage-1 environment for the clarvia-tts-tiny backend, sourced by clarvia-tts-tiny/submit_jobs.sh.
# Defines GEN_IMAGE and gen_setup.
#
# No Docker Space needed. clarvia-tts-tiny is a 4.5 M model whose inference is an ONNX graph that
# runs on CPU (apache-2.0), so the job uses the public slim Python image and pip-installs the pinned
# packages at job start. The weights (onnx/model.onnx) come from the model repo at the pinned
# revision set in run_eval.py.
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"

# The Turkish front end ships as an HF-only Python package (clarvia-phonemizer, pure Python, no
# dependencies). Pinned to its 2026-11-22 main HEAD.
CLARVIA_PHONEMIZER_REV="${CLARVIA_PHONEMIZER_REV:-bf1b1aab40e24ca3f6700980427c386f4f171f6f}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used.
# python:3.12-slim ships without git; the job apt-installs it so pip can clone the phonemizer repo.
# onnxruntime provides the CPU inference; the rest are the shared stage-1 deps (datasets/soundfile).
gen_setup() {
    cat <<INSTALL
apt-get update -qq && apt-get install -y --no-install-recommends git >/dev/null &&
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 \
    onnxruntime==1.20.1 \
    git+https://huggingface.co/wite-tech/clarvia-phonemizer@${CLARVIA_PHONEMIZER_REV} &&
INSTALL
}
