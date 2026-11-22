# Stage-1 environment for the apptor-tiny-tts backend, sourced by apptor-tiny-tts/submit_jobs.sh.
# Defines GEN_IMAGE and gen_setup.
#
# No Docker Space needed. apptor-tiny-tts is a ~4.5 M model whose inference is a three-stage ONNX
# pipeline (g2p + tts + vocos) that runs on CPU (apache-2.0; the vocos stage is MIT), so the job uses
# the public slim Python image and pip-installs the pinned packages at job start. The ONNX files, the
# in-repo ``tinyvoca`` front end and the G2P model come from the model repo at the pinned revision set
# in run_eval.py — the front end is self-contained (its own ONNX G2P, no eSpeak), so nothing else is
# cloned.
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used.
# onnxruntime provides the CPU inference; num2words spells German digits; the rest are the shared
# stage-1 deps (datasets/soundfile).
gen_setup() {
    cat <<INSTALL
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 \
    onnxruntime==1.20.1 num2words==0.5.14 &&
INSTALL
}
