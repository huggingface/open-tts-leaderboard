# Stage-1 environment for the tiny-broca backend, sourced by tiny-broca/submit_jobs.sh. Defines
# GEN_IMAGE and gen_setup.
#
# No Docker Space needed. tiny-broca is a 4.5 M model whose inference is an ONNX graph that runs on
# CPU (apache-2.0), so the job uses the public slim Python image and pip-installs the pinned packages
# at job start. The weights (onnx/model.onnx) and the character-level front end (text.py) come from
# the model repo at the pinned revision set in run_eval.py — the front end is self-contained (no
# dictionary, no external phonemizer), so nothing else is cloned.
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used.
# onnxruntime provides the CPU inference; the rest are the shared stage-1 deps (datasets/soundfile).
gen_setup() {
    cat <<INSTALL
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 \
    onnxruntime==1.20.1 &&
INSTALL
}
