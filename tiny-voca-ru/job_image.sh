# Stage-1 environment for the tiny-voca-ru backend, sourced by tiny-voca-ru/submit_jobs.sh. Defines
# GEN_IMAGE and gen_setup.
#
# No Docker Space needed. tiny-voca-ru is a 4.5 M model whose inference is an ONNX graph that runs on
# CPU (apache-2.0), so the job uses the public slim Python image and pip-installs the pinned packages at
# job start. Both the weights and the whole front end (pure-Python Russian normaliser + rule-based G2P +
# stress) ship in the model's own pip package ``tiny-voca-ru``; the weights (onnx/model.onnx,
# accent.onnx, the stress dictionary) are downloaded from the model repo at the pinned revision set in
# run_eval.py. No espeak / no git / no torch — the ONNX extra pulls onnxruntime only.
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used.
# The leaderboard stack (datasets/soundfile/huggingface_hub/onnxruntime) is pinned first; tiny-voca-ru
# 0.2.0 resolves cleanly against it (verified), and its [onnx] extra adds nothing beyond onnxruntime.
gen_setup() {
    cat <<INSTALL
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 \
    onnxruntime==1.20.1 "tiny-voca-ru[onnx]==0.2.0" &&
INSTALL
}
