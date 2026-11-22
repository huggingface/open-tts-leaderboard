# Stage-1 environment for the tiny-voca-arabic backend, sourced by tiny-voca-arabic/submit_jobs.sh.
# Defines GEN_IMAGE and gen_setup.
#
# No Docker Space needed. tiny-voca-arabic is a 4.5 M model whose inference is an ONNX graph that runs
# on CPU (apache-2.0), so the job uses the public slim Python image and pip-installs the pinned packages
# at job start. The weights (onnx/model.onnx) and the rule-based front end (code/) come from the model
# repo, and the CATT diacritiser package (catt_tashkeel) from the model's demo Space, both at the pinned
# revisions set in run_eval.py. The diacritiser is ONNX too (no torch): it downloads its encoder/decoder
# ONNX files from the abjadai/catt v2 GitHub release at first use, so the job needs network egress to
# github.com (HF Jobs have it). No espeak / no git needed — the whole front end is pure Python + ONNX.
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used.
# onnxruntime provides both the TTS and the diacritiser inference; the rest are the shared stage-1 deps.
gen_setup() {
    cat <<INSTALL
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 \
    onnxruntime==1.20.1 &&
INSTALL
}
