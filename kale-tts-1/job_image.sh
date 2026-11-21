# Stage-1 environment for the kale-tts-1 backend, sourced by kale-tts-1/submit_jobs.sh (eval) and
# submit_ttfa_jobs.sh (TTFA probe), so both measure the same install. Defines GEN_IMAGE and gen_setup.
#
# No Docker Space needed. kale-tts-1-nano (7.5M params) and kale-tts-1-tiny (45.7M params) are single
# ONNX graphs run with ONNX Runtime on CPU (apache-2.0). The Python package (python/kale_tts.py) and its
# text front end (python/kale_frontend.js, run in QuickJS; frontend/words/) ship inside each model repo,
# so the job only pip-installs onnxruntime, numpy and quickjs. No espeak-ng, no node.
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used.
gen_setup() {
    cat <<INSTALL
pip install --no-cache-dir -q \
    onnxruntime==1.32.0 numpy==2.4.2 quickjs==1.19.4 soundfile==0.13.1 \
    datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 &&
INSTALL
}
