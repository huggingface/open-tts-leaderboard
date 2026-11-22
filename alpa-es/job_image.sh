# Stage-1 environment for the alpa-es backend, sourced by alpa-es/submit_jobs.sh. Defines GEN_IMAGE
# and gen_setup.
#
# No Docker Space needed. alpa-es is a ~4.5 M PyTorch model that runs on CPU (apache-2.0), so the job
# uses the public slim Python image and pip-installs the pinned packages at job start. The weights and
# the repo's own code (the ``alpa`` package, which includes the rule-based Spanish front end) come from
# the model repo at the pinned revision set in run_eval.py — the front end is self-contained (no
# eSpeak, no dictionary), so nothing else is cloned. Torch is the CPU wheel from PyTorch's /whl/cpu
# index.
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used.
gen_setup() {
    cat <<INSTALL
pip install --no-cache-dir -q --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu &&
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 &&
INSTALL
}
