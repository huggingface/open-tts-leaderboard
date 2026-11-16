# Stage-1 environment for the sanoTTS backend, sourced by sanotts/submit_jobs.sh (eval) and
# submit_ttfa_jobs.sh (TTFA probe), so both measure the same install. Defines GEN_IMAGE and gen_setup.
#
# No Docker Space needed. sanotts is NumPy-only (no torch, no ONNX Runtime), so the job runs on the
# public slim Python image and pip-installs the pinned package plus the eval-loop deps at job start
# (seconds). The voice weights are downloaded by run_eval.py from the model repo at the revision
# pinned there.
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"
SANOTTS_VERSION="0.5.4"   # https://pypi.org/project/sanotts/0.5.4/ (synthesize_stream since 0.5.1; faster first chunk in 0.5.2-0.5.4)
# Every package pinned (the versions the submitted local run used), so eval and TTFA jobs on different days
# install the same numerics.
GEN_PIP_DEPS="${GEN_PIP_DEPS:-sanotts==${SANOTTS_VERSION} numpy==2.5.4 threadpoolctl==3.8.0 datasets==3.6.0 \
soundfile==0.14.0 tqdm==4.70.1 huggingface_hub==2.4.0}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used.
gen_setup() {
    echo "pip install --no-cache-dir -q ${GEN_PIP_DEPS} &&"
}
