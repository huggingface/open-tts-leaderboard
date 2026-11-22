# Stage-1 environment for the tiny-voca-gr backend, sourced by tiny-voca-gr/submit_jobs.sh. Defines
# GEN_IMAGE and gen_setup.
#
# No Docker Space needed. tiny-voca-gr is a 4.53 M PyTorch model that runs on CPU (cc-by-4.0), so the
# job uses the public slim Python image and pip-installs the pinned packages at job start. The Greek
# adapter (tiny_voca_gr) is installed from the model repo and pulls the upstream tiny_voca package; the
# front end is eSpeak NG's Greek phonemiser, so the job apt-installs espeak-ng. The checkpoint
# (inference.pt) is downloaded by run_eval.py at the pinned revision.
#
# The adapter checks the eSpeak version stored in the checkpoint (training used eSpeak NG 1.51 with
# data at /usr/lib/x86_64-linux-gnu/espeak-ng-data); python:3.12-slim is Debian, so apt gives that
# layout. Torch is pinned <2.7 (the adapter's requirement) from PyTorch's /whl/cpu index.
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"

# The Greek adapter is installed straight from the model repo; pin it to the same revision run_eval.py
# downloads the checkpoint at. Its pyproject pulls the upstream tiny-voca package (pinned there).
TINY_VOCA_GR_REV="${TINY_VOCA_GR_REV:-927ea15cf4edcb8695b75328c9c90c5c53928554}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used.
# git is needed for the git+https installs; espeak-ng is the Greek phonemiser. Torch (<2.7) goes in
# first from the CPU index so the adapter's torch requirement resolves to a CPU wheel.
gen_setup() {
    cat <<INSTALL
apt-get update -qq && apt-get install -y --no-install-recommends git espeak-ng >/dev/null &&
pip install --no-cache-dir -q --index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu &&
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 \
    git+https://huggingface.co/ComputationalLinguisticsPlatform/tiny-voca-gr@${TINY_VOCA_GR_REV} &&
INSTALL
}
