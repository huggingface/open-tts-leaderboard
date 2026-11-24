# Stage-1 environment for the tiny-voca-es backend, sourced by tiny-voca-es/submit_jobs.sh. Defines
# GEN_IMAGE and gen_setup.
#
# No Docker Space needed. tiny-voca-es is a 4.5 M PyTorch model that runs on CPU (apache-2.0). Its front
# end is the model's own fork of the tiny_voca package (installed from GitHub), driven by eSpeak NG's
# `es` phonemiser, so the job apt-installs espeak-ng (+ git for the install) and pip-installs the pinned
# package at job start. The checkpoint (tiny-voca-es.pt) is downloaded by run_eval.py at the pinned
# revision.
#
# The checkpoint records the eSpeak front-end fingerprint (training used eSpeak NG **1.52.0**). Unlike
# the other tiny-voca ports (trained with 1.51 → bookworm base), the Spanish port needs 1.52.0: its
# Spanish phonemes differ from 1.51 (e.g. "informe" is i n f o ɾ m e in 1.52 vs i m f o ɾ m e in 1.51).
# Debian 13 (trixie) ships eSpeak NG 1.52.0, so the rolling python:3.12-slim tag is exactly right here
# and the fingerprint passes natively (the check stays ON — no bypass).
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"

# The front-end fork is installed from the model's GitHub repo; pin it to the commit the checkpoint was
# exported from (same project as the model repo's pinned revision).
TINY_VOCA_ES_COMMIT="${TINY_VOCA_ES_COMMIT:-a4bf2bd979fc40b1bd7e34e13c0164db627d14ec}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used. Torch
# (CPU wheel) goes in first from the CPU index; then the leaderboard stack and the GitHub front-end fork.
gen_setup() {
    cat <<INSTALL
apt-get update -qq && apt-get install -y --no-install-recommends git espeak-ng >/dev/null &&
pip install --no-cache-dir -q --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu &&
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 \
    git+https://github.com/stdevMac/tiny-voca.git@${TINY_VOCA_ES_COMMIT} &&
INSTALL
}
