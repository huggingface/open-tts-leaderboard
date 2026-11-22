# Stage-1 environment for the tiny-voca-fr backend, sourced by tiny-voca-fr/submit_jobs.sh. Defines
# GEN_IMAGE and gen_setup.
#
# No Docker Space needed. tiny-voca-fr is a 4.5 M PyTorch model that runs on CPU (apache-2.0). Its front
# end is the model's own fork of the tiny_voca package (installed from GitHub), driven by eSpeak NG's
# fr-fr phonemiser, so the job apt-installs espeak-ng (+ git for the install) and pip-installs the pinned
# package at job start. The checkpoint (tiny-voca-fr.pt) is downloaded by run_eval.py at the pinned
# revision.
#
# The checkpoint records the eSpeak front-end fingerprint (training used eSpeak NG 1.51 at
# /usr/lib/x86_64-linux-gnu/espeak-ng-data). The rolling python:3.12-slim tag now tracks Debian 13
# (trixie) = eSpeak 1.52, which fails that fingerprint, so we pin Debian 12 (bookworm) = eSpeak 1.51 at
# the exact path, and run_eval.py keeps the check ON. The French eSpeak fr-fr G2P is byte-identical
# between 1.51 and 1.52 (verified on all eval sentences): this is a fingerprint pin, not an audio change.
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim-bookworm}"

# The front-end fork is installed from the model's GitHub repo; pin it to the commit the checkpoint was
# exported from (same project as the model repo's pinned revision).
TINY_VOCA_FR_COMMIT="${TINY_VOCA_FR_COMMIT:-4cc26ab23d934e3d33796eb06bf11f23f4c9ab34}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used. Torch
# (CPU wheel) goes in first from the CPU index; then the leaderboard stack and the GitHub front-end fork.
gen_setup() {
    cat <<INSTALL
apt-get update -qq && apt-get install -y --no-install-recommends git espeak-ng >/dev/null &&
pip install --no-cache-dir -q --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu &&
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 \
    git+https://github.com/DanielSuncost/tiny-voca-fr.git@${TINY_VOCA_FR_COMMIT} &&
INSTALL
}
