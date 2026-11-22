# Stage-1 environment for the tiny-voca-ja backend, sourced by tiny-voca-ja/submit_jobs.sh. Defines
# GEN_IMAGE and gen_setup.
#
# No Docker Space needed. tiny-voca-ja is a 4.5 M PyTorch model that runs on CPU (apache-2.0). Its stock
# front end is the model's own fork of the tiny_voca package (installed from GitHub): pykakasi reads
# kanji to kana and eSpeak NG's ja voice phonemises it, so the job apt-installs espeak-ng (+ git) and
# pip-installs the package plus pykakasi at job start. The checkpoint (tiny-voca-ja.pt) is downloaded by
# run_eval.py at the pinned revision. (No pyopenjtalk / no separate vocoder — those are the repo's
# experiments; the shipped checkpoint uses the stock pykakasi+eSpeak front end and the built-in
# source-filter synthesiser.)
#
# Base pinned to Debian 12 (bookworm) = eSpeak NG 1.51, for reproducibility; the rolling python:3.12-slim
# tag now tracks Debian 13 (trixie) = eSpeak 1.52. The eSpeak ja G2P is byte-identical between the two
# (verified on all eval sentences), so this is a reproducibility pin, not an audio change. run_eval.py
# loads with check_frontend=False (see its docstring: the recorded fingerprint differs but the tokens do
# not).
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim-bookworm}"

# The front-end fork is installed from the model's GitHub repo; pin it to the current main commit.
TINY_VOCA_JA_COMMIT="${TINY_VOCA_JA_COMMIT:-49632c552d54a13664c1beb96904bef0154fbfc3}"
# pykakasi (kanji -> kana reader) is pinned to the version this backend was validated with.
TINY_VOCA_JA_PYKAKASI="${TINY_VOCA_JA_PYKAKASI:-2.3.0}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used. Torch
# (CPU wheel) goes in first from the CPU index; then the leaderboard stack, pykakasi and the GitHub fork.
gen_setup() {
    cat <<INSTALL
apt-get update -qq && apt-get install -y --no-install-recommends git espeak-ng >/dev/null &&
pip install --no-cache-dir -q --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu &&
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 \
    pykakasi==${TINY_VOCA_JA_PYKAKASI} \
    git+https://github.com/gachimoto-code/tiny-voca-ja.git@${TINY_VOCA_JA_COMMIT} &&
INSTALL
}
