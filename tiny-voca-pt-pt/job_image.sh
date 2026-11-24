# Stage-1 environment for the tiny-voca-pt-pt backend, sourced by tiny-voca-pt-pt/submit_jobs.sh.
# Defines GEN_IMAGE and gen_setup.
#
# No Docker Space needed. tiny-voca-pt-pt is a 4.5 M PyTorch model that runs on CPU (apache-2.0). Its
# front end is the model's own fork of the tiny_voca package plus its companion `pronuncia-pt` package,
# both installed from the same GitHub repo. Unlike the other tiny-voca ports, pronuncia-pt is a
# **pure-Python** European Portuguese grapheme-to-phoneme (rules + exceptions) — there is no eSpeak
# binary in the loop, so the job needs no espeak-ng and no eSpeak-version pin; the base image is
# irrelevant to the phonemes. The checkpoint (tiny-voca-pt-pt.pt) is downloaded by run_eval.py at the
# pinned revision.
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"

# Both packages live in one GitHub repo; pin it to the commit the checkpoint was exported from. The
# main package (tiny_voca, src layout) installs from the repo root; pronuncia-pt is a subdirectory
# package, installed with pip's `#subdirectory=` at the same commit so the phonemizer version matches
# the checkpoint's recorded fingerprint.
TINY_VOCA_PTPT_COMMIT="${TINY_VOCA_PTPT_COMMIT:-744e42406fd481fbe61f3b67bf8012a7d29cb57a}"
TINY_VOCA_PTPT_REPO="${TINY_VOCA_PTPT_REPO:-https://github.com/joseruimemory/tiny-voca-pt-pt.git}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used. Torch
# (CPU wheel) goes in first from the CPU index; then the leaderboard stack, the GitHub fork and its
# pure-Python pronuncia-pt front end.
gen_setup() {
    cat <<INSTALL
apt-get update -qq && apt-get install -y --no-install-recommends git >/dev/null &&
pip install --no-cache-dir -q --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu &&
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 &&
pip install --no-cache-dir -q \
    "git+${TINY_VOCA_PTPT_REPO}@${TINY_VOCA_PTPT_COMMIT}" \
    "git+${TINY_VOCA_PTPT_REPO}@${TINY_VOCA_PTPT_COMMIT}#subdirectory=pronuncia-pt" &&
INSTALL
}
