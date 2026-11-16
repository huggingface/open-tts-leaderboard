# Stage-1 environment for the tiny-voca backend, sourced by tiny-voca/submit_jobs.sh (eval) and
# submit_ttfa_jobs.sh (TTFA probe), so both measure the same install. Defines GEN_IMAGE and
# gen_setup.
#
# No Docker Space needed. tiny-voca is a 4.5 M PyTorch model that runs on CPU (apache-2.0), so the
# job uses the public slim Python image and pip-installs the pinned packages at job start (~15 s).
# The weights come from the model repo at the pinned revision set in run_eval.py.
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"

# tiny-voca ships as a GitHub-only Python package; same for its phonemizer (dizione). Both pinned
# to their 2026-11-16 main HEADs. Torch is the CPU wheel from PyTorch's /whl/cpu index.
TINY_VOCA_REV="${TINY_VOCA_REV:-main}"
DIZIONE_REV="${DIZIONE_REV:-main}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used.
# python:3.12-slim ships without git, so apt-install it first — dizione and tiny-voca are
# GitHub-only (no PyPI release yet). Then torch from the CPU wheel index (so +cpu is resolvable),
# and the rest from PyPI (so --index-url doesn't apply to packages that only live there).
gen_setup() {
    cat <<INSTALL
apt-get update -qq && apt-get install -y --no-install-recommends git >/dev/null &&
pip install --no-cache-dir -q --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu &&
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 \
    git+https://github.com/n4ze3m/dizione@${DIZIONE_REV} \
    git+https://github.com/n4ze3m/tiny-voca@${TINY_VOCA_REV} &&
INSTALL
}
