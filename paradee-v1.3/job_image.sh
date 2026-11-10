# Stage-1 environment for the Paradee 1.3 backend, sourced by paradee-v1.3/submit_jobs.sh (eval) and
# submit_ttfa_jobs.sh (TTFA probe), so both measure the same install. Defines GEN_IMAGE and gen_setup.
#
# No Docker Space needed. The job runs on the public PyTorch image below (Python 3.11, CUDA 12.8,
# cuDNN 9; torch itself is unused) and pip-installs the paradee package pinned to the v1.3 release
# commit, its spaCy English model, and the eval-loop deps at job start. On a GPU, the CPU onnxruntime
# that paradee pulls in is then swapped for onnxruntime-gpu 1.23 (CUDA 12.x builds; 1.24+ needs CUDA 13),
# whose preload_dlls() (called in run_eval.py) loads cuDNN from the nvidia-cudnn-cu12 wheel. That wheel
# must be UPGRADED (-U): the image's own cuDNN 9.7 fails in onnxruntime-gpu 1.23's cuDNN frontend on some convs.
GEN_IMAGE="${GEN_IMAGE:-pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime}"
PARADEE_COMMIT="af5ca1a17daad860d1aea60510d93d8dbd1d7b6f"   # tag v1.3 of github.com/sahilmahendrakar/paradee
GEN_PIP_DEPS="${GEN_PIP_DEPS:-'paradee @ https://github.com/sahilmahendrakar/paradee/archive/${PARADEE_COMMIT}.zip' \
datasets soundfile tqdm onnx \
'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'}"
GEN_GPU_DEPS="${GEN_GPU_DEPS:-onnxruntime-gpu==1.23.0 nvidia-cudnn-cu12}"

# gen_setup <device> → `... &&` install snippet for the job body. A cpu device keeps the package's own
# CPU onnxruntime (no GPU swap), so a CPU run benchmarks the model as shipped.
gen_setup() {
    local setup="pip install --no-cache-dir -q ${GEN_PIP_DEPS} &&"
    [[ "$1" == cpu ]] || setup="${setup} pip uninstall -y -q onnxruntime && pip install --no-cache-dir -q -U ${GEN_GPU_DEPS} &&"
    echo "${setup}"
}
