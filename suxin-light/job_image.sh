# Stage-1 environment for the suxin-light backend, sourced by suxin-light/submit_jobs.sh (eval) and
# submit_ttfa_jobs.sh (TTFA probe), so both measure the same install. Defines GEN_IMAGE and
# gen_setup.
#
# No Docker Space. The job uses the public PyTorch CUDA 12.8 / cuDNN 9 image (Python 3.11; torch
# itself is unused) so onnxruntime-gpu 1.23.x can load the CUDA EP. 1.23 is the last ORT line built
# for CUDA 12 + cuDNN 9; 1.24+ needs CUDA 13 (see paradee-v1.3/job_image.sh). Weights and the
# in-repo ``suxin_tts`` package come from the model repo at the pinned revision in run_eval.py.
#
# English G2P: piper-tts → espeak-ng (GPL-3.0). MixedFrontend prefers piper-tts's bundled
# espeak-ng-data; apt-installing espeak-ng covers the system fallback. Chinese G2P is pypinyin
# (pure Python).
GEN_IMAGE="${GEN_IMAGE:-pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime}"

GEN_PIP_DEPS="${GEN_PIP_DEPS:-soundfile datasets==3.6.0 tqdm huggingface_hub pypinyin==0.55.0 piper-tts==1.8.0}"
# -U nvidia-cudnn-cu12: the image's own cuDNN can fail in ORT 1.23's cuDNN frontend on some convs
# (same pin as paradee-v1.3). run_eval.py calls ort.preload_dlls() so the wheel is on the loader path.
GEN_GPU_DEPS="${GEN_GPU_DEPS:-onnxruntime-gpu==1.23.2 nvidia-cudnn-cu12}"

# gen_setup <device> → `... &&` install snippet for the job body.
# A cpu device keeps piper-tts's CPU onnxruntime (no GPU swap). Any other device (cuda / cuda:0)
# uninstalls that CPU wheel and installs onnxruntime-gpu.
gen_setup() {
    local setup="apt-get update -qq && apt-get install -y --no-install-recommends espeak-ng libsndfile1 >/dev/null && pip install --no-cache-dir -q ${GEN_PIP_DEPS} &&"
    [[ "$1" == cpu ]] || setup="${setup} pip uninstall -y -q onnxruntime && pip install --no-cache-dir -q -U ${GEN_GPU_DEPS} &&"
    echo "${setup}"
}
