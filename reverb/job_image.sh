# Stage-1 environment for the Reverb-TTS backend, sourced by reverb/submit_jobs.sh (eval) and submit_ttfa_jobs.sh
# (TTFA probe), so both measure the same install. Defines GEN_IMAGE and gen_setup.
#
# No Docker Space: the job runs on the public PyTorch image and pip-installs reverb-tts from PyPI at job start.
GEN_IMAGE="${GEN_IMAGE:-pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime}"
GEN_PIP_DEPS="${GEN_PIP_DEPS:-reverb-tts==1.0.0 'datasets<4.0' librosa soundfile tqdm}"  # librosa: datasets<4 decodes the cloning prompts with it

# gen_setup <device> → `... &&` install snippet for the job body; the same install on a GPU and a CPU.
gen_setup() {
    echo "pip install --no-cache-dir -q ${GEN_PIP_DEPS} &&"
}
