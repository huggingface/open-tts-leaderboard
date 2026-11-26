# Stage-1 environment for the tiny-voca-ru backend, sourced by tiny-voca-ru/submit_jobs.sh. Defines
# GEN_IMAGE and gen_setup.
#
# No Docker Space needed. tiny-voca-ru is a 4.5 M model that runs on CPU (apache-2.0), so the job
# uses the public slim Python image and pip-installs the pinned packages at job start. The whole
# front end (pure-Python Russian normaliser + rule-based G2P + bundled stress.txt.gz & accent.onnx)
# ships in the model's own pip package ``tiny-voca-ru``; the weights (model.safetensors, config.json)
# are downloaded from the model repo at the pinned revision set in run_eval.py. Inference uses
# PyTorch CPU (native O(N log N) FFT/STFT in the source-filter vocoder, ~1.5x faster than the dense
# conv1d-DFT ONNX export); onnxruntime is still needed for the bundled accent.onnx stress predictor.
GEN_IMAGE="${GEN_IMAGE:-python:3.12-slim}"

# gen_setup <device> → `... &&` install snippet for the job body. CPU only: the device is not used.
# Torch (CPU wheel) goes in first from the CPU index; then the leaderboard stack + safetensors +
# onnxruntime (for accent.onnx) + tiny-voca-ru 0.2.3.
gen_setup() {
    cat <<INSTALL
pip install --no-cache-dir -q --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu &&
pip install --no-cache-dir -q \
    numpy==2.4.2 soundfile==0.13.1 datasets==3.6.0 tqdm==4.70.1 huggingface_hub==0.36.2 \
    safetensors==0.8.0 onnxruntime==1.20.1 "tiny-voca-ru[onnx]==0.2.3" &&
INSTALL
}
