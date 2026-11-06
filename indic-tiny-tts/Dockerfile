# Stage-1 (generation) image for the Open TTS Leaderboard indic-tiny-tts backend (indic-tiny-tts/run_eval.py is injected
# at runtime). Only needed if you want a Docker Space (set TTS_SPACE=<user>/<space> for submit_jobs.sh): by default the
# backend runs on this same base image and pip-installs the line below at job start.
# The model package (indic_tiny_tts) is NOT installed here: run_eval.py downloads the pinned HF model repo
# (sulabhkatiyar/indic-tiny-tts-hindi) at job time and imports the package it ships.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir datasets soundfile tqdm huggingface_hub safetensors numpy
WORKDIR /app
ENTRYPOINT ["bash"]
# Keep-alive CMD so the Space runtime stays healthy; HF Jobs override it.
EXPOSE 7860
CMD ["-c", "python -m http.server 7860"]
