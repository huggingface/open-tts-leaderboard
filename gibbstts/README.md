# GibbsTTS backend

Official model documentation: https://github.com/ydqmkkx/GibbsTTS

This backend evaluates the public base EN/ZH checkpoint with the upstream
`GibbsTTS.synthesize()` API. Voice cloning is required; the reference comes from
each benchmark sample. Inference is serial (`batch_size=1`, batch inference: no).
Japanese LoRA and the optional no-cloning TTFA benchmark are not included.

## Reproducibility

- Model: `ydqmkkx/GibbsTTS`, MIT, approximately 0.4B generative parameters.
- Code: `2af61715fe832b8dbef909b545090a56056428a3`.
- Weights: `a31945f20bf84498af41508ff3b126ea7f53b5c2`.
- Sampling: 32 steps, CFG 2.5, rescale CFG 0.75, temperature 0.6,
  top-p 1.0, speed 1.0, seed 42. No precision changes, retries, or ASR-based selection.
- Output: mono 24 kHz float WAV, with reference WAVs and the standard JSONL manifest.
- RTFx timing uses synchronized wall-clock time around the public synthesis call,
  including phonemization, prompt reading/encoding, generation, and codec decoding.
  Model loading, two warm-up utterances, preparing the reference files, and saving
  the generated files are excluded. Samples are re-seeded independently.
- `generation_config.json` accompanies each split. Resume skips complete rows,
  repairs missing waveform/reference files, and rejects changed sampling settings.

## HF Jobs

Create a public Docker Space for the environment. The matching build files are
prepared separately in `gibbstts-space/` alongside this checkout; backend Dockerfiles
normally live in their HF Spaces, as described in the repository README.
HF currently requires PRO to create a Docker Space on `cpu-basic`. HF Jobs itself
is a separate service. If you cannot build a Space, reuse the public leaderboard
image with `TTS_SPACE=bezzam/evals GIBBSTTS_BOOTSTRAP=true`; `bootstrap.sh` installs
the pinned dependencies and official source inside each generation job. Setup and
weight downloads are outside the inference timer, but incur job compute charges.

Prepare **private** `seed_tts_eval` and `cv3_eval` dataset copies with the existing
`scripts/prepare_seed_tts_eval.py` and `scripts/prepare_cv3_eval.py`.
The smoke launcher uses `ydqmkkx/seed_tts_eval_smoke` and `ydqmkkx/cv3_eval_smoke`,
private copies containing the first 8 English rows in official release order.
The launcher loads the locally configured token without printing it, checks Jobs
access and prerequisite resources, uses the existing `bezzam/evals` image with
runtime bootstrap, sets 20-minute timeouts for all six jobs, and disables automatic
retries. It completes the Seed-TTS chain before starting CV3 and stops on failure.
The token needs **Jobs → Start and manage Jobs**, in addition to repo/bucket access.

```bash
python gibbstts/run_smoke.py         # read-only readiness checks; no paid jobs
python gibbstts/run_smoke.py --run   # paid test; only after resources are ready and cost approved
```

Configure your own bucket and Docker Space explicitly:

```bash
export DATASET_NAMESPACE=ydqmkkx
export TTS_SPACE=ydqmkkx/evals-gibbstts
export RESULTS_BUCKET=ydqmkkx/gibbstts-smoke-h200
export HF_TOKEN="$(python -c 'from huggingface_hub import get_token; t=get_token(); assert t, "Run hf auth login first"; print(t)')"

# Paid smoke test: first 8 examples of Seed-TTS/en and CV3/en.
MAX_EVAL_SAMPLES=8 ONLY_LANGS=en GEN_TIMEOUT=20m bash gibbstts/submit_jobs.sh

# Paid full run: four splits. Use a fresh bucket and confirm a budget first.
RESULTS_BUCKET=ydqmkkx/gibbstts-full-h200 \
MAX_EVAL_SAMPLES=-1 ONLY_LANGS="en zh" bash gibbstts/submit_jobs.sh
```

HF Jobs charges for GPU compute. A timeout limits each generation job, not the
total run cost; shared ASR/SIM timeouts and image retries also contribute.
Bucket storage may have separate charges. Do not run these commands before
checking account credits and approving the planned compute cost.

To re-score existing results, set `STAGES="transcribe sim"` and use the same bucket.
If regenerating audio in an existing bucket, set `SIM_OVERWRITE=true ASR_OVERWRITE=true`.
All four splits run in parallel, and each split has generation → transcription → SIM.

## Use your existing GPU for development

With the same Python dependencies as the Docker environment installed, a checkout
of the official GibbsTTS source, and access to your private benchmark copy:

```bash
PYTHONPATH=scripts python gibbstts/run_eval.py \
  --source_dir=/path/to/GibbsTTS --checkpoint_path=/path/to/pretrained \
  --dataset_path=ydqmkkx/seed_tts_eval --dataset=tts --split=en \
  --device=cuda:0 --voice_clone --max_eval_samples=8
```

Missing weights are downloaded from the pinned public model revision. Use `--resume`
to continue an interrupted local generation. Local CPU/other GPU timing is for
development; official leaderboard generation timing must use H200.

## Offline validation

Install `pytest numpy soundfile datasets==3.6.0 librosa` and run:

```bash
python -m pytest gibbstts/test_backend.py -q
bash -n gibbstts/submit_jobs.sh scripts/tts_jobs_common.sh
```

Tests exercise real HF Audio decoding and WAV/manifest I/O with a fake synthesizer,
plus job-command capture and timeout/failure handling. They do not certify CUDA
inference or a Docker build. Complete the real smoke/full evaluation before
marking the upstream PR's smoke-test or full-results checklist items complete.
