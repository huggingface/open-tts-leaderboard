# API TTS evaluation

This backend evaluates hosted TTS services using the same Seed-TTS/CV3 text, Qwen3-ASR transcription, language-specific normalization and optional WavLM speaker similarity as the other backends. Generation runs locally in CPU Docker containers; only ASR and similarity scoring use Hugging Face Jobs. Provider adapters follow the [Open ASR Leaderboard API organization](https://github.com/huggingface/open_asr_leaderboard/tree/main/api).

API generation measures a client request, including networking, provider queuing and retries. It writes `time: null` so the shared scorer cannot turn those timings into H200 RTFx. Quality scores and API timing summaries are exported to `api/results/API_RESULTS.csv` and `API_RESULTS.json`; no unmeasured leaderboard rows are created.

## Setup

Use a running Docker daemon and Python 3.10 or newer on the host. The host Python dependencies and `hf` CLI are still needed for orchestration, bucket sync, Jobs and score export. From the repository root:

```bash
python -m venv .venv-api
source .venv-api/bin/activate
pip install -r api/requirements.txt
python api/run_eval.py --list_models
```

Direct `python api/run_eval.py` is a low-level development entry point using host dependencies. Use `api/submit_jobs.sh` below for Docker generation and the complete pipeline.

Set the key for the provider being evaluated. The wrapper passes `HF_TOKEN` and the selected provider key to the generation container by environment-variable name, keeping their values out of image builds and printed commands. Scorer Jobs receive only `HF_TOKEN`.

| Provider | Environment variable | Default voice |
| --- | --- | --- |
| ElevenLabs | `ELEVENLABS_API_KEY` | `JBFqnCBsd6RMkjVDRZzb` (George) |
| Cartesia | `CARTESIA_API_KEY` | `db6b0ed5-d5d3-463d-ae85-518a07d3c2b4` |
| MiniMax | `MINIMAX_API_KEY` | Native voices by language in `models.py`; English `English_expressive_narrator` |
| Inworld | `INWORLD_API_KEY` | `Ashley`; use the API credential's Base64 value for Basic authentication |
| Gemini | `GEMINI_API_KEY` | `Kore` |
| Fish Audio | `FISH_API_KEY` | First sample's reference speaker, fixed for the split |
| Mistral | `MISTRAL_API_KEY` | First sample's reference speaker, fixed for the split |
| Smallest AI | `SMALLEST_API_KEY` | Language-specific voices in `models.py`; English `blake` for Pro, `olivia` otherwise |
| Deepgram | `DEEPGRAM_API_KEY` | Voice is encoded in `aura-2-thalia-en` |

The model and voice registry is explicit in [models.py](models.py). `--voice` selects an existing provider voice for a single model. Smallest AI uses a voice trained for each language; overriding it requires choosing a voice compatible with the selected language. Deepgram voice changes require adding another voice-specific model ID to the registry.

ElevenLabs v4 uses a single-turn Text-to-Dialogue HTTP stream; v4 Turbo uses the
native Text-to-Dialogue WebSocket. Both receive the benchmark transcript verbatim.
The HTTP stream rejects inputs over its documented 2,000-character reliable streaming
limit instead of accepting potentially truncated audio. Inworld rejects inputs over
4,000 UTF-16 code units and Smallest AI over 8,000 characters. Any such sample is
recorded as a failure and prevents exporting an incomplete split.

Set `HF_TOKEN` with access to the private evaluation datasets. Prepare your private copies using [`scripts/prepare_seed_tts_eval.py`](../scripts/prepare_seed_tts_eval.py) and [`scripts/prepare_cv3_eval.py`](../scripts/prepare_cv3_eval.py), then set `DATASET_NAMESPACE` to their namespace. The pipeline also accepts `DATASET_PATH` and `CV3_EVAL_PATH` for explicit repository IDs.

For ASR/SIM, use a dedicated API bucket and an account with HF Jobs access. `RESULTS_BUCKET` is required for these stages; the official H200 bucket is rejected. `ORG_NAME` optionally selects the Jobs organization.

## Run

Commands below run from the repository root. When `generate` is selected, the wrapper tries to pull `registry.hf.space/hf-audio-open-tts-leaderboard-apis:latest` once, then runs a local container for each model/split. It overrides the image entrypoint with Python and uses `/app/api` as the working directory. Containers use the host UID/GID, mount this checkout's `api/` and shared scripts read-only at `/app/api` and `/app/scripts`, and overlay writable `api/results/` at `/app/api/results`. `MAX_WORKERS=1` is the default and should be retained for comparable request latency.

The shared environment lives in the public [API environment Space](https://huggingface.co/spaces/hf-audio/open-tts-leaderboard-apis). Update its Dockerfile and requirements to change dependencies; the Hub builds the registry image. If the registry pull fails, the wrapper resolves the public Docker Space's exact commit, downloads that revision with `hf download --revision`, and builds the environment locally under the same image tag. The downloaded source is cached at `<HF cache>/tts_api_spaces/<owner>--<space>/<commit SHA>`; the Dockerfile is maintained solely in the Hub Space.

Set `TTS_SPACE` or `--tts_space` to select another environment Space. `API_IMAGE` or `--api_image` overrides the registry reference, including an exact tag or digest. Explicit image references disable the fallback and fail if pulling fails. `--skip_image_pull` reuses a cached image, whether pulled or built from the Space. `API_DOCKER_PLATFORM` or `--docker_platform` defaults to `linux/amd64`, matching the Hub build; Apple Silicon runs this image through Docker's emulation.

The host cache is selected by `HF_CACHE_DIR`, then `HF_HOME`, then `~/.cache/huggingface`, and mounted at `/hf_cache`. Dataset and Hub caches use `/hf_cache/datasets_tts_api` and `/hf_cache/hub`, respectively.

Eight-sample generation smoke test for one model and the English Seed-TTS split:

```bash
MODEL=elevenlabs/eleven_v4 ONLY_LANGS=en MAX_EVAL_SAMPLES=8 \
  STAGES=generate bash api/submit_jobs.sh --datasets seed_tts
```

Add your dedicated bucket to run the complete smoke test, including ASR and score export. `--resume` reuses the generated WAVs and their measured timings:

```bash
MODEL=elevenlabs/eleven_v4 ONLY_LANGS=en MAX_EVAL_SAMPLES=8 \
  RESULTS_BUCKET=your-org/tts-api-results \
  bash api/submit_jobs.sh --datasets seed_tts --resume
```

Full standard-voice evaluation of selected models across supported Seed-TTS and CV3 languages:

```bash
DATASET_NAMESPACE=your-hf-namespace RESULTS_BUCKET=your-org/tts-api-results \
  MAX_EVAL_SAMPLES=-1 MAX_WORKERS=1 \
  bash api/submit_jobs.sh --models elevenlabs/eleven_v4 cartesia/sonic-3.6-2026-08-27
```

To replace smoke results with full results at the same paths, pass `--overwrite` rather than `--resume`: changing the sample set or settings invalidates resume. `--overwrite` regenerates WAVs and invalidates old predictions/similarity. Without either flag, an existing manifest causes generation to stop rather than silently overwrite measurements.

The default model list contains all registered models. Explicit `--models` avoids requiring every provider's credential. `ONLY_LANGS="en fr"` restricts the benchmark languages. Models skip unsupported languages: Mistral covers English/French/Spanish/Italian/German; Smallest AI Lightning v3.1 covers English/Spanish; this Deepgram voice covers English. The other registered models cover the nine benchmark languages.

Per-sample cloning and speaker similarity are implemented for Fish Audio and Mistral:

```bash
MODEL=fish/s2.1-pro VOICE_CLONE=true MAX_WORKERS=1 \
  RESULTS_BUCKET=your-org/tts-api-results \
  bash api/submit_jobs.sh --only_langs en
```

Cloning uses each sample's `prompt_audio` and `prompt_text`; generation saves reference WAVs beside generated WAVs so the shared similarity scorer can resolve them. In standard mode, Fish Audio and Mistral use the first reference consistently across the split unless a saved voice is selected. Other adapters use existing voices and reject per-sample cloning; they do not enroll persistent provider voices. Gemini voice replication is not used because the benchmark references lack its separate consent recording.

Stage selection uses `STAGES` or `--stages`. The fixed order is `generate`, `transcribe`, `sim`, `score`; SIM only runs in cloning mode. To rerun ASR/SIM against a previously uploaded bucket, skip generation. The pipeline downloads manifests first, scores the bucket WAVs, then downloads updated manifests while keeping local WAVs:

```bash
MODEL=fish/s2.1-pro VOICE_CLONE=true STAGES="transcribe sim score" \
  ASR_OVERWRITE=true SIM_OVERWRITE=true RESULTS_BUCKET=your-org/tts-api-results \
  bash api/submit_jobs.sh --only_langs en
```

Scorer Jobs use the shared public `bezzam/evals` image, inject this checkout's scorer scripts and default to `l4x1`. `SPACE`, `ASR_FLAVOR`, `SIM_FLAVOR`, `ASR_BATCH_SIZE`, `SIM_BATCH_SIZE`, `MAX_AUDIO_SECONDS` and `SIM_BACKEND` have the same meanings as in the existing backend pipelines. The default SIM backend is `wavlm_seed_tts`; its scores are separate from `xvector` scores. API manifests record SHA-256 hashes of the generated and reference WAVs. SIM resumes reuse a prior score only when both hashes match, including forks retained in the bucket after a generation-only overwrite. Older API rows without these hashes are recomputed. Result export rejects stale SIM forks.

Inspect the planned Docker pull/run and scoring commands without filesystem writes, API calls, uploads or Jobs:

```bash
MODEL=elevenlabs/eleven_v4 RESULTS_BUCKET=your-org/tts-api-results \
  bash api/submit_jobs.sh --datasets seed_tts --only_langs en --dry_run
```

## API time to first audio

A dedicated TTFA probe writes a JSON sidecar only. Run it with the generation stage alone; `--ttfa_probe=-1` probes the full selected split. The shared probe is sequential (batch size one), samples evenly across the split for a positive count and excludes the first three samples from its summary when enough samples exist.

```bash
MODEL=cartesia/sonic-3.6-2026-08-27 STAGES=generate \
  bash api/submit_jobs.sh --datasets cv3 --only_langs en --ttfa_probe=-1
```

Use `--ttfa_probe 8` for a quick probe. Leave `MAX_EVAL_SAMPLES=-1` for comparable full-split measurements. Files end in `__api.json`, keeping API probes out of the GPU/CPU streaming result groups. Streaming PCM providers timestamp the first complete audio frame. Fish WAV, MiniMax MP3 and Mistral WAV currently use full-utterance latency in the dedicated probe and are labeled accordingly. Per-sample `api_ttfa_ms` in generation manifests is null when a first-audio timestamp is unavailable.

TTFA and request latency depend on the caller's region, connection, provider quota, concurrency and retries. Report those run conditions with results. The exposed endpoints do not provide the local GPU clock or provider hardware, and API throughput must be presented separately from the leaderboard's H200 RTFx.

## Output and offline validation

Each split produces generated WAVs, a standard `MODEL_...jsonl` manifest and a matching `.run.json` metadata file containing provider/model/voice, dataset/language, input hash, concurrency, sample coverage, retries and API timing summaries. Failed requests remain failures; no silence or successful placeholder sample is substituted. An incomplete generation run exits unsuccessfully and can be resumed with unchanged settings.

Score already transcribed local manifests without HF Jobs:

```bash
python api/score_results.py --model_id elevenlabs/eleven_v4
```

Each manifest is scored with its recorded language. Exports retain completed measured results from earlier models, keyed by model/dataset/split/cloning mode, and include sample coverage. Incomplete generation, missing ASR and unfinished cloning SIM are rejected; partial runs remain in their manifests/metadata until resumed. Use these API exports for review; the generic H200 result publisher is not an API timing publishing path.

A private-data-free generation smoke test can use a local JSONL with `id`, `text`, and, for reference-based models, `prompt_audio_filepath` and `prompt_text`. Paths in that file should be absolute. The wrapper mounts the JSONL and its prompt audio files read-only at the expected absolute paths inside the container, preserving path aliases such as macOS `/tmp`:

```bash
MODEL=elevenlabs/eleven_v4 STAGES=generate \
  bash api/submit_jobs.sh --datasets seed_tts --only_langs en --input_jsonl /absolute/path/samples.jsonl
pip install pytest
python -m pytest api/tests -q
```

Offline tests use mocked HTTP responses and local audio to verify adapter schemas, fragmented streaming payloads, retry/decoding behavior, manifest compatibility, orchestration and scoring export. They do not measure provider quality or validate an account's model/voice availability.

## Results to collect before finalizing

For every model below, collect the full supported standard-voice splits, ASR quality, dedicated full-split API TTFA, and recorded run conditions. Collect cloning/SIM for Fish Audio and Mistral as well. Review coverage, failed requests, output audio and voice/model access during each provider's initial smoke test. No provider results have been added by this integration.

- [ ] `elevenlabs/eleven_v4`
- [ ] `elevenlabs/eleven_v4_turbo`
- [ ] `cartesia/sonic-3.6-2026-08-27`
- [ ] `minimax/speech-2.8-hd`
- [ ] `minimax/speech-2.8-turbo`
- [ ] `inworld/inworld-tts-2`
- [ ] `inworld/inworld-tts-2-flash`
- [ ] `gemini/gemini-3.8-flash-tts`
- [ ] `gemini/gemini-3.8-flash-lite-tts`
- [ ] `fish/s2.1-pro` (standard and cloning)
- [ ] `fish/s2-pro` (standard and cloning)
- [ ] `mistral/voxtral-mini-tts-2603` (standard and cloning)
- [ ] `smallestai/lightning_v3.1_pro`
- [ ] `smallestai/lightning_v3.1`
- [ ] `deepgram/aura-2-thalia-en`
