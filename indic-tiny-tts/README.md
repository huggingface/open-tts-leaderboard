# indic-tiny-tts (Hindi) backend

Open TTS Leaderboard backend for [`sulabhkatiyar/indic-tiny-tts-hindi`](https://huggingface.co/sulabhkatiyar/indic-tiny-tts-hindi),
a small single-language Hindi model (collection: [indic-tiny-tts-models](https://huggingface.co/collections/sulabhkatiyar/indic-tiny-tts-models)).

| | |
|---|---|
| Architecture | Matcha-style flow-matching acoustic model (8.9M, character input, 4 Euler steps) + Vocos vocoder (13.5M); no eSpeak / no GPL component |
| Size | 0.0225B parameters |
| Output | 24 kHz mono |
| Voices | 2 fixed (`female`, `male`); **no voice cloning** (only the "no cloning" table applies) |
| Batch inference | no (one text per call; the backend loops, batch size 1) |
| Languages | 1 (`hi`); `supports_language()` rejects every other split |
| `transformers` | no (the HF repo ships its own runtime package `indic_tiny_tts/`) |
| License | weights CC-BY-4.0 (attribution: AI4Bharat Rasa Hindi data, CC-BY-4.0); code and Vocos vocoder MIT |
| Training data | AI4Bharat Rasa Hindi only (no MiniMax, no leaderboard text) |

## How it runs

- `run_eval.py` downloads the model repo at a pinned revision (`snapshot_download(revision=...)`), puts the folder on `sys.path`,
  and loads `indic_tiny_tts.TinyTTS` (PyTorch class, `device=cuda:0`). The raw dataset text is synthesized as is (the model has its own
  bundled Hindi text normaliser that spells out digits); the leaderboard normaliser is only applied at scoring.
  One voice per run (`VOICE`, default `male`), serial loop, whole `synthesize()` call timed, 2 untimed warm-ups, per-call seed.
- `submit_jobs.sh` has no `declare -A` (runs on Linux and macOS bash 3.2). Stage 1 runs on the default `h200` flavor with a
  30 min timeout (`GEN_TIMEOUT`). By default it uses the public `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` image and
  `pip install`s `datasets soundfile tqdm huggingface_hub safetensors numpy` at job start, so no Docker Space is needed. The
  `Dockerfile` here does the same, for a maintainer who prefers a Space (`TTS_SPACE=<user>/<space>` then skips the pip step).
- Hindi only exists in MiniMax: `DATASET_CONFIGS=("tts hi hi ${MINIMAX_EVAL_PATH}")`, stages `generate transcribe`.

```bash
# RESULTS_BUCKET defaults to the official bucket if unset (shared script): always pass your own.
# MODEL_REVISION (default: the full 40-char sha e15ac436... of sulabhkatiyar/indic-tiny-tts-hindi, package version 0.1.0) and
# VOICE (default: male; female also available) are shown for clarity.
HF_TOKEN=hf_... \
RESULTS_BUCKET=YOUR_USERNAME/your_bucket \
MODEL_REVISION=e15ac436560eb24d115d47c0783f62f10a02d30a \
VOICE=male \
MAX_EVAL_SAMPLES=8 ONLY_LANGS=hi ONLY_DATASETS=minimax_eval \
bash indic-tiny-tts/submit_jobs.sh          # smoke test; drop MAX_EVAL_SAMPLES for the full 100 sentences
```

## Known limitations

- Devanagari only: Latin letters in the input are dropped by the model's front-end (MiniMax Hindi has a few such tokens, e.g. "5G", "3D").
- Trained on read speech; conversational prompts are out of domain.
