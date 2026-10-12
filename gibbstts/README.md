# GibbsTTS EN/ZH voice-cloning backend

Model: [ydqmkkx/GibbsTTS](https://huggingface.co/ydqmkkx/GibbsTTS). Upstream documentation: [GitHub](https://github.com/ydqmkkx/GibbsTTS).

The submitted configuration uses native batch32 DiT/CTMC inference, BF16 estimator autocast, TorchInductor compilation and a padded final batch. Text/reference preprocessing and codec calls remain per real utterance. The base checkpoint supports English and Chinese; the Japanese LoRA and optional no-cloning TTFA probe are separate.

## Environment and reproducibility

The public Docker Space [bezzam/evals](https://huggingface.co/spaces/bezzam/evals) is used with `bootstrap.sh`, which installs pinned dependencies and checks out GibbsTTS commit `2af61715fe832b8dbef909b545090a56056428a3`. The evaluated image runs PyTorch 2.8.0+cu128. Setup and downloads precede generation timing. Creating a dedicated Docker Space on this account returned HTTP402 requiring PRO, so this backend uses the existing public image and runtime bootstrap.

Weights: `a31945f20bf84498af41508ff3b126ea7f53b5c2`. Metadata: MIT; voice cloning yes; batch inference yes; two base languages; approximately 0.4B generative parameters (codec excluded); Transformers backend no.

Defaults: 32 steps, CFG2.5, rescale CFG0.75, temperature0.6, top-p1, speed1, seed42 per batch. Model/codec weights, logits and solver stay FP32; BF16 is estimator autocast only. `batched_inference.py` and `compiler_backend.py` are the exact files used by the full experiments. Static text/language embeddings and masks are prepared once per batch; sorting is skipped for top-p1. The RoPE cache is initialized to 16384 positions during setup.

## Timing, padding and resume

RTFx is real audio duration divided by synchronized wall-clock generation time on H200. Timing includes phonemization, local reference reading/resampling/encoding, packing/static cache, DiT/CFG/logits, sampling and real codec decoding/CPU return. Model loading, reference-file preparation, output writes/transfers and ASR/SIM are excluded.

Only the initial two warm-up calls on the first pending batch are excluded. No per-batch warm-up is used; later compilation stays timed and is recorded. Inductor uses its default mode, `fullgraph=True`, `dynamic=True`; CUDA Graph is not explicitly enabled.

The final batch repeats one valid prepared row to reach batch32. Dummy rows have no extra frontend/reference encoding or output decoding, and their outputs are discarded. All dummy DiT/CTMC compute is timed; duration and WER/CER/SIM include real rows only. Full batch time is divided by real count, including a four-row tail. Generation follows official order with seed42 per batch, with no retries or score-based selection.

`generation_config.json` records settings, warm-ups, batch counts/times and compiler events. `--resume` rejects changed settings and repairs whole incomplete original batches, preserving their RNG shape/timing allocation; complete runs are skipped. Batch shape and compilation can change waveforms; reported quality is from the actual optimized outputs.

## Run on HF Jobs

Prepare private copies named `seed_tts_eval` and `cv3_eval` with the repository's preparation scripts. Authenticate locally with `hf auth login`; the token needs Jobs management and access to your private datasets/bucket. Use a fresh bucket for a different configuration.

```bash
export DATASET_NAMESPACE=YOUR_USERNAME
export RESULTS_BUCKET=YOUR_USERNAME/your_private_results_bucket
export TTS_SPACE=bezzam/evals
export GIBBSTTS_BOOTSTRAP=true

# Paid eight-sample smoke test.
MAX_EVAL_SAMPLES=8 ONLY_LANGS=en STAGE_RETRIES=0 bash gibbstts/submit_jobs.sh

# Paid full run: generation -> ASR -> SIM -> local scoring.
MAX_EVAL_SAMPLES=-1 ONLY_LANGS="en zh" bash gibbstts/submit_jobs.sh
```

Generation uses H200; shared Qwen3-ASR and WavLM-Seed-TTS scoring use L4. To re-score existing audio, use `STAGES="transcribe sim"`; after regenerating audio set `ASR_OVERWRITE=true SIM_OVERWRITE=true`.

## Completed evaluation

All 4108 real rows were generated, scored and checked against official IDs/text/order and exact baseline reference PCM. Eight full GPU jobs completed. The native inference modules are unchanged from those runs; aggregate metrics, data revisions and job links are in `results.json`. Benchmark text and reference audio remain private.

| Dataset | N | WER/CER | RTFx | SIM |
|---|---:|---:|---:|---:|
| Seed-TTS/en | 1088 | 0.92% | 25.49 | 72.86% |
| Seed-TTS/zh | 2020 | 0.77% | 41.36 | 77.27% |
| CV3/en | 500 | 2.87% | 26.01 | 66.75% |
| CV3/zh | 500 | 2.50% | 31.20 | 77.44% |

Language composites: EN WER1.90%, RTFx25.71, SIM69.81%; ZH CER1.64%, RTFx38.11, SIM77.35%. Error/SIM use dataset macro-averages; RTFx pools audio/time. Compared with the previously evaluated uncompiled BF16/b32 path, CV3/zh CER changes 2.34% -> 2.50% and SIM77.66% ->77.44%. This is a throughput/quality tradeoff, not numerically identical generation.

## Validation

```bash
python -m pytest gibbstts/test_backend.py -q
bash -n gibbstts/submit_jobs.sh scripts/tts_jobs_common.sh
PYTHONPATH=scripts python gibbstts/run_eval.py --help
```

Offline tests use real WAV/HF Audio/manifest I/O with a fake batch generator/CUDA interface. They check real-only timing allocation, padded-tail counts, whole-batch repair and actual H200 job-command generation. They are not CUDA measurements. The [eight-row H200 production-entry smoke](https://huggingface.co/jobs/ydqmkkx/6accb4b1fee2c900701a534e) passed: 8 real outputs, execution batch32, 24 dummy rows discarded, no timed recompilation, and byte-identical completed resume. The private smoke results are separate from the full evaluation.

For CPU development use `--device=cpu --precision=fp32 --no-compile`; CPU/other-GPU timing is not official leaderboard timing.
