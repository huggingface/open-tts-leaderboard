#!/bin/bash
# Every ZONOS3-Mini job, launched in parallel: the full eval with the fixed voice and with voice cloning, and the
# TTFA probe on A100 and H200. Each launcher waits for its own jobs and scores them; this script waits for all four.
# Usage: HF_TOKEN=hf_... bash zonos3-mini/submit_all_jobs.sh
# No CPU TTFA: the model runs on Mini-SGLang CUDA kernels only. The two GPUs run different kernels (env.sh picks
# the Hopper-only SonicMoE experts and TileLang GDN on the H200, the portable Triton ones on the A100).
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

bash zonos3-mini/submit_jobs.sh &                                 # fixed voice (Savannah), no SIM
VOICE_CLONE=true bash zonos3-mini/submit_jobs.sh &                # zero-shot voice cloning + SIM
bash submit_ttfa_jobs.sh zonos3-mini &                            # TTFA on A100 (a100-large, the default)
TTFA_FLAVOR=h200 bash submit_ttfa_jobs.sh zonos3-mini &           # TTFA on H200
wait
