#!/bin/bash
# Every Inflect-Mini-v3 job, launched in parallel: the full eval (Seed-TTS, CV3 and MiniMax en; fixed
# voice) and the TTFA probe on A100 and on H200. Each launcher waits for its own jobs and scores them;
# this script waits for all three.
# Usage: HF_TOKEN=hf_... bash inflect-v3/submit_all_jobs.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

bash inflect-v3/submit_jobs.sh &                              # eval: generate → transcribe → UTMOS → score
bash submit_ttfa_jobs.sh inflect-mini-v3 &                    # TTFA on A100 (a100-large, the default)
TTFA_FLAVOR=h200 bash submit_ttfa_jobs.sh inflect-mini-v3 &   # TTFA on H200
wait
