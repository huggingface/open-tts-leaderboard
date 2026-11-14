#!/bin/bash
# Every Inflect-Mini-v3-Preview job, launched in parallel: the full eval (Seed-TTS, CV3 and MiniMax en; fixed
# voice) and the TTFA probe on A100 and on CPU. Each launcher waits for its own jobs and scores them; this
# script waits for all three.
# The model repo is gated: HF_TOKEN must belong to an account that accepted its preview terms.
# Usage: HF_TOKEN=hf_... bash inflect-v3/submit_all_jobs.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

bash inflect-v3/submit_jobs.sh &                              # eval: generate → transcribe → UTMOS → score
bash submit_ttfa_jobs.sh inflect-mini-v3 &                    # TTFA on A100 (a100-large, the default)
TTFA_DEVICE=cpu bash submit_ttfa_jobs.sh inflect-mini-v3 &    # TTFA on CPU (cpu-upgrade)
wait
