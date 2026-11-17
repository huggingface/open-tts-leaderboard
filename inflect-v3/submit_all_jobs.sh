#!/bin/bash
# Inflect v3 full launch: main eval (Seed-TTS, CV3, MiniMax en; fixed voice) + TTFA probes.
# Mini-v3 is already published (see tts_leaderboard_results PR #133); this script covers Micro
# (GPU — A100 + H200 TTFA) and Atom (CPU-only runtime — cpu-upgrade TTFA).
# Each launcher waits for its own jobs and scores them; this script waits for all of them.
# Usage: HF_TOKEN=hf_... bash inflect-v3/submit_all_jobs.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# Main eval (Micro + Atom; one job chain per model × dataset).
bash inflect-v3/submit_jobs.sh &

# GPU TTFA: A100 (default) + H200 for Micro.
bash submit_ttfa_jobs.sh inflect-micro-v3 &
TTFA_FLAVOR=h200 bash submit_ttfa_jobs.sh inflect-micro-v3 &

# CPU TTFA: Atom only (CPU-only runtime).
TTFA_DEVICE=cpu bash submit_ttfa_jobs.sh inflect-atom-v3 &

wait
