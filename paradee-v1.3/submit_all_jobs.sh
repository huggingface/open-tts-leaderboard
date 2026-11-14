#!/bin/bash
bash paradee-v1.3/submit_jobs.sh                                # full eval
bash submit_ttfa_jobs.sh paradee-v1.3                           # TTFA on GPU (a100-large)
TTFA_DEVICE=cpu bash submit_ttfa_jobs.sh paradee-v1.3           # TTFA on CPU (cpu-upgrade)
