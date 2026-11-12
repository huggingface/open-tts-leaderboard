#!/bin/bash
# Every ZONOS3-Mini job: the full eval with the fixed voice, then with voice cloning (sequential), in parallel with
# the TTFA probe on A100, H200 and CPU. Each launcher waits for its own jobs and scores them; this script waits for all.
# Usage: HF_TOKEN=hf_... bash zonos3-mini/submit_all_jobs.sh
# GPU: Mini-SGLang CUDA kernels; the two GPUs run different ones (env.sh picks the Hopper-only SonicMoE experts and
# TileLang GDN on the H200, the portable Triton ones on the A100). CPU: the runtime's native ggml server on its own
# image (Zyphra/evals-zonos3-mini-cpu, the quantized zonos3-ptq.gguf backbone), TTFA only.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# The two evals run one after the other (fixed voice first), so their HF Jobs don't all run at once.
{
    bash zonos3-mini/submit_jobs.sh                               # fixed voice (native voice per language), no SIM
    VOICE_CLONE=true bash zonos3-mini/submit_jobs.sh              # zero-shot voice cloning (speaker embedding) + SIM
} &
bash submit_ttfa_jobs.sh zonos3-mini &                            # TTFA on A100 (a100-large, the default)
TTFA_FLAVOR=h200 bash submit_ttfa_jobs.sh zonos3-mini &           # TTFA on H200
TTFA_DEVICE=cpu bash submit_ttfa_jobs.sh zonos3-mini-ggml &       # TTFA on CPU (cpu-upgrade, ggml server)
wait
