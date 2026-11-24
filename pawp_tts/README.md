# Submitting PAWP-TTS to `hf-audio/open_tts_leaderboard`

This directory is a drop-in backend package matching the exact structure of `sanotts/` and `pivo/` in [`huggingface/open-tts-leaderboard`](https://github.com/huggingface/open-tts-leaderboard).

## Steps to Open the PR

1. Fork and clone `https://github.com/huggingface/open-tts-leaderboard`.
2. Copy this folder into the fork root as `pawp_tts/`:
   ```bash
   cp -r tools/open_tts_leaderboard/pawp_tts /path/to/open-tts-leaderboard/pawp_tts
   chmod +x /path/to/open-tts-leaderboard/pawp_tts/submit_jobs.sh
   ```
3. Run a quick 5-sample local smoke test against `Seed-TTS en`:
   ```bash
   cd /path/to/open-tts-leaderboard
   PYTHONPATH=scripts python pawp_tts/run_eval.py \
     --model_id Firoj112/pawp-tts-1.5m \
     --dataset_path hf-audio/seed-tts-eval \
     --dataset en --split test --device cpu --max_eval_samples 5
   ```
4. Commit `pawp_tts/` and open a Pull Request to `huggingface/open-tts-leaderboard` requesting evaluation of `Firoj112/pawp-tts-1.5m` (1,490,390 params, 5.98 MB) and `Firoj112/pawp-tts-300k` (281,078 unique params, 1.12 MB) on `en` (`generate`, `transcribe`, `utmos`, and CPU `--ttfa_probe`).
