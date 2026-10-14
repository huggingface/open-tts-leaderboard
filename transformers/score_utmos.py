"""
Predicted MOS (UTMOS) for the Open TTS Leaderboard (stage 4).

Scores each generated wav with UTMOS22 strong (https://github.com/tarepan/SpeechMOS) and writes
`UTMOS_<manifest stem>.json` next to the manifest (or in `--output_dir`). Resumable unless `--overwrite`.
"""

import argparse
import json
import os

import torch
from transformers.audio_utils import load_audio

# v1.2.0, pinned by commit; the checkpoint is pinned by digest.
SPEECHMOS_REPO = "tarepan/SpeechMOS:ed25eacbfa42b99156c36ebec67a733b5dbb9b79"
SPEECHMOS_ENTRY = "utmos22_strong"
CKPT_URL = "https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt"
CKPT_SHA256 = "38aa51ab79e2a4e09a1449758a4b37e9cbb2e8235a49662a732d33a9ba1e9bff"
SAMPLING_RATE = 16_000
MIN_SAMPLES = SAMPLING_RATE // 10  # shorter clips crash the conv front end


def score_manifest(model, manifest_path, args):
    with open(manifest_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()][: args.max_rows if args.max_rows > 0 else None]
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    stem = os.path.splitext(os.path.basename(manifest_path))[0]
    out_path = os.path.join(args.output_dir or manifest_dir, f"UTMOS_{stem}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    scores = {}
    if os.path.exists(out_path) and not args.overwrite:
        with open(out_path, encoding="utf-8") as f:
            scores = json.load(f)["scores"]

    def save():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"utmos_model": f"{SPEECHMOS_ENTRY} ({CKPT_SHA256[:12]})", "scores": scores}, f)

    todo = [e["audio_filepath"] for e in entries if e.get("audio_filepath") and e["audio_filepath"] not in scores]
    print(f"{manifest_path}: {len(todo)}/{len(entries)} rows to score -> {out_path}")
    max_samples = int(args.max_audio_seconds * SAMPLING_RATE) if args.max_audio_seconds > 0 else None
    for n, rel in enumerate(todo, 1):
        try:
            speech = load_audio(os.path.join(manifest_dir, rel), sampling_rate=SAMPLING_RATE)[:max_samples]
        except Exception as e:  # left unscored, retried on resume
            print(f"  WARNING: could not load {rel}: {e}")
            continue
        if len(speech) < MIN_SAMPLES:
            scores[rel] = None
            continue
        with torch.no_grad():
            wav = torch.from_numpy(speech).float().unsqueeze(0).to(args.device)
            scores[rel] = round(model(wav, SAMPLING_RATE).item(), 6)
        if n % args.save_every == 0:
            save()
    save()

    valid = [scores[e["audio_filepath"]] for e in entries if isinstance(scores.get(e.get("audio_filepath")), float)]
    if valid:
        print(f"  UTMOS = {sum(valid) / len(valid):.3f} over {len(valid)}/{len(entries)} rows")


def main(args):
    model = torch.hub.load(SPEECHMOS_REPO, SPEECHMOS_ENTRY, trust_repo=True, pretrained=False)
    ckpt = os.path.join(torch.hub.get_dir(), "checkpoints", f"{CKPT_SHA256}.pt")
    if not os.path.exists(ckpt):
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.hub.download_url_to_file(CKPT_URL, ckpt, hash_prefix=CKPT_SHA256)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model = model.to(args.device).eval()
    for manifest_path in args.manifest_paths:
        score_manifest(model, manifest_path, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default=None, help="Default: next to each manifest.")
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--max_audio_seconds", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="cuda")
    main(parser.parse_args())
