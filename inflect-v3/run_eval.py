"""
TTS synthesis for the Open TTS Leaderboard (Inflect v3 backend, stage 1).

Covers Inflect v3 (`owensong/Inflect-Mini-v3`, 34.31M synthesis-core parameters): a fixed-voice
English model at 24 kHz with six voice ids and no reference conditioning, so there is no
voice-clone mode and no SIM stage. The repo is apache-2.0; no gating.

This script snapshot_downloads the repo (the `core/**` package) at a pinned revision, then calls
`core.synthesize.load_selected('mini', device, profile='base', runtime='canonical')` to build the
runtime. That helper runs the baseline integrity check (`verify_candidate`) over `core/baseline`
and wires in Mini's `MiniDelivery` wrapper from `core/selected_mini.py`. For base profile with
Cassidy the pitch-polish component is loaded at `MiniDelivery.__init__` time but not invoked.

Synthesis follows `core.synthesize.render()` exactly: full-input numeric preparation via
`baseline.synthesize.prepare_text` (conservative mode), chunking at <=220 chars on punctuation,
Misaki 0.9.4 + spaCy G2P with the bundled eSpeak fallback for each chunk, each chunk seeded with
`seed + index`, chunks joined with 0.12 s of silence. One sample at a time (no batched API); the
timed region covers G2P and synthesis.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, warm_up, wav_rel_path, write_entry,
)

SAMPLE_RATE = 24000
SEGMENT_GAP = 2880  # samples of silence between segments (0.12 s), as in core/synthesize.render.


def _load_module(path, name):
    """Load a file as a module without touching sys.path (so repeated runs don't collide)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_model(repo_dir, device):
    """Load the release's runtime via core.synthesize.load_selected. Return (cli, runtime, enhancer).

    `cli` is the baseline synthesize module (prepare_text, load_numeric); `runtime` is the 5-tuple
    (core, frontend, model, torch, soundfile) where `model` is MiniDelivery; `enhancer` is None for
    profile='base' (we don't publish the 48 kHz enhanced track).
    """
    core_dir = Path(repo_dir) / "core"
    core_syn = _load_module(core_dir / "synthesize.py", "_inflect_v3_core_synthesize")
    cli, runtime, enhancer = core_syn.load_selected(
        "mini", device, profile="base", runtime="canonical", root=core_dir,
    )
    base_model = runtime[2].base  # MiniDelivery.base is the raw nn.Module
    config = json.loads((Path(repo_dir) / "config.json").read_text(encoding="utf-8"))
    n_params = sum(p.numel() for p in base_model.parameters())
    assert n_params == config["core_parameters"], (n_params, config["core_parameters"])
    print(f"Loaded Inflect v3 mini; TTS model size: {n_params / 1e9:.4f}B parameters "
          f"({n_params:,} synthesis core)")
    return cli, runtime, enhancer


def main(args):
    from huggingface_hub import snapshot_download

    if args.checkpoint_path:
        repo_dir = Path(args.checkpoint_path)
    else:
        # `core/**` carries the runtime code + weights; the JSONs anchor verify_candidate's integrity
        # check. local_dir= gives real files (verify_candidate rejects symlinks).
        repo_dir = Path(snapshot_download(
            args.model_id, revision=args.revision,
            local_dir=Path(args.local_dir) / args.model_id.replace("/", "-"),
            allow_patterns=["core/**", "*.json", "LICENSE*", "README*"],
        ))
    print(f"Checkpoint: {repo_dir} (revision {args.revision})")

    # The release's CLI only accepts 'cpu' or 'cuda' (no ':N' suffix): its load_runtime does
    # `allow_cuda=device == 'cuda'`, so 'cuda:0' silently resolves to allow_cuda=False.
    release_device = "cuda" if args.device.startswith("cuda") else "cpu"
    cli, runtime, _enhancer = load_model(repo_dir, release_device)
    _core, _frontend, model, _torch, _sf = runtime
    numeric = cli.load_numeric(repo_dir / "core" / "baseline")
    if args.voice not in cli.VOICES:
        raise SystemExit(f"--voice must be one of {cli.VOICES}")

    is_cuda = args.device.startswith("cuda")
    failures = []

    def synthesize(text):
        """Mirror core.synthesize.render() for one text: prepare chunks, G2P + synth each, join."""
        prepared = cli.prepare_text(text, numeric, "conservative")
        waves = []
        with torch.inference_mode():
            for index, chunk in enumerate(prepared["chunks"]):
                phones = _core.prepare_phonemes(chunk, model.vocab, g2p=_frontend)["ps"]
                result = model.synthesize_phonemes(phones, args.voice,
                                                    seed=(args.seed + index) % 2**32)
                if result["sample_rate"] != SAMPLE_RATE:
                    raise ValueError(f"Unexpected sample rate {result['sample_rate']}")
                wave = result["audio"].detach().cpu().float().numpy().astype(np.float32, copy=False)
                if index:
                    waves.append(np.zeros(SEGMENT_GAP, dtype=np.float32))
                waves.append(wave)
        return np.concatenate(waves)

    def synth_one(text):
        """Return (audio, elapsed_s). A text the runtime refuses (unsupported phonemes, chunk-limit
        violation, clipping) is scored as silence."""
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            audio = synthesize(text)
        except ValueError as exc:
            failures.append(str(exc))
            print(f"WARNING: refused ({exc}); writing silence for: {text[:80]!r}")
            audio = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
        if is_cuda:
            torch.cuda.synchronize()
        return audio, time.perf_counter() - start

    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # No streaming API, so `first` is None and the probe records the whole-utterance time.
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        run_probe(
            lambda job: (synth_one(job["text"])[0], SAMPLE_RATE, None),
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            extra={"device": args.device, "voice": args.voice, "seed": args.seed, "revision": args.revision},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    # Serial synthesis: one sample per "batch"; warm-up runs untimed on the first pending samples.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)
    failures.clear()

    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        audio, elapsed = synth_one(text)
        sf.write(os.path.join(model_dir, rel_path), audio, SAMPLE_RATE)
        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLE_RATE, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    print(f"Refused texts (scored as silence): {len(failures)} of {len(starts)}")
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="owensong/Inflect-Mini-v3",
                        help="Inflect v3 repo (apache-2.0 release).")
    parser.add_argument("--revision", type=str, default="0490aee6a9c66f9375b48287a2c5e5788afd00cc",
                        help="Model repo revision (default: Mini's main as of 2026-11-12).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local copy of the repo. Default: download --model_id under --local_dir.")
    parser.add_argument("--local_dir", type=str, default="/tmp/inflect_v3_models",
                        help="Where to materialise the model repo as real files (not cache symlinks).")
    parser.add_argument("--voice", type=str, default="cassidy",
                        help="Fixed voice id: hope, sophia, cassidy, james, jasper or mark. The card's "
                             "first recommendation, and the voice of its own no-tag WER panel.")
    add_common_args(parser)
    parser.set_defaults(seed=1234)  # same as the Preview wrapper used; core/synthesize uses 42.

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
