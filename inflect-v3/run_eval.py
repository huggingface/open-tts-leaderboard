"""
TTS synthesis for the Open TTS Leaderboard (Inflect v3 backend, stage 1).

Covers all three Inflect v3 releases (apache-2.0, English, six voices, 24 kHz, fixed voice —
no reference conditioning, so no voice-clone mode and no SIM stage):

  * ``owensong/Inflect-Mini-v3``  — 34.31M synth-core params. Entry is ``core/synthesize.py``'s
    ``load_selected('mini', ..., profile='base', runtime='canonical')``; MiniDelivery wrapper
    (``core/selected_mini.py``) loads the pitch component but leaves it idle at the base profile.
  * ``owensong/Inflect-Micro-v3`` — 11.79M synth-core params. Root-level ``synthesize.py``
    (``verify_candidate``, ``load_numeric``, ``prepare_text``, ``load_runtime('micro', device)``,
    ``render_prepared`` — the same shape as Mini's ``core/baseline/synthesize.py`` but with no
    wrapper): we drive it directly from Python rather than its CLI. Runtime module is
    ``models/micro/code/fastmicro_inference_20261029.py``.
  * ``owensong/Inflect-Atom-v3`` — tiny CPU-only AtomDSP export (~4 MiB .pt). Root-level
    ``atom_runtime_20261024.AtomRuntime(export.pt, vocab.json, device='cpu',
    g2p=make_offline_misaki(), dsp_backend='torch')`` with ``model.synthesize_text(text, voice,
    seed=...)`` returning audio directly (no explicit chunking or phoneme pass in the caller).

The three share misaki 0.9.4 + spaCy G2P from the same stage-1 image (bezzam/evals-inflect-v3),
and the eval-loop plumbing (batching, resume, TTFA probe) is the same for all three. The variant
is detected from the repo's own files so ``--model_id`` alone picks the right code path.

Synthesis follows each release's own CLI (Mini: ``core.synthesize.render``; Micro: baseline-style
``render_prepared``; Atom: ``AtomRuntime.synthesize_text``) — chunk-at-punctuation segmentation,
Misaki + espeak fallback, each chunk/utterance seeded with ``seed + index``, chunks joined with
0.12 s of silence (for Mini/Micro only — Atom's runtime does its own chunking internally).
"""

import argparse
import importlib.util
import io
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
SEGMENT_GAP = 2880  # samples of silence between segments (0.12 s), as in each release's render path.


def _load_module(path, name):
    """Load a file as a module without touching sys.path (so repeated runs don't collide)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _detect_variant(repo_dir: Path) -> str:
    """One of ``'mini'``, ``'micro'``, ``'atom'`` — based on the files in the downloaded repo.

    * Mini ships ``core/synthesize.py`` (the MiniDelivery wrapper over the baseline).
    * Atom ships ``atom_runtime_20261024.py`` + ``export.pt`` at the repo root.
    * Micro ships a root-level ``synthesize.py`` plus ``models/micro/weights.pt``.
    """
    if (repo_dir / "core" / "synthesize.py").is_file():
        return "mini"
    if (repo_dir / "atom_runtime_20261024.py").is_file():
        return "atom"
    if (repo_dir / "synthesize.py").is_file() and (repo_dir / "models" / "micro" / "weights.pt").is_file():
        return "micro"
    raise RuntimeError(f"Unknown Inflect v3 package layout under {repo_dir}")


# Micro and Atom's ``verify_candidate`` / ``verify_manifest`` hashes every file the release
# manifest names (incl. ``.gitattributes``, ``LICENSES/*``, ``assets/*``). Pulling the whole repo
# is simpler than enumerating that list — ~50 MB for Micro, ~5 MB for Atom. Mini's verify only
# checks files under ``core/baseline`` (``core/synthesize.verify`` over the outer manifest is only
# run by its ``main()``, not by our ``load_selected`` call), so the narrower pattern stays.
_ALLOW_PATTERNS = {
    "mini":  ["core/**", "*.json", "LICENSE*", "README*"],
    "micro": None,   # pull everything
    "atom":  None,   # pull everything
}


def _snapshot(args, variant_hint: str | None = None):
    """Download the subset of the repo this variant needs. Variant is detected after download
    unless the caller already knows it (TTFA probes re-use a cached one).

    Micro and Atom's verify_candidate / verify_manifest hashes every file in their release
    manifest (including .gitattributes / LICENSES / assets), so for them we don't pass
    ``allow_patterns`` — pulling the whole repo is a few more MB and avoids missing a file.
    """
    from huggingface_hub import snapshot_download
    # Without a hint we don't know which allow_patterns to use, and Mini/Micro/Atom share no
    # distinguishing filename we could key on cheaply — so just pull everything in that case.
    # With a hint, use the per-variant patterns (None = pull everything).
    patterns = _ALLOW_PATTERNS[variant_hint] if variant_hint else None
    kwargs = dict(
        revision=args.revision,
        local_dir=Path(args.local_dir) / args.model_id.replace("/", "-"),
    )
    if patterns:
        kwargs["allow_patterns"] = patterns
    return Path(snapshot_download(args.model_id, **kwargs))


# ── Loaders ─────────────────────────────────────────────────────────────────
def _load_mini(repo_dir: Path, device: str):
    core_dir = repo_dir / "core"
    core_syn = _load_module(core_dir / "synthesize.py", "_inflect_v3_core_synthesize")
    cli, runtime, _ = core_syn.load_selected(
        "mini", device, profile="base", runtime="canonical", root=core_dir,
    )
    numeric = cli.load_numeric(core_dir / "baseline")
    base_model = runtime[2].base  # MiniDelivery.base is the raw nn.Module
    config = json.loads((repo_dir / "config.json").read_text(encoding="utf-8"))
    n_params = sum(p.numel() for p in base_model.parameters())
    assert n_params == config["core_parameters"], (n_params, config["core_parameters"])
    print(f"Loaded Inflect v3 mini; TTS model size: {n_params / 1e9:.4f}B parameters "
          f"({n_params:,} synthesis core)")
    return cli, runtime, numeric


def _load_micro(repo_dir: Path, device: str):
    cli = _load_module(repo_dir / "synthesize.py", "_inflect_v3_root_synthesize")
    cli.verify_candidate(repo_dir)
    runtime = cli.load_runtime(repo_dir, "micro", device)
    numeric = cli.load_numeric(repo_dir)
    # The runtime object (``FastMicro``) isn't an ``nn.Module`` — its weights are inside the
    # tensors it holds, not exposed as ``.parameters()``. Model-card lists 11.79M synth-core
    # parameters; we don't need the live count here, just a human-readable load confirmation.
    print("Loaded Inflect v3 micro (~11.79M synth-core parameters; see the model card).")
    return cli, runtime, numeric


def _load_atom(repo_dir: Path, device: str):
    """Atom is CPU-only; its CLI (``atom_preview_cli.run``) sets ``CUDA_VISIBLE_DEVICES=''`` and
    pins torch to 2 threads. We mirror that here — device=cuda is treated as cpu."""
    if device != "cpu":
        print("Atom runtime is CPU-only; overriding --device to cpu for inference.")
    sys.path.insert(0, str(repo_dir))
    native_mini_text = _load_module(repo_dir / "native_mini_text.py", "native_mini_text")
    native_espeak = _load_module(repo_dir / "native_mini_text_espeak.py", "native_mini_text_espeak")
    g2p = native_espeak.make_offline_misaki()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    atom_runtime = _load_module(repo_dir / "atom_runtime_20261024.py", "_atom_preview_runtime")
    model = atom_runtime.AtomRuntime(
        repo_dir / "export.pt", repo_dir / "vocab.json",
        device="cpu", g2p=g2p, dsp_backend="torch",
    )
    # AtomRuntime's own CLI verifies the manifest; the eval path trusts the download hash.
    # Expose a VOICES attribute like Mini/Micro's cli for voice validation.
    class _AtomCli:
        VOICES = ("hope", "sophia", "cassidy", "james", "jasper", "mark")
    return _AtomCli(), model, None


# ── Synth hot paths ─────────────────────────────────────────────────────────
def _synth_fn_mini_micro(cli, runtime, numeric, voice, seed):
    """Both Mini and Micro expose the same (core, frontend, model, torch, soundfile) tuple and
    the same ``synthesize_phonemes`` API, so the per-chunk inner loop is identical."""
    core, frontend, model, torch_mod, _sf = runtime

    def synthesize(text):
        prepared = cli.prepare_text(text, numeric, "conservative")
        waves = []
        with torch_mod.inference_mode():
            for index, chunk in enumerate(prepared["chunks"]):
                phones = core.prepare_phonemes(chunk, model.vocab, g2p=frontend)["ps"]
                result = model.synthesize_phonemes(phones, voice, seed=(seed + index) % 2**32)
                if result["sample_rate"] != SAMPLE_RATE:
                    raise ValueError(f"Unexpected sample rate {result['sample_rate']}")
                wave = result["audio"].detach().cpu().float().numpy().astype(np.float32, copy=False)
                if index:
                    waves.append(np.zeros(SEGMENT_GAP, dtype=np.float32))
                waves.append(wave)
        return np.concatenate(waves)
    return synthesize


def _synth_fn_atom(model, voice, seed):
    """Atom's ``synthesize_text`` returns ``{"audio": one_1d_tensor, "sample_rate": 24000, ...}``
    where chunks are already concatenated internally with 80 ms silence (see
    ``atom_runtime_20261024.AtomRuntime.synthesize_text``). We just pass it through."""
    def synthesize(text):
        result = model.synthesize_text(text, voice, seed=seed)
        if result.get("sample_rate", SAMPLE_RATE) != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate {result['sample_rate']}")
        audio = result["audio"]
        wave = audio.detach().cpu().float().numpy().astype(np.float32, copy=False) if hasattr(audio, "detach") else np.asarray(audio, dtype=np.float32)
        if wave.ndim != 1 or wave.size == 0:
            raise ValueError("Atom runtime returned empty or non-1D audio")
        return wave
    return synthesize


def main(args):
    repo_dir = Path(args.checkpoint_path) if args.checkpoint_path else _snapshot(args)
    print(f"Checkpoint: {repo_dir} (revision {args.revision})")

    variant = _detect_variant(repo_dir)
    print(f"Detected Inflect v3 variant: {variant}")

    # Per-variant release device: Mini/Micro accept 'cuda' or 'cpu'; Atom is CPU-only (its CLI
    # sets CUDA_VISIBLE_DEVICES=''). 'cuda:N' is normalized to 'cuda' for the release APIs.
    if variant == "atom":
        release_device = "cpu"
    else:
        release_device = "cuda" if args.device.startswith("cuda") else "cpu"

    if variant == "mini":
        cli, runtime, numeric = _load_mini(repo_dir, release_device)
        synth_text = _synth_fn_mini_micro(cli, runtime, numeric, args.voice, args.seed)
    elif variant == "micro":
        cli, runtime, numeric = _load_micro(repo_dir, release_device)
        synth_text = _synth_fn_mini_micro(cli, runtime, numeric, args.voice, args.seed)
    else:
        cli, atom_model, _ = _load_atom(repo_dir, release_device)
        synth_text = _synth_fn_atom(atom_model, args.voice, args.seed)

    if args.voice not in cli.VOICES:
        raise SystemExit(f"--voice must be one of {cli.VOICES}")

    # Atom is CPU-only; disable cuda.synchronize even if the caller asked for cuda.
    is_cuda = args.device.startswith("cuda") and variant != "atom"
    failures = []

    def synth_one(text):
        """Return (audio, elapsed_s). A text the runtime refuses (unsupported phonemes, chunk-limit
        violation, clipping) is scored as silence."""
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            audio = synth_text(text)
        except (ValueError, RuntimeError) as exc:
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
    if args.ttfa_probe != 0:
        from ttfa_probe import run_probe, sample_indices
        run_probe(
            lambda job: (synth_one(job["text"])[0], SAMPLE_RATE, None),
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            extra={"device": args.device, "voice": args.voice, "seed": args.seed,
                   "revision": args.revision, "variant": variant},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

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


# Pinned revisions (main HEAD at 2026-11-16). Chosen by --model_id; fall back to the Mini default.
_REVISIONS = {
    "owensong/Inflect-Mini-v3":  "0490aee6a9c66f9375b48287a2c5e5788afd00cc",
    "owensong/Inflect-Micro-v3": "077eac5c4f0c719bbb428cd29f06a3cf6c6e8a27",
    "owensong/Inflect-Atom-v3":  "1f06119fd84c0156132382ff7519cdcea6d48e46",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="owensong/Inflect-Mini-v3",
                        help="Inflect v3 repo id: Mini, Micro, or Atom (all apache-2.0).")
    parser.add_argument("--revision", type=str, default=None,
                        help="Model repo revision (default: the pinned main HEAD for this model_id).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local copy of the repo. Default: download --model_id under --local_dir.")
    parser.add_argument("--local_dir", type=str, default="/tmp/inflect_v3_models",
                        help="Where to materialise the model repo as real files (not cache symlinks).")
    parser.add_argument("--voice", type=str, default="cassidy",
                        help="Fixed voice id: hope, sophia, cassidy, james, jasper or mark. The card's "
                             "first recommendation, and the voice of its own no-tag WER panel.")
    add_common_args(parser)
    parser.set_defaults(seed=1234)

    args = parser.parse_args()
    if args.revision is None:
        args.revision = _REVISIONS.get(args.model_id, _REVISIONS["owensong/Inflect-Mini-v3"])

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
