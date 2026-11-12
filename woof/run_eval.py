"""
TTS synthesis for the Open TTS Leaderboard (Underdog Woof backend, stage 1).

Underdog Woof (ConwayResearch/Underdog-Woof) is a 38M-parameter English TTS (22.95M speech model +
15.3M Mimi-derived audio decoder) with two built-in voices, Juno (default) and Mira. It runs on CPU
with numpy only, from one int8 weights file. The model repo is not a pip package: the Docker image
bakes it at a pinned revision and puts it on PYTHONPATH (see the Space's Dockerfile).

`Woof.__call__` normalizes the text (its own rule-based normalizer), splits it into sentences (and
long sentences into clauses), and generates each piece in turn, redrawing a piece up to 3 times if
it runs too long or stalls. The speech model keeps one KV cache for a single sequence, so there is
no batched API: rows are processed serially, as in the Kokoro and Paradee backends.

Device: numpy CPU only. `--device` must be `cpu`; on the h200 flavor the model uses that host's CPU.
`--threads` caps the BLAS thread pool (threadpoolctl); the model card benchmarks one thread.

Seeding: Woof draws its sampling noise from its own torch-compatible RNG (not np.random), seeded per
call. Every row is synthesized with `seed=args.seed`, so a row's audio depends only on its text,
and a resumed run reproduces the same audio.

Stage 2 (`transformers/transcribe.py`) fills `pred_text` with Qwen3-ASR, and stage 4
(`normalizer.eval_utils.score_results`) computes WER + RTFx (no stage 3 SIM: fixed voice).
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
from threadpoolctl import threadpool_info, threadpool_limits
from tqdm import tqdm

from underdog_woof import SR, VOICES, Woof
from underdog_woof.textsplit import split_text

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

DEFAULT_VOICE = "juno"  # the package default

# Model card: 22.95M speech model + 15.3M audio decoder.
MODEL_PARAMS = 38.24e6


def _print_model_size(tts):
    """Count the weights the runtime actually holds (speech model fp32 after dequant + decoder)."""
    try:
        n = 0
        sp = tts.model.speech
        groups = [sp.layers, [sp.ce, sp.ada, sp.lat_in, sp.h_inp, sp.h_out], sp.hblocks]
        for group in groups:
            for item in group:
                for w, b in (item if isinstance(item, list) else [item]):
                    n += w.size + (0 if b is None else b.size)
        n += sp.text_emb.size + sp.bos.size
        print(f"Speech model: {n / 1e6:.2f}M parameters (model card: 22.95M); "
              f"TTS model size: {MODEL_PARAMS / 1e9:.3f}B parameters (model card total incl. 15.3M decoder)")
    except Exception as e:
        print(f"Could not count parameters ({e}); model card reports {MODEL_PARAMS / 1e6:.2f}M parameters.")


def main(args):
    if args.device != "cpu":
        # Woof has no GPU path. Refuse rather than label a CPU measurement with a GPU device.
        raise ValueError(f"Underdog Woof is numpy CPU inference only; got --device={args.device} (use cpu).")
    set_seed(args.seed)

    tts = Woof(voice=args.voice, temp=args.temp)
    print(f"Loaded Underdog Woof ({args.model_id}@{os.environ.get('WOOF_REVISION', '?')}, voice={args.voice}, "
          f"temp={args.temp}, threads={args.threads or 'unlimited'})")
    print(f"BLAS: {[(p.get('internal_api'), p.get('num_threads')) for p in threadpool_info()]}")
    _print_model_size(tts)

    def synth_one(text):
        """Synthesize one text with the package's own pipeline. Return (audio_1d_numpy, elapsed_s)."""
        start = time.perf_counter()
        audio = tts(text, seed=args.seed)
        elapsed = time.perf_counter() - start
        if audio.size == 0:
            audio = np.zeros(1, dtype=np.float32)
        return audio, elapsed

    def synth_one_streaming(text, early_stop=False):
        """TTFA probe only: Woof.__call__'s own loop (seed, normalize, split, one piece at a time),
        timestamping the first piece's audio. Returns (audio_1d_numpy, first_piece_ts, n_pieces).

        A one-sentence prompt has nothing to stream (n_pieces == 1): first audio is last audio.
        Woof's retry check needs the whole piece, so a piece is the finest unit it can hand back.
        """
        from underdog_woof.runtime import RNG

        RNG.seed(args.seed)
        if tts.norm is not None:
            text = tts.norm(text)
        pieces = split_text(text) or [text]
        first, wavs = None, []
        for piece in pieces:
            wavs.append(tts._piece(piece))
            if first is None:
                first = time.perf_counter()
                if early_stop:
                    break
        return np.concatenate(wavs).astype(np.float32), first, len(pieces)

    # Layout: results/<model_safe>/{MODEL_<safe>_DATASET_<dsid>.jsonl, <dsid>/output_<id>.wav}.
    # Manifest paths are relative to model_dir so later stages resolve them wherever it is copied.
    # Both voices share one model id, so a non-default voice gets its own manifest/sidecar suffix
    # (`_mira`; publish with `open_results_pr.py --variant _mira`). Juno, the default, has none.
    mode_suffix = "" if args.voice == DEFAULT_VOICE else f"_{args.voice}"
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # Timestamps the first sentence/clause piece (see synth_one_streaming).
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        piece_counts = []

        def _ttfa_gen(job, early_stop):
            audio, first, n_pieces = synth_one_streaming(job["text"], early_stop)
            piece_counts.append(n_pieces)
            return audio, SR, first

        run_probe(
            _ttfa_gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            # segments_per_utterance == 1 means one sentence in, nothing to stream.
            extra={"device": "cpu", "runtime": "numpy", "threads": args.threads, "voice": args.voice,
                   "temp": args.temp, "seed": args.seed, "revision": os.environ.get("WOOF_REVISION"),
                   "segments_per_utterance": piece_counts},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    # Serial synthesis: one sample per "batch"; warm-up runs untimed on the first pending samples.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)

    # ── Main loop: synthesize one sample at a time → write JSONL ─────────────
    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]

        # Store the path relative to model_dir (the manifest's dir); write to the full path.
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        path = os.path.join(model_dir, rel_path)

        audio, elapsed = synth_one(text)
        sf.write(path, audio, SR)

        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SR, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="ConwayResearch/Underdog-Woof",
                        help="HF model id (manifest/output naming; the image pins the repo revision).")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, choices=VOICES,
                        help="Built-in voice: juno (the package default) or mira (outputs suffixed _mira).")
    parser.add_argument("--temp", type=float, default=0.5, help="Sampling temperature (package default 0.5).")
    parser.add_argument("--threads", type=int, default=1,
                        help="BLAS thread cap via threadpoolctl. Default 1, as on the model card (submit_jobs.sh passes 8); "
                             "0 = leave the environment's setting (e.g. OMP_NUM_THREADS).")
    add_common_args(parser)
    parser.set_defaults(device="cpu")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    if args.threads > 0:
        with threadpool_limits(limits=args.threads):
            main(args)
    else:
        main(args)
