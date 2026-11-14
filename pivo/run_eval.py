"""
TTS synthesis for the Open TTS Leaderboard (PiVo-1.4M backend, stage 1).

PiVo-1.4M (empero-ai/PiVo-1.4M) is a 1.4M-parameter English TTS (584,965-parameter acoustic model +
816,597-parameter FARGAN-derived vocoder) with two built-in voices, Pi (default) and Vo, 16 kHz
output. It runs on CPU only, with NumPy and ONNX Runtime (CPUExecutionProvider). The model repo is
not a pip package: the Docker image bakes it at a pinned revision and this script imports the
`pivo` package it ships (see the Space's Dockerfile). Run locally, it downloads the same revision.

Text frontend. PiVo looks every word up in the CMU dictionary and RAISES on a word it does not
know (11% of Seed-TTS en and 20% of CV3 en rows have one, mostly names), and on characters its
normalizer does not handle (accented letters, "°"). The model card asks callers to pass
pronunciations through `overrides`, so this backend does what a deployment would:
  * unknown words get an ARPAbet pronunciation from g2p_en (`G2p.predict`, the neural CMU-trained
    G2P; same phone set as PiVo), passed through `overrides`;
  * accented letters are reduced to ASCII (é -> e), "°" is read "degrees", and any other character
    PiVo rejects becomes a space.
Both run inside the timed call (RTFx and TTFA include them), and each manifest row records what
was needed (`frontend` key), so the results are "PiVo + g2p_en", stated in the leaderboard Note.

Device: CPU only. `--device` must be `cpu`; on the h200 flavor the model uses that host's CPU.
`--threads` sets ONNX Runtime's intra-op threads (the model card benchmarks 1).

There is no batched API: rows are processed serially. Synthesis is deterministic (no sampling).

Stage 2 (`transformers/transcribe.py`) fills `pred_text` with Qwen3-ASR, and stage 4
(`normalizer.eval_utils.score_results`) computes WER + RTFx (no stage 3 SIM: fixed voice).
"""

import argparse
import os
import re
import sys
import time
import unicodedata

import numpy as np
import soundfile as sf
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

# Pinned model revision (the public 1.0.0 release). The Docker image bakes the same one.
MODEL_REVISION = "515d84fc3420da6ae138f1b81017fcbc24d622a8"
DEFAULT_VOICE = "Pi"  # the package default
MODEL_PARAMS = 1_401_562  # model card: learned parameters, vocoder included

# Characters PiVo's own normalizer handles (pivo/text.py normalize): letters, digits, the punctuation
# it reads or drops, the currency/symbol words it expands, and the typographic quotes/dashes it folds.
_PIVO_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t\n'.,?!;:-()[]{}\"/$£€%&+’‘“”—–…")
_SPOKEN_SYMBOLS = {"°": " degrees "}


def load_pivo(model_dir, threads):
    """Import the `pivo` package shipped in the model repo and load it (hash-checked by PiVo)."""
    sys.path.insert(0, model_dir)
    from pivo import PiVo
    from pivo.text import SYMBOL_TO_ID, phonemize
    return PiVo(model_dir, threads=threads), phonemize, SYMBOL_TO_ID


class Frontend:
    """Make arbitrary eval text acceptable to PiVo's strict frontend (see the module docstring)."""

    def __init__(self, phonemize, symbol_to_id):
        from g2p_en import G2p  # neural G2P; G2p() loads NLTK's cmudict (baked into the image)

        self.phonemize, self.symbols = phonemize, symbol_to_id
        self.g2p = G2p()
        self.cache = {}

    def clean(self, text):
        """ASCII-fold accented letters, speak "°", blank anything else PiVo would reject."""
        out = []
        for char in text:
            if char in _PIVO_CHARS:
                out.append(char)
            elif char in _SPOKEN_SYMBOLS:
                out.append(_SPOKEN_SYMBOLS[char])
            else:
                folded = "".join(c for c in unicodedata.normalize("NFKD", char) if not unicodedata.combining(c))
                out.append(folded if folded and all(c in _PIVO_CHARS for c in folded) else " ")
        return "".join(out)

    def pronounce(self, word):
        """ARPAbet for an out-of-dictionary word, from g2p_en's neural model (cached)."""
        if word not in self.cache:
            letters = re.sub(r"[^a-z]", "", word.lower())  # g2p_en's alphabet is a-z only
            phones = [p for p in self.g2p.predict(letters) if p in self.symbols] if letters else []
            self.cache[word] = phones
        return self.cache[word]

    def __call__(self, text):
        """(text for PiVo, overrides for PiVo, {what was needed} for the manifest)."""
        cleaned = self.clean(text)
        unknown = sorted(set(self.phonemize(cleaned, strict=False).unknown_words))
        overrides = {w: self.pronounce(w) for w in unknown}
        overrides = {w: p for w, p in overrides.items() if p}
        info = {}
        if cleaned != text:
            info["cleaned_text"] = cleaned
        if overrides:
            info["g2p_en"] = {w: " ".join(p) for w, p in overrides.items()}
        return cleaned, overrides or None, info


def main(args):
    if args.device != "cpu":
        # PiVo has no GPU path (CPUExecutionProvider only). Refuse rather than label a CPU
        # measurement with a GPU device.
        raise ValueError(f"PiVo runs on CPU only; got --device={args.device} (use cpu).")
    set_seed(args.seed)

    model_dir = args.model_dir or os.environ.get("PIVO_MODEL_DIR")
    if not model_dir:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(args.model_id, revision=MODEL_REVISION)
    # 0 = the job's CPU quota (submit_ttfa_jobs.sh exports it as OMP_NUM_THREADS).
    threads = args.threads or int(os.environ.get("OMP_NUM_THREADS") or 1)
    model, phonemize, symbol_to_id = load_pivo(model_dir, threads)
    frontend = Frontend(phonemize, symbol_to_id)
    sample_rate = model.sample_rate
    print(f"Loaded PiVo ({args.model_id}@{MODEL_REVISION[:7]} from {model_dir}, voice={args.voice}, "
          f"threads={threads}, {sample_rate} Hz); TTS model size: {MODEL_PARAMS / 1e9:.4f}B parameters")

    def synth_one(text):
        """Frontend + synthesis of one text, timed together. Return (audio, elapsed_s, frontend info)."""
        start = time.perf_counter()
        cleaned, overrides, info = frontend(text)
        audio = model.synthesize(cleaned, voice=args.voice, speed=args.speed, overrides=overrides)
        elapsed = time.perf_counter() - start
        if audio.size == 0:
            audio = np.zeros(1, dtype=np.float32)
        return audio, elapsed, info

    def synth_one_streaming(text, early_stop=False):
        """TTFA probe only: the same frontend, then PiVo's own `iter_audio`, timestamping its first
        10 ms frame. PiVo encodes the whole text before the first frame, so that cost is included.
        Returns (audio_1d, first_frame_ts)."""
        cleaned, overrides, _ = frontend(text)
        first, chunks = None, []
        for chunk in model.iter_audio(cleaned, voice=args.voice, speed=args.speed, overrides=overrides):
            chunks.append(chunk)
            if first is None:
                first = time.perf_counter()
                if early_stop:
                    break
        return (np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)), first

    # Layout: results/<model_safe>/{MODEL_<safe>_DATASET_<dsid>.jsonl, <dsid>/output_<id>.wav}.
    # The non-default voice gets its own manifest/sidecar suffix (`_vo`; publish with
    # `open_results_pr.py --variant _vo`). Pi, the default, has none.
    mode_suffix = "" if args.voice == DEFAULT_VOICE else f"_{args.voice.lower()}"
    paths = output_paths(args, mode_suffix)
    model_dir_out, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices

        def _ttfa_gen(job, early_stop):
            audio, first = synth_one_streaming(job["text"], early_stop)
            return audio, sample_rate, first

        run_probe(
            _ttfa_gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir_out, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={"device": "cpu", "runtime": "onnxruntime", "threads": threads, "voice": args.voice,
                   "revision": MODEL_REVISION, "frontend": "pivo + g2p_en fallback"},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    # Serial synthesis: one sample per "batch"; warm-up runs untimed on the first pending samples.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)

    # ── Main loop: synthesize one sample at a time → write JSONL ─────────────
    n_cleaned = n_g2p = 0
    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]

        # Store the path relative to model_dir (the manifest's dir); write to the full path.
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        path = os.path.join(model_dir_out, rel_path)

        audio, elapsed, info = synth_one(text)
        sf.write(path, audio, sample_rate)

        entry = manifest_entry(rel_path, len(audio) / sample_rate, elapsed, text)
        if info:
            entry["frontend"] = info
        n_cleaned += "cleaned_text" in info
        n_g2p += "g2p_en" in info
        write_entry(manifest_file, entry)
        manifest_file.flush()

    manifest_file.close()
    print(f"Frontend fallbacks over {len(starts)} generated rows: {n_g2p} with g2p_en pronunciations, "
          f"{n_cleaned} with characters folded/dropped (details in each row's `frontend` key).")
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="empero-ai/PiVo-1.4M",
                        help="HF model id (manifest/output naming; the revision is pinned in this script).")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Local snapshot of the model repo (default: $PIVO_MODEL_DIR, else download the pinned revision).")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, choices=["Pi", "Vo"],
                        help="Built-in voice: Pi (the package default) or Vo (outputs suffixed _vo).")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking rate, 0.5-2.0 (package default 1.0).")
    parser.add_argument("--threads", type=int, default=1,
                        help="ONNX Runtime intra-op threads. Default 1, as on the model card; "
                             "0 = $OMP_NUM_THREADS (the CPU quota submit_ttfa_jobs.sh exports).")
    add_common_args(parser)
    parser.set_defaults(device="cpu")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
