"""
TTS synthesis for the Open TTS Leaderboard (tiny-broca backend, stage 1).

tiny-broca (https://huggingface.co/brenorb/tiny-broca, code https://github.com/brenorb/tiny-broca)
is a 4,533,835-parameter Brazilian Portuguese TTS trained from scratch at 24 kHz with two voices
(F1 female, M1 male). It is the Brazilian Portuguese port of tiny-voca
(https://huggingface.co/n4ze3m/tiny-voca): the same compact acoustic model and source-filter
waveform generator (no neural vocoder), trained from scratch on Portuguese. CPU-only.

Inference is the ONNX graph shipped in the repo (``onnx/model.onnx``) with the repo's own front end
(``text.py``). The front end is character-level — Portuguese graphemes map to phones through regular
context rules the encoder learns — and spells digits, dates, money and identifiers in Brazilian
conventions (dot thousands separator, comma decimal) before tokenisation, so "R$ 1.234,56" and
"15/11/2026" reach the model as ordinary words.

Text handling mirrors the tiny-voca/clarvia convention: sentence split, each sentence tokenised and
synthesised as one ONNX call, long sentences cut at commas then between words (the model's 512-token
/ 32 s budget), and the audio joined peak-limited to 0.99. A text with nothing to tokenise is scored
as a tenth of a second of silence.

Each voice is its own leaderboard row:
  * ``F1``  default row (no manifest suffix).
  * ``M1``  variant row, manifest suffix ``_m1`` (publish with ``--variant m1``).

Device: CPU only. ``--threads`` caps onnxruntime intra-op threads (default 1: a fixed count keeps
RTFx independent of the host's core count, matching the tiny-voca/sanotts convention).
"""

import argparse
import importlib.util
import os
import re
import time

import numpy as np
import soundfile as sf
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, warm_up, wav_rel_path, write_entry,
)

SAMPLE_RATE = 24000
PEAK = 0.99
MAX_TOKENS = 512        # the model reads one utterance per call, up to 512 tokens / 32 s
TOKEN_MARGIN = 12       # leave room for start/end/boundary tokens, as the clarvia wrapper does

# Pinned model repo revision (brenorb/tiny-broca main on 2026-11-22).
MODEL_REVISION = "0ea879d8f532ad0c6e7308017b45105a74f81a95"

VOICES = {
    "F1": ("", 0),      # default row, voice index 0
    "M1": ("_m1", 1),   # variant row (publish with --variant m1), voice index 1
}
DEFAULT_VOICE = "F1"

# Sentence boundary: end punctuation followed by whitespace and an opening cap/digit (Portuguese caps
# included). Mirrors the clarvia wrapper's splitter.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[\"'(\[]?[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ0-9])")


def _load_text_module(repo_dir):
    """Import the repo's own ``text.py`` front end from the downloaded snapshot (unique module name
    so it never shadows anything on sys.path)."""
    path = os.path.join(repo_dir, "text.py")
    spec = importlib.util.spec_from_file_location("tiny_broca_text", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if text and not re.search(r"[.!?…,;:][\"'”’)\]»]*$", text):
        text += "."
    return [part for part in _SENTENCE_END.split(text) if re.search(r"\w", part)]


def main(args):
    if not args.device.startswith("cpu"):
        raise SystemExit("tiny-broca runs on CPU only; pass --device=cpu.")
    if args.voice not in VOICES:
        raise SystemExit(f"--voice must be one of {list(VOICES)}")
    mode_suffix, voice_index = VOICES[args.voice]

    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    if args.checkpoint_path:
        repo_dir = args.checkpoint_path
    else:
        repo_dir = snapshot_download(
            args.model_id, revision=args.revision,
            local_dir=os.path.join(args.local_dir, args.model_id.replace("/", "-")),
            allow_patterns=["onnx/**", "config.json", "text.py"],
        )
    print(f"Checkpoint: {repo_dir} (revision {args.revision})")

    text_mod = _load_text_module(repo_dir)
    phonemize = text_mod.phonemize

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        os.path.join(repo_dir, "onnx", "model.onnx"), options, providers=["CPUExecutionProvider"])
    print(f"Loaded tiny-broca voice={args.voice} | onnxruntime, {SAMPLE_RATE} Hz, threads={args.threads}.")

    def _run_chunk(ids, stress):
        audio, _ = session.run(None, {
            "tokens": np.asarray([ids], dtype=np.int64),
            "stress": np.asarray([stress], dtype=np.int64),
            "voice": np.asarray([voice_index], dtype=np.int64),
            "speed": np.asarray([1.0], dtype=np.float32),
            "pitch_shift": np.asarray([0.0], dtype=np.float32),
            "pitch_range": np.asarray([1.0], dtype=np.float32),
        })
        return np.asarray(audio).reshape(-1)

    failures = []

    def synth_one(text):
        """Return (audio float32, elapsed_s). Text with nothing to tokenise is scored as silence."""
        start = time.perf_counter()
        chunks = []
        pending = _sentences(text)
        try:
            while pending:
                sentence = pending.pop(0)
                try:
                    tokens = phonemize(sentence)
                except ValueError:   # empty / nothing to tokenise
                    continue
                if len(tokens.ids) > MAX_TOKENS - TOKEN_MARGIN:
                    # too long for one utterance: split at commas, else in half
                    parts = re.split(r"(?<=[,;:])\s+", sentence)
                    if len(parts) == 1:
                        words = sentence.split()
                        parts = ([" ".join(words[: len(words) // 2]), " ".join(words[len(words) // 2:])]
                                 if len(words) > 1 else [])
                    pending = parts + pending
                    continue
                chunks.append(_run_chunk(tokens.ids, tokens.stress))
        except (ValueError, RuntimeError) as exc:
            failures.append(str(exc))
            print(f"WARNING: refused ({exc}); writing silence for: {text[:80]!r}")
            chunks = []
        if not chunks:
            audio = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
        else:
            wave = np.concatenate(chunks)
            loudest = float(np.abs(wave).max())
            audio = (wave * (PEAK / loudest) if loudest > PEAK else wave).astype(np.float32)
        return audio, time.perf_counter() - start

    paths = output_paths(args, mode_suffix)
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
            out_dir=model_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={"device": args.device, "voice": args.voice, "threads": args.threads, "revision": args.revision},
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="brenorb/tiny-broca",
                        help="tiny-broca model repo (apache-2.0).")
    parser.add_argument("--revision", type=str, default=MODEL_REVISION,
                        help="Model repo revision (default: main on 2026-11-22).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local copy of the repo. Default: download --model_id under --local_dir.")
    parser.add_argument("--local_dir", type=str, default="/tmp/tiny_broca_models",
                        help="Where to materialise the model repo as real files.")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE,
                        help=f"One of {list(VOICES)}. {DEFAULT_VOICE} is the default row.")
    parser.add_argument("--threads", type=int, default=1,
                        help="onnxruntime intra-op threads (default 1 for RTFx independent of host cores).")
    add_common_args(parser)
    # ``--device`` comes from ``add_common_args``; override the default to cpu since tiny-broca is
    # CPU-only (``main`` refuses a GPU device).
    parser.set_defaults(seed=0, device="cpu")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} voice={args.voice} on "
          f"{args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
