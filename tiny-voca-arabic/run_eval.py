"""
TTS synthesis for the Open TTS Leaderboard (tiny-voca-arabic backend, stage 1).

tiny-voca-arabic (https://huggingface.co/oddadmix/tiny-voca-arabic) is a 4,531,723-parameter Modern
Standard Arabic (فصحى) TTS trained from scratch at 24 kHz with one synthetic voice (``a000``). It is
the Arabic port of tiny-voca (https://huggingface.co/n4ze3m/tiny-voca): the same compact acoustic
model and source-filter waveform generator (no neural vocoder), trained from scratch on Arabic.
CPU-only. Licence apache-2.0.

Inference is the ONNX graph shipped in the repo (``onnx/model.onnx``). Arabic is written without short
vowels, so the front end has three stages:

  1. ``expand_numbers`` (repo ``code/numbers.py``) reads digits as MSA words.
  2. The CATT diacritiser (``catt_tashkeel``, apache-2.0, encoder-decoder ONNX from the abjadai/catt v2
     GitHub release, downloaded at first use) restores the short vowels.
  3. A rule-based MSA converter (repo ``code/arabic_frontend.py``) turns the fully-vowelled text into
     49 phoneme symbols + a stress feature (sun-letter assimilation, hamzat al-wasl, taʾ marbuta,
     pausal forms, emphatic backing, etc.), exposed as ``tiny_voca_ar.text.phonemize``.

The front-end package (``tiny_voca_ar``) comes from the model repo's ``code/`` directory; ``catt_tashkeel``
is vendored from the model's demo Space (it is not in the model repo). Both are pinned.

Text handling mirrors the demo Space and the tiny-voca/broca convention: sentence split, each sentence
diacritised + tokenised and synthesised as one ONNX call, long sentences cut at Arabic/Latin clause
punctuation then between words (the model's 512-token / 32 s budget), and the audio joined peak-limited
to 0.99. A text with nothing to pronounce is scored as a tenth of a second of silence.

One synthetic voice, so one leaderboard row (no manifest suffix).

Device: CPU only. ``--threads`` caps onnxruntime intra-op threads for both the TTS model and the CATT
diacritiser (default 1: a fixed count keeps RTFx independent of the host's core count, matching the
tiny-voca/broca convention). RTFx includes diacritisation, as it is part of generating the audio.
"""

import argparse
import importlib.util
import os
import re
import sys
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
TOKEN_MARGIN = 12       # leave room for start/end/boundary tokens, as the demo Space does
VOICE_INDEX = 0         # the only voice, a000

# Pinned revisions (oddadmix/tiny-voca-arabic main on 2026-11-22). The model weights + front-end code
# come from the model repo; catt_tashkeel comes from the demo Space (not in the model repo).
MODEL_REVISION = "e529f716965142f51835adc9e5661f224c3fa103"
SPACE_REVISION = "64e3e2ae08f7553c38f280a58f05b955a0c296f4"

# Sentence boundary: end punctuation (Arabic + Latin) followed by whitespace. Mirrors the demo Space.
_SENTENCE_END = re.compile(r"(?<=[.!?؟…])\s+(?=\S)")
_CLAUSE_END = re.compile(r"(?<=[،,;؛:])\s+")


def _assemble_frontend(args):
    """Materialise the two front-end packages under --local_dir and put them on sys.path.

    ``tiny_voca_ar`` (expand_numbers + phonemize) is the model repo's ``code/`` directory; ``catt_tashkeel``
    (the diacritiser) is vendored from the demo Space. Returns the directory added to sys.path.
    """
    from huggingface_hub import hf_hub_download
    pkg_root = os.path.join(args.local_dir, "frontend")
    ar_dir = os.path.join(pkg_root, "tiny_voca_ar")
    catt_dir = os.path.join(pkg_root, "catt_tashkeel")
    os.makedirs(ar_dir, exist_ok=True)
    os.makedirs(catt_dir, exist_ok=True)
    open(os.path.join(ar_dir, "__init__.py"), "a").close()
    for name in ("arabic_frontend.py", "numbers.py", "text.py"):
        src = hf_hub_download(args.model_id, f"code/{name}", revision=args.revision)
        with open(os.path.join(ar_dir, name), "w", encoding="utf-8") as out:
            out.write(open(src, encoding="utf-8").read())
    for name in ("__init__.py", "bw2ar.py", "models.py", "tokenizer.py", "utils.py"):
        src = hf_hub_download(args.space_id, f"catt_tashkeel/{name}", revision=args.space_revision,
                              repo_type="space")
        with open(os.path.join(catt_dir, name), "w", encoding="utf-8") as out:
            out.write(open(src, encoding="utf-8").read())
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    return pkg_root


def main(args):
    if not args.device.startswith("cpu"):
        raise SystemExit("tiny-voca-arabic runs on CPU only; pass --device=cpu.")
    # Keep the CATT diacritiser single-threaded too, so wall-clock (and thus RTFx) is reproducible.
    os.environ.setdefault("CATT_THREADS", str(args.threads))
    os.environ.setdefault("THREADS", str(args.threads))

    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    onnx_path = hf_hub_download(args.model_id, "onnx/model.onnx", revision=args.revision)
    hf_hub_download(args.model_id, "onnx/model.json", revision=args.revision)  # fetched for completeness
    print(f"Checkpoint: {onnx_path} (revision {args.revision})")

    _assemble_frontend(args)
    from tiny_voca_ar.numbers import expand_numbers
    from tiny_voca_ar.text import NothingToPronounce, phonemize
    from tiny_voca_ar import arabic_frontend

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, options, providers=["CPUExecutionProvider"])
    # Load the diacritiser now (downloads its ONNX files from the abjadai/catt v2 release on first use),
    # so the download does not count against the first synthesis / warm-up timing.
    arabic_frontend.diacritize(["مرحبا"])
    print(f"Loaded tiny-voca-arabic | onnxruntime + CATT diacritiser, {SAMPLE_RATE} Hz, "
          f"threads={args.threads}.")

    def _run_chunk(ids, stress):
        audio, _ = session.run(None, {
            "tokens": np.asarray([ids], dtype=np.int64),
            "stress": np.asarray([stress], dtype=np.int64),
            "voice": np.asarray([VOICE_INDEX], dtype=np.int64),
            "speed": np.asarray([1.0], dtype=np.float32),
            "pitch_shift": np.asarray([0.0], dtype=np.float32),
            "pitch_range": np.asarray([1.0], dtype=np.float32),
        })
        return np.asarray(audio).reshape(-1)

    failures = []

    def _sentences(text):
        text = re.sub(r"\s+", " ", (text or "").strip())
        if not text:
            return []
        if text[-1] not in ".!?؟…":
            text += "."
        return [part for part in _SENTENCE_END.split(expand_numbers(text)) if re.search(r"\w", part)]

    def synth_one(text):
        """Return (audio float32, elapsed_s). Text with nothing to pronounce is scored as silence."""
        start = time.perf_counter()
        chunks = []
        pending = _sentences(text)
        try:
            while pending:
                sentence = pending.pop(0)
                try:
                    tokens = phonemize(sentence)
                except (ValueError, NothingToPronounce):   # empty / no Arabic to say
                    continue
                if len(tokens.ids) > MAX_TOKENS - TOKEN_MARGIN:
                    # too long for one utterance: split at clause punctuation, else in half
                    parts = [p for p in _CLAUSE_END.split(sentence) if p.strip()]
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

    paths = output_paths(args, "")
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
            out_dir=model_dir, mode_suffix="", n=args.ttfa_probe,
            extra={"device": args.device, "voice": "a000", "threads": args.threads, "revision": args.revision},
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
    parser.add_argument("--model_id", type=str, default="oddadmix/tiny-voca-arabic",
                        help="tiny-voca-arabic model repo (apache-2.0).")
    parser.add_argument("--revision", type=str, default=MODEL_REVISION,
                        help="Model repo revision (default: main on 2026-11-22).")
    parser.add_argument("--space_id", type=str, default="oddadmix/tiny-voca-arabic",
                        help="Demo Space that ships catt_tashkeel (not in the model repo).")
    parser.add_argument("--space_revision", type=str, default=SPACE_REVISION,
                        help="Space revision for catt_tashkeel (default: main on 2026-11-22).")
    parser.add_argument("--local_dir", type=str, default="/tmp/tiny_voca_arabic",
                        help="Where to materialise the front-end packages.")
    parser.add_argument("--threads", type=int, default=1,
                        help="onnxruntime intra-op threads (default 1 for RTFx independent of host cores).")
    add_common_args(parser)
    # ``--device`` comes from ``add_common_args``; override the default to cpu since tiny-voca-arabic is
    # CPU-only (``main`` refuses a GPU device).
    parser.set_defaults(seed=0, device="cpu")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
