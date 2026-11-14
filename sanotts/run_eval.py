"""
TTS synthesis for the Open TTS Leaderboard (sanoTTS backend, stage 1).

sanoTTS (https://huggingface.co/ampixa/sanoTTS, code https://github.com/Ampixa/sanoTTS) is a family of
small distilled TTS voices that run with NumPy only (no torch, no ONNX Runtime) through the `sanotts`
pip package. Three English voices are evaluated, one leaderboard row each:

  * heart       2,272,145 params, 24 kHz. "nano" graph (mel-100 -> ConvNeXt1D -> iSTFT), fp32,
                distilled from Kokoro-82M af_heart. The default row (no suffix).
  * amy         1,454,284 params, 22.05 kHz. "piperlite" graph (duration + token-context acoustic +
                time-domain decoder), fp16 weights, distilled from Piper en_US-amy. Suffix `_amy`.
  * heart-nano    294,279 params, 24 kHz. A separate, much smaller student of the same nano
                architecture (lineage en_us_e13b vs heart's en_us_r227f32), stored in int8.
                Suffix `_heartnano`.

Weights are read from the model repo pinned to the commit main pointed to on 2026-11-07 (the package's
own default fetch reads the moving `main`; the package checks every blob against the sha256 in the
voice's metadata).
The G2P is the package's default, espeak-free English front end for each voice (a lexicon plus an
out-of-vocabulary fallback model), so no GPL espeak-ng is involved.

Text handling. Every text is first synthesized WHOLE, exactly as `Synthesizer.synthesize` takes it.
The nano voices (heart, heart-nano) refuse inputs longer than their duration model's 207-token window
(`FrontendError(kind="too_long")`), and the package has no long-text loop of its own, so ONLY on that
error this backend (not the package) falls back: it splits the text into sentences (the regex of the
project's streaming integration, custom_components/sanotts/tts.py; fragments with no word character
stay attached to the preceding sentence), and any sentence still too long is cut at clause
punctuation (, ; :) in its middle third, else at the space nearest its middle, recursively. Pieces are
concatenated with no added silence. amy has no window and is never split. The fallback runs inside the
timed call (RTFx and TTFA include the failed whole-text attempt), and each manifest row that needed it
records `frontend = {"sentences": k, "pieces": n}` (n > k means mid-sentence cuts).

Device: CPU only (NumPy). `--device` must be `cpu`; on the h200 flavor the voice uses that host's
CPU. `--threads` caps NumPy's BLAS pool with threadpoolctl (default 1: the matrices are tiny, and one
thread keeps RTFx independent of the host's core count); the value is recorded in the TTFA sidecar.

There is no batched API: rows are processed serially. Synthesis is deterministic (the nano decoder's
noise excitation is seeded inside the package).

Streaming (TTFA probe only). `Synthesizer.synthesize_stream` (sanotts >= 0.5.1) runs the utterance-global
steps over the whole text (G2P, durations, the acoustic token stage, the decoder noise draw), then the
local stages chunk by chunk (for heart / heart-nano the acoustic frame stage and the decoder, for amy the
decoder) on windows widened by their receptive field, so the chunks concatenate to `synthesize()`'s audio
(to float32 rounding; the package tests this). The first chunk is 8 frames, about 64 ms of audio from
heart / heart-nano and 93 ms from amy; the pre-chunk work and the failed whole-text attempt of a too-long
text are inside the timed call. The eval loop below keeps calling `synthesize()`, so the scored audio is
the non-streamed rendering.

Stage 2 (`transformers/transcribe.py`) fills `pred_text` with Qwen3-ASR, and stage 4
(`normalizer.eval_utils.score_results`) computes WER + RTFx (no stage 3 SIM: fixed voice).
"""

import argparse
import itertools
import os
import re
import time

import numpy as np
import soundfile as sf
from threadpoolctl import threadpool_info, threadpool_limits
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

# Pinned model repo revision (ampixa/sanoTTS main on 2026-11-07). Every voice below is read from it.
MODEL_REVISION = "4a8194aa1dd958033ee449b3ad30c37e536b216e"

# voice -> (folder in the model repo, learned parameters, manifest suffix). heart is the default row.
VOICES = {
    "heart": ("heart", 2_272_145, ""),
    "amy": ("amy-en-1p46m", 1_454_284, "_amy"),
    "heart-nano": ("heartnano", 294_279, "_heartnano"),
}
DEFAULT_VOICE = "heart"

# Sentence splitter of the project's Home Assistant integration (custom_components/sanotts/tts.py).
_SENTENCE_RE = re.compile("[^.!?…。！？\n]*"
                          "[.!?…。！？\n]+"
                          "[\"'”’)\\]]*\\s*")
_CLAUSE_RE = re.compile(r"[,;:]\s+")
_SPEAKABLE_RE = re.compile(r"\w")
# A "sentence" ending in one of these is not a sentence end: merge it with the next piece.
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "no", "inc", "ltd", "co",
                  "mt", "ft", "gen", "gov", "sen", "rep", "fig", "approx", "dept", "est", "vol", "jan", "feb",
                  "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}


def _is_false_break(piece, following):
    """True when the period ending `piece` belongs to an abbreviation, initial or number, not a sentence end."""
    if not piece.rstrip().endswith("."):
        return False
    last = piece.rstrip()[:-1].split()[-1] if piece.rstrip()[:-1].split() else ""
    last = last.strip("\"'(\u201c\u2018").split(".")[-1]
    nxt = following.lstrip()[:1]
    return (len(last) == 1 and last.isalpha()          # initials, "U." of "U.S."
            or last.lower() in _ABBREVIATIONS          # "Dr.", "Jan."
            or nxt.isdigit() or nxt.islower())          # "4." + "2", "a." + "m."


def split_sentences(text):
    """Sentences (with their trailing punctuation), then any unterminated remainder.

    A fragment with no word character (a lone quote, "...", a symbol) is appended to the previous
    sentence instead of becoming a piece of its own, which the front end would reject as empty. A
    period after an initial, a common abbreviation, or before a digit / lowercase letter ("U.S.",
    "Dr.", "4.2", "a.m.") is not treated as a sentence end. This heuristic also merges some real
    sentence ends ("... the U.S. The ..."); that only makes a piece longer before halving.
    """
    raw, end = [], 0
    for match in _SENTENCE_RE.finditer(text):
        raw.append(match.group(0))
        end = match.end()
    raw.append(text[end:])
    sentences = []
    for part in raw:
        if not part.strip():
            continue
        if sentences and (not _SPEAKABLE_RE.search(part) or _is_false_break(sentences[-1], part)):
            sentences[-1] += part
        else:
            sentences.append(part)
    return [s.strip() for s in sentences]


def halve(text):
    """Split a too-long sentence in two: at a clause break in its middle third, else the space nearest the middle.

    Only cut points that leave a word character on both sides are used, so neither half is unspeakable.
    """
    n = len(text)
    middle = n / 2

    def usable(i):
        return _SPEAKABLE_RE.search(text[:i]) and _SPEAKABLE_RE.search(text[i:])

    breaks = [m.end() for m in _CLAUSE_RE.finditer(text) if n / 3 <= m.end() <= 2 * n / 3 and usable(m.end())]
    if not breaks:
        breaks = [m.start() for m in re.finditer(r"\s+", text) if usable(m.start())]
    if not breaks:
        raise ValueError(f"cannot split an unbreakable token that exceeds the voice's window: {text[:80]!r}")
    cut = min(breaks, key=lambda i: abs(i - middle))
    left, right = text[:cut].strip(), text[cut:].strip()
    if not left or not right:
        raise ValueError(f"split produced an empty half for: {text[:80]!r}")
    return left, right


class Voice:
    """A loaded sanoTTS voice plus the sentence loop (see the module docstring)."""

    def __init__(self, model_id, voice, revision):
        from huggingface_hub import snapshot_download
        from sanotts import Synthesizer
        from sanotts.nano_frontend import FrontendError as NanoFrontendError

        folder, self.parameters, self.suffix = VOICES[voice]
        repo_dir = snapshot_download(model_id, revision=revision, allow_patterns=[f"{folder}/*"])
        self.voice_dir = os.path.join(repo_dir, folder)
        self.synth = Synthesizer(voice_dir=self.voice_dir)
        self.sample_rate = self.synth.sample_rate
        self._too_long = NanoFrontendError

    def _try(self, text):
        """Audio for `text`, or None when it exceeds the voice's window (any other error propagates)."""
        try:
            return self.synth.synthesize(text).audio
        except self._too_long as exc:
            if getattr(exc, "kind", None) != "too_long":
                raise
            return None

    def _render(self, text):
        """Yield audio pieces for one sentence, halving it until each piece fits the window."""
        audio = self._try(text)
        if audio is not None:
            yield audio
            return
        left, right = halve(text)
        yield from self._render(left)
        yield from self._render(right)

    def _try_stream(self, text):
        """A chunk iterator for `text`, or None when it exceeds the voice's window.

        The window check is in the front end, which runs before the first chunk, so pulling the first
        chunk is what raises; a too_long after audio has been produced would be a package bug.
        """
        chunks = self.synth.synthesize_stream(text)
        try:
            first = next(chunks)
        except StopIteration as exc:
            raise RuntimeError(f"sanotts produced no audio for {text[:80]!r}") from exc
        except self._too_long as exc:
            if getattr(exc, "kind", None) != "too_long":
                raise
            return None
        return itertools.chain([first], chunks)

    def _render_stream(self, text):
        """`_render`, streaming: chunks of one sentence, halving it until each piece fits the window."""
        chunks = self._try_stream(text)
        if chunks is not None:
            yield from chunks
            return
        left, right = halve(text)
        yield from self._render_stream(left)
        yield from self._render_stream(right)

    def stream(self, text):
        """`pieces`, streaming: the same whole-text-first fallback, yielding decoder chunks as they come."""
        chunks = self._try_stream(text)
        if chunks is not None:
            yield from chunks
            return
        for sentence in split_sentences(text):
            yield from self._render_stream(sentence)

    def pieces(self, text, stats=None):
        """Yield audio pieces in order. The whole text first; sentences/halves only if it is too long.

        `stats`, when given, receives {"sentences": k, "pieces": n} for a text that needed the fallback.
        """
        audio = self._try(text)
        if audio is not None:
            yield audio
            return
        sentences = split_sentences(text)
        n = 0
        for sentence in sentences:
            for piece in self._render(sentence):
                n += 1
                yield piece
        if stats is not None:
            stats.update(sentences=len(sentences), pieces=n)


def main(args):
    if args.device != "cpu":
        # sanotts is NumPy-only; refuse rather than label a CPU measurement with a GPU device.
        raise ValueError(f"sanoTTS runs on CPU only (NumPy); got --device={args.device} (use cpu).")
    set_seed(args.seed)
    if args.threads < 1:
        raise ValueError(f"--threads must be >= 1, got {args.threads}")
    # Process-wide cap on NumPy's BLAS pool (kept for the whole run; see the module docstring). Verified:
    # a platform where threadpoolctl sees no BLAS pool would otherwise run uncapped under a "capped" label.
    _blas_limit = threadpool_limits(limits=args.threads)  # noqa: F841 (held for the whole run)
    blas_pools = [{k: pool.get(k) for k in ("internal_api", "num_threads")}
                  for pool in threadpool_info() if pool.get("user_api") == "blas"]
    if not blas_pools or any(pool["num_threads"] != args.threads for pool in blas_pools):
        raise RuntimeError(f"could not cap NumPy's BLAS pool at {args.threads} thread(s): {blas_pools}")
    print(f"BLAS pools: {blas_pools}")

    import sanotts
    sanotts_version = sanotts.__version__
    voice = Voice(args.model_id, args.voice, args.revision)
    sample_rate = voice.sample_rate
    print(f"Loaded sanoTTS ({args.model_id}@{args.revision[:7]}, voice={args.voice} from {voice.voice_dir}, "
          f"{sample_rate} Hz, BLAS threads={args.threads}, sanotts {sanotts_version}); "
          f"TTS model size: {voice.parameters / 1e9:.6f}B parameters")

    def synth_one(text):
        """Synthesis of one text (with the too_long fallback), timed. Return (audio, elapsed_s, stats)."""
        stats = {}
        start = time.perf_counter()
        parts = list(voice.pieces(text, stats))
        elapsed = time.perf_counter() - start
        audio = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
        return audio, elapsed, stats

    def synth_one_streaming(text, early_stop=False):
        """TTFA probe only: `Synthesizer.synthesize_stream` through the same fallback, timestamping the
        first chunk. The utterance-global steps run over the whole text before that chunk, so their cost
        is included. Returns (audio_1d, first_chunk_ts, n_chunks)."""
        first, parts = None, []
        for part in voice.stream(text):
            parts.append(part)
            if first is None:
                first = time.perf_counter()
                if early_stop:
                    break
        if not parts:
            raise RuntimeError(f"no audio for {text[:80]!r}")
        return np.concatenate(parts), first, len(parts)

    # Layout: results/<model_safe>/{MODEL_<safe>_DATASET_<dsid><suffix>.jsonl, <dsid><suffix>/output_<id>.wav}.
    # heart (the default) has no suffix; amy and heart-nano are variants (`open_results_pr.py --variant`).
    paths = output_paths(args, voice.suffix)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        chunk_counts = []

        def _ttfa_gen(job, early_stop):
            audio, first, n_chunks = synth_one_streaming(job["text"], early_stop)
            chunk_counts.append(n_chunks)
            return audio, sample_rate, first

        run_probe(
            _ttfa_gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, mode_suffix=voice.suffix, n=args.ttfa_probe,
            extra={"device": "cpu", "runtime": "numpy (sanotts)", "voice": args.voice,
                   "revision": args.revision, "threads": args.threads,
                   "blas": blas_pools, "sanotts": sanotts_version,
                   "streaming_api": "Synthesizer.synthesize_stream",
                   # chunks yielded per row; with early stop, 1 (only the first chunk is pulled)
                   "chunks_per_utterance": chunk_counts},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    # Serial synthesis: one sample per "batch"; warm-up runs untimed on the first pending samples.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)
    set_seed(args.seed)

    # ── Main loop: synthesize one sample at a time → write JSONL ─────────────
    n_split = n_cut = 0
    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]

        # Store the path relative to model_dir (the manifest's dir); write to the full path.
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        path = os.path.join(model_dir, rel_path)

        audio, elapsed, stats = synth_one(text)
        sf.write(path, audio, sample_rate)

        entry = manifest_entry(rel_path, len(audio) / sample_rate, elapsed, text)
        if stats:
            entry["frontend"] = stats
            n_split += 1
            n_cut += stats["pieces"] > stats["sentences"]
        write_entry(manifest_file, entry)
        manifest_file.flush()

    manifest_file.close()
    print(f"Too-long fallback over {len(starts)} generated rows: {n_split} split into sentences, {n_cut} of "
          f"them also cut mid-sentence (details in each row's `frontend` key).")
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="ampixa/sanoTTS",
                        help="HF model repo holding every voice (manifest/output naming).")
    parser.add_argument("--revision", type=str, default=MODEL_REVISION,
                        help="Model repo revision (default: the pinned commit above).")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, choices=sorted(VOICES),
                        help="heart (default row), amy (suffix _amy) or heart-nano (suffix _heartnano).")
    parser.add_argument("--threads", type=int, default=1,
                        help="NumPy BLAS threads (threadpoolctl). Default 1, also used by the TTFA target.")
    add_common_args(parser)
    parser.set_defaults(device="cpu")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} ({args.voice}) on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
