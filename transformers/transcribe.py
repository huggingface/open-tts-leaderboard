"""
ASR transcription for the Open TTS Leaderboard (stage 2).

Fills `pred_text` in a stage-1 manifest with Qwen3-ASR, in place and resumably. The ASR call
mirrors `transcribe()` in `transformers/run_eval.py`.

ZTTS2-Eval Mini panel manifests use `--asr_profile ztts2` instead: ZTTS2-Eval's ASR, i.e. Whisper-large-v3 or
Qwen3-ASR routed per language as ZTTS2-Eval routes them, at pinned revisions, with its repetition-loop guard. Each
row it transcribes is stamped with `asr_profile` / `asr_model`, which normalizer/ztts2.py requires.
"""

import argparse
import json
import os
import zlib

import torch
from transformers.audio_utils import load_audio

from transformers import AutoModelForMultimodalLM, AutoProcessor, WhisperForConditionalGeneration

DEFAULT_ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"

# ── ZTTS2-Eval profile: ZTTS2-Eval's ASR (utils/benchmark.py, scripts/score/run_wer.py) ───────────
ZTTS2_LANGS = {"en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru", "pt", "ar", "hi", "tr", "pl", "id", "th",
               "fil", "tl", "nl", "vi", "cs", "sv", "da", "fi", "el", "hu", "ro", "uk", "nb", "he", "bg", "hr",
               "sk", "ca", "ms", "fa"}
# Divergence from ZTTS2-Eval's own routing: we route to Qwen3-ASR every ZTTS2 language that already
# has Qwen3-ASR coverage on this leaderboard's other datasets (Seed-TTS / CV3 / MiniMax / Neyshekar),
# so the same ASR scores a given language wherever it appears. The remaining ZTTS2 languages go to
# Whisper. Keeping Qwen3 on exactly the languages our other panels use avoids silently mixing two
# ASR systems' per-language accuracy, at the cost of diverging from ZTTS2-Eval's published panels.
ZTTS2_QWEN_LANGS = {"en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru", "pt", "ar", "hi", "tr", "pl",
                    "id", "th", "nl", "vi", "cs", "fi", "el", "ro", "fa"}
# Pass the language hint for every ZTTS2 Qwen3-routed language — matches how the leaderboard
# profile invokes Qwen3-ASR for CV3 / MiniMax / Neyshekar (`language = args.asr_language or None`),
# so the same (language, ASR, hint) triple scores a given language wherever it appears here.
ZTTS2_QWEN_HINT_LANGS = ZTTS2_QWEN_LANGS
ZTTS2_QWEN_MODEL = ("Qwen/Qwen3-ASR-1.7B-hf", "bcd2b5b7f32b480ab5790554cfa8347f246a14f3")
ZTTS2_WHISPER_MODEL = ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1")
ZTTS2_WHISPER_CODE = {"fil": "tl", "nb": "no"}                    # else the code itself
# Whisper's 448-token decoder window minus its 4 prompt tokens (ZTTS2's vLLM greedy pass is uncapped).
ZTTS2_MAX_NEW_TOKENS = 444
# Repetition-loop guard: a transcript of >= 60 UTF-8 bytes compressing > 2.4x is re-decoded with this.
ZTTS2_LOOP_MIN_BYTES = 60
ZTTS2_LOOP_RATIO = 2.4
ZTTS2_RETRY_KWARGS = {"repetition_penalty": 1.2, "max_new_tokens": 256}


def _degenerate(text):
    """A repetition loop: the transcript compresses far more than natural text (run_wer.py `_degenerate`)."""
    b = text.encode("utf-8")
    return len(b) >= ZTTS2_LOOP_MIN_BYTES and len(b) / max(1, len(zlib.compress(b))) > ZTTS2_LOOP_RATIO


def main(args):
    torch_dtype = getattr(torch, args.dtype)
    ztts2 = args.asr_profile == "ztts2"
    if ztts2 and args.asr_language not in ZTTS2_LANGS:
        raise ValueError(f"--asr_profile ztts2 needs a ZTTS2-Eval language code, got {args.asr_language!r}")

    # ── Read manifest ────────────────────────────────────────────────────────
    with open(args.manifest_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if not entries:
        raise ValueError(f"No entries found in manifest {args.manifest_path}")
    print(f"Transcribing {len(entries)} samples from {args.manifest_path}")

    # Wav paths are relative to the manifest's directory; absolute or unresolvable paths are used as-is.
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest_path))

    def _resolve(p):
        if not p or os.path.isabs(p):
            return p
        cand = os.path.join(manifest_dir, p)
        return cand if os.path.exists(cand) else p

    # ── Resume: decide what's left BEFORE paying for the ASR model ───────────
    # Only transcribe rows without a prediction (unless --overwrite), or, under the ztts2 profile, rows
    # transcribed by another one. Checked before the model load so re-running a finished split is ~free.
    todo = list(range(len(entries)))
    if not args.overwrite:
        todo = [i for i in todo
                if not entries[i].get("pred_text") or (ztts2 and entries[i].get("asr_profile") != "ztts2")]
    if len(todo) < len(entries):
        print(f"Resuming: {len(entries) - len(todo)} already transcribed, {len(todo)} remaining.")
    if not todo:
        print("Nothing to transcribe; manifest already complete:", os.path.abspath(args.manifest_path))
        return

    # ── ASR model ─────────────────────────────────────────────────────────────
    whisper = ztts2 and args.asr_language not in ZTTS2_QWEN_LANGS
    backend = "whisper" if whisper else "qwen"
    if whisper:
        model_id, revision = ZTTS2_WHISPER_MODEL
    elif ztts2:
        model_id, revision = ZTTS2_QWEN_MODEL
    else:
        model_id, revision = args.asr_model_id, None
    asr_processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    asr_model = (WhisperForConditionalGeneration if whisper else AutoModelForMultimodalLM).from_pretrained(
        model_id, revision=revision, dtype=torch_dtype, device_map=args.device
    )
    asr_model.eval()
    asr_sr = 16_000
    if getattr(asr_processor, "feature_extractor", None) is not None and hasattr(
        asr_processor.feature_extractor, "sampling_rate"
    ):
        asr_sr = asr_processor.feature_extractor.sampling_rate

    if whisper:
        language = ZTTS2_WHISPER_CODE.get(args.asr_language, args.asr_language)
    elif ztts2:
        language = args.asr_language if args.asr_language in ZTTS2_QWEN_HINT_LANGS else None
    else:
        language = args.asr_language or None  # '' => None => Qwen3-ASR auto-detects
    max_new_tokens = ZTTS2_MAX_NEW_TOKENS if ztts2 else args.asr_max_new_tokens
    print(f"ASR: {model_id}@{revision or 'main'} ({backend}), language={language or 'auto'}")

    # Cap clip length: a runaway generation padded into a batch can OOM the ASR encoder, and
    # anything that long is degenerate and scores as an error anyway.
    max_samples = int(args.max_audio_seconds * asr_sr) if args.max_audio_seconds > 0 else None

    def generate(speech, **kw):
        """Greedy transcripts of `speech` (a list of 16 kHz arrays); `kw` goes to `generate`."""
        with torch.no_grad():
            if whisper:
                feats = asr_processor.feature_extractor(
                    speech, sampling_rate=asr_sr, return_tensors="pt"
                ).input_features.to(asr_model.device, asr_model.dtype)
                ids = asr_model.generate(feats, language=language, task="transcribe", do_sample=False,
                                         num_beams=1, **kw)
                return [t.strip() for t in asr_processor.batch_decode(ids, skip_special_tokens=True)]
            inputs = asr_processor.apply_transcription_request(speech, language=[language] * len(speech)).to(
                asr_model.device, asr_model.dtype
            )
            output_ids = asr_model.generate(**inputs, **kw)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        texts = asr_processor.decode(generated_ids, return_format="transcription_only")
        return [t.strip() for t in texts] if ztts2 else texts

    def transcribe(paths):
        speech = [load_audio(path, sampling_rate=asr_sr) for path in paths]
        if max_samples is not None:
            speech = [s[:max_samples] for s in speech]
        texts = generate(speech, max_new_tokens=max_new_tokens)
        if ztts2:
            # ZTTS2-Eval's loop guard: re-decode only the looping transcripts, with a repetition penalty, and keep
            # a retry that breaks the loop or at least shortens it.
            loops = [i for i, text in enumerate(texts) if _degenerate(text)]
            if loops:
                retries = generate([speech[i] for i in loops], **ZTTS2_RETRY_KWARGS)
                for i, retry in zip(loops, retries):
                    if not _degenerate(retry) or len(retry) < len(texts[i]):
                        print(f"  loop guarded: {os.path.basename(paths[i])} ({len(texts[i])} -> {len(retry)} chars)")
                        texts[i] = retry
        return texts

    def rewrite_manifest():
        with open(args.manifest_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Batch clips of similar length together so padding=longest wastes less memory/compute, and
    # any capped long clips cluster into a few batches rather than inflating every batch.
    todo.sort(key=lambda i: entries[i].get("duration") or 0.0)

    # ── Transcribe in ASR minibatches; fill pred_text and persist after each chunk ─
    for start in range(0, len(todo), args.asr_batch_size):
        idxs = todo[start : start + args.asr_batch_size]
        preds = transcribe([_resolve(entries[i]["audio_filepath"]) for i in idxs])
        for i, pred in zip(idxs, preds):
            entries[i]["pred_text"] = pred
            if ztts2:
                entries[i]["asr_profile"] = "ztts2"
                entries[i]["asr_model"] = f"{model_id}@{revision}"
        rewrite_manifest()  # persist progress so a crash doesn't lose completed chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  transcribed {min(start + args.asr_batch_size, len(todo))}/{len(todo)}")

    print("Manifest updated with transcriptions:", os.path.abspath(args.manifest_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=str, required=True, help="JSONL manifest to transcribe (pred_text filled in place).")
    parser.add_argument("--asr_model_id", type=str, default=DEFAULT_ASR_MODEL_ID, help="ASR model used to transcribe for WER.")
    parser.add_argument(
        "--asr_language",
        type=str,
        default="en",
        help="Language hint for the ASR model. Pass '' (empty) to let Qwen3-ASR auto-detect.",
    )
    parser.add_argument("--asr_max_new_tokens", type=int, default=256, help="Max tokens for ASR transcription.")
    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=30.0,
        help="Truncate each clip to at most this many seconds before transcribing (0 = no cap). "
             "Bounds ASR-encoder memory: a runaway TTS generation can be minutes long and OOM the "
             "conv frontend (a batch is padded to its longest clip). Mirrors score_similarity.py.",
    )
    parser.add_argument(
        "--asr_batch_size",
        type=int,
        default=32,
        help="Minibatch size for ASR transcription (clips are padded to the longest in the batch).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-transcribe every row even if it already has a pred_text (default: resume, skip filled rows).",
    )
    parser.add_argument(
        "--asr_profile",
        choices=["leaderboard", "ztts2"],
        default="leaderboard",
        help="'ztts2': ZTTS2-Eval's ASR (Whisper-large-v3 / Qwen3-ASR routed per language, pinned, loop guard); "
             "--asr_model_id is ignored.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="'cuda', 'cpu', or 'cuda:0'.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="ASR model dtype, e.g. 'bfloat16'.")
    args = parser.parse_args()

    main(args)
