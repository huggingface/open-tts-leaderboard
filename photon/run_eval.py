"""
Time-to-first-audio probe for the Open TTS Leaderboard (Moondream Photon backend).

A third-party engine for models the leaderboard already evaluates: Moondream's Photon runtime
(https://moondream.ai/blog/photon-can-now-speak) serves Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice
and hexgrad/Kokoro-82M from the same checkpoints through its Kestrel engine, streaming audio.

API (moondream>=2.5.0, which pins kestrel==0.8.2):

    with md.photon(model_id, device=...) as voice:
        stream = voice.synthesize(text=..., voice=..., language=..., stream=True)   # Qwen3-TTS
        stream = voice.synthesize(phonemes=..., voice=..., stream=True)             # Kokoro
        for update in stream: update["audio"], update["sample_rate"]                 # 24 kHz f32

First audio = first AUDIBLE audio: kestrel's SpeechOnsetTrimmer buffers output until three
consecutive 10 ms blocks exceed -45 dBFS and drops the leading silence. This clock is stricter than
other backends' first-chunk-of-anything, and audio_s is slightly shorter.

Kokoro takes phonemes: Photon ships no G2P, so we use kokoro's KPipeline(model=False) (the same
misaki G2P + chunking as kokoro/run_eval.py). G2P runs inside the timed call, as in kokoro's TTFA,
and its cost is recorded separately (g2p_ms).

TTFA only: Photon's audio is not what the eval scores, so running without --ttfa_probe is
refused. Sidecars are engine-tagged "photon" by submit_ttfa_jobs.sh. No voice cloning: only the
CustomVoice Qwen3-TTS checkpoints and Kokoro voice packs are supported.
"""

import argparse
import os
import time

import numpy as np
import torch

import moondream as md

from run_eval_utils import add_common_args, load_tts_dataset, set_seed

SAMPLE_RATE = 24_000  # Photon's fixed TTS output rate, for both models

# Kept identical to qwen3tts/run_eval.py: the submit scripts pass ISO codes, Qwen3-TTS wants the
# English language NAME. Photon casefolds it, so "English" and "english" are the same key.
LANGUAGE_CODE_TO_NAME = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


def _family(model_id):
    name = model_id.split("/")[-1].lower()
    if "kokoro" in name:
        return "kokoro"
    if "qwen3-tts" in name:
        if "customvoice" not in name:
            raise SystemExit(
                f"Photon serves only the Qwen3-TTS CustomVoice checkpoints, not {model_id}.\n"
                "Base (voice-clone) and VoiceDesign variants are not registered in kestrel 0.8.2."
            )
        return "qwen3tts"
    raise SystemExit(f"Unsupported model for the Photon backend: {model_id} "
                     f"(registered: {md.photon_models()})")


def main(args):
    if args.ttfa_probe == 0:
        raise SystemExit(
            "This backend is TTFA-only: pass --ttfa_probe=N (or -1 for the whole split).\n"
            "Photon's kernels and speech-onset trimming mean its audio is not the audio the\n"
            "leaderboard's WER/RTFx are scored from. Use qwen3tts/ or kokoro/ for those."
        )

    family = _family(args.model_id)
    if family == "qwen3tts" and not args.device.startswith("cuda"):
        # Photon itself raises at engine start, but deeper and less clearly.
        raise SystemExit(f"Photon runs Qwen3-TTS on CUDA only (got --device={args.device}). "
                         "Kokoro is the Photon model that runs on CPU.")

    seed = args.seed
    set_seed(seed)

    from importlib.metadata import version
    versions = {p: version(p) for p in ("moondream", "kestrel", "kestrel-kernels")}

    t0 = time.perf_counter()
    voice = md.photon(args.model_id, device=args.device)
    print(f"Loaded {args.model_id} via Photon on {args.device} in {time.perf_counter() - t0:.1f}s "
          f"({versions})")

    g2p = None
    if family == "kokoro":
        # G2P only: model=False builds the misaki pipeline without loading Kokoro's weights, so the
        # reference model never shares the GPU with Photon's copy.
        from kokoro import KPipeline
        g2p = KPipeline(lang_code=args.lang_code, repo_id=args.model_id, model=False)

    g2p_ms = []           # Kokoro only: phonemization time inside each timed call
    chunk_counts = []     # streamed updates per utterance — see the sidecar note below

    def _request(text):
        """Build the synthesize() kwargs for one utterance (G2P included for Kokoro)."""
        if family == "kokoro":
            t = time.perf_counter()
            # Same chunking kokoro/run_eval.py gets from KPipeline; Photon re-splits at its own
            # 510-phoneme limit, so joining the chunks back loses nothing.
            phonemes = " ".join(r.phonemes for r in g2p(text) if r.phonemes)
            g2p_ms.append((time.perf_counter() - t) * 1000.0)
            return dict(phonemes=phonemes, voice=args.voice, speed=args.speed)
        return dict(
            text=text, voice=args.voice, language=args.language,
            settings={
                "do_sample": args.do_sample, "temperature": args.temperature,
                "top_k": args.top_k, "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "max_tokens": args.max_new_tokens,
            },
        )

    # Kestrel captures graphs / selects kernels on first use; keep that out of the probe rows.
    for i in range(args.warmup_steps):
        t0 = time.perf_counter()
        voice.synthesize(**_request("Hello there, this is a warm-up sentence for the engine."))
        print(f"warm-up {i + 1}/{args.warmup_steps}: {time.perf_counter() - t0:.2f}s")
    g2p_ms.clear()

    model_safe = args.model_id.replace("/", "-")
    # Not created here: run_probe creates it on write, so a crash leaves no dir and the job fails.
    out_dir = os.path.join("results", model_safe)

    # Keeps only `id` + the text, so no Audio() column is ever decoded (no torchcodec needed).
    dataset = load_tts_dataset(args)

    # Imported HERE, not at module scope: the probe module is injected only by submit_ttfa_jobs.sh.
    from ttfa_probe import run_probe, sample_indices

    def _gen(job, early_stop):
        """One streaming synthesis at batch size 1, timestamping the first yielded update."""
        torch.manual_seed(seed)
        first, chunks, sample_rate = None, [], None
        # Leaving the block closes the stream, which cancels an early-stopped request inside the
        # engine: TTFA is already known and the rest of the utterance is pure cost.
        with voice.synthesize(stream=True, **_request(job["text"])) as stream:
            for update in stream:
                audio = np.asarray(update["audio"], dtype=np.float32).reshape(-1)
                if first is None:
                    if not audio.size:
                        continue
                    first = time.perf_counter()
                sample_rate = update["sample_rate"]
                chunks.append(audio)
                if early_stop:
                    break
        chunk_counts.append(len(chunks))
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return audio, sample_rate or SAMPLE_RATE, first

    jobs = [{"text": dataset[i][args.text_column]}
            for i in sample_indices(args.ttfa_probe, len(dataset))]

    extra = {
        "device": args.device,
        "note": f"driven at batch size 1, Moondream Photon streaming ({family})",
        "engine": "photon",
        "engine_versions": versions,
        "first_audio": "first audible (onset-trimmed, -45 dBFS x 3x10ms)",
        # 1 update means nothing was incremental (Kokoro streams per ~510-phoneme segment, so a
        # short prompt is one segment); many updates arriving late means buffering.
        "chunks_per_utterance": chunk_counts,
        "voice": args.voice,
    }
    if family == "kokoro":
        extra.update({
            "lang_code": args.lang_code, "speed": args.speed,
            "g2p": "kokoro.KPipeline(model=False) / misaki, timed inside ttfa_ms",
            "g2p_ms": g2p_ms,
        })
    else:
        extra.update({
            "language": args.language,
            "sampling": {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                         "do_sample": args.do_sample, "repetition_penalty": args.repetition_penalty},
        })

    try:
        run_probe(
            _gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=out_dir, n=args.ttfa_probe, extra=extra,
        )
    finally:
        voice.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice or hexgrad/Kokoro-82M — the SAME weights "
             "the qwen3tts/ and kokoro/ backends load.",
    )
    parser.add_argument(
        "--voice", type=str, default=None,
        help="Qwen3-TTS built-in speaker (default Ryan) or Kokoro voice pack (default af_heart; its "
             "prefix must match --lang_code).",
    )
    parser.add_argument("--language", type=str, default="English",
                        help="Qwen3-TTS only. ISO codes are mapped to the language NAME.")
    parser.add_argument("--lang_code", type=str, default="a",
                        help="Kokoro only: KPipeline G2P language ('a' = American English).")
    parser.add_argument("--speed", type=float, default=1.0, help="Kokoro only: speech speed.")
    # Qwen3-TTS sampling — defaults are kestrel's Qwen3TTSSampling, which match qwen-tts and
    # faster-qwen3-tts, so the three Qwen3-TTS engines are driven identically.
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    # --warmup_steps = untimed synthesize() calls before the probe (Kestrel graph capture / kernel
    # selection). --ttfa_probe is REQUIRED: this backend does nothing else.
    add_common_args(parser)
    parser.set_defaults(dataset_path="hf-audio/cv3_eval", dataset="zero_shot")

    args = parser.parse_args()
    args.language = LANGUAGE_CODE_TO_NAME.get(args.language.lower(), args.language)
    if args.voice is None:
        args.voice = "af_heart" if "kokoro" in args.model_id.lower() else "Ryan"

    print("*" * 100)
    print(f"TTFA probe: {args.model_id} via Moondream Photon on "
          f"{args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
