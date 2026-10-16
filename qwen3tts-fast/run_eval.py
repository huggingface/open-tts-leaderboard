"""
Time-to-first-audio probe for the Open TTS Leaderboard (faster-qwen3-tts backend).

A second ENGINE for the Qwen3-TTS checkpoints that `qwen3tts/run_eval.py` evaluates. Qwen's
`qwen-tts` has no streaming API, so its TTFA is a whole-utterance clock; andimarafioti/faster-qwen3-tts
re-implements inference over the same weights and exposes streaming generators, making a real TTFA
measurable. It needs a newer `transformers` than `qwen-tts`, hence its own directory and image.

--backend:
  * torch (default) — CUDA graphs, GPU only.
  * ggml            — the qwentts.cpp runtime (GGUF weights from `Serveurperso/Qwen3-TTS-GGUF`),
                      which also runs on CPU.
Both expose the same streaming generators, so _gen drives either unchanged.

TTFA ONLY: upstream notes its outputs can differ slightly from the reference implementation, so
running without --ttfa_probe is refused. Sidecars are engine-tagged by submit_ttfa_jobs.sh.
"""

import argparse
import os
import time

import numpy as np
import torch

from faster_qwen3_tts import FasterQwen3TTS

from run_eval_utils import add_common_args, load_tts_dataset, set_seed

torch.set_float32_matmul_precision("high")

# Kept identical to qwen3tts/run_eval.py: ISO code -> the English language NAME Qwen3-TTS wants.
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


def main(args):
    if args.ttfa_probe == 0:
        raise SystemExit(
            "This backend is TTFA-only: pass --ttfa_probe=N (or -1 for the whole split).\n"
            "It exists to measure streaming latency for an engine whose audio upstream does NOT\n"
            "guarantee matches the reference implementation, so it must not generate the wavs the\n"
            "leaderboard's WER/RTFx are scored from. Use qwen3tts/run_eval.py for those."
        )

    seed = args.seed
    set_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Same rule as qwen3tts/run_eval.py so the two engines resolve the same variant for a model id.
    if args.mode == "auto":
        mode = "voice_clone" if "base" in args.model_id.split("/")[-1].lower() else "custom_voice"
    else:
        mode = args.mode

    # The ggml backend's device is decided by the installed qwentts-cpp-python wheel, recorded only
    # in its local-version suffix ("0.3.1+cpu", "0.3.1+cu128").
    wheel_version = wheel_variant = None
    if args.backend == "ggml":
        from importlib.metadata import version, PackageNotFoundError
        try:
            wheel_version = version("qwentts-cpp-python")
            wheel_variant = wheel_version.split("+", 1)[1] if "+" in wheel_version else None
        except PackageNotFoundError:
            raise SystemExit(
                "--backend=ggml needs qwentts-cpp-python, which is not installed.\n"
                "See the Dockerfile of the bezzam/evals-ggml-qwentts Space — the +cpu wheel comes from the Hub index, "
                "not PyPI (PyPI ships the CUDA 12.8 build)."
            )

    if args.backend == "torch":
        if not args.device.startswith("cuda"):
            raise SystemExit(
                f"--backend=torch needs a CUDA device (got --device={args.device}). CUDA graph\n"
                "capture IS this backend's acceleration, so there is nothing to measure without it.\n"
                "For a CPU measurement use --backend=ggml, which runs the qwentts.cpp runtime."
            )
        model = FasterQwen3TTS.from_pretrained(
            args.model_id,
            device=args.device,
            dtype=getattr(torch, args.dtype),
            attn_implementation=args.attn_implementation,
        )
    else:
        # `quant` picks the GGUF (BF16 is lossless; Q8_0/Q4_K_M are a different model, recorded in
        # the sidecar). GGMLQwen3TTS ignores --device, so cross-check it against the installed wheel
        # to avoid filing a CPU measurement under a GPU flavor (or vice versa).
        if wheel_variant == "cpu" and args.device.startswith("cuda"):
            raise SystemExit(
                f"--backend=ggml with --device={args.device}, but the installed "
                f"qwentts-cpp-python is the CPU wheel ({wheel_version}).\n"
                "It would compute on the CPU and the sidecar would be filed under a GPU flavor.\n"
                "Run this target with TTFA_DEVICE=cpu, or build the image with a +cuNNN wheel."
            )
        if wheel_variant and wheel_variant.startswith("cu") and args.device == "cpu":
            raise SystemExit(
                f"--backend=ggml with --device=cpu, but the installed qwentts-cpp-python is a "
                f"CUDA wheel ({wheel_version}).\n"
                "The measurement would not be the CPU number it is labelled as. Install the "
                "+cpu wheel for a CPU probe."
            )
        model = FasterQwen3TTS.from_pretrained(
            args.model_id,
            backend="ggml",
            quant=args.quant,
        )
    print(f"Loaded {args.model_id} via faster-qwen3-tts backend={args.backend}"
          f"{'/' + args.quant if args.backend == 'ggml' else ''} "
          f"(mode={mode}, speaker={args.speaker}, language={args.language}, chunk_size={args.chunk_size})")

    # Keep one-time CUDA graph capture (torch) / buffer priming (ggml) out of the timed rows.
    if args.warmup:
        t0 = time.perf_counter()
        model.warmup()
        print(f"warm-up ({args.backend}): {time.perf_counter() - t0:.2f}s")

    model_safe = args.model_id.replace("/", "-")
    # NOT created here (run_probe does it before writing): if the probe crashes, the job's final
    # `cp -r results/<model> /results/` then fails instead of exiting 0 with no sidecar.
    out_dir = os.path.join("results", model_safe)

    # Unneeded columns are dropped BEFORE indexing rows: datasets decodes `prompt_audio` on
    # __getitem__, which needs torchcodec even in custom_voice mode.
    dataset = load_tts_dataset(args, extra_columns=("prompt_audio", "prompt_text") if mode == "voice_clone" else ())

    # Imported lazily: ttfa_probe.py is only injected by submit_ttfa_jobs.sh.
    from ttfa_probe import run_probe, sample_indices

    chunk_counts = []  # chunks per utterance

    def _gen(job, early_stop):
        """One streaming generation (batch size 1), timestamping the first yielded chunk."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        common = dict(
            text=job["text"],
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            chunk_size=args.chunk_size,
        )
        if mode == "voice_clone":
            stream = model.generate_voice_clone_streaming(
                ref_audio=job["prompt_path"], ref_text=job["prompt_text"], **common)
        else:
            stream = model.generate_custom_voice_streaming(speaker=args.speaker, **common)

        first, chunks, sample_rate = None, [], None
        for new_audio, sr, _timing in stream:
            if first is None:
                first = time.perf_counter()
            sample_rate = sr
            chunks.append(np.asarray(new_audio, dtype=np.float32).reshape(-1))
            if early_stop:
                stream.close()
                break

        chunk_counts.append(len(chunks))
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return audio, sample_rate or args.sample_rate, first

    jobs, prompt_dir = [], None
    for i in sample_indices(args.ttfa_probe, len(dataset)):
        row = dataset[i]
        job = {"text": row[args.text_column]}
        if mode == "voice_clone":
            import soundfile as sf
            # The ggml images ship no torchcodec; fail clearly instead of deep inside datasets.
            try:
                import torchcodec  # noqa: F401
            except ImportError:
                raise SystemExit(
                    "--mode=voice_clone needs torchcodec to decode the dataset's prompt_audio, "
                    "and it is not installed.\n"
                    "Add `pip install \"torchcodec==0.7.*\"` to this backend's Dockerfile, or run "
                    "with --mode=custom_voice (what every TTFA target uses)."
                )
            # generate_voice_clone_streaming takes a path, not an array. Written to a temp dir so the
            # results tree only ever receives the sidecar (and out_dir is not created early).
            if prompt_dir is None:
                import tempfile
                prompt_dir = tempfile.mkdtemp(prefix="ttfa_refs_")
            path = os.path.join(prompt_dir, f"prompt_{row.get('id', i)}.wav")
            sf.write(path, np.asarray(row["prompt_audio"]["array"], dtype=np.float32),
                     row["prompt_audio"]["sampling_rate"])
            job["prompt_path"], job["prompt_text"] = path, row["prompt_text"]
        jobs.append(job)

    run_probe(
        _gen, jobs,
        model_id=args.model_id, dataset_path=args.dataset_path,
        dataset_config=args.dataset, split=args.split,
        out_dir=out_dir, mode_suffix="_voice_clone" if mode == "voice_clone" else "",
        n=args.ttfa_probe,
        extra={
            "device": args.device,
            "note": f"driven at batch size 1, faster-qwen3-tts streaming, mode={mode}",
            # chunks_per_utterance distinguishes "no incremental output" (1 chunk) from late delivery.
            "chunks_per_utterance": chunk_counts,
            # chunk_size sets the TTFA floor; TTFAs at different chunk sizes are not comparable.
            "chunk_size": args.chunk_size,
            "engine": "faster-qwen3-tts",
            "backend": args.backend,
            # TTFA is only comparable within one quant.
            "quant": args.quant if args.backend == "ggml" else None,
            "gguf_repo": "Serveurperso/Qwen3-TTS-GGUF" if args.backend == "ggml" else None,
            # The ggml runtime wheel ("+cpu" vs "+cuNNN") is what decides CPU vs GPU.
            "qwentts_cpp": wheel_version,
            "sampling": {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                         "do_sample": args.do_sample, "repetition_penalty": args.repetition_penalty},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    # CV3-Eval by default. --device must be cuda for --backend=torch (ggml picks its own device).
    parser.set_defaults(dataset_path="hf-audio/cv3_eval", dataset="zero_shot")

    parser.add_argument(
        "--model_id", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="Qwen3-TTS checkpoint. faster-qwen3-tts supports the 0.6B/1.7B Base, CustomVoice and "
             "VoiceDesign variants — the SAME weights the qwen3tts backend loads.",
    )
    parser.add_argument(
        "--mode", type=str, default="auto", choices=["auto", "custom_voice", "voice_clone"],
        help="Matches qwen3tts/run_eval.py: 'custom_voice' uses a built-in --speaker, 'voice_clone' "
             "clones the dataset's prompt_audio/prompt_text, 'auto' picks voice_clone for Base ids.",
    )
    parser.add_argument("--speaker", type=str, default="Ryan",
                        help="Built-in speaker voice. Ignored in voice_clone mode.")
    parser.add_argument(
        "--language", type=str, default="English",
        help="Target language. ISO codes ('en', 'zh', ...) are mapped to the English NAME the model "
             "wants, so the submit scripts pass the same code they give every other backend.",
    )
    parser.add_argument(
        "--backend", type=str, default="torch", choices=["torch", "ggml"],
        help="Runtime. 'torch' = CUDA graphs (GPU only). 'ggml' = the qwentts.cpp runtime, which "
             "runs on CPU and is the only way to get a CPU TTFA for this model.",
    )
    parser.add_argument(
        "--quant", type=str, default="BF16", choices=["BF16", "F32", "Q8_0", "Q4_K_M"],
        help="GGUF quantization for --backend=ggml (from Serveurperso/Qwen3-TTS-GGUF). BF16 is "
             "lossless w.r.t. the reference weights; Q8_0/Q4_K_M are a different model and their "
             "TTFA is not comparable with a BF16 one. Ignored for --backend=torch.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype, e.g. 'bfloat16'.")
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        help="Attention impl ('sdpa'/'flash_attention_2').")
    parser.add_argument("--sample_rate", type=int, default=24000,
                        help="Fallback sample rate, used only if the generator yields none.")
    parser.add_argument(
        "--chunk_size", type=int, default=12,
        help="Codec frames per streamed chunk — the library default. This SETS the TTFA floor, so "
             "keep it fixed across runs and never compare TTFA across different values.",
    )
    # Sampling defaults copied from the library's streaming signatures.
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                        help="Warm up (torch: CUDA-graph capture; ggml: buffer priming) before the timed rows (on by default).")

    args = parser.parse_args()
    args.language = LANGUAGE_CODE_TO_NAME.get(args.language.lower(), args.language)

    print("*" * 100)
    print(f"TTFA probe: {args.model_id} via faster-qwen3-tts on "
          f"{args.dataset_path} / {args.dataset} / {args.split}")
    print(f"Target language: {args.language}, chunk_size: {args.chunk_size}")
    print("*" * 100)

    main(args)
