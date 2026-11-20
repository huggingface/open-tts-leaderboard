"""
TTS synthesis for the Open TTS Leaderboard (Breeze TTS 2 backend, stage 1).

Breeze TTS 2 (BreezeBlue/Breeze-TTS-2) is a ~3.5B model — a Qwen3 "llama-1B"-flavor backbone,
a T5Gemma2 text encoder, a 16-codebook depth decoder and a Mimi codec — run in-process from
the cloned `breeze-tts` inference repo (its `models` + `breeze_infer` packages register the
`breeze` architecture, which stock transformers does not know). This script loads the model,
tokenizer and audio tokenizer once, synthesizes each target text, times generation for RTFx,
saves wavs, and writes a manifest with `text`, `duration` and generation `time` (empty
`pred_text`).

Breeze has no built-in default voice, so the two tracks use its two templates:
  --no-voice_clone (default) `tts_instruction` — reference-free "Voice Design" from a fixed instruction
                                                 (no SIM; the speaker is NOT stable across utterances).
  --voice_clone              `ref_edit_tata`   — conditioned on the sample's reference clip + transcript.

Both run through `FastBreezeStreamingRuntime` (the runtime `infer.py` uses), one sample at a time;
--fast_all enables its CUDA-graph fast path. Output is 24 kHz mono.
"""

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from run_eval_utils import (
    add_common_args,
    count_parameters,
    load_done_entries,
    load_tts_dataset,
    manifest_entry,
    open_manifest,
    output_paths,
    pending_batches,
    print_next_steps,
    warm_up,
    wav_rel_path,
    write_entry,
)

# The breeze-tts inference repo is cloned (not pip-installed) at image build: its `models` and
# `breeze_infer` packages are imported by path, and `configs/fast.json` is read from the same tree.
BREEZE_REPO = Path(os.environ.get("BREEZE_REPO", "/opt/breeze-tts"))
sys.path.insert(0, str(BREEZE_REPO))

from breeze_infer.runtime import (  # noqa: E402
    load_runtime,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs  # noqa: E402
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig  # noqa: E402
from models.warmup_profile import (  # noqa: E402
    BackbonePrefillWarmupGraph,
    TextEncoderWarmupGraph,
    load_warmup_profile,
)

# infer.py's own ceilings, which override the checkpoint's generation_config (max_new_tokens=750).
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1

# Neutral instruction per language for the --no-voice_clone track; the model card says the
# instruction language should match the target text. `en` is infer.py's default, `zh` its translation.
DEFAULT_INSTRUCTIONS = {
    "en": "Speak clearly and naturally.",
    "zh": "请用清晰自然的语气朗读。",
}


def _cover_all_lengths(profile):
    """Declare every batch size's graphs up to the profile's longest length (step: its shortest).

    configs/fast.json stops batch-1 graphs at 256 tokens (batch 2, the CFG path, goes to 512), and the
    frozen runtime raises on an undeclared shape, so a longer reference prompt in no-CFG voice cloning
    (e.g. ZTTS2-Eval's) failed. Declaring the missing shapes only adds graphs, captured in the
    untimed warmup; shapes the profile already had are unchanged.
    """
    def extend(graphs, size_of, length_of, make):
        lengths = sorted({length_of(g) for g in graphs})
        step, longest = lengths[0], lengths[-1]
        have = {(size_of(g), length_of(g)) for g in graphs}
        extra = [make(b, n) for b in sorted({size_of(g) for g in graphs})
                 for n in range(step, longest + 1, step) if (b, n) not in have]
        return tuple(graphs) + tuple(extra)

    return replace(
        profile,
        text_encoder_graphs=extend(profile.text_encoder_graphs, lambda g: g.batch_size, lambda g: g.token_length,
                                   lambda b, n: TextEncoderWarmupGraph(batch_size=b, token_length=n)),
        backbone_prefill_graphs=extend(profile.backbone_prefill_graphs, lambda g: g.branch_batch_size,
                                       lambda g: g.sequence_length,
                                       lambda b, n: BackbonePrefillWarmupGraph(branch_batch_size=b, sequence_length=n)),
    )


def _default_instruction(language: str) -> str:
    return DEFAULT_INSTRUCTIONS.get((language or "en").split("_")[0], DEFAULT_INSTRUCTIONS["en"])


def main(args):
    device = args.device
    is_cuda = str(device).startswith("cuda")
    instruction = args.instruction or _default_instruction(args.language)

    # The weights are not baked into the image (licence), so they are downloaded here.
    # `load_runtime` resolves `<ckpt>/audio_tokenizer` as a Path, so it must not be given a str.
    if args.checkpoint_path:
        ckpt_dir = Path(args.checkpoint_path)
    else:
        from huggingface_hub import snapshot_download

        ckpt_dir = Path(snapshot_download(args.model_id))
    print(f"Checkpoint: {ckpt_dir}")
    tokenizer, model, audio_tokenizer = load_runtime(
        ckpt_dir, device=device, attn_implementation=args.attn_implementation
    )
    update_generation_config_for_breeze(model)

    streaming_config = FastStreamingConfig(
        max_new_tokens=args.max_new_tokens,
        max_seq_len=MAX_SEQ_LEN,
        fast_all=args.fast_all,
        repetition_penalty=REPETITION_PENALTY,
    )
    runtime = FastBreezeStreamingRuntime(model, audio_tokenizer, streaming_config, tokenizer=tokenizer)

    # Build the fast path's CUDA graphs up front so that one-time cost stays out of the timings.
    if runtime.fast_enabled:
        profile = load_warmup_profile(BREEZE_REPO / "configs" / "fast.json")
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        profile = _cover_all_lengths(profile)
        manifest = runtime.warmup_from_profile(profile)
        print(f"Fast-path warmup: {manifest['total_elapsed_ms']:.2f} ms")

    sampling_rate = runtime.sample_rate  # 24000
    template_name = "ref_edit_tata" if args.voice_clone else "tts_instruction"
    print(
        f"Loaded Breeze TTS 2 ({args.model_id}, sr={sampling_rate}, template={template_name}, "
        f"fast={runtime.fast_enabled})"
    )
    n_params = count_parameters(model)
    print(f"TTS model size: {n_params / 1e9:.2f}B parameters (backbone + text encoder + depth decoder + codec)")

    def synth_one(text, ref_path=None, ref_text=None, request_id="eval", early_stop=False):
        """Synthesize one text; return (audio_1d_numpy, elapsed_s, ttfa).

        With ref_path + ref_text the voice is cloned from that clip; otherwise it comes from
        `instruction`. `prepare_inputs` is timed because it encodes the reference clip (the model's
        own reference processing). `ttfa` holds two clocks, in ms:
          ttfa_model_ms  runtime's `ttfa_internal_ms`: prefill -> first decoded chunk.
          ttfa_ms        end to end, including prepare_inputs.
        """
        # Re-seed per sample for reproducibility (does NOT pin the voice-design speaker).
        set_all_seeds(args.seed)

        request = {"id": request_id, "text": text, "instruction": instruction, "speaker": args.speaker}
        if ref_path is not None:
            request["ref_audio_path"] = str(ref_path)
            request["ref_text"] = ref_text

        if is_cuda:
            torch.cuda.synchronize(device=device)
        start = time.perf_counter()

        inputs = prepare_inputs(
            tokenizer,
            audio_tokenizer,
            model,
            [request],
            get_template(template_name),
            guidance_scale=args.cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        chunks, ttfa = [], {"ttfa_ms": None, "ttfa_model_ms": None, "first_ts": None}
        for chunk in runtime.iter_audio_chunks(inputs, request_id=request_id):
            if not chunks:
                # `.audio` is already a host numpy array, so no extra synchronize is needed.
                first_ts = time.perf_counter()
                ttfa["ttfa_ms"] = (first_ts - start) * 1000.0
                ttfa["ttfa_model_ms"] = (chunk.timing or {}).get("ttfa_internal_ms")
                # Absolute timestamp for the TTFA probe; stripped before the manifest is written.
                ttfa["first_ts"] = first_ts
            chunks.append(chunk.audio)
            if early_stop:
                break

        if is_cuda:
            torch.cuda.synchronize(device=device)
        elapsed = time.perf_counter() - start

        if chunks:
            audio = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks])
        else:
            audio = np.zeros(int(0.1 * sampling_rate), dtype=np.float32)
        return audio, elapsed, ttfa

    # Layout: results/<model_safe>/MODEL_<safe>_DATASET_<dsid>.jsonl + <dsid>/output_<id>.wav.
    # Manifest paths are relative to model_dir so later stages resolve them wherever it is copied.
    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir
    # Reference clips are saved for stage 3 (SIM), and because the cloning template takes a path.
    prompt_dir = os.path.join(output_dir, "prompts")
    if args.voice_clone:
        os.makedirs(prompt_dir, exist_ok=True)

    dataset = load_tts_dataset(args, extra_columns=("prompt_text", "prompt_audio") if args.voice_clone else ())

    # ── TTFA probe: writes ONLY a JSON sidecar (no wavs, no manifest) ─────────
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported lazily: ttfa_probe.py is only injected by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        # Reference clips go to a temp dir so existing results are never touched.
        tmp_dir = tempfile.mkdtemp(prefix="ttfa_refs_")

        def _gen(job, early_stop):
            audio, _, ttfa = synth_one(
                job["text"], ref_path=job.get("ref_path"), ref_text=job.get("ref_text"),
                request_id=f"ttfa-{job['id']}", early_stop=early_stop,
            )
            return audio, sampling_rate, ttfa["first_ts"]

        jobs = []
        for i in sample_indices(args.ttfa_probe, len(dataset)):
            sample_id = dataset[i]["id"]
            job = {"id": sample_id, "text": dataset[i][args.text_column]}
            if args.voice_clone:
                pa = dataset[i]["prompt_audio"]
                ref = os.path.join(tmp_dir, f"prompt_{sample_id}.wav")
                sf.write(ref, np.asarray(pa["array"], dtype=np.float32), pa["sampling_rate"])
                job["ref_path"], job["ref_text"] = ref, dataset[i]["prompt_text"]
            jobs.append(job)

        run_probe(
            _gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={"device": args.device, "fast_all": args.fast_all, "cfg_scale": args.cfg_scale},
        )
        return

    manifest_path = paths.manifest_path
    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    def synth_row(row):
        """Synthesize one dataset row; return (audio, elapsed, ttfa, prompt_rel_path)."""
        sample_id = row["id"]
        prompt_rel_path = prompt_path = prompt_text = None
        if args.voice_clone:
            # Save the reference clip before the timer — file I/O is not generation.
            pa = row["prompt_audio"]
            prompt_rel_path = os.path.join(dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
            prompt_path = os.path.join(model_dir, prompt_rel_path)
            if not os.path.exists(prompt_path):
                sf.write(prompt_path, np.asarray(pa["array"], dtype=np.float32), pa["sampling_rate"])
            prompt_text = row["prompt_text"]
        audio, elapsed, ttfa = synth_one(
            row[args.text_column], ref_path=prompt_path, ref_text=prompt_text, request_id=f"eval-{sample_id}"
        )
        return audio, elapsed, ttfa, prompt_rel_path

    # ── Main loop: synthesize one sample at a time → write JSONL ─────────────
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda b: synth_row({k: v[0] for k, v in b.items()}), dataset, starts, 1, args.warmup_steps)
    for i in tqdm(starts, desc="Generating"):
        row = dataset[i]
        # Store the path relative to model_dir (the manifest's dir); write to the full path.
        rel_path = wav_rel_path(dataset_dir_name, row["id"])
        audio, elapsed, ttfa, prompt_rel_path = synth_row(row)
        sf.write(os.path.join(model_dir, rel_path), audio, sampling_rate)

        entry = manifest_entry(rel_path, len(audio) / sampling_rate, elapsed, row[args.text_column])
        entry.update({k: v for k, v in ttfa.items() if k != "first_ts"})
        if args.voice_clone:
            entry.update(prompt_audio_filepath=prompt_rel_path, sim="")
        write_entry(manifest_file, entry)
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser, voice_clone=False)

    parser.add_argument("--model_id", type=str, default="BreezeBlue/Breeze-TTS-2", help="Hub repo id (downloaded unless --checkpoint_path is set; also names the outputs).")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Local dir with the weights + bundled audio_tokenizer/. Default: snapshot_download(--model_id).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language of the target text; picks the default voice-design instruction. Defaults to --split.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help="Backbone attention. 'eager' is infer.py's default; the T5Gemma2 text encoder always "
             "uses flash_attention_2 (the checkpoint pins it), so flash-attn is required either way.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help="Max audio frames to generate.")
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance. 1.0 (infer.py's default) runs a single branch; >1 adds a "
             "negative branch. The card suggests 4 to strengthen instruction-following.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Natural-language delivery instruction. Defaults to a neutral one for --language.",
    )
    parser.add_argument("--speaker", type=str, default="S0", help="Speaker tag prefixed to each segment.")
    parser.add_argument(
        "--fast_all",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the CUDA-graph fast path for every stage (warmup excluded from timings). "
             "Needs ~14.4 GiB instead of ~7.7 GiB.",
    )

    args = parser.parse_args()
    if args.language is None:
        args.language = args.split

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
