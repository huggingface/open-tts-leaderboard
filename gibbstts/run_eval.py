"""GibbsTTS EN/ZH voice cloning: BF16 DiT autocast, batch32, compile, padded tail.

Weights, codec, logits and solver remain FP32. TTFA/Japanese LoRA are separate.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest,
    output_paths, print_model_size, print_next_steps, set_seed, wav_rel_path, write_entry,
)

SOURCE_REVISION = "2af61715fe832b8dbef909b545090a56056428a3"
MODEL_REVISION = "a31945f20bf84498af41508ff3b126ea7f53b5c2"
WEIGHT_FILES = ("GibbsTTS_large_ema.safetensors", "MaskGCT_codec_encoder.safetensors",
                "MaskGCT_codec_decoder.safetensors")
SAMPLE_RATE = 24000
ROPE_CACHE_POSITIONS = 16384


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, batch_size=32, voice_clone=True)
    parser.add_argument("--model_id", default="ydqmkkx/GibbsTTS")
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--checkpoint_path", default="/opt/models/GibbsTTS")
    parser.add_argument("--source_dir", default="/opt/GibbsTTS")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tail_padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.5)
    parser.add_argument("--rescale_cfg", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    return parser


def validate_args(args):
    if args.split not in {"en", "zh"}:
        raise ValueError("The base checkpoint supports only en/zh; Japanese requires a separate LoRA.")
    if not args.voice_clone:
        raise ValueError("GibbsTTS requires reference audio/text; use --voice_clone.")
    if args.ttfa_probe:
        raise ValueError("The optional no-cloning TTFA probe is not implemented.")
    if args.batch_size < 1 or args.steps < 1 or args.warmup_steps < 0:
        raise ValueError("batch_size/steps must be positive; warmup_steps must be non-negative.")
    if args.compile and args.warmup_steps < 1:
        raise ValueError("Compiled evaluation requires initial untimed warm-up.")
    if not all(math.isfinite(v) for v in
               (args.speed, args.temperature, args.top_p, args.cfg, args.rescale_cfg)):
        raise ValueError("Sampling parameters must be finite.")
    if args.speed <= 0 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("speed/temperature must be positive; top_p must be in (0, 1].")
    if args.device not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("Use cpu or the default CUDA device.")
    if args.device == "cpu" and (args.precision != "fp32" or args.compile):
        raise ValueError("CPU development requires --precision=fp32 --no-compile.")
    if args.model_id != "ydqmkkx/GibbsTTS" or args.model_revision != MODEL_REVISION:
        raise ValueError("This backend is pinned to the evaluated GibbsTTS base checkpoint.")


def load_model(args):
    import torch
    from huggingface_hub import snapshot_download
    source = Path(args.source_dir).resolve()
    if not (source / "config.py").is_file():
        raise FileNotFoundError(f"GibbsTTS checkout missing: {source}")
    sys.path.insert(0, str(source))
    from config import ModelConfig
    from models import GibbsTTS
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; leaderboard generation requires H200.")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    checkpoint = Path(args.checkpoint_path).resolve()
    snapshot_download(args.model_id, revision=args.model_revision, local_dir=str(checkpoint),
                      allow_patterns=list(WEIGHT_FILES))
    config = ModelConfig()
    config.infer_ckpt_dir = str(checkpoint)
    config.infer_ckpt_path = str(checkpoint / WEIGHT_FILES[0])
    for key in ("steps", "cfg", "rescale_cfg", "temperature", "top_p", "speed"):
        setattr(config, key, getattr(args, key))
    model = GibbsTTS(config).eval()
    if not all(p.dtype == torch.float32 for p in model.parameters()):
        raise RuntimeError("Expected FP32 weights; BF16 is estimator autocast only.")
    print_model_size(model.model)
    return model


def configure_inference(model, args):
    import torch
    import batched_inference as native
    native.enable_positioned_rope(model.model.estimator)
    for block in model.model.estimator.blocks:
        block.attn.rotary_pe._build_cache(ROPE_CACHE_POSITIONS, torch.device(args.device))
    backend = None
    if args.compile:
        from compiler_backend import CompileBackend
        backend = CompileBackend()
        native.estimator_infer = torch.compile(native.estimator_infer, backend=backend,
                                              fullgraph=True, dynamic=True)
    def generate(rows, info):
        set_seed(args.seed)  # inside the generation timer, matching the full experiments
        return native.synthesize_many(model, rows, args.split, args.precision,
            cache_static=True, skip_top_p_one=True,
            pad_to=args.batch_size if args.tail_padding else None, batch_info=info)
    return generate, backend


def prepare_prompt(prompt_audio, destination):
    import numpy as np
    import soundfile as sf
    audio = np.asarray(prompt_audio["array"], dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Reference audio must be nonempty, finite and mono.")
    sf.write(destination, audio, int(prompt_audio["sampling_rate"]), subtype="FLOAT")


def generate_batch(generate, args, rows):
    import numpy as np
    import torch
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    begin = time.perf_counter()
    info = {}
    audio = generate(rows, info)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - begin
    if len(audio) != len(rows):
        raise ValueError("Expected one output per real row; discard dummy outputs.")
    audio = [np.asarray(wave, dtype=np.float32) for wave in audio]
    if any(wave.ndim != 1 or not wave.size or not np.isfinite(wave).all() for wave in audio):
        raise ValueError("Empty, non-mono or nonfinite generated waveform.")
    return audio, elapsed, info


def evaluate(args, model):
    import soundfile as sf
    from tqdm import tqdm
    paths = output_paths(args, "_voice_clone")
    prompt_dir = Path(paths.output_dir) / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_tts_dataset(args, extra_columns=("prompt_text", "prompt_audio"))
    settings = {key: getattr(args, key) for key in
        ("model_id", "model_revision", "split", "dataset_path", "dataset", "seed", "steps", "cfg",
         "rescale_cfg", "temperature", "top_p", "speed", "warmup_steps", "device", "batch_size",
         "precision", "compile", "tail_padding", "max_eval_samples", "text_column")}
    settings.update(source_revision=SOURCE_REVISION, sample_rate=SAMPLE_RATE,
        rope_cache_positions=ROPE_CACHE_POSITIONS, voice_cloning=True, batch_inference=True,
        static_cache=True, top_p_one_fast_path=True,
        precision_scope=("BF16 DiT autocast; FP32 weights/codec/logits/solver"
                         if args.precision == "bf16" else "FP32 inference"),
        compiler="inductor/default/fullgraph/dynamic" if args.compile else None)
    metadata_path = Path(paths.output_dir) / "generation_config.json"
    previous = {}
    if args.resume and Path(paths.manifest_path).is_file():
        previous = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        if previous.get("settings") != settings:
            raise ValueError("Cannot resume with different generation settings.")
    done = load_done_entries(paths.manifest_path, args.resume)
    done = {key: row for key, row in done.items() if (Path(paths.model_dir) / key).is_file()
        and row.get("prompt_audio_filepath") and (Path(paths.model_dir) / row["prompt_audio_filepath"]).is_file()}
    ids = [str(sid) for sid in dataset["id"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Dataset IDs must be unique.")
    starts = []
    for start in range(0, len(ids), args.batch_size):
        keys = [wav_rel_path(paths.dataset_dir_name, sid) for sid in ids[start:start + args.batch_size]]
        if not all(key in done for key in keys):
            starts.append(start)
            for key in keys:
                done.pop(key, None)  # repair an entire original batch; preserve shape/RNG/timing
    with open(paths.manifest_path, "w", encoding="utf-8") as stream:
        for sid in ids:
            key = wav_rel_path(paths.dataset_dir_name, sid)
            if key in done:
                write_entry(stream, done[key])
    if not starts:
        print("All real rows are complete.")
        print_next_steps(True)
        return
    rows = []
    for item in dataset:
        sid = str(item["id"])
        if not sid or sid in {".", ".."} or "/" in sid or "\\" in sid:
            raise ValueError(f"Unsafe dataset ID: {sid!r}")
        text, prompt_text = item[args.text_column], item["prompt_text"]
        if not isinstance(text, str) or not text.strip() or not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError(f"Empty target/reference text for {sid}")
        prompt_relative = os.path.join(paths.dataset_dir_name, "prompts", f"prompt_{sid}.wav")
        prompt = Path(paths.model_dir) / prompt_relative
        prepare_prompt(item["prompt_audio"], prompt)
        rows.append(dict(id=sid, text=text, prompt_text=prompt_text, prompt_path=prompt,
                         prompt_relative=prompt_relative))
    generate, backend = configure_inference(model, args)
    events = lambda: len(backend.events) if backend is not None else 0
    warmups = []
    first = rows[starts[0]:starts[0] + args.batch_size]
    for _ in range(args.warmup_steps):
        before = events()
        _, elapsed, _ = generate_batch(generate, args, first)
        warmups.append(dict(seconds=elapsed, new_backend_attempts=events() - before))
    batches = [b for b in previous.get("batches", []) if b["start"] not in starts]
    metadata = dict(settings=settings, warmups=warmups, batches=batches,
        previous_warmups=previous.get("warmups", []),
        timing="Only initial warmups excluded; frontend/reference read+encode/pack/cache/DiT/CTMC/real decode included; dummy compute included; duration only real audio; writes/transfers/ASR/SIM excluded")
    def save_metadata():
        metadata["compiler_events"] = backend.events if backend is not None else []
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    save_metadata()
    with open_manifest(paths.manifest_path, done) as stream:
        for start in tqdm(starts, desc="Generating"):
            chunk = rows[start:start + args.batch_size]
            before = events()
            audio, elapsed, info = generate_batch(generate, args, chunk)
            batches.append(dict(start=start, count=len(chunk), time=elapsed,
                execution_batch_size=info["execution_batch_size"], dummy_count=info["dummy_count"],
                new_backend_attempts=events() - before))
            for row, wave in zip(chunk, audio):
                relative = wav_rel_path(paths.dataset_dir_name, row["id"])
                sf.write(Path(paths.model_dir) / relative, wave, SAMPLE_RATE, subtype="FLOAT")
                entry = manifest_entry(relative, len(wave) / SAMPLE_RATE, elapsed / len(chunk),
                                       row["text"], row["prompt_relative"])
                entry.update(sample_id=row["id"], precision=args.precision, batch_size=args.batch_size,
                    actual_batch_size=len(chunk), execution_batch_size=info["execution_batch_size"],
                    dummy_count=info["dummy_count"], batch_index=start // args.batch_size,
                    compiled=args.compile, tail_padded=args.tail_padding)
                write_entry(stream, entry)
            stream.flush()
            save_metadata()
    completed = load_done_entries(paths.manifest_path, True)
    with open(paths.manifest_path, "w", encoding="utf-8") as stream:
        for sid in ids:
            write_entry(stream, completed[wav_rel_path(paths.dataset_dir_name, sid)])
    print("Results saved at:", os.path.abspath(paths.manifest_path))
    print_next_steps(True)


def main(args):
    validate_args(args)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    set_seed(args.seed)
    evaluate(args, load_model(args))


if __name__ == "__main__":
    main(build_parser().parse_args())
