"""Fixed-preset Phonon stage 1, using the released ptts 0.4.0 API.

Follows paradee/run_eval.py's serial generation and common manifest conventions.
Downloads a pinned public Q8 checkpoint, or accepts an existing local checkpoint.
CPU completion timing is diagnostic, not official H200 RTFx.
Voice cloning (`--voice_clone`) conditions each row on a precomputed voice file for its reference
clip: the speaker encoder is not part of the released checkpoints. The files are pinned like the
checkpoints, and each is checked against the SHA-256 of the clip it was computed from.
The separate TTFA path measures 500 CV3 English requests on cpu-upgrade,
using the blocking API by default or an explicitly selected streaming API.
"""

import argparse
import array
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import struct
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace

from run_eval_utils import (
    add_common_args,
    load_done_entries,
    manifest_entry,
    open_manifest,
    output_paths,
    print_next_steps,
    wav_rel_path,
    write_entry,
)


MODEL_REVISIONS = {
    "gradium/phonon-31m-en": "92741c5b07d085ff840e26fea8ce43110b87852f",
    "gradium/phonon-40m-en": "cc738b7e4d182c2ba1bafb8062aa4846af000637",
    "gradium/phonon-40m-fr": "d91086970206b88efd2c88ca971d0628b5968f0f",
    "gradium/phonon-40m-de": "79cfc7b47fcfc0b666657b981eb366415023a862",
    "gradium/phonon-40m-es": "f0d2f39c9b58ad4c19a189e3ba989ee4df393cc5",
}

# Voice cloning: voices/<SHA-256 of a reference clip>.safetensors holds that clip's speaker latents,
# and index.json maps "<dataset>/<config>/<split>" to {row id: SHA-256}.
CLONE_VOICES_REPO = "gradium/phonon-leaderboard-clone-voices"
CLONE_VOICES_REVISION = "3f2a9ece98f4296c57db4f797591a9e021ee11a9"


def file_hash(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_audio(path, audio, sample_rate):
    """Match the evaluated rc1 path's mono PCM16 conversion, outside the timer."""
    view = memoryview(audio)
    if (view.ndim, view.itemsize, view.format) not in ((1, 4, "f"), (1, 4, "=f")):
        raise ValueError("Expected mono float32 audio.")
    if not len(view) or any(not math.isfinite(value) for value in view):
        raise ValueError("Expected nonempty, finite audio.")
    pcm = array.array("h", (round(max(-1.0, min(1.0, value)) * 32767) for value in view))
    if sys.byteorder != "little":
        pcm.byteswap()
    with open(path, "xb") as target, wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    return len(view) / sample_rate


def check_cpu_isa():
    """The Linux x86-64 wheel requires v3 instructions before native import."""
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        return
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError as error:
        raise ValueError("Cannot verify the x86-64-v3 CPU required by the Linux wheel.") from error
    rows = [set(line.split(":", 1)[1].split()) for line in cpuinfo.splitlines()
            if ":" in line and line.split(":", 1)[0].strip() == "flags"]
    flags = set.intersection(*rows) if rows else set()
    required = {"cx16", "lahf_lm", "popcnt", "pni", "ssse3", "sse4_1", "sse4_2",
                "avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe", "xsave"}
    if not required.issubset(flags) or not {"abm", "lzcnt"}.intersection(flags):
        raise ValueError("The Linux ptts wheel requires x86-64-v3 CPU instructions.")


def gguf_tensor_elements(path):
    """Count stored logical tensor elements (parameters and buffers) from GGUF v3.

    Quantized storage bytes do not change this dimension-product count. It is not
    a count of unique or trainable parameters and excludes external voice assets.
    Read only bounded metadata/descriptors, stopping before the tensor payload.
    """
    with open(path, "rb", buffering=0) as source:
        def read(size):
            if size < 0 or source.tell() + size > 16 * 1024 * 1024:
                raise ValueError("GGUF header exceeds the bounded metadata size.")
            value = source.read(size)
            if len(value) != size:
                raise ValueError("Truncated GGUF header.")
            return value

        def integer(code):
            return struct.unpack("<" + code, read(struct.calcsize("<" + code)))[0]

        def string(limit=1024 * 1024):
            size = integer("Q")
            if size > limit:
                raise ValueError("GGUF string exceeds the bounded metadata size.")
            return read(size).decode("utf-8")

        def skip_value(kind, depth=0):
            if depth > 8:
                raise ValueError("GGUF metadata nesting exceeds the supported depth.")
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                     7: 1, 10: 8, 11: 8, 12: 8}
            if kind in sizes:
                value = read(sizes[kind])
                if kind == 7 and value not in (b"\x00", b"\x01"):
                    raise ValueError("Invalid GGUF boolean metadata.")
            elif kind == 8:
                string()
            elif kind == 9:
                item_kind, count = integer("I"), integer("Q")
                if item_kind not in range(13) or count > 1_000_000:
                    raise ValueError("Invalid or oversized GGUF metadata array.")
                for _ in range(count):
                    skip_value(item_kind, depth + 1)
            else:
                raise ValueError("Unsupported GGUF metadata type.")

        if read(4) != b"GGUF" or integer("I") != 3:
            raise ValueError("Expected a little-endian GGUF v3 checkpoint.")
        tensors, metadata = integer("Q"), integer("Q")
        if not 1 <= tensors <= 100_000 or metadata > 100_000:
            raise ValueError("Invalid or oversized GGUF header counts.")
        keys = set()
        for _ in range(metadata):
            key = string(65535)
            if not key or key in keys:
                raise ValueError("Empty or duplicate GGUF metadata key.")
            keys.add(key)
            skip_value(integer("I"))
        names, total = set(), 0
        for _ in range(tensors):
            name, rank = string(1024), integer("I")
            if not name or name in names or not 1 <= rank <= 4:
                raise ValueError("Invalid or duplicate GGUF tensor descriptor.")
            names.add(name)
            dimensions = [integer("Q") for _ in range(rank)]
            if any(not 1 <= value <= 2**32 for value in dimensions):
                raise ValueError("Invalid GGUF tensor dimension.")
            elements = math.prod(dimensions)
            total += elements
            if total > 2**63 - 1:
                raise ValueError("GGUF tensor element count exceeds the supported bound.")
            integer("I")  # Quantization type affects bytes, not logical dimensions.
            integer("Q")  # Payload offset; no tensor payload is read.
        return total, tensors


def load_tts(args, checkpoint, distribution, sample_rate, *, verify_native=False):
    check_cpu_isa()
    elements, tensors = gguf_tensor_elements(checkpoint / "model.q8.gguf")
    print(f"TTS model size: {elements / 1e6:.2f}M parameters and buffers "
          f"({elements:,} elements in {tensors:,} GGUF tensors)")
    import ptts

    if (getattr(ptts, "__version__", None) != "0.4.0"
            or Path(ptts.__file__).resolve() != Path(distribution.locate_file("ptts/__init__.py")).resolve()):
        raise ValueError("Imported ptts differs from the recorded 0.4.0 runtime.")
    if verify_native:
        native = sys.modules.get("ptts._ptts")
        native_paths = {
            Path(distribution.locate_file(name)).resolve()
            for name in distribution.files or ()
            if name.parts[0] == "ptts" and name.suffix in (".so", ".pyd")
        }
        if (native is None or not getattr(native, "__file__", None)
                or Path(native.__file__).resolve() not in native_paths):
            raise ValueError("Loaded ptts native module differs from the recorded runtime.")
    ptts.set_num_threads(args.threads)
    if ptts.get_num_threads() != args.threads:
        raise ValueError("Runtime did not retain the requested thread count.")
    tts = ptts.TTS(model=str(checkpoint), device="cpu", quant="q8_0", voice=args.voice,
                   temperature=args.temperature, seed=args.seed, cfg_coef=None,
                   eos_threshold=args.eos_threshold, lang=args.split, rewrites=args.rewrites,
                   conditions=args.conditions)
    if (tts.device, tts.quant, tts.sample_rate) != ("cpu", "q8_0", sample_rate):
        raise ValueError("Loaded runtime differs from the requested CPU/checkpoint settings.")
    return tts


def validate_ttfa(payload, provenance, probe):
    """A completed, matching sidecar is reusable without another synthesis request."""
    streaming_mode = provenance["protocol"]["blocking"] is False
    full_rows = 50 if streaming_mode else 500
    expected = {
        "model_id": provenance["model_id"], "dataset_path": provenance["dataset"]["dataset_path"],
        "dataset": "zero_shot", "split": "en",
        "mode_suffix": "",
        "batch_size": 1, "n_samples": 500, "warmup_excluded": 3,
        "full_generation_rows": full_rows - 3, "early_stopped_rows": 500 - full_rows,
        "phonon_provenance": provenance,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing TTFA sidecar has different inputs or protocol.")
    rows = payload.get("samples", [])
    if len(rows) != 500:
        raise ValueError("TTFA requires all 500 completed measurements.")
    for index, row in enumerate(rows):
        full = index < full_rows
        if (row.get("index") != index or row.get("full_generation") is not full
                or row.get("first_reported") is not streaming_mode
                or any(not isinstance(row.get(key), (int, float))
                       or isinstance(row[key], bool)
                       or not math.isfinite(row[key]) or row[key] <= 0
                       for key in ("ttfa_ms", "gen_s", "audio_s"))
                or row["ttfa_ms"] > row["gen_s"] * 1000 + 1e-6):
            raise ValueError("TTFA sidecar contains an incomplete or invalid measurement.")
        fraction = row["ttfa_ms"] / (row["gen_s"] * 1000)
        recorded = row.get("ttfa_frac_of_gen")
        if ((full and (not isinstance(recorded, (int, float))
                       or isinstance(recorded, bool) or not math.isfinite(recorded)
                       or not math.isclose(recorded, fraction)))
                or (not full and recorded is not None)
                or (not streaming_mode and not math.isclose(fraction, 1.0))):
            raise ValueError("TTFA sidecar first-audio fractions differ from its measured rows.")
    warm = rows[3:]
    warm_full = [row for row in warm if row["full_generation"]]
    fractions = sorted(row["ttfa_frac_of_gen"] for row in warm_full)
    median = fractions[len(fractions) // 2]
    streaming = streaming_mode and median < probe.STREAMING_MAX_FRAC
    if (payload.get("ttfa_ms") != probe._summary([row["ttfa_ms"] for row in warm])
            or payload.get("gen_s") != probe._summary([row["gen_s"] for row in warm_full])
            or payload.get("streaming") is not streaming
            or payload.get("ttfa_frac_of_gen") != round(median, 3)
            or payload.get("first_reported_but_not_early") is not (streaming_mode and not streaming)
            or payload.get("rtfx_batch1") != round(
                sum(row["audio_s"] for row in warm_full) / sum(row["gen_s"] for row in warm_full), 2)):
        raise ValueError("TTFA sidecar summaries differ from its measured rows.")


def run_ttfa(args, checkpoint, distribution, hashes, runtime_hashes, quality_root):
    """One selected API path; no quality outputs, extra warmups, uploads or scoring."""
    import ttfa_probe as probe

    mode = getattr(args, "ttfa_mode", "blocking")
    if mode not in ("blocking", "streaming"):
        raise ValueError("Select blocking or streaming TTFA explicitly.")
    streaming_mode = mode == "streaming"
    if streaming_mode and (probe.FULL_GEN_SAMPLES != 50 or probe.WARMUP_DISCARD != 3):
        raise ValueError("Streaming TTFA requires the shared 50-full, three-discard protocol.")
    suffix = "__ptts-stream__cpu-upgrade" if streaming_mode else "__cpu-upgrade"
    if not getattr(args, "ttfa_output_root", None):
        raise ValueError("Supply a distinct private --ttfa_output_root.")
    output_root = Path(args.ttfa_output_root).expanduser().resolve()
    if any((parent / ".git").exists() for parent in (output_root, *output_root.parents)):
        raise ValueError("Keep TTFA output outside every Git checkout.")
    for other in (checkpoint, quality_root):
        if output_root == other or other in output_root.parents or output_root in other.parents:
            raise ValueError("TTFA, checkpoint and quality output roots must be separate.")
    if not any(Path(name).suffix in (".so", ".pyd") for name in runtime_hashes):
        raise ValueError("TTFA requires a recorded ptts native-library hash.")
    os.umask(0o077)
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output_root.stat().st_mode & 0o077:
        raise ValueError("Private TTFA output root must have mode 0700.")
    model_safe = args.model_id.replace("/", "-")
    output = output_root / "results" / model_safe
    for path in (output.parent, output):
        if path.is_symlink() or not path.resolve().is_relative_to(output_root):
            raise ValueError("TTFA paths must remain inside the private output root.")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    sidecar = output / probe.probe_filename(
        model_safe, args.dataset_path.replace("/", "-"), "zero_shot", "en", suffix,
    )
    provenance_path, lock = sidecar.with_suffix(".run.json"), sidecar.with_suffix(".lock")
    for path in (sidecar, provenance_path, lock):
        if path.is_symlink() or not path.resolve().is_relative_to(output_root):
            raise ValueError("TTFA files must remain inside the private output root.")

    from datasets import load_dataset

    dataset = load_dataset(args.dataset_path, name="zero_shot", split="en", revision=args.dataset_revision)
    if len(dataset) != 500:
        raise ValueError("The full pinned CV3 English split must contain exactly 500 rows.")
    if "id" not in dataset.column_names:
        dataset = dataset.add_column("id", list(range(len(dataset))))
    dataset = dataset.select_columns(["id", args.text_column])
    inventory = [{"id": str(row["id"]), "text": row[args.text_column]} for row in dataset]
    if (len({row["id"] for row in inventory}) != 500
            or any(not isinstance(row["text"], str) or not row["text"].strip() for row in inventory)):
        raise ValueError("CV3 must contain unique IDs and nonempty raw target text.")
    text_hash = hashlib.sha256(json.dumps(
        inventory, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    ).encode()).hexdigest()
    expected_text_hash = getattr(args, "ttfa_text_sha256", None)
    if expected_text_hash and expected_text_hash != text_hash:
        raise ValueError("CV3 raw ID/text inventory differs from its expected SHA256.")
    provenance = {
        "model_id": args.model_id, "checkpoint_files_sha256": hashes,
        "runtime_version": distribution.version, "runtime_files_sha256": runtime_hashes,
        "python_version": sys.version, "runner_sha256": file_hash(Path(__file__).resolve()),
        "shared_probe_sha256": file_hash(Path(probe.__file__).resolve()),
        "shared_helpers_sha256": file_hash(Path(manifest_entry.__code__.co_filename).resolve()),
        "dataset": {key: getattr(args, key) for key in
                    ("dataset_path", "dataset_revision", "dataset", "split", "text_column")},
        "ordered_id_text_sha256": text_hash,
        "settings": {key: getattr(args, key) for key in
                     ("voice", "temperature", "seed", "eos_threshold", "rewrites", "conditions", "threads")},
        "thread_pools": {key: os.environ.get(key) for key in
                         ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "device": "cpu", "flavor": "cpu-upgrade", "quant": "q8_0", "sample_rate": 24000,
        "protocol": {"samples": 500, "batch_size": 1, "warmup_excluded": 3,
                     "additional_warmup_calls": 0, "blocking": not streaming_mode},
        "official_h200_rtfx": False,
    }
    if streaming_mode:
        provenance["protocol"].update(streaming_api="ptts.TTS.stream", full_generation_samples=50)
    if sidecar.exists():
        if not provenance_path.exists() or json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Completed TTFA provenance differs from this request.")
        validate_ttfa(json.loads(sidecar.read_text()), provenance, probe)
        print(f"Reusing completed TTFA measurements: {sidecar}")
        return sidecar
    if lock.exists() or provenance_path.exists():
        raise FileExistsError("Earlier TTFA attempt is incomplete or uncertain; inspect it before any retry.")
    with lock.open("x") as attempt:
        attempt.write(str(os.getpid()))
    with provenance_path.open("x") as target:
        json.dump(provenance, target, indent=2, sort_keys=True)
    tts = load_tts(args, checkpoint, distribution, 24000, verify_native=True)

    def generate(job):
        audio = tts.synth(job["text"], voice=args.voice, temperature=args.temperature,
                          seed=args.seed, cfg_coef=None)
        view = memoryview(audio)
        if ((view.ndim, view.itemsize, view.format) not in ((1, 4, "f"), (1, 4, "=f"))
                or not len(view)):
            raise ValueError("Expected completed mono float32 audio.")
        # The worker's smoke checks finite samples; do not scan audio inside the TTFA clock.
        return audio, 24000, None

    def generate_stream(job, early_stop):
        samples, first = 0, None
        with tts.stream(job["text"], voice=args.voice, temperature=args.temperature,
                        seed=args.seed, cfg_coef=None) as stream:
            if stream.sample_rate != 24000:
                raise ValueError("Expected 24 kHz streaming audio.")
            for chunk in stream:
                view = memoryview(chunk)
                if (view.ndim, view.itemsize, view.format) not in ((1, 4, "f"), (1, 4, "=f")):
                    raise ValueError("Expected mono float32 streaming chunks.")
                if not len(view):
                    continue
                if first is None:
                    first = time.perf_counter()
                samples += len(view)
                if early_stop:
                    break
        if first is None:
            raise ValueError("Streaming generation returned no audio.")
        # Count samples without concatenating or scanning PCM within the clock.
        return samples / 24000, 24000, first

    payload = probe.run_probe(
        generate_stream if streaming_mode else generate, inventory,
        model_id=args.model_id, dataset_path=args.dataset_path,
        dataset_config="zero_shot", split="en", out_dir=str(output), n=500,
        mode_suffix=suffix,
        # Hardware/engine tags belong to the filename; the publication script uses
        # mode_suffix only for a model variant or voice-clone mode.
        extra={"mode_suffix": "", "device": "cpu", "flavor": "cpu-upgrade",
               "phonon_provenance": provenance},
    )
    validate_ttfa(payload, provenance, probe)
    with sidecar.open("rb") as target:
        os.fsync(target.fileno())
    lock.unlink()
    return sidecar


def load_clone_voices(args):
    """This split's voice files and row index, downloaded at the pinned revision."""
    from huggingface_hub import hf_hub_download, snapshot_download

    revision = args.clone_voices_revision or CLONE_VOICES_REVISION
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Supply an immutable clone voices revision.")
    split = f"{args.dataset_path.split('/')[-1]}/{args.dataset}/{args.split}"
    index_path = hf_hub_download(args.clone_voices_repo, "index.json", repo_type="dataset", revision=revision)
    keys = json.loads(Path(index_path).read_text()).get(split)
    if not keys:
        raise ValueError(f"The clone voices have no rows for {split}.")
    root = snapshot_download(args.clone_voices_repo, repo_type="dataset", revision=revision,
                             allow_patterns=[f"voices/{key}.safetensors" for key in sorted(set(keys.values()))])
    return SimpleNamespace(keys=keys, root=Path(root), provenance={
        "repo": args.clone_voices_repo, "revision": revision, "index_sha256": file_hash(index_path)})


def main(args):
    if args.voice_clone and (args.ttfa_probe or args.voice != "default"):
        raise ValueError("Voice cloning replaces the preset voice and has no TTFA path.")
    if args.device != "cpu" or args.split not in ("en", "fr", "de", "es"):
        raise ValueError("This q8_0 path supports English, French, German and Spanish on CPU only.")
    if args.ttfa_probe and args.split != "en":
        raise ValueError("TTFA measures the English split only; A100/CUDA TTFA is unavailable.")
    if getattr(args, "ttfa_mode", "blocking") not in ("blocking", "streaming"):
        raise ValueError("Select blocking or streaming TTFA explicitly.")
    if args.ttfa_probe and (getattr(args, "ttfa_flavor", "cpu-upgrade") != "cpu-upgrade"
                           or args.ttfa_probe not in (-1, 500) or args.dataset != "zero_shot"
                           or args.max_eval_samples not in (None, -1, 500)):
        raise ValueError("TTFA requires cpu-upgrade and all 500 CV3 zero_shot/en samples.")
    if args.threads < 1 or args.warmup_steps < 0 or not 0 <= args.seed < 2**64:
        raise ValueError("Invalid thread, warmup or seed setting.")
    if not all(math.isfinite(v) and v >= 0 for v in (args.temperature, args.eos_threshold)):
        raise ValueError("Temperature and EOS threshold must be finite and nonnegative.")
    for value in (args.model_id, args.dataset_path):
        if (not value or not re.fullmatch(r"[\w.-]+/[\w.-]+", value)
                or any(part in (".", "..") for part in value.split("/"))):
            raise ValueError("Use namespace/name model labels and Hub datasets.")
    if Path(args.dataset_path).exists():
        raise ValueError("Local datasets cannot satisfy an immutable Hub revision pin.")
    if args.dataset not in ("tts", "zero_shot") or not re.fullmatch(r"[\w-]+", args.voice):
        raise ValueError("Select a standard dataset configuration and an explicit preset name.")

    if not args.dataset_revision:
        from huggingface_hub import HfApi

        args.dataset_revision = HfApi().dataset_info(args.dataset_path).sha
    if not re.fullmatch(r"[0-9a-f]{40}", args.dataset_revision):
        raise ValueError("Supply the full immutable dataset revision.")
    if not args.checkpoint_path:
        from huggingface_hub import snapshot_download

        revision = getattr(args, "model_revision", None) or MODEL_REVISIONS.get(args.model_id)
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Supply an immutable model revision for an unlisted checkpoint.")
        args.checkpoint_path = snapshot_download(
            args.model_id, revision=revision,
            allow_patterns=["config.json", "tokenizer.json", "model.q8.gguf",
                            "default-voice.safetensors", "voices/*.safetensors"],
        )
    checkpoint = Path(args.checkpoint_path).expanduser().resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError("Checkpoint must be an existing local directory.")
    voice_file = "default-voice.safetensors" if args.voice == "default" else f"voices/{args.voice}.safetensors"
    names = ("config.json", "tokenizer.json", "model.q8.gguf", voice_file)
    hashes = {name: file_hash(checkpoint / name) for name in names}
    config = json.loads((checkpoint / "config.json").read_text())
    mimi = config["mimi"]
    if mimi["channels"] != 1 or mimi["sample_rate"] != 24000 or config["lsd_decode_steps"] != 1:
        raise ValueError("Expected the mono 24 kHz, one-decode-step checkpoint layout.")
    conditions = {row["name"]: row for row in config["conditioners"]}
    if set(args.conditions) != set(conditions):
        raise ValueError("Conditions must match the checkpoint's declared names.")
    for name, value in args.conditions.items():
        condition = conditions[name]
        if condition["type"] == "lut":
            valid = value in condition["lut"]["possible_values"]
        elif condition["type"] == "continuous":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        else:
            valid = False
        if not valid:
            raise ValueError("Conditions must match the checkpoint's declared values.")
    distribution = importlib.metadata.distribution("ptts")
    if distribution.version != "0.4.0":
        raise ValueError("Install the compatible ptts 0.4.0 runtime separately.")
    runtime_hashes = {
        str(name): file_hash(distribution.locate_file(name))
        for name in distribution.files or ()
        if name.parts[0] == "ptts" and name.suffix in (".py", ".pyi", ".so", ".pyd")
    }

    output_root = Path(args.output_root).expanduser().resolve()
    if any((parent / ".git").exists() for parent in (output_root, *output_root.parents)):
        raise ValueError("Keep private output outside every Git checkout.")
    if (output_root == checkpoint or checkpoint in output_root.parents
            or output_root in checkpoint.parents):
        raise ValueError("Output must be separate from the checkpoint.")
    if args.ttfa_probe:
        return run_ttfa(args, checkpoint, distribution, hashes, runtime_hashes, output_root)
    os.umask(0o077)
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output_root.stat().st_mode & 0o077:
        raise ValueError("Private output root must have mode 0700.")
    for path in (output_root / "results", output_root / "results" / args.model_id.replace("/", "-")):
        if path.is_symlink():
            raise ValueError("Existing output directories must not be symlinks.")
    clone = load_clone_voices(args) if args.voice_clone else None
    os.chdir(output_root)
    paths = output_paths(args, "_voice_clone" if clone else "")
    manifest_path = Path(paths.manifest_path)
    provenance_path = manifest_path.with_suffix(".run.json")
    attempt_path = manifest_path.with_suffix(".lock")
    for path in (Path(paths.model_dir), Path(paths.output_dir), manifest_path, provenance_path, attempt_path):
        if path.is_symlink() or not path.resolve().is_relative_to(output_root):
            raise ValueError("Output paths must stay inside the private output root.")
    provenance = {
        "model_id": args.model_id, "checkpoint_files_sha256": hashes,
        "runtime_version": distribution.version, "runtime_files_sha256": runtime_hashes,
        "runner_sha256": file_hash(Path(__file__).resolve()),
        "dataset": {key: getattr(args, key) for key in
                    ("dataset_path", "dataset_revision", "dataset", "split", "text_column")},
        "settings": {key: getattr(args, key) for key in
                     ("voice", "temperature", "seed", "eos_threshold", "rewrites", "conditions", "threads")},
        "device": "cpu", "quant": "q8_0", "official_h200_rtfx": False,
    }
    if clone:
        provenance["clone_voices"] = clone.provenance
    if manifest_path.exists() and not args.resume:
        raise FileExistsError("Existing manifest: inspect it before explicitly resuming.")
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Existing output has different generation inputs.")
    elif manifest_path.exists():
        raise ValueError("Cannot resume a manifest without matching provenance.")
    else:
        with provenance_path.open("x") as output:
            json.dump(provenance, output, indent=2, sort_keys=True)

    # The common loader currently omits revision; keep this one pinned load inline.
    from datasets import load_dataset

    if Path(args.dataset_path).exists():
        raise ValueError("Local datasets cannot satisfy an immutable Hub revision pin.")
    dataset = load_dataset(args.dataset_path, name=args.dataset, split=args.split,
                           revision=args.dataset_revision)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    if "id" not in dataset.column_names:
        dataset = dataset.add_column("id", list(range(len(dataset))))
    dataset = dataset.select_columns(["id", args.text_column] + (["prompt_audio"] if clone else []))
    if clone:
        from datasets import Audio

        # The reference clip's own bytes: hashed against its voice file, and kept for SIM.
        dataset = dataset.cast_column("prompt_audio", Audio(decode=False))
    if not len(dataset):
        raise ValueError("The selected dataset contains no samples.")
    done = load_done_entries(str(manifest_path), args.resume)
    if manifest_path.exists():
        entries = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        if len(entries) != len(done):
            raise ValueError("Existing manifest contains duplicate or invalid paths.")
    rows, prompts = {}, {}
    for row in dataset:
        sample_id, text = str(row["id"]), row[args.text_column]
        if "/" in sample_id or "\\" in sample_id or not isinstance(text, str) or not text.strip():
            raise ValueError("Expected safe sample IDs and nonempty raw text.")
        relative = wav_rel_path(paths.dataset_dir_name, sample_id)
        if relative in rows:
            raise ValueError("Dataset IDs must be unique.")
        rows[relative] = text
        if clone:
            clip = row["prompt_audio"]["bytes"]
            if not clip or clone.keys.get(sample_id) != hashlib.sha256(clip).hexdigest():
                raise ValueError(f"Row {sample_id}'s reference clip is not the one its voice was computed from.")
            prompts[relative] = (os.path.join(paths.dataset_dir_name, "prompts", f"prompt_{sample_id}.wav"),
                                 clip, clone.keys[sample_id])
    if not set(done) <= set(rows):
        raise ValueError("Existing manifest differs from the selected dataset rows.")
    for relative, entry in done.items():
        if entry["text"] != rows[relative] or file_hash(Path(paths.model_dir) / relative) != entry["audio_sha256"]:
            raise ValueError("Resume reference or audio differs from the completed entry.")
    pending = [(relative, text) for relative, text in rows.items() if relative not in done]
    if any((Path(paths.model_dir) / relative).exists() for relative, _ in pending):
        raise FileExistsError("Unrecorded WAV exists; inspect the interrupted attempt before resuming.")
    if attempt_path.exists():
        raise FileExistsError("An earlier attempt is active or uncertain; inspect its lock before resuming.")
    if not pending:
        print("All selected samples are already complete; no model loaded.")
        return

    # Exclusive admission; failures retain this marker for explicit inspection.
    with attempt_path.open("x") as attempt:
        attempt.write(str(os.getpid()))

    tts = load_tts(args, checkpoint, distribution, mimi["sample_rate"])
    registered = set()

    def voice_for(relative):
        """The preset voice, or this row's reference voice (registered on first use, untimed)."""
        if not clone:
            return args.voice
        key = prompts[relative][2]
        if key not in registered:
            tts.add_voice(key, str(clone.root / "voices" / f"{key}.safetensors"))
            registered.add(key)
        return key

    def synth_one(text, voice):
        start = time.perf_counter()
        audio = tts.synth(text, voice=voice, temperature=args.temperature,
                          seed=args.seed, cfg_coef=None)
        elapsed = time.perf_counter() - start
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Expected positive, finite generation time.")
        return audio, elapsed

    # Repeat the first text for untimed warmups; continuation reuses prior warmups.
    for _ in range(0 if done else args.warmup_steps):
        synth_one(pending[0][1], voice_for(pending[0][0]))
    with open_manifest(str(manifest_path), done) as manifest:
        for relative, text in pending:
            audio, elapsed = synth_one(text, voice_for(relative))  # raw text, including brackets/whitespace
            target = Path(paths.model_dir) / relative
            duration = write_audio(target, audio, tts.sample_rate)
            prompt_path = None
            if clone:
                # SIM compares the output with this reference clip, written unchanged.
                prompt_path, clip, _ = prompts[relative]
                (Path(paths.model_dir) / prompt_path).parent.mkdir(exist_ok=True)
                (Path(paths.model_dir) / prompt_path).write_bytes(clip)
            entry = manifest_entry(relative, duration, elapsed, text, prompt_audio_filepath=prompt_path)
            entry.update(device="cpu", cpu_generation_time_s=elapsed, official_h200_rtfx=False,
                         timing_scope="generation_complete", audio_sha256=file_hash(target))
            write_entry(manifest, entry)
            manifest.flush()
            os.fsync(manifest.fileno())
    attempt_path.unlink()
    print("Stage 1 complete; CPU timing is diagnostic, not official H200 RTFx.")
    print_next_steps(voice_clone=bool(clone))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", required=True, help="Public model repository, also used as the output label.")
    parser.add_argument("--model_revision", help="Immutable model commit; defaults to the evaluated release.")
    parser.add_argument("--checkpoint_path", help="Optional existing local q8 checkpoint directory.")
    parser.add_argument("--voice", default="default", help="Bundled default voice, or a local preset name.")
    parser.add_argument("--output_root", required=True, help="Private directory outside Git; results/ is created within it.")
    parser.add_argument("--dataset_revision", help="Immutable dataset commit; resolves current Hub revision if omitted.")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--eos_threshold", type=float, default=4.0)
    parser.add_argument("--conditions", type=json.loads, default={"num_speakers": "1", "duration_delta": 0.0})
    parser.add_argument("--rewrites", choices=("default", "all", "none"), default="default")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ttfa_output_root", help="Separate private root for TTFA sidecars only.")
    parser.add_argument("--ttfa_flavor", choices=("cpu-upgrade", "a100"), default="cpu-upgrade",
                        help="Only cpu-upgrade is supported by this local q8_0 path.")
    parser.add_argument("--ttfa_mode", choices=("blocking", "streaming"), default="blocking",
                        help="TTFA API path; streaming measures first chunks through ptts.TTS.stream.")
    parser.add_argument("--ttfa_text_sha256", help="Optional expected ordered CV3 ID/raw-text SHA256.")
    parser.add_argument("--clone_voices_repo", default=CLONE_VOICES_REPO, help="Dataset repo of reference voice files.")
    parser.add_argument("--clone_voices_revision", help="Immutable commit of it; defaults to the evaluated one.")
    add_common_args(parser, voice_clone=False)
    parser.set_defaults(device="cpu", dataset_path=None, seed=4242424242424242)
    main(parser.parse_args())
