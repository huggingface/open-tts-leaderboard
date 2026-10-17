"""Fish Audio Drama 3 generation and streaming TTFA via pinned local SGLang Omni.

Preview: bash fishaudio-drama-3/submit_jobs.sh
The launcher supports an explicit immutable image/bootstrap or a compatible public
Space image; --help lists bounded generation options. ASR/SIM use the shared
helpers, and scripts/open_results_pr.py handles result submission separately.
"""

import argparse
import array
import base64
import binascii
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_eval_utils import (
    add_common_args,
    load_done_entries,
    load_tts_dataset,
    manifest_entry,
    open_manifest,
    output_paths,
    print_next_steps,
    wav_rel_path,
    write_entry,
)

MODEL_ID = "fishaudio/drama-3"
MODEL_REVISION = "e0a9218aa48888270f06ef1ed9ea67622c0e17d2"
SERVER_REVISION = "12f65fa60fbb5217ad47f58b2e58b18092c818a1"
CODEC_ID = "fishaudio/s2-pro"
CODEC_REVISION = "1de9996b6be38b745688de084d87a5633f714e4e"
OUTPUT_HANDLING_POLICY = "retain_valid_audio_on_stop_or_length_v1"


@dataclass(frozen=True)
class SynthesisResult:
    audio: bytes
    finish_reason: str

    @property
    def generation_token_limit_reached(self):
        return self.finish_reason == "length"


@dataclass(frozen=True)
class StreamResult:
    duration: float
    first_audio_timestamp: float
    response_chunks: int
    finish_reason: str | None
    early_stopped: bool = False

    @property
    def generation_token_limit_reached(self):
        # The terminal reason is deliberately not observed after a first-frame abort.
        return None if self.early_stopped else self.finish_reason == "length"


def validate_finish_reason(reason):
    if reason not in ("stop", "length"):
        raise RuntimeError(f"Drama 3 returned a missing or unsupported terminal reason: {reason!r}")
    return reason


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 30
    max_new_tokens: int = 2048

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 1 <= self.top_k <= 2147483647 or not 1 <= self.max_new_tokens <= 2048:
            raise ValueError("Expected positive int32 top_k and max_new_tokens within the server's 2048 cap")


def speech_payload(text, reference=None, *, sampling=None, streaming=False):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Synthesis needs nonempty text")
    payload = {"model": MODEL_ID, "input": text, **asdict(sampling or Sampling())}
    if reference is not None:
        payload["references"] = reference
    # Drama 3 rejects seed, language and instructions. Do not inherit the S2/Higgs fields.
    if streaming:
        payload.update(stream=True, response_format="pcm", stream_format="sse")
    else:
        payload.update(stream=False, response_format="wav")
    return payload


def wav_duration(data):
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            frames, rate = audio.getnframes(), audio.getframerate()
            expected = frames * audio.getnchannels() * audio.getsampwidth()
            if frames <= 0 or rate <= 0 or len(audio.readframes(frames)) != expected:
                raise ValueError("Empty or truncated WAV response")
            return frames / rate
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid PCM WAV response") from exc


class DramaClient:
    def __init__(self, host="127.0.0.1", port=8000, timeout=300, sampling=None):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("This benchmark expects a local Drama 3 server")
        if not 1 <= port <= 65535 or timeout <= 0:
            raise ValueError("Invalid server port or request timeout")
        self.base_url = f"http://{'[::1]' if host == '::1' else host}:{port}"
        self.timeout = timeout
        self.sampling = sampling or Sampling()
        # Never send loopback benchmark references through an HTTP proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _open(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        request = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            return self.opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            raise RuntimeError(f"Drama 3 HTTP request failed ({status}); no retry was sent") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RuntimeError("Drama 3 server connection failed; no retry was sent") from None

    def wait_for_server(self, timeout=1200):
        deadline = time.monotonic() + timeout
        while True:
            try:
                with self._open("/health") as response:
                    ready = response.status == 200
            except RuntimeError:
                ready = False
            if ready:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Drama 3 server did not become healthy before timeout")
            time.sleep(3)
        with self._open("/v1/models") as response:
            models = json.load(response)
        if [item.get("id") for item in models.get("data", [])] != [MODEL_ID]:
            raise RuntimeError("Server model identity is not fishaudio/drama-3")

    def synthesize(self, text, reference=None):
        """Compatibility API returning audio bytes without discarding capped outputs."""
        return self.synthesize_result(text, reference).audio

    def synthesize_result(self, text, reference=None):
        """Keep one valid response, including explicit token-limit termination.

        A length termination is a model outcome, not a transport failure. Return its
        audio once and carry the reason with it so callers can score and flag it.
        """
        payload = speech_payload(text, reference, sampling=self.sampling)
        with self._open("/v1/audio/speech", payload) as response:
            reason = validate_finish_reason(response.headers.get("X-Finish-Reason"))
            data = response.read()
        wav_duration(data)
        return SynthesisResult(data, reason)

    def stream(self, text, reference=None, early_stop=False):
        """Compatibility API returning duration, first timestamp and event count."""
        result = self.stream_result(text, reference, early_stop=early_stop)
        return result.duration, result.first_audio_timestamp, result.response_chunks

    def stream_result(self, text, reference=None, early_stop=False):
        """Return audio measurements and an observed stop/length terminal reason.

        The pinned server's SSE endpoint provides an explicit terminal reason. A raw
        PCM EOF alone cannot distinguish a completed utterance from a failed stream.
        An intentional first-frame abort leaves the terminal reason unknown.
        There are no fallback requests or automatic retries.
        """
        payload = speech_payload(text, reference, sampling=self.sampling, streaming=True)
        with self._open("/v1/audio/speech", payload) as response:
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise RuntimeError("Expected the pinned SGLang Omni SSE speech protocol")
            try:
                rate = int(response.headers["X-Sample-Rate"])
                channels = int(response.headers["X-Channels"])
                bits = int(response.headers["X-Bit-Depth"])
            except (KeyError, TypeError, ValueError):
                raise RuntimeError("Missing PCM format headers") from None
            if rate <= 0 or channels != 1 or bits != 16:
                raise RuntimeError("Expected positive-rate mono PCM16")
            pending = bytearray()
            data_lines = []
            total = chunks = 0
            first = None
            # read1 returns available bytes without waiting to fill a large read buffer.
            while True:
                block = response.read1(65536)
                if not block:
                    break
                pending.extend(block)
                while b"\n" in pending:
                    line, _, rest = pending.partition(b"\n")
                    pending = bytearray(rest)
                    line = line.rstrip(b"\r")
                    if line.startswith(b"data:"):
                        data_lines.append(line[5:].lstrip(b" "))
                    elif not line and data_lines:
                        try:
                            event = json.loads(b"\n".join(data_lines))
                        except (ValueError, UnicodeDecodeError):
                            raise RuntimeError("Malformed speech stream event") from None
                        data_lines = []
                        if not isinstance(event, dict):
                            raise RuntimeError("Malformed speech stream event")
                        if event.get("type") == "speech.audio.delta":
                            try:
                                chunk = base64.b64decode(event["audio"], validate=True)
                            except (KeyError, ValueError, TypeError, binascii.Error):
                                raise RuntimeError("Invalid base64 speech audio") from None
                            if chunk:
                                chunks += 1
                                total += len(chunk)
                            if first is None and total >= 2:
                                first = time.perf_counter()
                                if early_stop:
                                    return StreamResult((total // 2) / rate, first, chunks, None, True)
                        elif event.get("type") == "speech.audio.done":
                            reason = validate_finish_reason(event.get("finish_reason"))
                            if not total or total % 2:
                                raise RuntimeError("Empty or incomplete PCM speech stream")
                            return StreamResult(total / 2 / rate, first, chunks, reason)
                        elif event.get("type") == "error":
                            raise RuntimeError("Drama 3 speech stream returned an error")
            raise RuntimeError("Speech stream ended without a completed terminal event")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_dataset_source(args):
    """Return loader arguments and provenance, retaining the canonical output ID."""
    local_root = getattr(args, "dataset_local_root", None)
    if not local_root:
        return args, None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/(seed_tts_eval|cv3_eval)", args.dataset_path):
        raise ValueError("Private mounted inputs require an explicit canonical owner/dataset ID")
    key = args.dataset_path.split("/")[1]
    expected_config = "tts" if key == "seed_tts_eval" else "zero_shot"
    if args.dataset != expected_config:
        raise ValueError("Private input benchmark and config differ")
    root = Path(local_root).resolve(strict=True)
    provenance_path = root / "input_provenance.json"
    raw_provenance = provenance_path.read_bytes()
    provenance = json.loads(raw_provenance)
    if provenance.get("format_version") != 1:
        raise ValueError("Unsupported private input provenance version")
    dataset = provenance["datasets"][key]
    split_key = f"{args.dataset}/{args.split}"
    entry = dataset["splits"][split_key]
    expected_rows = ({"en": 1088, "zh": 2020}.get(args.split)
                     if key == "seed_tts_eval" else
                     (500 if args.split in ("en", "zh", "fr", "ja", "ko", "de", "es", "it", "ru") else None))
    if expected_rows is None or entry.get("rows") != expected_rows:
        raise ValueError("Private input row count differs from the full canonical split")
    relative = Path(entry["parquet_file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Private input Parquet path must remain inside the mount")
    parquet = (root / relative).resolve(strict=True)
    dataset_root = (root / key).resolve(strict=True)
    if not dataset_root.is_relative_to(root) or not parquet.is_relative_to(dataset_root):
        raise ValueError("Private input Parquet path escapes its benchmark directory")
    if file_sha256(dataset_root / "README.md") != dataset.get("readme_sha256"):
        raise ValueError("Private input config mapping differs from preparation provenance")
    for field in ("parquet_sha256", "ordered_id_text_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get(field, "")):
            raise ValueError(f"Private input {field} must be an exact SHA256")
    if file_sha256(parquet) != entry["parquet_sha256"]:
        raise ValueError("Private input Parquet checksum differs from preparation provenance")
    loader_args = SimpleNamespace(**vars(args))
    loader_args.dataset_path = str(dataset_root)
    identity = {"type": "private_bucket_parquet", "dataset": key, "split": split_key,
                "provenance_sha256": hashlib.sha256(raw_provenance).hexdigest(),
                "prepared_split": entry, "source": dataset.get("source")}
    return loader_args, identity


def ordered_text_sha256(dataset, text_column):
    digest = hashlib.sha256()
    for row in dataset:
        digest.update(json.dumps([str(row["id"]), row[text_column]], ensure_ascii=False).encode() + b"\n")
    return digest.hexdigest()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model_id", default=MODEL_ID, choices=[MODEL_ID])
    result.add_argument("--model_revision", default=MODEL_REVISION, choices=[MODEL_REVISION])
    result.add_argument("--server_revision", default=SERVER_REVISION, choices=[SERVER_REVISION])
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8000)
    result.add_argument("--server_timeout", type=int, default=1200)
    result.add_argument("--request_timeout", type=int, default=300)
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--top_p", type=float, default=0.9)
    result.add_argument("--top_k", type=int, default=30)
    result.add_argument("--max_new_tokens", type=int, default=2048)
    result.add_argument("--dataset_local_root", help="Verified private Parquet mount; keeps dataset_path as benchmark ID")
    add_common_args(result, batch_size=1, voice_clone=False)
    return result


def write_prompt(path, audio):
    """Persist a mono PCM16 reference readable by both the local server and SIM."""
    values = audio["array"]
    rate = int(audio["sampling_rate"])
    if rate <= 0 or not len(values):
        raise ValueError("Reference audio must be nonempty with a positive sample rate")
    if len(values) / rate > 60:
        raise ValueError("Drama 3 reference audio exceeds the server's 60 second limit")
    pcm = array.array("h")
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Reference audio contains non-finite samples")
        pcm.append(round(max(-1.0, min(1.0, value)) * 32767))
    if sys.byteorder != "little":
        pcm.byteswap()
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(pcm.tobytes())


def prepare_reference(row, paths, voice_clone):
    if not voice_clone:
        return None, None
    relative = os.path.join(paths.dataset_dir_name, "prompts", f"prompt_{row['id']}.wav")
    path = Path(paths.model_dir, relative).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_prompt(path, row["prompt_audio"])
    return relative, [{"audio_path": str(path), "text": row["prompt_text"] or ""}]


def normalizer_identity():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "normalizer").glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    # HF jobs inject the backend without the Git checkout or normalizer files.
    # Preserve the launcher's source identity without claiming to run normalization.
    source_identity = os.environ.get("OPEN_TTS_SOURCE_IDENTITY")
    if not files and source_identity:
        identity = json.loads(source_identity)
        identity["note"] = "Source checkout provenance only; generation does not run normalization."
        return identity

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"checkout_head": git("rev-parse", "HEAD"),
            "committed_tree": git("rev-parse", "HEAD:normalizer"),
            "normalizer_file_sha256": (hashlib.sha256((root / "normalizer/normalizer.py").read_bytes()).hexdigest()
                                       if (root / "normalizer/normalizer.py").is_file() else None),
            "python_files_sha256": digest.hexdigest() if files else None,
            "note": "Generation does not use the normalizer; scoring revision must be recorded separately."}


def corpus_hash(dataset, args):
    digest = hashlib.sha256()
    for row in dataset:
        record = [str(row["id"]), row[args.text_column]]
        if args.voice_clone:
            record.append(row["prompt_text"])
            audio = row["prompt_audio"]
            record.append(int(audio["sampling_rate"]))
            # Hash the decoded reference without retaining private audio/text in metadata.
            samples = array.array("d", (float(x) for x in audio["array"]))
            if sys.byteorder != "little":
                samples.byteswap()
            record.append(hashlib.sha256(samples.tobytes()).hexdigest())
        digest.update(json.dumps(record, ensure_ascii=False).encode() + b"\n")
    return digest.hexdigest()


def write_metadata(path, payload):
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
    os.replace(temp, path)


def print_generation_progress(completed, expected, capped):
    print(f"Generation progress: completed={completed}/{expected}; token_limit_reached={capped}", flush=True)


def main(args):
    if args.batch_size <= 0 or args.warmup_steps < 0 or args.server_timeout <= 0:
        raise ValueError("batch_size/server_timeout must be positive; warmup_steps must be nonnegative")
    sampling = Sampling(args.temperature, args.top_p, args.top_k, args.max_new_tokens)
    paths = output_paths(args, "_voice_clone" if args.voice_clone else "")
    loader_args, input_source = resolve_dataset_source(args)
    dataset = load_tts_dataset(loader_args, ("prompt_text", "prompt_audio") if args.voice_clone else ())
    if not len(dataset):
        raise ValueError("No evaluation samples")
    if input_source:
        prepared = input_source["prepared_split"]
        expected = min(prepared["rows"], args.max_eval_samples) if args.max_eval_samples and args.max_eval_samples > 0 else prepared["rows"]
        if len(dataset) != expected:
            raise ValueError("Loaded private input row count differs from preparation provenance")
        if expected == prepared["rows"] and ordered_text_sha256(dataset, args.text_column) != prepared["ordered_id_text_sha256"]:
            raise ValueError("Loaded private input ID/text order differs from preparation provenance")
    ids = [str(row["id"]) for row in dataset]
    if len(set(ids)) != len(ids) or any("/" in sid or "\\" in sid or sid in (".", "..") for sid in ids):
        raise ValueError("Sample IDs must be unique filenames")
    client = DramaClient(args.host, args.port, args.request_timeout, sampling)

    if args.ttfa_probe != 0:
        from ttfa_probe import probe_filename, run_probe, sample_indices
        suffix = "_voice_clone" if args.voice_clone else ""
        name = probe_filename(paths.model_safe, args.dataset_path.replace("/", "-"), args.dataset, args.split, suffix)
        if Path(paths.model_dir, name).exists():
            raise ValueError("TTFA sidecar already exists; use a separate results directory for a new probe")
        client.wait_for_server(args.server_timeout)
        # Stay beneath the server's allowed results root, but never overwrite stage-1 references.
        with tempfile.TemporaryDirectory(dir=paths.model_dir, prefix=".ttfa_refs_") as probe_dir:
            probe_paths = SimpleNamespace(model_dir=paths.model_dir, dataset_dir_name=Path(probe_dir).name)
            jobs = []
            for index in sample_indices(args.ttfa_probe, len(dataset)):
                row = dataset[index]
                _, reference = prepare_reference(row, probe_paths, args.voice_clone)
                jobs.append((row[args.text_column], reference))
            chunks = []
            outcomes = []

            def generate(job, early_stop):
                result = client.stream_result(*job, early_stop=early_stop)
                chunks.append(result.response_chunks)
                outcomes.append({"finish_reason": result.finish_reason,
                                 "generation_token_limit_reached": result.generation_token_limit_reached,
                                 "early_stopped": result.early_stopped})
                return result.duration, None, result.first_audio_timestamp

            payload = run_probe(
                generate, jobs, model_id=args.model_id, dataset_path=args.dataset_path,
                dataset_config=args.dataset, split=args.split, out_dir=paths.model_dir,
                mode_suffix=suffix, n=args.ttfa_probe,
                extra={"engine": "sglang", "model_revision": args.model_revision,
                       "server_revision": args.server_revision, "codec_revision": CODEC_REVISION,
                       "sampling": asdict(sampling), "request_seed_supported": False,
                       "output_handling_policy": OUTPUT_HANDLING_POLICY,
                       "dataset_source": input_source,
                       "response_chunks": chunks, "batch_size": 1},
            )
            for row, outcome in zip(payload["samples"], outcomes):
                row.update(outcome)
            payload["finish_reason_counts"] = {
                reason: sum(outcome["finish_reason"] == reason for outcome in outcomes)
                for reason in ("stop", "length")
            }
            payload["generation_token_limit_reached_samples"] = payload["finish_reason_counts"]["length"]
            payload["terminal_reason_unobserved_samples"] = sum(outcome["early_stopped"] for outcome in outcomes)
            write_metadata(str(Path(paths.model_dir, name)), payload)
        return

    metadata_path = paths.manifest_path.removesuffix(".jsonl") + ".run.json"
    settings = {
        "model_id": args.model_id, "model_revision": args.model_revision,
        "server_revision": args.server_revision, "codec_id": CODEC_ID, "codec_revision": CODEC_REVISION,
        "dataset_path": args.dataset_path, "dataset": args.dataset, "split": args.split,
        "text_column": args.text_column, "voice_clone": args.voice_clone,
        "batch_size": args.batch_size, "sampling": asdict(sampling),
        "output_handling_policy": OUTPUT_HANDLING_POLICY,
        "request_seed_supported": False, "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "input_sha256": corpus_hash(dataset, args), "expected_samples": len(dataset),
        "max_eval_samples": args.max_eval_samples,
        "dataset_source": input_source,
    }
    previous = None
    if os.path.exists(paths.manifest_path) or os.path.exists(metadata_path):
        if not args.resume:
            raise ValueError("Results already exist; use --resume with identical settings or a separate results directory")
        if not os.path.exists(metadata_path):
            raise ValueError("Cannot resume without the run metadata")
        with open(metadata_path, encoding="utf-8") as source:
            previous = json.load(source)
        if previous["settings"] != settings:
            raise ValueError("Resume settings or input corpus differ from the recorded run")
        if previous.get("completed_samples", 0) and not os.path.exists(paths.manifest_path):
            raise ValueError("Recorded completed samples are missing their manifest")
    done = load_done_entries(paths.manifest_path, args.resume)
    if os.path.exists(paths.manifest_path):
        with open(paths.manifest_path, encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        if len(rows) != len(done):
            raise ValueError("Manifest contains duplicate or invalid rows")
    expected = {wav_rel_path(paths.dataset_dir_name, sid) for sid in ids}
    if not set(done) <= expected:
        raise ValueError("Manifest contains samples outside this corpus")
    finish_reason_counts = {"stop": 0, "length": 0}
    for relative, entry in done.items():
        reason = entry.get("finish_reason")
        if reason not in finish_reason_counts or entry.get("generation_token_limit_reached") is not (reason == "length"):
            raise ValueError("Completed sample is missing consistent generation termination metadata")
        finish_reason_counts[reason] += 1
        duration = wav_duration(Path(paths.model_dir, relative).read_bytes())
        if not math.isclose(duration, entry["duration"], abs_tol=1e-6):
            raise ValueError("Recorded audio duration does not match completed sample")
        if args.voice_clone and not Path(paths.model_dir, entry["prompt_audio_filepath"]).is_file():
            raise ValueError("Completed sample is missing its reference audio")
    pending = [dataset[i] for i, sid in enumerate(ids) if wav_rel_path(paths.dataset_dir_name, sid) not in done]
    metadata = previous or {"settings": settings, "normalizer_at_generation": normalizer_identity(),
                            "metrics_ready": False}
    metadata.update(completed_samples=len(done), status="running", finish_reason_counts=finish_reason_counts,
                    generation_token_limit_reached_samples=finish_reason_counts["length"])
    if previous and pending:
        metadata.setdefault("resume_normalizer_snapshots", []).append(normalizer_identity())
    write_metadata(metadata_path, metadata)
    if not pending:
        metadata.update(status="complete", completed_samples=len(dataset))
        write_metadata(metadata_path, metadata)
        print_generation_progress(len(dataset), len(dataset), finish_reason_counts["length"])
        print("All samples already complete; no synthesis requests sent.")
        return
    client.wait_for_server(args.server_timeout)

    def generate(row):
        return client.synthesize_result(row[args.text_column], row["_reference"])

    def prepare(batch):
        prepared = []
        for row in batch:
            prompt_path, reference = prepare_reference(row, paths, args.voice_clone)
            prepared.append({**row, "_prompt_path": prompt_path, "_reference": reference})
        return prepared

    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        for start in range(0, min(len(pending), args.warmup_steps * args.batch_size), args.batch_size):
            list(executor.map(generate, prepare(pending[start:start + args.batch_size])))
        with open_manifest(paths.manifest_path, done) as manifest:
            for start in range(0, len(pending), args.batch_size):
                batch = prepare(pending[start:start + args.batch_size])
                begin = time.perf_counter()
                outputs = list(executor.map(generate, batch))
                elapsed = (time.perf_counter() - begin) / len(batch)
                for row, result in zip(batch, outputs):
                    relative = wav_rel_path(paths.dataset_dir_name, row["id"])
                    Path(paths.model_dir, relative).write_bytes(result.audio)
                    entry = manifest_entry(
                        relative, wav_duration(result.audio), elapsed, row[args.text_column], row["_prompt_path"],
                    )
                    entry.update(finish_reason=result.finish_reason,
                                 generation_token_limit_reached=result.generation_token_limit_reached)
                    write_entry(manifest, entry)
                    finish_reason_counts[result.finish_reason] += 1
                manifest.flush()
                previous_completed = metadata["completed_samples"]
                metadata["completed_samples"] += len(batch)
                metadata["generation_token_limit_reached_samples"] = finish_reason_counts["length"]
                write_metadata(metadata_path, metadata)
                if (metadata["completed_samples"] < len(dataset)
                        and metadata["completed_samples"] // 50 > previous_completed // 50):
                    print_generation_progress(metadata["completed_samples"], len(dataset), finish_reason_counts["length"])
    metadata["status"] = "complete"
    write_metadata(metadata_path, metadata)
    print_generation_progress(len(dataset), len(dataset), finish_reason_counts["length"])
    print("Results saved at path:", os.path.abspath(paths.manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    main(parser().parse_args())
