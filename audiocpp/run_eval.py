"""
Time-to-first-audio probe for the Open TTS Leaderboard (audio.cpp backend).

0xShug0/audio.cpp (https://github.com/0xShug0/audio.cpp) is a ggml C++ runtime whose
`audiocpp_server` streams many TTS families over an OpenAI-style SSE endpoint. Some are a second
ENGINE for models the leaderboard already evaluates (Qwen3-TTS, VoxCPM2, Supertonic,
Pocket TTS, Breeze, Fish Audio, and Voxtral); the
rest are only available here (see FAMILIES).

    POST /v1/audio/speech  {"model": ..., "input": ..., "stream_format": "sse", "response_format": "pcm"}
      -> data: {"type": "speech.audio.delta", "audio": "<base64 PCM16 mono>"}   (repeated)
      -> data: {"type": "speech.audio.done", "timing": {"ttft_ms": ...}}
      -> data: [DONE]

The job runs the server itself: this script installs the model's GGUF package with
`audiocpp_model_manager`, writes a one-model server config with `mode: "streaming"`, starts
`audiocpp_server`, and is then a plain HTTP client of it. First audio = the first non-empty
`speech.audio.delta` reaching the client, so the clock includes the (loopback) transport.

Only families audio.cpp STREAMS are registered. Some registered families
(Supertonic, OmniVoice) "stream" one event per TEXT chunk, so a CV3 sentence arrives as one event;
the probe reports those as not streaming, which is the honest label. Left out: segment-level,
clone-only ports (ZipVoice, Sopro), MiraTTS (no published package), PersonaPlex (speech-to-speech).

Packages are the full-precision GGUFs (bf16 / f16 / orig) where one is published, not the Q8
defaults: a quantized model is a different model. Audio8 and VoxCPM1 publish only Q8_0; the
package is recorded in the sidecar.

Voice mode: --voice_clone sends each row's prompt_audio/prompt_text as the reference. Clone-only
families (Confucius4) always clone; their sidecar then carries the `_voice_clone` suffix, so it
is never mixed up with a fixed-voice number.

TTFA only: audio.cpp's audio is not what the eval scores, so running without --ttfa_probe is
refused. Sidecars are engine-tagged "audiocpp" by submit_ttfa_jobs.sh.
"""

import argparse
import base64
import io
import json
import os
import subprocess
import tempfile
import time
import wave

import requests

from run_eval_utils import add_common_args, load_tts_dataset, set_seed

AUDIOCPP_DIR = os.environ.get("AUDIOCPP_DIR", "/app")  # binaries of the ghcr.io/0xshug0/audio.cpp image

# Leaderboard model id -> how audio.cpp serves it. Request fields mirror the reference backend's
# defaults (its run_eval.py / the TTFA_TARGETS entry), so only the engine differs.
FAMILIES = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": {
        "family": "qwen3_tts",
        "clone": False,  # reference-audio cloning uses the separate Base checkpoint
        "package": "qwen3_tts_1_7b_customvoice_bf16",
        "session_options": {"qwen3_tts.perf_mode": "cuda_perf"},
        # CustomVoice preset Aiden and 12 frames per stream event.
        "request": {"voice": "Aiden"},
        "options": {"stream_frames_per_event": "12"},
        # The talker takes the English language NAME, as in qwen3tts/run_eval.py; unknown codes pass through.
        "language": {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "de": "German",
                     "fr": "French", "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian"},
    },
    "openbmb/VoxCPM2": {
        "family": "voxcpm2",
        "package": "voxcpm2_bf16",
        "session_options": {"voxcpm2.perf_mode": "cuda_perf"},
        # voxcpm2/run_eval.py: cfg_value 2.0, inference_timesteps 10, no voice (plain TTS).
        # retry_badcase must be off: retrying a finished bad case is offline-only.
        # The reference emits one codec patch per event, represented here by one frame per event.
        "request": {"guidance_scale": 2.0, "num_inference_steps": 10},
        "options": {"retry_badcase": "false", "stream_frames_per_event": "1"},
    },
    "Supertone/supertonic-3": {
        "family": "supertonic",
        "clone": False,  # preset voices only
        "package": "supertonic_3_orig",
        # supertonic/run_eval.py: voice M1, total_steps 8, speed 1.05.
        "request": {"voice": "M1", "num_inference_steps": 8, "speed": 1.05},
        "language": True,
    },
    "kyutai/pocket-tts": {
        "family": "pocket_tts",
        # English only: other languages are separate packages (and models).
        "package": "pocket_tts_english_bf16",
        # pocket-tts/run_eval.py: English default voice `alba`; sampling left at the engine's defaults.
        "request": {"voice": "alba"},
    },
    "BreezeBlue/Breeze-TTS-2": {
        "family": "breeze_tts",
        "package": "breeze_tts_2_bf16",
        "session_options": {"breeze_tts.perf_mode": "cuda_perf"},
        # breeze-tts/run_eval.py: reference-free voice design from a neutral per-language instruction
        # (filled in main()), cfg 1.0.
        # Default grow starts at one frame with zero lookahead, matching Python's fast-mode first chunk.
        "request": {"guidance_scale": 1.0},
        "instructions": {"en": "Speak clearly and naturally.", "zh": "请用清晰自然的语气朗读。"},
        # Prompt-audio cloning is a separate task ("clon") for this family.
        "clone_task": "clon",
    },
    "k2-fsa/OmniVoice": {
        "family": "omnivoice",
        "package": "omnivoice_bf16",
        "session_options": {"omnivoice.perf_mode": "flash_attention"},
        # omnivoice/run_eval.py: auto voice, num_step 32. One stream event per text chunk.
        "request": {"num_inference_steps": 32},
        "language": True,
    },
    "fishaudio/s2-pro": {
        "family": "fish_audio",
        "package": "fish_audio_s2_pro_bf16",
        # The streaming SGLang target uses 2048 tokens, temperature 0.8, top-k 30, and top-p 0.8.
        "request": {"max_tokens": 2048, "temperature": 0.8, "top_k": 30, "top_p": 0.8},
    },
    "microsoft/VibeVoice-1.5B": {
        "family": "vibevoice",
        "package": "vibevoice_1_5b_bf16",
        # VibeVoice 1.5B uses its documented guidance scale of 1.3.
        "request": {"guidance_scale": 1.3},
        # A multi-speaker script model: the server rejects text without `Speaker N:` lines.
        "text_template": "Speaker 1: {}",
        "reference_text": False,  # speaker conditioning consumes audio, not its transcript
    },
    "mistralai/Voxtral-4B-TTS-2603": {
        "family": "voxtral_tts",
        "clone": False,
        "package": "voxtral_4b_tts_2603_bf16",
        # vLLM-Omni's Voxtral profile: casual_male, 2048 tokens, seven acoustic steps, CFG 1.2.
        "request": {"voice": "casual_male", "max_tokens": 2048,
                    "num_inference_steps": 7, "guidance_scale": 1.2},
    },
    # ── Not on the leaderboard: audio.cpp's streaming families, at their documented defaults ──
    "kugelaudio/kugelaudio-0-open": {
        "family": "kugelaudio",
        "clone": False,  # preset voices only
        "package": "kugelaudio_0_open_bf16",
        # `default`/`clear` are German voices; English text gets the British English preset.
        "options": {"voice_id": "english_female"},
    },
    "neuphonic/neutts-2e": {
        "family": "neutts",
        "clone": False,  # preset voices only
        "package": "neutts_2e_orig",
        "options": {"voice_id": "emily"},
    },
    "Edge0/Audio8-TTS-Preview-0.6b": {
        "family": "audio8_tts",
        "package": "audio8_tts_preview_0_6b_q8_0",  # the only published package
    },
    "LiquidAI/LFM2.5-Audio-1.5B": {
        "family": "lfm2_audio",
        "clone": False,  # preset voices only
        "package": "lfm2_audio_1_5b_f16",
        "request": {"voice": "us_female"},
        # Two GGUFs (model + mmproj) with no embedded config: the server loads the package DIRECTORY.
        "path": "dir",
    },
    "ekwek/Soprano-1.1-80M": {
        "family": "soprano_tts",
        "clone": False,  # preset voices only
        "package": "soprano_1_1_80m_bf16",
    },
    "openbmb/VoxCPM-0.5B": {
        "family": "voxcpm1",
        "package": "voxcpm1_0_5b_q8_0",  # the only published package
        "options": {"retry_badcase": "false"},
    },
    "dots-studio/dots.tts-soar": {
        "family": "dots_tts",
        "package": "dots_tts_soar_bf16",
    },
    "dots-studio/dots.tts-mf": {
        "family": "dots_tts",
        "package": "dots_tts_mf_bf16",
    },
    "netease-youdao/Confucius4-TTS": {
        "family": "confucius4_tts",
        "package": "confucius4_tts_orig",
        "language": True,
        "clone_only": True,
        "clone_task": "clon",
        "reference_text": False,  # rejected as an unknown option: it clones from the audio alone
    },
}


def _package_model_path(models_dir, family, package, as_dir=False):
    """Install `package` (idempotent) and return the path of its .gguf (its directory if `as_dir`).

    The file comes from the image's own model_specs/<family>.json (package `files` are relative to
    the models root), so it always matches the binaries the job runs.
    """
    with open(os.path.join(AUDIOCPP_DIR, "model_specs", f"{family}.json"), encoding="utf-8") as f:
        packages = {p["id"]: p for p in json.load(f)["packages"]}
    if package not in packages:
        raise SystemExit(f"No package {package!r} for family {family}; available: {sorted(packages)}")
    gguf = [p for p in packages[package]["files"] if p.endswith(".gguf")]
    if not gguf or (len(gguf) > 1 and not as_dir):
        raise SystemExit(f"Expected one .gguf in package {package}, got {gguf}")

    cmd = [os.path.join(AUDIOCPP_DIR, "audiocpp_model_manager"), "install", package, "--models-dir", models_dir]
    print("$", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    # Retried: large downloads from the model host occasionally drop the connection.
    for attempt in range(1, 4):
        if subprocess.run(cmd, cwd=AUDIOCPP_DIR).returncode == 0:
            break
        if attempt == 3:
            raise SystemExit(f"audiocpp_model_manager install {package} failed 3 times.")
        print(f"install attempt {attempt} failed; retrying in {30 * attempt}s", flush=True)
        time.sleep(30 * attempt)
    print(f"installed {package} in {time.perf_counter() - t0:.1f}s", flush=True)

    # Some packages install under a directory their spec does not name (e.g. Audio8's
    # `Audio8-TTS-Preview-0.6B-GGUF/`), so fall back to finding the file by name.
    path = os.path.join(models_dir, gguf[0])
    if not os.path.exists(path):
        found = [os.path.join(root, os.path.basename(gguf[0]))
                 for root, _, files in os.walk(models_dir) if os.path.basename(gguf[0]) in files]
        if len(found) != 1:
            raise SystemExit(f"Package {package} should provide {gguf[0]}; found {found} under {models_dir}.")
        path = found[0]
    return os.path.dirname(path) if as_dir else path


def _start_server(args, cfg, model_path, workdir, task):
    """Write a one-model streaming config, start audiocpp_server, wait for /health. Returns the Popen."""
    backend = "cpu" if args.device == "cpu" else "cuda"
    device = int(args.device.split(":", 1)[1]) if args.device.startswith("cuda:") else 0
    # CPU: one worker per vCPU of the job's quota (submit_ttfa_jobs.sh exports OMP_NUM_THREADS from
    # the cgroup). GPU: the CLI default; the compute is on the device.
    threads = args.threads
    if not threads:
        threads = (int(os.environ.get("OMP_NUM_THREADS") or 0) or os.cpu_count()) if backend == "cpu" else 4
    config = {
        "host": "127.0.0.1",
        "port": args.port,
        "backend": backend,
        "device": device,
        "threads": threads,
        # Load at startup, so /health means "ready" and the first timed row pays no load.
        "lazy_load": False,
        "models": [{
            "id": "probe",
            "family": cfg["family"],
            "path": model_path,
            "task": task,
            "mode": "streaming",
        }],
    }
    # Session controls such as perf_mode belong in the server model config, not request options.
    if cfg.get("session_options"):
        config["models"][0]["session_options"] = cfg["session_options"]
    config_path = os.path.join(workdir, "server.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=1)
    print("server config:", json.dumps(config), flush=True)

    log_path = os.path.join(workdir, "server.log")
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [os.path.join(AUDIOCPP_DIR, "audiocpp_server"), "--config", config_path, "--no-ui"],
        cwd=AUDIOCPP_DIR, stdout=log, stderr=subprocess.STDOUT,
    )
    base_url = f"http://127.0.0.1:{args.port}"
    deadline = time.time() + args.server_startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            log.flush()
            with open(log_path, encoding="utf-8", errors="replace") as f:
                print("--- audiocpp_server log (full) ---\n" + f.read())
            raise SystemExit(f"audiocpp_server exited with code {proc.returncode} before becoming healthy.")
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                print(f"audiocpp_server healthy at {base_url} (backend={backend}, threads={threads})", flush=True)
                return proc, base_url, log_path, threads
        except requests.RequestException:
            pass
        time.sleep(2)
    proc.kill()
    raise SystemExit(f"audiocpp_server not healthy after {args.server_startup_timeout}s — see {log_path}")


def main(args):
    if args.ttfa_probe == 0:
        raise SystemExit(
            "This backend is TTFA-only: pass --ttfa_probe=N (or -1 for the whole split).\n"
            "audio.cpp re-implements inference in ggml, so its audio is not the audio the\n"
            "leaderboard's WER/RTFx are scored from. Use the model's own backend for those."
        )
    if args.model_id not in FAMILIES:
        raise SystemExit(f"{args.model_id} is not registered for audio.cpp streaming. "
                         f"Registered: {sorted(FAMILIES)} (see the module docstring for what is not).")
    cfg = FAMILIES[args.model_id]
    package = args.package or cfg["package"]
    set_seed(args.seed)

    # Clone-only families always clone (their sidecar is suffixed `_voice_clone`, so it never
    # passes for a fixed-voice number).
    # Fixed-voice families ignore --voice_clone, like the other backends without a clone mode.
    voice_clone = cfg.get("clone_only", False) or (args.voice_clone and cfg.get("clone", True))
    if voice_clone != args.voice_clone:
        print(f"NOTE: {cfg['family']} {'only clones' if voice_clone else 'has no clone mode'}; "
              f"probing with voice_clone={voice_clone}.")
    task = cfg.get("clone_task", "tts") if voice_clone else "tts"
    mode_suffix = "_voice_clone" if voice_clone else ""

    # Loaded BEFORE the server so a dataset problem fails fast. Audio is left undecoded (no
    # torchcodec in this image): the reference is sent to the server as the original file bytes.
    dataset = load_tts_dataset(
        args, extra_columns=("prompt_audio", "prompt_text") if voice_clone else (), decode_audio=False)

    # Imported HERE, not at module scope: the probe module is injected only by submit_ttfa_jobs.sh.
    from ttfa_probe import run_probe, sample_indices

    def _reference(row):
        """(base64 WAV, transcript) of a row's prompt, converted with ffmpeg if it is not a WAV."""
        audio = row["prompt_audio"]
        data = audio.get("bytes")
        if data is None:
            with open(audio["path"], "rb") as f:
                data = f.read()
        if data[:4] != b"RIFF":
            data = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "wav", "pipe:1"],
                                  input=data, capture_output=True, check=True).stdout
        return base64.b64encode(data).decode("ascii"), row["prompt_text"]

    jobs = []
    for i in sample_indices(args.ttfa_probe, len(dataset)):
        row = dataset[i]
        job = {"text": row[args.text_column]}
        if voice_clone:
            job["ref"] = _reference(row)
        jobs.append(job)

    workdir = tempfile.mkdtemp(prefix="audiocpp_")
    model_path = _package_model_path(args.models_dir, cfg["family"], package, as_dir=cfg.get("path") == "dir")
    proc, base_url, log_path, threads = _start_server(args, cfg, model_path, workdir, task)

    try:
        version = subprocess.run([os.path.join(AUDIOCPP_DIR, "audiocpp_server"), "--version"],
                                 capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:  # noqa: BLE001 — diagnostics only
        version = None
    print(f"audio.cpp server version: {version}, image: {os.environ.get('AUDIOCPP_IMAGE')}")

    def _body(text, stream, ref=None):
        text = cfg.get("text_template", "{}").format(text)
        body = {"model": "probe", "input": text, "seed": args.seed, **cfg.get("request", {})}
        options = dict(cfg.get("options", {}))
        if "instructions" in cfg:
            body["instructions"] = cfg["instructions"].get(args.language, cfg["instructions"]["en"])
        if cfg.get("language"):
            # True sends the ISO code; a dict maps it to the name the family expects.
            names = cfg["language"] if isinstance(cfg["language"], dict) else {}
            body["language"] = names.get(args.language, args.language)
        if ref is not None:
            # The row's own prompt replaces any preset voice, as in the reference backends' clone mode.
            body.pop("voice", None)
            options.pop("voice_id", None)
            body["voice_ref"] = {"type": "base64", "data": ref[0]}
            if cfg.get("reference_text", True):
                body["reference_text"] = ref[1]
        if stream:
            body.update(stream_format="sse", response_format="pcm")
        if options:
            body["options"] = options
        return body

    def _post(body, stream):
        r = requests.post(f"{base_url}/v1/audio/speech", json=body, stream=stream, timeout=args.request_timeout)
        if r.status_code != 200:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                tail = f.read()[-4000:]
            raise SystemExit(f"/v1/audio/speech returned {r.status_code}: {r.text[:2000]}\n"
                             f"--- server log (tail) ---\n{tail}")
        return r

    # Clone-mode warm-up uses a prompt from OUTSIDE the probe set: the server caches prepared
    # references, so warming up on a probed row would make that row's TTFA artificially low.
    warm_ref = None
    if voice_clone:
        probed = set(sample_indices(args.ttfa_probe, len(dataset)))
        spare = next((i for i in range(len(dataset)) if i not in probed), None)
        warm_ref = _reference(dataset[spare]) if spare is not None else jobs[0]["ref"]

    # Untimed warm-up. The NON-streaming request returns a WAV, whose header is the only place the
    # sample rate appears (SSE deltas are bare PCM16); the streaming ones prime the streaming path.
    r = _post(_body("Hello there, this is a warm-up sentence for the engine.", stream=False, ref=warm_ref),
              stream=False)
    with wave.open(io.BytesIO(r.content)) as w:
        sample_rate, channels = w.getframerate(), w.getnchannels()
    print(f"output: {sample_rate} Hz, {channels} channel(s)")
    if channels != 1:
        raise SystemExit(f"Expected mono output, got {channels} channels.")

    chunk_counts = []      # audio deltas per utterance: 1 means nothing was incremental
    server_ttft_ms = []    # the server's own first-audio clock, for comparison with the client's

    def _gen(job):
        """One streaming request at batch size 1, timestamping the first audio delta.

        Always a full generation (no early_stop parameter): the server runs one request at a time,
        so abandoning a stream would leave it generating and inflate the NEXT row's TTFA.
        Reference encoding (clone mode) happens server-side inside the timed request.
        """
        first, n_bytes, n_chunks, ttft = None, 0, 0, None
        with _post(_body(job["text"], stream=True, ref=job.get("ref")), stream=True) as r:
            for line in r.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                kind = event.get("type")
                if kind == "speech.audio.delta":
                    pcm = base64.b64decode(event["audio"])
                    if pcm and first is None:
                        first = time.perf_counter()
                    n_bytes += len(pcm)
                    n_chunks += 1
                elif kind == "speech.audio.done":
                    ttft = (event.get("timing") or {}).get("ttft_ms")
                elif kind == "error":
                    raise SystemExit(f"stream error event: {event}")
        chunk_counts.append(n_chunks)
        server_ttft_ms.append(ttft)
        # PCM16 mono: 2 bytes per sample. Passed as seconds — only the duration is used.
        return n_bytes / 2 / sample_rate, first

    for i in range(args.warmup_steps):
        t0 = time.perf_counter()
        _gen({"text": "This is another warm-up sentence, streamed this time.", "ref": warm_ref})
        print(f"streaming warm-up {i + 1}/{args.warmup_steps}: {time.perf_counter() - t0:.2f}s")
    chunk_counts.clear()
    server_ttft_ms.clear()

    model_safe = args.model_id.replace("/", "-")
    # Not created here: run_probe creates it on write, so a crash leaves no dir and the job fails.
    out_dir = os.path.join("results", model_safe)

    try:
        run_probe(
            _gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=out_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={
                "device": args.device,
                "voice_clone": voice_clone,
                "task": task,
                "note": f"driven at batch size 1, audiocpp_server SSE streaming ({cfg['family']})",
                "engine": "audio.cpp",
                "engine_version": version,
                "image": os.environ.get("AUDIOCPP_IMAGE"),
                "family": cfg["family"],
                "package": package,
                "server_threads": threads,
                "session_options": cfg.get("session_options", {}),
                "first_audio": "first non-empty speech.audio.delta received by the client (loopback HTTP)",
                "request": {k: v for k, v in _body("<text>", stream=True, ref=("<prompt>", "<prompt_text>") if voice_clone else None).items()
                            if k not in ("model", "input")},
                "sample_rate": sample_rate,
                "chunks_per_utterance": chunk_counts,
                # The server's own first-audio clock (excludes HTTP/SSE transport).
                "server_ttft_ms": server_ttft_ms,
            },
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser, voice_clone=False)
    parser.set_defaults(dataset_path="hf-audio/cv3_eval", dataset="zero_shot")

    parser.add_argument("--model_id", type=str, default="openbmb/VoxCPM2",
                        help=f"Model id (names the sidecar); one of {sorted(FAMILIES)}.")
    parser.add_argument("--package", type=str, default=None,
                        help="Override the audio.cpp package (e.g. a Q8_0 one). Recorded in the sidecar; "
                             "a different precision is a different model, so do not compare it naively.")
    parser.add_argument("--language", type=str, default="en",
                        help="ISO language code; sent to families that take one, and picks Breeze's instruction.")
    parser.add_argument("--models_dir", type=str, default="/tmp/audiocpp_models",
                        help="Where audiocpp_model_manager installs the package.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=0,
                        help="Server worker threads; 0 = OMP_NUM_THREADS / nproc on CPU, 4 on GPU.")
    parser.add_argument("--server_startup_timeout", type=int, default=1800,
                        help="Seconds to wait for /health (includes loading the model).")
    parser.add_argument("--request_timeout", type=int, default=600)

    args = parser.parse_args()

    print("*" * 100)
    print(f"TTFA probe: {args.model_id} via audio.cpp on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
