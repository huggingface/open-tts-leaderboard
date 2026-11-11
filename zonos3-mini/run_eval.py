"""
TTS synthesis for the Open TTS Leaderboard (ZONOS3-Mini backend, stage 1).

ZONOS3-Mini (Zyphra/ZONOS3-Mini, Apache-2.0) is a multilingual MoE TTS model (44.1 kHz) with the runtime of
https://github.com/Zyphra/ZONOS3-Mini.

`--voice_clone` clones each sample's `prompt_audio` in speaker-embedding mode (the release speaker encoder embeds the
clip; the model's default cloning mode, `prompt_text` unused) through the runtime's server: submit_jobs.sh starts
servers/sgl/scripts/serve.sh in the job (as the README serves the model) and this script sends `--batch_size`
concurrent POST /v2/generate requests per batch, which the server's scheduler runs as one batch while it decodes audio
beside the generation. By default (`--no-voice_clone`) every sample uses the prefix of `--voice` (a native default
voice of the split's language), encoded once, through the offline `zonos3_sgl.tts.TTSLLM` API, which generates a whole
minibatch in one call (no SIM). Either way, per-sample generation time is the minibatch time divided by its size, as in
the other batched backends.

Text: the runtime normalizes en, zh, ja, ko, de, es, fr, it and pt (the NeMo grammars). A voice-clone split in any other
language is sent as en_us without normalization, its numbers spelled out here with num2words.

`--device cpu` (submit_ttfa_jobs.sh's zonos3-mini-ggml target, on its CPU image) is the TTFA probe of the runtime's
native ggml server (servers/ggml of the same repository) instead: see ttfa_probe_ggml.
"""

import argparse
import base64
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from tqdm import tqdm

CODEBOOK_SIZE = 4096  # valid codec token ids are [0, 4096) (zonos3.tokenizer.matched_codec.decode)

# The runtime (torch, zonos3, zonos3_sgl) is imported where the GPU paths use it: the CPU image has only the ggml server.
from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

# --device cpu: the post-training-quantized backbone the ggml runtime publishes (gguf/zonos3-ptq.gguf of the model repo,
# Q4_0 experts and Q8_0 mixers; the README's CPU default) and the files the server picks up next to it (the keep2_plain
# decoder zac-keep2.gguf, the speaker encoder spk-encoder.gguf), plus the default voices.
GGML_BACKBONE = "zonos3-ptq.gguf"

# ASR language -> ZONOS3 language code (text normalization).
LANGUAGES = {"en": "en_us", "zh": "cmn", "fr": "fr_fr", "de": "de", "es": "es", "it": "it", "pt": "pt_br",
             "ja": "ja", "ko": "ko"}

# ASR language -> fixed voice (--no-voice_clone): a native default voice of the model, female where there is one.
# The model continues the reference clip's codec prefix, so a clip in another language derails the speech (the
# English Savannah clip gave 54-144% WER on fr/es/it/ko/zh/ja). There is no Japanese default voice.
VOICE_DIR = "/opt/ZONOS3-Mini/assets/default_voices"
VOICES = {
    "en": "English/Savannah_en-American_F_1.mp3",
    "zh": "Mandarin Chinese/Meilin_zh-energetic_F_1.wav",
    "fr": "French/Gaspard_fr_M_1.mp3",
    "de": "German/Liesel_de_F_1.mp3",
    "es": "Spanish/Ximena_es_F_1.wav",
    "it": "Italian/Ginevra_it_F_1.mp3",
    "ko": "Korean/Seoyeon_ko_F_1.wav",
    "pt": "Portuguese/Leonor_pt_F_1.wav",
}

# ── Voice clone through the runtime's server ─────────────────────────────────
EMPTY_AUDIO_SECONDS = 0.1  # an empty response is written as this much silence, so it scores as a full miss

# Languages without runtime normalization whose numbers num2words reads: decimal separator, and the word for "%"
# (Turkish says it before the number).
NUMBER_DECIMAL = {"ru": ",", "nl": ",", "pl": ",", "cs": ",", "ro": ",", "fi": ",", "tr": ",", "id": ",", "vi": ",",
                  "ar": ".", "th": "."}
NUMBER_PERCENT = {"ru": "процентов", "nl": "procent", "pl": "procent", "cs": "procent", "ro": "la sută", "fi": "prosenttia",
                  "tr": "yüzde", "id": "persen", "vi": "phần trăm", "ar": "بالمئة", "th": "เปอร์เซ็นต์"}
# num2words' decimal reading is wrong for these ("3,5" -> vi "ba phẩy năm mươi"): integer part, this word, fraction digits
DECIMAL_WORD = {"tr": "virgül", "vi": "phẩy", "ar": "فاصلة", "th": "จุด"}
# an optional "%" before (Turkish writing) or after the number: thousand groups split by spaces, or digits with "." /
# "," inside (one separator: a decimal or thousands; more: a date or version, read part by part)
NUMBER = re.compile(r"(%\s?)?(?<![\w.,])(\d{1,3}(?:[ \u00a0]\d{3})+(?!\d)|\d+(?:[.,]\d+)*)(\s?%)?")


def number_words(match, lang):
    """One number (thousand groups, decimal, optional %) in words; the match unchanged if num2words fails."""
    from num2words import num2words  # installed by submit_jobs.sh (only the voice-clone eval needs it)

    raw, percent = match.group(2), match.group(1) or match.group(3)
    dec = NUMBER_DECIMAL[lang]
    try:
        if re.fullmatch(r"\d+", raw) or any(c in raw for c in " \u00a0") or re.fullmatch(r"\d+[.,]\d{3}", raw) and dec not in raw:
            words = num2words(int(re.sub(r"\D", "", raw)), lang=lang)  # an integer (the other separator groups thousands)
        elif dec in raw and raw.count(dec) == 1 and not re.search(r"[.,]", raw.replace(dec, "", 1)):
            whole, frac = raw.split(dec)
            if lang in DECIMAL_WORD:
                zeros = len(frac) - len(frac.lstrip("0"))
                words = " ".join([num2words(int(whole), lang=lang), DECIMAL_WORD[lang]]
                                 + [num2words(0, lang=lang)] * zeros + ([num2words(int(frac), lang=lang)] if frac.lstrip("0") else []))
            else:
                words = num2words(float(f"{whole}.{frac}"), lang=lang)
        else:  # the other separator (not a thousands group), or several: a date, version or ordinal, part by part
            words = " ".join(num2words(int(part), lang=lang) for part in re.split(r"[.,]", raw))
    except (NotImplementedError, OverflowError, ValueError):
        return match.group(0)
    if percent:
        words = f"{NUMBER_PERCENT[lang]} {words}" if lang == "tr" else f"{words} {NUMBER_PERCENT[lang]}"
    return words


def spell_numbers(text, lang):
    """Digits -> words for a language without runtime normalization (unchanged where num2words has no reading)."""
    if lang not in NUMBER_DECIMAL:
        return text

    def replace(m):
        before, after = text[m.start() - 1 : m.start()], text[m.end() : m.end() + 1]
        return (" " if before.isalnum() else "") + number_words(m, lang) + (" " if after.isalnum() else "")
    return NUMBER.sub(replace, text)


def http(url, payload=None, timeout=60):
    """GET (payload None) or POST JSON; returns (body bytes, response headers)."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as resp:
        return resp.read(), resp.headers


def process_alive(pid):
    """False once `pid` has exited (a zombie counts as exited)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def wait_for_server(base_url, timeout, server_pid, server_log=None):
    """Poll GET /v2/health; fail fast if the server process dies or its log shows a traceback (a scheduler or tokenizer
    subprocess that crashes leaves the API process up but never healthy). The first start JIT-compiles kernels."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            http(f"{base_url}/v2/health", timeout=10)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        if server_pid and not process_alive(server_pid):
            raise RuntimeError("ZONOS3-Mini server exited before it became healthy (its log tail follows).")
        if server_log and os.path.exists(server_log):
            with open(server_log, errors="replace") as f:
                if "Traceback (most recent call last)" in f.read():
                    raise RuntimeError("ZONOS3-Mini server raised during startup (its log tail follows).")
        if time.monotonic() > deadline:
            raise RuntimeError(f"ZONOS3-Mini server not healthy after {timeout}s.")
        time.sleep(10)


def synthesize(base_url, payload, timeout, retries):
    """POST /v2/generate -> (float32 mono audio, sample rate). Retries 5xx/connection errors, raises on 4xx."""
    for attempt in range(retries + 1):
        try:
            body, headers = http(f"{base_url}/v2/generate", payload, timeout=timeout)
            return np.frombuffer(body, dtype="<f4").copy(), int(headers.get("X-Audio-Sample-Rate", 44100))
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == retries:
                raise RuntimeError(f"/v2/generate HTTP {e.code}: {e.read()[:500]!r}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(2.0 * 2**attempt)


def server_voice_clone(args):
    """Voice-clone eval through the runtime's server: each minibatch as `--batch_size` concurrent requests in
    speaker-embedding mode (the server embeds each prompt clip inside the timed requests)."""
    server_language = LANGUAGES.get(args.language, "en_us")
    text_normalization = args.language in LANGUAGES
    wait_for_server(args.server_url, args.server_timeout, args.server_pid, args.server_log)
    caps = json.loads(http(f"{args.server_url}/v2/capabilities")[0])
    if "embedding" not in caps["speaker_modes"]:
        raise SystemExit(f"The served checkpoint has no embedding speaker mode: {caps['speaker_modes']}")
    served = json.loads(http(f"{args.server_url}/v1/models")[0])["data"][0]["root"]
    print(f"ZONOS3-Mini server: model={served} codec={caps['audio_codec']} decoder={json.dumps(caps['decoder'])}")
    print(f"language={args.language} -> {server_language}, text_normalization={text_normalization}")

    paths = output_paths(args, "_voice_clone")
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path
    os.makedirs(os.path.join(paths.output_dir, "prompts"), exist_ok=True)
    dataset = load_tts_dataset(args, ("prompt_audio",))
    executor = ThreadPoolExecutor(max_workers=args.batch_size)

    def run_batch(batch):
        """Synthesize a minibatch -> (list of 1d float32 arrays, sample rate, prompt paths, elapsed_s)."""
        payloads, prompt_rels = [], []
        for sample_id, text, prompt in zip(batch["id"], batch[args.text_column], batch["prompt_audio"]):
            # The prompt is written under results/ (untimed) so stage 3 (SIM) reads the clip the server embedded.
            prompt_rel = os.path.join(dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
            prompt_path = os.path.join(model_dir, prompt_rel)
            if not os.path.exists(prompt_path):
                sf.write(prompt_path, np.asarray(prompt["array"], dtype=np.float32), prompt["sampling_rate"])
            with open(prompt_path, "rb") as f:
                speaker_audio = base64.b64encode(f.read()).decode("ascii")
            payloads.append({
                "text": spell_numbers(text, args.language), "language": server_language,
                "text_normalization": text_normalization, "speaker_mode": "embedding",
                "speaker_audio_base64": speaker_audio, "speaker_audio_name": os.path.basename(prompt_path),
                "seed": args.seed, "max_tokens": args.max_tokens, "stream": False, "format": "pcm",
            })
            prompt_rels.append(prompt_rel)
        start = time.perf_counter()
        outputs = list(executor.map(
            lambda p: synthesize(args.server_url, p, args.request_timeout, args.retries), payloads))
        elapsed = time.perf_counter() - start
        return [audio for audio, _ in outputs], outputs[0][1], prompt_rels, elapsed

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)
    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(run_batch, dataset, starts, args.batch_size, args.warmup_steps)

    for start in tqdm(starts, desc="Generating"):
        batch = dataset[start : start + args.batch_size]
        texts = list(batch[args.text_column])
        wavs, sampling_rate, prompt_rels, elapsed = run_batch(batch)
        per_sample = elapsed / len(texts)
        for sample_id, text, audio, prompt_rel in zip(batch["id"], texts, wavs, prompt_rels):
            if audio.size == 0:
                print(f"WARNING: empty audio for sample {sample_id}; writing {EMPTY_AUDIO_SECONDS}s of silence.")
                audio = np.zeros(int(EMPTY_AUDIO_SECONDS * sampling_rate), dtype=np.float32)
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            sf.write(os.path.join(model_dir, rel_path), audio, sampling_rate)
            write_entry(manifest_file, manifest_entry(rel_path, len(audio) / sampling_rate, per_sample, text, prompt_rel))
        manifest_file.flush()

    manifest_file.close()
    executor.shutdown()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(True)


def ttfa_probe_ggml(args):
    """--device cpu: TTFA of the runtime's native ggml server (zonos3-server, built in the CPU image) on GGML_BACKBONE.
    This job starts the server on the CPU and is then its HTTP client: one streaming POST /v2/generate at a time, the
    first audio bytes reaching the client as first audio (the loopback transport is inside the clock), every response
    read to its end. The voice is the fixed native voice of --language in speaker-embedding mode, as the GPU probe
    conditions generate_stream; --voice_clone sends each row's prompt clip as its original file bytes (the server
    decodes and embeds it inside the timed request). Text is sent as the eval sends it (the server normalizes)."""
    from ttfa_probe import run_probe, sample_indices  # injected only by submit_ttfa_jobs.sh

    if args.ttfa_probe == 0:
        raise SystemExit("--device cpu is the TTFA probe of the ggml runtime only (--ttfa_probe); the eval runs on the GPU.")
    language = LANGUAGES.get(args.language, "en_us")
    if not args.voice_clone and args.language not in VOICES:
        raise SystemExit(f"No native default voice for '{args.language}': use --voice_clone.")
    mode_suffix = "_voice_clone" if args.voice_clone else ""
    dataset = load_tts_dataset(args, ("prompt_audio",) if args.voice_clone else (), decode_audio=False)
    jobs = []
    for i in sample_indices(args.ttfa_probe, len(dataset)):
        row = dataset[i]
        job = {"text": spell_numbers(row[args.text_column], args.language), "language": language,
               "text_normalization": args.language in LANGUAGES, "seed": args.seed, "stream": True, "format": "pcm",
               "speaker_mode": "embedding"}
        if args.voice_clone:
            prompt = row["prompt_audio"]
            data = prompt["bytes"] if prompt.get("bytes") else open(prompt["path"], "rb").read()
            job.update(speaker_audio_base64=base64.b64encode(data).decode("ascii"),
                       speaker_audio_name=os.path.basename(prompt.get("path") or "prompt.wav"))
        jobs.append(job)

    with socket.socket() as s:  # a free loopback port for the server
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    log_path = os.path.abspath("zonos3_server.log")
    command = [args.ggml_server, os.path.join(args.gguf_dir, GGML_BACKBONE), "--cpu", "--host", "127.0.0.1",
               "--port", str(port), "--tts-default-voices-dir", os.path.join(args.gguf_dir, "voices")]
    # The speaker encoder (voice clone) defaults to the host's core count; the job's thread cap (the CPU preamble's
    # OMP_NUM_THREADS, the cgroup quota) keeps it from oversubscribing the vCPUs. The backbone and decoder keep ggml's.
    env = dict(os.environ)
    if env.get("OMP_NUM_THREADS"):
        env.setdefault("SPK_THREADS", env["OMP_NUM_THREADS"])
    print("Starting:", " ".join(command), f"(SPK_THREADS={env.get('SPK_THREADS', 'default')})", flush=True)
    with open(log_path, "w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        try:
            wait_for_server(base_url, args.server_timeout, server.pid, log_path)
        except RuntimeError:
            with open(log_path, errors="replace") as f:
                print("--- zonos3-server log ---\n" + f.read()[-5000:])
            raise
        caps = json.loads(http(f"{base_url}/v2/capabilities")[0])
        voice = None
        if not args.voice_clone:  # default voices are listed by a hashed id: match the file's stem to the label
            key = lambda name: re.sub(r"[^a-z0-9]", "", name.lower())
            stem = os.path.splitext(os.path.basename(VOICES[args.language]))[0]
            speakers = json.loads(http(f"{base_url}/v2/speakers")[0])["speakers"]
            voice = next((s for s in speakers if key(s.get("label") or s["id"]) == key(stem)), None)
            if voice is None:
                raise SystemExit(f"{stem} is not one of the server's voices: {[s.get('label') for s in speakers]}")
            for job in jobs:
                job["speaker_embedding_id"] = voice["id"]
        print(f"zonos3-server: backbone={GGML_BACKBONE} decoder={caps.get('decoder', {}).get('id')} "
              f"language={args.language} -> {language} voice={voice['label'] if voice else 'voice_clone'}", flush=True)

        def gen(job):
            request = urllib.request.Request(f"{base_url}/v2/generate", data=json.dumps(job).encode("utf-8"),
                                             headers={"Content-Type": "application/json"})
            first, parts = None, []
            with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
                sample_rate = int(response.headers.get("X-Audio-Sample-Rate", 44100))
                while chunk := response.read1(1 << 16):
                    if first is None:
                        first = time.perf_counter()
                    parts.append(chunk)
            body = b"".join(parts)
            return np.frombuffer(body[: len(body) // 4 * 4], dtype="<f4"), sample_rate, first

        run_probe(
            gen, jobs, model_id=args.model_id, dataset_path=args.dataset_path, dataset_config=args.dataset,
            split=args.split, out_dir=output_paths(args, mode_suffix).model_dir, mode_suffix=mode_suffix,
            n=args.ttfa_probe,
            extra={"device": "cpu", "engine": "zonos3-server (ggml, servers/ggml of the ZONOS3-Mini repo)",
                   "backbone": f"{GGML_BACKBONE} (post-training quantized: Q4_0 experts, Q8_0 mixers)",
                   "decoder": caps.get("decoder", {}).get("id"), "language": language,
                   "streaming_api": "HTTP POST /v2/generate (stream)", "speaker_mode": "embedding",
                   "voice": "voice_clone (prompt embedding)" if args.voice_clone else voice["label"],
                   "threads": "zonos3-server defaults (ggml CPU backend: 4)",
                   "zonos3_commit": os.environ.get("ZONOS3_COMMIT")},
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


def main(args):
    set_seed(args.seed)
    if args.device == "cpu":
        ttfa_probe_ggml(args)
        return
    if args.voice_clone and args.ttfa_probe == 0:
        server_voice_clone(args)
        return

    import torch
    from zonos3.generation import SAMPLE_RATE  # generate_stream's chunk rate (44.1 kHz)
    from zonos3.tts.delay import shear_up
    from zonos3_sgl.tts import TTSLLM, TTSSamplingParams

    # The README's offline settings: FlashAttention, per-request cache, half the GPU for the KV cache (codec fits).
    tts = TTSLLM(model_path=f"hf://{args.model_id}/lm/zonos3_mini",
                 attention_backend="fa", cache_type="naive", memory_ratio=0.5)
    print(f"Loaded ZONOS3-Mini ({args.model_id})")
    print_model_size(tts.engine)
    language = LANGUAGES.get(args.language, args.language)
    if not args.voice_clone and args.voice is None:
        if args.language not in VOICES:
            raise SystemExit(f"No native default voice for '{args.language}': pass --voice or use --voice_clone.")
        args.voice = os.path.join(VOICE_DIR, VOICES[args.language])
    if not args.voice_clone:
        print(f"Fixed voice: {args.voice}")
    voice = None
    if not args.voice_clone:
        wav, sr = sf.read(args.voice, dtype="float32", always_2d=True)
        voice = tts.encode_reference(torch.from_numpy(wav.T.copy()), sr)

    def synth_batch(texts):
        """Synthesize a list of texts in the fixed voice. Return (list of 1d float32 arrays, sample rate, elapsed_s)."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        results = tts.generate(texts, TTSSamplingParams(seed=args.seed), language=language, decode_audio=False,
                               speaker_reference_codes=voice)
        wavs = [decode(r) for r in results]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return wavs, SAMPLE_RATE, elapsed

    def decode(result):
        """Decode one result as TTSLLM.generate does (undo the codebook delay, cut at EOS, codec decode), also when
        the sample never emitted EOS: it then ran to the token cap and undoing the delay leaves pad ids in its last
        frames, which upstream hands to the codec and crashes the whole batch ("Invalid matched codec tokens").
        Here those frames are cut, so a runaway sample is scored (as the long take it is) instead."""
        if not result["audio_tokens"]:
            return np.zeros(1, dtype=np.float32)
        codes = shear_up(torch.tensor(result["audio_tokens"], dtype=torch.int64, device="cuda"), tts.audio_pad_id)
        if result["eos_frame"] is not None:
            codes = codes[: max(0, result["eos_frame"])]
        invalid = ((codes < 0) | (codes >= CODEBOOK_SIZE)).any(dim=1).nonzero()
        if len(invalid):
            print(f"WARNING: no EOS (eos_frame={result['eos_frame']}); cutting {len(codes) - int(invalid[0])} "
                  f"padded frame(s) of {len(codes)}")
            codes = codes[: int(invalid[0])]
        if codes.numel() == 0:
            return np.zeros(1, dtype=np.float32)
        audio = tts._vocoder.decode_all(codes.unsqueeze(0), apply_shear_up=False)
        return audio[0].numpy().astype(np.float32)

    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args, ("prompt_audio",) if args.voice_clone else ())

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # `TTSLLM.generate_stream` is the server's own streaming path (/v2/generate): one prompt at a time, float32
    # chunks yielded as the codec decodes them. It conditions on a speaker EMBEDDING, not on reference codes, so the
    # probe embeds the fixed voice once (untimed, like the eval's encoded prefix) or, with --voice_clone, each
    # sample's prompt clip inside the timed call (as the eval encodes each prompt inside its timed batch).
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        fixed_embedding = None if args.voice_clone else tts.embed_speaker_file(args.voice)

        def _ttfa_gen(job, early_stop):
            torch.cuda.synchronize()
            embedding = fixed_embedding
            if embedding is None:
                prompt = job["prompt_audio"]
                wav = torch.from_numpy(np.asarray(prompt["array"], dtype=np.float32)).reshape(1, -1)
                embedding = tts.embed_speaker(wav, prompt["sampling_rate"])
            first, chunks = None, []
            stream = tts.generate_stream(job["text"], TTSSamplingParams(seed=args.seed), language=language,
                                         speaker_embedding=embedding)
            for chunk in stream:
                if chunk.size == 0:
                    continue
                if first is None:
                    first = time.perf_counter()
                chunks.append(chunk)
                if early_stop:
                    # Closing the generator runs the request to its end without decoding (the scheduler has
                    # no abort), so the next probe starts on an idle engine.
                    stream.close()
                    break
            audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
            return audio, SAMPLE_RATE, first

        jobs = [{"text": dataset[i][args.text_column],
                 "prompt_audio": dataset[i]["prompt_audio"] if args.voice_clone else None}
                for i in sample_indices(args.ttfa_probe, len(dataset))]
        run_probe(
            _ttfa_gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={"device": args.device, "language": language, "streaming_api": "TTSLLM.generate_stream",
                   "voice": "voice_clone (prompt embedding)" if args.voice_clone else args.voice,
                   "flash_tilelang": os.environ.get("FLA_TILELANG"),
                   "sonic_experts": os.environ.get("ZONOS3_USE_TRAINING_SONIC_EXPERTS")},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    def run_batch(batch):
        return synth_batch(list(batch[args.text_column]))

    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(run_batch, dataset, starts, args.batch_size, args.warmup_steps)
    set_seed(args.seed)  # warm-up must not shift the RNG stream of the timed run

    for start in tqdm(starts, desc="Generating"):
        batch = dataset[start : start + args.batch_size]
        texts = list(batch[args.text_column])
        wavs, sampling_rate, elapsed = run_batch(batch)
        per_sample = elapsed / len(texts)
        for sample_id, text, audio in zip(batch["id"], texts, wavs):
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            sf.write(os.path.join(model_dir, rel_path), audio, sampling_rate)
            write_entry(manifest_file, manifest_entry(rel_path, len(audio) / sampling_rate, per_sample, text, None))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Zyphra/ZONOS3-Mini")
    parser.add_argument("--language", type=str, default="en", help="ASR language code of the split (e.g. 'en', 'zh').")
    parser.add_argument("--voice", type=str, default=None,
                        help="Reference clip of the fixed voice (--no-voice_clone); default: the native voice of "
                             "--language in VOICES.")
    # --voice_clone: the runtime's server (started by submit_jobs.sh)
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:1919", help="The runtime's server (serve.sh's port).")
    parser.add_argument("--server_pid", type=int, default=0, help="Server PID: the startup wait fails fast if it exits.")
    parser.add_argument("--server_log", type=str, default=None, help="Server log: the startup wait fails fast on a traceback.")
    parser.add_argument("--server_timeout", type=int, default=3600, help="Seconds to wait for /v2/health.")
    parser.add_argument("--request_timeout", type=int, default=900, help="Seconds per /v2/generate request.")
    parser.add_argument("--retries", type=int, default=4, help="Retries per request on 5xx / connection errors.")
    parser.add_argument("--max_tokens", type=int, default=1350, help="Frame cap per request (15 Hz: 1350 = 90 s).")
    # --device cpu: the ggml runtime of the CPU image (started by ttfa_probe_ggml)
    parser.add_argument("--ggml_server", type=str, default="/opt/ZONOS3-Mini/servers/ggml/build/zonos3-server",
                        help="The runtime's native ggml server binary (--device cpu).")
    parser.add_argument("--gguf_dir", type=str, default="/opt/gguf",
                        help=f"Directory of {GGML_BACKBONE}, zac-keep2.gguf, spk-encoder.gguf and voices/ (--device cpu).")
    add_common_args(parser, batch_size=32, voice_clone=False)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
