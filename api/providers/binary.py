"""HTTP streaming adapters using raw PCM or framed, base64-encoded PCM.

These endpoints synthesize with existing voices. They do not create persistent
voice clones from benchmark references.
"""

import base64
import binascii
import json
import time
from types import SimpleNamespace
from urllib.parse import quote, urlencode

from .base import APIError, AudioResponse, Provider, SynthesisRequest, binary_audio


def _voice(request: SynthesisRequest) -> str:
    if request.reference_audio is not None:
        raise APIError("This provider requires an existing voice; inline cloning is unsupported")
    if not request.voice:
        raise APIError("An existing voice ID is required")
    return request.voice


def _sample_rate(request: SynthesisRequest, supported: set[int]) -> int:
    if request.sample_rate not in supported:
        raise APIError("Unsupported PCM sample rate for this provider")
    return request.sample_rate


def _lines(response):
    """Frame lines independently of HTTP read boundaries, including final EOF."""
    pending = bytearray()
    for chunk in response.iter_content(chunk_size=1024):
        if not chunk:
            continue
        pending.extend(chunk)
        while True:
            newline = pending.find(b"\n")
            if newline < 0:
                break
            line = bytes(pending[:newline]).rstrip(b"\r")
            del pending[: newline + 1]
            yield line
    if pending:
        yield bytes(pending).rstrip(b"\r")


def _json(raw: bytes) -> dict:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        raise APIError("Invalid JSON in synthesis stream") from None
    if not isinstance(payload, dict):
        raise APIError("Invalid message in synthesis stream")
    return payload


def _sse(response):
    """Parse complete SSE events, including multiline data and keepalives."""
    data = []
    event = b""
    for line in _lines(response):
        if not line:
            if data:
                yield event, _json(b"\n".join(data))
            data = []
            event = b""
        elif line.startswith(b":"):
            continue
        else:
            field, _, value = line.partition(b":")
            if value.startswith(b" "):
                value = value[1:]
            if field == b"data":
                data.append(value)
            elif field == b"event":
                event = value
    # SSE dispatch requires a blank line. An unterminated frame is incomplete.
    if data:
        raise APIError("Synthesis stream ended inside an event", retryable=True)


class _PCM:
    def __init__(self):
        self.audio = bytearray()
        self.first_audio_at = None

    def append(self, encoded):
        if not isinstance(encoded, str):
            raise APIError("Invalid audio payload in synthesis stream")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise APIError("Invalid base64 audio in synthesis stream") from None
        self.audio.extend(decoded)
        if self.first_audio_at is None and len(self.audio) >= 2:
            self.first_audio_at = time.perf_counter()

    def finish(self, response, sample_rate: int) -> AudioResponse:
        if not self.audio or len(self.audio) % 2:
            raise APIError("Synthesis returned empty or incomplete PCM audio")
        return AudioResponse(
            audio_bytes=bytes(self.audio),
            encoding="pcm_s16le",
            sample_rate=sample_rate,
            first_audio_at=self.first_audio_at,
            request_id=response.headers.get("X-Request-Id") or response.headers.get("x-request-id"),
        )


class ElevenLabsProvider(Provider):
    """Single-turn dialogue using the native v4 HTTP / Turbo WebSocket APIs.

    https://elevenlabs.io/docs/api-reference/text-to-dialogue/stream
    https://elevenlabs.io/docs/eleven-api/guides/how-to/websockets/realtime-tdd
    """

    def synthesize(self, model: str, request: SynthesisRequest) -> AudioResponse:
        voice = _voice(request)
        sample_rate = _sample_rate(request, {16000, 22050, 24000, 44100, 48000})
        if model == "eleven_v4_turbo":
            return self._dialogue_socket(model, request, voice, sample_rate)
        if model == "eleven_v4":
            # The HTTP dialogue stream warns that longer inputs can terminate
            # early. Fail explicitly rather than score a truncated utterance.
            if len(request.text) > 2000:
                raise APIError("ElevenLabs HTTP dialogue requires at most 2000 characters")
            url = "https://api.elevenlabs.io/v1/text-to-dialogue/stream"
            body = {
                "inputs": [{"text": request.text, "voice_id": voice}],
                "model_id": model,
                "language_code": request.language,
            }
        else:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{quote(voice, safe='')}/stream"
            body = {"text": request.text, "model_id": model}
            # Multilingual v2 explicitly does not accept language_code.
            if model != "eleven_multilingual_v2":
                body["language_code"] = request.language
        with self.post(
            url,
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            params={"output_format": f"pcm_{sample_rate}"},
            json=body,
        ) as response:
            return binary_audio(response, "pcm_s16le", sample_rate)

    def _dialogue_socket(self, model, request, voice, sample_rate):
        try:
            import websocket
        except ImportError:
            raise APIError("Install api/requirements.txt for ElevenLabs Turbo WebSocket support") from None

        url = "wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input?" + urlencode({
            "model_id": model,
            "output_format": f"pcm_{sample_rate}",
            "language_code": request.language,
        })
        connection = None
        deadline = time.perf_counter() + self.timeout
        try:
            connection = websocket.create_connection(
                url, header={"xi-api-key": self.api_key}, timeout=min(10, self.timeout),
            )
            connection.settimeout(self.timeout)
            connection.send(json.dumps({"voices": [voice]}))
            connection.send(json.dumps({"inputs": [{
                "text": request.text, "voice_id": voice, "new_turn": False,
            }]}))
            # Closing flushes short texts that have not reached the buffer threshold.
            connection.send(json.dumps({"close_socket": True}))
            pcm = _PCM()
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise APIError("ElevenLabs dialogue stream timed out", retryable=True)
                connection.settimeout(remaining)
                raw = connection.recv()
                if not raw:
                    raise APIError("ElevenLabs dialogue stream closed without completion", retryable=True)
                payload = _json(raw)
                if payload.get("error") or payload.get("error_code"):
                    raise APIError("ElevenLabs dialogue stream returned an error")
                if payload.get("audio"):
                    pcm.append(payload["audio"])
                if payload.get("is_final") is True:
                    response = SimpleNamespace(headers=connection.getheaders() or {})
                    return pcm.finish(response, sample_rate)
        except websocket.WebSocketBadStatusException as exc:
            status = exc.status_code if isinstance(exc.status_code, int) else None
            raise APIError(
                "ElevenLabs dialogue handshake failed", status_code=status,
                retryable=status in (408, 429) or bool(status and status >= 500),
            ) from None
        except (websocket.WebSocketException, OSError):
            raise APIError("ElevenLabs dialogue connection failed", retryable=True) from None
        finally:
            if connection is not None:
                try:
                    connection.close(timeout=1)
                except (websocket.WebSocketException, OSError):
                    pass


class CartesiaProvider(Provider):
    """https://docs.cartesia.ai/api-reference/tts/bytes"""

    API_VERSION = "2026-08-14"

    def synthesize(self, model: str, request: SynthesisRequest) -> AudioResponse:
        voice = _voice(request)
        sample_rate = _sample_rate(request, {8000, 16000, 22050, 24000, 44100, 48000})
        with self.post(
            "https://api.cartesia.ai/tts/bytes",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Cartesia-Version": self.API_VERSION,
                "Content-Type": "application/json",
            },
            json={
                "model_id": model,
                "transcript": request.text,
                "voice": {"id": voice},
                "language": request.language,
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": sample_rate,
                },
            },
        ) as response:
            return binary_audio(response, "pcm_s16le", sample_rate)


class InworldProvider(Provider):
    """https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech-stream"""

    def synthesize(self, model: str, request: SynthesisRequest) -> AudioResponse:
        voice = _voice(request)
        sample_rate = _sample_rate(request, {8000, 16000, 22050, 24000, 32000, 44100, 48000})
        if len(request.text.encode("utf-16-le")) // 2 > 4000:
            raise APIError("Inworld streaming text exceeds 4000 UTF-16 code units")
        with self.post(
            "https://api.inworld.ai/tts/v1/voice:stream",
            headers={
                "Authorization": f"Basic {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "text": request.text,
                "voiceId": voice,
                "modelId": model,
                "language": request.language,
                "audioConfig": {"audioEncoding": "PCM", "sampleRateHertz": sample_rate},
            },
        ) as response:
            pcm = _PCM()
            for line in _lines(response):
                if not line.strip():
                    continue
                payload = _json(line)
                error = payload.get("error")
                if error is not None or payload.get("code") not in (None, 0):
                    code = error.get("code") if isinstance(error, dict) else payload.get("code")
                    raise APIError("Inworld synthesis stream returned an error", retryable=code in (4, 8, 13, 14))
                result = payload.get("result")
                if not isinstance(result, dict):
                    raise APIError("Invalid result in Inworld synthesis stream")
                if "audioContent" in result:
                    pcm.append(result["audioContent"])
            return pcm.finish(response, sample_rate)


class DeepgramProvider(Provider):
    """https://developers.deepgram.com/reference/text-to-speech/speak-request"""

    def synthesize(self, model: str, request: SynthesisRequest) -> AudioResponse:
        if request.reference_audio is not None:
            raise APIError("Deepgram does not support inline reference cloning")
        if request.voice not in (None, model):
            raise APIError("Deepgram voice is part of the model ID; select a voice-specific model")
        sample_rate = _sample_rate(request, {8000, 16000, 24000, 32000, 48000})
        with self.post(
            "https://api.deepgram.com/v1/speak",
            headers={"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"},
            params={"model": model, "encoding": "linear16", "container": "none", "sample_rate": sample_rate},
            json={"text": request.text},
        ) as response:
            return binary_audio(response, "pcm_s16le", sample_rate)


class SmallestProvider(Provider):
    """https://docs.smallest.ai/api-reference/models/text-to-speech/synthesize-speech-sse"""

    def synthesize(self, model: str, request: SynthesisRequest) -> AudioResponse:
        voice = _voice(request)
        sample_rate = _sample_rate(request, {8000, 16000, 24000, 44100})
        if not request.text.strip() or len(request.text.strip()) > 8000:
            raise APIError("Smallest requires 1–8000 text characters")
        with self.post(
            "https://api.smallest.ai/waves/v1/tts/live",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            json={
                "text": request.text,
                "voice_id": voice,
                "model": model,
                "sample_rate": sample_rate,
                "output_format": "pcm",
                # The registry selects a voice trained in this language.
                "language": request.language,
            },
        ) as response:
            pcm = _PCM()
            completed = False
            for event, payload in _sse(response):
                status = payload.get("status")
                if event == b"error" or "error" in payload or status not in (None, "200", "206", 200, 206):
                    raise APIError("Smallest synthesis stream returned an error", retryable=status in ("429", "500", "503", 429, 500, 503))
                if completed:
                    raise APIError("Smallest returned data after stream completion")
                if "audio" in payload:
                    pcm.append(payload["audio"])
                if payload.get("done") is True:
                    if status not in (None, "200", 200):
                        raise APIError("Smallest returned an unsuccessful completion status")
                    completed = True
            if not completed:
                raise APIError("Smallest synthesis stream ended without completion", retryable=True)
            return pcm.finish(response, sample_rate)
