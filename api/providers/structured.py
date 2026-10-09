"""TTS adapters for providers with structured requests or audio responses.

Provider schemas are linked alongside the implementations. No voice enrollment
or remote reference downloads are performed by these adapters.
"""

import base64
import binascii
import io
import time
import wave

from .base import (
    APIError,
    AudioResponse,
    Provider,
    SynthesisRequest,
    binary_audio,
    iter_sse,
)


def _base64_audio(value, provider):
    try:
        audio = base64.b64decode(value, validate=True)
    except (ValueError, TypeError, binascii.Error):
        raise APIError(f"{provider} returned invalid base64 audio") from None
    if not audio:
        raise APIError(f"{provider} returned empty audio")
    return audio


def _json_response(response, provider):
    try:
        payload = response.json()
    except ValueError:
        raise APIError(f"{provider} returned invalid JSON") from None
    if not isinstance(payload, dict):
        raise APIError(f"{provider} returned an invalid response")
    if payload.get("error"):
        # Response messages can echo input or credentials; never log them.
        raise APIError(f"{provider} reported a synthesis error")
    return payload


class FishProvider(Provider):
    """https://docs.fish.audio/api-reference/openapi.json"""

    supports_reference = True

    def synthesize(self, model: str, request: SynthesisRequest) -> AudioResponse:
        payload = {
            "text": request.text,
            "format": "wav",
            "sample_rate": request.sample_rate,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "model": model}
        kwargs = {}
        if request.reference_audio is not None:
            if not request.reference_text.strip():
                raise APIError("Fish Audio reference audio requires its transcript")
            # Inline references contain raw audio bytes and require MessagePack.
            import msgpack

            payload["references"] = [
                {"audio": request.reference_audio, "text": request.reference_text}
            ]
            headers["Content-Type"] = "application/msgpack"
            kwargs["data"] = msgpack.packb(payload, use_bin_type=True)
        else:
            if request.voice:
                payload["reference_id"] = request.voice
            headers["Content-Type"] = "application/json"
            kwargs["json"] = payload
        with self.post("https://api.fish.audio/v1/tts", headers=headers, **kwargs) as response:
            # WAV headers alone are not playable audio; use completion latency.
            return binary_audio(response, "wav", request.sample_rate)


class MiniMaxProvider(Provider):
    """https://platform.minimax.io/docs/api-reference/speech-t2a-http"""

    supports_reference = False

    def synthesize(self, model: str, request: SynthesisRequest) -> AudioResponse:
        if request.reference_audio is not None:
            raise APIError("MiniMax inline reference cloning is not supported")
        if not request.voice:
            raise APIError("MiniMax requires a voice ID")
        payload = {
            "model": model,
            "text": request.text,
            "stream": True,
            # The final packet repeats the entire audio. Preserve status=1
            # chunks, falling back to status=2 audio only if no chunks arrived.
            "stream_options": {"exclude_aggregated_audio": False},
            "output_format": "hex",
            "language_boost": "auto",
            "voice_setting": {"voice_id": request.voice},
            "audio_setting": {
                "sample_rate": request.sample_rate,
                "format": "mp3",
                "channel": 1,
            },
        }
        chunks = []
        final_audio = b""
        completed = False
        request_id = None
        sample_rate = request.sample_rate
        with self.post(
            "https://api.minimax.io/v1/t2a_v2",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
        ) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            events = [_json_response(response, "MiniMax")] if "application/json" in content_type else iter_sse(response)
            for event in events:
                base_response = event.get("base_resp") or {}
                code = base_response.get("status_code", 0)
                if code != 0:
                    raise APIError(
                        f"MiniMax reported synthesis error {code}",
                        retryable=code in {1000, 1001, 1002, 1039},
                    )
                if event.get("error"):
                    raise APIError("MiniMax reported a synthesis error")
                request_id = event.get("trace_id") or request_id
                metadata = event.get("extra_info") or {}
                sample_rate = metadata.get("audio_sample_rate") or sample_rate
                data = event.get("data") or {}
                status = data.get("status")
                try:
                    audio = bytes.fromhex(data.get("audio") or "")
                except (ValueError, TypeError):
                    raise APIError("MiniMax returned invalid hexadecimal audio") from None
                if status == 1 and audio:
                    chunks.append(audio)
                elif status == 2:
                    completed = True
                    final_audio = audio
        if not completed:
            raise APIError("MiniMax audio stream ended before completion", retryable=True)
        audio = b"".join(chunks) if chunks else final_audio
        if not audio:
            raise APIError("MiniMax returned empty audio")
        # Compressed chunks can contain headers or incomplete frames. Do not
        # equate the first network byte with the time of playable audio.
        return AudioResponse(audio, "mp3", sample_rate, request_id=request_id)


class GeminiProvider(Provider):
    """https://ai.google.dev/gemini-api/docs/speech-generation"""

    supports_reference = False

    def synthesize(self, model: str, request: SynthesisRequest) -> AudioResponse:
        if request.reference_audio is not None:
            raise APIError("Gemini cloning requires a separate consent recording and is not supported")
        payload = {
            "model": model,
            # This field is a verbatim transcript, without prompting or style
            # directives that might affect the leaderboard transcription.
            "input": [{"type": "user_input", "content": [{"type": "text", "text": request.text}]}],
            "response_format": {
                "type": "audio",
                "mime_type": "audio/l16",
                "sample_rate": 24000,
            },
            "generation_config": {"speech_config": [{"voice": request.voice or "Kore"}]},
            "stream": True,
            "store": False,
        }
        chunks = []
        first_audio_at = None
        completed = False
        request_id = None
        total_size = 0
        with self.post(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json=payload,
        ) as response:
            for event in iter_sse(response):
                event_type = event.get("event_type")
                if event_type == "error" or event.get("error"):
                    raise APIError("Gemini reported a synthesis error")
                if event_type == "step.delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "audio" and delta.get("data"):
                        mime_type = delta.get("mime_type", "audio/l16").split(";")[0]
                        if mime_type != "audio/l16":
                            raise APIError("Gemini returned an unexpected audio encoding")
                        audio = _base64_audio(delta["data"], "Gemini")
                        chunks.append(audio)
                        total_size += len(audio)
                        # A PCM16 sample needs two bytes, even if the transport
                        # happens to split one sample over consecutive chunks.
                        if first_audio_at is None and total_size >= 2:
                            first_audio_at = time.perf_counter()
                elif event_type in {"interaction.created", "interaction.completed"}:
                    interaction = event.get("interaction") or {}
                    request_id = interaction.get("id") or request_id
                    if event_type == "interaction.completed":
                        if interaction.get("status") != "completed":
                            raise APIError("Gemini synthesis did not complete successfully")
                        completed = True
        if not completed:
            raise APIError("Gemini audio stream ended before completion", retryable=True)
        audio = b"".join(chunks)
        if not audio or len(audio) % 2:
            raise APIError("Gemini returned empty or incomplete PCM audio")
        return AudioResponse(audio, "pcm_s16le", 24000, first_audio_at, request_id)


class MistralProvider(Provider):
    """https://docs.mistral.ai/studio/audio/text_to_speech/speech"""

    supports_reference = True

    def synthesize(self, model: str, request: SynthesisRequest) -> AudioResponse:
        payload = {"model": model, "input": request.text, "response_format": "wav", "stream": False}
        if request.reference_audio is not None:
            payload["ref_audio"] = base64.b64encode(request.reference_audio).decode("ascii")
        elif request.voice:
            payload["voice_id"] = request.voice
        else:
            raise APIError("Mistral requires a saved voice ID or reference audio")
        with self.post(
            "https://api.mistral.ai/v1/audio/speech",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
        ) as response:
            result = _json_response(response, "Mistral")
            audio = _base64_audio(result.get("audio_data"), "Mistral")
            try:
                with wave.open(io.BytesIO(audio), "rb") as wav:
                    sample_rate = wav.getframerate()
            except (wave.Error, EOFError):
                raise APIError("Mistral returned invalid WAV audio") from None
            # No API sample-rate override is documented. Retain the returned
            # WAV rate; the evaluation runner retains it when saving output.
            return AudioResponse(audio, "wav", sample_rate, request_id=response.headers.get("x-request-id"))
