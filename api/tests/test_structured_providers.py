"""Schema and audio reconstruction checks without billable API requests."""

import base64
import io
import json
import unittest
import wave
from contextlib import contextmanager
from unittest.mock import patch

from api.providers.base import APIError, SynthesisRequest
from api.providers.structured import (
    FishProvider,
    GeminiProvider,
    MiniMaxProvider,
    MistralProvider,
)


class FakeResponse:
    def __init__(self, events=(), payload=None, chunks=(), headers=None):
        self.events = events
        self.payload = payload
        self.chunks = chunks
        self.headers = headers or {"Content-Type": "text/event-stream"}

    def iter_lines(self, **kwargs):
        for event in self.events:
            yield ("data: " + json.dumps(event)).encode()
            yield b""

    def iter_content(self, **kwargs):
        yield from self.chunks

    def json(self):
        return self.payload


def wav_bytes(rate=24000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\x00\x00\x01\x00")
    return buffer.getvalue()


class StructuredProviderTests(unittest.TestCase):
    def request(self, **kwargs):
        return SynthesisRequest(text="Hello.", language="en", voice="voice-id", **kwargs)

    def transport(self, provider, response):
        calls = []

        @contextmanager
        def post(url, **kwargs):
            calls.append((url, kwargs))
            yield response

        provider.post = post
        return calls

    def test_fish_saved_voice_uses_json_and_model_header(self):
        provider = FishProvider("secret")
        audio = wav_bytes()
        calls = self.transport(provider, FakeResponse(chunks=[audio[:44], audio[44:]], headers={"Content-Type": "audio/wav"}))
        result = provider.synthesize("s2.1-pro", self.request())
        url, kwargs = calls[0]
        self.assertEqual(url, "https://api.fish.audio/v1/tts")
        self.assertEqual(kwargs["headers"]["model"], "s2.1-pro")
        self.assertEqual(kwargs["json"]["reference_id"], "voice-id")
        self.assertEqual(result.audio_bytes, audio)
        self.assertIsNone(result.first_audio_at)

    def test_fish_inline_reference_uses_messagepack_raw_bytes(self):
        import msgpack

        provider = FishProvider("secret")
        calls = self.transport(provider, FakeResponse(chunks=[wav_bytes()], headers={"Content-Type": "audio/wav"}))
        provider.synthesize("s2-pro", self.request(reference_audio=b"reference", reference_text="Reference words."))
        kwargs = calls[0][1]
        self.assertNotIn("json", kwargs)
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/msgpack")
        self.assertEqual(
            msgpack.unpackb(kwargs["data"], raw=False)["references"],
            [{"audio": b"reference", "text": "Reference words."}],
        )

    def test_fish_requires_reference_transcript_before_post(self):
        provider = FishProvider("secret")
        calls = self.transport(provider, FakeResponse())
        with self.assertRaisesRegex(APIError, "transcript"):
            provider.synthesize("s2-pro", self.request(reference_audio=b"reference"))
        self.assertEqual(calls, [])

    def test_minimax_does_not_duplicate_final_aggregated_audio(self):
        provider = MiniMaxProvider("secret")
        events = [
            {"data": {"audio": b"part1".hex(), "status": 1}, "base_resp": {"status_code": 0}},
            {"data": {"audio": b"part2".hex(), "status": 1}},
            {"data": {"audio": b"part1part2".hex(), "status": 2}, "trace_id": "trace"},
        ]
        calls = self.transport(provider, FakeResponse(events=events))
        result = provider.synthesize("speech-2.8-hd", self.request())
        self.assertEqual(result.audio_bytes, b"part1part2")
        self.assertEqual(result.request_id, "trace")
        self.assertIsNone(result.first_audio_at)
        self.assertFalse(calls[0][1]["json"]["stream_options"]["exclude_aggregated_audio"])

    def test_minimax_accepts_only_final_audio_response(self):
        provider = MiniMaxProvider("secret")
        response = FakeResponse(
            payload={"data": {"audio": b"audio".hex(), "status": 2}, "extra_info": {"audio_sample_rate": 32000}},
            headers={"Content-Type": "application/json"},
        )
        self.transport(provider, response)
        result = provider.synthesize("speech-2.8-turbo", self.request())
        self.assertEqual(result.audio_bytes, b"audio")
        self.assertEqual(result.sample_rate, 32000)

    def test_minimax_handles_http200_rate_limit_without_echoing_body(self):
        provider = MiniMaxProvider("secret")
        self.transport(provider, FakeResponse(events=[{
            "base_resp": {"status_code": 1002, "status_msg": "sensitive input secret"}
        }]))
        with self.assertRaises(APIError) as error:
            provider.synthesize("speech-2.8-hd", self.request())
        self.assertTrue(error.exception.retryable)
        self.assertNotIn("secret", str(error.exception))

    def test_minimax_rejects_truncated_stream(self):
        provider = MiniMaxProvider("secret")
        self.transport(provider, FakeResponse(events=[{"data": {"audio": "0000", "status": 1}}]))
        with self.assertRaisesRegex(APIError, "before completion"):
            provider.synthesize("speech-2.8-hd", self.request())

    def test_minimax_rejects_malformed_hex(self):
        provider = MiniMaxProvider("secret")
        self.transport(provider, FakeResponse(events=[{"data": {"audio": "not-hex", "status": 2}}]))
        with self.assertRaisesRegex(APIError, "hexadecimal"):
            provider.synthesize("speech-2.8-hd", self.request())

    def test_gemini_verbatim_input_and_pcm_first_frame(self):
        provider = GeminiProvider("secret")
        events = [
            {"event_type": "interaction.created", "interaction": {"id": "interaction-id"}},
            {"event_type": "step.delta", "delta": {"type": "audio", "data": "AA=="}},
            {"event_type": "step.delta", "delta": {"type": "audio", "data": "AQ=="}},
            {"event_type": "interaction.completed", "interaction": {"status": "completed"}},
        ]
        calls = self.transport(provider, FakeResponse(events=events))
        with patch("api.providers.structured.time.perf_counter", return_value=12.5) as timer:
            result = provider.synthesize("gemini-3.8-flash-tts", self.request())
        self.assertEqual(timer.call_count, 1)
        self.assertEqual(result.first_audio_at, 12.5)
        self.assertEqual(result.audio_bytes, b"\x00\x01")
        self.assertEqual(result.encoding, "pcm_s16le")
        self.assertEqual(result.request_id, "interaction-id")
        payload = calls[0][1]["json"]
        self.assertEqual(payload["input"][0]["content"], [{"type": "text", "text": "Hello."}])
        self.assertEqual(payload["response_format"]["mime_type"], "audio/l16")
        self.assertFalse(payload["store"])

    def test_gemini_rejects_failed_final_status(self):
        provider = GeminiProvider("secret")
        self.transport(provider, FakeResponse(events=[{
            "event_type": "interaction.completed", "interaction": {"status": "incomplete"}
        }]))
        with self.assertRaisesRegex(APIError, "successfully"):
            provider.synthesize("gemini-3.8-flash-lite-tts", self.request())

    def test_gemini_rejects_error_after_partial_audio(self):
        provider = GeminiProvider("secret")
        self.transport(provider, FakeResponse(events=[
            {"event_type": "step.delta", "delta": {"type": "audio", "data": "AAA="}},
            {"event_type": "error", "error": {"message": "sensitive input secret"}},
        ]))
        with self.assertRaises(APIError) as error:
            provider.synthesize("gemini-3.8-flash-tts", self.request())
        self.assertNotIn("secret", str(error.exception))

    def test_gemini_rejects_truncated_stream(self):
        provider = GeminiProvider("secret")
        self.transport(provider, FakeResponse(events=[{
            "event_type": "step.delta", "delta": {"type": "audio", "data": "AAA="}
        }]))
        with self.assertRaisesRegex(APIError, "before completion"):
            provider.synthesize("gemini-3.8-flash-tts", self.request())

    def test_gemini_rejects_reference_cloning_before_post(self):
        provider = GeminiProvider("secret")
        calls = self.transport(provider, FakeResponse())
        with self.assertRaisesRegex(APIError, "consent"):
            provider.synthesize("gemini-3.8-flash-tts", self.request(reference_audio=b"reference"))
        self.assertEqual(calls, [])

    def test_mistral_inline_reference_and_returned_wav_rate(self):
        provider = MistralProvider("secret")
        audio = wav_bytes(rate=22050)
        calls = self.transport(provider, FakeResponse(payload={"audio_data": base64.b64encode(audio).decode()}))
        result = provider.synthesize("voxtral-mini-tts-2603", self.request(reference_audio=b"reference"))
        payload = calls[0][1]["json"]
        self.assertEqual(payload["ref_audio"], base64.b64encode(b"reference").decode())
        self.assertNotIn("voice_id", payload)
        self.assertEqual(result.audio_bytes, audio)
        self.assertEqual(result.sample_rate, 22050)
        self.assertIsNone(result.first_audio_at)

    def test_mistral_saved_voice(self):
        provider = MistralProvider("secret")
        calls = self.transport(provider, FakeResponse(payload={"audio_data": base64.b64encode(wav_bytes()).decode()}))
        provider.synthesize("voxtral-mini-tts-2603", self.request())
        self.assertEqual(calls[0][1]["json"]["voice_id"], "voice-id")

    def test_mistral_rejects_invalid_audio_without_echoing_body(self):
        provider = MistralProvider("secret")
        self.transport(provider, FakeResponse(payload={"audio_data": "secret invalid audio"}))
        with self.assertRaises(APIError) as error:
            provider.synthesize("voxtral-mini-tts-2603", self.request())
        self.assertNotIn("secret", str(error.exception))


if __name__ == "__main__":
    unittest.main()
