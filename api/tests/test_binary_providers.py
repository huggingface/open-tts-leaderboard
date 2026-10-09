import base64
import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

from requests.structures import CaseInsensitiveDict

from api.providers.base import APIError, SynthesisRequest
from api.providers.binary import (
    CartesiaProvider,
    DeepgramProvider,
    ElevenLabsProvider,
    InworldProvider,
    SmallestProvider,
)


class Response:
    def __init__(self, chunks, content_type="audio/pcm"):
        self.chunks = chunks
        self.headers = CaseInsensitiveDict({"Content-Type": content_type, "X-Request-Id": "test-request"})
        self.status_code = 200

    def iter_content(self, chunk_size=1024):
        yield from self.chunks


def attach_response(provider, response):
    @contextmanager
    def post(*args, **kwargs):
        yield response

    provider.post = Mock(side_effect=post)
    return provider


def audio_message(data, **kwargs):
    return json.dumps({"audio": base64.b64encode(data).decode(), **kwargs}).encode()


def inworld_message(data):
    return json.dumps({"result": {"audioContent": base64.b64encode(data).decode()}}).encode()


class BinaryProvidersTests(unittest.TestCase):
    def request(self, **kwargs):
        return SynthesisRequest(**{"text": "Hello", "language": "en", "voice": "voice", **kwargs})

    def test_elevenlabs_pcm_endpoint_and_auth(self):
        provider = attach_response(ElevenLabsProvider("key"), Response([b"\x01", b"\x00\x02\x00"]))
        output = provider.synthesize("eleven_v4", self.request(voice="voice/id"))
        args, kwargs = provider.post.call_args
        self.assertEqual(args[0], "https://api.elevenlabs.io/v1/text-to-dialogue/stream")
        self.assertEqual(kwargs["headers"]["xi-api-key"], "key")
        self.assertEqual(kwargs["params"], {"output_format": "pcm_24000"})
        self.assertEqual(kwargs["json"]["model_id"], "eleven_v4")
        self.assertEqual(kwargs["json"]["language_code"], "en")
        self.assertEqual(kwargs["json"]["inputs"], [{"text": "Hello", "voice_id": "voice/id"}])
        self.assertEqual(output.audio_bytes, b"\x01\x00\x02\x00")
        self.assertEqual(output.encoding, "pcm_s16le")
        self.assertIsNotNone(output.first_audio_at)

    def test_elevenlabs_multilingual_v2_omits_unsupported_language_field(self):
        provider = attach_response(ElevenLabsProvider("key"), Response([b"\0\0"]))
        provider.synthesize("eleven_multilingual_v2", self.request())
        self.assertNotIn("language_code", provider.post.call_args.kwargs["json"])

    def test_elevenlabs_legacy_tts_escapes_voice_path(self):
        provider = attach_response(ElevenLabsProvider("key"), Response([b"\0\0"]))
        provider.synthesize("eleven_flash_v2_5", self.request(voice="voice/id"))
        self.assertEqual(provider.post.call_args.args[0], "https://api.elevenlabs.io/v1/text-to-speech/voice%2Fid/stream")
        self.assertEqual(provider.post.call_args.kwargs["json"]["text"], "Hello")

    def test_elevenlabs_dialogue_rejects_long_stream_input_before_post(self):
        provider = ElevenLabsProvider("key")
        provider.post = Mock()
        with self.assertRaises(APIError):
            provider.synthesize("eleven_v4", self.request(text="a" * 2001))
        provider.post.assert_not_called()

    def websocket_module(self, responses):
        class WebSocketException(Exception):
            pass

        class WebSocketBadStatusException(WebSocketException):
            def __init__(self, status_code):
                super().__init__("private URL and key")
                self.status_code = status_code

        socket = Mock()
        socket.recv.side_effect = responses
        socket.getheaders.return_value = {"x-request-id": "socket-request"}
        module = SimpleNamespace(
            create_connection=Mock(return_value=socket),
            WebSocketException=WebSocketException,
            WebSocketBadStatusException=WebSocketBadStatusException,
        )
        return module, socket

    def test_elevenlabs_turbo_native_dialogue_websocket(self):
        module, socket = self.websocket_module([
            audio_message(b"\x01").decode(),
            '{"is_final_audio_for_turn":true}',
            audio_message(b"\x00", is_final=True).decode(),
        ])
        provider = ElevenLabsProvider("secret", timeout=45)
        provider.post = Mock()
        with patch.dict("sys.modules", {"websocket": module}):
            output = provider.synthesize("eleven_v4_turbo", self.request())
        args, kwargs = module.create_connection.call_args
        parsed = urlsplit(args[0])
        self.assertEqual(parsed.scheme, "wss")
        self.assertEqual(parsed.netloc, "api.elevenlabs.io")
        self.assertEqual(parsed.path, "/v1/text-to-dialogue/stream-input")
        self.assertEqual(parse_qs(parsed.query), {"model_id": ["eleven_v4_turbo"], "output_format": ["pcm_24000"], "language_code": ["en"]})
        self.assertEqual(kwargs["header"], {"xi-api-key": "secret"})
        self.assertNotIn("secret", args[0])
        self.assertEqual([json.loads(call.args[0]) for call in socket.send.call_args_list], [
            {"voices": ["voice"]},
            {"inputs": [{"text": "Hello", "voice_id": "voice", "new_turn": False}]},
            {"close_socket": True},
        ])
        self.assertEqual(socket.recv.call_count, 3)
        self.assertEqual(output.audio_bytes, b"\x01\x00")
        self.assertEqual(output.request_id, "socket-request")
        self.assertIsNotNone(output.first_audio_at)
        socket.close.assert_called_once_with(timeout=1)
        provider.post.assert_not_called()

    def test_elevenlabs_turbo_requires_final_frame_and_rejects_stream_errors(self):
        for messages in [
            [""],
            [audio_message(b"\0\0").decode(), ""],
            [audio_message(b"\0\0").decode(), '{"error":"private provider text"}'],
            ['{"is_final":true}'],
            ['{"audio":"!","is_final":true}'],
        ]:
            with self.subTest(messages=messages):
                module, socket = self.websocket_module(messages)
                with patch.dict("sys.modules", {"websocket": module}), self.assertRaises(APIError) as raised:
                    ElevenLabsProvider("key").synthesize("eleven_v4_turbo", self.request())
                self.assertNotIn("private provider text", str(raised.exception))
                socket.close.assert_called_once_with(timeout=1)

    def test_elevenlabs_turbo_handshake_auth_and_transport_errors_are_redacted(self):
        for status in (401, 429, 503):
            with self.subTest(status=status):
                module, _ = self.websocket_module([])
                module.create_connection.side_effect = module.WebSocketBadStatusException(status)
                with patch.dict("sys.modules", {"websocket": module}), self.assertRaises(APIError) as raised:
                    ElevenLabsProvider("key").synthesize("eleven_v4_turbo", self.request())
                self.assertEqual(raised.exception.status_code, status)
                self.assertEqual(raised.exception.retryable, status != 401)
                self.assertTrue(raised.exception.__suppress_context__)
                self.assertNotIn("private URL and key", str(raised.exception))
        module, socket = self.websocket_module([])
        socket.recv.side_effect = module.WebSocketException("private URL and key")
        with patch.dict("sys.modules", {"websocket": module}), self.assertRaises(APIError) as raised:
            ElevenLabsProvider("key").synthesize("eleven_v4_turbo", self.request())
        self.assertTrue(raised.exception.retryable)
        self.assertTrue(raised.exception.__suppress_context__)
        socket.close.assert_called_once_with(timeout=1)

    def test_cartesia_current_version_and_voice_schema(self):
        provider = attach_response(CartesiaProvider("key"), Response([b"\0\0"]))
        provider.synthesize("sonic-3.6-2026-08-27", self.request(language="fr"))
        args, kwargs = provider.post.call_args
        self.assertEqual(args[0], "https://api.cartesia.ai/tts/bytes")
        self.assertEqual(kwargs["headers"]["Cartesia-Version"], "2026-08-14")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer key")
        self.assertEqual(kwargs["json"]["voice"], {"id": "voice"})
        self.assertEqual(kwargs["json"]["language"], "fr")
        self.assertEqual(kwargs["json"]["output_format"], {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000})

    def test_deepgram_voice_is_model_and_output_has_no_container(self):
        provider = attach_response(DeepgramProvider("key"), Response([b"\0\0"]))
        provider.synthesize("aura-2-thalia-en", self.request(voice=None))
        args, kwargs = provider.post.call_args
        self.assertEqual(args[0], "https://api.deepgram.com/v1/speak")
        self.assertEqual(kwargs["headers"]["Authorization"], "Token key")
        self.assertEqual(kwargs["params"], {"model": "aura-2-thalia-en", "encoding": "linear16", "container": "none", "sample_rate": 24000})
        self.assertEqual(kwargs["json"], {"text": "Hello"})

    def test_inworld_ndjson_spans_http_boundaries_and_finishes_at_eof(self):
        stream = inworld_message(b"\x01") + b"\r\n" + inworld_message(b"\x00") + b"\n" + b'{"result":{"timestampInfo":{}}}'
        chunks = [stream[i:i + 3] for i in range(0, len(stream), 3)]
        provider = attach_response(InworldProvider("already-base64-key"), Response(chunks, "application/json"))
        with patch("api.providers.binary.time.perf_counter", return_value=12.5):
            output = provider.synthesize("inworld-tts-2", self.request())
        self.assertEqual(output.audio_bytes, b"\x01\x00")
        self.assertEqual(output.first_audio_at, 12.5)
        self.assertEqual(output.request_id, "test-request")
        kwargs = provider.post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["Authorization"], "Basic already-base64-key")
        self.assertEqual(kwargs["json"]["audioConfig"], {"audioEncoding": "PCM", "sampleRateHertz": 24000})
        self.assertEqual(kwargs["json"]["voiceId"], "voice")

    def test_inworld_http_200_terminal_error_is_not_success(self):
        stream = inworld_message(b"\0\0") + b'\n{"error":{"code":14,"message":"private request text"}}\n'
        provider = attach_response(InworldProvider("key"), Response([stream], "application/json"))
        with self.assertRaises(APIError) as raised:
            provider.synthesize("inworld-tts-2", self.request())
        self.assertTrue(raised.exception.retryable)
        self.assertNotIn("private request text", str(raised.exception))

    def test_inworld_counts_utf16_units_before_network_request(self):
        provider = InworldProvider("key")
        provider.post = Mock()
        with self.assertRaises(APIError):
            provider.synthesize("inworld-tts-2", self.request(text="😀" * 2001))
        provider.post.assert_not_called()

    def test_smallest_sse_pcm_events_and_required_completion(self):
        stream = b": keepalive\r\n\r\nevent: audio\r\n" + b"data: " + audio_message(b"\x01", status="206", done=False) + b"\r\n\r\n"
        stream += b"event: audio\ndata: " + audio_message(b"\x00", status="206", done=False) + b'\n\ndata: {"done":true,"status":"200"}\n\n'
        provider = attach_response(SmallestProvider("key"), Response([stream[i:i + 5] for i in range(0, len(stream), 5)], "text/event-stream"))
        output = provider.synthesize("lightning_v3.1_pro", self.request(voice="manon", language="fr"))
        self.assertEqual(output.audio_bytes, b"\x01\x00")
        self.assertEqual(output.request_id, "test-request")
        args, kwargs = provider.post.call_args
        self.assertEqual(args[0], "https://api.smallest.ai/waves/v1/tts/live")
        self.assertEqual(kwargs["json"]["output_format"], "pcm")
        self.assertEqual(kwargs["json"]["model"], "lightning_v3.1_pro")
        self.assertEqual(kwargs["json"]["voice_id"], "manon")
        self.assertEqual(kwargs["json"]["language"], "fr")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer key")

    def test_smallest_accepts_multiline_sse_json(self):
        stream = b'data: {"audio":\ndata: "AAA=", "done":false}\n\ndata: {"done":true}\n\n'
        provider = attach_response(SmallestProvider("key"), Response([stream], "text/event-stream"))
        self.assertEqual(provider.synthesize("lightning_v3.1", self.request()).audio_bytes, b"\0\0")

    def test_smallest_rejects_empty_truncated_and_terminal_errors(self):
        audio = b"data: " + audio_message(b"\0\0", done=False, status="206") + b"\n\n"
        streams = [
            b"",
            audio,
            audio + b'data: {"done":true}',
            b'data: {"done":true}\n\n',
            audio + b'event: error\ndata: {"message":"private provider message"}\n\n',
            audio + b'data: {"status":"500","message":"private provider message"}\n\n',
            audio + b'data: {"status":"206","done":true}\n\n',
            audio + b'data: {"done":true}\n\ndata: {"error":"private provider message"}\n\n',
        ]
        for stream in streams:
            with self.subTest(stream=stream):
                provider = attach_response(SmallestProvider("key"), Response([stream], "text/event-stream"))
                with self.assertRaises(APIError) as raised:
                    provider.synthesize("lightning_v3.1", self.request())
                self.assertNotIn("private provider message", str(raised.exception))

    def test_framed_streams_reject_invalid_audio_json_and_pcm(self):
        for cls, bodies, content_type in [
            (InworldProvider, [b'{"result":{"audioContent":"!"}}\n', b'[]\n', b'not JSON\n', inworld_message(b"\0") + b"\n"], "application/json"),
            (SmallestProvider, [b'data: {"audio":"!"}\n\n', b'data: []\n\n', b'data: not JSON\n\n', b'data: {"audio":"AA=="}\n\ndata: {"done":true}\n\n'], "text/event-stream"),
        ]:
            for body in bodies:
                with self.subTest(provider=cls.__name__, body=body):
                    provider = attach_response(cls("key"), Response([body], content_type))
                    with self.assertRaises(APIError):
                        provider.synthesize("model", self.request())

    def test_existing_voice_adapters_reject_reference_audio_before_post(self):
        for cls in [ElevenLabsProvider, CartesiaProvider, InworldProvider, DeepgramProvider, SmallestProvider]:
            with self.subTest(provider=cls.__name__):
                provider = cls("key")
                provider.post = Mock()
                with self.assertRaises(APIError):
                    provider.synthesize("model", self.request(reference_audio=b"reference"))
                provider.post.assert_not_called()

    def test_invalid_sample_rate_and_deepgram_voice_fail_before_post(self):
        for cls in [ElevenLabsProvider, CartesiaProvider, InworldProvider, DeepgramProvider, SmallestProvider]:
            with self.subTest(provider=cls.__name__):
                provider = cls("key")
                provider.post = Mock()
                with self.assertRaises(APIError):
                    provider.synthesize("model", self.request(voice=None if cls is DeepgramProvider else "voice", sample_rate=12345))
                provider.post.assert_not_called()
        provider = DeepgramProvider("key")
        provider.post = Mock()
        with self.assertRaises(APIError):
            provider.synthesize("aura-2-thalia-en", self.request(voice="different-voice"))
        provider.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
