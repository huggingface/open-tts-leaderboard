"""Small provider contract; transport errors never expose credentials or response bodies."""

import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests


@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    language: str
    voice: str | None
    reference_audio: bytes | None = None
    reference_text: str = ""
    sample_rate: int = 24000


@dataclass(frozen=True)
class AudioResponse:
    audio_bytes: bytes
    encoding: str
    sample_rate: int
    first_audio_at: float | None = None
    request_id: str | None = None


class APIError(Exception):
    def __init__(self, message, status_code=None, retryable=False, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after


def retry_after_seconds(value):
    if not value:
        return None
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


class Provider:
    supports_reference = False

    def __init__(self, api_key, timeout=120):
        self.api_key = api_key
        self.timeout = timeout

    @contextmanager
    def post(self, url, *, headers=None, json=None, data=None, params=None):
        response = None
        try:
            response = requests.post(
                url, headers=headers, json=json, data=data, params=params,
                timeout=(10, self.timeout), stream=True,
            )
            status = response.status_code
            if not 200 <= status < 300:
                raise APIError(
                    f"Provider returned HTTP {status}", status_code=status,
                    retryable=status in (408, 429) or status >= 500,
                    retry_after=retry_after_seconds(response.headers.get("Retry-After")),
                )
            yield response
        except requests.RequestException:
            # Exception strings can include URLs, query keys and response bodies.
            raise APIError("Provider connection failed", retryable=True) from None
        finally:
            if response is not None:
                response.close()

    def synthesize(self, model, request):
        raise NotImplementedError


def binary_audio(response, encoding, sample_rate):
    """Only raw PCM establishes playable first audio; container bytes do not."""
    content_type = response.headers.get("Content-Type", "").lower()
    if "json" in content_type or content_type.startswith("text/"):
        raise APIError("Provider returned an error document instead of audio")
    chunks, size, first = [], 0, None
    frame_size = {"pcm_s16le": 2, "pcm_f32le": 4}.get(encoding)
    for chunk in response.iter_content(chunk_size=1024):
        if not chunk:
            continue
        chunks.append(chunk)
        size += len(chunk)
        if first is None and frame_size and size >= frame_size:
            first = time.perf_counter()
    if not chunks:
        raise APIError("Provider returned no audio")
    if frame_size and size % frame_size:
        raise APIError("Provider returned incomplete PCM frames")
    audio = b"".join(chunks)
    if frame_size and audio.startswith((b"RIFF", b"OggS", b"ID3", b"fLaC")):
        raise APIError("Provider returned container audio instead of requested raw PCM")
    return AudioResponse(
        audio, encoding, sample_rate, first,
        response.headers.get("x-request-id") or response.headers.get("request-id"),
    )


def iter_sse(response):
    """Parse complete SSE data events, including a final event without a blank line."""
    lines = []

    def parse():
        data = "\n".join(lines)
        if not data or data == "[DONE]":
            return None
        try:
            event = json.loads(data)
        except (ValueError, TypeError) as exc:
            raise APIError("Provider returned malformed SSE data") from exc
        if not isinstance(event, dict):
            raise APIError("Provider returned an unexpected SSE event")
        if event.get("error"):
            raise APIError("Provider reported a streaming error")
        return event

    for line in response.iter_lines(decode_unicode=True):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line:
            event = parse()
            if event is not None:
                yield event
            lines.clear()
        elif line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
    event = parse()
    if event is not None:
        yield event
