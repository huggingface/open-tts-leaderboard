"""Provider registry, mirroring the Open ASR Leaderboard's API backend."""

import os

from .base import APIError, AudioResponse, Provider, SynthesisRequest

KEY_ENV = {
    "elevenlabs": "ELEVENLABS_API_KEY", "cartesia": "CARTESIA_API_KEY",
    "minimax": "MINIMAX_API_KEY", "inworld": "INWORLD_API_KEY",
    "gemini": "GEMINI_API_KEY", "fish": "FISH_API_KEY",
    "mistral": "MISTRAL_API_KEY", "smallestai": "SMALLEST_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
}


def get_provider(name, timeout=120):
    from .binary import (
        CartesiaProvider,
        DeepgramProvider,
        ElevenLabsProvider,
        InworldProvider,
        SmallestProvider,
    )
    from .structured import (
        FishProvider,
        GeminiProvider,
        MiniMaxProvider,
        MistralProvider,
    )

    providers = {
        "elevenlabs": ElevenLabsProvider, "cartesia": CartesiaProvider,
        "minimax": MiniMaxProvider, "inworld": InworldProvider,
        "gemini": GeminiProvider, "fish": FishProvider, "mistral": MistralProvider,
        "smallestai": SmallestProvider, "deepgram": DeepgramProvider,
    }
    if name not in providers:
        raise ValueError(f"Unknown provider: {name}")
    env = KEY_ENV[name]
    key = os.environ.get(env)
    if not key:
        raise ValueError(f"Set {env} to evaluate {name}")
    return providers[name](key, timeout=timeout)


__all__ = ["APIError", "AudioResponse", "Provider", "SynthesisRequest", "get_provider"]
