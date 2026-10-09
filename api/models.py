"""Explicit model/voice settings; provider defaults must not silently change an evaluation."""

from dataclasses import dataclass, field

BENCHMARK_LANGUAGES = ("en", "zh", "fr", "es", "ja", "ko", "it", "de", "ru")
MINIMAX_VOICES = {
    "en": "English_expressive_narrator", "zh": "Chinese (Mandarin)_News_Anchor",
    "ja": "Japanese_CalmLady", "ko": "Korean_CalmLady", "es": "Spanish_Narrator",
    "fr": "French_MaleNarrator", "de": "German_FriendlyMan", "ru": "Russian_ReliableMan",
    "it": "Italian_Narrator",
}


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    voice: str | None = None
    languages: tuple[str, ...] = BENCHMARK_LANGUAGES
    reference_mode: str = "none"
    voices: dict[str, str] = field(default_factory=dict)

    def voice_for_language(self, language):
        return self.voices.get(language, self.voice)


MODELS = {
    "elevenlabs/eleven_v4": ModelConfig("elevenlabs", "eleven_v4", "JBFqnCBsd6RMkjVDRZzb"),
    "elevenlabs/eleven_v4_turbo": ModelConfig("elevenlabs", "eleven_v4_turbo", "JBFqnCBsd6RMkjVDRZzb"),
    "cartesia/sonic-3.6-2026-08-27": ModelConfig(
        "cartesia", "sonic-3.6-2026-08-27", "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
    ),
    "minimax/speech-2.8-hd": ModelConfig("minimax", "speech-2.8-hd", voices=MINIMAX_VOICES),
    "minimax/speech-2.8-turbo": ModelConfig("minimax", "speech-2.8-turbo", voices=MINIMAX_VOICES),
    "inworld/inworld-tts-2": ModelConfig("inworld", "inworld-tts-2", "Ashley"),
    "inworld/inworld-tts-2-flash": ModelConfig("inworld", "inworld-tts-2-flash", "Ashley"),
    "gemini/gemini-3.8-flash-tts": ModelConfig("gemini", "gemini-3.8-flash-tts", "Kore"),
    "gemini/gemini-3.8-flash-lite-tts": ModelConfig("gemini", "gemini-3.8-flash-lite-tts", "Kore"),
    "fish/s2.1-pro": ModelConfig("fish", "s2.1-pro", reference_mode="inline"),
    "fish/s2-pro": ModelConfig("fish", "s2-pro", reference_mode="inline"),
    "mistral/voxtral-mini-tts-2603": ModelConfig(
        "mistral", "voxtral-mini-tts-2603", languages=("en", "fr", "es", "it", "de"),
        reference_mode="inline",
    ),
    "smallestai/lightning_v3.1_pro": ModelConfig(
        "smallestai", "lightning_v3.1_pro", "blake", voices={
            "en": "blake", "zh": "hazel", "fr": "manon", "es": "martina", "de": "hanna",
            "it": "silvia", "ja": "aria", "ko": "june", "ru": "anastasia",
        },
    ),
    "smallestai/lightning_v3.1": ModelConfig(
        "smallestai", "lightning_v3.1", "olivia", languages=("en", "es"),
        voices={"en": "olivia", "es": "daniella"},
    ),
    "deepgram/aura-2-thalia-en": ModelConfig("deepgram", "aura-2-thalia-en", languages=("en",)),
}


def get_model(model_id):
    try:
        return MODELS[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_id!r}; use --list_models to see configured IDs") from exc
