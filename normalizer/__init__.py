# The normalizer classes live in the `asr-normalizer` package (shared with the Open ASR Leaderboard);
# re-exported here so `from normalizer import ...` keeps working.
from asr_normalizer import (
    ArmenianTextNormalizer,
    BasicMultilingualTextNormalizer,
    EnglishTextNormalizer,
    MultilingualNormalizer,
)

__all__ = ["EnglishTextNormalizer", "BasicMultilingualTextNormalizer", "ArmenianTextNormalizer", "MultilingualNormalizer"]
