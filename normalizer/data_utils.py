import re

import num2words
from normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer
from normalizer.persian import PersianTextNormalizer


class MultilingualNormalizer(BasicMultilingualTextNormalizer):
    """BasicMultilingualTextNormalizer with optional number normalization.

    Pass lang= to also convert digits to words via num2words.
    """

    def _normalize_numbers(self, text, lang):
        # Join space-separated thousand groups (e.g. "10 000" -> "10000")
        text = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", text)
        # Convert remaining digit sequences to words
        def _replace(m):
            try:
                return num2words.num2words(int(m.group()), lang=lang)
            except Exception:
                return m.group()
        return re.sub(r"\d+", _replace, text)

    def __call__(self, s, lang=None):
        if lang == "fa":
            return persian_normalizer(s)
        s = super().__call__(s)
        if lang is not None:
            s = self._normalize_numbers(s, lang)
        return s


normalizer = EnglishTextNormalizer()

persian_normalizer = PersianTextNormalizer()

ml_normalizer = MultilingualNormalizer(remove_diacritics=False)

