import re

import num2words
from normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer


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

    @staticmethod
    def _normalize_vi_numbers(text):
        # Vietnamese conventions, applied before punctuation is stripped: "." groups thousands
        # ("10.000"), "," is the decimal separator ("3,5" -> "3 phẩy 5", leading zeros read as
        # "không"), and "%" is read "phần trăm". Otherwise "3,5%" would become "ba năm".
        text = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)
        def _decimal(m):
            frac = m.group(2)
            zeros = len(frac) - len(frac.lstrip("0"))
            return " ".join([m.group(1), "phẩy"] + ["không"] * zeros + ([frac[zeros:]] if frac[zeros:] else []))
        text = re.sub(r"(\d+),(\d+)", _decimal, text)
        return text.replace("%", " phần trăm")

    def __call__(self, s, lang=None):
        if lang == "vi":
            s = self._normalize_vi_numbers(s)
        s = super().__call__(s)
        if lang is not None:
            s = self._normalize_numbers(s, lang)
        return s


normalizer = EnglishTextNormalizer()

ml_normalizer = MultilingualNormalizer(remove_diacritics=False)

