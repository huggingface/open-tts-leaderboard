import re

import num2words
from normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer
from normalizer.persian import PersianTextNormalizer
from normalizer.hindi import normalize_hi_numbers_after, normalize_hi_numbers_before
from normalizer.spoken_forms import expand_written_forms, fold_spoken_forms, number_words

# Decimal numbers, in the languages whose transcripts read them in words ("drei Komma sieben", "cinco punto ocho",
# "drie en een half procent"): the word for the separator. The separator is "," (and "." in Spanish, whose
# references write "2.5") followed by one or two digits, so "10.000" and "1,000" stay thousands; "." in the other
# languages is left alone, as the ASR writes times with it ("20.30").
DECIMAL_WORDS = {"de": "komma", "fr": "virgule", "es": "coma", "it": "virgola", "pt": "vírgula", "nl": "komma",
                 "pl": "przecinek", "cs": "celá"}
# "<n> and a half" for n,5, as the ASR wrote it.
HALF_READINGS = {"nl": ("en een half", "en half", "en halve"), "pl": ("i pół",), "cs": ("a půl",), "pt": ("e meio",)}


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

    @staticmethod
    def _normalize_decimals(text, lang):
        # "3,5" -> "3 komma 5", before punctuation is stripped (it was "drei fünf"). German reads the decimals one
        # by one; the other languages read them as a number, after any leading zeros ("0,01" -> "0 vírgula 0 1").
        separators = ",." if lang == "es" else ","
        def _decimal(m):
            frac = m.group(2)
            zeros = len(frac) if lang == "de" else len(frac) - len(frac.lstrip("0"))
            return " ".join([m.group(1), DECIMAL_WORDS[lang]] + list(frac[:zeros]) + ([frac[zeros:]] if frac[zeros:] else []))
        return re.sub(rf"(?<![\d.,:])(\d+)[{separators}](\d{{1,2}})(?!\d|[.,]\d)", _decimal, text)

    @staticmethod
    def _fold_decimal_readings(text, lang):
        # The ASR's other readings of a decimal, after the numbers are words: Spanish "punto" between two numbers is
        # "coma"; "<n> and a half" is "<n> <separator> <five>".
        numbers = number_words(lang)
        if lang == "es":
            text = re.sub(r"(?<!\S)(\S+) punto (?=(\S+))",
                          lambda m: f"{m.group(1)} {'coma' if m.group(1) in numbers and m.group(2) in numbers else 'punto'} ",
                          text)
        for half in HALF_READINGS.get(lang, ()):
            five = num2words.num2words(5, lang=lang)
            text = re.sub(rf"(?<!\S)(\S+) {half}(?!\S)",
                          lambda m: f"{m.group(1)} {DECIMAL_WORDS[lang]} {five}" if m.group(1) in numbers else m.group(0),
                          text)
        return text

    def __call__(self, s, lang=None):
        if lang == "fa":
            return persian_normalizer(s)
        if lang == "vi":
            s = self._normalize_vi_numbers(s)
        elif lang == "hi":
            s = normalize_hi_numbers_before(s)
        elif lang is not None:
            if lang in DECIMAL_WORDS:
                s = self._normalize_decimals(s, lang)
            # Units, symbols and abbreviations to their spoken words, before symbols and dots are stripped.
            s = expand_written_forms(s, lang)
        s = super().__call__(s)
        if lang == "hi":
            s = normalize_hi_numbers_after(s)
        elif lang is not None:
            s = self._normalize_numbers(s, lang)
            s = fold_spoken_forms(s, lang)
            if lang in DECIMAL_WORDS:
                s = self._fold_decimal_readings(s, lang)
        return s


normalizer = EnglishTextNormalizer()

persian_normalizer = PersianTextNormalizer()

ml_normalizer = MultilingualNormalizer(remove_diacritics=False)

