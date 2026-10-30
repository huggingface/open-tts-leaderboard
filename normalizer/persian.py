"""Persian orthographic normalization, applied identically to references and ASR text."""

import re
import unicodedata

from num2words import num2words

from .normalizer import remove_non_speech_annotations


_LETTERS = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ه"})
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
# Short vowels, tanwin, shadda, sukun, dagger alif and Quranic annotation marks.
# Keep hamza/madda: أ/آ/ؤ/ئ must not collapse into ا/و/ی.
_VOCALIZATION = re.compile("[\u0610-\u061a\u064b-\u0652\u0656-\u065f\u0670\u06d6-\u06ed]")


class PersianTextNormalizer:
    """Use WER with Persian spelling conventions and the shared compound-boundary policy.

    ZWNJ becomes a space; joined vs split compounds are handled by score_results, as for
    other multilingual WER languages. Do not stem or equate colloquial/formal words.
    Numeric digit scripts are folded and integers expanded with num2words' Persian rules.
    Decimal/group separators are handled before punctuation, so ۱۲٫۵ is not read as ۱۲۵.
    """

    def __call__(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text).lower().translate(_LETTERS).translate(_DIGITS)
        text = remove_non_speech_annotations(text)
        text = _VOCALIZATION.sub("", text).replace("ـ", "")
        # Optional written ezafe on heh (خانهٔ / خانۀ); preserve hamza elsewhere.
        text = text.replace("ه\u0654", "ه")
        text = text.replace("\u200c", " ")
        text = "".join(c for c in text if unicodedata.category(c) != "Cf")
        # Only remove grouping separators in groups of three, never every comma blindly.
        text = re.sub(r"(?<=\d)[,٬](?=\d{3}(?:\D|$))", "", text)
        text = text.replace("٫", ".").replace("−", "-")

        def expand(match):
            number = match.group()
            # Pass integers: num2words' Persian implementation cannot parse strings and
            # its fractional helper conflates e.g. .05 and .5. Spell decimal notation
            # with ممیز, retaining leading fractional zeros, without float rounding.
            try:
                integer, dot, fraction = number.partition(".")
                value = int(integer)
                if abs(value) >= 10**15:
                    return number
                words = num2words(value, lang="fa")
                if number.startswith("-") and value == 0:
                    words = "منفی " + words
                if dot:
                    zeros = len(fraction) - len(fraction.lstrip("0"))
                    tail = (["صفر"] * zeros)
                    if fraction.lstrip("0"):
                        if len(fraction.lstrip("0")) > 15:
                            return number
                        tail.append(num2words(int(fraction), lang="fa"))
                    words += " ممیز " + " ".join(tail)
                return words
            except (ValueError, OverflowError):
                return number

        text = re.sub(r"(?<!\w)-?\d+(?:\.\d+)?", expand, text)
        # The number expander can introduce ZWNJ; normalize its output as well.
        text = text.translate(_LETTERS).replace("\u200c", " ")
        text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
        return " ".join(text.split())
