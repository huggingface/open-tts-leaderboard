"""Spoken forms of the units, symbols and abbreviations that occur in the benchmark references.

For these items the ASR writes the written form in some transcripts and the spoken words in others ("12%" or "12
Prozent", "km/h" or "kilómetros por hora", "z.B." or "zum Beispiel"). Which one it writes is its formatting choice,
not something the TTS controls, so before alignment both the reference and the transcript are mapped to one spoken
form per item:

- `expand_written_forms` runs on the raw text, before punctuation and symbols are stripped: a unit or symbol after a
  number, and an abbreviation, become their spoken words.
- `fold_spoken_forms` runs on the normalized text: the other spellings of those spoken words that the ASR writes
  ("Kilo" for "Kilogramm", "kilomètres heure" for "kilomètres par heure", an inflected form) become the one spoken
  form, and a "plus" read as the sign of a number is dropped, as the "+" sign itself already is.

Every entry is an item that occurs in the leaderboard's references, and every variant a spelling the ASR wrote in the
published results. A reading the TTS got wrong (letters spelled out, a unit left out) still differs from the reference.
"""

import re
from functools import lru_cache

# A unit or symbol after a number ("12 %", "480 km/h", "+30°C"), as written in the references. Case-sensitive.
# "°C" also matches "° C". The number may be written in words ("forty mph"), except before the units in DIGIT_ONLY,
# which are also words or too short to tell apart.
#
# Each unit is read the way the references use it. Short abbreviations can mean something else in other text: in
# technical writing "50m", "2h" and "30s" are often 50 minutes, 2 hours and 30 seconds. English has no one-letter unit
# here, so those stay as written; German and Spanish "m" after a number is read as metres, as in the references
# ("30 m", "100 m a 250 m"), and would be wrong where it means minutes. A rule for such units needs the context.
UNITS = {
    "en": {"km": "kilometers", "km/h": "kilometers per hour", "kph": "kilometers per hour",
           "mph": "miles per hour", "°C": "degrees celsius"},
    "de": {"%": "prozent", "kg": "kilogramm", "m": "meter", "km/h": "kilometer pro stunde",
           "m/s": "meter pro sekunde", "MPH": "meilen pro stunde", "mph": "meilen pro stunde",
           "°C": "grad celsius"},
    "es": {"%": "por ciento", "km": "kilómetros", "m": "metros", "mi": "millas", "km/h": "kilómetros por hora",
           "mph": "millas por hora", "°C": "grados celsius", "°F": "grados fahrenheit"},
    "fr": {"%": "pour cent", "mm": "millimètres", "km/h": "kilomètres par heure", "mph": "miles par heure",
           "°C": "degrés celsius", "°F": "degrés fahrenheit"},
    "it": {"%": "per cento", "mm": "millimetri", "km/h": "chilometri orari", "°C": "gradi celsius"},
    "pt": {"%": "por cento"},
    "nl": {"%": "procent"},
    "pl": {"%": "procent"},
    "cs": {"%": "procent"},
    "ro": {"%": "la sută"},
    "el": {"%": "τοις εκατό"},
    "ru": {"%": "процентов", "км/ч": "километров в час", "миль/ч": "миль в час", "кв. км": "квадратных километров"},
}
DIGIT_ONLY = {"m", "mi"}

# Abbreviations, as regular expressions over the raw text (the forms in the references, and the other ways the ASR
# writes the same abbreviation: a space or a dot left out, another case), with their spoken form.
ABBREVIATIONS = {
    "de": [(r"z\.\s*B\.", "zum beispiel"), (r"bzw\.", "beziehungsweise"), (r"etc\.", "et cetera"),
           (r"Dr\.", "doktor"), (r"St\.?(?=\s+[A-Z])", "sankt")],
    "es": [(r"EE\.\s*UU\.|EEUU", "estados unidos"), (r"etc\.", "etcétera"), (r"a\.\s*C\.", "antes de cristo"),
           (r"n\.º|No\.(?=\s*\d)", "número"), (r"St\.", "saint")],
    "fr": [(r"av\.\s*J\.?-?\s*C\.?", "avant jésus christ"), (r"J\.?-\s*C\.?|J\.C\.", "jésus christ"),
           (r"Dr\.?(?=\s+[A-ZÀ-Ý])", "docteur"), (r"St\.?(?=\s+[A-ZÀ-Ý])", "saint"), (r"M\.(?=\s+[A-ZÀ-Ý])", "monsieur")],
    "it": [(r"(?i:ad es)\.?", "ad esempio"), (r"a\.\s*C\.", "avanti cristo"), (r"St\.?(?=\s+[A-Z])", "saint")],
    "ru": [(r"т\.\s*е\.", "то есть"), (r"см\.", "смотрите"), (r"гг\.", "годов")],
}

# Other spellings of the spoken forms above, as the ASR wrote them (after normalization), folded on both sides.
VARIANTS = {
    # "degrees C" is a common English reading of "°C".
    "en": {"kilometers an hour": "kilometers per hour", "miles an hour": "miles per hour", "degrees c": "degrees celsius"},
    "de": {"kilo": "kilogramm", "metern": "meter", "saint": "sankt"},
    "es": {"kilómetros hora": "kilómetros por hora", "kilómetros h": "kilómetros por hora"},
    "fr": {"kilomètres heure": "kilomètres par heure"},
    "it": {"chilometri per ora": "chilometri orari", "chilometri all ora": "chilometri orari",
           "chilometri ora": "chilometri orari", "chilometro ora": "chilometri orari"},
    "pl": {"procenta": "procent"},
    "cs": {"procenta": "procent"},
    "ru": {"процента": "процентов", "процент": "процентов", "километра": "километров",
           "километров ч": "километров в час", "км ч": "километров в час", "миль ч": "миль в час"},
}

# The word for a "+" sign, dropped before a number (the "+" itself is stripped with the other symbols).
PLUS_WORDS = {"de": {"plus"}, "es": {"más", "mas"}, "ru": {"плюс"}}


@lru_cache(maxsize=None)
def _number_words(lang):
    """The words num2words uses for the numbers 0-1000 (and the powers of a thousand) in `lang`."""
    import num2words
    words = set()
    for n in list(range(1001)) + [10 ** k for k in range(6, 13, 3)]:
        try:
            words.update(re.split(r"[\s-]+", num2words.num2words(n, lang=lang)))
        except Exception:
            pass
    return frozenset(w for w in words if w)


@lru_cache(maxsize=None)
def _compiled(lang):
    unit_re = None
    units = UNITS.get(lang, {})
    if units:
        # Longest first, so "km/h" wins over "km" and "m/s" over "m".
        alternatives = "|".join(re.escape(u).replace("°", r"°\s?") for u in sorted(units, key=len, reverse=True))
        # The number before the unit: its last digit, or a whole word (checked against the number words).
        unit_re = re.compile(rf"(\d|(?<![\w-])[^\W\d_]+(?:-[^\W\d_]+)*\s)\s?({alternatives})(?![\w/])")
    abbreviations = [
        (re.compile(rf"(?<![\w.])(?:{pattern})(?!\w)"), spoken) for pattern, spoken in ABBREVIATIONS.get(lang, [])
    ]
    variants = VARIANTS.get(lang, {})
    variant_re = None
    if variants:
        alternatives = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
        variant_re = re.compile(rf"(?<!\S)({alternatives})(?!\S)")
    return unit_re, abbreviations, variant_re


def expand_written_forms(text: str, lang: str) -> str:
    """Raw text: units and symbols after a number, and abbreviations, to their spoken words."""
    unit_re, abbreviations, _ = _compiled(lang)
    for pattern, spoken in abbreviations:
        text = pattern.sub(f" {spoken} ", text)
    if unit_re is not None:
        units = {re.sub(r"\s", "", u): spoken for u, spoken in UNITS[lang].items()}

        def unit(m):
            number, written = m.group(1), re.sub(r"\s", "", m.group(2))
            last = number.strip().lower().split("-")[-1]   # "sixty-four"
            if not number[0].isdigit() and (written in DIGIT_ONLY or last not in _number_words(lang)):
                return m.group(0)
            return f"{number} {units[written]} "

        text = unit_re.sub(unit, text)
    return text


def _is_number(token, lang):
    if lang == "en":
        return bool(re.match(r"^[+-]?\d", token))
    return token.split("-")[0] in _number_words(lang)


def fold_spoken_forms(text: str, lang: str) -> str:
    """Normalized text: the ASR's other spellings of a spoken form to that form; a spoken plus sign dropped."""
    _, _, variant_re = _compiled(lang)
    if variant_re is not None:
        variants = VARIANTS[lang]
        text = variant_re.sub(lambda m: variants[m.group(1)], text)
    if lang == "en":
        # The number normalizer writes "plus thirty" as "+30"; "+" before a number is a sign unless a number
        # precedes it ("2 +2" is a sum).
        words = text.split()
        for i, w in enumerate(words):
            if re.match(r"^\+\d", w) and not (i and _is_number(words[i - 1], lang)):
                words[i] = w[1:]
        text = " ".join(words)
    elif lang in PLUS_WORDS:
        words = text.split()
        text = " ".join(
            w for i, w in enumerate(words)
            if not (w in PLUS_WORDS[lang] and i + 1 < len(words) and _is_number(words[i + 1], lang)
                    and not (i and _is_number(words[i - 1], lang)))
        )
    return text
