"""Times, ordinals and Roman numerals in the benchmark references, read the way people say them.

The references write these in figures ("10:00 Uhr", "7 h 19", "1:15 a.m.", "il 1° reggimento", "XVe siècle",
"siglo IX", "World War II"). Before this, "10:00 Uhr" was normalized to "zehn null uhr", so the usual reading "zehn
Uhr" counted an error; "7h19" became one word ("septhdix") and "1er" became "uner"; and an ordinal or a Roman numeral
was kept as written, so "primo", "premier", "quinzième", "noveno" or "World War One" counted as errors. Both the
reference and the transcript are mapped to one form:

- `expand_times_and_ordinals` runs on the raw text, before punctuation and symbols are stripped: a time loses ":00"
  ("zehn Uhr"), a French time is read with "heures", an ordinal and a Roman numeral in these contexts become words.
- `fold_time_readings` runs on the normalized text: the ASR's other readings of the same thing ("alle venti e
  trenta", "a las once y veinte", "siglo noveno", "world war ii") become that form.

Every rule is for an item in the leaderboard's references, and every folded reading one the ASR wrote in the results.
"""

import re
from functools import lru_cache

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}


def roman_to_int(numeral):
    values = [_ROMAN[c] for c in numeral]
    return sum(-v if i + 1 < len(values) and v < values[i + 1] else v for i, v in enumerate(values))


@lru_cache(maxsize=None)
def _ordinal(n, lang):
    import num2words
    return num2words.num2words(n, lang=lang, to="ordinal")


def _french_ordinal(m):
    value = roman_to_int(m.group(1)) if m.group(1)[0] in _ROMAN else int(m.group(1))
    if value == 1:
        return "première" if m.group(2) in ("re", "ère") else "premier"
    return _ordinal(value, "fr")


def expand_times_and_ordinals(text: str, lang: str) -> str:
    """Raw text: times, ordinals and Roman numerals to the words they are read with."""
    if lang == "en":
        # "1:15 a.m." is read "one fifteen", which the number normalizer writes "115"; "5:00 a.m." is "five a.m.".
        return re.sub(r"\b(\d{1,2}):(\d{2})(?=\s?(?i:[ap]\.?\s?m)\b)",
                      lambda m: m.group(1) if m.group(2) == "00" else m.group(1) + m.group(2), text)
    if lang in ("de", "es", "it"):
        # "10:00 Uhr" is read "zehn Uhr", "alle 10:00" "alle dieci", "15:00 UTC" "quince".
        text = re.sub(r"\b(\d{1,2}):00\b(?![.,]\d)", r"\1", text)
    if lang == "fr":
        # "7 h 19" and "7h19" are read "sept heures dix-neuf", "11 h 00" "onze heures"; the ASR also writes "7h19h
        # locale" for "... heure locale", and "7:19" before "heure", "GMT" or "UTC" (elsewhere "4:41" is a race time).
        def _time(m):
            minutes = "" if m.group(2) == "00" else f" {m.group(2)}"
            return f"{m.group(1)} heures{minutes}" + (" heure" if m.group(3) else "")
        text = re.sub(r"\b(\d{1,2})\s?h\s?(\d{2})(?!\d)(h\b)?", _time, text)
        text = re.sub(r"\b(\d{1,2}):(\d{2})(?!\d)()(?=\s+(?:heures?|GMT|UTC)\b)", _time, text)
        text = re.sub(r"\b(\d{1,2})\s?h\b(?![\w/])", r"\1 heures", text)
        # Ordinals: "1er" premier, "XVe" / "XVème" / "15e" quinzième (only I, V and X, so "Le" is not fifty).
        text = re.sub(r"\b([IVX]+|\d+)(er|re|ère|e|è|ème|eme)\b", _french_ordinal, text)
    if lang == "it":
        # "il 1° e il 3° reggimento" is "primo", "terzo"; "35°W", "35° W", "35° di latitudine", "30°C" are degrees.
        text = re.sub(r"\b(\d{1,2})°(?=\s+[a-zà-ù])(?!\s+(?:gradi|grado|di|nord|sud|est|ovest)\b)",
                      lambda m: _ordinal(int(m.group(1)), "it"), text)
        # Centuries are ordinals: "XV secolo" quindicesimo.
        text = re.sub(r"\b([IVX]+)(?=\s+secolo\b)", lambda m: _ordinal(roman_to_int(m.group(1)), "it"), text)
    if lang == "es":
        # "siglo IX" is read "siglo nueve" (or "noveno", folded below).
        text = re.sub(r"\b([Ss]iglo)\s+([IVXL]+)\b", lambda m: f"{m.group(1)} {roman_to_int(m.group(2))}", text)
    return text


@lru_cache(maxsize=None)
def _spanish_century_ordinals():
    import num2words
    return {num2words.num2words(n, lang="es", to="ordinal"): num2words.num2words(n, lang="es") for n in range(1, 11)}


def fold_time_readings(text: str, lang: str) -> str:
    """Normalized text: the ASR's other readings of a time, ordinal or Roman numeral, to the form above."""
    if lang == "en":
        # "World War I" and "World War II" are read "One" and "Two" (normalized "one" and "2").
        text = re.sub(r"\bworld war i\b", "world war one", text)
        return re.sub(r"\bworld war ii\b", "world war 2", text)
    from normalizer.spoken_forms import number_words
    numbers = number_words(lang)
    if lang in ("es", "it"):
        # "a las once y veinte", "alle venti e trenta": the hour and the minutes joined by "y" / "e".
        lead, conj = ("las", "y") if lang == "es" else ("alle", "e")
        text = re.sub(rf"(?<!\S){lead} (\S+) {conj} (?=(\S+))",
                      lambda m: f"{lead} {m.group(1)} " if m.group(1) in numbers and m.group(2) in numbers else m.group(0),
                      text)
    if lang == "it":
        # "alle dieci zero zero" reads "10:00" digit by digit.
        text = re.sub(r"(?<!\S)(\S+) zero zero(?!\S)", lambda m: m.group(1) if m.group(1) in numbers else m.group(0), text)
    if lang == "es":
        ordinals = _spanish_century_ordinals()
        text = re.sub(r"(?<!\S)siglo (\S+)", lambda m: f"siglo {ordinals.get(m.group(1), m.group(1))}", text)
    return text
