"""ZTTS2-Eval Mini scoring, as ZTTS2-Eval (https://github.com/Zyphra/ZTTS2-Eval) scores its coreset panels.

Ports ZTTS2-Eval's text preparation (`utils/text_norm.py` `prepare` / `process_one`) and its panel
estimator (`scripts/coreset/score_panel.py` `weighted_macro`): per language, the panel-weighted mean of
the per-utterance WER clipped at 1 (CER for zh ja ko th). The transcripts must come from
transformers/transcribe.py's `ztts2` ASR profile (Whisper-large-v3 / Qwen3-ASR routed per language), which
stamps every row it transcribes with `asr_profile`.
"""

import functools
import json
import os
import re
import unicodedata

import num2words as _n2w
import zhconv
from kaldialign import edit_distance

# Split names of the two panels, in ZTTS2-Eval's order (utils/benchmark.py CLEAN_LANGS / ITW_LANGS).
PANEL_LANGS = {
    "clean": ("en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru", "pt", "ar", "hi", "tr", "pl", "id", "th",
              "fil", "nl", "vi", "cs", "sv", "da", "fi", "el", "hu", "ro", "uk", "nb", "he", "bg", "hr", "sk",
              "ca", "ms", "fa"),
    "in_the_wild": ("en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru", "pt", "ar", "hi", "id", "tr", "tl",
                    "pl", "th", "nl", "uk", "vi"),
}
# {config: {split: {item id: panel weight}}}, written by scripts/prepare_ztts2_eval_mini.py.
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ztts2_eval_mini_weights.json")
ASR_PROFILE = "ztts2"   # `asr_profile` that transcribe.py writes on rows it transcribed with ZTTS2 routing
_PANEL_RE = re.compile(r"ztts2_eval_mini_(clean|in_the_wild)_([a-z]+)")
CHAR_LANGS = {"zh", "ja", "ko", "th", "fa"}
# ZTTS2-Eval's num2words 0.5.14 converters (nb via `no`); pinned because newer num2words adds e.g. hi,
# which ZTTS2-Eval leaves in digits.
NUM2WORDS_LANGS = frozenset({"ar", "ca", "cs", "da", "de", "en", "es", "fa", "fi", "fr", "he", "hu", "id", "it",
                             "ja", "ko", "nb", "nl", "pl", "pt", "ro", "ru", "sk", "sv", "th", "tr", "uk", "vi"})

# ── Verbatim from ZTTS2-Eval utils/text_norm.py ──────────────────────────────
_CJK_TOKEN_RE = re.compile(r"[A-Za-z]+|\S")
# apostrophe glyphs: ' U+2018 U+2019 U+02BC U+02B9 U+2032 backtick
_APOS_RE = re.compile("['‘’ʼʹ′`]")

# zero-width joiners / bidi controls / soft hyphen / BOM (never pronounced)
_ZW = {c: None for c in (0x200C, 0x200D, 0x200E, 0x200F, 0x00AD, 0x2060, 0xFEFF)}
# Arabic combining marks: harakat/tanwin U+064B-064F, shadda/sukun U+0650-0652,
# maddah/hamza-above/below U+0653-0655, superscript alef U+0670, tatweel U+0640
_AR_STRIP = {c: None for c in range(0x064B, 0x0656)}
_AR_STRIP.update({0x0670: None, 0x0640: None})


def _t(mapping):
    return {ord(k): v for k, v in mapping.items()}


_AR_FOLD = _t({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ٲ": "ا", "ٳ": "ا",
               "ى": "ي", "ئ": "ي", "ؤ": "و", "ء": None, "ة": "ه"})
_FA_FOLD = _t({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
               "ي": "ی", "ك": "ک", "ى": "ی", "ئ": "ی", "ؤ": "و", "ء": None, "ة": "ه"})
_HE_STRIP = {c: None for c in range(0x0591, 0x05C8)
             if unicodedata.category(chr(c)) == "Mn"}
_HE_FOLD = _t({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})

# Pronunciation-neutral orthographic folding of the abjad languages (ZTTS2-Eval's default ORTHO_NORM_LANGS).
_ORTHO = {
    "ar": {**_AR_STRIP, **_ZW, **_AR_FOLD},
    "fa": {**_AR_STRIP, **_ZW, **_FA_FOLD},
    "he": {**_HE_STRIP, **_ZW, **_HE_FOLD},
}

_N2W_CODE = {"nb": "no"}
_NUM_RE = re.compile(r"(?<![\w])(\d+(?:[.,]\d+)*)(st|nd|rd|th)?(?![\w])")
_SEP_RE = re.compile(r"[.,]")


def _say(value, code, **kw):
    return " " + _n2w.num2words(value, lang=code, **kw) + " "


def expand_numbers(text, base):
    """Replace digit tokens by words; identity for languages without num2words support."""
    if base not in NUM2WORDS_LANGS:
        return text
    code = _N2W_CODE.get(base, base)

    def repl(m):
        tok, suffix = m.group(1), m.group(2)
        if suffix and base != "en":
            return m.group(0)
        try:
            groups = _SEP_RE.split(tok)
            if len(groups) > 1 and all(len(g) == 3 for g in groups[1:]) \
                    and not groups[0].startswith("0"):
                tok = "".join(groups)               # 1,000 / 1.000.000 -> integer
            elif len(groups) == 2:
                return _say(f"{groups[0]}.{groups[1]}", code)   # decimal
            elif len(groups) > 2:
                return " ".join(_say(int(g), code) for g in groups)
            n = int(tok)
            if base == "en" and suffix:
                return _say(n, code, to="ordinal")
            if base == "en" and 1100 <= n <= 2099 and len(tok) == 4:
                return _say(n, code, to="year")
            return _say(n, code)
        except Exception:  # num2words: NotImplementedError/OverflowError/... -> keep digits
            return m.group(0)

    return _NUM_RE.sub(repl, text)


# ── Scoring ──────────────────────────────────────────────────────────────────
def parse_panel(dataset_id):
    """(config, split) of a ztts2_eval_mini manifest's dataset id, else None."""
    m = _PANEL_RE.search(dataset_id)
    return (m.group(1), m.group(2)) if m else None


def prepare(text, lang):
    """Tokens of `text` after ZTTS2-Eval's prep (utils/text_norm.py `prepare`)."""
    if lang == "zh":
        text = zhconv.convert(text, "zh-cn")
    table = _ORTHO.get(lang)
    if table:
        text = text.translate(table)
    text = expand_numbers(text, lang)
    text = _APOS_RE.sub("", text)
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    if lang in CHAR_LANGS:
        return [t.lower() for t in _CJK_TOKEN_RE.findall(text)]
    return text.lower().split()


def utterance_errors(ref, hyp, lang):
    """(wer, ins, del, sub) of one utterance, as ZTTS2-Eval's `process_one`."""
    ref_toks, hyp_toks = prepare(ref, lang), prepare(hyp, lang)
    if not ref_toks:
        return float(bool(hyp_toks)), len(hyp_toks), 0, 0
    r = edit_distance(ref_toks, hyp_toks)
    return r["total"] / len(ref_toks), r["ins"], r["del"], r["sub"]


def item_id(audio_filepath):
    """Dataset id of a generated wav (`.../output_<id>.wav`, run_eval_utils.wav_rel_path)."""
    name = os.path.basename(audio_filepath)
    if not (name.startswith("output_") and name.endswith(".wav")):
        raise ValueError(f"cannot read an item id from {audio_filepath!r}")
    return name[len("output_"):-len(".wav")]


@functools.cache
def panel_weights():
    with open(WEIGHTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def panel_estimate(config, split, values):
    """score_panel.py `weighted_macro` for one language: the weighted mean of `values`
    ({item id: per-item metric}) with the panel weights renormalized over the items present."""
    weights = panel_weights()[config][split]
    if not values:
        raise ValueError(f"no items to score for {config}/{split}")
    unknown = sorted(set(values) - set(weights))
    if unknown:
        raise ValueError(f"{len(unknown)} item(s) not in the {config}/{split} panel, e.g. {unknown[0]!r}")
    total = sum(weights[i] for i in values)
    return sum(weights[i] * v for i, v in values.items()) / total


def score_manifest(manifest, config, split, use_sim):
    """WER (CER for zh ja ko th) and SIM of one panel manifest, scored as ZTTS2-Eval scores it.
    Raises ValueError when the manifest cannot be scored that way."""
    stale = sum(1 for row in manifest if row.get("asr_profile") != ASR_PROFILE)
    if stale:
        raise ValueError(f"{stale}/{len(manifest)} row(s) were not transcribed with the '{ASR_PROFILE}' ASR "
                         f"profile; re-run the transcribe stage (STAGES=transcribe ASR_OVERWRITE=true)")
    clipped, sims, ins, dels, subs = {}, {}, 0, 0, 0
    for row in manifest:
        uid = item_id(row["audio_filepath"])
        if uid in clipped:
            raise ValueError(f"duplicate item {uid!r}")
        wer, i, d, s = utterance_errors(row["text"], row.get("pred_text") or "", split)
        clipped[uid] = min(wer, 1.0)
        ins, dels, subs = ins + i, dels + d, subs + s
        if use_sim and isinstance(row.get("sim"), (int, float)):
            sims[uid] = row["sim"]
    return {
        "wer": round(100 * panel_estimate(config, split, clipped), 2),
        "metric": "CER" if split in CHAR_LANGS else "WER",
        "sim": round(100 * panel_estimate(config, split, sims), 2) if sims else None,
        "ins": ins, "del": dels, "sub": subs,
        "n_items": len(clipped), "n_panel": len(panel_weights()[config][split]),
    }
