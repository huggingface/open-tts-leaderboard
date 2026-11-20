"""Arabic orthographic folding, applied identically to references and ASR text.

The same pronunciation-neutral folding as ZTTS2-Eval (utils/text_norm.py, ORTHO_NORM_LANGS "ar"): optional
vowel marks and spelling variants that the audio cannot distinguish are not TTS errors. Without it, a
reference with harakat (يُطلب) against an ASR output without them (يطلب) counts a word error.
"""

# Harakat / tanwin (U+064B-064F), shadda / sukun (U+0650-0652), maddah / hamza above / below (U+0653-0655),
# superscript alef (U+0670), tatweel (U+0640).
_STRIP = {c: None for c in range(0x064B, 0x0656)}
_STRIP.update({0x0670: None, 0x0640: None})
# Hamza-carrier and alef variants, alef maqsura, and taa marbuta, each folded onto one letter; lone hamza dropped.
_FOLD = {ord(k): v for k, v in {
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ٲ": "ا", "ٳ": "ا",
    "ى": "ي", "ئ": "ي", "ؤ": "و", "ء": None, "ة": "ه",
}.items()}
_TABLE = {**_STRIP, **_FOLD}


def fold_arabic(text: str) -> str:
    """Strip Arabic vowel marks and tatweel and fold letter variants (see the module docstring)."""
    return " ".join(text.translate(_TABLE).split())
