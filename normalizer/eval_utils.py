import os
import glob
import json
import re
import unicodedata
from difflib import SequenceMatcher

from collections import defaultdict
from kaldialign import batch_error_rate


# ── Chinese / CJK scoring (CER instead of WER) ────────────────────────────────
# CJK text has no word delimiters, so WER is meaningless; score per character (CER), as
# seed-tts-eval's run_wer.py does for zh (it still labels it "WER").

# CJK Unified Ideographs (+ Ext A/B+ and compat), Hiragana/Katakana, Hangul syllables.
_CJK_PATTERN = re.compile(
    "["
    "㐀-䶿"          # CJK Unified Ideographs Extension A
    "一-鿿"          # CJK Unified Ideographs
    "豈-﫿"          # CJK Compatibility Ideographs
    "\U00020000-\U0002ffff"  # CJK Unified Ideographs Extension B and beyond
    "぀-ヿ"          # Hiragana + Katakana
    "가-힯"          # Hangul syllables
    "]"
)

# Language codes that must be scored with CER. The single source of truth for the CJK list.
CJK_LANGUAGES = {"zh", "yue", "cmn", "ja", "jpn", "ko", "kor"}
# Other scripts without word delimiters: ml_normalizer, then scored per character (CER).
CHAR_LANGUAGES = {"th"}


def is_cjk_language(language: str) -> bool:
    """Whether `language` is CJK, matched on the part before the first `_`/`-` (e.g. "zh-cn")."""
    return re.split(r"[_-]", language)[0] in CJK_LANGUAGES


def default_multilingual(language: str) -> bool:
    """Compound-boundary pre-folding: off for English (own normalizer) and CJK (scored per char)."""
    return language != "en" and not is_cjk_language(language)


# Kana/Hangul mark a string as Japanese/Korean, gating the Chinese-only folds below.
_KANA_HANGUL_PATTERN = re.compile("[぀-ヿ가-힯]")
_JAPANESE_KOREAN_LANGUAGES = {"ja", "jpn", "ko", "kor"}


def _is_chinese(text: str, language: str = None) -> bool:
    """Whether to apply the Chinese-specific folds to `text`.

    Not Chinese if `language` is ja/ko (kanji-only Japanese is indistinguishable by script) or if
    the text contains kana/Hangul (`language` may be 'en' for auto-detected CJK manifests).
    """
    if language and re.split(r"[_-]", language)[0] in _JAPANESE_KOREAN_LANGUAGES:
        return False
    return not _KANA_HANGUL_PATTERN.search(text)


def _to_simplified(text: str) -> str:
    """Fold traditional Chinese to simplified (as seed-tts-eval's run_wer.py does via zhconv).

    ASR sometimes emits traditional characters while references are simplified; applied to both sides.
    """
    try:
        from zhconv import convert
    except ImportError as e:
        raise ImportError(
            "Scoring Chinese requires `zhconv` to fold traditional characters to simplified "
            "before alignment (matching seed-tts-eval's run_wer.py). Install it with "
            "`pip install zhconv`."
        ) from e
    return convert(text, "zh-cn")


# ── Chinese homophone folding (lenient metric only) ───────────────────────────
# Closed-class groups whose written distinction is a convention on one identical spoken form, so
# the ASR's spelling choice is not a TTS error (cf. `english_spelling_normalizer`). Open-class
# homophones (在/再, 做/作, ...) are excluded: context resolves them. Groups are chosen from
# phonology and references only, never fitted to model outputs.
CJK_HOMOPHONE_GROUPS = [
    # Structural particle `de`. 地 (dì) and 得 (dé) are polyphonic, so this over-reaches, but the fold
    # is applied to both sides and can only lower CER.
    ("的", "地得"),
    # Invariantly `tā`; gender/animacy is absent from the audio.
    ("他", "她它"),
]
# variant char -> canonical char
_CJK_HOMOPHONE_MAP = str.maketrans(
    {variant: canonical for canonical, variants in CJK_HOMOPHONE_GROUPS for variant in variants}
)

# ── Japanese script folding (lenient metric only) ─────────────────────────────
# Katakana and hiragana spellings of a word sound identical. Katakana→hiragana is a 1:1,
# length-preserving codepoint map (U+30A1-U+30F6 → U+3041-U+3096); ー and ヷヸヹヺ are left alone.
# Kanji↔kana is NOT folded: it needs a reading dictionary, is ambiguous and not length-preserving.
_KATAKANA_TO_HIRAGANA = str.maketrans({cp: cp - 0x60 for cp in range(0x30A1, 0x30F7)})

# ── CJK numeral notation folding (lenient metric only) ────────────────────────
# ASR may write "2006年" where the reference has "二零零六年". A per-digit map matches the
# positional readings used by Seed-TTS references and is length-preserving (unlike num2words).
_CJK_NUMERAL_MAP = str.maketrans({
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
    # Ideographic zero and the circle sometimes used for it in years (二〇二〇).
    "〇": "零", "○": "零",
})


def cjk_normalizer(text: str, fold_homophones: bool = False, fold_numerals: bool = False, language: str = None) -> str:
    """Normalize a CJK string for character-level (CER) scoring.

    Mirrors seed-tts-eval's `process_one`: strip punctuation, fold traditional→simplified (Chinese
    only), lowercase embedded Latin, and drop whitespace. With both flags off this is the strict,
    seed-tts-eval-comparable form. `fold_homophones` applies CJK_HOMOPHONE_GROUPS (Chinese) or
    katakana→hiragana (Japanese); `fold_numerals` maps Arabic digits to CJK numerals. Every fold is
    length-preserving and applied to both sides, so it can only lower the error rate.
    """
    text = unicodedata.normalize("NFKC", text)
    # Chinese only: on Japanese, zhconv rewrites kanji (俳優 → 俳优); on Korean it folds hanja.
    is_chinese = _is_chinese(text, language)
    if is_chinese:
        text = _to_simplified(text)
    text = text.lower()
    # Drop every mark/symbol/punctuation (CJK and ASCII); keep the apostrophe, as seed-tts-eval does.
    text = "".join(c for c in text if c == "'" or unicodedata.category(c)[0] not in "MSP")
    # The Chinese groups must not run on Japanese (的/地/得 are distinct morphemes there).
    if fold_homophones:
        text = text.translate(_CJK_HOMOPHONE_MAP if is_chinese else _KATAKANA_TO_HIRAGANA)
    if fold_numerals:
        text = text.translate(_CJK_NUMERAL_MAP)
    return "".join(text.split())


def _is_mostly_cjk(texts, threshold: float = 0.5) -> bool:
    """True when most of `texts` contain CJK characters (so the manifest wants CER)."""
    sample = [t for t in texts if t and t.strip()]
    if not sample:
        return False
    return sum(1 for t in sample if _CJK_PATTERN.search(t)) / len(sample) >= threshold


def normalize_compound_pairs(refs, preds):
    """Align compound word boundaries between ref/pred pairs.

    When a mismatch region has identical characters ignoring whitespace,
    normalize both sides to the joined form.
    """
    new_refs, new_preds = [], []
    for ref_text, pred_text in zip(refs, preds):
        ref_words = ref_text.split()
        pred_words = pred_text.split()

        sm = SequenceMatcher(None, ref_words, pred_words)
        new_rw, new_pw = [], []

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                new_rw.extend(ref_words[i1:i2])
                new_pw.extend(pred_words[j1:j2])
            else:
                rc = "".join(ref_words[i1:i2])
                pc = "".join(pred_words[j1:j2])
                if rc == pc:
                    new_rw.append(rc)
                    new_pw.append(pc)
                else:
                    new_rw.extend(ref_words[i1:i2])
                    new_pw.extend(pred_words[j1:j2])

        new_refs.append(" ".join(new_rw))
        new_preds.append(" ".join(new_pw))
    return new_refs, new_preds


def read_manifest(manifest_path: str):
    """
    Reads a manifest file (jsonl format) and returns a list of dictionaries containing samples.
    """
    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(line) > 0:
                datum = json.loads(line)
                data.append(datum)
    return data


# SIM backends that score into their own manifest fork, <manifest>_<backend>.jsonl (keep in sync
# with `sim_manifest_path` in transformers/score_similarity.py). `xvector` writes `sim` in place.
SIM_BACKEND_SUFFIXES = ("wavlm_seed_tts",)


def score_results(directory: str, model_id: str = None, multilingual: bool = None, csv_only: bool = False, language: str = "en", manifests: list = None, sim_backend: str = "wavlm_seed_tts"):
    """
    Scores all result files in a directory and returns a composite score over all evaluated datasets.

    Args:
        directory: Path to the result directory, containing one or more jsonl files.
        model_id: Optional, model name to filter out result files based on model name.
        multilingual: If True, apply compound word boundary normalization before
                      WER computation. Default (None): `default_multilingual(language)`.
        csv_only: If True, suppress the per-dataset and composite printouts.
        language: Language code used for normalization (e.g. 'en', 'de', 'fr'). Non-'en' uses
                  ml_normalizer. CJK is scored per character (CER); it is also auto-detected per
                  manifest from the reference text.
        manifests: Optional list of manifest file names (basenames or paths) to restrict scoring
                  to. `language` applies to every manifest in a call and only CJK is auto-detected,
                  so multi-language folders must be scored one language per call.
        sim_backend: Which SIM manifest family to score; only one per call, since the SIM scales
                  differ. 'wavlm_seed_tts' prefers each `_wavlm_seed_tts.jsonl` fork, falling back
                  to the plain manifest (WER/RTFx only, SIM suppressed). 'xvector' reads only the
                  plain manifests.

    Returns:
        Composite score over all evaluated datasets and a dictionary of all results. Each
        result carries a `metric` field ('WER' or 'CER') saying which error rate `wer` holds.
    """

    if multilingual is None:
        multilingual = default_multilingual(language)

    # Strip trailing slash
    if directory.endswith(os.sep):
        directory = directory[:-1]

    # Find all result files in the directory
    result_files = list(glob.glob(f"{directory}/**/*.jsonl", recursive=True))
    result_files = list(sorted(result_files))

    # Keep exactly one SIM manifest family — see the `sim_backend` arg.
    def _sim_suffix_of(fp: str):
        stem = os.path.basename(fp).removesuffix(".jsonl")
        return next((s for s in SIM_BACKEND_SUFFIXES if stem.endswith(f"_{s}")), None)

    def _canonical(name):
        """Path with any SIM-backend suffix removed — the identity of a (model, dataset) manifest."""
        head, base = os.path.split(name)
        stem = base.removesuffix(".jsonl")
        suffix = _sim_suffix_of(name)
        return os.path.join(head, (stem.removesuffix(f"_{suffix}") if suffix else stem) + ".jsonl")

    # Reject unknown names rather than silently reporting the other SIM scale ($SIM_BACKEND is
    # otherwise unvalidated).
    if sim_backend not in ("xvector",) + SIM_BACKEND_SUFFIXES:
        raise ValueError(
            f"Unknown sim_backend {sim_backend!r}; expected 'xvector' or one of {list(SIM_BACKEND_SUFFIXES)}."
        )

    # Files kept for WER/RTFx whose `sim` is on a different backend's scale and must not be reported.
    sim_unavailable = set()
    if sim_backend in SIM_BACKEND_SUFFIXES:
        # One file per (model, dataset): the requested fork if present, else the plain manifest
        # (e.g. fixed-voice models, which have no SIM stage).
        by_canonical = defaultdict(dict)
        for fp in result_files:
            by_canonical[_canonical(fp)][_sim_suffix_of(fp)] = fp
        chosen = []
        for variants in by_canonical.values():
            if sim_backend in variants:
                chosen.append(variants[sim_backend])
            elif None in variants:
                chosen.append(variants[None])
                sim_unavailable.add(variants[None])
        result_files = sorted(chosen)
    else:
        result_files = [fp for fp in result_files if _sim_suffix_of(fp) is None]

    # Filter files belonging to a specific model id
    if model_id is not None and model_id != "":
        print("Filtering models by id:", model_id)
        model_id = model_id.replace("/", "-")
        result_files = [
            fp for fp in result_files
            if f"/{model_id}/" in fp or f"MODEL_{model_id}_DATASET_" in fp
        ]

    # Compared by basename with any SIM-backend suffix stripped from both sides, so callers can
    # name the plain manifests whichever SIM family is scored.
    if manifests:
        wanted = {os.path.basename(_canonical(m)) for m in manifests}
        result_files = [fp for fp in result_files if os.path.basename(_canonical(fp)) in wanted]

    # Check if any result files were found
    if len(result_files) == 0:
        raise ValueError(f"No result files found in {directory}")

    # Utility function to parse the file path and extract model id, dataset path, dataset name and split
    def parse_filepath(fp: str):
        model_index = fp.find("MODEL_")
        fp = fp[model_index:]
        ds_index = fp.find("DATASET_")
        model_id = fp[:ds_index].replace("MODEL_", "").rstrip("_")
        author_index = model_id.find("-")
        model_id = model_id[:author_index] + "/" + model_id[author_index + 1 :]

        ds_fp = fp[ds_index:]
        dataset_id = ds_fp.replace("DATASET_", "").removesuffix(".jsonl")
        # Drop the SIM-backend suffix so dataset ids are the same whichever family is scored.
        for suffix in SIM_BACKEND_SUFFIXES:
            dataset_id = dataset_id.removesuffix(f"_{suffix}")
        return model_id, dataset_id

    # Compute WER results per dataset, and RTFx over all datasets
    from normalizer import data_utils
    results = {}
    stale_sim_files = []  # (basename, sim_model) for manifests whose SIM was on the other scale

    for result_file in result_files:
        manifest = read_manifest(result_file)
        model_id_of_file, dataset_id = parse_filepath(result_file)

        raw_references = [datum["text"] for datum in manifest]
        raw_predictions = [datum["pred_text"] for datum in manifest]

        # CJK is scored per character (CER); detected from `language` or the reference text.
        use_cer = is_cjk_language(language) or _is_mostly_cjk(raw_references)
        per_char = not use_cer and language in CHAR_LANGUAGES

        if use_cer:
            # `language` also gates the Chinese-only folds.
            normalize = lambda t: cjk_normalizer(t, language=language)
        elif language == "en":
            normalize = data_utils.normalizer
        else:
            normalize = lambda t: data_utils.ml_normalizer(t, lang=language)
        references = [normalize(t) for t in raw_references]
        predictions = [normalize(t) for t in raw_predictions]

        time = [datum["time"] for datum in manifest]
        duration = [datum["duration"] for datum in manifest]
        compute_rtfx = all(time) and all(duration)

        # SIM is optional (voice-cloning backends only); averaged over rows that have it. Files in
        # `sim_unavailable` carry another backend's scale, so no SIM is reported for them.
        if result_file in sim_unavailable:
            sims = []
            if any(isinstance(datum.get("sim"), (int, float)) for datum in manifest):
                stale_sim_files.append((os.path.basename(result_file), manifest[0].get("sim_model")))
        else:
            sims = [datum["sim"] for datum in manifest if isinstance(datum.get("sim"), (int, float))]
        sim = round(100 * sum(sims) / len(sims), 2) if sims else None

        if use_cer:
            # Per-character alignment (merge_compounds is moot). The headline CJK number is the
            # lenient CER (homophone + numeral folds), always <= the unnormalized CER.
            _lenient = lambda t: cjk_normalizer(t, fold_homophones=True, fold_numerals=True, language=language)
            folded = batch_error_rate(
                [tuple(_lenient(t)) for t in raw_references],
                [tuple(_lenient(t)) for t in raw_predictions],
                merge_compounds=False,
            )
            total_ins, total_del, total_sub = folded["ins"], folded["del"], folded["sub"]
            wer = folded["err_rate"]

            # Unnormalized CER, directly comparable with seed-tts-eval's run_wer.py.
            r = batch_error_rate(
                [tuple(ref) for ref in references],
                [tuple(pred) for pred in predictions],
                merge_compounds=False,
            )
            cer_unnormalized = round(100 * r["err_rate"], 2)
        else:
            cer_unnormalized = None
            # Hyphens are inaudible: fold them to whitespace on both sides. ml_normalizer keeps
            # them, and num2words hyphenates e.g. French numerals ("dix-sept").
            references = [r.replace("-", " ") for r in references]
            predictions = [p.replace("-", " ") for p in predictions]
            if multilingual:
                # Pre-fold compound boundaries (e.g. German compounds); complements merge_compounds
                # below, which also catches split compounds adjacent to other errors.
                references, predictions = normalize_compound_pairs(references, predictions)
            # Use kaldialign batch_error_rate with merge_compounds=True so that
            # split compounds (e.g. "white paper" vs "whitepaper") count as
            # 0 errors in either direction.
            if per_char:
                refs_split  = [tuple(r.replace(" ", "")) for r in references]
                preds_split = [tuple(p.replace(" ", "")) for p in predictions]
            else:
                refs_split  = [tuple(r.split()) for r in references]
                preds_split = [tuple(p.split()) for p in predictions]
            r = batch_error_rate(refs_split, preds_split, merge_compounds=not per_char)
            total_ins, total_del, total_sub = r["ins"], r["del"], r["sub"]
            wer = r["err_rate"]

        wer = round(100 * wer, 2)

        if compute_rtfx:
            audio_length = sum(duration)
            inference_time = sum(time)
            rtfx = round(audio_length / inference_time, 4)
        else:
            audio_length = inference_time = rtfx = None

        result_key = f"{model_id_of_file} | {dataset_id}"
        extra = {"ins": total_ins, "del": total_del, "sub": total_sub}
        # `wer` holds the headline error rate; `metric` says whether it is a WER or a CER. For CJK
        # it is the lenient CER, with `cer_unnormalized` the seed-tts-eval-comparable number.
        results[result_key] = {"wer": wer, "metric": "CER" if use_cer or per_char else "WER", "cer_unnormalized": cer_unnormalized, "audio_length": audio_length, "inference_time": inference_time, "rtfx": rtfx, "sim": sim, **extra}

    # Name the manifests whose SIM was dropped, even under csv_only, so a blank SIM is explained.
    if stale_sim_files:
        print(
            f"NOTE: {len(stale_sim_files)} manifest(s) have a SIM from a different backend, so no SIM "
            f"is reported for them. Re-run the SIM stage with --sim_backend={sim_backend}:"
        )
        for name, was in sorted(stale_sim_files):
            print(f"    {name}  (scored by: {was or 'unknown'})")

    if not csv_only:
        print("*" * 80)
        print("Results per dataset:")
        print("*" * 80)

        for k, v in results.items():
            metrics = f"{k}: {v['metric']} = {v['wer']:0.2f} %"
            if v.get("cer_unnormalized") is not None:
                metrics += f" (unnormalized {v['cer_unnormalized']:0.2f} %)"
            if v["rtfx"] is not None:
                metrics += f", RTFx = {v['rtfx']:0.2f}"
            if v.get("sim") is not None:
                metrics += f", SIM = {v['sim']:0.2f} %"
            print(metrics)

    # composite WER should be computed over all datasets and with the same key
    composite_wer = defaultdict(float)
    composite_audio_length = defaultdict(float)
    composite_inference_time = defaultdict(float)
    composite_sim = defaultdict(float)
    count_sim_entries = defaultdict(int)
    count_entries = defaultdict(int)
    composite_metrics = defaultdict(set)
    for k, v in results.items():
        key = k.split("|")[0].strip()
        composite_metrics[key].add(v["metric"])
        composite_wer[key] += v["wer"]
        # Composite RTFx pools only the datasets that have timings.
        if v["rtfx"] is not None:
            composite_audio_length[key] += v["audio_length"]
            composite_inference_time[key] += v["inference_time"]
        if v.get("sim") is not None:
            composite_sim[key] += v["sim"]
            count_sim_entries[key] += 1
        count_entries[key] += 1

    # normalize scores & print
    if not csv_only:
        print()
        print("*" * 80)
        print("Composite Results:")
        print("*" * 80)
        for k, v in composite_wer.items():
            wer = v / count_entries[k]
            # Label mixed WER/CER averages as such.
            label = "/".join(sorted(composite_metrics[k]))
            print(f"{k}: {label} = {wer:0.2f} %")
        for k in composite_audio_length:
            rtfx = composite_audio_length[k] / composite_inference_time[k]
            print(f"{k}: RTFx = {rtfx:0.2f}")
        for k in composite_sim:
            sim = composite_sim[k] / count_sim_entries[k]
            print(f"{k}: SIM = {sim:0.2f} %")
        print("*" * 80)

    return composite_wer, results
