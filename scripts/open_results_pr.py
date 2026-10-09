#!/usr/bin/env python3
"""Pull a model's results from the HF bucket and open a PR against the published TTS CSVs.

Scores the bucket manifests exactly the way `scripts/score_model.py` does (same
`score_results`, one call per language so each gets its own normalizer), turns the per-language
numbers into one row per benchmark file, upserts those rows into
`hf-audio/tts_leaderboard_results`, and opens a single pull request carrying every changed file.

Files written (all in that dataset repo):
    model_info.csv            metadata -- scoring cannot derive it, so it comes from the CLI flags
    seed_tts.csv              Seed-TTS-Eval, model's own/default voice   (WER, RTFx, UTMOS)
    seed_tts_voice_clone.csv  Seed-TTS-Eval, cloning each prompt         (WER, RTFx, SIM x2, UTMOS)
    cv3.csv                   CV3-Eval, model's own/default voice        (WER, RTFx, UTMOS + averages)
    cv3_voice_clone.csv       CV3-Eval, cloning each prompt              (WER, RTFx, SIM, UTMOS + averages)
    streaming.csv             time-to-first-audio, CPU and GPU           (TTFA, RTFx at batch 1)

Nothing is pushed unless --open_pr is passed: the default is a dry run that prints the rows.

Usage:
    # dry run (default): show the rows that would be written
    python scripts/open_results_pr.py --model_id openbmb/VoxCPM2

    # the same, with the model_info.csv metadata, and actually open the PR
    python scripts/open_results_pr.py --model_id openbmb/VoxCPM2 \
        --license apache-2.0 --voice_cloning yes --batch_inference no \
        --num_languages 9 --model_size_b 0.5 --transformers no --open_pr

    # one benchmark only
    python scripts/open_results_pr.py --model_id tencent/AuK --targets cv3_voice_clone --open_pr

    # a backend variant that is published as its own row
    python scripts/open_results_pr.py --model_id ResembleAI/chatterbox --variant multilingual \
        --row_name "ResembleAI/chatterbox (ChatterboxMultilingualTTS)" --open_pr

    # TTFA only, for a model probed through a separate inference engine
    python scripts/open_results_pr.py --model_id Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --targets streaming \
        --engine photon --row_name "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (photon)" --open_pr \
        --note "https://moondream.ai/blog/photon-can-now-speak" --model_size_b 1.92

    # re-use results already synced locally (e.g. by submit_jobs.sh)
    python scripts/open_results_pr.py --model_id openbmb/VoxCPM2 --skip_sync
"""

import argparse
import contextlib
import csv
import io
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # so `from normalizer import ...` resolves regardless of cwd

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download  # noqa: E402

from normalizer.eval_utils import SIM_BACKEND_SUFFIXES, default_multilingual, score_results  # noqa: E402

# Where the results live, and where they are published.
RESULTS_BUCKET = "hf-audio/tts_leaderboard_h200"
RESULTS_REPO = "hf-audio/tts_leaderboard_results"

MODEL_INFO_FILE = "model_info.csv"

# model_info.csv columns, and the flag each one is supplied by. Scoring cannot know any of these.
# A flag that is not passed leaves the published cell untouched; a flag passed as "" clears it.
METADATA_ARGS = {
    "license": "license",
    "voice cloning": "voice_cloning",
    "batch inference": "batch_inference",
    "# languages": "num_languages",
    "Transformers": "transformers",
    "model size (B)": "model_size_b",
}

# Second SIM column on seed_tts_voice_clone.csv: the `xvector` scale (WavLMForXVector), published
# alongside the seed-tts-eval scale; the two are NOT comparable (see --sim_backend in
# scripts/tts_jobs_common.sh).
XVECTOR_SIM_COLUMN = "{lang} SIM (microsoft/wavlm-base-plus-sv)"
XVECTOR_BACKEND = "xvector"

CLONE_SUFFIX = "_voice_clone"


class Target:
    """One published benchmark file and the manifests that feed it.

    marker: the `<eval>_<config>` substring a manifest name carries for this benchmark (a
        substring, since the dataset namespace in front of it differs per user; see
        scripts/prepare_*_eval.py).
    voice_clone: which side of the `_voice_clone` split this file publishes.
    xvector_sim: this file also carries the xvector-scale SIM column, which needs a second scoring
        pass (one `score_results` call reports one SIM scale).
    """

    def __init__(self, filename, marker, voice_clone, xvector_sim=False):
        self.filename = filename
        self.marker = marker
        self.voice_clone = voice_clone
        self.xvector_sim = xvector_sim


# Keyed by --targets value; the order here is the order the files are reported and committed in.
TARGETS = {
    "seed_tts": Target("seed_tts.csv", "seed_tts_eval_tts_", voice_clone=False),
    "seed_tts_voice_clone": Target(
        "seed_tts_voice_clone.csv", "seed_tts_eval_tts_", voice_clone=True, xvector_sim=True
    ),
    "cv3": Target("cv3.csv", "cv3_eval_zero_shot_", voice_clone=False),
    "cv3_voice_clone": Target("cv3_voice_clone.csv", "cv3_eval_zero_shot_", voice_clone=True),
}

# streaming.csv is filled from the TTFA sidecars rather than by scoring, so it is not a Target.
STREAMING_KEY = "streaming"
STREAMING_FILE = "streaming.csv"
TTFA_PREFIX = "TTFA_"
# Sidecar flavor suffix -> the prefix of the streaming.csv columns it fills
TTFA_COLUMN_GROUPS = {"cpu-upgrade": "CPU", "h200": "H200", "a100-large": "A100"}
# Which group's streaming verdict wins when they disagree: the GPU probes first, newest hardware first.
TTFA_VERDICT_ORDER = ["A100", "H200", "CPU"]
# Row-level columns of streaming.csv that are not per-device measurements.
STREAMING_API_COLUMN = "Streaming API"
STREAMING_NOTE_COLUMN = "Note"
STREAMING_SIZE_COLUMN = "Size (B)"


# ── Bucket ───────────────────────────────────────────────────────────────────
def sync_bucket(bucket, model_folder, local_dir, hf_token=None, clean=False, include=None):
    """Sync one model's folder out of the bucket, manifests only (scoring never reads the wavs).

    Same call `_wait_and_score` in scripts/tts_jobs_common.sh makes, so either's folder is usable.
    """
    source = f"hf://buckets/{bucket}/{model_folder}"
    dest = os.path.join(local_dir, model_folder)
    if clean and os.path.isdir(dest):
        # `hf buckets sync` never deletes, so stale local manifests would otherwise still be scored.
        print(f"Removing {dest} (--clean) ...")
        subprocess.run(["rm", "-rf", dest], check=True)
    print(f"Syncing {source}  →  {dest} ...")
    os.makedirs(dest, exist_ok=True)
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
    subprocess.run(
        ["hf", "buckets", "sync", source, dest, *(["--include", include] if include else ["--exclude", "*.wav"])],
        check=True,
        env=env,
    )
    print("Sync complete.\n")


# ── Manifest discovery ───────────────────────────────────────────────────────
def canonical_stem(basename):
    """Manifest name with the `.jsonl` and any SIM-backend suffix removed.

    A non-xvector SIM backend scores into a fork `<manifest>_<backend>.jsonl`; `score_results`
    picks the right fork itself given the plain name.
    """
    stem = basename.removesuffix(".jsonl")
    for suffix in SIM_BACKEND_SUFFIXES:
        stem = stem.removesuffix(f"_{suffix}")
    return stem


def select_manifests(model_dir, model_safe, target, variant):
    """Manifests in `model_dir` belonging to `target` (and `variant`).

    Returns ({canonical stem: language}, [stems of the right benchmark that did not match]); the
    latter usually means a backend variant (`_multilingual`, `_rl`) needs naming with --variant.
    """
    selected, unmatched = {}, []
    prefix = f"MODEL_{model_safe}_DATASET_"
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".jsonl"):
            continue
        stem = canonical_stem(name)
        if not stem.startswith(prefix):
            continue
        index = stem.find(target.marker)
        if index < 0:
            continue
        rest = stem[index + len(target.marker):]
        is_clone = rest.endswith(CLONE_SUFFIX)
        if is_clone != target.voice_clone:
            continue
        if is_clone:
            rest = rest[: -len(CLONE_SUFFIX)]
        # What remains is `<lang><variant>`. The variant comes from the CLI rather than being
        # guessed: Seed-TTS's `zh_hard` split would otherwise read as `zh` + variant `_hard`.
        if variant:
            if not rest.endswith(variant):
                unmatched.append(stem)
                continue
            rest = rest[: -len(variant)]
        selected[stem] = rest
    return selected, unmatched


def select_ttfa_sidecars(model_dir, model_safe, variant, engine):
    """TTFA sidecars for this model/variant/engine, as {column group: (path, payload)}, the group
    being a TTFA_COLUMN_GROUPS value ("CPU", "H200", "A100").

    One sidecar per group is published; sidecars from a flavor with no group are listed and
    skipped. When a group has several (different splits, or
    both a default-voice and a voice-clone probe) the default-voice one wins -- TTFA does not depend on
    the corpus (see scripts/ttfa_probe.py) -- then the most recent; the others are listed.
    """
    prefix = f"{TTFA_PREFIX}MODEL_{model_safe}_DATASET_"
    candidates = {}
    for name in sorted(os.listdir(model_dir)):
        if not (name.startswith(prefix) and name.endswith(".json")):
            continue
        # `<dataset>_<config>_<split><mode>[__<engine>]__<flavor>`
        parts = name[len(prefix):].removesuffix(".json").split("__")
        if len(parts) < 2:
            continue  # untagged sidecar (a probe run outside submit_ttfa_jobs.sh): hardware unknown
        rest, flavor = parts[0], parts[-1]
        tag = "__".join(parts[1:-1])
        if tag != engine:
            continue
        is_clone = rest.endswith(CLONE_SUFFIX)
        rest = rest.removesuffix(CLONE_SUFFIX)
        # Same rule as the manifests: a variant is only matched when named.
        if variant and not rest.endswith(variant):
            continue
        path = os.path.join(model_dir, name)
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not variant and payload.get("mode_suffix", "").removesuffix(CLONE_SUFFIX):
            continue  # a variant run (e.g. `_rl`); its own row needs --variant
        device = TTFA_COLUMN_GROUPS.get(flavor)
        if device is None:
            print(f"  NOTE: skipping {name}: {STREAMING_FILE} has no columns for flavor {flavor} "
                  f"(published flavors: {', '.join(TTFA_COLUMN_GROUPS)})")
            continue
        candidates.setdefault(device, []).append((is_clone, -os.path.getmtime(path), path, payload))
    chosen = {}
    for device, found in candidates.items():
        found.sort(key=lambda c: (c[0], c[1]))
        chosen[device] = (found[0][2], found[0][3])
        for other in found[1:]:
            print(f"  NOTE: {device} TTFA taken from {os.path.basename(found[0][2])}; "
                  f"ignoring {os.path.basename(other[2])}")
    return chosen


def streaming_column(header, device, metric, stat=None):
    """The header's `<device> ... <metric> ... <stat>` column (the published names are not uniform:
    `CPU RTFx` but `H200 (RTFx)`), or None. `stat` ("(p50)", "(p95)") tells apart the several TTFA
    columns of one device."""
    for column in header:
        if column.startswith(device + " ") and metric in column and (stat is None or stat in column):
            return column
    return None


def collect_streaming_values(header, sidecars):
    """streaming.csv row body from the chosen sidecars. Returns (values, notes)."""
    values, notes = {}, []
    for device, (path, payload) in sorted(sidecars.items()):
        ttfa = payload.get("ttfa_ms") or {}
        # p95 is absent from sidecars written before the probe reported it: that column is then
        # left as published (empty for those models) rather than guessed.
        for metric, stat, value in (("TTFA", "(p50)", ttfa.get("p50")),
                                    ("TTFA", "(p95)", ttfa.get("p95")),
                                    ("RTFx", None, payload.get("rtfx_batch1"))):
            column = streaming_column(header, device, metric, stat)
            if column and value is not None:
                values[column] = fmt(value)
        notes.append(f"{device}: {os.path.basename(path)}")
        if payload.get("first_reported_but_not_early"):
            notes.append(f"NOTE: {device} probe reported first-chunk timestamps that were not early "
                         f"(ttfa/gen {payload.get('ttfa_frac_of_gen')}); published as not streaming.")
    # A GPU probe decides when they disagree: it is the headline measurement.
    ordered = [g for g in TTFA_VERDICT_ORDER if g in sidecars]
    verdicts = [sidecars[g][1].get("streaming") for g in ordered]
    if verdicts and verdicts[0] is not None:
        values[STREAMING_API_COLUMN] = "yes" if verdicts[0] else "no"
        if len(set(verdicts)) > 1:
            notes.append(f"NOTE: the {', '.join(ordered)} probes disagree on streaming; "
                         f"the {ordered[0]} verdict is used.")
    return values, notes


# ── Scoring ──────────────────────────────────────────────────────────────────
def run_scoring(model_dir, model_id, manifests_by_language, sim_backend):
    """Score each language separately; return {dataset_id: result}.

    One call per language because `score_results` applies its `language` to every manifest it
    reads (same as `_wait_and_score` in scripts/tts_jobs_common.sh).
    """
    scored = {}
    for language, manifests in sorted(manifests_by_language.items()):
        multilingual = default_multilingual(language)
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                _, results = score_results(
                    model_dir,
                    model_id=model_id,
                    csv_only=True,
                    language=language,
                    multilingual=multilingual,
                    manifests=sorted(f"{stem}.jsonl" for stem in manifests),
                    sim_backend=sim_backend,
                )
        except ValueError as exc:
            # No manifests for this language -- normal when a model was not run on it.
            print(f"  [{language}] skipped: {exc}")
            continue
        # Surface anything score_results had to say (notably the stale-SIM NOTE behind a blank SIM).
        for line in buffer.getvalue().splitlines():
            if line.strip() and not line.startswith("Filtering models by id:"):
                print(f"  {line}")
        for key, value in results.items():
            # Keyed "<model> | <dataset id>"; only the dataset half is used, since the model half
            # is rebuilt from the file name and mangles hyphenated namespaces (k2-fsa/OmniVoice).
            scored[key.split("|", 1)[1].strip()] = value
        print(f"  [{language}] scored {len(manifests)} manifest(s) (multilingual={multilingual})")
    return scored


def fmt(value, digits=2):
    """Published-CSV spelling of a metric: 2 decimals, no trailing zeros, blank when absent."""
    if value is None:
        return ""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def collect_values(target, header_languages, records, scored, scored_xvector):
    """Merge one target's per-language results into a {column: value} row body."""
    values, unknown = {}, []
    for stem, language in sorted(records.items()):
        if language not in header_languages:
            unknown.append((stem, language))
            continue
        # The dataset id is what `score_results` keys its results by: everything after `_DATASET_`.
        dataset_id = stem.split("_DATASET_", 1)[1]
        result = scored.get(dataset_id)
        if result is None:
            continue
        values[f"{language} WER"] = fmt(result["wer"])
        values[f"{language} RTFx"] = fmt(result["rtfx"])
        if result.get("sim") is not None:
            values[f"{language} SIM"] = fmt(result["sim"])
        if result.get("utmos") is not None:
            values[f"{language} UTMOS"] = fmt(result["utmos"])
        if target.xvector_sim:
            legacy = scored_xvector.get(dataset_id)
            if legacy is not None and legacy.get("sim") is not None:
                values[XVECTOR_SIM_COLUMN.format(lang=language)] = fmt(legacy["sim"])
    return values, unknown


# ── Published CSVs ───────────────────────────────────────────────────────────
class Published:
    """A published CSV, kept as its own bytes alongside the parsed rows.

    The raw text lets an update splice a single line in (see `splice_row`): a csv round-trip is not
    byte-identical, so every PR would otherwise be a whole-file rewrite.
    """

    def __init__(self, text):
        self.text = text
        self.lines = text.splitlines(keepends=True)
        reader = csv.reader(io.StringIO(text, newline=""))
        self.header = next(reader)
        # Each row's span of physical lines, so it can be found again in `self.lines`. A quoted
        # field may hold a newline, so a row is not necessarily one line.
        self.rows, self.spans = [], []
        start = reader.line_num
        for row in reader:
            end = reader.line_num
            if row:
                self.rows.append(row)
                self.spans.append((start, end))
            start = end
        self.eol = line_terminator(self.lines[0]) if self.lines else "\n"
        self.trailing_newline = bool(self.lines) and bool(line_terminator(self.lines[-1]))

    def index_of(self, row_name):
        """Position of `row_name` in `self.rows`, or None if the model is not published yet."""
        for i, row in enumerate(self.rows):
            if row and row[0].strip() == row_name:
                return i
        return None


def line_terminator(line):
    """The line's own ending, or "" for a last line that has none."""
    for ending in ("\r\n", "\n", "\r"):
        if line.endswith(ending):
            return ending
    return ""


def fetch_csv(filename, hf_token):
    path = hf_hub_download(RESULTS_REPO, filename, repo_type="dataset", token=hf_token)
    with open(path, newline="", encoding="utf-8") as handle:
        return Published(handle.read())


def serialize_row(row):
    """One CSV record, with no line ending of its own."""
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerow(row)
    return buffer.getvalue()[:-1]


def splice_row(published, index, new_row):
    """The file's content with just this model's line replaced, or appended.

    Every other byte is the published file's own, so the PR diff is one line and merges cleanly
    with PRs for other models. The one exception: a missing final newline is added, so later
    appends are clean single-line insertions too.
    """
    lines = list(published.lines)
    serialized = serialize_row(new_row)
    eol = published.eol or "\n"
    if index is None:
        if lines and not published.trailing_newline:
            lines[-1] += eol
        lines.append(serialized + eol)
    else:
        first, last = published.spans[index]
        lines[first:last] = [serialized + (line_terminator(lines[last - 1]) or eol)]
    return "".join(lines)


def rewrite_sorted(published, index, new_row):
    """Whole file, rows re-sorted by model. Only --sort takes this path, since it moves lines on
    purpose; the file's own line ending is still kept."""
    rows = list(published.rows)
    if index is None:
        rows.append(new_row)
    else:
        rows[index] = new_row
    rows.sort(key=lambda row: (row[0] or "").lower())
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator=published.eol or "\n")
    writer.writerow(published.header)
    writer.writerows(rows)
    return buffer.getvalue()  # ends with a newline, like splice_row -- see its docstring


def render(published, index, new_row, sort):
    return rewrite_sorted(published, index, new_row) if sort else splice_row(published, index, new_row)


def add_utmos_columns(published, languages):
    """Append `<lang> UTMOS` (+ `avg UTMOS` if the file has averages) to a header without them."""
    header = published.header
    if any(column.endswith(" UTMOS") for column in header):
        return published
    header = header + [f"{lang} UTMOS" for lang in languages]
    if any(column.startswith("avg ") for column in header):
        header.append("avg UTMOS")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator=published.eol or "\n")
    writer.writerow(header)
    for row in published.rows:
        writer.writerow(row + [""] * (len(header) - len(row)))
    return Published(buffer.getvalue())


def header_languages(header):
    """Languages the file publishes, read off its own header (every one has a `<lang> WER`)."""
    return [
        column[: -len(" WER")]
        for column in header
        if column.endswith(" WER") and column != "avg WER"
    ]


def upsert(published, row_name, values, metadata):
    """Build `row_name`'s row. Returns (action, index, new_row).

    `index` is the model's position in `published.rows`, or None when the model is new; the
    caller hands both to `render`, which splices only that one line.
    """
    header = published.header
    new_row = [row_name if column == "model" else values.get(column, "") for column in header]

    index = published.index_of(row_name)
    action = "added" if index is None else "updated"
    if index is not None:
        # Keep anything already published in a column this run has no value for: a re-run of
        # one language must not blank the other languages, and hand-curated metadata must
        # survive a results-only submission.
        published_row = published.rows[index]
        padded = published_row + [""] * (len(header) - len(published_row))
        new_row = [new if new != "" else old for new, old in zip(new_row, padded)]

    # Metadata is applied after the merge, not through it, so that a flag passed as "" clears the
    # published cell instead of reading as "this run has no opinion".
    columns = {column: i for i, column in enumerate(header)}
    for column, value in metadata.items():
        if column in columns:
            new_row[columns[column]] = value
    return action, index, new_row


def recompute_averages(header, row, languages):
    """Fill the `avg <metric>` columns from the MERGED row's per-language columns, so
    re-submitting one language keeps the average over everything published. Returns notes to print.
    """
    notes = []
    index = {column: i for i, column in enumerate(header)}
    for metric in ("WER", "RTFx", "SIM", "UTMOS"):
        avg_column = f"avg {metric}"
        if avg_column not in index:
            continue
        present, missing = [], []
        for language in languages:
            column = f"{language} {metric}"
            if column not in index:
                continue
            raw = row[index[column]].strip()
            if raw:
                present.append(float(raw))
            else:
                missing.append(language)
        if not present:
            continue
        if missing:
            notes.append(
                f"NOTE: '{avg_column}' is averaged over the {len(present)} language(s) with a "
                f"result; no {metric} for: {', '.join(missing)}."
            )
        # Published averages carry full precision; 6 decimals only drops binary-float noise.
        mean = sum(present) / len(present)
        row[index[avg_column]] = f"{mean:.6f}".rstrip("0").rstrip(".")
    return notes


def show_row(header, row, action, filename, notes=()):
    print(f"\n{filename} ({action}):")
    for column, value in zip(header, row):
        if value != "":
            print(f"  {column:<40} {value}")
    blanks = [column for column, value in zip(header, row) if value == ""]
    if blanks:
        print(f"  (blank: {', '.join(blanks)})")
    for note in notes:
        print(f"  {note}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Open a PR adding a model's results to the published TTS leaderboard CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[1],
    )
    parser.add_argument(
        "--model_id",
        required=True,
        help="Model id as evaluated, e.g. openbmb/VoxCPM2. Selects the bucket folder and the "
             "manifests, and labels the row unless --row_name overrides it.",
    )
    parser.add_argument(
        "--row_name",
        default=None,
        help="Value written into the 'model' column (default: --model_id). Needed when one "
             "checkpoint is published as several rows, e.g. "
             "'ResembleAI/chatterbox (ChatterboxMultilingualTTS)'.",
    )
    parser.add_argument(
        "--variant",
        default="",
        help="Backend variant embedded in the manifest names between the language and the mode, "
             "e.g. 'multilingual' (Chatterbox) or 'rl' (CosyVoice 3). Leading '_' optional. "
             "Default: the plain run, i.e. manifests with no variant.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        choices=[*TARGETS, STREAMING_KEY],
        help="Benchmark files to update (default: every one the model has results for).",
    )
    parser.add_argument(
        "--skip_model_info",
        action="store_true",
        help=f"Do not touch {MODEL_INFO_FILE}.",
    )
    parser.add_argument(
        "--engine",
        default="",
        help=f"{STREAMING_FILE} only: the inference-engine tag in the TTFA sidecar name (the "
             "`engine_tag` of a submit_ttfa_jobs.sh target, e.g. 'sglang' or 'ovgguf'). Default: "
             "the sidecar with no engine tag.",
    )
    parser.add_argument(
        "--note",
        default=None,
        help=f"{STREAMING_FILE} 'Note' column. Omitted leaves it as published; '' clears it.",
    )
    parser.add_argument("--bucket", default=RESULTS_BUCKET, help="Override the source bucket.")
    parser.add_argument(
        "--local_dir",
        default=None,
        help="Where to sync results (default: <repo>/results, same place submit_jobs.sh uses).",
    )
    parser.add_argument("--utmos_bucket", default=None, help="Bucket with the UTMOS sidecars (default: --bucket).")
    parser.add_argument("--skip_sync", action="store_true", help="Score already-downloaded results.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the local copy of the model's folder before syncing, so the bucket is the "
             "only source of truth (sync alone never deletes).",
    )
    parser.add_argument(
        "--sim_backend",
        default="wavlm_seed_tts",
        choices=[XVECTOR_BACKEND, *SIM_BACKEND_SUFFIXES],
        help="Which SIM scale fills the 'SIM' columns (default: wavlm_seed_tts, the published "
             "one). The legacy 'xvector' scale has its own column and is always read as well.",
    )
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"), help="Defaults to $HF_TOKEN.")
    parser.add_argument("--open_pr", action="store_true", help="Actually open the PR (default: dry run).")
    parser.add_argument("--commit_message", default=None, help="PR title.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Dry run: also write the updated CSVs to this directory.",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Re-sort rows case-insensitively by model, which rewrites the whole file (default: "
             "append new rows at the end, so the PR diff is the model's one line and merges "
             f"cleanly with other open PRs). {MODEL_INFO_FILE}, seed_tts.csv, "
             f"seed_tts_voice_clone.csv, cv3.csv and {STREAMING_FILE} are alphabetical today; cv3_voice_clone.csv "
             "is ordered by avg WER, so do not sort that one.",
    )

    meta = parser.add_argument_group(
        "model_info.csv columns",
        "Scoring cannot derive these. An omitted flag leaves the published cell as it is; "
        "passing an empty string clears it.",
    )
    meta.add_argument("--license", default=None, help="License, e.g. 'apache-2.0' or a URL to the model's LICENSE.")
    meta.add_argument("--voice_cloning", default=None, choices=["yes", "no"], help="Model supports voice cloning.")
    meta.add_argument("--batch_inference", default=None, choices=["yes", "no"], help="Model supports batched inference.")
    meta.add_argument("--num_languages", default=None, help="Number of languages the model supports.")
    meta.add_argument("--transformers", default=None, choices=["yes", "no"], help="Model is supported in Transformers.")
    meta.add_argument("--model_size_b", default=None, help="Total parameters, in billions.")
    args = parser.parse_args()

    row_name = args.row_name or args.model_id
    model_safe = args.model_id.replace("/", "-")
    model_folder = model_safe
    variant = args.variant
    if variant and not variant.startswith("_"):
        variant = f"_{variant}"
    local_dir = args.local_dir or os.path.join(REPO_ROOT, "results")
    model_dir = os.path.join(local_dir, model_folder)

    if not args.skip_sync:
        try:
            sync_bucket(args.bucket, model_folder, local_dir, args.hf_token, args.clean)
        except subprocess.CalledProcessError as exc:
            print(
                f"ERROR: could not sync {args.bucket}/{model_folder} (exit {exc.returncode}). "
                f"Check the model folder exists in the bucket and that $HF_TOKEN can read it.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.utmos_bucket:
            sync_bucket(args.utmos_bucket, model_folder, local_dir, args.hf_token, include="UTMOS_*.json")
    if not os.path.isdir(model_dir):
        print(f"ERROR: no results directory for {args.model_id}: {model_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Which manifests feed which file ──────────────────────────────────────
    target_keys = args.targets or [*TARGETS, STREAMING_KEY]
    sidecars = {}
    if STREAMING_KEY in target_keys:
        sidecars = select_ttfa_sidecars(model_dir, model_safe, variant, args.engine)
    selection, unmatched = {}, []
    for key in [k for k in target_keys if k in TARGETS]:
        records, skipped = select_manifests(model_dir, model_safe, TARGETS[key], variant)
        if records:
            selection[key] = records
        unmatched += skipped

    if not selection and not sidecars:
        print(f"No manifests or TTFA sidecars for {args.model_id} in {model_dir} matching:")
        print(f"  targets: {', '.join(target_keys)}")
        print(f"  variant: {variant or '(none)'}")
        print(f"  engine:  {args.engine or '(none)'}")
        if unmatched:
            print("Manifests of the right benchmark that a different --variant would select:")
            for stem in sorted(set(unmatched)):
                print(f"    {stem}")
        sys.exit(1)

    # ── Score, one pass per SIM scale ────────────────────────────────────────
    # Languages are pooled across targets: one call scores every manifest of that language,
    # whatever file each ends up in.
    by_language = {}
    for records in selection.values():
        for stem, language in records.items():
            by_language.setdefault(language, set()).add(stem)
    scored = {}
    if by_language:
        print(f"Scoring {sum(len(v) for v in by_language.values())} manifest(s) "
              f"[{args.sim_backend}] ...")
        scored = run_scoring(model_dir, args.model_id, by_language, args.sim_backend)

    # The xvector SIM column needs its own pass: one score_results call reports one SIM scale.
    scored_xvector = {}
    legacy_by_language = {}
    if args.sim_backend != XVECTOR_BACKEND:
        for key, records in selection.items():
            if not TARGETS[key].xvector_sim:
                continue
            for stem, language in records.items():
                legacy_by_language.setdefault(language, set()).add(stem)
    if legacy_by_language:
        print(f"Scoring {sum(len(v) for v in legacy_by_language.values())} manifest(s) "
              f"[{XVECTOR_BACKEND}, for the legacy SIM column] ...")
        scored_xvector = run_scoring(model_dir, args.model_id, legacy_by_language, XVECTOR_BACKEND)

    # ── Build the updated files ──────────────────────────────────────────────
    metadata = {
        column: getattr(args, arg)
        for column, arg in METADATA_ARGS.items()
        if getattr(args, arg) is not None
    }
    # 'Transformers' is published as 'Yes' or blank -- there is no 'No' in the file.
    if metadata.get("Transformers") is not None:
        metadata["Transformers"] = "Yes" if metadata["Transformers"] == "yes" else ""

    updated = {}  # filename -> new content
    for key in [k for k in TARGETS if k in selection]:
        target = TARGETS[key]
        published = fetch_csv(target.filename, args.hf_token)
        header = published.header
        languages = header_languages(header)
        values, unknown = collect_values(
            target, languages, selection[key], scored, scored_xvector
        )
        for stem, language in unknown:
            sys.stdout.flush()  # keep the warning next to the file it is about when piped
            print(
                f"\nWARNING: {target.filename} has no '{language}' columns; skipping {stem}.jsonl."
                + (f" Is '{language}' a language plus a variant? Try --variant." if not variant else ""),
                file=sys.stderr,
            )
        if not values:
            print(f"\n{target.filename}: nothing scored; skipping.")
            continue
        if any(column.endswith(" UTMOS") for column in values):
            published = add_utmos_columns(published, languages)
            header = published.header
        action, index, new_row = upsert(published, row_name, values, {})
        notes = recompute_averages(header, new_row, languages)
        show_row(header, new_row, action, target.filename, notes)
        updated[target.filename] = render(published, index, new_row, args.sort)

    if sidecars:
        published = fetch_csv(STREAMING_FILE, args.hf_token)
        values, notes = collect_streaming_values(published.header, sidecars)
        # Row-level columns shared with other flags: --model_size_b also fills 'Size (B)' here.
        extra = {}
        if args.model_size_b is not None:
            extra[STREAMING_SIZE_COLUMN] = args.model_size_b
        if args.note is not None:
            extra[STREAMING_NOTE_COLUMN] = args.note
        action, index, new_row = upsert(published, row_name, values, extra)
        show_row(published.header, new_row, action, STREAMING_FILE, notes)
        updated[STREAMING_FILE] = render(published, index, new_row, args.sort)
    elif STREAMING_KEY in (args.targets or []):
        print(f"\n{STREAMING_FILE}: no TTFA sidecar for {args.model_id} "
              f"(variant: {variant or '(none)'}, engine: {args.engine or '(none)'}); skipping.")

    if not updated:
        print("\nNo results to submit.")
        sys.exit(1)

    # model_info.csv: written when metadata was passed, or to give a model its first row.
    if not args.skip_model_info:
        published = fetch_csv(MODEL_INFO_FILE, args.hf_token)
        exists = published.index_of(row_name) is not None
        # A row published to a *_voice_clone file is a model that clones voices; fill that in
        # rather than leaving the column blank, unless the flag says otherwise.
        if "voice cloning" not in metadata and any(TARGETS[k].voice_clone for k in updated_keys(updated)):
            if not published_value(published, row_name, "voice cloning"):
                metadata["voice cloning"] = "yes"
                print("\nmodel_info.csv: 'voice cloning' set to 'yes' (a voice-clone file is being "
                      "written); pass --voice_cloning to override.")
        if metadata or not exists:
            action, index, new_row = upsert(published, row_name, {}, metadata)
            show_row(published.header, new_row, action, MODEL_INFO_FILE)
            updated[MODEL_INFO_FILE] = render(published, index, new_row, args.sort)
        else:
            print(f"\n{MODEL_INFO_FILE}: row already published and no metadata flags given; "
                  f"leaving it alone.")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for filename, content in updated.items():
            path = os.path.join(args.out_dir, filename)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
            print(f"Wrote {path}")

    if not args.open_pr:
        print(f"\nDry run - not opening a PR. {len(updated)} file(s) would change: "
              f"{', '.join(sorted(updated))}.\nRe-run with --open_pr to submit.")
        return

    message = args.commit_message or f"Add {row_name} results"
    api = HfApi(token=args.hf_token)
    commit = api.create_commit(
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=filename, path_or_fileobj=content.encode("utf-8"))
            for filename, content in sorted(updated.items())
        ],
        commit_message=message,
        commit_description=(
            f"Scored from `{args.bucket}` with `scripts/open_results_pr.py`.\n\n"
            f"Files: {', '.join(sorted(updated))}"
        ),
        create_pr=True,
    )
    print(f"\nPR opened: {commit.pr_url}")


def updated_keys(updated):
    """Target keys behind the files written so far."""
    return [key for key, target in TARGETS.items() if target.filename in updated]


def published_value(published, row_name, column):
    """The value `row_name` already has in `column`, or "" if there is no such row/column."""
    if column not in published.header:
        return ""
    row_index = published.index_of(row_name)
    column_index = published.header.index(column)
    if row_index is None:
        return ""
    row = published.rows[row_index]
    return row[column_index].strip() if len(row) > column_index else ""


if __name__ == "__main__":
    main()
