#!/usr/bin/env python3
"""Rescore every published (model, benchmark) row after a normalizer change.

Normalization is applied at scoring time to the raw `text` / `pred_text` stored in each manifest,
so a normalizer change only needs the manifests rescored -- no generation or ASR rerun. This walks
every row of the published benchmark CSVs in `hf-audio/tts_leaderboard_results`, rescores it from
the results bucket with the same code path as `scripts/open_results_pr.py`, and writes the new WER
(CER for CJK) columns and `avg WER` back into the CSVs, all in a single PR.

Only already-published WER cells change; a blank cell stays blank. RTFx and SIM do not depend on the normalizer, so they are left as
published; they are instead used as a check that the right manifests were scored: a language
whose rescored RTFx differs from the published one is skipped (pass --force to update it anyway).
streaming.csv and model_info.csv are never touched.

Nothing is pushed unless --open_pr is passed: the default is a dry run that prints the changes.

Usage:
    # dry run over every row of every benchmark file
    python scripts/rescore_all.py

    # some models / benchmarks only, keeping the updated CSVs locally for inspection
    python scripts/rescore_all.py --models k2-fsa/OmniVoice openbmb/VoxCPM2 --targets cv3 \\
        --out_dir /tmp/rescored

    # re-use results already synced locally, and open the PR
    python scripts/rescore_all.py --skip_sync --open_pr
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from huggingface_hub import CommitOperationAdd, HfApi  # noqa: E402

from open_results_pr import (  # noqa: E402
    RESULTS_BUCKET,
    RESULTS_REPO,
    TARGETS,
    Published,
    collect_values,
    fetch_csv,
    header_languages,
    recompute_averages,
    run_scoring,
    select_manifests,
    splice_row,
    sync_bucket,
)

# Published row name -> (model id in the bucket, manifest variant). Rows not listed here are
# scored as their own name with no variant.
ROW_OVERRIDES = {
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (RL)": ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "_rl"),
    "ResembleAI/chatterbox (ChatterboxTTS)": ("ResembleAI/chatterbox", ""),
    "ResembleAI/chatterbox (ChatterboxMultilingualTTS)": ("ResembleAI/chatterbox", "_multilingual"),
}

SIM_BACKEND = "wavlm_seed_tts"  # same default as open_results_pr.py; WER does not depend on it


def resolve(row_name):
    return ROW_OVERRIDES.get(row_name, (row_name, ""))


def bucket_folders(bucket, hf_token):
    return {
        entry.path.rstrip("/")
        for entry in HfApi(token=hf_token).list_bucket_tree(bucket, recursive=False)
        if not hasattr(entry, "xet_hash")
    }


def as_float(raw):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def rescore_row(header, old_row, values, rtfx_tol, force):
    """New row with only the WER cells (and avg WER) replaced. Returns (row, changes, skipped)."""
    columns = {column: i for i, column in enumerate(header)}
    padded = old_row + [""] * (len(header) - len(old_row))
    new_row = list(padded)
    skipped = []
    for language in header_languages(header):
        wer_column = f"{language} WER"
        if wer_column not in values or wer_column not in columns:
            continue
        if not padded[columns[wer_column]].strip():
            continue  # never published (e.g. left out on purpose): rescoring must not add it
        old_rtfx = as_float(padded[columns[f"{language} RTFx"]]) if f"{language} RTFx" in columns else None
        new_rtfx = as_float(values.get(f"{language} RTFx"))
        if not force and old_rtfx is not None and new_rtfx is not None and abs(old_rtfx - new_rtfx) > rtfx_tol:
            skipped.append(f"{language} (RTFx {old_rtfx:g} published vs {new_rtfx:g} rescored)")
            continue
        new_row[columns[wer_column]] = values[wer_column]

    if "avg WER" in columns:
        # recompute_averages fills every avg column; keep only avg WER, the others are unchanged.
        averaged = list(new_row)
        recompute_averages(header, averaged, header_languages(header))
        new_row[columns["avg WER"]] = averaged[columns["avg WER"]]

    changes = [
        (column, old, new)
        for column, old, new in zip(header, padded, new_row)
        if old != new and as_float(old) != as_float(new)
    ]
    return new_row, changes, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Rescore every published leaderboard row after a normalizer change.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[1],
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(TARGETS),
        choices=list(TARGETS),
        help="Benchmark files to rescore (default: all; streaming.csv is never rescored).",
    )
    parser.add_argument("--models", nargs="+", default=None, help="Only these published row names.")
    parser.add_argument("--bucket", default=RESULTS_BUCKET, help="Override the source bucket.")
    parser.add_argument("--local_dir", default=None, help="Where to sync results (default: <repo>/results).")
    parser.add_argument("--skip_sync", action="store_true", help="Score already-downloaded results.")
    parser.add_argument("--clean", action="store_true", help="Delete each local model folder before syncing.")
    parser.add_argument(
        "--rtfx_tol",
        type=float,
        default=0.01,
        help="Max |published - rescored| RTFx before a language is treated as scored from other "
             "manifests and skipped (default: 0.01, the published rounding).",
    )
    parser.add_argument("--force", action="store_true", help="Update WER even when RTFx does not match.")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"), help="Defaults to $HF_TOKEN.")
    parser.add_argument("--out_dir", default=None, help="Also write the updated CSVs to this directory.")
    parser.add_argument("--open_pr", action="store_true", help="Actually open the PR (default: dry run).")
    parser.add_argument("--commit_message", default="Rescore WER after normalizer change", help="PR title.")
    args = parser.parse_args()

    local_dir = args.local_dir or os.path.join(REPO_ROOT, "results")
    every_file = {key: fetch_csv(target.filename, args.hf_token) for key, target in TARGETS.items()}
    published = {key: every_file[key] for key in args.targets}

    # Only rows published on every benchmark (Seed-TTS and CV3, plain or voice-clone file), so
    # partial entries such as Seed-TTS-only models are left alone.
    names_per_benchmark = {}
    for key, csv_file in every_file.items():
        benchmark = TARGETS[key].marker
        names_per_benchmark.setdefault(benchmark, set()).update(row[0].strip() for row in csv_file.rows)
    on_all_benchmarks = set.intersection(*names_per_benchmark.values())

    # Every (model id, variant) that some selected file publishes, and the files it appears in.
    jobs = {}
    for key, csv_file in published.items():
        seen = set()
        for row in csv_file.rows:
            name = row[0].strip()
            if args.models and name not in args.models:
                continue
            if name not in on_all_benchmarks:
                continue
            if name in seen:
                print(f"WARNING: {TARGETS[key].filename} lists '{name}' more than once; all copies are updated.")
            seen.add(name)
            jobs.setdefault(resolve(name), {}).setdefault(key, set()).add(name)

    folders = None if args.skip_sync else bucket_folders(args.bucket, args.hf_token)
    synced, skipped_models, report = set(), [], []
    for (model_id, variant), keys in sorted(jobs.items()):
        model_safe = model_id.replace("/", "-")
        model_dir = os.path.join(local_dir, model_safe)
        label = f"{model_id}{f' [{variant}]' if variant else ''}"
        if not args.skip_sync and model_safe not in synced:
            if model_safe not in folders:
                skipped_models.append(f"{label}: no '{model_safe}' folder in {args.bucket}")
                continue
            try:
                sync_bucket(args.bucket, model_safe, local_dir, args.hf_token, args.clean)
            except subprocess.CalledProcessError as exc:
                skipped_models.append(f"{label}: sync failed (exit {exc.returncode})")
                continue
            synced.add(model_safe)
        if not os.path.isdir(model_dir):
            skipped_models.append(f"{label}: no local results in {model_dir}")
            continue

        print(f"\n=== {label}")
        selection, by_language = {}, {}
        for key in keys:
            records, _ = select_manifests(model_dir, model_safe, TARGETS[key], variant)
            if records:
                selection[key] = records
                for stem, language in records.items():
                    by_language.setdefault(language, set()).add(stem)
        if not by_language:
            skipped_models.append(f"{label}: no matching manifests for {', '.join(sorted(keys))}")
            continue
        scored = run_scoring(model_dir, model_id, by_language, SIM_BACKEND)

        for key, records in selection.items():
            csv_file = published[key]
            header = csv_file.header
            values, _ = collect_values(TARGETS[key], header_languages(header), records, scored, {})
            for index, row in enumerate(csv_file.rows):
                if row[0].strip() not in keys[key]:
                    continue
                new_row, changes, skipped = rescore_row(header, row, values, args.rtfx_tol, args.force)
                report.append((TARGETS[key].filename, row[0].strip(), changes, skipped))
                if changes:
                    # Splicing keeps every row's position, so the indices stay valid after re-parsing.
                    csv_file = Published(splice_row(csv_file, index, new_row))
            published[key] = csv_file

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80 + "\nChanges\n" + "=" * 80)
    changed_files = set()
    for filename, row_name, changes, skipped in report:
        if not changes and not skipped:
            continue
        print(f"\n{filename} | {row_name}")
        for column, old, new in changes:
            print(f"  {column:<12} {old or '(blank)':>8} -> {new or '(blank)'}")
        for note in skipped:
            print(f"  SKIPPED {note}")
        if changes:
            changed_files.add(filename)
    unchanged = sum(1 for _, _, changes, skipped in report if not changes and not skipped)
    print(f"\n{unchanged} row(s) unchanged.")
    if skipped_models:
        print("\nNot rescored:")
        for note in skipped_models:
            print(f"  {note}")

    updated = {
        TARGETS[key].filename: csv_file.text
        for key, csv_file in published.items()
        if TARGETS[key].filename in changed_files
    }
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for filename, content in updated.items():
            path = os.path.join(args.out_dir, filename)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
            print(f"Wrote {path}")

    if not updated:
        print("\nNothing to submit.")
        return
    if not args.open_pr:
        print(f"\nDry run - not opening a PR. {len(updated)} file(s) would change: "
              f"{', '.join(sorted(updated))}.\nRe-run with --open_pr to submit.")
        return

    commit = HfApi(token=args.hf_token).create_commit(
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=filename, path_or_fileobj=content.encode("utf-8"))
            for filename, content in sorted(updated.items())
        ],
        commit_message=args.commit_message,
        commit_description=(
            f"WER rescored from `{args.bucket}` with `scripts/rescore_all.py`.\n\n"
            f"Files: {', '.join(sorted(updated))}"
        ),
        create_pr=True,
    )
    print(f"\nPR opened: {commit.pr_url}")


if __name__ == "__main__":
    main()
