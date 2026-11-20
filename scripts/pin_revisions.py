#!/usr/bin/env python3
"""Maintain model_revisions.json: the Hub commit each model repo is pinned to in the eval jobs.

Every HF Job runs with scripts/hf_pin/sitecustomize.py, which makes a download of a listed model
repo at `main` (or with no revision) use the pinned commit instead (see that file). This script
keeps the list:

  python scripts/pin_revisions.py                  # pin repos not pinned yet, at their current main:
                                                   # every backend's MODEL_CONFIGS, the models in
                                                   # submit_ztts2_jobs.sh, the scorers, EXTRA_REPOS
  python scripts/pin_revisions.py --add org/name   # also pin these (e.g. a repo logged UNPINNED)
  python scripts/pin_revisions.py --check          # list pinned repos whose main has moved, with
                                                   # the new commits (review before bumping!)
  python scripts/pin_revisions.py --bump org/name  # re-pin to the current main after review

A job log line `[hf-pin] UNPINNED <repo>` names a repo a backend downloads that is not listed yet
(e.g. a codec or tokenizer loaded inside a library): add it with --add.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINS_PATH = os.path.join(REPO_ROOT, "model_revisions.json")
COMMENT = ("Hub commit each model repo is pinned to in the eval jobs (scripts/hf_pin/sitecustomize.py). "
           "Maintained with scripts/pin_revisions.py; review a repo's new commits before --bump.")

# Repos loaded by the shared scorer stages or inside backends (not a MODEL_CONFIGS model id).
EXTRA_REPOS = [
    # scorers (transformers/transcribe.py, score_similarity.py)
    "Qwen/Qwen3-ASR-1.7B-hf", "openai/whisper-large-v3",
    "microsoft/wavlm-base-plus-sv", "microsoft/wavlm-large", "bezzam/wavlm_large_finetune_seed_tts_eval",
    # loaded by backends besides their model id
    "microsoft/speecht5_hifigan", "neuphonic/distill-neucodec", "charactr/vocos-mel-24khz",
]


def backend_model_ids():
    """Model ids of every backend's default MODEL_CONFIGS (from a DRY_RUN that submits nothing)."""
    scripts = sorted(f"{d}/submit_jobs.sh" for d in os.listdir(REPO_ROOT)
                     if os.path.isfile(os.path.join(REPO_ROOT, d, "submit_jobs.sh")))
    env = dict(os.environ, DRY_RUN="true", ONLY_DATASETS="__none__")

    def ids(script):
        out = subprocess.run(["bash", script], cwd=REPO_ROOT, env=env, capture_output=True, text=True).stdout
        return re.findall(r"^\s*Evaluating: (\S+)", out, flags=re.M)

    with ThreadPoolExecutor(8) as pool:
        found = [i for lst in pool.map(ids, scripts) for i in lst]
    runs = subprocess.run(["bash", "submit_ztts2_jobs.sh"], cwd=REPO_ROOT, env=dict(os.environ, DRY_RUN="true"),
                          capture_output=True, text=True).stdout
    found += [line.split("|")[1].split()[0] for line in runs.splitlines() if line.count("|") == 3]
    return found


def load_pins():
    if not os.path.exists(PINS_PATH):
        return {}
    with open(PINS_PATH, encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def save_pins(pins):
    with open(PINS_PATH, "w", encoding="utf-8") as f:
        json.dump({"_comment": COMMENT, **dict(sorted(pins.items(), key=lambda kv: kv[0].lower()))}, f, indent=2)
        f.write("\n")


def main_sha(api, repo):
    try:
        return api.model_info(repo).sha
    except RepositoryNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--add", nargs="*", default=[], help="Extra model repos to pin.")
    parser.add_argument("--check", action="store_true", help="Report pinned repos whose main has moved.")
    parser.add_argument("--bump", nargs="*", default=[], help="Re-pin these repos to their current main.")
    args = parser.parse_args()
    api = HfApi()
    pins = load_pins()

    if args.check:
        repos = sorted(pins)
        with ThreadPoolExecutor(16) as pool:
            heads = dict(zip(repos, pool.map(lambda r: main_sha(api, r), repos)))
        moved = [r for r in repos if heads[r] and heads[r] != pins[r]]
        for repo in repos:
            if heads[repo] is None:
                print(f"GONE   {repo} (pinned {pins[repo][:12]}): repo not found or no access")
        for repo in moved:
            print(f"MOVED  {repo}: pinned {pins[repo][:12]}, main {heads[repo][:12]}")
            try:
                for c in api.list_repo_commits(repo):
                    if c.commit_id == pins[repo]:
                        break
                    print(f"         {c.commit_id[:12]} {c.created_at:%Y-%m-%d} {', '.join(c.authors)}: {c.title}")
            except HfHubHTTPError as e:  # e.g. a gated repo: its info is readable, its history is not
                print(f"         (commit history not readable: {type(e).__name__})")
        print(f"{len(repos)} pinned, {len(moved)} moved since pinning")
        return

    for repo in args.bump:
        sha = main_sha(api, repo)
        if sha is None:
            sys.exit(f"{repo}: not found")
        print(f"bump   {repo}: {pins.get(repo, '-')[:12]} -> {sha[:12]}")
        pins[repo] = sha

    wanted = sorted(set(backend_model_ids() + EXTRA_REPOS + args.add) - set(pins), key=str.lower)
    with ThreadPoolExecutor(16) as pool:
        shas = dict(zip(wanted, pool.map(lambda r: main_sha(api, r), wanted)))
    for repo in wanted:
        if shas[repo] is None:
            print(f"skip   {repo}: not a model repo on the Hub (or no access)")
        else:
            pins[repo] = shas[repo]
            print(f"pin    {repo} @ {shas[repo][:12]}")
    save_pins(pins)
    print(f"{len(pins)} repos pinned in {os.path.relpath(PINS_PATH, REPO_ROOT)}")


if __name__ == "__main__":
    main()
