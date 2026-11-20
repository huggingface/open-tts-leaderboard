"""Pin Hugging Face Hub model downloads to reviewed commits (injected into every HF Job).

Python imports `sitecustomize` at startup from the first sys.path entry that has one, and the jobs
put this directory first on PYTHONPATH. Every Hub download (transformers/diffusers `from_pretrained`,
library loaders, llama-cpp, ...) goes through `hf_hub_download` or `snapshot_download`, so wrapping
those two (and the cache lookup `try_to_load_from_cache`) pins every repo listed in model_revisions.json (next to this file, or $HF_PIN_FILE):

  - a download of a listed model repo at `main` (or with no revision) gets the pinned commit;
  - an explicit other revision (a tag or commit) is left as is;
  - a model repo that is not listed is downloaded from `main` and logged as `[hf-pin] UNPINNED`,
    or refused when HF_PIN_STRICT=1.

This protects runs against a silently updated model repo: a new commit on `main` is
not used until it is reviewed and pinned (scripts/pin_revisions.py). Downloads outside the two
functions (raw `resolve/main` URLs, `git clone`, weights baked into a Space image at build time)
are not covered.

The wrap is applied when `huggingface_hub.file_download` / `huggingface_hub._snapshot_download`
are first imported (not here), so this file does not import huggingface_hub at startup and does
not freeze its env-var configuration before the job's own code runs.
"""

import functools
import importlib.util
import inspect
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TAG = "[hf-pin]"


def _load_pins():
    path = os.environ.get("HF_PIN_FILE") or os.path.join(_HERE, "model_revisions.json")
    try:
        with open(path, encoding="utf-8") as f:
            return {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    except FileNotFoundError:
        return None
    except Exception as e:  # a broken pins file must not take down every Python process
        print(f"{_TAG} could not read {path}: {e!r}; downloads are NOT pinned", file=sys.stderr)
        return None


_PINS = _load_pins()
_STRICT = os.environ.get("HF_PIN_STRICT", "") not in ("", "0", "false")
_LOGGED = set()


def _log_once(key, msg):
    if key not in _LOGGED:
        _LOGGED.add(key)
        print(f"{_TAG} {msg}", file=sys.stderr, flush=True)


def _pin_revision(repo_id, repo_type, revision, strict=True):
    if repo_type not in (None, "model") or revision not in (None, "main") or not isinstance(repo_id, str):
        return revision
    if os.path.exists(repo_id):  # a local directory, not a Hub repo
        return revision
    sha = _PINS.get(repo_id)
    if sha:
        _log_once(repo_id, f"{repo_id} pinned to {sha}")
        return sha
    if _STRICT and strict:
        raise RuntimeError(f"{_TAG} {repo_id} is not pinned in model_revisions.json (HF_PIN_STRICT=1); "
                           "add it with scripts/pin_revisions.py")
    _log_once(repo_id, f"UNPINNED {repo_id} (downloading from main)")
    return revision


def _wrap(fn, strict):
    if getattr(fn, "_hf_pinned", False):
        return fn
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            bound = sig.bind_partial(*args, **kwargs)
        except TypeError:
            return fn(*args, **kwargs)  # let the real function raise its own error
        a = bound.arguments
        if "repo_id" in a and "revision" in sig.parameters:
            a["revision"] = _pin_revision(a["repo_id"], a.get("repo_type"), a.get("revision"), strict)
        return fn(*bound.args, **bound.kwargs)

    wrapper._hf_pinned = True
    return wrapper


# module -> [(function, refuse unpinned repos under HF_PIN_STRICT)]. try_to_load_from_cache is the
# cache lookup transformers does after a download: with the same mapping it finds the file stored
# under the pinned commit (a download by commit does not write the cache's refs/main).
_TARGETS = {
    "huggingface_hub.file_download": [("hf_hub_download", True), ("try_to_load_from_cache", False)],
    "huggingface_hub._snapshot_download": [("snapshot_download", True)],
}


def _patch(module_name, module):
    for attr, strict in _TARGETS[module_name]:
        _patch_one(module, attr, strict)


def _patch_one(module, attr, strict):
    original = getattr(module, attr, None)
    if original is None or getattr(original, "_hf_pinned", False):
        return
    wrapped = _wrap(original, strict)
    # Rebind every reference already taken (e.g. the package's own lazy attribute); modules that
    # import the name later get the wrapped one from here.
    for mod in list(sys.modules.values()):
        d = getattr(mod, "__dict__", None)
        if not d:
            continue
        for key, value in list(d.items()):
            if value is original:
                try:
                    setattr(mod, key, wrapped)
                except Exception:
                    pass


class _PinFinder:
    """Meta-path finder that wraps the target modules right after they execute."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            try:
                _patch(fullname, module)
            except Exception as e:
                print(f"{_TAG} could not wrap {fullname}: {e!r}; downloads are NOT pinned", file=sys.stderr)

        loader.exec_module = exec_and_patch
        return spec


if _PINS is not None:
    sys.meta_path.insert(0, _PinFinder())


# This file shadows any sitecustomize the image already has (e.g. Debian's): run that one too.
def _chain_next_sitecustomize():
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == _HERE:
            continue
        candidate = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("_image_sitecustomize", candidate)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"{_TAG} image sitecustomize {candidate} failed: {e!r}", file=sys.stderr)
            return


_chain_next_sitecustomize()
