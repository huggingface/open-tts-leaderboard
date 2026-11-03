# /// script
# requires-python = ">=3.10"
# dependencies = ["komos==0.0.2"]
# ///
# Run: uv run --script transformers/score_komos.py --manifest_paths <Korean manifest> --device cpu

"""Resumable Korean MOS sidecars for the Open TTS Leaderboard.

Run in a separate environment with komos==0.0.2. The generation manifest is
read-only; scores and checkpoint/preprocessing provenance live in KOMOS_*.json.
"""

import argparse
import json
import math
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Final, Protocol, TypedDict

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

MODEL_ID: Final = "upskyy/komos"
MODEL_REVISION: Final = "c558983b9777d4f01aa61003b151dea37db0cc2d"
KOMOS_VERSION: Final = "0.0.2"
MIN_AUDIO_SECONDS: Final = 0.1


class ScoringModel(Protocol):
    """KOMOS capability used by the manifest scorer."""

    def score_long_form(self, audio: NDArray[np.float32], sr: int) -> float:
        """Score a complete mono waveform up to the backend's 60-second limit."""
        ...


class Provenance(TypedDict):
    model_id: str
    revision: str
    komos_version: str
    method: str
    max_audio_seconds: float
    min_audio_seconds: float


class Sidecar(TypedDict):
    komos_model: Provenance
    scores: dict[str, float | None]
    errors: dict[str, str]


@dataclass(frozen=True, slots=True)
class ScoreOptions:
    """Preprocessing, output and resume policy for one scoring run."""

    output_dir: Path | None = None
    max_rows: int = -1
    overwrite: bool = False
    save_every: int = 100
    max_audio_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.save_every <= 0:
            message = "save_every must be positive"
            raise ValueError(message)
        if (
            not math.isfinite(self.max_audio_seconds)
            or not 0 <= self.max_audio_seconds <= 60
        ):
            message = "max_audio_seconds must be between 0 (no clipping) and 60"
            raise ValueError(message)


def score_manifest(
    model: ScoringModel, manifest_path: Path, options: ScoreOptions
) -> Path:
    """Resume per-file scoring without modifying a generation or ASR manifest."""
    with manifest_path.open(encoding="utf-8") as source:
        entries = [json.loads(line) for line in source if line.strip()]
    if options.max_rows > 0:
        entries = entries[: options.max_rows]
    out_path = (
        options.output_dir or manifest_path.parent
    ) / f"KOMOS_{manifest_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    provenance: Provenance = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "komos_version": KOMOS_VERSION,
        "method": "score_long_form",
        "max_audio_seconds": options.max_audio_seconds,
        "min_audio_seconds": MIN_AUDIO_SECONDS,
    }
    scores: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    if out_path.exists() and not options.overwrite:
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        if previous.get("komos_model") != provenance:
            message = f"{out_path}: scoring provenance changed; use --overwrite"
            raise ValueError(message)
        scores = previous["scores"]
        errors = previous.get("errors", {})

    def save() -> None:
        """Atomically replace the sidecar so an interrupted save remains resumable."""
        data: Sidecar = {"komos_model": provenance, "scores": scores, "errors": errors}
        temporary = out_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
        temporary.replace(out_path)

    todo = dict.fromkeys(
        e["audio_filepath"]
        for e in entries
        if e.get("audio_filepath") and e["audio_filepath"] not in scores
    )
    print(f"{manifest_path}: {len(todo)}/{len(entries)} rows to score -> {out_path}")
    for n, rel in enumerate(todo, 1):
        try:
            with sf.SoundFile(manifest_path.parent / rel) as audio:
                sr = audio.samplerate
                frames = (
                    int(options.max_audio_seconds * sr)
                    if options.max_audio_seconds
                    else -1
                )
                speech = audio.read(
                    frames=frames, dtype="float32", always_2d=True
                ).mean(axis=1)
        except (OSError, sf.LibsndfileError) as exc:
            errors[rel] = str(exc)  # Missing/unreadable audio is retried on resume.
            print(f"  WARNING: could not load {rel}: {exc}")
        else:
            if len(speech) < MIN_AUDIO_SECONDS * sr:
                scores[rel] = None
                errors[rel] = "Audio shorter than 0.1 seconds"
            else:
                # Do not hide inference/device failures or silently truncate >60s with cap=0.
                score = float(model.score_long_form(speech, sr=sr))
                if not math.isfinite(score) or not 1 <= score <= 5:
                    save()
                    message = f"Invalid KOMOS score for {rel}: {score}"
                    raise ValueError(message)
                scores[rel] = round(score, 6)
                errors.pop(rel, None)
        if n % options.save_every == 0:
            save()
    save()
    valid = [
        scores[e["audio_filepath"]]
        for e in entries
        if scores.get(e.get("audio_filepath")) is not None
    ]
    if valid:
        print(
            f"  KOMOS = {sum(valid) / len(valid):.3f} over {len(valid)}/{len(entries)} rows"
        )
    return out_path


def get_parser() -> argparse.ArgumentParser:
    """Build the scorer CLI, following score_utmos.py's manifest/output flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=30.0,
        help="Evaluate this prefix (<=60s); 0 scores full files up to 60s.",
    )
    parser.add_argument(
        "--device", default="cpu", help="PyTorch device (default: cpu)."
    )
    return parser


def main() -> None:
    """Load one pinned checkpoint and score every requested Korean manifest."""
    args = get_parser().parse_args()
    options = ScoreOptions(
        args.output_dir,
        args.max_rows,
        args.overwrite,
        args.save_every,
        args.max_audio_seconds,
    )
    if version("komos") != KOMOS_VERSION:
        message = f"Install komos=={KOMOS_VERSION} to reproduce this scoring protocol"
        raise RuntimeError(message)
    from komos import KOMOS

    model = (
        KOMOS.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(args.device).eval()
    )
    for manifest_path in args.manifest_paths:
        score_manifest(model, manifest_path, options)


if __name__ == "__main__":
    main()
