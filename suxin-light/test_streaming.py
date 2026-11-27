#!/usr/bin/env python3
"""Local probe: suxin-light first-audio latency vs full utterance (no HF Jobs).

Loads the pinned ONNX from --checkpoint_path (default: the local snapshot used in development)
and checks that the TTFA path emits the first sentence/clause before generation finishes, while
the package's synthesize_long / first-chunk-hold path does not.

  PYTHONPATH=scripts python3 suxin-light/test_streaming.py \
      --checkpoint_path /tmp/suxin_light_models/TEQUILA123-suxin-light
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent
REPO_ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_eval import synth_streaming  # noqa: E402
from ttfa_probe import STREAMING_MAX_FRAC  # noqa: E402

ONE = "Please bring your ID card to the registration desk."
TWO = (
    "Please bring your ID card to the registration desk. "
    "Then wait in the lobby until your name is called."
)
THREE = (
    "Hello there. How are you today? "
    "I would like to register for the afternoon session."
)


def _load(checkpoint, graph):
    sys.path.insert(0, checkpoint)
    from suxin_tts import SuxinTTS  # noqa: E402

    return SuxinTTS.from_pretrained(
        checkpoint,
        model_file=graph,
        intra_op_num_threads=1,
        inter_op_num_threads=1,
        noise_scale=0.0,
        noise_w=0.0,
        pad_mode="group",
        use_lang_tags=False,
        normalize_audio=False,
    )


def _hold_stream(tts, text):
    """Package synthesize_long stitch: first piece is held until the second exists."""
    silence = np.zeros(int(tts.sample_rate * tts._INTER_SENTENCE_SILENCE), dtype=np.float32)
    start = time.perf_counter()
    first = None
    prev = None
    n = 0
    chunks = []
    for sent in tts.split_sentences(text):
        audio = tts.synthesize(sent)
        if audio is None or len(audio) == 0:
            continue
        n += 1
        merged, prev = tts._stitch_stream_chunk(prev, audio, silence)
        if merged is not None:
            if first is None:
                first = time.perf_counter()
            chunks.append(merged)
    if prev is not None:
        if first is None:
            first = time.perf_counter()
        chunks.append(prev)
    end = time.perf_counter()
    return first, start, end, n


def _time_emit(tts, text, early_stop=False):
    start = time.perf_counter()
    audio, first, n = synth_streaming(tts, text, early_stop=early_stop)
    end = time.perf_counter()
    return first, start, end, n, audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        default=os.environ.get(
            "SUXIN_CHECKPOINT", "/tmp/suxin_light_models/TEQUILA123-suxin-light"
        ),
    )
    parser.add_argument("--graph", default="zh_en-suxin-light.int8-excldec.onnx")
    args = parser.parse_args()
    if not Path(args.checkpoint_path, args.graph).is_file():
        raise SystemExit(f"missing {args.checkpoint_path}/{args.graph}")

    tts = _load(args.checkpoint_path, args.graph)
    # Untimed warm-up (same rule as the eval): first ONNX shapes are not the TTFA clock.
    synth_streaming(tts, TWO)

    print(f"STREAMING_MAX_FRAC={STREAMING_MAX_FRAC}")
    print(f"split TWO -> {tts.split_sentences(TWO)}")
    print(f"split ONE -> {tts.split_sentences(ONE)}")

    failures = []

    fh, sh, eh, nh = _hold_stream(tts, TWO)
    hold_frac = (fh - sh) / (eh - sh)
    print(f"hold TWO: n={nh} first/gen={hold_frac:.3f} (package stitch; should be ~1.0)")
    if nh != 2:
        failures.append(f"hold TWO expected 2 pieces, got {nh}")
    if hold_frac < STREAMING_MAX_FRAC:
        failures.append(
            f"hold TWO first/gen={hold_frac:.3f} < {STREAMING_MAX_FRAC}: "
            "the crossfade hold should look like whole-utterance"
        )

    fe, se, ee, ne, audio = _time_emit(tts, TWO)
    emit_frac = (fe - se) / (ee - se)
    first_ms = (fe - se) * 1000
    gen_ms = (ee - se) * 1000
    print(
        f"emit TWO: n={ne} first={first_ms:.1f}ms gen={gen_ms:.1f}ms "
        f"first/gen={emit_frac:.3f} audio={len(audio)/tts.sample_rate:.2f}s"
    )
    if ne != 2:
        failures.append(f"emit TWO expected 2 pieces, got {ne}")
    if emit_frac >= STREAMING_MAX_FRAC:
        failures.append(
            f"emit TWO first/gen={emit_frac:.3f} >= {STREAMING_MAX_FRAC}: "
            "first audio did not arrive before generation ended"
        )
    if first_ms >= gen_ms * 0.9:
        failures.append(f"emit TWO first_ms={first_ms:.1f} not substantially before gen_ms={gen_ms:.1f}")

    fe1, se1, ee1, n1, _ = _time_emit(tts, ONE)
    one_frac = (fe1 - se1) / (ee1 - se1)
    print(f"emit ONE: n={n1} first/gen={one_frac:.3f} (honest: nothing to stream)")
    if n1 != 1:
        failures.append(f"emit ONE expected 1 piece, got {n1}")
    if one_frac < 0.95:
        failures.append(f"emit ONE first/gen={one_frac:.3f}: a single piece must be whole-utterance")

    # three sentences: first should be even earlier
    fe3, se3, ee3, n3, _ = _time_emit(tts, THREE)
    three_frac = (fe3 - se3) / (ee3 - se3)
    print(f"emit THREE: n={n3} first/gen={three_frac:.3f}")
    if n3 != 3:
        failures.append(f"emit THREE expected 3 pieces, got {n3}")
    if three_frac >= STREAMING_MAX_FRAC:
        failures.append(f"emit THREE first/gen={three_frac:.3f} >= {STREAMING_MAX_FRAC}")

    # early_stop: generation should stop near first audio
    fe_e, se_e, ee_e, n_e, _ = _time_emit(tts, THREE, early_stop=True)
    early_frac = (fe_e - se_e) / (ee_e - se_e) if (ee_e - se_e) else 1.0
    print(f"early THREE: n={n_e} first/gen={early_frac:.3f} gen={(ee_e-se_e)*1000:.1f}ms")
    if n_e != 1:
        failures.append(f"early_stop should yield 1 piece, got {n_e}")

    if failures:
        print("FAIL:")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print("OK: first audio arrives before generation ends on multi-sentence text; "
          "quality path (synthesize_long hold) is unchanged and not used for TTFA.")


if __name__ == "__main__":
    main()
