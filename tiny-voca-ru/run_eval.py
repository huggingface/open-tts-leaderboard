"""
TTS synthesis for the Open TTS Leaderboard (tiny-voca-ru backend, stage 1).

tiny-voca-ru (https://huggingface.co/egnch/tiny-voca-ru) is a 4,534,411-parameter Russian TTS trained
from scratch at 24 kHz with two fixed voices (F female, M male). It is the Russian port of tiny-voca
(https://huggingface.co/n4ze3m/tiny-voca): the same compact acoustic model and source-filter waveform
generator (no neural vocoder), trained from scratch on Russian. CPU-only. Licence apache-2.0 (its
bundled ruaccent stress data is MIT).

Inference uses the model's own pip package ``tiny_voca_ru`` (ONNX Runtime backend, no PyTorch). The
front end is pure-Python: Russian text normalisation, rule-based G2P into a 61-symbol Russian IPA set,
and stress placement. We use the default ``lite`` stress (bundled dictionary + ``accent.onnx``), which
is what the model card's WER/CER numbers are measured with; ``full`` (the 10 MB ``omograph.onnx``) is a
slower homograph refinement not needed here. ``Synthesizer.synthesize`` does normalisation, chunking to
the 512-token budget, sentence joining and peak-limiting internally, so one call handles any text. A
text with nothing to pronounce is scored as a tenth of a second of silence.

Each voice is its own leaderboard row:
  * ``F``  default row (no manifest suffix).
  * ``M``  variant row, manifest suffix ``_m`` (publish with ``--variant m``).

Device: CPU only. ``--threads`` caps onnxruntime intra-op threads (default 1: a fixed count keeps RTFx
independent of the host's core count, matching the tiny-voca/broca convention). The package does not
expose a thread knob, so we cap it by wrapping ``onnxruntime.InferenceSession`` before the model loads.
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, warm_up, wav_rel_path, write_entry,
)

PEAK = 0.99
STRESS = "lite"   # the model card's reported WER/CER are measured with the default lite stress

# Pinned model repo revision (egnch/tiny-voca-ru main on 2026-11-22).
MODEL_REVISION = "2723b0e71fea216001bd4f6caf0866eb06879339"

VOICES = {
    "F": "",      # default row
    "M": "_m",    # variant row (publish with --variant m)
}
DEFAULT_VOICE = "F"


def _cap_onnx_threads(threads):
    """Make every onnxruntime session the package builds use a fixed intra-op thread count.

    ``tiny_voca_ru`` creates ``ort.InferenceSession(path, providers=...)`` with no SessionOptions, so it
    would otherwise default to all host cores and make RTFx depend on the machine. We wrap the
    constructor to inject a SessionOptions with the fixed thread count when the caller gives none.
    """
    import onnxruntime as ort
    original = ort.InferenceSession

    def wrapped(*args, **kwargs):
        if "sess_options" not in kwargs and len(args) < 2:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            kwargs["sess_options"] = opts
        return original(*args, **kwargs)

    ort.InferenceSession = wrapped


def main(args):
    if not args.device.startswith("cpu"):
        raise SystemExit("tiny-voca-ru runs on CPU only; pass --device=cpu.")
    if args.voice not in VOICES:
        raise SystemExit(f"--voice must be one of {list(VOICES)}")
    mode_suffix = VOICES[args.voice]

    _cap_onnx_threads(args.threads)
    from tiny_voca_ru import Synthesizer
    synth = Synthesizer.from_pretrained(args.model_id, backend="onnx", revision=args.revision)
    sample_rate = int(synth.sample_rate)
    print(f"Loaded tiny-voca-ru voice={args.voice} (revision {args.revision}) | onnxruntime, "
          f"{sample_rate} Hz, stress={STRESS}, threads={args.threads}.")

    failures = []

    def synth_one(text):
        """Return (audio float32, elapsed_s). Text with nothing to pronounce is scored as silence."""
        start = time.perf_counter()
        try:
            result = synth.synthesize(text, voice=args.voice, stress=STRESS, peak=PEAK)
            audio = np.asarray(result.audio, dtype=np.float32)
        except (ValueError, RuntimeError) as exc:
            failures.append(str(exc))
            print(f"WARNING: refused ({exc}); writing silence for: {text[:80]!r}")
            audio = np.zeros(sample_rate // 10, dtype=np.float32)
        if audio.size == 0:
            audio = np.zeros(sample_rate // 10, dtype=np.float32)
        return audio, time.perf_counter() - start

    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # No streaming API, so `first` is None and the probe records the whole-utterance time.
    if args.ttfa_probe != 0:
        from ttfa_probe import run_probe, sample_indices
        run_probe(
            lambda job: (synth_one(job["text"])[0], sample_rate, None),
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={"device": args.device, "voice": args.voice, "threads": args.threads, "revision": args.revision},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)
    failures.clear()

    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        audio, elapsed = synth_one(text)
        sf.write(os.path.join(model_dir, rel_path), audio, sample_rate)
        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / sample_rate, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    print(f"Refused texts (scored as silence): {len(failures)} of {len(starts)}")
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="egnch/tiny-voca-ru",
                        help="tiny-voca-ru model repo (apache-2.0).")
    parser.add_argument("--revision", type=str, default=MODEL_REVISION,
                        help="Model repo revision (default: main on 2026-11-22).")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE,
                        help=f"One of {list(VOICES)}. {DEFAULT_VOICE} is the default row.")
    parser.add_argument("--threads", type=int, default=1,
                        help="onnxruntime intra-op threads (default 1 for RTFx independent of host cores).")
    add_common_args(parser)
    # ``--device`` comes from ``add_common_args``; override the default to cpu since tiny-voca-ru is
    # CPU-only (``main`` refuses a GPU device).
    parser.set_defaults(seed=0, device="cpu")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} voice={args.voice} on "
          f"{args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
