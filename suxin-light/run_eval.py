"""
TTS synthesis for the Open TTS Leaderboard (suxin-light backend, stage 1).

suxin-light (https://huggingface.co/TEQUILA123/suxin-light) is a 28.65 M-parameter Mandarin–English
VITS student distilled from Fun-CosyVoice3-0.5B, with the SuxinVoc decoder, at 22.05 kHz mono. One
fixed voice; no voice cloning. Inference is the INT8-excldec ONNX graph
(``zh_en-suxin-light.int8-excldec.onnx``, ~67 MB) on onnxruntime ``CPUExecutionProvider`` only,
driven by the repo's own ``suxin_tts`` package (``SuxinTTS`` / ``MixedFrontend``).

The front end is mixed ``phoneme_type`` (stock piper-tts cannot load it): Chinese via pypinyin,
English via piper-tts's ``EspeakPhonemizer`` (espeak-ng data, GPL-3.0). PAD is inserted only after
syllable/word ends (``pad_mode=group``). In a Chinese context, Arabic digits are read as Chinese
numerals inside the frontend (``请到3号窗口`` → 三); a purely English number stays on espeak
(``in his 30s``). That is the model's own text handling, not an extra leaderboard normaliser.

Text handling is exactly the CLI default (``suxin-tts`` / ``infer.py`` without ``--no-split``):
``SuxinTTS.synthesize_long`` splits on sentence then clause punctuation, synthesises each piece,
and stitches with a 50 ms gap plus a short crossfade. A text with nothing to pronounce is scored
as a tenth of a second of silence.

Device: CPU only. ``--threads`` caps onnxruntime intra-op threads (default 1: a fixed count keeps
RTFx independent of the host's core count, matching the tiny-voca / tiny-broca convention). The
model card's on-device recommendation of 4 threads is for an 8-core Cortex-A55, not this eval.

Streaming / TTFA. The package exposes sentence/chunk generators (``synthesize_stream``,
``synthesize_stream_from_tokens``) but they are not native intra-model streaming: each ONNX call
still emits a whole clause, and ``synthesize_stream`` withholds the first piece until the next one
arrives so it can crossfade. The eval path is ``synthesize_long`` (one complete wav). ``--ttfa_probe``
therefore uses the whole-utterance clock (``first`` is None), same as tiny-voca / tiny-broca.
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, warm_up, wav_rel_path, write_entry,
)

SAMPLE_RATE = 22050
# Model card: 28.65 M VITS student. The INT8 ONNX session is not a torch nn.Module, so this is
# printed as a constant rather than counted from parameters().
N_PARAMS = 28_650_000
DEFAULT_ONNX = "zh_en-suxin-light.int8-excldec.onnx"

# Pinned model repo revision (TEQUILA123/suxin-light main on 2026-10-13).
MODEL_REVISION = "5ee2da546f3635da263602842d238d8624cb7a81"


def main(args):
    if not args.device.startswith("cpu"):
        raise SystemExit("suxin-light runs on CPU only; pass --device=cpu.")

    from huggingface_hub import snapshot_download
    if args.checkpoint_path:
        repo_dir = args.checkpoint_path
    else:
        repo_dir = snapshot_download(
            args.model_id, revision=args.revision,
            local_dir=os.path.join(args.local_dir, args.model_id.replace("/", "-")),
            allow_patterns=[
                "suxin_tts/**",
                DEFAULT_ONNX,
                DEFAULT_ONNX + ".json",
            ],
        )
    print(f"Checkpoint: {repo_dir} (revision {args.revision})")

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from suxin_tts import SuxinTTS  # noqa: E402

    tts = SuxinTTS.from_pretrained(
        repo_dir,
        model_file=DEFAULT_ONNX,
        intra_op_num_threads=args.threads,
        inter_op_num_threads=1,
        noise_scale=0.0,
        noise_w=0.0,
        pad_mode="group",
        use_lang_tags=False,
        normalize_audio=False,
    )
    sample_rate = int(tts.sample_rate)
    if sample_rate != SAMPLE_RATE:
        print(f"WARNING: frontend sample_rate={sample_rate}, expected {SAMPLE_RATE}")
    print(f"Loaded suxin-light | onnxruntime CPUExecutionProvider, {sample_rate} Hz, "
          f"threads={args.threads} | TTS model size: {N_PARAMS / 1e9:.4f}B parameters "
          f"({N_PARAMS:,}).")

    failures = []

    def synth_one(text):
        """Return (audio float32, elapsed_s). Empty / unspeakable text is scored as silence."""
        start = time.perf_counter()
        try:
            audio = tts.synthesize_long(text)
            if audio is None:
                audio = np.zeros(0, dtype=np.float32)
            else:
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        except (ValueError, RuntimeError) as exc:
            failures.append(str(exc))
            print(f"WARNING: refused ({exc}); writing silence for: {text[:80]!r}")
            audio = np.zeros(0, dtype=np.float32)
        if audio.size == 0:
            audio = np.zeros(sample_rate // 10, dtype=np.float32)
        return audio, time.perf_counter() - start

    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # Sentence/chunk APIs do not yield playable audio before the next clause (crossfade hold), and
    # there is no intra-model stream, so `first` is None and the probe records the whole-utterance
    # time of synthesize_long — the same path as the scored wavs.
    if args.ttfa_probe != 0:
        from ttfa_probe import run_probe, sample_indices
        run_probe(
            lambda job: (synth_one(job["text"])[0], sample_rate, None),
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            extra={"device": args.device, "threads": args.threads, "revision": args.revision,
                   "onnx": DEFAULT_ONNX, "streaming_api": "none (synthesize_long whole-utterance)"},
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
    parser.add_argument("--model_id", type=str, default="TEQUILA123/suxin-light",
                        help="suxin-light model repo (MIT; English G2P via piper-tts/espeak-ng, GPL-3.0).")
    parser.add_argument("--revision", type=str, default=MODEL_REVISION,
                        help="Model repo revision (default: main on 2026-10-13).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local copy of the repo. Default: download --model_id under --local_dir.")
    parser.add_argument("--local_dir", type=str, default="/tmp/suxin_light_models",
                        help="Where to materialise the model repo as real files.")
    parser.add_argument("--threads", type=int, default=1,
                        help="onnxruntime intra-op threads (default 1 for RTFx independent of host cores).")
    add_common_args(parser)
    # ``--device`` comes from ``add_common_args``; override the default to cpu since suxin-light is
    # CPU-only (``main`` refuses a GPU device).
    parser.set_defaults(seed=0, device="cpu")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on "
          f"{args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
