"""
TTS synthesis for the Open TTS Leaderboard (Kokoro-Vietnamese backend, stage 1).

Kokoro-Vietnamese (contextboxai/Kokoro-Vietnamese, Apache-2.0) is a Vietnamese fine-tune of the
Kokoro-82M StyleTTS2 architecture, loaded in-process via the `kokoro_vietnamese` package
(https://github.com/iamdinhthuan/Kokoro-Vietnamese). Fixed built-in voice, so no stage 3 (SIM).
Synthesis is serial (no batched API), like the `kokoro/` backend.

Output is 24 kHz. Vietnamese G2P is `vig2p`, which does not declare a license (see
https://github.com/iamdinhthuan/Kokoro-Vietnamese/issues/5).
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from kokoro_vietnamese import KokoroVietnamese

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

SAMPLING_RATE = 24_000


def main(args):
    set_seed(args.seed)

    tts = KokoroVietnamese(device=args.device, voice=args.voice)
    print(f"Loaded Kokoro-Vietnamese (voice={args.voice}, device={args.device})")
    print_model_size(getattr(tts, "model", None))

    is_cuda = args.device.startswith("cuda")

    def synth_one(text):
        """Synthesize one text. Return (audio_1d_numpy, elapsed_s)."""
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        audio, _phonemes = tts.synthesize(text)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            audio = np.zeros(1, dtype=np.float32)
        return audio, elapsed

    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    if args.ttfa_probe != 0:
        raise SystemExit("--ttfa_probe is not implemented for this backend (no streaming API).")

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)
    set_seed(args.seed)  # warm-up must not shift the RNG stream of the timed run

    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]

        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        path = os.path.join(model_dir, rel_path)

        audio, elapsed = synth_one(text)
        sf.write(path, audio, SAMPLING_RATE)

        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLING_RATE, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="contextboxai/Kokoro-Vietnamese",
                        help="Model repo id (recorded in the manifest folder name; weights are fetched by the package).")
    parser.add_argument("--voice", type=str, default="diem_trinh",
                        help="Built-in voice (diem_trinh, hung_thinh, mai_linh, mai_loan, manh_dung, ...).")
    add_common_args(parser)
    parser.set_defaults(split="vi")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
