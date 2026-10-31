"""
TTS synthesis for the Open TTS Leaderboard (SteerTTS edge backend, stage 1).

SteerTTS edge is the 12 MB on-device build of SteerTTS: int4 acoustic model (9.25M parameters) + int4 vocoder
(6.22M), 15.5M parameters in total, loaded in-process from the `steertts_edge` package (PyTorch runtime, `EdgeTTS`).
English only, so only the `en` splits. Default voice = the release's own default voice (`voice_1688` in v2.0.0; no
test-set voice selection); no voice cloning. Rows are synthesized one at a time (no batched API); generation is timed
after `--warmup_steps` untimed samples.

    PYTHONPATH=scripts python steertts-edge/run_eval.py --model_id WhissleAI/steer-tts-edge --revision v2.0.0 \
        --dataset_path YOUR_USERNAME/seed_tts_eval \
        --dataset tts --split en --device cuda:0 [--model_path /local/steer-tts-edge]
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from steertts_edge import EdgeTTS
from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

SAMPLING_RATE = 24_000


def main(args):
    set_seed(args.seed)
    if args.model_path:
        tts = EdgeTTS(args.model_path, device=args.device)
    else:
        tts = EdgeTTS.from_pretrained(args.model_id, revision=args.revision, device=args.device)
    voice = args.voice or tts.default_voice
    print(f"Loaded {tts} voice={voice}")
    print_model_size(tts.model, tts.vocoder)
    is_cuda = str(args.device).startswith("cuda")

    paths = output_paths(args, "")
    model_dir, dataset_dir_name = paths.model_dir, paths.dataset_dir_name

    def synth_one(text):
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        audio = np.asarray(tts(text, voice=voice), dtype=np.float32)
        if is_cuda:
            torch.cuda.synchronize()
        return audio, time.perf_counter() - start

    dataset = load_tts_dataset(args)
    done_entries = load_done_entries(paths.manifest_path, args.resume)
    manifest_file = open_manifest(paths.manifest_path, done_entries)
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)

    def gen_batch(batch):
        return [synth_one(text) for text in batch[args.text_column]]

    warm_up(gen_batch, dataset, starts, 1, args.warmup_steps)
    set_seed(args.seed)
    for i in tqdm(starts, desc="Generating"):
        row = dataset[i]
        audio, elapsed = synth_one(row[args.text_column])
        rel_path = wav_rel_path(dataset_dir_name, row["id"])
        sf.write(os.path.join(model_dir, rel_path), audio, SAMPLING_RATE)
        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLING_RATE, elapsed, row[args.text_column], None))
        manifest_file.flush()
    manifest_file.close()
    print("Results saved at path:", os.path.abspath(paths.manifest_path))
    print_next_steps(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="WhissleAI/steer-tts-edge",
                        help="Name used for the results folder; also the HF repo loaded unless --model_path is set.")
    parser.add_argument("--revision", type=str, default=None, help="HF revision (tag / commit) of --model_id.")
    parser.add_argument("--model_path", type=str, default=None, help="Local package directory (overrides --model_id).")
    parser.add_argument("--voice", type=str, default=None, help="Built-in voice (default: the package's default_voice).")
    add_common_args(parser, voice_clone=False)
    args = parser.parse_args()
    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)
    main(args)
