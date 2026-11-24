"""
TTS synthesis for the Open TTS Leaderboard (nano-edge-tts-standalone backend, stage 1).

A 3.27M-parameter English model distilled from Kokoro-82M, using Kokoro's front end and voices
(https://huggingface.co/drmhse/nano-edge-tts-standalone). One sample at a time, on the CPU with one thread, so RTFx
does not depend on the host's core count. Fixed voice, so no stage 3 (SIM).
"""

import argparse
import importlib.util
import os
import time

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

SAMPLING_RATE = 24_000
MODEL_FILES = ("nano_tts.py", "config.json", "nano-tts.safetensors")


def load_model(model_id):
    paths = [hf_hub_download(model_id, f) for f in MODEL_FILES]
    spec = importlib.util.spec_from_file_location("nano_tts", paths[0])
    nano_tts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nano_tts)
    return nano_tts.load(folder=os.path.dirname(paths[0]), device="cpu")


def main(args):
    if args.device != "cpu":
        raise ValueError("nano-edge-tts is evaluated on the CPU (--device=cpu).")
    torch.set_num_threads(args.threads)
    if torch.get_num_threads() != args.threads:
        raise RuntimeError(f"could not set {args.threads} thread(s); torch reports {torch.get_num_threads()}")
    set_seed(args.seed)

    tts = load_model(args.model_id)
    print(f"Loaded {args.model_id} (voice={args.voice}, threads={torch.get_num_threads()})")
    print_model_size(tts)
    lang = args.voice[0]  # a = American, b = British English
    tts.voice(args.voice)  # load the voice and front end before anything is timed
    tts.phonemize("Warm up.", lang)

    def synth_one(text):
        start = time.perf_counter()
        audio = tts(text, voice=args.voice, speed=args.speed)
        return audio, time.perf_counter() - start

    def synth_streaming(text, early_stop=False):
        """TTFA probe: first audio is the front end's first chunk."""
        pack, first, parts = tts.voice(args.voice), None, []
        for ids, tags in tts.phonemize(text, lang):
            parts.append(tts.synth(ids, tags, pack[min(len(ids) - 3, len(pack) - 1)], args.speed))
            if first is None:
                first = time.perf_counter()
                if early_stop:
                    break
        audio = torch.cat(parts).numpy() if parts else torch.zeros(1).numpy()
        return audio, first, len(parts)

    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path
    dataset = load_tts_dataset(args)

    if args.ttfa_probe != 0:
        from ttfa_probe import run_probe, sample_indices
        chunks = []

        def _ttfa_gen(job, early_stop):
            audio, first, n = synth_streaming(job["text"], early_stop)
            chunks.append(n)
            return audio, SAMPLING_RATE, first

        run_probe(
            _ttfa_gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            extra={"device": "cpu", "threads": args.threads, "chunks_per_utterance": chunks},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)
    set_seed(args.seed)

    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        audio, elapsed = synth_one(text)
        sf.write(os.path.join(model_dir, rel_path), audio, SAMPLING_RATE)
        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLING_RATE, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="drmhse/nano-edge-tts-standalone")
    parser.add_argument("--voice", type=str, default="af_heart",
                        help="A Kokoro-82M English voice: af_*/am_* (American) or bf_*/bm_* (British).")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier.")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads (1 for the leaderboard).")
    add_common_args(parser)
    args = parser.parse_args()
    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)
    main(args)
