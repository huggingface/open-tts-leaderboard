"""
TTS synthesis for the Open TTS Leaderboard (Unose-Voice-v1.0 backend, stage 1).

Unose-Voice-v1.0 (igunosalabs/Unose-Voice-v1.0) is a 60.5 MB neural VITS ONNX model
trained on Nigerian English / West African cadence. This script downloads the ONNX graph
and metadata directly from the Hugging Face Hub, loads it into the PiperVoice runtime,
synthesizes benchmark sentences with its native voice, times generation for RTFx, saves
wavs, and writes the standardized manifest.

Stage 2 (transcription via Qwen3-ASR) and stage 4 (scoring WER + RTFx) run using the shared
leaderboard harness. Fixed voice, so no stage 3 (SIM).
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from piper.voice import PiperVoice
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

DEFAULT_SAMPLING_RATE = 22_050


def main(args):
    set_seed(args.seed)

    onnx_file = args.onnx_file or "Unose-Voice-v1.0.onnx"
    json_file = args.json_file or "Unose-Voice-v1.0.onnx.json"

    print(f"Loading Unose-Voice from Hub ({args.model_id})...")
    onnx_path = hf_hub_download(repo_id=args.model_id, filename=onnx_file)
    json_path = hf_hub_download(repo_id=args.model_id, filename=json_file)

    use_cuda = (args.device or ("cuda" if torch.cuda.is_available() else "cpu")).startswith("cuda")
    voice = PiperVoice.load(onnx_path, config_path=json_path, use_cuda=use_cuda)
    sampling_rate = voice.config.sample_rate or DEFAULT_SAMPLING_RATE
    print(f"Loaded Unose-Voice ({args.model_id}, sample_rate={sampling_rate}, cuda={use_cuda})")

    def synth_one(text):
        """Synthesize one text row. Returns (audio_1d_numpy, elapsed_seconds)."""
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        segments = []
        for chunk in voice.synthesize(text):
            segments.append(chunk.audio_float_array)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if not segments:
            return np.zeros(1, dtype=np.float32), elapsed
        return np.concatenate(segments), elapsed

    def synth_one_streaming(text, early_stop=False):
        """Streaming synthesis helper for the TTFA probe."""
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        first_audio_time = None
        segments = []
        n_chunks = 0

        for chunk in voice.synthesize(text):
            now = time.perf_counter()
            if first_audio_time is None:
                first_audio_time = now - start
                if early_stop:
                    return np.zeros(1, dtype=np.float32), first_audio_time, 1
            segments.append(chunk.audio_float_array)
            n_chunks += 1

        if not segments:
            return np.zeros(1, dtype=np.float32), first_audio_time or 0.0, 0
        return np.concatenate(segments), first_audio_time, n_chunks

    # Setup paths
    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing JSON sidecar ─────────
    if args.ttfa_probe != 0:
        from ttfa_probe import run_probe, sample_indices
        segment_counts = []

        def _ttfa_gen(job, early_stop):
            audio, first, n_chunks = synth_one_streaming(job["text"], early_stop)
            segment_counts.append(n_chunks)
            return audio, sampling_rate, first

        run_probe(
            _ttfa_gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            extra={"device": args.device, "segments_per_utterance": segment_counts},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)
    set_seed(args.seed)

    # ── Main generation loop ─────────────
    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]

        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        full_path = os.path.join(model_dir, rel_path)

        audio, elapsed = synth_one(text)
        sf.write(full_path, audio, sampling_rate)

        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / sampling_rate, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    print_next_steps(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unose-Voice TTS generation")
    add_common_args(parser)
    parser.add_argument("--model_id", type=str, default="igunosalabs/Unose-Voice-v1.0")
    parser.add_argument("--onnx_file", type=str, default="Unose-Voice-v1.0.onnx")
    parser.add_argument("--json_file", type=str, default="Unose-Voice-v1.0.onnx.json")
    main(parser.parse_args())
