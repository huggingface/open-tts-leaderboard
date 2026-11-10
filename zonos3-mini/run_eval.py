"""
TTS synthesis for the Open TTS Leaderboard (ZONOS3-Mini backend, stage 1).

ZONOS3-Mini (Zyphra/ZONOS3-Mini, Apache-2.0) is a multilingual MoE TTS model (44.1 kHz) driven through the offline
`zonos3_sgl.tts.TTSLLM` API of https://github.com/Zyphra/ZONOS3-Mini, which generates a whole minibatch in one call.
The model clones from a raw reference clip (codec prefix): `--voice_clone` encodes each sample's `prompt_audio`
(timed); by default (`--no-voice_clone`) every sample uses the prefix of `--voice`, encoded once (no SIM).

Per-sample generation time is the minibatch time divided by the minibatch size, as in the other batched backends.
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from zonos3_sgl.tts import TTSLLM, TTSSamplingParams

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

# ASR language -> ZONOS3 language code (text normalization).
LANGUAGES = {"en": "en_us", "zh": "cmn", "fr": "fr_fr", "pt": "pt_br"}


def main(args):
    set_seed(args.seed)

    # The README's offline settings: FlashAttention, per-request cache, half the GPU for the KV cache (codec fits).
    tts = TTSLLM(model_path=f"hf://{args.model_id}/lm/zonos3_mini",
                 attention_backend="fa", cache_type="naive", memory_ratio=0.5)
    print(f"Loaded ZONOS3-Mini ({args.model_id})")
    print_model_size(tts.engine)
    language = LANGUAGES.get(args.language, args.language)
    voice = None
    if not args.voice_clone:
        wav, sr = sf.read(args.voice, dtype="float32", always_2d=True)
        voice = tts.encode_reference(torch.from_numpy(wav.T.copy()), sr)

    def synth_batch(texts, prompt_audios):
        """Synthesize a list of texts. Return (list of 1d float32 arrays, sample rate, elapsed_s)."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        if voice is not None:
            reference = {"speaker_reference_codes": voice}
        else:
            reference = {"speaker_reference_audio": [
                (torch.from_numpy(np.asarray(p["array"], dtype=np.float32)).reshape(1, -1), p["sampling_rate"])
                for p in prompt_audios
            ]}
        results = tts.generate(texts, TTSSamplingParams(seed=args.seed), language=language, **reference)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        wavs = [np.frombuffer(r["audio"], dtype=np.float32) for r in results]
        return [w if w.size else np.zeros(1, dtype=np.float32) for w in wavs], results[0]["sample_rate"], elapsed

    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path
    if args.voice_clone:
        os.makedirs(os.path.join(paths.output_dir, "prompts"), exist_ok=True)

    dataset = load_tts_dataset(args, ("prompt_audio",) if args.voice_clone else ())

    if args.ttfa_probe != 0:
        raise SystemExit("--ttfa_probe is not implemented for this backend.")

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    def run_batch(batch):
        return synth_batch(list(batch[args.text_column]), batch["prompt_audio"] if args.voice_clone else None)

    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(run_batch, dataset, starts, args.batch_size, args.warmup_steps)
    set_seed(args.seed)  # warm-up must not shift the RNG stream of the timed run

    for start in tqdm(starts, desc="Generating"):
        batch = dataset[start : start + args.batch_size]
        texts = list(batch[args.text_column])
        wavs, sampling_rate, elapsed = run_batch(batch)
        per_sample = elapsed / len(texts)
        for i, (sample_id, text, audio) in enumerate(zip(batch["id"], texts, wavs)):
            prompt_rel = None
            if args.voice_clone:
                prompt = batch["prompt_audio"][i]
                prompt_rel = os.path.join(dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
                sf.write(os.path.join(model_dir, prompt_rel), prompt["array"], prompt["sampling_rate"])
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            sf.write(os.path.join(model_dir, rel_path), audio, sampling_rate)
            write_entry(manifest_file, manifest_entry(rel_path, len(audio) / sampling_rate, per_sample, text, prompt_rel))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Zyphra/ZONOS3-Mini")
    parser.add_argument("--language", type=str, default="en", help="ASR language code of the split (e.g. 'en', 'zh').")
    parser.add_argument("--voice", type=str,
                        default="/opt/ZONOS3-Mini/assets/default_voices/English/Savannah_en-American_F_1.mp3",
                        help="Reference clip of the fixed voice (--no-voice_clone).")
    add_common_args(parser, batch_size=32, voice_clone=False)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
