"""
TTS synthesis for the Open TTS Leaderboard (ZONOS3-Mini backend, stage 1).

ZONOS3-Mini (Zyphra/ZONOS3-Mini, Apache-2.0) is a multilingual MoE TTS model (44.1 kHz) driven through the offline
`zonos3_sgl.tts.TTSLLM` API of https://github.com/Zyphra/ZONOS3-Mini, which generates a whole minibatch in one call.
The model clones from a raw reference clip (codec prefix): `--voice_clone` encodes each sample's `prompt_audio`
(timed); by default (`--no-voice_clone`) every sample uses the prefix of `--voice` (a native default voice of the
split's language), encoded once (no SIM).

Per-sample generation time is the minibatch time divided by the minibatch size, as in the other batched backends.
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from zonos3.generation import SAMPLE_RATE  # generate_stream's chunk rate (44.1 kHz)
from zonos3.tts.delay import shear_up
from zonos3_sgl.tts import TTSLLM, TTSSamplingParams

CODEBOOK_SIZE = 4096  # valid codec token ids are [0, 4096) (zonos3.tokenizer.matched_codec.decode)

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

# ASR language -> ZONOS3 language code (text normalization).
LANGUAGES = {"en": "en_us", "zh": "cmn", "fr": "fr_fr", "pt": "pt_br"}

# ASR language -> fixed voice (--no-voice_clone): a native default voice of the model, female where there is one.
# The model continues the reference clip's codec prefix, so a clip in another language derails the speech (the
# English Savannah clip gave 54-144% WER on fr/es/it/ko/zh/ja). There is no Japanese default voice.
VOICE_DIR = "/opt/ZONOS3-Mini/assets/default_voices"
VOICES = {
    "en": "English/Savannah_en-American_F_1.mp3",
    "zh": "Mandarin Chinese/Meilin_zh-energetic_F_1.wav",
    "fr": "French/Gaspard_fr_M_1.mp3",
    "de": "German/Liesel_de_F_1.mp3",
    "es": "Spanish/Ximena_es_F_1.wav",
    "it": "Italian/Ginevra_it_F_1.mp3",
    "ko": "Korean/Seoyeon_ko_F_1.wav",
    "pt": "Portuguese/Leonor_pt_F_1.wav",
}


def main(args):
    set_seed(args.seed)

    # The README's offline settings: FlashAttention, per-request cache, half the GPU for the KV cache (codec fits).
    tts = TTSLLM(model_path=f"hf://{args.model_id}/lm/zonos3_mini",
                 attention_backend="fa", cache_type="naive", memory_ratio=0.5)
    print(f"Loaded ZONOS3-Mini ({args.model_id})")
    print_model_size(tts.engine)
    language = LANGUAGES.get(args.language, args.language)
    if not args.voice_clone and args.voice is None:
        if args.language not in VOICES:
            raise SystemExit(f"No native default voice for '{args.language}': pass --voice or use --voice_clone.")
        args.voice = os.path.join(VOICE_DIR, VOICES[args.language])
    if not args.voice_clone:
        print(f"Fixed voice: {args.voice}")
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
        results = tts.generate(texts, TTSSamplingParams(seed=args.seed), language=language, decode_audio=False,
                               **reference)
        wavs = [decode(r) for r in results]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return wavs, SAMPLE_RATE, elapsed

    def decode(result):
        """Decode one result as TTSLLM.generate does (undo the codebook delay, cut at EOS, codec decode), also when
        the sample never emitted EOS: it then ran to the token cap and undoing the delay leaves pad ids in its last
        frames, which upstream hands to the codec and crashes the whole batch ("Invalid matched codec tokens").
        Here those frames are cut, so a runaway sample is scored (as the long take it is) instead."""
        if not result["audio_tokens"]:
            return np.zeros(1, dtype=np.float32)
        codes = shear_up(torch.tensor(result["audio_tokens"], dtype=torch.int64, device="cuda"), tts.audio_pad_id)
        if result["eos_frame"] is not None:
            codes = codes[: max(0, result["eos_frame"])]
        invalid = ((codes < 0) | (codes >= CODEBOOK_SIZE)).any(dim=1).nonzero()
        if len(invalid):
            print(f"WARNING: no EOS (eos_frame={result['eos_frame']}); cutting {len(codes) - int(invalid[0])} "
                  f"padded frame(s) of {len(codes)}")
            codes = codes[: int(invalid[0])]
        if codes.numel() == 0:
            return np.zeros(1, dtype=np.float32)
        audio = tts._vocoder.decode_all(codes.unsqueeze(0), apply_shear_up=False)
        return audio[0].numpy().astype(np.float32)

    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path
    if args.voice_clone:
        os.makedirs(os.path.join(paths.output_dir, "prompts"), exist_ok=True)

    dataset = load_tts_dataset(args, ("prompt_audio",) if args.voice_clone else ())

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # `TTSLLM.generate_stream` is the server's own streaming path (/v2/generate): one prompt at a time, float32
    # chunks yielded as the codec decodes them. It conditions on a speaker EMBEDDING, not on reference codes, so the
    # probe embeds the fixed voice once (untimed, like the eval's encoded prefix) or, with --voice_clone, each
    # sample's prompt clip inside the timed call (as the eval encodes each prompt inside its timed batch).
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        fixed_embedding = None if args.voice_clone else tts.embed_speaker_file(args.voice)

        def _ttfa_gen(job, early_stop):
            torch.cuda.synchronize()
            embedding = fixed_embedding
            if embedding is None:
                prompt = job["prompt_audio"]
                wav = torch.from_numpy(np.asarray(prompt["array"], dtype=np.float32)).reshape(1, -1)
                embedding = tts.embed_speaker(wav, prompt["sampling_rate"])
            first, chunks = None, []
            stream = tts.generate_stream(job["text"], TTSSamplingParams(seed=args.seed), language=language,
                                         speaker_embedding=embedding)
            for chunk in stream:
                if chunk.size == 0:
                    continue
                if first is None:
                    first = time.perf_counter()
                chunks.append(chunk)
                if early_stop:
                    # Closing the generator runs the request to its end without decoding (the scheduler has
                    # no abort), so the next probe starts on an idle engine.
                    stream.close()
                    break
            audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
            return audio, SAMPLE_RATE, first

        jobs = [{"text": dataset[i][args.text_column],
                 "prompt_audio": dataset[i]["prompt_audio"] if args.voice_clone else None}
                for i in sample_indices(args.ttfa_probe, len(dataset))]
        run_probe(
            _ttfa_gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={"device": args.device, "language": language, "streaming_api": "TTSLLM.generate_stream",
                   "voice": "voice_clone (prompt embedding)" if args.voice_clone else args.voice,
                   "flash_tilelang": os.environ.get("FLA_TILELANG"),
                   "sonic_experts": os.environ.get("ZONOS3_USE_TRAINING_SONIC_EXPERTS")},
        )
        return

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
    parser.add_argument("--voice", type=str, default=None,
                        help="Reference clip of the fixed voice (--no-voice_clone); default: the native voice of "
                             "--language in VOICES.")
    add_common_args(parser, batch_size=32, voice_clone=False)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
