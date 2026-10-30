"""
TTS synthesis for the Open TTS Leaderboard (LFM3-Audio backend, stage 1).

LFM3-Audio-2B (LiquidAI/LFM3-Audio-2B) is Liquid AI's end-to-end speech + text model: an LFM3
backbone predicts 12 LiquidCodec codes per 80 ms speech frame, which the LiquidCodec tokenizer
(`processor.audio_tokenizer`) turns into 24 kHz audio. It is loaded through Transformers with the
checkpoint's own code (`trust_remote_code=True`). TTS is selected by the system prompt
"Perform TTS in <language> using a <gender> voice.", with the assistant turn prefilled with
`<|audio_start|>`. There is no voice cloning: the voice is one of two built-in ones (female/male).

`model.generate` takes a batch of left-padded prompts and returns every row's speech codes
(`audio_codes`, padded with -1, cut at `audio_lengths`), so texts are synthesized in minibatches.
Each row's codes are decoded separately inside the timed region (generation + decoding = RTFx).

Model card (usage): https://huggingface.co/LiquidAI/LFM3-Audio-2B

Stage 2 (`transformers/transcribe.py`, run in the shared `bezzam/evals` image) transcribes the
wavs with Qwen3-ASR to fill `pred_text`, and stage 4 (`normalizer.eval_utils.score_results`, run
locally) computes WER + RTFx from the completed manifest. Fixed voice, so no stage 3 (SIM).
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor
from transformers.generation.streamers import BaseStreamer

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

# LiquidCodec decodes every 80 ms frame to 1920 samples at 24 kHz.
SAMPLING_RATE = 24_000

# Dataset split → the language (or locale) named in the TTS system prompt (model card, "Speech
# languages / locales"). Splits not listed here are not supported by the model.
LANGUAGES = {
    "en": "American English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "el": "Greek",
    "fi": "Finnish",
    "tr": "Turkish",
    "ar": "Arabic",
    "da": "Danish",
    "he": "Hebrew",
    "sv": "Swedish",
}


class _StopAfterFirstAudio(Exception):
    """Raised from the TTFA streamer to abort generation once the first audio is in hand."""


class FirstAudioStreamer(BaseStreamer):
    """TTFA probe only: decodes the first `n_frames` speech frames the moment they are sampled.

    `model.generate` hands every speech frame, `(num_codebooks,)`, to `put` synchronously, and the
    codec decodes chunk by chunk through a `LiquidCodecStreamingCache`, so the first chunk's audio
    is exactly what a streaming client could start playing.
    """

    def __init__(self, codec, n_frames, early_stop):
        self.codec, self.n_frames, self.early_stop = codec, n_frames, early_stop
        self.frames, self.first, self.chunk = [], None, None

    def put(self, value):
        self.frames.append(value)
        if self.first is None and len(self.frames) == self.n_frames:
            # The cache class lives in the codec's remote-code module (modeling_liquid_codec.py).
            cache = sys.modules[type(self.codec).__module__].LiquidCodecStreamingCache(self.codec.config)
            codes = torch.stack(self.frames, -1).unsqueeze(0)  # (1, num_codebooks, n_frames)
            self.chunk = self.codec.decode(codes, streaming_cache=cache, finalize=False).audio_values[0]
            self.chunk = self.chunk.float().cpu().numpy()  # .cpu() syncs: the audio is in hand
            self.first = time.perf_counter()
            if self.early_stop:
                raise _StopAfterFirstAudio

    def end(self):
        pass


def main(args):
    set_seed(args.seed)

    if args.split not in LANGUAGES:
        raise ValueError(f"LFM3-Audio has no TTS language for split '{args.split}'.")
    language = args.language or LANGUAGES[args.split]
    system_prompt = f"Perform TTS in {language} using a {args.voice} voice."

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id, dtype=dtype, device_map=args.device, trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"  # generate() expects left-padded prompts
    codec = processor.audio_tokenizer.to(device=args.device).eval()
    print(f"Loaded LFM3-Audio ({args.model_id}): system prompt {system_prompt!r}")
    print_model_size(model, codec)

    sampling = dict(
        max_new_tokens=args.max_new_tokens,
        audio_temperature=args.audio_temperature,
        audio_top_k=args.audio_top_k,
        audio_repetition_penalty=args.audio_repetition_penalty,
        audio_repetition_window=args.audio_repetition_window,
        return_dict_in_generate=True,
    )
    truncated = []  # rows that hit --max_new_tokens before the end-of-audio frame

    def prompts(texts):
        """Left-padded TTS prompts for a batch of texts, ending on `<|audio_start|>`."""
        conversations = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
                {"role": "assistant", "content": "<|audio_start|>"},
            ]
            for text in texts
        ]
        return processor.apply_chat_template(
            conversations, continue_final_message=True, tokenize=True, return_dict=True,
            return_tensors="pt", processor_kwargs={"padding": True},
        ).to(args.device)

    def decode(codes):
        """(num_codebooks, frames) codes of one row → 1-D float32 numpy waveform."""
        if not codes.shape[-1]:
            return np.zeros(1, dtype=np.float32)
        audio = codec.decode(codes.unsqueeze(0)).audio_values[0]
        return audio.float().cpu().numpy()

    @torch.inference_mode()
    def generate_tts(texts):
        """Synthesize a batch of texts; time generation + decoding for RTFx.

        Returns (list of 1-D float32 audio arrays, per-sample time in seconds).
        """
        inputs = prompts(texts)  # tokenization is text preprocessing, not generation

        # START TIMING (batched generation + codec decoding)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        start = time.perf_counter()
        outputs = model.generate(**inputs, **sampling)
        audios = []
        for row, text in enumerate(texts):
            length = int(outputs.audio_lengths[row])
            audios.append(decode(outputs.audio_codes[row, :, :length]))
            if length >= args.max_new_tokens:
                truncated.append(text)
        # END TIMING (decode() ends with .cpu(), so the audio is materialised)
        elapsed = time.perf_counter() - start
        return audios, elapsed / len(texts)

    # Layout: results/<model_safe>/{MODEL_<safe>_DATASET_<dsid>.jsonl, <dsid>/output_<id>.wav}.
    # Manifest paths are relative to model_dir so later stages resolve them wherever it is copied.
    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # Batch size 1; the first audio is the first `--ttfa_chunk_frames` frames decoded as soon as
    # they are sampled (FirstAudioStreamer). The rest of the utterance is decoded at the end.
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices

        @torch.inference_mode()
        def _ttfa_gen(job, early_stop):
            streamer = FirstAudioStreamer(codec, args.ttfa_chunk_frames, early_stop)
            try:
                outputs = model.generate(**prompts([job["text"]]), audio_streamer=streamer, **sampling)
            except _StopAfterFirstAudio:
                return streamer.chunk, SAMPLING_RATE, streamer.first
            audio = decode(outputs.audio_codes[0, :, : int(outputs.audio_lengths[0])])
            # Shorter than one chunk: the first audio is the whole utterance.
            return audio, SAMPLING_RATE, streamer.first or time.perf_counter()

        # The probe discards its first rows as warm-up but does not run one: compile the kernels
        # here so the first probed rows are not dominated by CUDA initialisation.
        for i in range(min(args.warmup_steps, len(dataset))):
            generate_tts([dataset[i][args.text_column]])
        set_seed(args.seed)

        run_probe(
            _ttfa_gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            # The chunk size is the floor under TTFA, so it is recorded with the numbers.
            extra={"device": args.device, "voice": args.voice, "ttfa_chunk_frames": args.ttfa_chunk_frames},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    def run_batch(batch):
        return generate_tts(list(batch[args.text_column]))

    # ── Main loop: TTS → write JSONL, one batch at a time ───────────────────
    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(run_batch, dataset, starts, args.batch_size, args.warmup_steps)
    set_seed(args.seed)  # so warm-up does not shift the RNG stream of the timed run
    truncated.clear()

    for batch_start in tqdm(starts, desc="Generating"):
        batch = dataset[batch_start : batch_start + args.batch_size]
        texts = list(batch[args.text_column])
        audios, per_sample_time = run_batch(batch)

        for sample_id, text, audio in zip(batch["id"], texts, audios):
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            sf.write(os.path.join(model_dir, rel_path), audio, SAMPLING_RATE)
            write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLING_RATE, per_sample_time, text))
        manifest_file.flush()

    manifest_file.close()
    if truncated:
        print(f"WARNING: {len(truncated)} sample(s) hit --max_new_tokens={args.max_new_tokens} "
              "before the end-of-audio frame (audio cut off).")
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="LiquidAI/LFM3-Audio-2B", help="LFM3-Audio repo id.")
    parser.add_argument("--voice", type=str, default="female", choices=["female", "male"],
                        help="Built-in voice (selected through the system prompt).")
    parser.add_argument("--language", type=str, default=None,
                        help="Language/locale named in the system prompt (default: from --split, see "
                             "LANGUAGES), e.g. 'British English'.")
    # Speech sampling settings from the model card ("Audio sampling parameters").
    parser.add_argument("--audio_temperature", type=float, default=1.0, help="Speech-code sampling temperature.")
    parser.add_argument("--audio_top_k", type=int, default=128, help="Speech-code top-k.")
    parser.add_argument("--audio_repetition_penalty", type=float, default=1.1, help="Speech-code repetition penalty.")
    parser.add_argument("--audio_repetition_window", type=int, default=4096,
                        help="Past speech frames covered by the repetition penalty.")
    # 1 frame = 80 ms of audio, so 1024 frames is ~82 s: far beyond any eval utterance.
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max generated slots (speech frames) per sample.")
    parser.add_argument("--ttfa_chunk_frames", type=int, default=1,
                        help="TTFA probe: frames (80 ms each) decoded for the first playable chunk.")
    add_common_args(parser, batch_size=32)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
