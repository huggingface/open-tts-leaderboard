"""
TTS synthesis for the Open TTS Leaderboard (VieNeu-TTS v3 Turbo backend, stage 1).

VieNeu-TTS v3 Turbo (pnnbao-ump/VieNeu-TTS-v3-Turbo, Apache-2.0) is a Vietnamese/English TTS model
(48 kHz) driven through the `vieneu` Python SDK (https://github.com/pnnbao97/VieNeu-TTS). On a CUDA
machine the SDK uses its PyTorch engine and batches the text chunks of many inputs into one forward
pass (`infer_batch`). Default preset voice, no voice cloning, so no stage 3 (SIM).

Per-sample generation time is the minibatch time divided by the minibatch size, as in the other
batched backends. `--batch_size` is also the manifest-writing / resume granularity.

`--mode=v3nano` runs VieNeu-TTS v3 Nano (pnnbao-ump/VieNeu-TTS-v3-Nano, 24 kHz), a small flow model
the SDK ships as four ONNX graphs on the CPU EP only. On CUDA the graphs are reloaded under the CUDA EP
(as in `paradee/`), which needs `onnxruntime-gpu`; Nano's `infer_batch` is sequential.
"""

import argparse
import os
import time
from importlib.metadata import version

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from vieneu import Vieneu

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)


def _use_cuda_sessions(engine, device):
    """Reload the Nano engine's four ONNX graphs under the CUDA EP (the SDK hardcodes the CPU EP)."""
    import onnxruntime as ort

    device_id = int(device.split(":")[1]) if ":" in device else 0
    # Load pip-installed CUDA/cuDNN (nvidia-* wheels) when they are not on LD_LIBRARY_PATH.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    graph_dir = engine._fetch(engine.repo, engine.hf_token)  # already in the HF cache
    providers = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
    for attr, fn in (("s_text", "text_encoder.onnx"), ("s_dur", "duration_predictor.onnx"),
                     ("s_ve", "vector_estimator.onnx"), ("s_dec", "codec_decoder.onnx")):
        sess = ort.InferenceSession(str(graph_dir / fn), engine._so, providers=providers)
        # The CUDA EP silently falls back to CPU (e.g. missing cuDNN); refuse rather than mislabel.
        if "CUDAExecutionProvider" not in sess.get_providers():
            raise RuntimeError(f"--device={device} but the CUDA EP did not load for {fn}: {sess.get_providers()}")
        setattr(engine, attr, sess)
    engine._prov = providers


def main(args):
    set_seed(args.seed)

    device = "cuda" if args.device.startswith("cuda") else "cpu"
    tts = Vieneu(mode=args.mode, device=device)
    if args.mode == "v3nano" and device == "cuda":
        _use_cuda_sessions(tts.engine, args.device)
    print(f"Loaded VieNeu-TTS {args.mode} (vieneu {version('vieneu')}, device={device}, "
          f"sample_rate={tts.sample_rate}, voice={args.voice or 'default'})")
    print_model_size(getattr(tts, "model", None))

    sampling_rate = int(tts.sample_rate)
    is_cuda = device == "cuda"

    def synth_batch(texts):
        """Synthesize a list of texts. Return (list of 1d float32 arrays, elapsed_s)."""
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        wavs = tts.infer_batch(texts, voice=args.voice)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        out = []
        for w in wavs:
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            out.append(w if w.size else np.zeros(1, dtype=np.float32))
        return out, elapsed

    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    if args.ttfa_probe != 0:
        raise SystemExit("--ttfa_probe is not implemented for this backend.")

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(lambda batch: synth_batch(list(batch[args.text_column])), dataset, starts, args.batch_size, args.warmup_steps)
    set_seed(args.seed)  # warm-up must not shift the RNG stream of the timed run

    for start in tqdm(starts, desc="Generating"):
        batch = dataset[start : start + args.batch_size]
        texts = list(batch[args.text_column])
        wavs, elapsed = synth_batch(texts)
        per_sample = elapsed / len(texts)
        for sample_id, text, audio in zip(batch["id"], texts, wavs):
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            sf.write(os.path.join(model_dir, rel_path), audio, sampling_rate)
            write_entry(manifest_file, manifest_entry(rel_path, len(audio) / sampling_rate, per_sample, text))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="pnnbao-ump/VieNeu-TTS-v3-Turbo",
                        help="Model repo id (recorded in the manifest folder name; weights are fetched by the SDK).")
    parser.add_argument("--mode", type=str, default="v3turbo", choices=["v3turbo", "v3nano"],
                        help="SDK engine: v3turbo (PyTorch, batched) or v3nano (ONNX).")
    parser.add_argument("--voice", type=str, default=None,
                        help="Preset voice name or id (default: the SDK's default preset voice).")
    add_common_args(parser, batch_size=32)
    parser.set_defaults(split="vi")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
