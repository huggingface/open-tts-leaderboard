"""
TTS synthesis for the Open TTS Leaderboard (Mithi backend, stage 1).

Mithi (https://huggingface.co/Oomni/mithi-v2-flash, https://huggingface.co/Oomni/mithi-v1-edge) speaks English and ten
Indian languages with built-in voices (no voice cloning). Each model repo ships a self-contained `run.py` that defines
the `Mithi` class; this script downloads the pinned repo, imports that class, and synthesizes every target text with
one fixed voice.

Batching: the texts of a batch are synthesized together, each seeded exactly as `Mithi.synthesize` seeds it, so the
audio matches `Mithi.synthesize` whatever batch a text is in. Long texts are split and joined as `Mithi.synthesize` does.

Stage 2 (`transformers/transcribe.py`) transcribes the wavs with Qwen3-ASR to fill `pred_text`; scoring computes WER
and RTFx from the manifest. Fixed voice, so no SIM stage.
"""

import argparse
import importlib.util
import os
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

SAMPLING_RATE = 24_000
PIECE_GAP_S = 0.15          # silence between the pieces of a long text (Mithi.synthesize default)


def load_mithi(model_id, revision, model_dir, device):
    """Import `Mithi` from the model repo's run.py and load it. A local `model_dir` skips the download."""
    if not model_dir:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(model_id, revision=revision)
    spec = importlib.util.spec_from_file_location("mithi_run", os.path.join(model_dir, "run.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Mithi(model_dir, device=device)


def _sample(logits, temp, top_p, generator):
    """Same as the model's own sampler, with an explicit generator."""
    if temp <= 0:
        return int(logits.argmax())
    p = F.softmax(logits / temp, -1)
    sp, idx = p.sort(descending=True)
    sp = sp * (sp.cumsum(-1) - sp < top_p)
    return int(idx[torch.multinomial(sp / sp.sum(), 1, generator=generator)])


class BatchedMithi:
    """Batched version of `Mithi.synth_sentence` (see the module docstring)."""

    def __init__(self, tts, voice):
        import mithi_tts.prosody as P
        import mithi_tts.tts as T

        if voice not in tts.voices:
            raise ValueError(f"unknown voice {voice} (one of {', '.join(tts.voices)})")
        self.tts, self.voice, self.T, self.P = tts, voice, T, P
        self.dev = tts.dev
        self.s = tts.settings

    # ── text -> model inputs ──
    def prepare(self, text, lang):
        tts, fe, dev = self.tts, self.T.fe, self.dev
        norm, ps = fe.phonemes(text, lang)
        ids = torch.LongTensor([0, *[i for i in map(tts.vocab.get, ps) if i is not None], 0])
        words = fe.words_of(text, norm)
        tok_word, n_groups = fe.token_words(ids.tolist())
        if n_groups == len(words):
            wv, _ = tts.text_enc.word_vectors(" ".join(words))
            tw = torch.tensor(tok_word)
            word_vec, has = wv[tw.clamp(min=0)], (tw >= 0).to(dev)
        else:
            word_vec = torch.zeros(len(ids), tts.text_enc.dim, device=dev)
            has = torch.zeros(len(ids), dtype=torch.bool, device=dev)
        return ids, word_vec, has, len(ps)

    # ── batch of pieces ──
    @torch.no_grad()
    def write(self, items, lang, generators):
        tts, P, dev, s = self.tts, self.P, self.dev, self.s
        w = tts.writer
        B, L = len(items), max(len(it[0]) for it in items)
        x = torch.zeros(B, L, dtype=torch.long, device=dev)
        wv = torch.zeros(B, L, items[0][1].shape[-1], device=dev)
        has = torch.zeros(B, L, dtype=torch.bool, device=dev)
        pad = torch.ones(B, L, dtype=torch.bool, device=dev)
        for b, (ids, word_vec, h, _) in enumerate(items):
            n = len(ids)
            x[b, :n], wv[b, :n], has[b, :n], pad[b, :n] = ids.to(dev), word_vec, h, False
        spk = torch.full((B,), tts.voices.index(self.voice), device=dev)
        lng = torch.full((B,), self.T.WRITER_LANGS.index(lang), device=dev)
        H = w.read(x, wv, has, spk, lng, pad)

        pause_ids, breath_id = self.T.fe.PAUSE_IDS, self.T.fe.BREATH_ID
        head_temp = {k: s["voicing_temp"] for k in ("voiced", "uv0", "uv1")}
        temp, top_p, n_codes = s["writer_temp"], s["writer_top_p"], len(P.NAMES)
        queues = [[(int(t), i) for i, t in enumerate(it[0].tolist())] for it in items]
        dec_ids, dec_map, rows = [[] for _ in range(B)], [[] for _ in range(B)], [[] for _ in range(B)]
        prev_p5, prev_e3 = [P.REF_P] * B, [P.REF_E] * B
        while True:
            act = [b for b in range(B) if queues[b]]
            if not act:
                break
            cur = {}
            for b in act:
                tok, i = queues[b].pop(0)
                dec_ids[b].append(tok)
                dec_map[b].append(i)
                cur[b] = tok
            ai = torch.tensor(act, device=dev)
            ids = torch.tensor([dec_ids[b] for b in act], device=dev)
            ctx = torch.tensor([rows[b] + [[0] * n_codes] for b in act], device=dev)
            acc = w.write(H[ai], pad[ai], ids, torch.tensor([dec_map[b] for b in act], device=dev), ctx,
                          torch.zeros_like(ids, dtype=torch.bool))[:, -1]
            new = [[] for _ in act]
            for k, head in enumerate(w.depth):
                t = head_temp.get(P.NAMES[k], temp)
                logits = head(acc)
                for j, b in enumerate(act):
                    r = new[j]
                    if 2 <= k <= 8 and r[P.VOICED] == 0:
                        a = P.PITCH_NONE if k >= 4 else 0
                    else:
                        c = _sample(logits[j], t, 1.0 if k == P.BREATH_NEXT else top_p, generators[b])
                        a = int(w._abs_from_rel(k, torch.tensor(c), torch.tensor(r + [0]), torch.tensor(prev_p5[b]),
                                                torch.tensor(prev_e3[b]))) if 4 <= k <= 11 else c
                    if k == P.BREATH_NEXT and (cur[b] not in pause_ids or cur[b] == breath_id or not queues[b]):
                        a = 0
                    r.append(a)
                acc = acc + w.code_emb[k](torch.tensor([r[k] for r in new], device=dev))
            for j, b in enumerate(act):
                r = new[j]
                rows[b].append(r)
                if r[P.VOICED] == 1:
                    prev_p5[b] = r[8]
                prev_e3[b] = r[11]
                if r[P.BREATH_NEXT] == 1:
                    queues[b].insert(0, (breath_id, dec_map[b][-1]))
        out = []
        for b in range(B):
            p5, breath = P.abs_to_p5(torch.tensor(rows[b]))
            out.append((torch.tensor(dec_ids[b]), torch.cat([p5, breath[:, None]], 1)))
        return out

    # ── one piece -> audio ──
    @torch.no_grad()
    def render(self, dec_ids, codes, n_ps):
        tts, s, dev, T = self.tts, self.s, self.dev, self.T
        codes[[0, -1], 0] = codes[[0, -1], 0].clamp(max=s["edge_max"])
        dur, f0, en = T.decode(codes[:, :T.P5_COLS].numpy())
        f0 = f0 * np.float32(2 ** (s["pitch_shift"] / 12))
        y = dec_ids[None].to(dev)
        n = int(dur.sum())
        aln = torch.zeros(1, y.shape[1], n, device=dev)
        aln[0, torch.repeat_interleave(torch.arange(y.shape[1], device=dev), torch.from_numpy(dur).to(dev)),
            torch.arange(n, device=dev)] = 1
        pack = tts.packs[self.voice]
        ref = pack[min(n_ps, len(pack)) - 1]
        h = tts.kokoro.text_encoder(y, torch.tensor([y.shape[1]], device=dev), torch.zeros_like(y, dtype=torch.bool)) @ aln
        audio = tts.kokoro.decoder(h, torch.from_numpy(f0)[None].to(dev), torch.from_numpy(en)[None].to(dev), ref[:, :128])
        return audio.squeeze().float().cpu().numpy()

    def _set_rng(self, generator):
        if self.dev.type == "cuda":
            torch.cuda.set_rng_state(generator.get_state(), self.dev)
        else:
            torch.set_rng_state(generator.get_state())

    def synthesize(self, texts, lang):
        """List of texts -> list of 24 kHz float32 waveforms in [-1, 1]."""
        pieces, owner = [], []
        for r, text in enumerate(texts):
            if text and text.strip():
                for p in self.tts.split(text, lang):
                    pieces.append(p)
                    owner.append(r)
        items = [self.prepare(p, lang) for p in pieces]
        gens = [torch.Generator(device=self.dev).manual_seed(self.s["seed"]) for _ in items]
        audios = [[] for _ in texts]
        if items:
            for k, (dec_ids, codes) in enumerate(self.write(items, lang, gens)):
                self._set_rng(gens[k])
                audios[owner[k]].append(self.render(dec_ids, codes, items[k][3]).astype(np.float32))
        gap = np.zeros(int(PIECE_GAP_S * SAMPLING_RATE), np.float32)
        out = []
        for segs in audios:
            if not segs:
                out.append(np.zeros(1, np.float32))
                continue
            joined = [segs[0]]
            for seg in segs[1:]:
                joined += [gap, seg]
            out.append(np.clip(np.concatenate(joined), -1.0, 1.0))
        return out


def main(args):
    set_seed(args.seed)
    tts = load_mithi(args.model_id, args.revision, args.model_dir, args.device)
    lang = args.language or args.split
    if lang not in tts.LANGS:
        raise ValueError(f"{args.model_id} does not speak '{lang}' (it speaks {', '.join(tts.LANGS)})")
    synth = BatchedMithi(tts, args.voice)
    print(f"Loaded Mithi ({args.model_id}, voice={args.voice}, lang={lang}, batch_size={args.batch_size})")
    print_model_size(tts)
    is_cuda = synth.dev.type == "cuda"

    def run_batch(batch):
        texts = batch[args.text_column]
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        audios = synth.synthesize(texts, lang)
        if is_cuda:
            torch.cuda.synchronize()
        return audios, (time.perf_counter() - start) / len(texts)

    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path
    dataset = load_tts_dataset(args)

    # ── TTFA probe: batch size 1. Mithi reads a long text piece by piece, so first audio = first piece done. ──
    if args.ttfa_probe != 0:
        from ttfa_probe import run_probe, sample_indices

        def _ttfa_gen(job, early_stop):
            text = job["text"]
            pieces = tts.split(text, lang) if text and text.strip() else []
            first, segs = None, []
            for p in pieces:
                segs.append(synth.synthesize([p], lang)[0])
                if first is None:
                    first = time.perf_counter()
                    if early_stop:
                        break
            audio = np.concatenate(segs) if segs else np.zeros(1, np.float32)
            return audio, SAMPLING_RATE, first

        run_probe(
            _ttfa_gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            extra={"device": args.device, "voice": args.voice},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(run_batch, dataset, starts, args.batch_size, args.warmup_steps)
    set_seed(args.seed)

    for start in tqdm(starts, desc="Generating"):
        batch = dataset[start : start + args.batch_size]
        audios, per_sample_time = run_batch(batch)
        for sample_id, text, audio in zip(batch["id"], batch[args.text_column], audios):
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            if rel_path in done_entries:
                continue
            sf.write(os.path.join(model_dir, rel_path), audio, SAMPLING_RATE)
            write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLING_RATE, per_sample_time, text))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Oomni/mithi-v1-edge", help="Mithi model repo id.")
    parser.add_argument("--revision", type=str, default="main", help="Model repo revision (full commit sha for runs).")
    parser.add_argument("--model_dir", type=str, default=None, help="Local copy of the model repo (skips the download).")
    parser.add_argument("--voice", type=str, default="charvi", help="Built-in voice, e.g. charvi, satvik.")
    parser.add_argument("--language", type=str, default=None, help="Language code; default: the split name (en, hi).")
    add_common_args(parser, batch_size=32)
    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)
    main(args)
