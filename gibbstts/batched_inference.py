"""Experimental native DiT batching for pinned GibbsTTS, without changing weights.

Text/reference preprocessing and codec calls remain per utterance. All 32
utterances share each DiT forward and CTMC step. Padding is excluded from
attention and per-utterance CFG statistics; RoPE positions remain contiguous.
BF16 uses autocast only for estimator compute; logits/solver/codec stay FP32.
"""
from contextlib import nullcontext
from dataclasses import dataclass
from types import MethodType

import torch
import torch.nn.functional as F


@dataclass
class Prepared:
    text: torch.Tensor
    prompt: torch.Tensor
    length: int
    lang: int


def precision_context(precision):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16" else nullcontext())


def enable_positioned_rope(estimator):
    """Reuse prebuilt caches; avoid a CUDA-to-CPU max() sync in every block."""
    for block in estimator.blocks:
        rope = block.attn.rotary_pe
        original = rope.forward

        def forward(self, x, position_ids=None, _original=original):
            if position_ids is None:
                return _original(x)
            cos = self.cos_cached[:, 0, 0][position_ids].unsqueeze(1)
            sin = self.sin_cached[:, 0, 0][position_ids].unsqueeze(1)
            # Keep the upstream arithmetic (FP32 RoPE cache, even under autocast).
            rotating, rest = x[..., :self.d], x[..., self.d:]
            half = self.d // 2
            negative = torch.cat([-rotating[..., half:], rotating[..., :half]], dim=-1)
            return torch.cat([rotating * cos + negative * sin, rest], dim=-1)

        rope.forward = MethodType(forward, rope)


def pack_inputs(prepared, codebook_size):
    b = len(prepared)
    device = prepared[0].text.device
    text_max = max(p.text.numel() for p in prepared)
    prompt_max = max(p.prompt.shape[0] for p in prepared)
    target_max = max(p.length for p in prepared)
    channels = prepared[0].prompt.shape[-1]
    text = torch.zeros((b, text_max), dtype=torch.long, device=device)
    xt = torch.randint(codebook_size, (b, prompt_max + target_max, channels), device=device)
    xt[:, :prompt_max] = codebook_size
    valid = torch.zeros((b, text_max + prompt_max + target_max), dtype=torch.bool, device=device)
    prompt_mask = torch.zeros((b, prompt_max + target_max), dtype=torch.bool, device=device)
    position = torch.zeros_like(valid, dtype=torch.long)
    for i, item in enumerate(prepared):
        nt, np = item.text.numel(), item.prompt.shape[0]
        text[i, text_max-nt:] = item.text
        xt[i, prompt_max-np:prompt_max] = item.prompt
        valid[i, text_max-nt:text_max] = True
        valid[i, text_max+prompt_max-np:text_max+prompt_max+item.length] = True
        prompt_mask[i, prompt_max-np:prompt_max] = True
        position[i, text_max-nt:text_max] = torch.arange(nt, device=device)
        position[i, text_max+prompt_max-np:text_max+prompt_max+item.length] = (
            nt + torch.arange(np + item.length, device=device))
    return dict(text=text, xt=xt, valid=valid, prompt_mask=prompt_mask,
                position=position, prompt_max=prompt_max,
                lang=torch.tensor([p.lang for p in prepared], device=device),
                lengths=[p.length for p in prepared])


def masked_std(x, valid):
    weight = valid.unsqueeze(-1)
    count = valid.sum(dim=1, keepdim=True).unsqueeze(-1) * x.shape[-1]
    mean = (x * weight).sum(dim=(1, 2), keepdim=True) / count
    variance = ((x - mean).square() * weight).sum(dim=(1, 2), keepdim=True) / (count - 1)
    return variance.sqrt()


def estimator_infer(estimator, t, packed, cfg, rescale_cfg, precision, cache=None):
    text, xt = packed["text"], packed["xt"]
    b, l, c = xt.shape
    if cache is None:
        mask = packed["valid"].unsqueeze(-1).float()
        mask = torch.cat([mask, mask], dim=0)
        positions = torch.cat([packed["position"], packed["position"]], dim=0)
    else:
        mask, positions = cache["mask"], cache["positions"]
    with precision_context(precision):
        if cache is None:
            texts_embed = estimator.text_embed(text)
            texts_embed = torch.cat([texts_embed,
                                    estimator.text_cfg_embed[None, None].expand_as(texts_embed)], dim=0)
            indices = torch.arange(c, device=xt.device).view(1, 1, c).expand(b, l, c).reshape(-1)
        else:
            texts_embed, indices = cache["texts_embed"], cache["indices"]
        embed = estimator.token_embed[indices, xt.reshape(-1)].view(b, l, -1)
        x = estimator.input_proj(embed)
        x.add_(packed["prompt_mask"].unsqueeze(-1) * estimator.prompt_embed)
        timing = estimator.time_mlp(estimator.time_embeddings(t)).expand(b, -1)
        language = estimator.lang_embed(packed["lang"]) if cache is None else cache["language"]
        cond = torch.cat([timing, language], dim=-1)
        cond = torch.cat([cond, cond], dim=0)
        x_cfg = torch.where(packed["prompt_mask"].unsqueeze(-1),
                            estimator.token_cfg_embed.to(x.dtype), x)
        x = torch.cat([texts_embed, torch.cat([x, x_cfg], dim=0)], dim=1)
        if cache is None:
            attention = mask * mask.transpose(1, 2)
            attention = torch.zeros_like(attention).masked_fill(
                attention == 0, -torch.finfo(x.dtype).max).unsqueeze(1)
        else:
            assert x.dtype == cache["activation_dtype"]
            attention = cache["attention"]
        for block in estimator.blocks:
            x = block(x, cond, mask, attention, positions)
        x = estimator.final_layer(x[:, text.shape[-1]:], cond)
        x, unconditional = x[:b], x[b:]
        acoustic_valid = packed["valid"][:, text.shape[-1]:]
        original_std = masked_std(x, acoustic_valid)
        x = x + cfg * (x - unconditional)
        rescaled = x * original_std / masked_std(x, acoustic_valid)
        x = rescale_cfg * rescaled + (1 - rescale_cfg) * x
        logits = torch.einsum("blh,ckh->blck", x,
                              estimator.token_embed[:, :estimator.codebook_size])
    return logits.float()


def make_static_cache(estimator, packed, precision):
    """Same values and arithmetic as each original step; preparation stays timed."""
    b, l, c = packed["xt"].shape
    mask = packed["valid"].unsqueeze(-1).float()
    mask = torch.cat([mask, mask], dim=0)
    positions = torch.cat([packed["position"], packed["position"]], dim=0)
    with precision_context(precision):
        texts_embed = estimator.text_embed(packed["text"])
        texts_embed = torch.cat([texts_embed,
                                estimator.text_cfg_embed[None, None].expand_as(texts_embed)], dim=0)
        language = estimator.lang_embed(packed["lang"])
    # Embedding is FP32 under CUDA autocast; concatenation promotes x to FP32.
    assert texts_embed.dtype == torch.float32
    attention = mask * mask.transpose(1, 2)
    attention = torch.zeros_like(attention).masked_fill(
        attention == 0, -torch.finfo(texts_embed.dtype).max).unsqueeze(1)
    indices = torch.arange(c, device=packed["xt"].device).view(1, 1, c).expand(b, l, c).reshape(-1)
    return dict(mask=mask, positions=positions, texts_embed=texts_embed,
                language=language, attention=attention, indices=indices,
                activation_dtype=texts_embed.dtype)


class PhaseEvents:
    """Stream intervals, including CPU gaps; used only in separate profile calls."""
    def __init__(self, enabled):
        self.enabled, self.events = enabled, []

    def begin(self, name):
        if not self.enabled:
            return None
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        self.events.append((name, start, end))
        return end

    def end(self, event):
        if event is not None:
            event.record()

    def finish(self):
        if not self.enabled:
            return None
        torch.cuda.synchronize()
        result = {}
        for name, start, end in self.events:
            result[name] = result.get(name, 0.0) + start.elapsed_time(end) / 1000
        return result


@torch.no_grad()
def synthesize_many(wrapper, rows, language, precision="fp32", *,
                    cache_static=False, skip_top_p_one=False, profile=None, pad_to=None, batch_info=None):
    import torchaudio
    from text import cleaned_text_to_sequence
    from text.phonemize import phonemize
    from models.utils import logits_top_p

    model, config, device = wrapper.model, wrapper.configs, wrapper.device
    phases = PhaseEvents(profile is not None)
    section = phases.begin("frontend_reference_encode")
    prepared = []
    for row in rows:
        prompt_phone, _ = phonemize(row["prompt_text"], language)
        target_phone, _ = phonemize(row["text"], language)
        prompt_ids = cleaned_text_to_sequence(prompt_phone)
        target_ids = wrapper.space_id + cleaned_text_to_sequence(target_phone)
        text = torch.tensor(prompt_ids + target_ids + [model.n_vocab-1], device=device)
        wav, sr = torchaudio.load(str(row["prompt_path"]))
        wav = wrapper.resampler(wav.to(device), sr).unsqueeze(0)
        if wav.shape[1] > 1:
            wav = wav.mean(dim=1, keepdim=True)
        prompt = wrapper.codec.encode(wav)[0]
        center = 3.224 if language == "en" else 3.286
        ratio = max(center * .8, min(prompt.shape[0] / len(prompt_ids), center * 1.25))
        length = int(len(target_ids) * ratio / config.speed)
        prepared.append(Prepared(text, prompt, length, wrapper.language_dict[language]))
    real_count = len(prepared)
    assert real_count > 0
    if pad_to is not None:
        assert real_count <= pad_to
        # Reuse a valid prepared row, preserving real-row maximum lengths.
        # No dummy frontend, audio read, reference encoding or output decoding.
        prepared.extend([prepared[0]] * (pad_to - real_count))
    if batch_info is not None:
        batch_info.update(real_count=real_count, execution_batch_size=len(prepared),
                          dummy_count=len(prepared)-real_count,
                          frontend_rows=real_count, codec_decode_rows=real_count)
    phases.end(section)
    section = phases.begin("packing_rope_cache_static_cache")
    packed = pack_inputs(prepared, model.codebook_size)
    max_position = packed["text"].shape[-1] + packed["xt"].shape[1]
    for block in model.estimator.blocks:
        block.attn.rotary_pe._build_cache(max_position, packed["xt"].device)
    cache = make_static_cache(model.estimator, packed, precision) if cache_static else None
    start = packed["prompt_max"]
    times = torch.linspace(0, 1, config.steps+1, device=device)
    phases.end(section)
    for step in range(config.steps):
        section = phases.begin("dit_cfg_logits")
        t = times[step].unsqueeze(0)
        h = times[step+1].unsqueeze(0) - t
        logits = estimator_infer(model.estimator, t, packed, config.cfg,
                                 config.rescale_cfg, precision, cache=cache)[:, start:]
        phases.end(section)
        if step == config.steps-1:
            section = phases.begin("final_argmax")
            packed["xt"][:, start:] = logits.argmax(dim=-1)
            phases.end(section)
        else:
            section = phases.begin("top_p_temperature")
            if not (skip_top_p_one and config.top_p == 1.0):
                logits = logits_top_p(logits, config.top_p)
            logits = logits / config.temperature
            phases.end(section)
            section = phases.begin("ctmc_solver")
            packed["xt"][:, start:] = model.solver(t, h, packed["xt"][:, start:], logits)
            phases.end(section)
    section = phases.begin("codec_decode_cpu_return")
    audio = [wrapper.codec.decode(packed["xt"][i:i+1, start:start+n].clamp(0, 1023))
             for i, n in enumerate(packed["lengths"][:real_count])]
    phases.end(section)
    if profile is not None:
        profile.update(phases.finish())
    return audio


@torch.no_grad()
def synthesize_single(wrapper, row, language, precision="fp32"):
    original = wrapper.model.estimator.infer
    if precision == "bf16":
        def mixed_infer(*args, **kwargs):
            with precision_context("bf16"):
                logits = original(*args, **kwargs)
            return logits.float()
        wrapper.model.estimator.infer = mixed_infer
    try:
        return [wrapper.synthesize(prompt_audio=str(row["prompt_path"]),
                                   prompt_text=row["prompt_text"], target_text=row["text"],
                                   prompt_lang=language, target_lang=language)]
    finally:
        wrapper.model.estimator.infer = original
