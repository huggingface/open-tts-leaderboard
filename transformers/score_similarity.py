"""
Speaker similarity (SIM) for the Open TTS Leaderboard (stage 3, voice cloning only).

Embeds each generated wav and its reference prompt (`prompt_audio_filepath`) with a
speaker-verification model and writes their cosine similarity into the manifest's `sim` field.
`normalizer.eval_utils.score_results` then reports the mean SIM alongside WER + RTFx.

Backends (`--sim_backend`); the two scales are NOT comparable:

  `wavlm_seed_tts` (default)
      seed-tts-eval's SIM model (WavLM-Large + ECAPA-TDNN, UniSpeech `wavlm_large_finetune.pth`),
      comparable with published Seed-TTS / F5-TTS / CosyVoice SIM. Runs one clip at a time, like
      upstream. Pure transformers + torch: the backbone is an HF `WavLMModel` with the checkpoint's
      weights remapped onto it (no s3prl / torch.hub).

  `xvector`
      HF `WavLMForXVector` (microsoft/wavlm-base-plus-sv). Cheap, but compressed: different speakers
      already score ~0.80, so it is too narrow to rank strong cloning models. Writes `sim` in place.

Resumable: a row is re-scored unless it already has a numeric `sim` from the requested backend
(recorded as `sim_model`). `--overwrite` forces every row. Progress is saved every `--save_every` rows.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.audio_utils import load_audio

from transformers import AutoFeatureExtractor, WavLMConfig, WavLMForXVector, WavLMModel

DEFAULT_XVECTOR_MODEL_ID = "microsoft/wavlm-base-plus-sv"

# SIM checkpoint from microsoft/UniSpeech (downstreams/speaker_verification, "WavLM large / Fix
# pre-train: No"; https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view), mirrored
# on the Hub by scripts/upload_sim_checkpoint.py (which parses these three constants) and pinned by the
# digest of the official file.
WAVLM_SEED_TTS_REPO_ID = "bezzam/wavlm_large_finetune_seed_tts_eval"
WAVLM_SEED_TTS_FILENAME = "wavlm_large_finetune.pth"
WAVLM_SEED_TTS_SHA256 = "51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b"
# Config only (config.json, ~2 KB) — the weights all come from the checkpoint above.
WAVLM_BACKBONE_CONFIG_ID = "microsoft/wavlm-large"

# Assumed `sim_model` for rows that have a `sim` but no `sim_model` field.
LEGACY_SIM_MODEL = DEFAULT_XVECTOR_MODEL_ID
# The xvector backend writes `sim` into the manifest in place; any other backend writes a separate
# `<manifest>_<backend>.jsonl` so both scales coexist. Keep the suffix in sync with
# `normalizer.eval_utils.score_results`.
LEGACY_SIM_BACKEND = "xvector"


def sim_manifest_path(manifest_path, sim_backend):
    """Where `sim_backend`'s scores belong: the manifest itself for the legacy backend, else a fork."""
    if sim_backend == LEGACY_SIM_BACKEND:
        return manifest_path
    stem, ext = os.path.splitext(manifest_path)
    return f"{stem}_{sim_backend}{ext}"


def _min_samples_for_embedding(wavlm, min_frames=2):
    """Shortest waveform (in samples) whose WavLM conv frontend emits at least `min_frames` frames.

    Below this, both heads crash (time-axis InstanceNorm / TDNN need >1 frame). For WavLM-Large this
    is 720 samples (45 ms at 16 kHz). It is the hard floor, not a quality threshold.
    """
    length_of = wavlm._get_feat_extract_output_lengths
    lo, hi = 1, 16_000
    while int(length_of(hi)) < min_frames:
        hi *= 2
    while lo < hi:  # binary search for the smallest length reaching min_frames
        mid = (lo + hi) // 2
        if int(length_of(mid)) >= min_frames:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Backend 1 — HF WavLMForXVector
# ══════════════════════════════════════════════════════════════════════════════════════════════
class XVectorEmbedder:
    """WavLM + TDNN x-vector head via `WavLMForXVector`. Batches freely (padding is masked)."""

    max_batch_size = None  # no constraint

    def __init__(self, model_id, device, dtype):
        self.name = model_id
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMForXVector.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.sampling_rate = self.feature_extractor.sampling_rate  # 16_000
        self.device = device
        self.min_samples = _min_samples_for_embedding(self.model.wavlm)

    def __call__(self, speech):
        inputs = self.feature_extractor(
            speech, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        ).to(self.device)
        inputs = inputs.to(self.model.dtype)
        with torch.no_grad():
            embeddings = self.model(**inputs).embeddings
        return F.normalize(embeddings, dim=-1).float().cpu()


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Backend 2 — WavLM-Large + ECAPA-TDNN (seed-tts-eval canonical)
#
# ECAPA-TDNN head transcribed from UniSpeech (MIT):
# https://github.com/microsoft/UniSpeech/blob/main/downstreams/speaker_verification/models/
# Module/parameter names match upstream so the checkpoint loads with `strict=True`. The 25 hidden
# states come from an HF `WavLMModel` (same tensors/order as upstream's s3prl front end).
# ══════════════════════════════════════════════════════════════════════════════════════════════
class Res2Conv1dReluBn(nn.Module):
    """in_channels == out_channels == channels."""

    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, f"{channels} % {scale} != 0"
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = nn.ModuleList(
            nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
            for _ in range(self.nums)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(self.width) for _ in range(self.nums))

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    """The SE connection of the 1D case."""

    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


class SE_Res2Block(nn.Module):
    """SE-Res2Block of the ECAPA-TDNN architecture."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)

        return x + residual


class AttentiveStatsPool(nn.Module):
    """Attentive weighted mean and standard deviation pooling."""

    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att

        # Conv1d with stride == 1 rather than Linear, so inputs need no transpose.
        in_channels = in_dim * 3 if global_context_att else in_dim
        self.linear1 = nn.Conv1d(in_channels, attention_channels, kernel_size=1)  # W and b in the paper
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)      # V and k in the paper

    def forward(self, x):
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here — upstream found it hard to converge.
        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class WavLMEcapaTdnn(nn.Module):
    """ECAPA-TDNN over a softmax-weighted sum of WavLM-Large's 25 hidden states.

    Equivalent to upstream's `ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large')`, with the
    s3prl front end replaced by `wavlm` (an HF `WavLMModel`). Head parameter names match the
    checkpoint; the backbone lives under `wavlm.*` and is loaded separately.
    """

    def __init__(self, wavlm, feat_dim=1024, channels=512, emb_dim=256):
        super().__init__()
        self.wavlm = wavlm
        self.feat_num = 1 + wavlm.config.num_hidden_layers  # 25 for WavLM-Large
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))

        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.channels = [channels] * 4 + [1536]

        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(self.channels[0], self.channels[1], kernel_size=3, stride=1, padding=2, dilation=2, scale=8, se_bottleneck_dim=128)
        self.layer3 = SE_Res2Block(self.channels[1], self.channels[2], kernel_size=3, stride=1, padding=3, dilation=3, scale=8, se_bottleneck_dim=128)
        self.layer4 = SE_Res2Block(self.channels[2], self.channels[3], kernel_size=3, stride=1, padding=4, dilation=4, scale=8, se_bottleneck_dim=128)

        self.conv = nn.Conv1d(channels * 3, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(self.channels[-1], attention_channels=128, global_context_att=False)
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def get_feat(self, wav):
        with torch.no_grad():
            hidden_states = self.wavlm(wav, output_hidden_states=True).hidden_states
        x = torch.stack(hidden_states, dim=0)  # (25, B, T, C)
        norm_weights = F.softmax(self.feature_weight, dim=-1).view(-1, 1, 1, 1)
        x = (norm_weights * x).sum(dim=0)      # (B, T, C)
        x = torch.transpose(x, 1, 2) + 1e-6    # (B, C, T)
        return self.instance_norm(x)

    def forward(self, wav):
        x = self.get_feat(wav)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        return self.linear(out)


def _convert_wavlm_backbone_state_dict(orig, target_keys):
    """Remap original-WavLM (`feature_extract.model.*`) parameter names onto HF `WavLMModel` ones.

    Same correspondence as transformers' `convert_wavlm_original_pytorch_checkpoint_to_pytorch.py`,
    spelled out explicitly so an unknown name fails loudly. `target_keys` (the destination key set)
    picks the weight-norm spelling and is used to assert exact coverage.
    """
    prefix = "feature_extract.model."
    # torch >= 2.1 `nn.utils.parametrizations.weight_norm` stores weight_g/weight_v as
    # `parametrizations.weight.original0/1`; older `nn.utils.weight_norm` uses `weight_g`/`weight_v`.
    parametrized = "encoder.pos_conv_embed.conv.parametrizations.weight.original0" in target_keys
    pos_conv_g = "parametrizations.weight.original0" if parametrized else "weight_g"
    pos_conv_v = "parametrizations.weight.original1" if parametrized else "weight_v"

    def convert(key):
        # Feature encoder: Sequential(conv, dropout, Sequential(transpose, layer_norm, transpose), gelu)
        if key.startswith("feature_extractor.conv_layers."):
            rest = key[len("feature_extractor.conv_layers.") :]
            layer_id, sub = rest.split(".", 1)
            if sub == "0.weight":
                return f"feature_extractor.conv_layers.{layer_id}.conv.weight"
            if sub.startswith("2.1."):
                return f"feature_extractor.conv_layers.{layer_id}.layer_norm.{sub[len('2.1.'):]}"
            return None
        if key.startswith("layer_norm."):
            return f"feature_projection.layer_norm.{key[len('layer_norm.'):]}"
        if key.startswith("post_extract_proj."):
            return f"feature_projection.projection.{key[len('post_extract_proj.'):]}"
        if key == "mask_emb":
            return "masked_spec_embed"
        if key.startswith("encoder.pos_conv.0."):
            suffix = key[len("encoder.pos_conv.0.") :]
            mapped = {"bias": "bias", "weight_g": pos_conv_g, "weight_v": pos_conv_v}.get(suffix)
            return f"encoder.pos_conv_embed.conv.{mapped}" if mapped else None
        if key.startswith("encoder.layer_norm."):
            return key
        if key.startswith("encoder.layers."):
            rest = key[len("encoder.layers.") :]
            layer_id, sub = rest.split(".", 1)
            out = f"encoder.layers.{layer_id}."
            for src, dst in (
                ("self_attn.q_proj.", "attention.q_proj."),
                ("self_attn.k_proj.", "attention.k_proj."),
                ("self_attn.v_proj.", "attention.v_proj."),
                ("self_attn.out_proj.", "attention.out_proj."),
                ("self_attn.grep_linear.", "attention.gru_rel_pos_linear."),
                ("self_attn.relative_attention_bias.", "attention.rel_attn_embed."),
                ("self_attn_layer_norm.", "layer_norm."),
                ("final_layer_norm.", "final_layer_norm."),
                ("fc1.", "feed_forward.intermediate_dense."),
                ("fc2.", "feed_forward.output_dense."),
            ):
                if sub.startswith(src):
                    return out + dst + sub[len(src) :]
            if sub == "self_attn.grep_a":
                return out + "attention.gru_rel_pos_const"
            return None
        return None

    converted, unmapped = {}, []
    for key, value in orig.items():
        if not key.startswith(prefix):
            continue
        mapped = convert(key[len(prefix) :])
        if mapped is None:
            unmapped.append(key)
        else:
            converted[mapped] = value

    if unmapped:
        raise RuntimeError(f"Unmapped WavLM backbone keys in checkpoint: {sorted(unmapped)[:10]}")
    missing = sorted(target_keys - converted.keys())
    unexpected = sorted(converted.keys() - target_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"WavLM backbone key mismatch after conversion — missing {len(missing)}: {missing[:10]}; "
            f"unexpected {len(unexpected)}: {unexpected[:10]}"
        )
    return converted


def _resolve_wavlm_seed_tts_checkpoint(path, repo_id, filename, expected_sha256):
    """Return a local path to `wavlm_large_finetune.pth`, downloading + digest-checking if needed."""
    if path:
        return path

    from huggingface_hub import hf_hub_download

    resolved = hf_hub_download(repo_id=repo_id, filename=filename)
    if expected_sha256:
        import hashlib

        digest = hashlib.sha256()
        with open(resolved, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {repo_id}/{filename}: expected {expected_sha256}, got {actual}. "
                "The mirror may have been re-uploaded; verify it before trusting any SIM numbers."
            )
    return resolved


class WavLMSeedTtsEmbedder:
    """seed-tts-eval's SIM model: WavLM-Large + ECAPA-TDNN, one clip per forward pass.

    No batching: instance norm and attentive pooling take time-axis statistics, so padding would
    shift the embeddings of shorter clips. Upstream also embeds one clip at a time.
    """

    max_batch_size = 1
    sampling_rate = 16_000

    def __init__(self, checkpoint_path, config_id, device, dtype):
        self.name = f"wavlm_large_finetune+ecapa_tdnn ({os.path.basename(checkpoint_path)})"
        self.device = device

        config = WavLMConfig.from_pretrained(config_id)
        config.apply_spec_augment = False
        config.layerdrop = 0.0  # eval() already disables it; make it explicit
        backbone = WavLMModel(config)

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
        backbone.load_state_dict(
            _convert_wavlm_backbone_state_dict(state_dict, set(backbone.state_dict())), strict=True
        )

        self.model = WavLMEcapaTdnn(backbone, feat_dim=config.hidden_size)
        # `loss_calculator.*` is the training-time AM-softmax speaker classifier; inference only
        # needs the 256-d embedding the head produces, so it has no counterpart here.
        head = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("feature_extract.") and not k.startswith("loss_calculator.")
        }
        missing, unexpected = self.model.load_state_dict(head, strict=False)
        # The backbone (`wavlm.*`) is already loaded, so it is the only expected "missing" group.
        stray = [k for k in missing if not k.startswith("wavlm.")]
        if stray or unexpected:
            raise RuntimeError(f"ECAPA head key mismatch — missing {stray[:10]}; unexpected {list(unexpected)[:10]}")

        self.model.to(device=device, dtype=dtype).eval()
        self.min_samples = _min_samples_for_embedding(backbone)

    def __call__(self, speech):
        assert len(speech) == 1, "WavLMSeedTtsEmbedder embeds one clip at a time"
        wav = torch.from_numpy(speech[0]).float().unsqueeze(0)
        # WavLM-Large was pre-trained on layer-normalized waveforms (`WavLMConfig.normalize` is
        # True for the large model), which is what s3prl applies before the forward pass.
        wav = F.layer_norm(wav, wav.shape)
        wav = wav.to(device=self.device, dtype=self.model.linear.weight.dtype)
        with torch.no_grad():
            embeddings = self.model(wav)
        return F.normalize(embeddings, dim=-1).float().cpu()


def build_embedder(args, torch_dtype):
    if args.sim_backend == "xvector":
        return XVectorEmbedder(args.sim_model_id or DEFAULT_XVECTOR_MODEL_ID, args.device, torch_dtype)
    checkpoint_path = _resolve_wavlm_seed_tts_checkpoint(
        args.sim_ckpt, args.sim_ckpt_repo, args.sim_ckpt_file, args.sim_ckpt_sha256
    )
    return WavLMSeedTtsEmbedder(checkpoint_path, args.wavlm_config_id, args.device, torch_dtype)


def main(args):
    torch_dtype = getattr(torch, args.dtype)

    # ── Read manifest ────────────────────────────────────────────────────────
    with open(args.manifest_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if not entries:
        raise ValueError(f"No entries found in manifest {args.manifest_path}")
    print(f"Scoring speaker similarity for {len(entries)} samples from {args.manifest_path}")

    out_path = args.output_manifest_path or sim_manifest_path(args.manifest_path, args.sim_backend)
    if out_path != args.manifest_path:
        print(f"Writing scores to {out_path} (the source manifest is left untouched)")
        # Rebuild the fork from the source manifest every run, carrying over only previous sims, so
        # upstream fields (`pred_text`, `duration`, ...) stay in sync with re-run earlier stages.
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                prior = {}
                for line in f:
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    if e.get("audio_filepath") is not None:
                        prior[e["audio_filepath"]] = e
            carried = 0
            for e in entries:
                hit = prior.get(e.get("audio_filepath"))
                if hit and isinstance(hit.get("sim"), (int, float)):
                    # API regeneration reuses filenames, including when the old fork remains
                    # in the bucket. Reuse a score only for the same audio/reference pair.
                    if e.get("timing_backend") == "api" and any(
                        not e.get(field) or e[field] != hit.get(field)
                        for field in ("audio_sha256", "prompt_audio_sha256")
                    ):
                        continue
                    e["sim"], e["sim_model"] = hit["sim"], hit.get("sim_model")
                    carried += 1
            print(f"  carried over {carried} existing score(s) from the previous run")

    # Wav paths are relative to the manifest's directory (the fork lives there too); absolute or
    # unresolvable paths are used as-is. Mirrors transcribe.py.
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest_path))

    def _resolve(p):
        if not p or os.path.isabs(p):
            return p
        cand = os.path.join(manifest_dir, p)
        return cand if os.path.exists(cand) else p

    # ── Speaker-verification model ─────────────────────────────────────────────
    embedder = build_embedder(args, torch_dtype)
    print(f"SIM backend: {args.sim_backend} — {embedder.name}")
    batch_size = args.batch_size
    if embedder.max_batch_size and batch_size > embedder.max_batch_size:
        print(
            f"  note: the {args.sim_backend} backend is length-sensitive, so batch_size is pinned to "
            f"{embedder.max_batch_size} (requested {batch_size}) to match the reference implementation."
        )
        batch_size = embedder.max_batch_size

    sim_sr = embedder.sampling_rate
    max_samples = int(args.max_audio_seconds * sim_sr) if args.max_audio_seconds > 0 else None

    def load_clip(path):
        """Read one clip at the embedder's sample rate, capped to --max_audio_seconds."""
        speech = load_audio(path, sampling_rate=sim_sr)
        if max_samples is not None:
            # Cap length: very long (runaway) clips OOM WavLM; a few seconds suffice for SIM.
            speech = speech[:max_samples]
        return speech

    def rewrite_manifest():
        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Only score rows that have a reference and (unless --overwrite) no usable sim yet. A sim from a
    # different backend is not usable: the two scales are incompatible, so it must be recomputed.
    def needs_scoring(e):
        if not e.get("prompt_audio_filepath"):
            return False
        if args.overwrite or not isinstance(e.get("sim"), (int, float)):
            return True
        return e.get("sim_model", LEGACY_SIM_MODEL) != embedder.name

    todo = [i for i, e in enumerate(entries) if needs_scoring(e)]
    # Batch clips of similar length together so padding=True wastes less memory/compute
    # (otherwise one long clip pads its whole batch up to its length).
    todo.sort(key=lambda i: entries[i].get("duration") or 0.0)
    skipped = len(entries) - len(todo)
    if skipped:
        print(f"Resuming/skipping: {skipped} rows already scored by this backend or missing a reference.")

    # ── Score in minibatches; fill `sim` and persist every --save_every rows ───
    done = 0
    too_short = []  # (audio_filepath, n_samples) for rows the embedder cannot process at all
    for start in range(0, len(todo), args.save_every):
        chunk = todo[start : start + args.save_every]
        for b in range(0, len(chunk), batch_size):
            idxs = chunk[b : b + batch_size]
            gen_speech = [load_clip(_resolve(entries[i]["audio_filepath"])) for i in idxs]
            ref_speech = [load_clip(_resolve(entries[i]["prompt_audio_filepath"])) for i in idxs]

            # Too-short clips would crash the embedder (see _min_samples_for_embedding); give them
            # `sim: null`, which eval_utils skips when averaging.
            keep = [
                k
                for k in range(len(idxs))
                if len(gen_speech[k]) >= embedder.min_samples and len(ref_speech[k]) >= embedder.min_samples
            ]
            for k in range(len(idxs)):
                if k not in keep:
                    i = idxs[k]
                    entries[i]["sim"] = None
                    entries[i]["sim_model"] = embedder.name
                    entries[i]["sim_note"] = (
                        f"clip shorter than the {embedder.min_samples}-sample minimum the speaker "
                        f"embedder needs ({len(gen_speech[k])} generated / {len(ref_speech[k])} "
                        f"reference samples at {sim_sr} Hz); no SIM computed"
                    )
                    too_short.append((entries[i]["audio_filepath"], len(gen_speech[k])))

            if keep:
                gen_emb = embedder([gen_speech[k] for k in keep])
                ref_emb = embedder([ref_speech[k] for k in keep])
                sims = F.cosine_similarity(gen_emb, ref_emb, dim=-1)
                for k, sim in zip(keep, sims.tolist()):
                    entries[idxs[k]]["sim"] = round(float(sim), 6)
                    entries[idxs[k]]["sim_model"] = embedder.name
            done += len(idxs)
        rewrite_manifest()  # persist progress so a crash doesn't lose completed chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  scored {done}/{len(todo)}")

    if too_short:
        print(
            f"WARNING: {len(too_short)} row(s) were too short to embed (< {embedder.min_samples} "
            f"samples = {embedder.min_samples / sim_sr * 1000:.0f} ms) and have sim=null; the mean "
            f"SIM is over the remaining rows. Usually a degenerate/near-empty generation:"
        )
        for name, n in too_short[:10]:
            print(f"    {os.path.basename(name)}  ({n} samples)")

    if not todo and out_path != args.manifest_path:
        # Nothing to score, but still write the fork: it may not exist yet, or the rebuilt entries
        # may carry fresher upstream fields (e.g. re-transcribed `pred_text`).
        rewrite_manifest()

    print("Manifest updated with speaker similarity:", os.path.abspath(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=str, required=True, help="JSONL manifest to read samples from.")
    parser.add_argument(
        "--output_manifest_path",
        type=str,
        default=None,
        help="Where to write the scored manifest. Default: in place for the xvector backend, else alongside it as <manifest>_<sim_backend>.jsonl so both "
             "SIM scales coexist. Must stay in the manifest's directory — wav paths are relative to it.",
    )
    parser.add_argument(
        "--sim_backend",
        type=str,
        default="wavlm_seed_tts",
        choices=["xvector", "wavlm_seed_tts"],
        help="Speaker embedder. 'wavlm_seed_tts' (default): seed-tts-eval's WavLM-Large + "
             "ECAPA-TDNN — comparable with published SIM and far more discriminative. 'xvector': HF "
             "WavLMForXVector (--sim_model_id). The two scales are not interchangeable.",
    )
    parser.add_argument(
        "--sim_model_id",
        type=str,
        default=DEFAULT_XVECTOR_MODEL_ID,
        help=f"xvector backend only: speaker-verification model (default {DEFAULT_XVECTOR_MODEL_ID}).",
    )
    parser.add_argument(
        "--sim_ckpt",
        type=str,
        default=None,
        help=f"wavlm_seed_tts backend only: local path to {WAVLM_SEED_TTS_FILENAME}. "
             f"Default: download from the pinned Hub mirror ({WAVLM_SEED_TTS_REPO_ID}).",
    )
    parser.add_argument("--sim_ckpt_repo", type=str, default=WAVLM_SEED_TTS_REPO_ID, help="Hub repo holding the wavlm_seed_tts checkpoint.")
    parser.add_argument("--sim_ckpt_file", type=str, default=WAVLM_SEED_TTS_FILENAME, help="Filename of the wavlm_seed_tts checkpoint in --sim_ckpt_repo.")
    parser.add_argument(
        "--sim_ckpt_sha256",
        type=str,
        default=WAVLM_SEED_TTS_SHA256,
        help="Expected sha256 of the downloaded wavlm_seed_tts checkpoint ('' to skip the check).",
    )
    parser.add_argument("--wavlm_config_id", type=str, default=WAVLM_BACKBONE_CONFIG_ID, help="Repo to read the WavLM-Large config.json from (weights come from the checkpoint).")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Minibatch size for embedding extraction. Lower it if long clips OOM. Pinned to 1 for "
             "the wavlm_seed_tts backend, whose pooling is padding-sensitive.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=64,
        help="Rewrite the manifest every N scored rows (crash-resume granularity).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-score every row even if it already has a sim from this backend (default: resume).",
    )
    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=30.0,
        help="Truncate each clip to at most this many seconds before embedding (0 = no cap). "
             "Bounds GPU memory: WavLM OOMs on very long/runaway clips, and a speaker embedding "
             "needs only a few seconds of audio.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="'cuda', 'cpu', or 'cuda:0'.")
    parser.add_argument("--dtype", type=str, default="float32", help="Model dtype, e.g. 'float32'.")
    args = parser.parse_args()

    main(args)
