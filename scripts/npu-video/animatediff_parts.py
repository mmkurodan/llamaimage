#!/usr/bin/env python3
"""SD1.5 UNet + AnimateDiff motion module (diffusers UNetMotionModel), split into chained parts for
the NPU (ONNX Runtime QNN EP), like unet_parts.py does for the image UNet.

Static shapes: the F frames are the batch (sample [F,4,h,w]); timestep [1] and
encoder_hidden_states [1,77,768] are shared by all frames and broadcast inside the parts.

HTP friendliness: diffusers' motion module (AnimateDiffTransformer3D) goes through 5-D tensors
([B,C,F,H,W] for the GroupNorm, [B,H,W,F,C] permutes). `patch_4d` replaces its forward with an
equivalent that stays at rank <= 4 (B = 1 is fixed), checked against the original in the probe.

Parts (same split points as unet_parts.py, so the app's name-chaining works unchanged):
  unet_0  sample, timestep, encoder_hidden_states -> h0, emb, r0..r11   (down blocks + mid block)
  unet_1  h0, emb, r9..r11                       -> h1                 (up block 0)
  unet_2  h1, emb, encoder_hidden_states, r0..r8  -> noise_pred         (up blocks 1-3, conv_out)
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "npu"))
import unet_parts  # noqa: E402

NAMES = unet_parts.NAMES


def _temporal_forward_4d(self, hidden_states, encoder_hidden_states=None, timestep=None,
                         class_labels=None, num_frames=1, cross_attention_kwargs=None):
    # hidden_states [F,C,H,W] (batch 1). Same math as AnimateDiffTransformer3D.forward.
    f, c, h, w = hidden_states.shape
    residual = hidden_states
    # GroupNorm over (C groups) x (F,H,W): as [1, C, F*H*W]
    x = hidden_states.permute(1, 0, 2, 3).reshape(1, c, f * h * w)
    x = self.norm(x)
    # -> [H*W, F, C]
    x = x.reshape(c, f, h * w).permute(2, 1, 0)
    x = self.proj_in(input=x)
    for block in self.transformer_blocks:
        x = block(hidden_states=x, encoder_hidden_states=encoder_hidden_states, timestep=timestep,
                  cross_attention_kwargs=cross_attention_kwargs, class_labels=class_labels)
    x = self.proj_out(input=x)                       # [H*W, F, C]
    x = x.permute(1, 2, 0).reshape(f, c, h, w)
    return x + residual


def patch_4d(unet):
    from diffusers.models.unets.unet_motion_model import AnimateDiffTransformer3D
    n = 0
    for m in unet.modules():
        if isinstance(m, AnimateDiffTransformer3D):
            m.forward = types.MethodType(_temporal_forward_4d, m)
            n += 1
    return n


class PlainAttnProcessor:
    """softmax((q * scale) @ k^T) @ v with a constant scale and rank <= 4 tensors: no Shape-derived
    1/sqrt(d) (SDPA export) and no zero score tensor (AttnProcessor's baddbmm folds into a
    [B*heads, N, M] constant -- 134 MB per attention at 256 px)."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 *args, **kwargs):
        assert attention_mask is None and attn.spatial_norm is None and attn.group_norm is None
        ctx = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q, k, v = attn.to_q(hidden_states), attn.to_k(ctx), attn.to_v(ctx)
        b, n, inner = q.shape
        m, hd = ctx.shape[1], attn.heads
        d = inner // hd
        q = q.reshape(b, n, hd, d).transpose(1, 2) * attn.scale         # [b,h,n,d]
        k = k.reshape(b, m, hd, d).permute(0, 2, 3, 1)                  # [b,h,d,m]
        v = v.reshape(b, m, hd, d).transpose(1, 2)                      # [b,h,m,d]
        o = torch.matmul(torch.softmax(torch.matmul(q, k), dim=-1), v)
        o = o.transpose(1, 2).reshape(b, n, inner)
        o = attn.to_out[1](attn.to_out[0](o))
        if attn.residual_connection:
            o = o + hidden_states
        return o / attn.rescale_output_factor


def _blk(b, **kw):
    if getattr(b, "has_cross_attention", False) and b.has_cross_attention:
        return b(**kw)
    kw.pop("encoder_hidden_states", None)
    return b(**kw)


class Part0(torch.nn.Module):
    def __init__(self, unet, frames):
        super().__init__()
        self.unet, self.frames = unet, frames

    def forward(self, sample, timestep, encoder_hidden_states):
        u, f = self.unet, self.frames
        emb = u.time_embedding(u.time_proj(timestep).to(sample.dtype))      # [1,1280]
        emb = emb.expand(f, emb.shape[1])
        ehs = encoder_hidden_states.expand(f, -1, -1)
        h = u.conv_in(sample)
        res = [h]
        for b in u.down_blocks:
            h, r = _blk(b, hidden_states=h, temb=emb, encoder_hidden_states=ehs, num_frames=f)
            res.extend(r)
        h = u.mid_block(h, emb, encoder_hidden_states=ehs, num_frames=f)
        return (h, emb, *res)


class UpPart(unet_parts.UpPart):
    def __init__(self, unet, first, last, frames):
        super().__init__(unet, first, last)
        self.frames = frames

    def forward(self, h, emb, *rest):
        u, f = self.unet, self.frames
        ehs = rest[0].expand(f, -1, -1) if self.cross else None
        res = dict(zip(self.skips, rest[1:] if self.cross else rest))
        for i, rng in zip(range(self.first, self.last), self.ranges):
            h = _blk(u.up_blocks[i], hidden_states=h, temb=emb, res_hidden_states_tuple=tuple(res[j] for j in rng),
                     encoder_hidden_states=ehs, num_frames=f)
        if self.last == len(u.up_blocks):
            if u.conv_norm_out is not None:
                h = u.conv_act(u.conv_norm_out(h))
            h = u.conv_out(h)
        return h


def parts(unet, frames):
    n, _ = unet_parts._skip_ranges(unet)
    s = unet_parts.UP_SPLIT
    p1, p2 = UpPart(unet, 0, s, frames), UpPart(unet, s, len(unet.up_blocks), frames)
    return [
        ("unet_0", Part0(unet, frames), ["sample", "timestep", "encoder_hidden_states"],
         ["h0", "emb"] + [f"r{j}" for j in range(n)]),
        ("unet_1", p1, p1.input_names("h0"), ["h1"]),
        ("unet_2", p2, p2.input_names("h1"), ["noise_pred"]),
    ]


def submodules(unet, name):
    """The modules a part actually runs (to cast just those to fp32 for export)."""
    s = unet_parts.UP_SPLIT
    if name == "unet_0":
        return [unet.time_proj, unet.time_embedding, unet.conv_in, *unet.down_blocks, unet.mid_block]
    if name == "unet_1":
        return list(unet.up_blocks[:s])
    return [*unet.up_blocks[s:], unet.conv_norm_out, unet.conv_out]


def run_chain(mods, sample, timestep, hidden):
    """PyTorch chain; returns (per-part input tuples, noise_pred)."""
    env = {"sample": sample, "timestep": timestep, "encoder_hidden_states": hidden}
    args = []
    with torch.no_grad():
        for name, mod, ins, outs in mods:
            a = tuple(env[i] for i in ins)
            args.append(a)
            y = mod(*a)
            y = y if isinstance(y, tuple) else (y,)
            env.update(zip(outs, y))
    return args, env["noise_pred"]
