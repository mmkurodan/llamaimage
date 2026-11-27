#!/usr/bin/env python3
"""SD1.5 UNet + AnimateDiff motion module (diffusers UNetMotionModel), split into chained parts for
the NPU (ONNX Runtime QNN EP), like unet_parts.py does for the image UNet.

Static shapes: the F frames are the batch (sample [F,4,h,w]); timestep [1] and
encoder_hidden_states [1,77,768] are shared by all frames and broadcast inside the parts.

HTP friendliness: diffusers' motion module (AnimateDiffTransformer3D) goes through 5-D tensors
([B,C,F,H,W] for the GroupNorm, [B,H,W,F,C] permutes). `patch_4d` replaces its forward with an
equivalent that stays at rank <= 4 (B = 1 is fixed), checked against the original in the probe.

Parts: the UNet is flattened into LAYERS (conv_in+time embedding, every resnet / spatial
attention / motion module / down- or upsampler, the mid block's layers, conv_out) and greedily
grouped into parts of at most `budget_mb` fp16 weights. Compiling a 1.1 GB part on the phone killed
the app (2026-11-27), and a single up block is 733 MB, hence layer granularity. Tensors between
parts are chained by NAME (the app does the same): the running hidden state "h<k>" (k = index of
the first layer of the next part), "emb", the skip tensors "r<j>" still needed later, and the
shared inputs "timestep"/"encoder_hidden_states". The first part takes "sample", the last returns
"noise_pred". Part names: unet_0 .. unet_{n-1}.
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "npu"))
import unet_parts  # noqa: E402,F401  (kept for path setup symmetry)

DEFAULT_BUDGET_MB = 450


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


class TemporalElementwiseAttnProcessor:
    """Motion-module (temporal) self-attention without MatMul: the sequence is the F frames of one
    pixel (F = 8), the batch is H*W pixels. As a batched MatMul that is H*W*heads tiny F x F
    products (8192 at the 256 px top level), and the HTP compile of that one 5 MB layer grew
    > 5 GB on the device (2026-11-27). Same math with F Mul + ReduceSum for the scores and F
    Mul + Add for the output, all rank <= 4:
        s[p,i,h,j] = sum_d q[p,i,h,d] * k[p,j,h,d] * scale ;  w = softmax_j(s)
        o[p,i,h,:] = sum_j w[p,i,h,j] * v[p,j,h,:]
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 *args, **kwargs):
        assert attention_mask is None and attn.spatial_norm is None and attn.group_norm is None
        ctx = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q, k, v = attn.to_q(hidden_states), attn.to_k(ctx), attn.to_v(ctx)
        p, n, inner = q.shape
        m, hd = ctx.shape[1], attn.heads
        d = inner // hd
        q = (q * attn.scale).reshape(p, n, hd, d)                       # [p, n, h, d]
        k = k.reshape(p, m, hd, d)
        v = v.reshape(p, m, hd, d)
        s = torch.cat([(q * k[:, j:j + 1]).sum(dim=-1, keepdim=True) for j in range(m)], dim=-1)  # [p,n,h,m]
        w = torch.softmax(s, dim=-1)
        o = w[..., 0:1] * v[:, 0:1]
        for j in range(1, m):
            o = o + w[..., j:j + 1] * v[:, j:j + 1]                     # [p, n, h, d]
        o = o.reshape(p, n, inner)
        o = attn.to_out[1](attn.to_out[0](o))
        if attn.residual_connection:
            o = o + hidden_states
        return o / attn.rescale_output_factor


def patch_temporal_attention(unet):
    """Motion-module attentions -> TemporalElementwiseAttnProcessor (after set_attn_processor)."""
    from diffusers.models.unets.unet_motion_model import AnimateDiffTransformer3D
    n = 0
    for mod in unet.modules():
        if isinstance(mod, AnimateDiffTransformer3D):
            for sub in mod.modules():
                if hasattr(sub, "set_processor") and hasattr(sub, "to_q"):
                    sub.set_processor(TemporalElementwiseAttnProcessor())
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


class Layer:
    """One step of the flattened UNet. fn(h, ctx) -> h; may push or pop a skip tensor."""

    def __init__(self, name, mods, fn, push=False, pop=False, needs_ehs=False):
        self.name, self.mods, self.fn = name, [m for m in mods if m is not None], fn
        self.push, self.pop, self.needs_ehs = push, pop, needs_ehs
        self.skip = None          # index of the skip tensor pushed / popped (filled by layers())

    def mb(self):
        return sum(p.numel() * p.element_size() for m in self.mods for p in m.parameters()) / 1e6


def layers(unet):
    """The UNet as a flat list of Layers (pre-processing is layer 0, conv_out the last)."""
    u = unet
    out = [Layer("pre", [u.time_proj, u.time_embedding, u.conv_in], None, push=True)]

    def res(r):
        return lambda h, c: r(input_tensor=h, temb=c["emb"])

    def att(a):
        return lambda h, c: a(h, encoder_hidden_states=c["ehs"], return_dict=False)[0]

    def mm(m):
        return lambda h, c: m(h, num_frames=c["frames"])

    for i, b in enumerate(u.down_blocks):
        atts = getattr(b, "attentions", None) or [None] * len(b.resnets)
        for j, (r, a, m) in enumerate(zip(b.resnets, atts, b.motion_modules)):
            out.append(Layer(f"d{i}.res{j}", [r], res(r)))
            if a is not None:
                out.append(Layer(f"d{i}.attn{j}", [a], att(a), needs_ehs=True))
            out.append(Layer(f"d{i}.mm{j}", [m], mm(m), push=True))
        if b.downsamplers is not None:
            ds = b.downsamplers[0]
            out.append(Layer(f"d{i}.down", [ds], lambda h, c, ds=ds: ds(hidden_states=h), push=True))
    mb = u.mid_block
    out.append(Layer("mid.res0", [mb.resnets[0]], res(mb.resnets[0])))
    for j, (a, r, m) in enumerate(zip(mb.attentions, mb.resnets[1:], mb.motion_modules)):
        out.append(Layer(f"mid.attn{j}", [a], att(a), needs_ehs=True))
        out.append(Layer(f"mid.mm{j}", [m], mm(m)))
        out.append(Layer(f"mid.res{j + 1}", [r], res(r)))
    for i, b in enumerate(u.up_blocks):
        atts = getattr(b, "attentions", None) or [None] * len(b.resnets)
        for j, (r, a, m) in enumerate(zip(b.resnets, atts, b.motion_modules)):
            out.append(Layer(f"u{i}.res{j}", [r],
                             lambda h, c, r=r: r(input_tensor=torch.cat([h, c["skip"]], dim=1), temb=c["emb"]),
                             pop=True))
            if a is not None:
                out.append(Layer(f"u{i}.attn{j}", [a], att(a), needs_ehs=True))
            out.append(Layer(f"u{i}.mm{j}", [m], mm(m)))
        if b.upsamplers is not None:
            us = b.upsamplers[0]
            out.append(Layer(f"u{i}.up", [us], lambda h, c, us=us: us(hidden_states=h)))
    out.append(Layer("out", [u.conv_norm_out, u.conv_out], lambda h, c: u.conv_out(u.conv_act(u.conv_norm_out(h)))))
    stack, n = [], 0
    for L in out:                 # skip indices: pushed in order, popped from the end (diffusers)
        if L.push:
            L.skip = n
            stack.append(n)
            n += 1
        if L.pop:
            L.skip = stack.pop()
    assert not stack, stack
    return out


def group(lays, budget_mb):
    """[(first, last)) layer ranges, each <= budget_mb unless a single layer is larger."""
    ranges, start, acc = [], 0, 0.0
    for k, L in enumerate(lays):
        m = L.mb()
        if k > start and acc + m > budget_mb:
            ranges.append((start, k))
            start, acc = k, 0.0
        acc += m
    ranges.append((start, len(lays)))
    return ranges


class Part(torch.nn.Module):
    def __init__(self, unet, lays, first, last, frames):
        super().__init__()
        self.unet, self.lays, self.first, self.last, self.frames = unet, lays, first, last, frames
        mine = lays[first:last]
        pushed_here = {L.skip for L in mine if L.push}
        popped_later = {L.skip for L in lays[last:] if L.pop}
        self.skips_in = sorted(L.skip for L in mine if L.pop and L.skip not in pushed_here)
        self.skips_out = sorted(j for j in pushed_here if j in popped_later)
        self.ehs = any(L.needs_ehs for L in mine)
        later_emb = last < len(lays)
        if first == 0:
            self.input_names = ["sample", "timestep"] + (["encoder_hidden_states"] if self.ehs else [])
        else:
            uses_emb = any(".res" in L.name for L in mine)
            self.input_names = [f"h{first}"] + (["emb"] if uses_emb else []) \
                + (["encoder_hidden_states"] if self.ehs else [])
        self.input_names += [f"r{j}" for j in self.skips_in]
        if last == len(lays):
            self.output_names = ["noise_pred"]
        else:
            self.output_names = [f"h{last}"] + (["emb"] if first == 0 and later_emb else []) \
                + [f"r{j}" for j in self.skips_out]

    def forward(self, *args):
        u, f = self.unet, self.frames
        a = dict(zip(self.input_names, args))
        c = {"frames": f, "ehs": a["encoder_hidden_states"].expand(f, -1, -1) if self.ehs else None}
        skips = {j: a[f"r{j}"] for j in self.skips_in}
        if self.first == 0:
            sample = a["sample"]
            emb = u.time_embedding(u.time_proj(a["timestep"]).to(sample.dtype))      # [1,1280]
            c["emb"] = emb.expand(f, emb.shape[1])
            h = u.conv_in(sample)
            skips[0] = h
            k0 = 1
        else:
            h, c["emb"], k0 = a[f"h{self.first}"], a.get("emb"), self.first
        for L in self.lays[k0:self.last]:
            if L.pop:
                c["skip"] = skips.pop(L.skip)
            h = L.fn(h, c)
            if L.push:
                skips[L.skip] = h
        if self.last == len(self.lays):
            return h
        outs = [h] + ([c["emb"]] if "emb" in self.output_names else []) + [skips[j] for j in self.skips_out]
        return tuple(outs)


def parts(unet, frames, budget_mb=DEFAULT_BUDGET_MB):
    """[(name, module, input_names, output_names, submodules)] in execution order."""
    lays = layers(unet)
    out = []
    for k, (a, b) in enumerate(group(lays, budget_mb)):
        p = Part(unet, lays, a, b, frames)
        subs = [m for L in lays[a:b] for m in L.mods]
        out.append((f"unet_{k}", p, p.input_names, p.output_names, subs))
    return out


def describe(unet, budget_mb=DEFAULT_BUDGET_MB):
    lays = layers(unet)
    return [(f"unet_{k}", lays[a].name, lays[b - 1].name, round(sum(L.mb() for L in lays[a:b])))
            for k, (a, b) in enumerate(group(lays, budget_mb))]


def run_chain(mods, sample, timestep, hidden):
    """PyTorch chain; returns (per-part input tuples, noise_pred)."""
    env = {"sample": sample, "timestep": timestep, "encoder_hidden_states": hidden}
    args = []
    with torch.no_grad():
        for name, mod, ins, outs, _ in mods:
            a = tuple(env[i] for i in ins)
            args.append(a)
            y = mod(*a)
            y = y if isinstance(y, tuple) else (y,)
            env.update(zip(outs, y))
    return args, env["noise_pred"]
