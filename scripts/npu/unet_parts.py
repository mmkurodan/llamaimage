#!/usr/bin/env python3
"""The SD1.5 UNet split into consecutive parts, each exported as its own ONNX model.

Compiling the whole UNet (1.7 GB fp16) for the HTP on the phone needs ~10 GB of RAM and the app
was killed (SIGKILL) near the end. Each part is compiled (and cached) on its own, so the peak is
roughly one part's worth.

Parts (SD1.5: 4 down blocks, mid block, 4 up blocks; 12 skip tensors r0..r11):
  unet_0  sample, timestep, encoder_hidden_states -> h0, emb, r0..r11   (conv_in, time embedding,
                                                                          down blocks, mid block)
  unet_1  h0, emb, r9..r11                       -> h1                 (up block 0, no attention)
  unet_2  h1, emb, encoder_hidden_states, r0..r8  -> noise_pred         (up blocks 1-3, conv_out)

The app chains them by tensor NAME: every part's inputs are taken from the original inputs or from
earlier parts' outputs (see run_unet), so the split points can change without app changes as long
as NAMES (below) stays in sync with NpuPipeline.UNET_PARTS.
"""
import torch

NAMES = ["unet_0", "unet_1", "unet_2"]
UP_SPLIT = 1          # up blocks [0, UP_SPLIT) in unet_1, the rest in unet_2
# fp16 weights per part (SD1.5): unet_0 697 MB, unet_1 324 MB, unet_2 698 MB


def _skip_ranges(unet):
    """Skip tensor indices consumed by each up block (taken from the end, as diffusers does)."""
    n = 1 + sum(len(b.resnets) + (1 if b.downsamplers else 0) for b in unet.down_blocks)
    out, end = [], n
    for up in unet.up_blocks:
        k = len(up.resnets)
        out.append(list(range(end - k, end)))
        end -= k
    assert end == 0, end
    return n, out


def _up(unet, i, h, emb, ehs, res):
    up = unet.up_blocks[i]
    if getattr(up, "has_cross_attention", False) and up.has_cross_attention:
        return up(hidden_states=h, temb=emb, res_hidden_states_tuple=tuple(res), encoder_hidden_states=ehs)
    return up(hidden_states=h, temb=emb, res_hidden_states_tuple=tuple(res))


class Part0(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        u = self.unet
        t = timestep.expand(sample.shape[0])
        emb = u.time_embedding(u.time_proj(t).to(sample.dtype))
        h = u.conv_in(sample)
        res = [h]
        for b in u.down_blocks:
            if getattr(b, "has_cross_attention", False) and b.has_cross_attention:
                h, r = b(hidden_states=h, temb=emb, encoder_hidden_states=encoder_hidden_states)
            else:
                h, r = b(hidden_states=h, temb=emb)
            res.extend(r)
        h = u.mid_block(h, emb, encoder_hidden_states=encoder_hidden_states)
        return (h, emb, *res)


class UpPart(torch.nn.Module):
    """Up blocks [first, last) (+ conv_out if last is the end). Inputs: h, emb, [ehs], r[skips]."""

    def __init__(self, unet, first, last):
        super().__init__()
        self.unet, self.first, self.last = unet, first, last
        _, ranges = _skip_ranges(unet)
        self.ranges = ranges[first:last]
        self.skips = sorted(j for r in self.ranges for j in r)
        self.cross = any(getattr(unet.up_blocks[i], "has_cross_attention", False) for i in range(first, last))

    def input_names(self, h):
        return [h, "emb"] + (["encoder_hidden_states"] if self.cross else []) + [f"r{j}" for j in self.skips]

    def forward(self, h, emb, *rest):
        u = self.unet
        ehs = rest[0] if self.cross else None
        res = dict(zip(self.skips, rest[1:] if self.cross else rest))
        for i, rng in zip(range(self.first, self.last), self.ranges):
            h = _up(u, i, h, emb, ehs, [res[j] for j in rng])
        if self.last == len(u.up_blocks):
            if u.conv_norm_out is not None:
                h = u.conv_act(u.conv_norm_out(h))
            h = u.conv_out(h)
        return h


def parts(unet):
    """[(name, module, input_names, output_names)] in execution order."""
    n, _ = _skip_ranges(unet)
    p1, p2 = UpPart(unet, 0, UP_SPLIT), UpPart(unet, UP_SPLIT, len(unet.up_blocks))
    return [
        ("unet_0", Part0(unet), ["sample", "timestep", "encoder_hidden_states"],
         ["h0", "emb"] + [f"r{j}" for j in range(n)]),
        ("unet_1", p1, p1.input_names("h0"), ["h1"]),
        ("unet_2", p2, p2.input_names("h1"), ["noise_pred"]),
    ]


def run_unet(sessions, sample, timestep, hidden):
    """Runs the chained ONNX parts ({name: onnxruntime session}); returns noise_pred."""
    env = {"sample": sample, "timestep": timestep, "encoder_hidden_states": hidden}
    for name in NAMES:
        s = sessions[name]
        outs = s.run(None, {i.name: env[i.name] for i in s.get_inputs()})
        env.update({o.name: v for o, v in zip(s.get_outputs(), outs)})
    return env["noise_pred"]


def example_inputs(unet, sample, timestep, hidden):
    """PyTorch inputs for each part (runs the chain), for torch.onnx.export."""
    out = []
    env = {"sample": sample, "timestep": timestep, "encoder_hidden_states": hidden}
    with torch.no_grad():
        for name, mod, ins, outs in parts(unet):
            args = tuple(env[i] for i in ins)
            out.append(args)
            y = mod(*args)
            y = y if isinstance(y, tuple) else (y,)
            env.update(zip(outs, y))
    return out, env["noise_pred"]
