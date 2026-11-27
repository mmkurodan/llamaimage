#!/usr/bin/env python3
"""Feasibility probe: AnimateDiff-Lightning (ByteDance, CreativeML OpenRAIL-M) on the NPU path.

1. Builds UNetMotionModel = SD1.5 single-file checkpoint + Lightning motion adapter (fp16).
2. Checks the rank-4 motion-module rewrite and the 3-part split against diffusers' own forward.
3. Exports the 3 parts as static-shape fp16 ONNX (frames x HxW fixed), prints sizes + op census,
   and compares ONNX Runtime (CPU) with PyTorch.
4. --generate: a whole Lightning txt2vid run through the ONNX parts (diffusers' Euler "trailing"
   scheduler, linear betas, CFG 1 = one UNet pass per step) + TAESD, written as a GIF, timed.

Memory: everything stays fp16; only the part being exported is temporarily fp32.
"""
import argparse
import collections
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "npu"))
import animatediff_parts as adp  # noqa: E402
import export_onnx  # noqa: E402
import fix_mixed_types  # noqa: E402

SD_CONFIG_REPO = "Lykon/dreamshaper-8-lcm"
SD_CONFIG_REV = "4645d8bc6a8e6b106d21606d63e8460cdad4f1a6"


def load(ckpt, adapter_path):
    from diffusers import MotionAdapter, StableDiffusionPipeline, UNetMotionModel
    from safetensors.torch import load_file
    pipe = StableDiffusionPipeline.from_single_file(ckpt, config=SD_CONFIG_REPO, revision=SD_CONFIG_REV,
                                                    torch_dtype=torch.float16, safety_checker=None,
                                                    load_safety_checker=False)
    adapter = MotionAdapter().to(torch.float16)
    missing = adapter.load_state_dict(load_file(adapter_path), strict=True)
    unet = UNetMotionModel.from_unet2d(pipe.unet, adapter).to(torch.float16).eval()
    te, tok = pipe.text_encoder.eval(), pipe.tokenizer
    del pipe, adapter
    return unet, te, tok


def _isolate_outputs(model):
    """Graph outputs that are also consumed inside the graph (r0.. skips) get their own Identity, so
    the fp16 converter's output Cast (keep_io_types) is not wired into the internal consumers too
    (onnxconverter-common 1.16 does that; ORT then rejects mixed float/float16 inputs)."""
    from onnx import helper
    outs = {o.name for o in model.graph.output}
    used = {i for n in model.graph.node for i in n.input}
    for o in outs & used:
        tmp = o + "__internal"
        for n in model.graph.node:
            n.input[:] = [tmp if i == o else i for i in n.input]
            n.output[:] = [tmp if x == o else x for x in n.output]
        model.graph.node.append(helper.make_node("Identity", [tmp], [o], name=o + "__out"))
    return model


def _install_converter_hook():
    from onnxconverter_common import float16
    orig = float16.convert_float_to_float16
    if getattr(orig, "_hooked", False):
        return

    def conv(model, *a, **k):
        return orig(_isolate_outputs(model), *a, **k)
    conv._hooked = True
    float16.convert_float_to_float16 = conv


def mb(mod):
    return sum(p.numel() * p.element_size() for p in mod.parameters()) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--prompt", default="a woman walking on a beach at sunset, waves, wind in her hair")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--budget-mb", type=float, default=adp.DEFAULT_BUDGET_MB,
                    help="max fp16 weights per part (layer granularity)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    F, L = a.frames, a.size // 8
    torch.set_num_threads(8)

    t0 = time.time()
    unet, te, tok = load(a.checkpoint, a.adapter)
    
    _install_converter_hook()
    print(f"[probe] loaded in {time.time() - t0:.0f}s; UNetMotionModel fp16 {mb(unet):.0f} MB "
          f"(motion modules {sum(mb(m) for n, m in unet.named_modules() if n.endswith('motion_modules')):.0f} MB)")

    # --- 2. correctness of the rewrite + split (tiny shape, fp32 compute on an fp16 model is too
    # big for RAM here, so compare in fp16; same ops -> differences are rounding only) ---
    g = torch.Generator().manual_seed(0)
    tf, tl = 4, 16
    xs = torch.randn(1, 4, tf, tl, tl, generator=g).half()
    ts = torch.tensor([749.0])
    eh = torch.randn(1, 77, 768, generator=g).half()
    with torch.no_grad():
        ref = unet(xs, ts, eh.repeat_interleave(tf, 0), return_dict=False)[0]      # [1,4,F,h,w]
        n = adp.patch_4d(unet)
        unet.set_attn_processor(adp.PlainAttnProcessor())   # ref above used diffusers' default (SDPA)
        x4 = xs[0].permute(1, 0, 2, 3).contiguous()                               # [F,4,h,w]
        _, chained = adp.run_chain(adp.parts(unet, tf, a.budget_mb), x4, ts, eh)
    ref4 = ref[0].permute(1, 0, 2, 3)
    d = (ref4 - chained).abs().max().item()
    print(f"[probe] patched {n} motion modules to rank<=4; max|diffusers(SDPA) - chained parts (rank-4 + plain attention)| = {d:.4g} "
          f"(|ref| max {ref4.abs().max().item():.3g})")
    if d > 2e-2:
        raise SystemExit("rewrite/split does not match diffusers")

    mods = adp.parts(unet, F, a.budget_mb)
    for nm, first, last, size in adp.describe(unet, a.budget_mb):
        print(f"[probe] {nm}: {first} .. {last}  {size} MB")
    names = [m[0] for m in mods]
    x = torch.randn(F, 4, L, L, generator=g).half()
    t = torch.tensor([999.0])
    ehs = torch.randn(1, 77, 768, generator=g).half()

    if not a.skip_export:
        import onnx
        # Each part: fp32 copy of just its modules -> export_onnx.export (fp32 trace, fp16 convert).
        args16, out16 = adp.run_chain(mods, x, t, ehs)
        for (name, mod, ins, outs, subs), pargs in zip(mods, args16):
            t1 = time.time()
            for sm in subs:
                sm.float()
            export_onnx.export(mod, tuple(p.float() for p in pargs), os.path.join(a.out, f"{name}.onnx"), ins, outs)
            for sm in subs:
                sm.half()
            fix_mixed_types.main_one(os.path.join(a.out, f"{name}.onnx"))
            m = onnx.load(os.path.join(a.out, f"{name}.onnx"), load_external_data=False)
            ops = collections.Counter(nd.op_type for nd in m.graph.node)
            print(f"[probe] {name}: export {time.time() - t1:.0f}s; ops {dict(ops.most_common())}")
            del m
        # The part modules share the unet; mod.float()/half() round-trips fp16 weights exactly.
        # TAESD decoder for this latent size (run per frame on the CPU provider in the app).
        from diffusers import AutoencoderTiny
        vae = AutoencoderTiny.from_pretrained(export_onnx.TAESD_REPO, revision=export_onnx.TAESD_REV,
                                              torch_dtype=torch.float32)
        tpath = os.path.join(a.out, "taesd_decoder_video.onnx")
        export_onnx.export(export_onnx.TaesdDecoder(vae), (torch.zeros(1, 4, L, L),), tpath, ["latents"], ["image"])
        fix_mixed_types.main_one(tpath)
        del vae

    # --- 3. ONNX Runtime CPU vs PyTorch at the export shape ---
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    # Host CPU only: the extended optimizer fuses Erf-GELU into com.microsoft.Gelu, which ORT-CPU
    # has no float16 kernel for (the HTP path does its own lowering).
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = {nm: ort.InferenceSession(os.path.join(a.out, f"{nm}.onnx"), so, providers=["CPUExecutionProvider"])
            for nm in names}
    _, ref = adp.run_chain(mods, x, t, ehs)
    t1 = time.time()
    got = run_onnx(sess, names, x.float().numpy(), t.numpy(), ehs.float().numpy())
    dt = time.time() - t1
    r = ref.float().numpy()
    cos = float((r * got).sum() / np.linalg.norm(r) / np.linalg.norm(got))
    print(f"[probe] ORT-CPU vs PyTorch fp16: max|d| {np.abs(r - got).max():.4g}, cos {cos:.6f}; "
          f"one UNet pass {F}x{a.size}^2 on ORT-CPU {dt:.1f}s")

    if a.generate:
        generate(a, sess, names, te, tok)


def run_onnx(sess, names, sample, timestep, hidden):
    env = {"sample": sample, "timestep": timestep.astype(np.float32), "encoder_hidden_states": hidden}
    for nm in names:
        s = sess[nm]
        outs = s.run(None, {i.name: env[i.name] for i in s.get_inputs()})
        env.update({o.name: v for o, v in zip(s.get_outputs(), outs)})
    return env["noise_pred"]


def generate(a, sess, names, te, tok):
    from diffusers import AutoencoderTiny, EulerDiscreteScheduler
    from PIL import Image
    F, L = a.frames, a.size // 8
    # Lightning's recommended scheduler: Euler, trailing spacing, LINEAR betas (not scaled_linear).
    sch = EulerDiscreteScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                                 beta_schedule="linear", timestep_spacing="trailing", steps_offset=1,
                                 prediction_type="epsilon")
    sch.set_timesteps(a.steps)
    print(f"[gen] timesteps {sch.timesteps.tolist()} sigmas {[round(s, 4) for s in sch.sigmas.tolist()]}")
    ids = tok(a.prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    with torch.no_grad():
        hidden = te(ids, return_dict=False)[0].float().numpy()
    g = torch.Generator().manual_seed(a.seed)
    lat = torch.randn(1, 4, F, L, L, generator=g)[0].permute(1, 0, 2, 3).contiguous() * sch.init_noise_sigma
    t0 = time.time()
    for i, ts in enumerate(sch.timesteps):
        inp = sch.scale_model_input(lat, ts)
        t1 = time.time()
        eps = torch.from_numpy(run_onnx(sess, names, inp.numpy(), np.array([float(ts)], np.float32), hidden))
        lat = sch.step(eps, ts, lat).prev_sample
        print(f"[gen] step {i + 1}/{a.steps} t={int(ts)} {time.time() - t1:.1f}s")
    print(f"[gen] sampling {time.time() - t0:.1f}s")
    vae = AutoencoderTiny.from_pretrained(export_onnx.TAESD_REPO, revision=export_onnx.TAESD_REV)
    t1 = time.time()
    with torch.no_grad():
        img = (vae.decode(lat, return_dict=False)[0] * 0.5 + 0.5).clamp(0, 1)
    print(f"[gen] TAESD decode {F} frames {time.time() - t1:.1f}s")
    frames = [Image.fromarray((f.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)) for f in img]
    path = os.path.join(a.out, f"lightning_{F}f_{a.size}_{a.steps}step_s{a.seed}.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=125, loop=0)
    w = a.size
    sheet = Image.new("RGB", (w * min(F, 8), w * ((F + 7) // 8)))
    for k, fr in enumerate(frames):
        sheet.paste(fr, ((k % 8) * w, (k // 8) * w))
    sheet.save(path.replace(".gif", "_sheet.png"))
    print(f"[gen] wrote {path}")


if __name__ == "__main__":
    main()
