#!/usr/bin/env python3
"""Compare each exported ONNX model with its PyTorch fp32 original (same inputs), and render the
diffusers reference image (fp32, LCMScheduler, full VAE and TAESD) for the same prompt/seed."""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from unet_parts import NAMES as UNET_PARTS, run_unet

DS_REPO = "Lykon/dreamshaper-8-lcm"
DS_REV = "4645d8bc6a8e6b106d21606d63e8460cdad4f1a6"
TAESD_REPO = "madebyollin/taesd"
TAESD_REV = "614f76814bbe30edbe2e627ace1c2234c81a2c0e"


def stats(name, ref, got):
    ref, got = ref.astype(np.float64).ravel(), got.astype(np.float64).ravel()
    cos = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))
    print(f"{name:14s} cos={cos:.6f} max_abs={np.abs(ref - got).max():.4f} "
          f"ref_std={ref.std():.4f} got_std={got.std():.4f} nan={int(np.isnan(got).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="a photo of a cat sitting on a wooden table, soft light")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from diffusers import AutoencoderTiny, LCMScheduler, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(DS_REPO, revision=DS_REV, torch_dtype=torch.float32,
                                                   safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    sess = lambda n: ort.InferenceSession(os.path.join(a.models, n), providers=["CPUExecutionProvider"])

    ids = pipe.tokenizer(a.prompt, padding="max_length", max_length=77, truncation=True,
                         return_tensors="pt").input_ids
    with torch.no_grad():
        hid_ref = pipe.text_encoder(ids)[0].numpy()
    hid = sess("text_encoder.onnx").run(None, {"input_ids": ids.numpy().astype(np.int32)})[0]
    stats("text_encoder", hid_ref, hid)

    unet = {n: sess(f"{n}.onnx") for n in UNET_PARTS}
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 4, 64, 64)).astype(np.float32)
    for t in (999.0, 499.0, 99.0):
        with torch.no_grad():
            eps_ref = pipe.unet(torch.from_numpy(x), torch.tensor([t]), torch.from_numpy(hid_ref)).sample.numpy()
        eps = run_unet(unet, x, np.array([t], np.float32), hid_ref)
        stats(f"unet t={int(t)}", eps_ref, eps)

    tae = AutoencoderTiny.from_pretrained(TAESD_REPO, revision=TAESD_REV, torch_dtype=torch.float32)
    lat = (rng.standard_normal((1, 4, 64, 64)) * 0.8).astype(np.float32)
    with torch.no_grad():
        img_ref = (tae.decode(torch.from_numpy(lat)).sample * 0.5 + 0.5).clamp(0, 1).numpy()
    stats("taesd_decoder", img_ref, sess("taesd_decoder.onnx").run(None, {"latents": lat})[0])

    g = torch.Generator().manual_seed(42)
    img = pipe(a.prompt, num_inference_steps=6, guidance_scale=1.0, generator=g, height=512, width=512).images[0]
    img.save(os.path.join(a.out, "diffusers_fp32_vae.png"))
    pipe.vae = tae
    g = torch.Generator().manual_seed(42)
    pipe(a.prompt, num_inference_steps=6, guidance_scale=1.0, generator=g).images[0].save(
        os.path.join(a.out, "diffusers_fp32_taesd.png"))
    print("scheduler timesteps:", [int(t) for t in pipe.scheduler.timesteps])
    print("scheduler config:", json.dumps({k: v for k, v in pipe.scheduler.config.items()
                                            if not k.startswith("_")}))


if __name__ == "__main__":
    main()
