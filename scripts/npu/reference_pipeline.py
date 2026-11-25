#!/usr/bin/env python3
"""Reference LCM txt2img / img2img over the exported ONNX models (ONNX Runtime, CPU).

Checks that the export is correct (writes PNGs) and is the spec the app's NPU pipeline follows:
CLIP tokenization (77 tokens, BOS/EOS, pad with EOS), LCM scheduler (diffusers LCMScheduler,
original_inference_steps=50), CFG 1 (no unconditional pass), TAESD decode of the final denoised
latent. Run with --provider QNNExecutionProvider on a Snapdragon to try the HTP.
"""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image


def session(path, provider):
    opts = ort.SessionOptions()
    if provider == "QNNExecutionProvider":
        return ort.InferenceSession(path, opts, providers=[(provider, {
            "backend_path": "libQnnHtp.so", "enable_htp_fp16_precision": "1", "htp_performance_mode": "burst"})])
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="a photo of a cat sitting on a wooden table, soft light")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from diffusers import LCMScheduler
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer(os.path.join(a.models, "tokenizer", "vocab.json"),
                        os.path.join(a.models, "tokenizer", "merges.txt"))
    ids = tok(a.prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np").input_ids
    np.save(os.path.join(a.out, "token_ids.npy"), ids.astype(np.int32))   # golden data for the app tokenizer

    t0 = time.time()
    te = session(os.path.join(a.models, "text_encoder.onnx"), a.provider)
    unet = session(os.path.join(a.models, "unet.onnx"), a.provider)
    dec = session(os.path.join(a.models, "taesd_decoder.onnx"), a.provider)
    enc = session(os.path.join(a.models, "taesd_encoder.onnx"), a.provider)
    print(f"sessions: {time.time() - t0:.1f}s")

    hidden = te.run(None, {"input_ids": ids.astype(np.int32)})[0]
    import json
    with open(os.path.join(a.models, "scheduler_config.json")) as f:
        sched = LCMScheduler.from_config(json.load(f))   # PNDM config of the repo -> LCM

    def run(init_image=None):
        rng = np.random.default_rng(a.seed)
        sched.set_timesteps(a.steps, original_inference_steps=50,
                            strength=a.strength if init_image is not None else 1.0)
        print(("img2img" if init_image is not None else "txt2img") + " timesteps:",
              [int(t) for t in sched.timesteps])
        noise = rng.standard_normal((1, 4, 64, 64)).astype(np.float32)
        if init_image is None:
            latents = noise * float(sched.init_noise_sigma)
        else:
            import torch
            x0 = enc.run(None, {"image": init_image})[0]
            latents = sched.add_noise(torch.from_numpy(x0), torch.from_numpy(noise),
                                      sched.timesteps[:1]).numpy()
        import torch
        denoised = None
        # ONE generator for the whole run: LCM re-noises between steps, and re-seeding per step
        # injected the same noise every step (over-sharpened, grainy images).
        gen = torch.Generator().manual_seed(a.seed)
        for t in sched.timesteps:
            t1 = time.time()
            eps = unet.run(None, {"sample": latents, "timestep": np.array([float(t)], np.float32),
                                  "encoder_hidden_states": hidden})[0]
            out = sched.step(torch.from_numpy(eps), t, torch.from_numpy(latents), generator=gen)
            latents, denoised = out.prev_sample.numpy(), out.denoised.numpy()
            print(f"  t={int(t)} {time.time() - t1:.2f}s")
        return dec.run(None, {"latents": denoised.astype(np.float32)})[0]

    img = run()
    Image.fromarray((img[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)).save(os.path.join(a.out, "txt2img.png"))
    img2 = run(img.astype(np.float32))
    Image.fromarray((img2[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)).save(os.path.join(a.out, "img2img.png"))
    print("timesteps:", [int(t) for t in sched.timesteps])


if __name__ == "__main__":
    main()
