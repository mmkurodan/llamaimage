# NPU (Hexagon HTP) backend — plan

Status: preparation. CPU and GPU (OpenCL) use stable-diffusion.cpp; the NPU path is a separate
pipeline because the HTP cannot run ggml graphs.

## Architecture

```
prompt ─ CLIP BPE tokenizer (Java) ─ text_encoder.onnx ─┐
                                                         ├─ LCM loop (Java, N steps) ── unet.onnx (HTP)
init image (img2img) ─ taesd_encoder.onnx ─ add noise ──┘                │
                                         final denoised latent ─ taesd_decoder.onnx ─ RGB
```

- Runtime: ONNX Runtime Android with the QNN execution provider
  (`com.microsoft.onnxruntime:onnxruntime-android-qnn`, MIT) which pulls
  `com.qualcomm.qti:qnn-runtime` (libQnnHtp*, HTP skels V68–V85).
- ORT compiles each ONNX graph for the device's HTP on first use (libQnnHtpPrepare) and caches the
  QNN context (`ep.context_enable`) in the app's files dir — no per-SoC offline conversion.
- Static shapes: batch 1, 77 tokens, 512×512 (64×64 latents). Other sizes = other models.
- The UNet is split into three chained ONNX parts (`scripts/npu/unet_parts.py`: down+mid / up 0 /
  up 1–3+out, fp16 weights 697 / 324 / 698 MB), chained by tensor name in `NpuPipeline`. Compiling
  the whole 1.7 GB UNet on the phone used ~10 GB and the process was SIGKILLed (2026-11-25, SM8850);
  each part is now compiled, cached and closed before the next one (release `npu-models-v2`).
- Weights fp16, float32 I/O; HTP runs fp16 (`enable_htp_fp16_precision=1`). If fp16 is too slow or
  too large for the HTP, the next step is W8A16 quantization (calibration from the reference
  pipeline's activations).
- Samplers: LCM (LCM checkpoints) and, for regular SD1.5 checkpoints, Euler a / Euler / DPM++ 2M
  (`KDiffusionSampler`, a port of sd.cpp's CompVisDenoiser + DiscreteSchedule + img2img t_enc rule,
  so CPU and NPU settings mean the same) with CFG (two UNet passes per step). The templates fit any
  SD1.5 fp16 single file with the standard keys: AbsoluteReality 1.8.1, Photon v1 and Juggernaut
  Reborn were checked against every map key + shape from their safetensors headers (2026-11-25).
- Java side: tokenizer + LCM scheduler + loop + previews (latent projection, as on CPU) + cancel
  between steps. `scripts/npu/reference_pipeline.py` is the executable spec (golden token ids,
  timesteps, images).
- Fallback: if the QNN EP cannot be created or a graph fails to compile, use the CPU/GPU engine.

## Rights

| Component | License | Use |
|---|---|---|
| ONNX Runtime (Microsoft) | MIT | bundled |
| QNN runtime (`qnn-runtime` AAR) | QTI AI Stack License | bundled: §1(iv) allows distributing in object code *as incorporated in the app* (not standalone); no reverse engineering; keep notices (ship NOTICE.txt); revocable; indemnity; prohibited-use list (image generation not on it) |
| QAIRT SDK (if offline tools are needed) | same AI Stack License (identical LICENSE.pdf) | development only, not shipped |
| DreamShaper 8 LCM | CreativeML OpenRAIL-M | converted ONNX = derivative; distribute with the license + use restrictions |
| TAESD | MIT | converted ONNX |
| Local Dream (reference app) | CC BY-NC 4.0 | **not used** — no code copied; its converted models carry no license and are not used |

## Steps

1. ✅ License review (runtime, SDK, reference app).
2. ✅ `scripts/npu/export_onnx.py` + `reference_pipeline.py` + workflow `NPU models (ONNX export)`
   (manual): export + CPU reference images as artifacts.
3. Host the ONNX models (HF repo of the project, with LICENSE files) and add them to the catalog as
   an "NPU" model with URL + size + SHA-256.
4. App: add the ORT-QNN dependency (+ NOTICE), Java NPU pipeline, backend choice CPU / GPU / NPU,
   first-run compile progress ("NPU 用に最適化中…"), fallback.
5. Measure on the device (the DSP is not reachable from the proot dev shell — test in the app).

## Video (experimental): AnimateDiff-Lightning

Status: host-verified, device untested. Button 「NPU 動画（試験）」 in the 動画 tab.

- Model: SD1.5 checkpoint (the selected NPU-capable model) + AnimateDiff-Lightning 4-step motion
  module (ByteDance, CreativeML OpenRAIL-M, HF commit-pinned, 907 MB, downloaded by the app).
  Euler, "trailing" timesteps (999/749/499/249), **linear** betas, CFG 1 → one UNet pass per step.
- Static shape 8 frames × 256 px: sample `[8,4,32,32]` (frames = batch), timestep `[1]`,
  encoder_hidden_states `[1,77,768]` (broadcast inside). Same 3-part split and tensor names as the
  image UNet; fp16 weights 1103 / 541 / 981 MB (motion modules +906 MB in total).
- HTP-friendly export (`scripts/npu-video/animatediff_parts.py`): the motion module's 5-D
  reshapes/permutes rewritten at rank ≤ 4; attention as plain `softmax(q·s @ kᵀ) @ v` with a
  constant scale (no Shape-derived scale, no folded zero score tensor).
- On-device conversion like the image path: `vunet_N.template.onnx` + map (keys `motion:*` come
  from the motion file, the rest from the checkpoint); only the sinusoidal position tables
  `[1,8,C]` are inline. Release `npu-video-v1` (workflow "NPU video models").
- Only the UNet runs on the HTP; text encoder (shared with the image model dir) and a 256 px TAESD
  decoder run on the ORT CPU provider. The image model is released before the test.
- Host results (ORT-CPU, same ONNX): parts vs diffusers max diff 0.006 (fp16), ORT vs PyTorch
  cos 0.99998, 15 s/step on CPU. Device: compile memory (parts up to 1.1 GB) and s/step unknown.
