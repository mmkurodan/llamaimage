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
- Weights fp16, float32 I/O; HTP runs fp16 (`enable_htp_fp16_precision=1`). If fp16 is too slow or
  too large for the HTP, the next step is W8A16 quantization (calibration from the reference
  pipeline's activations).
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
