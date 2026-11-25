#!/usr/bin/env python3
"""Split exported ONNX models into a weightless template + a weight map, so the app can build the
NPU model on the device from an SD1.5 checkpoint it already downloaded (any SD1.5-architecture
single-file .safetensors: DreamShaper 8 LCM, AbsoluteReality LCM, Photon LCM, ...).

For every large initializer of the exported fp16 ONNX, find the checkpoint tensor it came from by
VALUE (fp16 bytes, as-is or transposed), so no hand-written name table is needed. Matched
initializers become external data in "<name>.weights" (written by the app, in map order); small
graph constants (shapes, timestep frequencies, ...) stay inline in the template.

Outputs per model: <name>.template.onnx, <name>.map.json
map.json: {"weights_file": ..., "total_bytes": N, "tensors": [{"name", "key", "transpose",
           "dims", "offset", "length"}]}   (all fp16, little endian)
"""
import argparse
import hashlib
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

MATCH_MIN_ELEMS = 64      # smaller ones (scalars, shapes) are graph constants, never looked up
MUST_MATCH_ELEMS = 4096   # a larger fp16 initializer that is not in the checkpoint is an error


def fp16_digest(a):
    return hashlib.sha1(np.ascontiguousarray(a.astype(np.float16)).tobytes()).hexdigest()


def load_checkpoint(path, prefixes):
    from safetensors.numpy import load_file
    sd = load_file(path)
    out = {}
    for k, v in sd.items():
        if any(k.startswith(p) for p in prefixes) and v.dtype in (np.float16, np.float32) and v.size >= MATCH_MIN_ELEMS:
            out[k] = v
    return out


def build_index(ckpt):
    idx = {}
    for k, v in ckpt.items():
        idx.setdefault(fp16_digest(v), []).append((k, False))
        if v.ndim == 2:
            idx.setdefault(fp16_digest(v.T), []).append((k, True))
        if v.ndim == 4 and v.shape[2] == 1 and v.shape[3] == 1:   # 1x1 conv stored as linear (or back)
            idx.setdefault(fp16_digest(v.reshape(v.shape[0], v.shape[1])), []).append((k, False))
            idx.setdefault(fp16_digest(v.reshape(v.shape[0], v.shape[1]).T), []).append((k, True))
    return idx


def make(onnx_path, ckpt, name, out_dir):
    model = onnx.load(onnx_path)
    idx = build_index(ckpt)
    used = set()
    tensors, offset, missing, inline, ambiguous = [], 0, [], [], 0
    for init in model.graph.initializer:
        if init.data_type != TensorProto.FLOAT16:
            continue
        arr = numpy_helper.to_array(init)
        if arr.size < MATCH_MIN_ELEMS:
            continue
        cands = [c for c in idx.get(fp16_digest(arr), []) if c[0] not in used] or idx.get(fp16_digest(arr), [])
        if not cands:
            (missing if arr.size >= MUST_MATCH_ELEMS else inline).append((init.name, list(arr.shape)))
            continue
        if len(set(c[0] for c in cands)) > 1:
            ambiguous += 1   # identical values in this checkpoint; first unused key is taken
        key, transpose = cands[0]
        used.add(key)
        length = arr.size * 2
        tensors.append({"name": init.name, "key": key, "transpose": transpose, "dims": list(arr.shape),
                        "offset": offset, "length": length})
        # turn into external data pointing at the app-written weights file
        init.ClearField("raw_data")
        del init.float_data[:]
        del init.int32_data[:]
        init.data_location = TensorProto.EXTERNAL
        del init.external_data[:]
        for k, v in (("location", f"{name}.weights"), ("offset", str(offset)), ("length", str(length))):
            e = init.external_data.add()
            e.key, e.value = k, v
        offset += length
    if inline:
        print(f"[template] {name}: {len(inline)} small fp16 constants kept inline: {inline[:20]}")
    if missing:
        raise SystemExit(f"[template] {name}: {len(missing)} initializers not found in checkpoint: {missing[:10]}")
    tpath = os.path.join(out_dir, f"{name}.template.onnx")
    onnx.save(model, tpath)
    with open(os.path.join(out_dir, f"{name}.map.json"), "w") as f:
        json.dump({"weights_file": f"{name}.weights", "total_bytes": offset, "dtype": "float16",
                   "tensors": tensors}, f)
    print(f"[template] {name}: {len(tensors)} external tensors, {offset / 1e6:.1f} MB weights, "
          f"template {os.path.getsize(tpath) / 1e6:.2f} MB, transposed {sum(t['transpose'] for t in tensors)}, "
          f"ambiguous {ambiguous}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="dir with unet_*.onnx / text_encoder.onnx (fp16)")
    ap.add_argument("--checkpoint", required=True, help="SD1.5 single-file .safetensors")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from unet_parts import NAMES as UNET_PARTS
    unet_ck = load_checkpoint(a.checkpoint, ["model.diffusion_model."])
    for name in UNET_PARTS:   # the UNet is exported in parts (unet_parts.py)
        make(os.path.join(a.models, f"{name}.onnx"), unet_ck, name, a.out)
    make(os.path.join(a.models, "text_encoder.onnx"),
         load_checkpoint(a.checkpoint, ["cond_stage_model.transformer."]), "text_encoder", a.out)


if __name__ == "__main__":
    main()
