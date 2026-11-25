#!/usr/bin/env python3
"""Device-equivalent assembly: template + map + SD1.5 checkpoint -> runnable ONNX (+ .weights).

This is the exact algorithm the app implements in Java (read safetensors tensor by `key`, convert to
fp16, transpose 2-D if `transpose`, reshape to `dims`, append at `offset`). With --verify it checks
the assembled initializers byte-for-byte against the original exported ONNX (same checkpoint), and
with --run it executes one forward pass of each model on CPU (any checkpoint).
"""
import argparse
import json
import os
import shutil

import numpy as np


def assemble(templates, name, ckpt_path, out):
    from safetensors import safe_open
    with open(os.path.join(templates, f"{name}.map.json")) as f:
        m = json.load(f)
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(templates, f"{name}.template.onnx"), os.path.join(out, f"{name}.onnx"))
    with safe_open(ckpt_path, "np") as ck, open(os.path.join(out, m["weights_file"]), "wb") as w:
        for t in m["tensors"]:
            a = ck.get_tensor(t["key"]).astype(np.float16)
            if t["transpose"]:
                a = a.reshape(a.shape[0], -1).T
            a = np.ascontiguousarray(a.reshape(t["dims"]))
            assert w.tell() == t["offset"] and a.nbytes == t["length"], t["name"]
            w.write(a.tobytes())
    print(f"[assemble] {name}: {len(m['tensors'])} tensors, {m['total_bytes'] / 1e6:.1f} MB")
    return os.path.join(out, f"{name}.onnx")


def verify(assembled, original):
    import onnx
    from onnx import numpy_helper
    a = {i.name: i for i in onnx.load(assembled, load_external_data=True).graph.initializer}
    o = {i.name: i for i in onnx.load(original).graph.initializer}
    bad = [n for n in o if not np.array_equal(numpy_helper.to_array(o[n]), numpy_helper.to_array(a[n]))]
    print(f"[verify] {os.path.basename(original)}: {len(o) - len(bad)}/{len(o)} initializers identical")
    if bad:
        raise SystemExit(f"mismatch: {bad[:10]}")


def run(out):
    import onnxruntime as ort
    s = lambda n: ort.InferenceSession(os.path.join(out, n), providers=["CPUExecutionProvider"])
    ids = np.full((1, 77), 49407, np.int32)
    ids[0, 0] = 49406
    hid = s("text_encoder.onnx").run(None, {"input_ids": ids})[0]
    eps = s("unet.onnx").run(None, {"sample": np.random.default_rng(0).standard_normal((1, 4, 64, 64)).astype(np.float32),
                                     "timestep": np.array([499.0], np.float32), "encoder_hidden_states": hid})[0]
    print(f"[run] hidden std={hid.std():.4f} eps std={eps.std():.4f} nan={int(np.isnan(eps).sum())}")
    if not np.isfinite(eps).all() or not 0.1 < eps.std() < 5:
        raise SystemExit("assembled model produced implausible output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", help="dir with the original exported ONNX for byte comparison")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    for name in ("text_encoder", "unet"):
        p = assemble(a.templates, name, a.checkpoint, a.out)
        if a.verify:
            verify(p, os.path.join(a.verify, f"{name}.onnx"))
    if a.run:
        run(a.out)


if __name__ == "__main__":
    main()
