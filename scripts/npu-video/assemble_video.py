#!/usr/bin/env python3
"""Device-equivalent assembly of the video UNet parts (= NpuModelAssembler with a motion file):
template + map + SD1.5 checkpoint + motion module -> vunet_N.onnx + vunet_N.weights.

--verify DIR  compares every initializer byte for byte with the exported unet_N.onnx in DIR
--run         one chained forward pass on ORT-CPU (shapes from the graph), output must be sane
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "npu"))
from animatediff_parts import NAMES  # noqa: E402
from make_video_template import MOTION_PREFIX, video_name  # noqa: E402


def assemble(templates, name, ckpt_path, motion_path, out):
    from safetensors import safe_open
    with open(os.path.join(templates, f"{name}.map.json")) as f:
        m = json.load(f)
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(templates, f"{name}.template.onnx"), os.path.join(out, f"{name}.onnx"))
    n_motion = 0
    with safe_open(ckpt_path, "np") as ck, safe_open(motion_path, "np") as mo, \
            open(os.path.join(out, m["weights_file"]), "wb") as w:
        for t in m["tensors"]:
            key = t["key"]
            if key.startswith(MOTION_PREFIX):
                a = mo.get_tensor(key[len(MOTION_PREFIX):])
                n_motion += 1
            else:
                a = ck.get_tensor(key)
            a = a.astype(np.float16)
            if t["transpose"]:
                a = a.reshape(a.shape[0], -1).T
            a = np.ascontiguousarray(a.reshape(t["dims"]))
            assert w.tell() == t["offset"] and a.nbytes == t["length"], t["name"]
            w.write(a.tobytes())
    print(f"[assemble] {name}: {len(m['tensors'])} tensors ({n_motion} from the motion module), "
          f"{m['total_bytes'] / 1e6:.1f} MB")
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
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC   # no fp16 Gelu on CPU
    sess = [ort.InferenceSession(os.path.join(out, f"{video_name(p)}.onnx"), so,
                                 providers=["CPUExecutionProvider"]) for p in NAMES]
    shape = sess[0].get_inputs()[0].shape        # [F, 4, h, w]
    rng = np.random.default_rng(0)
    env = {"sample": rng.standard_normal(shape).astype(np.float32),
           "timestep": np.array([749.0], np.float32),
           "encoder_hidden_states": (rng.standard_normal((1, 77, 768)) * 0.5).astype(np.float32)}
    for s in sess:
        outs = s.run(None, {i.name: env[i.name] for i in s.get_inputs()})
        env.update({o.name: v for o, v in zip(s.get_outputs(), outs)})
    eps = env["noise_pred"]
    print(f"[run] sample {shape} -> noise_pred {list(eps.shape)} std={eps.std():.4f} "
          f"nan={int(np.isnan(eps).sum())}")
    if not np.isfinite(eps).all() or not 0.1 < eps.std() < 5:
        raise SystemExit("assembled model produced implausible output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    for part in NAMES:
        p = assemble(a.templates, video_name(part), a.checkpoint, a.motion, a.out)
        if a.verify:
            verify(p, os.path.join(a.verify, f"{part}.onnx"))
    if a.run:
        run(a.out)


if __name__ == "__main__":
    main()
