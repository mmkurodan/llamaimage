#!/usr/bin/env python3
"""Weightless templates + weight maps for the AnimateDiff video UNet parts (see make_template.py).

Weights come from TWO files on the device: the SD1.5 checkpoint (keys "model.diffusion_model.*")
and the motion module (AnimateDiff-Lightning diffusers safetensors). Motion keys are written to
the map with a "motion:" prefix; the app reads those from the motion file.

Outputs: vunet_{0,1,2}.template.onnx + .map.json (weights file vunet_N.weights).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "npu"))
import make_template  # noqa: E402
from animatediff_parts import NAMES  # noqa: E402

MOTION_PREFIX = "motion:"


def video_name(part):
    return "v" + part      # unet_0 -> vunet_0


def load_motion(path):
    from safetensors.numpy import load_file
    return {MOTION_PREFIX + k: v for k, v in load_file(path).items()
            if v.dtype in (np.float16, np.float32) and v.size >= make_template.MATCH_MIN_ELEMS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="dir with the exported unet_0..2.onnx (video)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = make_template.load_checkpoint(a.checkpoint, ["model.diffusion_model."])
    ck.update(load_motion(a.motion))
    # The motion modules' positional encodings are sliced to the frame count at export ([1, F, C]
    # constants). They match no stored tensor, are tiny and deterministic: keep them inline. Any
    # other unmatched large initializer is still an error (it would ship checkpoint weights).
    must = make_template.MUST_MATCH_ELEMS
    make_template.MUST_MATCH_ELEMS = 1 << 62
    try:
        for part in NAMES:
            make_template.make(os.path.join(a.models, f"{part}.onnx"), ck, video_name(part), a.out)
            check_inline(os.path.join(a.out, f"{video_name(part)}.template.onnx"), must)
    finally:
        make_template.MUST_MATCH_ELEMS = must


def check_inline(path, must):
    import onnx
    from onnx import TensorProto
    m = onnx.load(path, load_external_data=False)
    bad = []
    for i in m.graph.initializer:
        n = int(np.prod(i.dims)) if i.dims else 1
        if i.data_location != TensorProto.EXTERNAL and n >= must:
            if not (len(i.dims) == 3 and i.dims[0] == 1 and i.dims[1] <= 32):   # pos. encoding [1, F, C]
                bad.append((i.name, list(i.dims)))
    if bad:
        raise SystemExit(f"[template] {path}: unmatched initializers (not positional encodings): {bad[:10]}")


if __name__ == "__main__":
    main()
