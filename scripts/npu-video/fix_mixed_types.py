#!/usr/bin/env python3
"""Repairs fp16-converted ONNX graphs where a node gets both float and float16 inputs (seen with
onnxconverter-common 1.16 + torch 2.10 exports): inserts Cast(to=float16) on the float inputs.
If the float16 inputs are all constants they are cast up instead (precision). Resize's roi/scales (inputs 1, 2) must stay float and are left alone.

  python3 fix_mixed_types.py model.onnx [more.onnx ...]   (rewritten in place)
"""
import sys

import onnx
from onnx import TensorProto, helper, shape_inference

KEEP_FLOAT = {"Resize": {1, 2}}


_N = [0]


def _name(prefix):
    _N[0] += 1
    return f"{prefix}_{_N[0]}"


def fix(model):
    inf = shape_inference.infer_shapes(model, strict_mode=False)
    types = {v.name: v.type.tensor_type.elem_type for v in
             list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output)}
    types.update({i.name: i.data_type for i in model.graph.initializer})
    for n in model.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    types[n.output[0]] = a.t.data_type
    consts = {i.name for i in model.graph.initializer} | {n.output[0] for n in model.graph.node if n.op_type == "Constant"}
    new_nodes, added = [], 0
    for n in model.graph.node:
        if n.op_type != "Cast":
            keep = KEEP_FLOAT.get(n.op_type, set())
            ts = {types.get(x) for k, x in enumerate(n.input) if x and k not in keep}
            if TensorProto.FLOAT in ts and TensorProto.FLOAT16 in ts:
                half = [x for k, x in enumerate(n.input) if x and k not in keep and types.get(x) == TensorProto.FLOAT16]
                if all(x in consts for x in half):
                    # fp16 side is only constants (e.g. time_proj frequencies x timestep): compute in
                    # float -- t*freq up to ~1000 rad would lose its phase in fp16.
                    for k, x in enumerate(n.input):
                        if x in half:
                            y = _name(x + "__to32")
                            new_nodes.append(helper.make_node("Cast", [x], [y], to=TensorProto.FLOAT,
                                                              name=_name("fixcast")))
                            n.input[k] = y
                            types[y] = TensorProto.FLOAT
                            added += 1
                    new_nodes.append(n)
                    continue
                for k, x in enumerate(n.input):
                    if x and k not in keep and types.get(x) == TensorProto.FLOAT:
                        y = _name(x + "__to16")
                        new_nodes.append(helper.make_node("Cast", [x], [y], to=TensorProto.FLOAT16,
                                                          name=_name("fixcast")))
                        n.input[k] = y
                        types[y] = TensorProto.FLOAT16
                        added += 1
        new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return added


def main():
    for p in sys.argv[1:]:
        main_one(p)


def main_one(p):
    if True:
        m = onnx.load(p)
        total = 0
        for _ in range(8):     # a fix can make a downstream node's type known; iterate
            k = fix(m)
            total += k
            if not k:
                break
        if total:
            del m.graph.value_info[:]   # stale float16 annotations; ORT re-infers
        onnx.save(m, p)
        print(f"[fix] {p}: inserted {total} casts")


if __name__ == "__main__":
    main()
