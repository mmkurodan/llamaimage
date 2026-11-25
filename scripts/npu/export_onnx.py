#!/usr/bin/env python3
"""Export DreamShaper 8 LCM (SD1.5) + TAESD to static-shape ONNX for the NPU path.

The NPU backend runs these with ONNX Runtime's QNN execution provider (Hexagon HTP). QNN needs
static shapes, so everything is fixed to batch 1, 77 tokens and 512x512 (64x64 latents).

Outputs (in --out):
  text_encoder.onnx   input_ids int32 [1,77]                     -> hidden float32 [1,77,768]
  unet.onnx           sample f32 [1,4,64,64], timestep f32 [1],
                      encoder_hidden_states f32 [1,77,768]        -> noise_pred f32 [1,4,64,64]
  taesd_decoder.onnx  latents f32 [1,4,64,64]                    -> image f32 [1,3,512,512] in [0,1]
  taesd_encoder.onnx  image f32 [1,3,512,512] in [0,1]           -> latents f32 [1,4,64,64]
  tokenizer/ (vocab.json, merges.txt), scheduler_config.json, LICENSE notes

Weights are stored as fp16 (I/O stays float32), which halves the files and is what the HTP runs
anyway (QNN EP enable_htp_fp16_precision). Models are pinned to exact HF revisions.

Licenses: DreamShaper 8 LCM — CreativeML OpenRAIL-M (Lykon); TAESD — MIT (madebyollin).
Converted files are derivatives and must be distributed with those licenses.
"""
import argparse
import json
import os
import shutil

import torch

DS_REPO = "Lykon/dreamshaper-8-lcm"
DS_REV = "4645d8bc6a8e6b106d21606d63e8460cdad4f1a6"
TAESD_REPO = "madebyollin/taesd"
TAESD_REV = "614f76814bbe30edbe2e627ace1c2234c81a2c0e"
OPSET = 17


class TextEncoder(torch.nn.Module):
    def __init__(self, te):
        super().__init__()
        self.te = te

    def forward(self, input_ids):
        return self.te(input_ids.long(), return_dict=False)[0]


class UNet(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(sample, timestep, encoder_hidden_states, return_dict=False)[0]


class TaesdDecoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        # AutoencoderTiny.decode returns [-1, 1]; hand the app [0, 1].
        return (self.vae.decode(latents, return_dict=False)[0] * 0.5 + 0.5).clamp(0, 1)


class TaesdEncoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        return self.vae.encode(image * 2.0 - 1.0, return_dict=False)[0]


def fix_resize_params(model):
    """Resize's roi/scales (inputs 1, 2) must be float32. The fp16 converter turns the Constant
    nodes (and initializers) feeding them into float16 even when Resize itself is blocked; turn
    those back into float32."""
    import numpy as np
    from onnx import TensorProto, numpy_helper

    wanted = set()
    for n in model.graph.node:
        if n.op_type == "Resize":
            wanted.update(x for x in list(n.input)[1:3] if x)
    fixed = 0
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output[0] in wanted:
            for att in n.attribute:
                if att.name == "value" and att.t.data_type == TensorProto.FLOAT16:
                    att.t.CopyFrom(numpy_helper.from_array(
                        numpy_helper.to_array(att.t).astype(np.float32), att.t.name))
                    fixed += 1
    for i, init in enumerate(model.graph.initializer):
        if init.name in wanted and init.data_type == TensorProto.FLOAT16:
            model.graph.initializer[i].CopyFrom(numpy_helper.from_array(
                numpy_helper.to_array(init).astype(np.float32), init.name))
            fixed += 1
    # Drop stale (float16) type annotations for those tensors; shape inference re-derives them.
    keep = [vi for vi in model.graph.value_info if vi.name not in wanted]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep)
    if fixed:
        print(f"[export] restored {fixed} Resize roi/scales tensors to float32")


def export(module, args, path, input_names, output_names):
    tmp = path + ".fp32"
    os.makedirs(tmp, exist_ok=True)
    fp32 = os.path.join(tmp, "model.onnx")
    with torch.no_grad():
        torch.onnx.export(module.eval(), args, fp32, input_names=input_names, output_names=output_names,
                          opset_version=OPSET, do_constant_folding=True, dynamo=False)
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(fp32, load_external_data=True)
    # Resize's scales/roi inputs must stay float32 (ONNX spec); keep those nodes in fp32 (the
    # converter inserts the casts). Upsampling in the UNet and TAESD decoder uses Resize.
    block = list(float16.DEFAULT_OP_BLOCK_LIST) + ["Resize"]
    model16 = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True,
                                               op_block_list=block)
    fix_resize_params(model16)
    onnx.save(model16, path, save_as_external_data=False)
    shutil.rmtree(tmp)
    print(f"[export] {path}: {os.path.getsize(path) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from diffusers import AutoencoderTiny, StableDiffusionPipeline
    from huggingface_hub import hf_hub_download

    # Load from the SINGLE-FILE checkpoint (the format the app downloads and later converts on the
    # device), with the repo's diffusers configs. Exported weights then equal the checkpoint's
    # tensors bit for bit, which make_template.py relies on. (The repo's diffusers fp32 folders
    # were saved separately and differ slightly.)
    ckpt = hf_hub_download(DS_REPO, "DreamShaper8_LCM.safetensors", revision=DS_REV)
    pipe = StableDiffusionPipeline.from_single_file(ckpt, config=DS_REPO, revision=DS_REV,
                                                    torch_dtype=torch.float32, safety_checker=None,
                                                    load_safety_checker=False)
    te, unet = pipe.text_encoder, pipe.unet
    del pipe
    export(TextEncoder(te), (torch.zeros(1, 77, dtype=torch.int32),), os.path.join(a.out, "text_encoder.onnx"),
           ["input_ids"], ["hidden"])
    del te

    vae = AutoencoderTiny.from_pretrained(TAESD_REPO, revision=TAESD_REV, torch_dtype=torch.float32)
    export(TaesdDecoder(vae), (torch.zeros(1, 4, 64, 64),), os.path.join(a.out, "taesd_decoder.onnx"),
           ["latents"], ["image"])
    export(TaesdEncoder(vae), (torch.zeros(1, 3, 512, 512),), os.path.join(a.out, "taesd_encoder.onnx"),
           ["image"], ["latents"])
    del vae

    export(UNet(unet), (torch.zeros(1, 4, 64, 64), torch.tensor([999.0]), torch.zeros(1, 77, 768)),
           os.path.join(a.out, "unet.onnx"), ["sample", "timestep", "encoder_hidden_states"], ["noise_pred"])
    del unet

    tok = os.path.join(a.out, "tokenizer")
    os.makedirs(tok, exist_ok=True)
    for f in ("vocab.json", "merges.txt"):
        shutil.copy(hf_hub_download(DS_REPO, f"tokenizer/{f}", revision=DS_REV), os.path.join(tok, f))
    shutil.copy(hf_hub_download(DS_REPO, "scheduler/scheduler_config.json", revision=DS_REV),
                os.path.join(a.out, "scheduler_config.json"))
    with open(os.path.join(a.out, "SOURCES.json"), "w") as f:
        json.dump({"dreamshaper_8_lcm": {"repo": DS_REPO, "revision": DS_REV, "file": "DreamShaper8_LCM.safetensors",
                                         "license": "creativeml-openrail-m"},
                   "taesd": {"repo": TAESD_REPO, "revision": TAESD_REV, "license": "mit"},
                   "opset": OPSET, "weights": "fp16 (float32 I/O)", "resolution": 512}, f, indent=2)


if __name__ == "__main__":
    main()
