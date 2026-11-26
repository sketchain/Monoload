"""Write a random LoRA for a real diffusion model file (safetensors or GGUF), for real-machine tests of models
whose downloaded LoRAs are not there yet. Keys use ComfyUI's generic format (diffusion_model.<layer>.lora_up /
lora_down / alpha), so LoraLoader maps them like any downloaded LoRA.

Every --every-th 2D `.weight` gets a rank --rank LoRA (up * 0.02, down * 0.1, alpha = rank: a visible but sane
change). --small also adds a `.diff` to each non-layer parameter (Wan `modulation`, LTX `scale_shift_table`), which
native ComfyUI drops (docs/GGUF-QUANT.md §5); leave it off when comparing with native for max|Δ| = 0.

    python tests/make_model_lora.py MODEL OUT.safetensors [--every 4] [--rank 16] [--small] [--max-layers N]
"""

import argparse
import json
import re
import struct

import torch
from safetensors.torch import save_file  # writing a small test file, not loading a model

SMALL = re.compile(r"(^|\.)(modulation|scale_shift_table)$")


def shapes_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    out = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        out[k] = tuple(v["shape"])
    return out


def shapes_gguf(path):
    import gguf
    r = gguf.GGUFReader(path)
    orig = {}
    for f in r.fields.values():          # ComfyUI-GGUF stores reshaped tensors' shapes as comfy.gguf.orig_shape.<name>
        if f.name.startswith("comfy.gguf.orig_shape."):
            orig[f.name[len("comfy.gguf.orig_shape."):]] = tuple(int(f.parts[i][0]) for i in f.data)
    return {t.name: orig.get(t.name, tuple(int(v) for v in reversed(t.shape))) for t in r.tensors}


def strip(k):
    for p in ("model.diffusion_model.", "diffusion_model."):
        if k.startswith(p):
            return k[len(p):]
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--max-layers", type=int, default=0)
    a = ap.parse_args()
    shapes = shapes_gguf(a.model) if a.model.endswith(".gguf") else shapes_safetensors(a.model)
    g = torch.Generator().manual_seed(3)
    sd, n_lin, n_small = {}, 0, 0
    layers = sorted(k for k, s in shapes.items() if k.endswith(".weight") and len(s) == 2 and min(s) >= 16)
    for i, k in enumerate(layers):
        if i % a.every:
            continue
        if a.max_layers and n_lin >= a.max_layers:
            break
        out_f, in_f = shapes[k]
        base = "diffusion_model." + strip(k)[: -len(".weight")]
        sd[base + ".lora_up.weight"] = (torch.randn(out_f, a.rank, generator=g) * 0.02).to(torch.bfloat16)
        sd[base + ".lora_down.weight"] = (torch.randn(a.rank, in_f, generator=g) * 0.1).to(torch.bfloat16)
        sd[base + ".alpha"] = torch.tensor(float(a.rank))
        n_lin += 1
    if a.small:
        for k, s in sorted(shapes.items()):
            if SMALL.search(strip(k)):
                sd["diffusion_model." + strip(k) + ".diff"] = (torch.randn(*s, generator=g) * 0.01).to(torch.bfloat16)
                n_small += 1
    save_file(sd, a.out, metadata={"monoload_synthetic": "1", "source": a.model.rsplit("/", 1)[-1]})
    print("wrote {}: {} linear layers (of {}), {} small parameters".format(a.out, n_lin, len(layers), n_small))


if __name__ == "__main__":
    main()
