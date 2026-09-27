"""Make an fp8 "scaled_fp8" diffusion model from an SD1.5 checkpoint (test material).

Every Linear (2D) weight of the UNet is stored as float8_e4m3fn with a per-tensor
float32 `.scale_weight`; the `scaled_fp8` marker makes ComfyUI's
convert_old_quants turn it into mixed-precision (QuantizedTensor) layers.

    python tests/make_fp8_unet.py CHECKPOINT OUT
"""

import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file  # writing small/medium test files only

PREFIX = "model.diffusion_model."


def main(src, out):
    sd = {}
    with safe_open(src, "pt") as f:
        for k in f.keys():
            if not k.startswith(PREFIX):
                continue
            t = f.get_tensor(k)
            name = k[len(PREFIX):]
            if name.endswith(".weight") and t.ndim == 2 and t.numel() >= 4096:  # Linear only (mixed-precision ops)
                w = t.float()
                scale = (w.abs().max() / 448.0).clamp(min=1e-12)
                sd[name] = (w / scale).to(torch.float8_e4m3fn)
                sd[name[: -len(".weight")] + ".scale_weight"] = scale.reshape(())
            else:
                sd[name] = t
    sd["scaled_fp8"] = torch.empty(0, dtype=torch.float8_e4m3fn)
    save_file(sd, out)
    n = sum(1 for k in sd if k.endswith(".scale_weight"))
    print("wrote {} ({} fp8 layers)".format(out, n))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
