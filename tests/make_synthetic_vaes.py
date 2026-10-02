"""Write random-weight VAE files with the real structure of the three target
VAEs, for running tests/bench_vae.py without the real models (CPU smoke runs;
numbers mean nothing beyond exercising the code paths).

    python tests/make_synthetic_vaes.py OUT_DIR [--full]

  synthetic_kl_z4.safetensors     SDXL-style AutoencoderKL (post_quant_conv, 4 latent channels)
  synthetic_kl_z16.safetensors    Flux ae-style AutoencodingEngine (16 latent channels)
  synthetic_wan.safetensors       Wan 2.1 / qwen_image_vae-style WanVAE (16 latent channels, 5D)
Default decoder width is small (ch 32 / dim 16); --full uses the real widths
(ch 128 / dim 96). Only the decoder (and what VAE detection needs) is written.
"""

import math
import os
import sys

import torch
from safetensors.torch import save_file  # writing a small test file, not loading a model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("COMFY_ARGS", "--cpu")
import common  # noqa: E402,F401  (ComfyUI environment)
from comfy.ldm.modules.diffusionmodules.model import Decoder  # noqa: E402
import comfy.ldm.wan.vae as wan_vae  # noqa: E402


def init_random(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2 and not name.endswith("gamma"):
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p[0].numel()))
            elif name.endswith("gamma") or (p.ndim == 1 and name.endswith("weight")):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
    return model


def kl(z, ch, post_quant_conv):
    dec = init_random(Decoder(ch=ch, out_ch=3, ch_mult=[1, 2, 4, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0,
                              in_channels=3, resolution=256, z_channels=z))
    sd = {"decoder." + k: v.contiguous() for k, v in dec.state_dict().items()}
    if post_quant_conv:
        g = torch.Generator().manual_seed(2)
        sd["post_quant_conv.weight"] = torch.randn(z, z, 1, 1, generator=g) / math.sqrt(z)
        sd["post_quant_conv.bias"] = torch.zeros(z)
    return sd


def wan(dim):
    m = init_random(wan_vae.WanVAE(dim=dim, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                                   temperal_downsample=[False, True, True], image_channels=3, conv_out_channels=3, dropout=0.0))
    return {k: v.contiguous() for k, v in m.state_dict().items() if k.startswith(("decoder.", "conv2.")) or k == "encoder.conv1.weight"}


def main(out_dir, full):
    os.makedirs(out_dir, exist_ok=True)
    ch, dim = (128, 96) if full else (32, 16)
    for name, sd in (("synthetic_kl_z4", kl(4, ch, True)), ("synthetic_kl_z16", kl(16, ch, False)), ("synthetic_wan", wan(dim))):
        p = os.path.join(out_dir, name + ".safetensors")
        save_file({k: v.float() for k, v in sd.items()}, p)
        print("wrote {} ({} tensors, {:.1f} MB)".format(p, len(sd), sum(v.numel() * 4 for v in sd.values()) / 1e6))


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(args) != 1:
        raise SystemExit(__doc__)
    main(args[0], "--full" in sys.argv)
