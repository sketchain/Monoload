"""G group of docs/PLAN-video.md: the VAEs that tile / chunk on their own stay native under Monoload.

MiniMax H3 (video: stock ViT3DDecoder and Sol NonCausalViT3DDecoder; images: MiniMax-H3-Image and P-H3-Image),
SeedVR2 (VideoAutoencoderKLWrapper) and RelayVSR Flash (RelayFlash) all decode into their own preallocated
output (comfy_has_chunked_io) and bound their working memory themselves: 256 px tiles x 5+2-latent temporal
chunks, Sol's ring of 5-latent blocks, per-frame windowed attention, SeedVR2's frame slices with conv / GroupNorm
memory limits. Measured on CT 700 (tests/bench_selftiled.py, PLAN-video §2 G group): the MiniMax decode adds
at most a few GiB over its weights at 1080p, about half of it the output itself. Layer 1 / layer 2 would only
add work, so Monoload passes these through: this test locks the routing (every latent layout the VAE takes,
stills included, also after the multi-frame entry H1 opens 5D latents for other decoders).

Builds the decoder objects without weights (class only, no __init__): no GPU, no model files.
    python tests/test_vae_selftiled.py
"""

import types

from common import check, finish
import torch

from monoload import vae as mvae
from monoload.messages import msg


def fake(cls, latent_dim):
    fsm = cls.__new__(cls)   # routing reads only class attributes and the type name
    return types.SimpleNamespace(first_stage_model=fsm, latent_dim=latent_dim, vae_dtype=torch.bfloat16)


def main():
    import comfy.ldm.minimax.vae as mm_vae
    import comfy.ldm.seedvr.vae as sv_vae
    import comfy.ldm.wan.model_vsr as vsr

    cases = [
        # (class, latent_dim, latent shapes)
        (mm_vae.MiniMaxH3VideoVAE, 3, [(1, 24, 1, 45, 80), (1, 24, 7, 45, 80), (1, 24, 32, 68, 120), (2, 24, 22, 45, 80)]),
        (mm_vae.MiniMaxH3ImageVAE, 3, [(1, 24, 1, 64, 64), (1, 24, 64, 64)]),
        (mm_vae.PH3ImageVAE, 3, [(1, 24, 1, 64, 64), (1, 24, 64, 64)]),
        (sv_vae.VideoAutoencoderKLWrapper, 3, [(1, 16, 1, 90, 160), (1, 16, 9, 90, 160), (1, 16, 21, 135, 240)]),
        (vsr.RelayFlash, 3, [(1, 48, 3, 68, 120), (1, 48, 5, 136, 240)]),
    ]
    for cls, ld, shapes in cases:
        check("{} declares comfy_has_chunked_io".format(cls.__name__), getattr(cls, "comfy_has_chunked_io", False) is True)
        want = msg("vae.nr_chunked_io", model=cls.__name__)
        for shape in shapes:
            v = fake(cls, ld)
            got = mvae._native_reason(v, torch.empty(shape, device="meta"))
            check("{} latent {} stays native".format(cls.__name__, shape), got == want, repr(got))

    # every subclass of the MiniMax video VAE (image decoders, still-frame variants) inherits the flag
    subs = [c for c in vars(mm_vae).values() if isinstance(c, type) and issubclass(c, mm_vae.MiniMaxH3VideoVAE)]
    check("all MiniMax H3 VAE classes inherit comfy_has_chunked_io ({})".format(", ".join(c.__name__ for c in subs)),
          all(getattr(c, "comfy_has_chunked_io", False) for c in subs))
    finish()


if __name__ == "__main__":
    main()
