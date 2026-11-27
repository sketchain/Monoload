"""Group D on the real machine: MONOLOAD=0 (master switch off) decodes Mochi and Cosmos 1.0 bit for bit like native
(the unwrapped VAE.decode), and with the switch on the same decode goes through the streamed layer 2 (DESIGN §9.43).

Real files (models/vae/mochi_vae.safetensors, cosmos_cv8x8x8_1.0.safetensors), bf16, small random latents so native
fits: Mochi 424x240x31, Cosmos 640x352x33. A missing file is skipped.

    python tests/check_group_d_zero.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bench_vae  # noqa: E402  (sets up ComfyUI on the device)
import torch  # noqa: E402

from common import check, finish  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import folder_paths  # noqa: E402
import nodes  # noqa: E402
from monoload import settings, vae as mvae  # noqa: E402

CASES = (("mochi_vae.safetensors", 424, 240, 31), ("cosmos_cv8x8x8_1.0.safetensors", 640, 352, 33))


def main():
    mvae.install()
    for name, w, h, f in CASES:
        if folder_paths.get_full_path("vae", name) is None:
            print("SKIP {}: not in models/vae".format(name))
            continue
        vae = nodes.VAELoader().load_vae(name)[0]
        lat = bench_vae.random_latent(vae, w, h, 0, 1.0, f)
        with torch.inference_mode():
            ref = mvae._ORIG["decode"](vae, lat).cpu()
            mm.soft_empty_cache()
            settings.set_master(False)
            try:
                zero = vae.decode(lat).cpu()
            finally:
                settings.set_master(True)
            mm.soft_empty_cache()
            on = vae.decode(lat).cpu()
        check("{} {}x{}x{}: MONOLOAD=0 == native bit for bit".format(name, w, h, f),
              zero.shape == ref.shape and torch.equal(zero, ref), str(tuple(zero.shape)))
        d = float((on.float() - ref.float()).abs().max()) if on.shape == ref.shape else float("nan")
        check("{}: Monoload on, same shape, close to native (pixel max {:.4f})".format(name, d), on.shape == ref.shape and d < 0.05)
        del vae, ref, zero, on
        mm.unload_all_models()
        mm.soft_empty_cache()
    finish()


if __name__ == "__main__":
    main()
