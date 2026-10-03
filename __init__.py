"""ComfyUI custom node entry point for Monoload.

Importing this package (ComfyUI does so at startup, before any model is
loaded) installs
  * the runtime LoRA merge on comfy.model_patcher.ModelPatcher,
  * the per-prompt LoRA release on execution.PromptExecutor
    (skipped when MONOLOAD_KEEP_LORA=1), and
  * the managed VAE decode on comfy.sd.VAE.decode (layer 1: stripe decoding
    of recognized decoders, MONOLOAD_DISABLE_VAE_STRIPE=1 turns it off;
    layer 2: op-level chunking; own memory estimate, no tiled fallback;
    skipped when MONOLOAD_DISABLE_VAE=1 or MONOLOAD_EXACT=1).
Set MONOLOAD_DISABLE=1 to leave ComfyUI completely native, MONOLOAD_EXACT=1
for the bit-exact merge instead of the fused default (monoload/hotpatch.py)
and a native VAE decode.
"""

import logging
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


if _flag("MONOLOAD_DISABLE"):
    logging.info("[Monoload] MONOLOAD_DISABLE is set: runtime LoRA merge and VAE decode management NOT installed, ComfyUI stays native")
else:
    from .monoload import hotpatch, release

    hotpatch.install()
    logging.info("[Monoload] runtime LoRA merge installed on ModelPatcher (no in-place LoRA, no weight backups), merge: {}".format(
        "bit-exact (MONOLOAD_EXACT=1)" if hotpatch.is_exact() else "fused fp16 addmm / relaxed (set MONOLOAD_EXACT=1 for bit-exact)"))
    if _flag("MONOLOAD_KEEP_LORA"):
        logging.info("[Monoload] MONOLOAD_KEEP_LORA is set: LoRA state is kept between prompts")
    else:
        release.install()
        logging.info("[Monoload] LoRA is released after every prompt (base models stay loaded)")

    if _flag("MONOLOAD_DISABLE_VAE"):
        logging.info("[Monoload] VAE decode: native (MONOLOAD_DISABLE_VAE is set)")
    elif hotpatch.is_exact():
        logging.info("[Monoload] VAE decode: native (MONOLOAD_EXACT=1; chunking changes GEMM shapes, so it is not bit-exact)")
    else:
        from .monoload import vae
        from .monoload.vae_ops import fmt_bytes

        if vae.install():
            logging.info("[Monoload] VAE decode managed: op-level chunking (conv row blocks, attention query blocks), workspace {} "
                         "(MONOLOAD_VAE_WORKSPACE), own memory estimate, OOM -> smaller blocks, never tiled; "
                         "images only (4D / 5D T=1), set MONOLOAD_DISABLE_VAE=1 for native".format(fmt_bytes(vae.workspace())))
            if vae.stripe_enabled():
                logging.info("[Monoload] VAE layer 1 (stripe decoding) on for recognized decoders (Wan 2.1 / qwen_image_vae single frame; "
                             "LDM Decoder of SD1.5 / SDXL / SD3 / Flux ae with whole-image GroupNorm statistics, scheme {} (MONOLOAD_VAE_GN_SCHEME); "
                             "self-tested on first use): {}{}; other decoders use layer 2; "
                             "set MONOLOAD_DISABLE_VAE_STRIPE=1 to use layer 2 everywhere".format(
                                 vae.gn_scheme(),
                                 "peak budget {} (MONOLOAD_VAE_BUDGET)".format(fmt_bytes(vae.budget())) if vae.budget() else
                                 "default stripe policy: the peak of {}-row stripes, tallest stripes within it (MONOLOAD_VAE_BUDGET to choose a budget)".format(vae.DEFAULT_POLICY_ROWS),
                                 ", stripe height forced to {} rows (MONOLOAD_VAE_STRIPE_ROWS)".format(vae.stripe_rows()) if vae.stripe_rows() else ""))
            else:
                logging.info("[Monoload] VAE layer 1 (stripe decoding) off (MONOLOAD_DISABLE_VAE_STRIPE): every managed decode uses layer 2")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
