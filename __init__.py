"""ComfyUI custom node entry point for Monoload.

Importing this package (ComfyUI does so at startup, before any model is
loaded) installs
  * the runtime LoRA merge on comfy.model_patcher.ModelPatcher,
  * the per-prompt LoRA release on execution.PromptExecutor, and
  * the managed VAE decode on comfy.sd.VAE.decode (layer 1: stripe decoding
    of recognized decoders; layer 2: op-level chunking; own memory estimate,
    no tiled fallback).

Master switch MONOLOAD (monoload/settings.py): unset or 1 -> Monoload's
default strategy for every model and VAE; 0 -> native ComfyUI everywhere
(the hooks pass every call through), except for what a Monoload node
explicitly switches on for its own model / VAE. MONOLOAD_DISABLE=1 installs
nothing at all. The advanced variables (MONOLOAD_EXACT, MONOLOAD_KEEP_LORA,
MONOLOAD_DISABLE_VAE, MONOLOAD_DISABLE_VAE_STRIPE, MONOLOAD_VAE_*) are global
defaults that a node's explicit choice overrides for its model / VAE.

Nodes (monoload/nodes, registered in every case): Monoload VAE Settings --
a copy of a VAE with its own settings for the managed decode.
"""

import logging

# the nodes (monoload/nodes) are registered whatever the switches say, so that saved workflows load; with
# MONOLOAD_DISABLE they pass their input through and say so
from .monoload.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .monoload import settings

if settings.disabled():
    logging.info("[Monoload] MONOLOAD_DISABLE is set: nothing installed, ComfyUI stays native and the Monoload nodes pass their input through")
else:
    from .monoload import hotpatch, release, vae
    from .monoload.vae_ops import fmt_bytes

    hotpatch.install()
    release.install()
    installed_vae = vae.install()
    logging.info("[Monoload] master switch MONOLOAD: {}".format(settings.master_note()))
    if settings.master():
        logging.info("[Monoload] LoRA: runtime merge (no in-place LoRA, no weight backups), merge {}; {}".format(
            "bit-exact (MONOLOAD_EXACT=1)" if hotpatch.is_exact() else "fused fp16 addmm / relaxed (MONOLOAD_EXACT=1 for bit-exact)",
            "kept between prompts (MONOLOAD_KEEP_LORA=1)" if release.keep() else "released after every prompt (base models stay loaded)"))
    if installed_vae:
        mode, var = vae.global_mode()
        if mode == "native":
            logging.info("[Monoload] VAE decode: native by default ({}); a Monoload VAE Settings node with mode auto manages "
                         "its VAE".format(var))
        else:
            logging.info("[Monoload] VAE decode managed: op-level chunking (conv row blocks, attention query blocks), workspace {} "
                         "(MONOLOAD_VAE_WORKSPACE), own memory estimate, OOM -> smaller blocks, never tiled; "
                         "images only (4D / 5D T=1), set MONOLOAD_DISABLE_VAE=1 for native".format(fmt_bytes(vae.workspace())))
            if mode == "auto":
                logging.info("[Monoload] VAE layer 1 (stripe decoding) on for recognized decoders (Wan 2.1 / qwen_image_vae single frame; "
                             "LDM Decoder of SD1.5 / SDXL / SD3 / Flux ae with whole-image GroupNorm statistics, scheme {}; "
                             "self-tested on first use): {}{}; other decoders use layer 2; "
                             "set MONOLOAD_DISABLE_VAE_STRIPE=1 to use layer 2 everywhere".format(
                                 "{} (forced, MONOLOAD_VAE_GN_SCHEME)".format(vae.gn_scheme()) if vae.gn_scheme_forced() else "{} (default)".format(vae.gn_scheme()),
                                 "peak budget {} (MONOLOAD_VAE_BUDGET): the fastest of layer 2 and the layer-1 configurations whose estimate "
                                 "fits it".format(fmt_bytes(vae.budget())) if vae.budget() else
                                 "default stripe policy: the peak of {}-row stripes, tallest stripes within it (MONOLOAD_VAE_BUDGET to choose a budget)".format(vae.DEFAULT_POLICY_ROWS),
                                 ", stripe height forced to {} rows (MONOLOAD_VAE_STRIPE_ROWS)".format(vae.stripe_rows()) if vae.stripe_rows() else ""))
            else:
                logging.info("[Monoload] VAE layer 1 (stripe decoding) off (MONOLOAD_DISABLE_VAE_STRIPE): every managed decode uses layer 2")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
