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
nothing at all. MONOLOAD_LANG=zh: log and error messages in Chinese
(monoload/messages.py; the node UI follows ComfyUI's language through
locales/). The advanced variables (MONOLOAD_EXACT, MONOLOAD_KEEP_LORA,
MONOLOAD_DISABLE_VAE, MONOLOAD_DISABLE_VAE_STRIPE, MONOLOAD_VAE_*) are global
defaults that a node's explicit choice overrides for its model / VAE.

Nodes (monoload/nodes, registered in every case): Monoload LoRA Settings,
Monoload VAE Settings (per model / VAE settings), Monoload Info (what
Monoload is doing; its text is shown by web/monoload_info.js).
"""

import logging

# the nodes (monoload/nodes) are registered whatever the switches say, so that saved workflows load; with
# MONOLOAD_DISABLE they pass their input through and say so
from .monoload.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .monoload import settings

WEB_DIRECTORY = "./web"   # web/monoload_info.js: the Monoload Info node's text in the node box

from .monoload.messages import msg, warn_bad_lang

warn_bad_lang()
if settings.disabled():
    logging.info(msg("entry.disabled"))
else:
    from .monoload import hotpatch, release

    hotpatch.install()
    release.install()
    # the VAE part depends on more of ComfyUI's internals: if a ComfyUI update breaks its import or its API check,
    # the LoRA part and the nodes still work (the VAE decode stays native)
    try:
        from .monoload import vae
        from .monoload.vae_ops import fmt_bytes
        installed_vae = vae.install()
    except Exception as e:
        logging.warning(msg("vae.api_differs", bad="{}: {}".format(type(e).__name__, e)))
        installed_vae = False
    try:
        import nodes as _comfy_nodes
        from .monoload import lora_overrides
        lora_overrides.install_names(_comfy_nodes.LoraLoader)   # LoRA file names for the Monoload Info node
    except Exception:
        logging.warning(msg("entry.names_missing"))
    logging.info(msg("entry.master", state=settings.master_note()))
    if settings.master():
        logging.info(msg("entry.lora", merge=msg("entry.merge_exact") if hotpatch.is_exact() else msg("entry.merge_fused"),
                         after=msg("entry.after_keep") if release.keep() else msg("entry.after_release")))
    if installed_vae:
        mode, var = vae.global_mode()
        if mode == "native":
            logging.info(msg("entry.vae_native", var=var))
        else:
            logging.info(msg("entry.vae_managed", workspace=fmt_bytes(vae.workspace())))
            if mode == "auto":
                scheme = (msg("entry.scheme_forced", scheme=vae.gn_scheme()) if vae.gn_scheme_forced()
                          else msg("entry.scheme_default", scheme=vae.gn_scheme()))
                policy = (msg("entry.policy_budget", budget=fmt_bytes(vae.budget())) if vae.budget()
                          else msg("entry.policy_default", rows=vae.DEFAULT_POLICY_ROWS))
                rows = msg("entry.rows_forced", rows=vae.stripe_rows()) if vae.stripe_rows() else ""
                logging.info(msg("entry.vae_layer1", scheme=scheme, policy=policy, rows=rows))
            else:
                logging.info(msg("entry.vae_layer2_only"))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
