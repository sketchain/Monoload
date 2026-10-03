"""Monoload VAE Settings: the managed VAE decode's settings for one VAE.

Takes a VAE and returns a copy of it that carries its own settings (peak
budget, GroupNorm scheme, stripe height, mode). The copy shares everything
with the input -- weights, model patcher, loading and unloading by ComfyUI's
model management -- and the input is not changed, so other branches of the
workflow that use the same VAE keep the global settings. Every node that
calls vae.decode() on the copy (VAEDecode and the like) gets these settings;
VAEDecodeTiled stays native as always.

Each setting left at its default ("default" / 0) falls back to the
environment variable (MONOLOAD_VAE_BUDGET, MONOLOAD_VAE_GN_SCHEME,
MONOLOAD_VAE_STRIPE_ROWS, MONOLOAD_DISABLE_VAE_STRIPE), and without one to
Monoload's default -- item by item (monoload/vae.py resolve_settings). Chained nodes: what a node leaves unset
keeps the upstream node's value.. The
meanings are those of the environment variables. The global switches
MONOLOAD_DISABLE / MONOLOAD_DISABLE_VAE / MONOLOAD_EXACT win: with them the
decode stays native and the node's settings are ignored (logged once).
"""

import logging
import os

from ..vae_overrides import MODE_CHOICES, SCHEME_CHOICES, with_settings

_NOTED = []


def _flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class MonoloadVAESettings:
    DISPLAY_NAME = "Monoload VAE Settings"
    CATEGORY = "Monoload"
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "apply"
    DESCRIPTION = ("A copy of the VAE (weights shared, the input unchanged) with its own settings for Monoload's managed "
                   "decode. Settings left at their default follow the environment variables, then Monoload's defaults.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE", {"tooltip": "The VAE; it is not changed, the output is a copy that shares its weights."}),
            "budget_gib": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4096.0, "step": 0.25,
                                     "tooltip": "Peak budget in GiB, as MONOLOAD_VAE_BUDGET: the fastest decode whose estimate fits it "
                                                "(error naming what is needed when none fits). 0 = not set here."}),
            "gn_scheme": (list(SCHEME_CHOICES), {"default": "default",
                                                "tooltip": "GroupNorm scheme of an LDM decoder's layer 1 (SDXL / SD1.5 / SD3 / Flux ae), "
                                                           "as MONOLOAD_VAE_GN_SCHEME: forces it. default = not set here."}),
            "stripe_rows": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 8,
                                    "tooltip": "Layer-1 stripe height in output rows, as MONOLOAD_VAE_STRIPE_ROWS: forces it. 0 = not set here."}),
            "mode": (list(MODE_CHOICES), {"default": "default",
                                          "tooltip": "auto: layer 1 (stripes) where the decoder is recognized, else layer 2; layer 2 only: "
                                                     "as MONOLOAD_DISABLE_VAE_STRIPE=1; native: ComfyUI's own decode for this VAE. "
                                                     "default = not set here."}),
        }}

    def apply(self, vae, budget_gib=0.0, gn_scheme="default", stripe_rows=0, mode="default"):
        off = self._global_off()
        if off and off not in _NOTED:
            _NOTED.append(off)
            logging.info("[Monoload] Monoload VAE Settings: {}, so VAE decodes stay native and the node's settings are ignored".format(off))
        return (with_settings(vae, budget=budget_gib, gn_scheme=gn_scheme, stripe_rows=stripe_rows, mode=mode),)

    @staticmethod
    def _global_off():
        """Why the managed decode is off (a global switch), or None."""
        if _flag("MONOLOAD_DISABLE"):
            return "MONOLOAD_DISABLE is set"
        from .. import vae
        if not vae.is_installed():
            if _flag("MONOLOAD_DISABLE_VAE"):
                return "MONOLOAD_DISABLE_VAE is set"
            if _flag("MONOLOAD_EXACT"):
                return "MONOLOAD_EXACT=1 is set"
            return "the managed VAE decode is not installed"
        return None
