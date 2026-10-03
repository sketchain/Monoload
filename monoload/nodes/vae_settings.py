"""Monoload VAE Settings: the managed VAE decode's settings for one VAE.

Takes a VAE and returns a copy of it that carries its own settings (mode,
peak budget, GroupNorm scheme, stripe height). The copy shares everything
with the input -- weights, model patcher, loading and unloading by ComfyUI's
model management -- and the input is not changed, so other branches of the
workflow that use the same VAE keep the global settings. Every node that
calls vae.decode() on the copy (VAEDecode and the like) gets these settings;
VAEDecodeTiled stays native as always.

Priority, item by item (monoload/vae.py resolve_settings): a value chosen on
the node > the global default (the advanced environment variables:
MONOLOAD=0 / MONOLOAD_DISABLE_VAE / MONOLOAD_EXACT -> native,
MONOLOAD_DISABLE_VAE_STRIPE, MONOLOAD_VAE_BUDGET, MONOLOAD_VAE_GN_SCHEME,
MONOLOAD_VAE_STRIPE_ROWS) > Monoload's built-in default. An item left at
"default" (0 for stripe_rows) follows the global setting; with chained nodes
it keeps the upstream node's value. mode "auto" switches the managed decode
on for this VAE even when it is globally off (MONOLOAD=0 included).

MONOLOAD_DISABLE=1 (nothing installed): the node returns its input unchanged
and says so once.

Widget order: budget (the dropdown) right before budget_gib, which only
counts with budget "custom"; budget_gib keeps 0.01 GiB (step / round 0.01:
the frontend derives the stored precision from the step). Workflows saved
before this order load with shifted values (no compatibility, by decision).
"""

import logging

from .. import settings
from ..vae_overrides import BUDGET_CHOICES, MODE_CHOICES, SCHEME_CHOICES, with_settings
from ..messages import msg

_NOTED = []


class MonoloadVAESettings:
    DISPLAY_NAME = "Monoload VAE Settings"
    CATEGORY = "Monoload"
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "apply"
    DESCRIPTION = ("A copy of the VAE (weights shared, the input unchanged) with its own settings for Monoload's managed "
                   "decode. Each setting left at default follows the global setting (the environment variables), "
                   "then Monoload's built-in default.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE", {"tooltip": "The VAE; it is not changed, the output is a copy that shares its weights."}),
            "budget": (list(BUDGET_CHOICES), {"default": "default",
                                              "tooltip": "default: follow the global setting (MONOLOAD_VAE_BUDGET); unlimited: no "
                                                         "budget for this VAE (Monoload's default stripe policy); custom: budget_gib."}),
            "budget_gib": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4096.0, "step": 0.01, "round": 0.01,
                                     "tooltip": "Peak budget in GiB, used only when budget is custom: the fastest decode whose "
                                                "estimate fits it (error naming what is needed when none fits)."}),
            "gn_scheme": (list(SCHEME_CHOICES), {"default": "default",
                                                "tooltip": "GroupNorm scheme of an LDM decoder's layer 1 (SDXL / SD1.5 / SD3 / Flux ae): "
                                                           "forces it. default = follow the global setting (MONOLOAD_VAE_GN_SCHEME)."}),
            "stripe_rows": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 8,
                                    "tooltip": "Layer-1 stripe height in output rows: forces it. "
                                               "0 = follow the global setting (MONOLOAD_VAE_STRIPE_ROWS)."}),
            "mode": (list(MODE_CHOICES), {"default": "default",
                                          "tooltip": "default: follow the global setting (MONOLOAD=0 -> native); auto: Monoload's "
                                                     "managed decode for this VAE (layer 1 stripes where the decoder is recognized, "
                                                     "else layer 2), also when it is globally off; layer 2 only: no stripes; "
                                                     "native: ComfyUI's own decode for this VAE."}),
        }}

    def apply(self, vae, budget_gib=0.0, gn_scheme="default", stripe_rows=0, mode="default", budget="default"):
        if settings.disabled():
            if not _NOTED:
                _NOTED.append(True)
                logging.info(msg("node.vae_disabled"))
            return (vae,)
        b = str(budget or "default").strip().lower()
        if b not in BUDGET_CHOICES:
            raise ValueError(msg("node.bad_choice", item="budget", value=budget, choices="/".join(BUDGET_CHOICES)))
        if b == "custom":
            if not budget_gib or float(budget_gib) <= 0:
                raise ValueError(msg("node.custom_zero", gib=budget_gib))
            value = budget_gib
        else:
            value = "unlimited" if b == "unlimited" else 0.0
            if budget_gib and float(budget_gib) > 0:
                logging.info(msg("node.budget_unused", gib=budget_gib, budget=msg("v.follow") if b == "default" else msg("vae.unlimited")))
        return (with_settings(vae, budget=value, gn_scheme=gn_scheme, stripe_rows=stripe_rows, mode=mode),)
