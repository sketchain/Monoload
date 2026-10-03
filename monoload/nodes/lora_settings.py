"""Monoload LoRA Settings: how Monoload handles the LoRA of one model.

Typical place: LoRA loader(s) -> Monoload LoRA Settings -> KSampler (it works
before the loaders too: the settings live in the patcher's model_options,
which every ModelPatcher.clone() -- the loaders' included -- copies).

Inputs MODEL (required) and CLIP (optional); outputs MODEL and CLIP (None
when no CLIP is connected). The outputs are ComfyUI clones (weights shared,
the inputs unchanged) carrying the settings (monoload/lora_overrides.py):
  mode          default (follow the global setting: MONOLOAD) / enable
                (Monoload's runtime merge for this model, also with
                MONOLOAD=0) / native (ComfyUI's own LoRA handling)
  merge         default (follow MONOLOAD_EXACT) / fused (the fused fp16
                addmm default path) / exact (bit-identical to native ComfyUI)
  after_prompt  default (follow the global setting: release unless
                MONOLOAD_KEEP_LORA=1; a native model keeps it) / release /
                keep
Priority, item by item: the node's choice > the global default > built-in.
Chained nodes: what a node leaves at default keeps the upstream node's value.

MONOLOAD_DISABLE=1 (nothing installed): the node returns its inputs
unchanged and says so once.
"""

import logging

from .. import lora_overrides, settings
from ..lora_overrides import AFTER_CHOICES, MERGE_CHOICES, MODE_CHOICES

_NOTED = []


class MonoloadLoRASettings:
    DISPLAY_NAME = "Monoload LoRA Settings"
    CATEGORY = "Monoload"
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply"
    DESCRIPTION = ("How Monoload handles the LoRA of this model (and CLIP): a clone (weights shared, the inputs unchanged) "
                   "with its own settings. Each setting left at default follows the global setting (the environment "
                   "variables), then Monoload's built-in default. Works after or before the LoRA loaders.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model (usually after the LoRA loaders); not changed, the output is a clone."}),
                "mode": (list(MODE_CHOICES), {"default": "default",
                                              "tooltip": "default: follow the global setting (MONOLOAD=0 -> native); enable: Monoload's "
                                                         "runtime LoRA merge for this model (no weight backups), also when it is "
                                                         "globally off; native: ComfyUI's own LoRA handling for this model."}),
                "merge": (list(MERGE_CHOICES), {"default": "default",
                                                "tooltip": "default: follow the global setting (MONOLOAD_EXACT); fused: one fused fp16 "
                                                           "addmm per plain LoRA (fast, rounding-level differences); exact: "
                                                           "bit-identical to native ComfyUI (slower per step)."}),
                "after_prompt": (list(AFTER_CHOICES), {"default": "default",
                                                       "tooltip": "default: follow the global setting (release unless "
                                                                  "MONOLOAD_KEEP_LORA=1; a native model keeps it); release: drop this "
                                                                  "model's LoRA state after every prompt (the base model stays loaded); "
                                                                  "keep: keep it for the next prompt."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Optional: the CLIP (text encoder) to get the same settings."}),
            },
        }

    def apply(self, model, mode="default", merge="default", after_prompt="default", clip=None):
        if settings.disabled():
            if not _NOTED:
                _NOTED.append(True)
                logging.info("[Monoload] Monoload LoRA Settings: MONOLOAD_DISABLE is set, so the node passes MODEL and CLIP through unchanged")
            return (model, clip)
        m = lora_overrides.with_settings(model, mode=mode, merge=merge, after_prompt=after_prompt)
        c = lora_overrides.with_settings_clip(clip, mode=mode, merge=merge, after_prompt=after_prompt) if clip is not None else None
        logging.info("[Monoload] Monoload LoRA Settings: {}".format(lora_overrides.note(*lora_overrides.resolve(m))))
        return (m, c)
