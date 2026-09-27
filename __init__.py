"""ComfyUI custom node entry point for Monoload.

Importing this package (ComfyUI does so at startup, before any model is
loaded) installs
  * the runtime LoRA merge on comfy.model_patcher.ModelPatcher, and
  * the per-prompt LoRA release on execution.PromptExecutor
    (skipped when MONOLOAD_KEEP_LORA=1).
Set MONOLOAD_DISABLE=1 to leave ComfyUI completely native.
"""

import logging
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


if _flag("MONOLOAD_DISABLE"):
    logging.info("[Monoload] MONOLOAD_DISABLE is set: runtime LoRA merge NOT installed, ComfyUI stays native")
else:
    from .monoload import hotpatch, release

    hotpatch.install()
    logging.info("[Monoload] runtime LoRA merge installed on ModelPatcher (no in-place LoRA, no weight backups)")
    if _flag("MONOLOAD_KEEP_LORA"):
        logging.info("[Monoload] MONOLOAD_KEEP_LORA is set: LoRA state is kept between prompts")
    else:
        release.install()
        logging.info("[Monoload] LoRA is released after every prompt (base models stay loaded)")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
