"""ComfyUI custom node entry point for Monoload.

Importing this package (ComfyUI does so at startup, before any model is
loaded) installs the runtime LoRA merge on comfy.model_patcher.ModelPatcher.
Set MONOLOAD_DISABLE=1 to leave ComfyUI completely native.
"""

import logging
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if os.environ.get("MONOLOAD_DISABLE", "").strip().lower() in ("1", "true", "yes", "on"):
    logging.info("[Monoload] MONOLOAD_DISABLE is set: runtime LoRA merge NOT installed, ComfyUI stays native")
else:
    from .monoload import hotpatch

    hotpatch.install()
    logging.info("[Monoload] runtime LoRA merge installed on ModelPatcher (no in-place LoRA, no weight backups)")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
