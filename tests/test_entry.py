"""Plugin entry: loading the custom node the way ComfyUI does installs the
runtime merge on ModelPatcher; MONOLOAD_DISABLE=1 leaves ComfyUI native.

    python tests/test_entry.py            # expects installed
    MONOLOAD_DISABLE=1 python tests/test_entry.py
"""

import asyncio
import os

from common import check, finish
import comfy.model_patcher
import nodes

native = {n: comfy.model_patcher.ModelPatcher.__dict__[n] for n in ("load", "patch_weight_to_device", "unpatch_model", "patch_hooks")}
ok = asyncio.run(nodes.load_custom_node("/opt/ComfyUI/custom_nodes/monoload"))
check("custom node imported by ComfyUI's loader", ok)
disabled = os.environ.get("MONOLOAD_DISABLE", "") == "1"
now = {n: comfy.model_patcher.ModelPatcher.__dict__[n] for n in native}
changed = [n for n in native if now[n] is not native[n]]
if disabled:
    check("MONOLOAD_DISABLE=1: ModelPatcher untouched", not changed, str(changed))
else:
    check("installed: ModelPatcher methods replaced by Monoload", len(changed) == len(native) and all("hotpatch" in now[n].__module__ for n in changed), str(changed))
    check("CoreModelPatcher alias covered", comfy.model_patcher.CoreModelPatcher.patch_weight_to_device is now["patch_weight_to_device"])
finish()
