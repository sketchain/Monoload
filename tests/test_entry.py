"""Plugin entry: loading the custom node the way ComfyUI does installs the
runtime merge on ModelPatcher; MONOLOAD_DISABLE=1 leaves ComfyUI native.

    python tests/test_entry.py            # expects installed
    MONOLOAD_DISABLE=1 python tests/test_entry.py
    MONOLOAD_KEEP_LORA=1 python tests/test_entry.py
    MONOLOAD_EXACT=1 python tests/test_entry.py
"""

import asyncio
import os

from common import check, finish
import comfy.model_patcher
import execution
import nodes

native = {n: comfy.model_patcher.ModelPatcher.__dict__[n] for n in ("load", "patch_weight_to_device", "unpatch_model", "patch_hooks")}
native_exec = execution.PromptExecutor.execute_async
ok = asyncio.run(nodes.load_custom_node("/opt/ComfyUI/custom_nodes/monoload"))
check("custom node imported by ComfyUI's loader", ok)
disabled = os.environ.get("MONOLOAD_DISABLE", "") == "1"
keep = os.environ.get("MONOLOAD_KEEP_LORA", "") == "1"
release_hooked = execution.PromptExecutor.execute_async is not native_exec
now = {n: comfy.model_patcher.ModelPatcher.__dict__[n] for n in native}
changed = [n for n in native if now[n] is not native[n]]
if disabled:
    check("MONOLOAD_DISABLE=1: ModelPatcher untouched", not changed, str(changed))
    check("MONOLOAD_DISABLE=1: PromptExecutor untouched", not release_hooked)
else:
    check("installed: ModelPatcher methods replaced by Monoload", len(changed) == len(native) and all("hotpatch" in now[n].__module__ for n in changed), str(changed))
    check("CoreModelPatcher alias covered", comfy.model_patcher.CoreModelPatcher.patch_weight_to_device is now["patch_weight_to_device"])
    from monoload import hotpatch
    exact = os.environ.get("MONOLOAD_EXACT", "") == "1"
    check("merge path: {}".format("bit-exact (MONOLOAD_EXACT=1)" if exact else "fused/relaxed default"), hotpatch.is_exact() == exact)
    if keep:
        check("MONOLOAD_KEEP_LORA=1: per-prompt release NOT installed", not release_hooked)
    else:
        check("per-prompt LoRA release installed on PromptExecutor.execute_async", release_hooked)
finish()
