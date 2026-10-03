"""Plugin entry: loading the custom node the way ComfyUI does installs the
runtime merge on ModelPatcher, the per-prompt LoRA release and the managed
VAE decode on comfy.sd.VAE in every combination except MONOLOAD_DISABLE=1,
which leaves ComfyUI native; the other switches are global defaults read at
runtime.

    python tests/test_entry.py            # expects installed, master switch on
    MONOLOAD=0 python tests/test_entry.py # installed, master switch off: everything native by default
    MONOLOAD_DISABLE=1 python tests/test_entry.py
    MONOLOAD_KEEP_LORA=1 python tests/test_entry.py   # release installed, global default "keep"
    MONOLOAD_EXACT=1 python tests/test_entry.py       # bit-exact merge, VAE decode native by default
    MONOLOAD_DISABLE_VAE=1 python tests/test_entry.py # VAE decode native by default, LoRA part as usual
    MONOLOAD_DISABLE_VAE_STRIPE=1 python tests/test_entry.py  # VAE layer 1 off, layer 2 on
    MONOLOAD_VAE_BUDGET=2G MONOLOAD_VAE_STRIPE_ROWS=64 python tests/test_entry.py
    MONOLOAD_VAE_GN_SCHEME=D python tests/test_entry.py   # LDM layer-1 GroupNorm scheme
The Monoload VAE Settings node is registered in every combination.
"""

import asyncio
import os

from common import check, finish
import comfy.model_patcher
import comfy.sd
import execution
import nodes

native = {n: comfy.model_patcher.ModelPatcher.__dict__[n] for n in ("load", "patch_weight_to_device", "unpatch_model", "patch_hooks")}
native_exec = execution.PromptExecutor.execute_async
native_vae_decode = comfy.sd.VAE.decode
native_vae_tiled = (comfy.sd.VAE.decode_tiled, comfy.sd.VAE.decode_tiled_)
ok = asyncio.run(nodes.load_custom_node("/opt/ComfyUI/custom_nodes/monoload"))
check("custom node imported by ComfyUI's loader", ok)
disabled = os.environ.get("MONOLOAD_DISABLE", "") == "1"
keep = os.environ.get("MONOLOAD_KEEP_LORA", "") == "1"
release_hooked = execution.PromptExecutor.execute_async is not native_exec
vae_hooked = comfy.sd.VAE.decode is not native_vae_decode
check("VAE.decode_tiled / decode_tiled_ never touched", (comfy.sd.VAE.decode_tiled, comfy.sd.VAE.decode_tiled_) == native_vae_tiled)
node_cls = nodes.NODE_CLASS_MAPPINGS.get("MonoloadVAESettings")
check("node registered by ComfyUI's loader in every switch combination: MonoloadVAESettings ({}, category {})".format(
      nodes.NODE_DISPLAY_NAME_MAPPINGS.get("MonoloadVAESettings"), getattr(node_cls, "CATEGORY", None)),
      node_cls is not None and nodes.NODE_DISPLAY_NAME_MAPPINGS.get("MonoloadVAESettings") == "Monoload VAE Settings" and node_cls.CATEGORY == "Monoload")
flag = lambda n: os.environ.get(n, "").strip().lower() in ("1", "true", "yes", "on")
now = {n: comfy.model_patcher.ModelPatcher.__dict__[n] for n in native}
changed = [n for n in native if now[n] is not native[n]]
if disabled:
    check("MONOLOAD_DISABLE=1: ModelPatcher untouched", not changed, str(changed))
    check("MONOLOAD_DISABLE=1: PromptExecutor untouched", not release_hooked)
    check("MONOLOAD_DISABLE=1: VAE.decode untouched", not vae_hooked)
else:
    check("installed: ModelPatcher methods replaced by Monoload", len(changed) == len(native) and all("hotpatch" in now[n].__module__ for n in changed), str(changed))
    check("CoreModelPatcher alias covered", comfy.model_patcher.CoreModelPatcher.patch_weight_to_device is now["patch_weight_to_device"])
    from monoload import hotpatch, release, settings
    from monoload import vae
    exact = os.environ.get("MONOLOAD_EXACT", "") == "1"
    master = os.environ.get("MONOLOAD", "").strip() != "0"
    check("master switch MONOLOAD {}".format("on" if master else "off (MONOLOAD=0)"), settings.master() == master)
    check("merge path: {}".format("bit-exact (MONOLOAD_EXACT=1)" if exact else "fused/relaxed default"), hotpatch.is_exact() == exact)
    check("per-prompt LoRA release installed on PromptExecutor.execute_async; global default {}".format(
          "keep (MONOLOAD_KEEP_LORA=1)" if keep else "release"), release_hooked and release.keep() == keep and release.enabled() == (master and not keep))
    check("VAE.decode wrapped by Monoload (workspace {} bytes)".format(vae.workspace()),
          vae_hooked and comfy.sd.VAE.decode.__module__.endswith("monoload.vae") and comfy.sd.VAE.decode.__wrapped__ is native_vae_decode)
    want_mode = ("native", "MONOLOAD=0") if not master else ("native", "MONOLOAD_DISABLE_VAE=1") if flag("MONOLOAD_DISABLE_VAE") else \
        ("native", "MONOLOAD_EXACT=1") if exact else ("layer2", "MONOLOAD_DISABLE_VAE_STRIPE=1") if flag("MONOLOAD_DISABLE_VAE_STRIPE") else ("auto", None)
    check("VAE global default mode {} ({})".format(*vae.global_mode()), vae.global_mode() == want_mode)
    want = os.environ.get("MONOLOAD_VAE_WORKSPACE", "")
    if want:
        check("MONOLOAD_VAE_WORKSPACE={} honoured".format(want), vae.workspace() == vae.parse_size(want))
    no_stripe = flag("MONOLOAD_DISABLE_VAE_STRIPE")
    names = [a.__name__.rsplit(".", 1)[-1] for a in vae.STRIPE_ADAPTERS]
    check("VAE layer 1 (stripes) {}, adapters {}".format("off (MONOLOAD_DISABLE_VAE_STRIPE=1)" if no_stripe else "on", names),
          vae.stripe_enabled() == (not no_stripe) and names == ["vae_wan", "vae_ldm"])
    want = os.environ.get("MONOLOAD_VAE_GN_SCHEME", "").strip().upper()
    check("LDM GroupNorm scheme {} ({})".format(vae.gn_scheme(), "MONOLOAD_VAE_GN_SCHEME={}: forced".format(want) if want else "unset: default, not forced"),
          vae.gn_scheme() == (want or "B") and vae.gn_scheme_forced() == bool(want))
    want = os.environ.get("MONOLOAD_VAE_BUDGET", "")
    check("layer-1 budget {} ({})".format(vae.budget(), "MONOLOAD_VAE_BUDGET={}".format(want) if want else "unset: default stripe policy"),
          vae.budget() == (vae.parse_size(want) if want else None))
    want = os.environ.get("MONOLOAD_VAE_STRIPE_ROWS", "")
    check("stripe height {}".format("forced to {}".format(want) if want else "from the policy / budget"), vae.stripe_rows() == (int(want) if want else None))
finish()
