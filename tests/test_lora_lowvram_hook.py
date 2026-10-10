"""A hook LoRA on a key that also has a normal LoRA, on a lowvram layer
(native LowVramPatch in its weight_function): the normal LoRA is applied once,
by native's LowVramPatch, and the hook by Monoload's runtime patch in front of
it (native's order: the hook merged into the stored weight, the LoRA added
when the layer runs). Before the fix the runtime patch applied the normal
LoRA too, so it was added twice. Compared with native ComfyUI (Monoload's
methods uninstalled) on a two-layer comfy.ops model, CPU, no model files:
lowvram and full load, keys with a normal LoRA only / a hook only / both,
exact and fused; no weight backups, the weights never changed. Also a native
partial unload after the hooks were applied (review 2026-10 item 04): the
hook must survive on the keys that also have a normal LoRA. After the hooks
are taken off the modules stay flagged as loaded, so a native partial unload
frees what native frees (item 05).

    MODELS=/tmp/nomodels tests/docker_run.sh python tests/test_lora_lowvram_hook.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import check, finish  # noqa: E402
import torch  # noqa: E402

import comfy.hooks  # noqa: E402
import comfy.model_patcher  # noqa: E402
from monoload import hotpatch  # noqa: E402
from test_master_switch import make_lora, make_net, same, weights  # noqa: E402

CPU = torch.device("cpu")
X = torch.randn(4, 96, generator=torch.Generator().manual_seed(3))


def run(base_keys, hook_keys, lowvram):
    net = make_net()
    w0 = weights(net)
    p = comfy.model_patcher.ModelPatcher(net, CPU, CPU)
    p.add_patches({k: v for k, v in make_lora(5).items() if k in base_keys}, 0.8)
    group = comfy.hooks.create_hook_lora(None, 0.6, 0.0)
    p.add_hook_patches(group.hooks[0], {k: v for k, v in make_lora(6).items() if k in hook_keys}, 0.6)
    p.hook_mode = comfy.hooks.EnumHookMode.MinVram
    if lowvram:
        p.load(CPU, lowvram_model_memory=1)
    else:
        p.load(CPU, full_load=True)
    p.patch_hooks(group)
    y = net(X)
    info = {"fns": sorted(type(f).__name__ for m in (net.a, net.b) for f in (m.__dict__.get("weight_function") or [])),
            "weights untouched while patched": same(weights(net), w0), "backups": len(p.backup) + len(p.hook_backup)}
    p.unpatch_hooks()
    y_off = net(X)
    p.unpatch_model(CPU)
    info["restored"] = same(weights(net), w0)
    return y, y_off, info


def run_unload(base_keys, hook_keys):
    """Review 2026-10 item 04: full load, hooks applied, then a native partial unload of everything (free_memory under
    memory pressure) and the sampler's next apply_hooks(same group), as ComfyUI does every step."""
    net = make_net()
    w0 = weights(net)
    p = comfy.model_patcher.ModelPatcher(net, CPU, CPU)
    p.add_patches({k: v for k, v in make_lora(5).items() if k in base_keys}, 0.8)
    group = comfy.hooks.create_hook_lora(None, 0.6, 0.0)
    p.add_hook_patches(group.hooks[0], {k: v for k, v in make_lora(6).items() if k in hook_keys}, 0.6)
    p.hook_mode = comfy.hooks.EnumHookMode.MinVram
    p.load(CPU, full_load=True)
    p.apply_hooks(group)
    p.partially_unload(CPU, memory_to_free=1 << 40)
    p.apply_hooks(group)
    y = net(X)
    info = {"fns": sorted(type(f).__name__ for m in (net.a, net.b) for f in (m.__dict__.get("weight_function") or [])),
            "backups": len(p.backup) + len(p.hook_backup)}
    p.unpatch_hooks()
    y_off = net(X)
    p.unpatch_model(CPU)
    info["restored"] = same(weights(net), w0)
    return y, y_off, info


def run_flags(base_keys, hook_keys):
    """Review 2026-10 item 05: after the hooks are taken off (every sampling ends with cleanup -> unpatch_hooks) the
    fully loaded modules keep comfy_patched_weights, so a native partial unload can still offload them."""
    net = make_net()
    p = comfy.model_patcher.ModelPatcher(net, CPU, CPU)
    p.add_patches({k: v for k, v in make_lora(5).items() if k in base_keys}, 0.8)
    group = comfy.hooks.create_hook_lora(None, 0.6, 0.0)
    p.add_hook_patches(group.hooks[0], {k: v for k, v in make_lora(6).items() if k in hook_keys}, 0.6)
    p.load(CPU, full_load=True)
    p.apply_hooks(group)
    p.unpatch_hooks()
    flags = {n: getattr(getattr(net, n), "comfy_patched_weights", None) for n in ("a", "b")}
    freed = p.partially_unload(CPU, memory_to_free=1 << 40)
    p.unpatch_model(CPU)
    return flags, freed


def main():
    cases = []
    for lowvram in (True, False):
        for base_keys, hook_keys, label in ((("b.weight",), ("a.weight", "b.weight"), "LoRA on b, hook on a and b"),
                                            (("a.weight", "b.weight"), ("b.weight",), "LoRA on a and b, hook on b")):
            cases.append((lowvram, base_keys, hook_keys, label))
    flag_cases = ((("b.weight",), ("a.weight", "b.weight"), "LoRA on b, hook on a and b"), ((), ("a.weight", "b.weight"), "hooks only"))
    unload_cases = [(b, h, label) for lv, b, h, label in cases if lv]
    hotpatch.uninstall()
    try:
        ref_flags = [run_flags(b, h) for b, h, _ in flag_cases]
        ref = [run(b, h, lv) for lv, b, h, _ in cases]
        ref_unload = [run_unload(b, h) for b, h, _ in unload_cases]
    finally:
        hotpatch.install()
    for exact in (True, False):
        hotpatch.set_exact(exact)
        for (lv, b, h, label), (r, r_off, rinfo) in zip(cases, ref):
            y, y_off, info = run(b, h, lv)
            d, d_off = float((y - r).abs().max()), float((y_off - r_off).abs().max())
            tol = 0.0 if exact else 2e-4
            check("{} {}, {}: == native (max|Δ| {:.2g}; hooks off {:.2g}); weight functions {} (native {}); no backups, weights never changed".format(
                "exact" if exact else "fused", "lowvram" if lv else "full load", label, d, d_off, info["fns"], rinfo["fns"]),
                d <= tol and d_off <= tol and info["backups"] == 0 and info["weights untouched while patched"] and info["restored"]
                and (not lv or info["fns"].count("LowVramPatch") == rinfo["fns"].count("LowVramPatch") > 0))
        for (b, h, label), (r, r_off, rinfo) in zip(unload_cases, ref_unload):
            y, y_off, info = run_unload(b, h)
            d, d_off = float((y - r).abs().max()), float((y_off - r_off).abs().max())
            tol = 0.0 if exact else 2e-4
            check("{} full load, hooks applied, partially_unload of everything, apply_hooks(same group), {}: == native (max|Δ| {:.2g}; "
                  "hooks off {:.2g}); weight functions {} (native {}, the same LowVramPatches); no backups".format(
                      "exact" if exact else "fused", label, d, d_off, info["fns"], rinfo["fns"]),
                  d <= tol and d_off <= tol and info["backups"] == 0 and info["restored"]
                  and info["fns"].count("LowVramPatch") == rinfo["fns"].count("LowVramPatch") > 0)
    hotpatch.set_exact(False)
    for (b, h, label), ref_f in zip(flag_cases, ref_flags):
        got = run_flags(b, h)
        check("{}: after unpatch_hooks the modules keep comfy_patched_weights {} and partially_unload frees {} bytes (native {} / {})".format(
              label, got[0], got[1], ref_f[0], ref_f[1]), got == ref_f and ref_f[1] > 0)
    finish()


if __name__ == "__main__":
    main()
