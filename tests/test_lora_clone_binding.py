"""Runtime LoRA patches when clones share a model (review 01, DESIGN §7): the
runtime patches compute with the model's binding (monoload/hotpatch.py
_Binding: the active patcher's patches and merge, the hook patches in effect,
device copies), which load / partially_load / install point at the patcher
being loaded -- also when a clone with the same patches_uuid loads without
unpatching (native partially_load returns early). Before, a kept runtime patch
read the hook state, patches and merge of the clone that installed it.

Everything is compared with native ComfyUI (Monoload's methods uninstalled):
exact bit for bit, fused within 5e-4; keys with a normal LoRA only, a hook
only, and both; ComfyUI's real load_models_gpu / partially_load / apply_hooks /
cleanup. No weight backups, the weights never changed. CPU, no model files.

  1 MODEL, hooks registered on the patcher at node time (as
    comfy.hooks.load_hook_lora_for_models), B = A.clone() (same uuid):
    A -> B and B -> A with cleanup in between; B removes the hook; A cleans up
    after B was activated
  2 CLIP, scheduled keyframes, encode with A then its clone B
  3 CLIP forced hooks, same uuid, different hook groups: A -> B -> A -> a clone
    without hooks; unpatch_hooks(whitelist)
  4 ComfyUI's nodes: LoraLoader -> SetHookKeyframes -> SetClipHooks(schedule_clip)
    -> CLIPSetLastLayer / Monoload LoRA Settings (default), encode A, B, C, A
  5 merge: a same-uuid clone with another merge (constructed; the node rolls
    the uuid): the runtime patches follow the patcher being loaded
  6 lifecycle: the binding releases a clone that is gone; no device copies left

    MODELS=/tmp/nomodels tests/docker_run.sh python tests/test_lora_clone_binding.py
"""

import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import check, finish  # noqa: E402
import torch  # noqa: E402

import comfy.hooks  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.sd  # noqa: E402
import nodes  # noqa: E402
from comfy_extras import nodes_hooks  # noqa: E402
from monoload import hotpatch, lora_overrides  # noqa: E402
from test_master_switch import Net, make_lora, make_net, same, weights  # noqa: E402

CPU = torch.device("cpu")
X = torch.randn(4, 96, generator=torch.Generator().manual_seed(3))
LAYOUTS = (("LoRA on b, hook on a and b", ("b.weight",), ("a.weight", "b.weight")),
           ("LoRA on a and b, hook on b", ("a.weight", "b.weight"), ("b.weight",)))


def only(patches, keys):
    return {k: v for k, v in patches.items() if k in keys}


class Watch:
    """Backups and weight changes over a scenario (Monoload must make none)."""

    def __init__(self, net):
        self.net, self.w0, self.patchers, self.ok = net, weights(net), [], True

    def step(self):
        self.ok = self.ok and same(weights(self.net), self.w0) if hotpatch.is_installed() else self.ok

    def result(self):
        backups = sum(len(p.backup) + len(p.hook_backup) for p in self.patchers)
        return {"unchanged": self.ok and same(weights(self.net), self.w0), "backups": backups}


def model_scenarios(info):
    out = {}
    for lname, base, hooked in LAYOUTS:
        for order in ("A,B", "B,A"):
            mm.unload_all_models()
            net = make_net()
            w = Watch(net)
            L = comfy.model_patcher.ModelPatcher(net, CPU, CPU)
            L.add_patches(only(make_lora(5), base), 0.8)
            H = comfy.hooks.create_hook_lora(None, 0.6, 0.0)
            A = L.clone()
            A.add_hook_patches(H.hooks[0], only(make_lora(6), hooked), 0.6)
            B = A.clone()
            w.patchers += [A, B]
            info["MODEL same uuid"] = A.patches_uuid == B.patches_uuid
            seq = (("A", A), ("B", B)) if order == "A,B" else (("B", B), ("A", A))
            for name, p in seq:
                mm.load_models_gpu([p])
                p.apply_hooks(H)
                out["1 {} {} {}".format(lname, order, name)] = net(X)
                w.step()
                p.cleanup()
            out["1 {} {} after cleanup".format(lname, order)] = net(X)
            # B removes the hook (samples with none); then A cleans up after B was activated with H again
            mm.load_models_gpu([B])
            B.apply_hooks(None)
            out["1 {} {} B without hook".format(lname, order)] = net(X)
            mm.load_models_gpu([A])
            A.apply_hooks(H)
            mm.load_models_gpu([B])
            B.apply_hooks(H)
            A.cleanup()
            out["1 {} {} B after A cleaned up".format(lname, order)] = net(X)
            w.step()
            mm.unload_all_models()
            info["1 {} {}".format(lname, order)] = w.result()
    return out


def clip_patcher(net, base):
    L = comfy.model_patcher.ModelPatcher(net, CPU, CPU)
    L.is_clip = True
    L.hook_mode = comfy.hooks.EnumHookMode.MinVram   # as comfy.sd.CLIP sets it
    L.add_patches(only(make_lora(5), base), 0.8)
    return L


def clip_scheduled(info):
    out = {}
    for lname, base, hooked in LAYOUTS:
        mm.unload_all_models()
        net = make_net()
        w = Watch(net)
        L = clip_patcher(net, base)
        H = comfy.hooks.create_hook_lora(None, 0.0, 1.0)
        kf = comfy.hooks.HookKeyframeGroup()
        kf.add(comfy.hooks.HookKeyframe(1.0, 0.0))
        kf.add(comfy.hooks.HookKeyframe(0.25, 0.5))
        H.set_keyframes_on_hooks(kf)
        A = L.clone()
        A.forced_hooks = H.clone()
        A.add_hook_patches(H.hooks[0], only(make_lora(6), hooked), 1.0)
        B = A.clone()
        w.patchers += [A, B]
        for name, p in (("A", A), ("B", B), ("A again", A)):
            mm.load_models_gpu([p])
            hooks = p.forced_hooks
            sched = hooks.get_hooks_for_clip_schedule()
            hooks.reset()
            p.patch_hooks(None)
            for i, (_rng, hk) in enumerate(sched):
                for hook, k in hk:
                    hook.hook_keyframe._current_keyframe = k
                p.patch_hooks(hooks)
                out["2 {} {} kf{}".format(lname, name, i)] = net(X)
                w.step()
            hooks.reset()
        mm.unload_all_models()
        info["2 " + lname] = w.result()
    return out


def clip_switch(info):
    out = {}
    for lname, base, hooked in LAYOUTS:
        mm.unload_all_models()
        net = make_net()
        w = Watch(net)
        L = clip_patcher(net, base)
        Ha = comfy.hooks.create_hook_lora(None, 0.0, 1.0)
        Hb = comfy.hooks.create_hook_lora(None, 0.0, 1.0)
        A = L.clone()
        A.add_hook_patches(Ha.hooks[0], only(make_lora(6), hooked), 0.7)
        A.add_hook_patches(Hb.hooks[0], only(make_lora(7), ("a.weight", "b.weight")), -0.5)
        B, N = A.clone(), A.clone()
        A.forced_hooks = Ha.clone_and_combine(Hb)
        B.forced_hooks = Ha.clone()
        N.forced_hooks = None
        w.patchers += [A, B, N]
        for name, p in (("A", A), ("B", B), ("A again", A), ("N without hooks", N)):
            mm.load_models_gpu([p])        # is_clip: apply_hooks(forced_hooks, force_apply=True) on load or early return
            out["3 {} {}".format(lname, name)] = net(X)
            w.step()
        mm.load_models_gpu([A])
        A.unpatch_hooks({"a.weight"})
        out["3 {} A unpatch_hooks(a)".format(lname)] = net(X)
        mm.unload_all_models()
        info["3 " + lname] = w.result()
    return out


class TE(Net):
    def reset_clip_options(self):
        pass

    def set_clip_options(self, o):
        pass

    def encode_token_weights(self, tokens):
        y = self(X)
        return y, y.sum(-1)


def make_clip():
    g = torch.Generator().manual_seed(11)
    te = TE()
    with torch.no_grad():
        for p in te.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    c = comfy.sd.CLIP(no_init=True)
    c.patcher = comfy.model_patcher.ModelPatcher(te, CPU, CPU)
    c.patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
    c.patcher.is_clip = True
    c.cond_stage_model = te
    c.tokenizer = None
    c.layer_idx = None
    c.tokenizer_options = {}
    c.use_clip_schedule = False
    c.apply_hooks_to_conds = None
    return c


def clip_nodes(info):
    out = {}
    for sched in (True, False):
        mm.unload_all_models()
        clip0 = make_clip()
        w = Watch(clip0.cond_stage_model)
        clipL = clip0.clone()
        clipL.add_patches(only(make_lora(5), ("b.weight",)), 0.8)                      # as LoraLoader
        H = comfy.hooks.create_hook_lora(None, 0.0, 1.0)
        H.hooks[0].weights_clip = make_lora(6)
        H.hooks[0].need_weight_init = False
        kf = nodes_hooks.CreateHookKeyframe().create_hook_keyframe(1.0, 0.0)[0]
        kf = nodes_hooks.CreateHookKeyframe().create_hook_keyframe(0.25, 0.5, kf)[0]
        Hkf = nodes_hooks.SetHookKeyframes().set_hook_keyframes(H, kf)[0]
        A = nodes_hooks.SetClipHooks().apply_hooks(clipL, schedule_clip=sched, apply_to_conds=False, hooks=Hkf)[0]
        B = nodes.CLIPSetLastLayer().set_last_layer(A, -1)[0]
        C = lora_overrides.with_settings_clip(A)                     # Monoload LoRA Settings, everything 'default'
        w.patchers += [A.patcher, B.patcher, C.patcher]
        info["4 uuid A==B==C (schedule_clip {})".format(sched)] = A.patcher.patches_uuid == B.patcher.patches_uuid == C.patcher.patches_uuid
        for name, c in (("A", A), ("B", B), ("C", C), ("A again", A)):
            for i, (cond, d) in enumerate(c.encode_from_tokens_scheduled({})):
                out["4 schedule_clip {} {} range{}".format(sched, name, i)] = cond
            w.step()
        mm.unload_all_models()
        info["4 schedule_clip {}".format(sched)] = w.result()
    return out


def merge_follows(info):
    """Monoload only: the runtime patches' merge follows the patcher being loaded."""
    mm.unload_all_models()
    net = make_net()
    P0 = comfy.model_patcher.ModelPatcher(net, CPU, CPU)
    L = P0.clone()
    L.add_patches(make_lora(5), 0.8)
    S = lora_overrides.with_settings(L, merge="exact")
    F = S.clone()
    F.model_options[lora_overrides.KEY] = {"merge": "fused"}       # same uuid, other merge: constructed by hand
    F2 = lora_overrides.with_settings(S, merge="fused")
    got = []
    for name, p, want in (("S exact", S, True), ("F fused, same uuid", F, False), ("F2 fused (node)", F2, False), ("S again", S, True)):
        mm.load_models_gpu([p])
        f = [f for f in net.b.weight_function if getattr(f, "is_monoload_patch", False)][0]
        got.append((name, f.exact, want, f.patches is p.patches))
    mm.unload_all_models()
    check("merge: the runtime patches' merge and patches follow the patcher loaded ({})".format(
        ", ".join("{} -> exact {}".format(n, e) for n, e, _, _ in got)), all(e == want and own for _, e, want, own in got))


def lifecycle():
    mm.unload_all_models()
    net = make_net()
    L = comfy.model_patcher.ModelPatcher(net, CPU, CPU)
    L.add_patches(make_lora(5), 0.8)
    H = comfy.hooks.create_hook_lora(None, 0.6, 0.0)
    A = L.clone()
    A.add_hook_patches(H.hooks[0], make_lora(6), 0.6)
    B = A.clone()
    mm.load_models_gpu([A])
    A.apply_hooks(H)
    mm.load_models_gpu([B])
    B.apply_hooks(H)
    b = net.__dict__.get("_monoload_binding")
    bound_b = b is not None and b.patches is B.patches
    # (A itself stays alive natively: B.parent is A.) Nothing of Monoload's refers to A's patches dict any more
    gc.collect()
    a_gone = not [r for r in gc.get_referrers(A.patches) if r is not A.__dict__ and not isinstance(r, type(sys._getframe()))]
    mm.unload_all_models()
    gc.collect()
    left = (len(b.patches), len(b.hook_patches), len(b.device_cache)) if b is not None else None
    check("lifecycle: after switching to B the binding holds B's patches ({}); nothing refers to A's patches dict but A ({}); "
          "after unloading the binding holds nothing (patches, hooks, device copies: {})".format(bound_b, a_gone, left),
          bound_b and a_gone and left == (0, 0, 0))


def run_all():
    info, out = {}, {}
    for fn in (model_scenarios, clip_scheduled, clip_switch, clip_nodes):
        out.update(fn(info))
    return out, info


def main():
    hotpatch.uninstall()
    try:
        ref, ref_info = run_all()
    finally:
        hotpatch.install()
    check("native: same uuid for the clones in every scenario ({})".format(
        {k: v for k, v in ref_info.items() if "uuid" in k}), all(v for k, v in ref_info.items() if "uuid" in k))
    for exact in (True, False):
        hotpatch.set_exact(exact)
        got, info = run_all()
        tag = "exact" if exact else "fused"
        groups = {}
        for k in ref:
            d = float((got[k] - ref[k]).abs().max())
            ok = torch.equal(got[k], ref[k]) if exact else d <= 5e-4
            g = k.split(" ")[0]
            groups.setdefault(g, []).append((k, d, ok))
        for g, items in sorted(groups.items()):
            bad = [(k, d) for k, d, ok in items if not ok]
            check("[{}] scenario {}: {} outputs == native (worst max|Δ| {:.2g}){}".format(
                tag, g, len(items), max(d for _, d, _ in items), "; differ: {}".format(bad[:3]) if bad else ""), not bad)
        res = {k: v for k, v in info.items() if isinstance(v, dict)}
        check("[{}] no weight backups, weights never changed, in every scenario ({} runs)".format(tag, len(res)),
              all(v["backups"] == 0 and v["unchanged"] for v in res.values()), str({k: v for k, v in res.items() if v["backups"] or not v["unchanged"]}))
    hotpatch.set_exact(False)
    merge_follows({})
    lifecycle()
    finish()


if __name__ == "__main__":
    main()
