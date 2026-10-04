"""The master switch MONOLOAD and the global defaults (monoload/settings.py).

No model files needed: a small comfy.ops model with a LoRA, and a small
SDXL-like VAE with random weights (tests/test_vae_ldm.py), fp32 on the CPU.

  1. parsing: MONOLOAD unset / 1 / true -> on, 0 / false / off -> off,
     anything else -> on with a warning;
  2. MONOLOAD=0 and no Monoload node: the installed hooks pass through, so a
     LoRA'd model gives bit for bit the outputs of native ComfyUI (hooks
     uninstalled) -- full load, partial (lowvram) load, hook LoRA -- with
     native's own weight handling (weights baked in, backups, restored
     bit-exactly on unpatch); with the switch on the same model runs through
     Monoload (no backups) as a control;
  3. MONOLOAD=0: VAE.decode == native bit for bit, nothing logged at INFO;
     the per-prompt LoRA release returns at once (no LoRA node used); the wrappers' own cost per
     call (measured against a stub) is microseconds;
  4. priority, item by item, node > global > built-in: under MONOLOAD=0, or
     with MONOLOAD_DISABLE_VAE / MONOLOAD_EXACT as the global default, a VAE
     copy whose node chose mode auto is managed (layer 1), one left at
     default is native; the budget dropdown (default / unlimited / custom);
  5. MONOLOAD_DISABLE=1: the node passes its input through unchanged and
     says so once.

    python tests/test_master_switch.py
"""

import asyncio
import logging
import os
import time

import torch

from common import check, expect_raises, finish
import comfy.hooks
import comfy.lora
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.sd
from monoload import hotpatch, release, settings
from monoload import vae as mvae
from monoload import vae_overrides as vo
from test_vae import native_decode
from test_vae_ldm import ldm_vae
from test_vae_node import LogCapture, node_apply, registration, reset_globals

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. parsing
# ---------------------------------------------------------------------------

def parse_tests():
    saved = os.environ.get("MONOLOAD")
    got = {}
    try:
        for raw in (None, "", "1", "true", "ON", "0", "false", "off", "maybe"):
            if raw is None:
                os.environ.pop("MONOLOAD", None)
            else:
                os.environ["MONOLOAD"] = raw
            with LogCapture() as cap:
                got[raw] = settings._master_from_env()
            if raw == "maybe":
                warned = any("not understood" in line for line in cap.lines)
    finally:
        if saved is None:
            os.environ.pop("MONOLOAD", None)
        else:
            os.environ["MONOLOAD"] = saved
    check("MONOLOAD parsing: {}".format(", ".join("{}->{}".format("unset" if k is None else repr(k), "on" if v else "off") for k, v in got.items())),
          [got[k] for k in (None, "", "1", "true", "ON")] == [True] * 5 and [got[k] for k in ("0", "false", "off")] == [False] * 3
          and got["maybe"] is True and warned)


# ---------------------------------------------------------------------------
# 2. LoRA: MONOLOAD=0 == native ComfyUI
# ---------------------------------------------------------------------------

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = comfy.ops.manual_cast.Linear(96, 64, dtype=torch.float16)
        self.b = comfy.ops.manual_cast.Linear(64, 32, dtype=torch.float16)

    def forward(self, x):
        return self.b(torch.nn.functional.gelu(self.a(x)))


def make_net():
    g = torch.Generator().manual_seed(11)
    net = Net()
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return net


def make_lora(seed, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for name, (o, i) in (("a", (64, 96)), ("b", (32, 64))):
        sd[name + ".lora_up.weight"] = (torch.randn(o, 8, generator=g) * scale).half()
        sd[name + ".lora_down.weight"] = (torch.randn(8, i, generator=g) * scale).half()
        sd[name + ".alpha"] = torch.tensor(4.0)
    return comfy.lora.load_lora(sd, {"a": "a.weight", "b": "b.weight"}, log_missing=False)


def weights(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a) and a.keys() == b.keys()


def lora_run():
    """Outputs of the LoRA'd model in three loading situations, the weights
    after each unpatch, and the backups native-style loading made."""
    x = torch.randn(4, 96, generator=torch.Generator().manual_seed(3))
    lora, hook_lora = make_lora(5), make_lora(6)
    out, info = {}, {}
    net = make_net()
    w0 = weights(net)

    p = comfy.model_patcher.ModelPatcher(net, CPU, CPU)
    p.add_patches(lora, 0.8)
    p.load(CPU, full_load=True)
    out["full"] = net(x)
    info["full backups"] = len(p.backup)
    p.unpatch_model(CPU)
    info["full restored"] = same(weights(net), w0)

    q = p.clone()
    q.add_patches(lora, -0.3)
    q.load(CPU, lowvram_model_memory=1)   # nothing fits: every patched layer through its weight function
    out["lowvram"] = net(x)
    q.unpatch_model(CPU)
    info["lowvram restored"] = same(weights(net), w0)

    h = comfy.model_patcher.ModelPatcher(net, CPU, CPU)
    group = comfy.hooks.create_hook_lora(None, 0.6, 0.0)
    h.add_hook_patches(group.hooks[0], hook_lora, 0.6)
    h.add_patches(lora, 0.5)
    h.load(CPU, full_load=True)
    h.patch_hooks(group)
    out["hooks"] = net(x)
    h.unpatch_hooks()
    out["hooks off"] = net(x)
    h.unpatch_model(CPU)
    info["hooks restored"] = same(weights(net), w0)
    info["weight functions left"] = sum(len(m.__dict__.get("weight_function", []) or []) for m in net.modules())
    return out, info


def lora_tests():
    hotpatch.uninstall()
    try:
        ref, ref_info = lora_run()
    finally:
        hotpatch.install()
    settings.set_master(False)
    try:
        got, info = lora_run()
    finally:
        settings.set_master(True)
    on, on_info = lora_run()
    eq = {k: torch.equal(got[k], ref[k]) for k in ref}
    check("MONOLOAD=0, no node: LoRA'd model == native ComfyUI bit for bit ({})".format(
          ", ".join("{} {}".format(k, "==" if v else "DIFFERS") for k, v in eq.items())), all(eq.values()))
    check("... with native's own weight handling: {} backups as native ({}), weights restored bit-exactly on unpatch, "
          "no weight functions left ({})".format(info["full backups"], ref_info["full backups"], info["weight functions left"]),
          info == ref_info and info["full backups"] > 0 and info["full restored"] and info["lowvram restored"] and info["hooks restored"])
    check("control, MONOLOAD on: the same model through Monoload's runtime merge (no backups: {}), outputs within fp tolerance of native "
          "(max|Δ| {:.2g})".format(on_info["full backups"], max(float((on[k] - ref[k]).abs().max()) for k in ref)),
          on_info["full backups"] == 0 and all(torch.allclose(on[k], ref[k], atol=2e-3, rtol=1e-2) for k in ref))


# ---------------------------------------------------------------------------
# 3. VAE / release / cost under MONOLOAD=0
# ---------------------------------------------------------------------------

def vae_native_tests(sd, lat):
    ref = native_decode(sd, lat)
    settings.set_master(False)
    try:
        with LogCapture() as cap:
            out = comfy.sd.VAE.decode(sd, lat)
        last = mvae.last_decode()
        eff, src = mvae.resolve_settings(sd)
    finally:
        settings.set_master(True)
    check("MONOLOAD=0, no node: VAE.decode == native bit for bit, recorded as native ({}), mode from the global setting "
          "({} {}), no INFO line".format(last.get("reason"), eff["mode"], eff.get("mode_env")),
          torch.equal(out, ref) and last.get("strategy") == "native" and eff["mode"] == "native" and src["mode"] == "env"
          and eff.get("mode_env") == "MONOLOAD=0" and not [line for line in cap.lines if "[Monoload]" in line])


def release_tests():
    import execution
    calls = []
    real = release._release_loaded_models
    saved = execution.PromptExecutor.execute_async

    async def fake_execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        return "done"

    release.uninstall()
    execution.PromptExecutor.execute_async = fake_execute
    release.install()
    release._release_loaded_models = lambda: calls.append(1) or 0
    try:
        runs = []
        for master, keep in ((True, False), (False, False), (True, True)):
            settings.set_master(master)
            release.set_keep(keep)
            n = len(calls)
            r = asyncio.run(execution.PromptExecutor.execute_async(object(), {}, "id"))
            runs.append((master, keep, len(calls) - n, r))
    finally:
        settings.set_master(True)
        release.set_keep(False)
        release._release_loaded_models = real
        release.uninstall()
        execution.PromptExecutor.execute_async = saved
        release.install()
    check("per-prompt LoRA release: looks at the loaded models with the switch on; returns at once with MONOLOAD=0 and with "
          "MONOLOAD_KEEP_LORA, no LoRA node used ({})".format(", ".join("master {} keep {} -> {} scan(s)".format(m, k, n) for m, k, n, _ in runs)),
          [n for _, _, n, _ in runs] == [1, 0, 0] and all(r == "done" for *_, r in runs))


def cost_tests(sd, lat):
    """The wrappers' own cost under MONOLOAD=0, against the original method
    replaced by a stub (so only the wrapper is timed)."""
    n = 20000
    p = comfy.model_patcher.ModelPatcher(make_net(), CPU, CPU)
    p.add_patches(make_lora(5), 0.8)
    settings.set_master(False)
    saved = dict(hotpatch._ORIG)
    stub = lambda *a, **k: None
    hotpatch._ORIG["patch_weight_to_device"] = stub
    try:
        t0 = time.perf_counter()
        for _ in range(n):
            p.patch_weight_to_device("a.weight")
        t_wrap = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for _ in range(n):
            stub(p, "a.weight")
        t_stub = (time.perf_counter() - t0) / n
        saved_decode = mvae._ORIG["decode"]
        mvae._ORIG["decode"] = lambda self, s, o={}: s
        try:
            m = 2000
            t0 = time.perf_counter()
            for _ in range(m):
                comfy.sd.VAE.decode(sd, lat)
            t_vae = (time.perf_counter() - t0) / m
        finally:
            mvae._ORIG["decode"] = saved_decode
    finally:
        hotpatch._ORIG.clear()
        hotpatch._ORIG.update(saved)
        settings.set_master(True)
    extra = t_wrap - t_stub
    check("MONOLOAD=0 cost per call: ModelPatcher.patch_weight_to_device +{:.2f} µs (once per patched weight per load), "
          "VAE.decode wrapper {:.1f} µs (once per decode)".format(extra * 1e6, t_vae * 1e6),
          extra < 20e-6 and t_vae < 500e-6)


# ---------------------------------------------------------------------------
# 4. node > global > built-in under the new global defaults
# ---------------------------------------------------------------------------

def priority_tests(cls, sd, lat):
    ref = native_decode(sd, lat)
    auto = node_apply(cls, sd, mode="auto")
    plain = node_apply(cls, sd)
    for label, setup, undo in (("MONOLOAD=0", lambda: settings.set_master(False), lambda: settings.set_master(True)),
                               ("MONOLOAD_DISABLE_VAE=1", lambda: mvae.set_native("MONOLOAD_DISABLE_VAE=1"), lambda: mvae.set_native(None)),
                               ("MONOLOAD_EXACT=1", lambda: mvae.set_native("MONOLOAD_EXACT=1"), lambda: mvae.set_native(None))):
        setup()
        try:
            out_auto = comfy.sd.VAE.decode(auto, lat)
            last_auto = mvae.last_decode()
            out_plain = comfy.sd.VAE.decode(plain, lat)
            last_plain = mvae.last_decode()
            out_sd = comfy.sd.VAE.decode(sd, lat)
        finally:
            undo()
        check("{} globally: node mode auto -> managed (layer 1, mode from the node); node left at default and the plain VAE -> "
              "native (== native bit for bit)".format(label),
              last_auto.get("strategy") == "layer1" and last_auto["settings_source"]["mode"] == "node"
              and last_plain.get("strategy") == "native" and last_plain["settings_source"]["mode"] == "env"
              and torch.equal(out_plain, ref) and torch.equal(out_sd, ref) and float((out_auto - ref).abs().max()) <= 1e-4)
    mvae.set_budget(2 << 30)
    try:
        rows = {}
        for b, gib in (("default", 0.0), ("unlimited", 0.0), ("custom", 1.5), ("default", 7.0)):
            with LogCapture() as cap:
                v = node_apply(cls, sd, budget=b, budget_gib=gib)
            eff, src = mvae.resolve_settings(v)
            rows[(b, gib)] = (eff["budget"], src["budget"], any("not used" in line for line in cap.lines))
    finally:
        reset_globals()
    check("budget dropdown with MONOLOAD_VAE_BUDGET=2G: default -> 2 GiB (env), unlimited -> none (node), custom 1.5 -> 1.5 GiB (node), "
          "default with budget_gib 7 -> 2 GiB (env, budget_gib not used, logged)",
          rows[("default", 0.0)] == (2 << 30, "env", False) and rows[("unlimited", 0.0)] == (None, "node", False)
          and rows[("custom", 1.5)] == (int(1.5 * (1 << 30)), "node", False) and rows[("default", 7.0)] == (2 << 30, "env", True))
    v = node_apply(cls, sd, budget="unlimited")
    note = mvae.settings_note(*mvae.resolve_settings(v))
    check("... the log names it: {}".format(note[:60]), "budget unlimited (node)" in note)
    expect_raises("budget custom with budget_gib 0 -> ValueError", ValueError, lambda: node_apply(cls, sd, budget="custom", budget_gib=0.0), "budget_gib")
    up = node_apply(cls, sd, budget="custom", budget_gib=3.0)
    down = node_apply(cls, up, budget="unlimited")
    check("chained: a downstream unlimited overrides an upstream custom budget ({} -> {})".format(vo.overrides(up), vo.overrides(down)),
          vo.overrides(up) == {"budget": 3 << 30} and vo.overrides(down) == {"budget": None})


def disable_tests(cls, sd):
    from monoload.nodes import vae_settings as nv
    saved = os.environ.get("MONOLOAD_DISABLE")
    os.environ["MONOLOAD_DISABLE"] = "1"
    nv._NOTED.clear()
    try:
        with LogCapture() as cap:
            a = node_apply(cls, sd, mode="auto", budget="custom", budget_gib=1.0)
            b = node_apply(cls, sd, gn_scheme="A")
    finally:
        if saved is None:
            os.environ.pop("MONOLOAD_DISABLE")
        else:
            os.environ["MONOLOAD_DISABLE"] = saved
        nv._NOTED.clear()
    notes = [line for line in cap.lines if "Monoload VAE Settings" in line]
    check("MONOLOAD_DISABLE=1: the node returns its input VAE itself, unchanged, and says so once ({})".format(notes[0][:90] if notes else ""),
          a is sd and b is sd and vo.overrides(sd) == {} and len(notes) == 1)


def main():
    if not mvae.is_installed():
        check("vae.install() on this ComfyUI", mvae.install())
    if not hotpatch.is_installed():
        hotpatch.install()
    if not release.is_installed():
        release.install()
    reset_globals()
    mvae.set_native(None)
    settings.set_master(True)
    parse_tests()
    lora_tests()
    cls = registration()
    sd = ldm_vae(4, True)
    lat = torch.randn(1, 4, 12, 10, generator=torch.Generator().manual_seed(7))
    vae_native_tests(sd, lat)
    release_tests()
    cost_tests(sd, lat)
    priority_tests(cls, sd, lat)
    disable_tests(cls, sd)
    finish()


if __name__ == "__main__":
    main()
