"""The Monoload Info node (monoload/nodes/info.py, monoload/info.py, the
per-VAE decode records in monoload/vae.py, the LoRA names recorded by the
LoraLoader wrapper in monoload/lora_overrides.py).

No model files needed: a small SDXL-like VAE with random weights
(tests/test_vae_ldm.py), a two-layer comfy.ops model for MODEL, and a stand-in
loader class for the LoraLoader wrapper; CPU.

  1. interface: output node, all inputs optional (vae, model, images), STRING
     output, never cached (IS_CHANGED is NaN), the UI text == the output;
  2. no input: version / commit, master switch, every global default with its
     source (env / runtime / built-in); MONOLOAD=0;
  3. vae: settings with sources; "not decoded yet"; records belong to the
     VAE object -- a copy made by Monoload VAE Settings and the original are
     never mixed; native decodes recorded too; refreshed on every run;
  4. model: LoRA names and strengths from the loader (chained, MODEL only),
     patched weights, settings with sources, state in memory;
  5. combinations (vae + model + images, images only) and MONOLOAD_DISABLE=1
     never raise.

    python tests/test_info_node.py
"""

import os
import re
import time

import torch

from common import check, finish
import comfy.model_patcher
import comfy.sd
from monoload import info, lora_overrides as lo, settings
from monoload import vae as mvae
from test_master_switch import make_lora, make_net
from test_vae import native_decode
from test_vae_ldm import ldm_vae
from test_vae_node import node_apply, reset_globals

NODE = "MonoloadInfo"
CPU = torch.device("cpu")


def info_cls():
    from monoload.nodes import NODE_CLASS_MAPPINGS
    return NODE_CLASS_MAPPINGS[NODE]


def run_node(**kw):
    cls = info_cls()
    out = getattr(cls(), cls.FUNCTION)(**kw)
    return out


def text(**kw):
    out = run_node(**kw)
    return out["result"][0]


class FakeLoader:
    """What LoraLoader.load_lora does to the patchers: clones with patches."""

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
        m = model.clone()
        m.add_patches(make_lora(5), strength_model)
        return (m, clip)


class FakeLoaderModelOnly(FakeLoader):
    def load_lora_model_only(self, model, lora_name, strength_model):
        return (self.load_lora(model, None, lora_name, strength_model, 0)[0],)


def interface_tests():
    from monoload.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    cls = NODE_CLASS_MAPPINGS.get(NODE)
    it = cls.INPUT_TYPES()
    nan = cls.IS_CHANGED()
    out = run_node()
    check("node in the registry: {} ({}), category {}, output node {}, inputs required {} optional {}, returns {}".format(
          NODE, NODE_DISPLAY_NAME_MAPPINGS.get(NODE), cls.CATEGORY, cls.OUTPUT_NODE, list(it["required"]), list(it["optional"]), cls.RETURN_TYPES),
          NODE_DISPLAY_NAME_MAPPINGS.get(NODE) == "Monoload Info" and cls.CATEGORY == "Monoload" and cls.OUTPUT_NODE is True
          and it["required"] == {} and list(it["optional"]) == ["vae", "model", "images"] and cls.RETURN_TYPES == ("STRING",))
    check("never cached: IS_CHANGED is NaN (not equal to itself); UI text == STRING output",
          nan != nan and out["ui"]["text"] == [out["result"][0]] and isinstance(out["result"][0], str))


def global_tests():
    t = text()
    commit, branch = info.git_commit()
    lines = t.splitlines()
    check("no input: version and commit ({})".format(lines[0]), lines[0].startswith("Monoload 0.") and (commit is None or commit in lines[0]))
    want = ["LoRA mode", "LoRA merge", "LoRA after prompt", "VAE mode", "VAE budget", "VAE GroupNorm scheme", "VAE stripe rows", "VAE workspace"]
    rows = {w: next((l for l in lines if l.strip().startswith(w + " ")), "") for w in want}
    check("no input: master switch and every global default with its source:\n" + "\n".join(lines[1:]),
          "master switch MONOLOAD: on [built-in]" in t and all(rows[w] and "[built-in]" in rows[w] for w in want))
    os.environ["MONOLOAD_VAE_BUDGET"] = "3G"
    mvae.set_budget(3 << 30)
    settings.set_master(False)
    settings.set_exact(True)
    try:
        t = text()
    finally:
        os.environ.pop("MONOLOAD_VAE_BUDGET")
        reset_globals()
        settings.set_master(True)
        settings.set_exact(False)
    lines = t.splitlines()
    row = lambda w: next((l for l in lines if l.strip().startswith(w + " ")), "")
    check("sources follow the state: env MONOLOAD_VAE_BUDGET=3G, set at runtime (MONOLOAD=0 / exact by tests), VAE mode env MONOLOAD=0",
          "[env MONOLOAD_VAE_BUDGET=3G]" in row("VAE budget") and "3.00 GiB" in row("VAE budget")
          and "off" in t and "[set at runtime]" in row("LoRA merge") and "exact" in row("LoRA merge")
          and "native" in row("VAE mode") and "[env MONOLOAD=0]" in row("VAE mode"))


def vae_tests():
    sd = ldm_vae(4, True)
    lat = torch.randn(1, 4, 12, 10, generator=torch.Generator().manual_seed(7))
    cls_v = __import__("monoload.nodes", fromlist=["NODE_CLASS_MAPPINGS"]).NODE_CLASS_MAPPINGS["MonoloadVAESettings"]
    copy_ = node_apply(cls_v, sd, budget="custom", budget_gib=1024.0)
    t = text(vae=sd)
    check("vae never decoded: settings with sources, 'not decoded yet' ({})".format(next(l for l in t.splitlines() if "last decode" in l).strip()[:60]),
          "settings: mode auto [built-in], budget none (default policy) [built-in]" in t and "not decoded yet" in t)
    comfy.sd.VAE.decode(sd, lat)
    t_orig = text(vae=sd)
    t_copy = text(vae=copy_)
    check("after decoding the original only: the original shows its layer-1 decode, the copy still 'not decoded yet' "
          "(records belong to the VAE object)", "layer 1 (LDM stripes" in t_orig and "not decoded yet" in t_copy)
    check("the first decode of this structure in the process says it includes the first-use self-test",
          "(includes the first-use self-test)" in t_orig)
    comfy.sd.VAE.decode(copy_, lat)
    t_copy = text(vae=copy_)
    check("... a later decode does not", "self-test" not in t_copy)
    t_orig2 = text(vae=sd)
    last = next(l for l in t_copy.splitlines() if "last decode" in l)
    check("after decoding the copy: copy -> layer 2 with budget 1024 GiB [node]; the original still shows its own layer-1 decode\n  "
          + last.strip(), "layer 2 (op-level chunking)" in t_copy and "budget 1.00 TiB [node]" in t_copy.replace("1024.00 GiB", "1.00 TiB")
          and "layer 1 (LDM stripes" in t_orig2 and re.search(r"workspace .*, estimate .*, measured peak .*, [0-9.]+ s, OOM retries 0", last))
    check("the scheme the decode used is explained in one line: {}".format(next((l for l in t_orig2.splitlines() if "GroupNorm scheme B:" in l), "").strip()),
          "GroupNorm scheme B: keeps the H/4 and H/2 level outputs" in t_orig2)
    nat = node_apply(cls_v, sd, mode="native")
    native_decode(nat, lat)   # the original method: no record
    comfy.sd.VAE.decode(nat, lat)
    t = text(vae=nat)
    check("native decode recorded: {}".format(next(l for l in t.splitlines() if "last decode" in l).strip()[:110]), "native ComfyUI decode (mode native (Monoload VAE Settings node))" in t)
    sline = next(l for l in t.splitlines() if "settings: mode" in l)
    check("mode native: budget, scheme and stripe rows marked unused ({})".format(sline.strip()[:150]),
          sline.count("(not used in native mode)") == 3)
    l2 = node_apply(cls_v, sd, mode="layer 2 only")
    sline = next(l for l in text(vae=l2).splitlines() if "settings: mode" in l)
    check("mode layer 2 only: scheme and stripe rows marked unused, budget not ({})".format(sline.strip()[:150]),
          sline.count("(not used: layer 2 only)") == 2 and "budget none (default policy) [built-in]," in sline)
    lines = t.splitlines()
    check("with a vae connected the global defaults table follows at the end",
          any("global defaults" in l for l in lines) and lines.index(next(l for l in lines if "global defaults" in l)) > lines.index(next(l for l in lines if l.startswith("VAE ("))))
    time.sleep(1.1)
    t2 = text(vae=nat)
    check("refreshed on every run (the age of the record moves)", t2 != t)
    return sd


def error_record_tests(sd):
    """A decode that fails gets a record of its own error (strategy "error", kind, message), never the fields of the
    previous decode of another VAE (or of its own earlier success); the exception propagates unchanged."""
    import comfy.model_management as mm
    from monoload import vae_engine as eng
    lat = torch.randn(1, 4, 12, 10, generator=torch.Generator().manual_seed(7))
    cls_v = __import__("monoload.nodes", fromlist=["NODE_CLASS_MAPPINGS"]).NODE_CLASS_MAPPINGS["MonoloadVAESettings"]
    stale = ("adapter", "stripes", "rows", "seconds", "estimate", "plan", "workspace", "retries")

    def fail(v, label, patch, exc, kind, words):
        comfy.sd.VAE.decode(sd, lat)                    # A succeeds right before: _LAST holds A's layer-1 fields
        a_before = mvae.decode_record(sd)
        obj, name, repl = patch
        orig = getattr(obj, name)
        setattr(obj, name, repl)
        raised = None
        try:
            comfy.sd.VAE.decode(v, lat)
        except Exception as e:
            raised = e
        finally:
            setattr(obj, name, orig)
        r = mvae.decode_record(v) or {}
        line = info.describe_decode(r)
        same_a = v is sd or mvae.decode_record(sd) == a_before
        check("{}: {} raised and propagated; record: {} {} ({}), none of the previous decode's fields; Info: {}".format(
            label, type(raised).__name__, r.get("strategy"), r.get("kind"), r.get("error"), line.split("): ", 1)[-1][:90]),
            isinstance(raised, exc) and r.get("strategy") == "error" and r.get("kind") == kind and not any(k in r for k in stale)
            and all(w in line for w in words) and same_a and a_before.get("strategy") == "layer1")

    b = node_apply(cls_v, sd)
    fail(b, "VAE B fails while loading", (mm, "load_models_gpu", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("load refused"))),
         RuntimeError, "other", ["the decode failed", "RuntimeError: load refused"])
    fail(b, "VAE B fails during the stripes", (eng, "run_stripes", lambda *a, **k: (_ for _ in ()).throw(ValueError("stripe broke"))),
         ValueError, "other", ["the decode failed", "ValueError: stripe broke"])
    oom = mm.OOM_EXCEPTION("simulated")
    fail(b, "VAE B out of memory at every retry", (eng, "run_stripes", lambda *a, **k: (_ for _ in ()).throw(oom)),
         mvae.MonoloadVAEOOMError, "oom", ["out of memory", "MonoloadVAEOOMError"])
    fail(sd, "VAE A succeeds, then fails itself", (eng, "run_stripes", lambda *a, **k: (_ for _ in ()).throw(ValueError("second time"))),
         ValueError, "other", ["the decode failed", "second time"])
    comfy.sd.VAE.decode(sd, lat)
    check("... and its next successful decode replaces the error record", (mvae.decode_record(sd) or {}).get("strategy") == "layer1")
    from monoload import messages
    check("the new Info lines have English and Chinese text", all(len(messages.M[k]) == 2 and all(messages.M[k]) for k in ("info.decode_oom", "info.decode_failed")))


def probe_tests():
    """_MemProbe on a GPU (simulated here): the allocator's cache is emptied
    before the starting point is taken, so blocks cached by earlier work (the
    sampler) do not hide the decode's footprint."""
    names = ("is_available", "memory_reserved", "reset_peak_memory_stats", "max_memory_reserved")
    saved = {n: getattr(torch.cuda, n) for n in names}
    saved_empty, saved_files = mvae.mm.soft_empty_cache, mvae._gtt_files
    reserved, order = [5 << 30], []      # 5 GiB reserved, 4 of them cached free blocks left by the sampler

    def empty(*a, **k):
        order.append("empty")
        reserved[0] = 1 << 30
    torch.cuda.is_available = lambda: True
    torch.cuda.memory_reserved = lambda d=None: reserved[0]
    torch.cuda.reset_peak_memory_stats = lambda d=None: order.append("reset")
    torch.cuda.max_memory_reserved = lambda d=None: 3 << 30   # the decode's peak
    mvae.mm.soft_empty_cache = empty
    mvae._gtt_files = lambda: []
    try:
        with mvae._MemProbe(torch.device("cuda")) as p:
            pass
    finally:
        for n, f in saved.items():
            setattr(torch.cuda, n, f)
        mvae.mm.soft_empty_cache, mvae._gtt_files = saved_empty, saved_files
    check("memory probe: cache emptied before the starting point ({}), increase = peak - live memory at the start = {} GiB "
          "(not 3 - 5)".format(order, p.result["reserved_peak"] / (1 << 30)), order == ["empty", "reset"] and p.result["reserved_peak"] == 2 << 30)


def model_tests(sd):
    base = comfy.model_patcher.ModelPatcher(make_net(), CPU, CPU)
    t = text(model=base)
    check("model without LoRA: 'LoRA: none', 0 patched weights, settings from the globals", "LoRA: none" in t and "patched weights: 0" in t
          and "mode enable (default)" in t and "now: not loaded" in t)
    lo.uninstall_names()
    lo.install_names(FakeLoader)
    try:
        m1, _ = FakeLoader().load_lora(base, None, "a.safetensors", 0.8, 0.0)
        m2, _ = FakeLoader().load_lora(m1, None, "b.safetensors", -0.3, 0.0)
        mo = FakeLoaderModelOnly().load_lora_model_only(base, "c.safetensors", 0.5)[0]
        same = FakeLoader().load_lora(base, None, "zero.safetensors", 0.0, 0.0)[0]
    finally:
        lo.uninstall_names()
    t2 = text(model=m2)
    check("LoRA names and strengths from the loader nodes, chained in order: {}".format(next(l for l in t2.splitlines() if "LoRA:" in l).strip()),
          "LoRA: a.safetensors x 0.8, b.safetensors x -0.3" in t2 and "patched weights: 2" in t2 and lo.lora_names(base) == [] and lo.lora_names(m1) == [{"name": "a.safetensors", "strength": 0.8}])
    check("MODEL-only loader: 'c.safetensors x 0.5'; strength 0 returns the input unchanged, nothing recorded",
          "LoRA: c.safetensors x 0.5" in text(model=mo) and same is base and lo.lora_names(base) == [])
    nat = lo.with_settings(m2, mode="native", merge="exact")
    sline = next(l for l in text(model=nat).splitlines() if "LoRA settings:" in l)
    check("model mode native: merge marked unused ({})".format(sline.strip()), "merge exact (not used in native mode) (node)" in sline)
    tuned = lo.with_settings(m2, merge="exact", after_prompt="keep")
    t3 = text(model=tuned)
    check("settings with sources: {}".format(next(l for l in t3.splitlines() if "settings:" in l).strip()),
          "settings: mode enable (default), merge exact (node), after prompt keep (node)" in t3 and "a.safetensors x 0.8" in t3)
    import comfy.model_management as mm
    mm.load_models_gpu([tuned])
    t4 = text(model=tuned)
    mm.unload_all_models()
    check("state in memory while loaded: {}".format(next(l for l in t4.splitlines() if "now:" in l).strip()),
          "now: loaded; Monoload runtime merge on 2 weights" in t4)
    return m2


def combo_tests(sd, m):
    ok = True
    for kw in ({"vae": sd, "model": m, "images": torch.zeros(1, 8, 8, 3)}, {"images": torch.zeros(1, 8, 8, 3)}, {"vae": sd}, {"model": m}):
        try:
            t = text(**kw)
            ok = ok and t.startswith("Monoload ") and (("VAE (" in t) == ("vae" in kw)) and (("MODEL (" in t) == ("model" in kw))
        except Exception as e:
            ok = False
            print(e)
    os.environ["MONOLOAD_DISABLE"] = "1"
    try:
        t = text(vae=sd, model=m)
    finally:
        os.environ.pop("MONOLOAD_DISABLE")
    check("input combinations (vae + model + images, images only, vae, model) and MONOLOAD_DISABLE=1 run without error; "
          "MONOLOAD_DISABLE says nothing is installed", ok and "nothing installed" in t)


def main():
    from monoload import hotpatch
    if not mvae.is_installed():
        mvae.install()
    if not hotpatch.is_installed():
        hotpatch.install()
    reset_globals()
    interface_tests()
    global_tests()
    sd = vae_tests()
    error_record_tests(sd)
    probe_tests()
    m = model_tests(sd)
    combo_tests(sd, m)
    finish()


if __name__ == "__main__":
    main()
