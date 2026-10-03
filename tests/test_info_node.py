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
    check("vae never decoded: settings with sources, 'not decoded yet' ({})".format(t.splitlines()[-1].strip()[:60]),
          "settings: mode auto [built-in], budget none (default policy) [built-in]" in t and "not decoded yet" in t)
    comfy.sd.VAE.decode(sd, lat)
    t_orig = text(vae=sd)
    t_copy = text(vae=copy_)
    check("after decoding the original only: the original shows its layer-1 decode, the copy still 'not decoded yet' "
          "(records belong to the VAE object)", "layer 1 (LDM stripes" in t_orig and "not decoded yet" in t_copy)
    comfy.sd.VAE.decode(copy_, lat)
    t_copy = text(vae=copy_)
    t_orig2 = text(vae=sd)
    last = t_copy.splitlines()[-1]
    check("after decoding the copy: copy -> layer 2 with budget 1024 GiB [node]; the original still shows its own layer-1 decode\n  "
          + last.strip(), "layer 2 (op-level chunking)" in t_copy and "budget 1.00 TiB [node]" in t_copy.replace("1024.00 GiB", "1.00 TiB")
          and "layer 1 (LDM stripes" in t_orig2 and re.search(r"workspace .*, estimate .*, measured peak .*, [0-9.]+ s, OOM retries 0", last))
    check("the scheme the decode used is explained in one line: {}".format(next((l for l in t_orig2.splitlines() if "GroupNorm scheme B:" in l), "").strip()),
          "GroupNorm scheme B: keeps the H/4 and H/2 level outputs" in t_orig2)
    nat = node_apply(cls_v, sd, mode="native")
    native_decode(nat, lat)   # the original method: no record
    comfy.sd.VAE.decode(nat, lat)
    t = text(vae=nat)
    check("native decode recorded: {}".format(t.splitlines()[-1].strip()[:110]), "native ComfyUI decode (mode native (Monoload VAE Settings node))" in t)
    time.sleep(1.1)
    t2 = text(vae=nat)
    check("refreshed on every run (the age of the record moves)", t2 != t)
    return sd


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
    m = model_tests(sd)
    combo_tests(sd, m)
    finish()


if __name__ == "__main__":
    main()
