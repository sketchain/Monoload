"""The Monoload LoRA Settings node (monoload/nodes/lora_settings.py,
monoload/lora_overrides.py, per-patcher decisions in hotpatch / release).

Needs the random-weight SD1.5 checkpoint and LoRA of
tests/make_synthetic_checkpoint.py under $MODELS (checkpoints/,
loras/); ComfyUI's own CheckpointLoaderSimple, LoraLoader and
LoraLoaderModelOnly nodes; CPU, one sampling step on an 8x8 latent.

  1. interface: registry, category, inputs (MODEL required, CLIP optional),
     outputs MODEL + CLIP (None without a CLIP);
  2. priority item by item: node > global (MONOLOAD, MONOLOAD_EXACT,
     MONOLOAD_KEEP_LORA) > built-in, with sources;
  3. clone: weights shared, inputs unchanged, a new patches_uuid when the
     settings differ; the settings survive LoraLoader when the node sits
     before it (model_options are copied by every clone); chained nodes;
  4. results against native ComfyUI (Monoload's hooks uninstalled):
     merge exact == native bit for bit (text-encoder output and latent),
     mode native == native bit for bit with native's backups, enable /
     fused == the global default path, no backups; for LoraLoader,
     LoraLoaderModelOnly (MODEL only, no CLIP), two chained LoraLoaders, and
     the node before the loader;
  5. per ModelPatcher: two clones of one model with different settings,
     loaded alternately, each gives its own result and the base weights are
     restored bit for bit afterwards;
  6. MONOLOAD=0: no node -> native; node enable -> Monoload's runtime merge;
  7. after the prompt: release / keep / default (global, native keeps) for
     loaded models and cached outputs;
  8. MONOLOAD_DISABLE=1: the node passes its inputs through.

    python tests/make_synthetic_checkpoint.py $MODELS   # once
    python tests/test_lora_node.py
"""

import hashlib
import os

import torch

from common import check, expect_raises, finish, free_all, set_runtime
import comfy.model_management as mm
import comfy.sample
import folder_paths
import nodes
from monoload import hotpatch, lora_overrides as lo, release, settings

CKPT = "synthetic_sd15.safetensors"
LORA = "synthetic_sd15_lora.safetensors"
NODE = "MonoloadLoRASettings"


def node_cls():
    from monoload.nodes import NODE_CLASS_MAPPINGS
    return NODE_CLASS_MAPPINGS[NODE]


def node(model, clip=None, **kw):
    cls = node_cls()
    args = {k: v[1]["default"] for k, v in cls.INPUT_TYPES()["required"].items() if k != "model"}
    args.update(kw)
    return getattr(cls(), cls.FUNCTION)(model=model, clip=clip, **args)


def lora(model, clip, s_model=0.8, s_clip=0.6):
    return nodes.LoraLoader().load_lora(model, clip, LORA, s_model, s_clip)


def lora_model_only(model, s=0.8):
    return nodes.LoraLoaderModelOnly().load_lora_model_only(model, LORA, s)[0]


def run(model, clip):
    """(text-encoder output, latent after one step, backups) of this MODEL / CLIP."""
    pos = nodes.CLIPTextEncode().encode(clip, "a photo of a duck")[0]
    lat = torch.zeros(1, 4, 8, 8)
    noise = comfy.sample.prepare_noise(lat, 1, None)
    out = comfy.sample.sample(model, noise, 1, 1.0, "euler", "normal", pos, pos, lat, denoise=1.0, disable_pbar=True, seed=1)
    return pos[0][0].clone(), out.clone(), len(model.backup) + len(clip.patcher.backup)


def weights_digest(model):
    h = hashlib.blake2b(digest_size=16)
    for k, v in sorted(model.model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def runtime_patches(model):
    return sum(1 for m in model.model.modules() for a in ("weight_function", "bias_function")
               for f in (m.__dict__.get(a) or ()) if getattr(f, "is_monoload_patch", False))


# ---------------------------------------------------------------------------

def interface_tests():
    from monoload.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    cls = NODE_CLASS_MAPPINGS.get(NODE)
    it = cls.INPUT_TYPES()
    check("node in the registry: {} ({}), category {}, required {}, optional {}, returns {}".format(
          NODE, NODE_DISPLAY_NAME_MAPPINGS.get(NODE), cls.CATEGORY, list(it["required"]), list(it["optional"]), cls.RETURN_TYPES),
          NODE_DISPLAY_NAME_MAPPINGS.get(NODE) == "Monoload LoRA Settings" and cls.CATEGORY == "Monoload"
          and list(it["required"]) == ["model", "mode", "merge", "after_prompt"] and list(it["optional"]) == ["clip"]
          and cls.RETURN_TYPES == ("MODEL", "CLIP") and it["required"]["mode"][0] == ["default", "enable", "native"]
          and it["required"]["merge"][0] == ["default", "fused", "exact"] and it["required"]["after_prompt"][0] == ["default", "release", "keep"])


def priority_tests(model):
    plain = node(model)[0]
    rows = []
    eff, src = lo.resolve(plain)
    ok = eff == {"mode": "enable", "merge": "fused", "after_prompt": "release"} and set(src.values()) == {"default"}
    rows.append("defaults: {}".format(lo.note(eff, src)))
    settings.set_master(False)
    settings.set_exact(True)
    settings.set_keep(True)
    try:
        eff, src = lo.resolve(plain)
        ok_env = eff == {"mode": "native", "merge": "exact", "after_prompt": "keep", "after_from_mode": True} and set(src.values()) == {"env"}
        rows.append("globals: {}".format(lo.note(eff, src)))
        got = []
        good = True
        for item, value in (("mode", "enable"), ("merge", "fused"), ("after_prompt", "release")):
            eff, src = lo.resolve(node(model, **{item: value})[0])
            others = [k for k in src if k != item]
            good = good and eff[item] == value and src[item] == "node" and all(src[k] == "env" for k in others)
            got.append("{} -> {}".format(item, value))
        eff, src = lo.resolve(node(model, mode="enable")[0])
        good = good and eff["after_prompt"] == "keep" and src["after_prompt"] == "env"   # enabled: the global keep
    finally:
        settings.set_master(True)
        settings.set_exact(False)
        settings.set_keep(False)
    check("priority: {}".format(" | ".join(rows)), ok and ok_env)
    check("item by item under MONOLOAD=0 + MONOLOAD_EXACT + MONOLOAD_KEEP_LORA: the node's item, the globals for the others ({})".format(
          ", ".join(got)), good)
    eff, src = lo.resolve(node(model, mode="native")[0])
    check("mode native: after the prompt kept as native ComfyUI does ({})".format(lo.note(eff, src)),
          eff["after_prompt"] == "keep" and src["after_prompt"] == "node")
    expect_raises("unknown merge -> ValueError", ValueError, lambda: node(model, merge="fast"), "merge")


def clone_tests(model, clip):
    m, c = node(model, clip, mode="enable", merge="exact", after_prompt="keep")
    check("clone: MODEL and CLIP share the weights, the inputs carry no settings, new patches_uuid",
          m is not model and m.model is model.model and c is not clip and c.patcher.model is clip.patcher.model
          and lo.overrides(model) == {} and lo.overrides(clip.patcher) == {} and m.patches_uuid != model.patches_uuid
          and lo.overrides(m) == lo.overrides(c.patcher) == {"mode": "enable", "merge": "exact", "after_prompt": "keep"})
    same = node(model)[0]
    check("a node left at default: a clone with the same patches_uuid (nothing to reload)", same.patches_uuid == model.patches_uuid)
    m2, c2 = lora(m, c)
    check("node BEFORE LoraLoader: the loader's clones keep the settings (model_options are copied by clone) ({})".format(lo.overrides(m2)),
          lo.overrides(m2) == lo.overrides(c2.patcher) == lo.overrides(m) and len(m2.patches) > 0)
    chained = node(m2, merge="fused")[0]
    check("chained nodes: the downstream item overrides, the rest is kept ({})".format(lo.overrides(chained)),
          lo.overrides(chained) == {"mode": "enable", "merge": "fused", "after_prompt": "keep"})
    mo, co = node(lora_model_only(model), None, merge="exact")
    check("MODEL only (LoraLoaderModelOnly, no CLIP connected): CLIP output None, no error", co is None and lo.overrides(mo) == {"merge": "exact"})


def results_tests(model, clip):
    chains = {
        "LoraLoader": lambda m, c: lora(m, c),
        "LoraLoaderModelOnly": lambda m, c: (lora_model_only(m), c),
        "2 x LoraLoader": lambda m, c: lora(*lora(m, c, 0.5, 0.4), -0.3, 0.2),
    }
    set_runtime(False)
    try:
        ref = {name: run(*f(model, clip)) for name, f in chains.items()}
    finally:
        set_runtime(True)
    for name, f in chains.items():
        lm, lc = f(model, clip)
        r_cond, r_lat, r_bk = ref[name]
        rows = {}
        for label, kw in (("exact", {"merge": "exact"}), ("native", {"mode": "native"}), ("fused", {"merge": "fused"}), ("no node", None)):
            free_all()
            m, c = (lm, lc) if kw is None else node(lm, lc, **kw)
            cond, lat, bk = run(m, c)
            rows[label] = (cond, lat, bk)
        eq = lambda a, b: torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        d = float((rows["fused"][1] - r_lat).abs().max()) / float(r_lat.abs().max())
        check("{}: exact == native bit for bit (no backups: {}); native == native bit for bit with native's backups ({} = {}); "
              "fused == no node (the global default), no backups, max|Δ latent| / max|latent| vs native {:.2g}".format(
                  name, rows["exact"][2], rows["native"][2], r_bk, d),
              eq(rows["exact"], ref[name]) and rows["exact"][2] == 0 and eq(rows["native"], ref[name]) and rows["native"][2] == r_bk > 0
              and eq(rows["fused"], rows["no node"]) and rows["fused"][2] == 0 and 0 < d < 1e-3)
    free_all()
    m, c = lora(*node(model, clip, merge="exact"))
    cond, lat, bk = run(m, c)
    check("node before LoraLoader, merge exact: == native bit for bit", torch.equal(cond, ref["LoraLoader"][0]) and torch.equal(lat, ref["LoraLoader"][1]) and bk == 0)
    return ref["LoraLoader"]


def per_patcher_tests(model, clip, ref):
    free_all()
    base = weights_digest(model)
    lm, lc = lora(model, clip)
    a = node(lm, lc, merge="fused")
    b = node(lm, lc, mode="native")
    e = node(lm, lc, merge="exact")
    got = []
    for label, (m, c) in (("fused", a), ("native", b), ("exact", e), ("fused", a), ("native", b)):
        cond, lat, bk = run(m, c)   # no free between: ComfyUI switches the shared model itself
        got.append((label, lat, bk, runtime_patches(m)))
    fused_lat = got[0][1]
    ok = all(torch.equal(lat, fused_lat if label == "fused" else ref[1]) for label, lat, _, _ in got)
    ok = ok and all((bk == 0 and rp > 0) if label != "native" else (bk > 0 and rp == 0) for label, _, bk, rp in got)
    free_all()
    check("per ModelPatcher: clones of one model loaded alternately (fused, native, exact, fused, native) each give their own "
          "result (native / exact == native, fused == fused) with their own weight handling; base weights restored bit for bit "
          "afterwards", ok and weights_digest(model) == base)


def master_off_tests(model, clip, ref):
    lm, lc = lora(model, clip)
    fused = None
    free_all()
    fused = run(*node(lm, lc, mode="enable"))
    settings.set_master(False)
    try:
        free_all()
        plain = run(lm, lc)
        free_all()
        on = run(*node(lm, lc, mode="enable"))
        free_all()
    finally:
        settings.set_master(True)
    check("MONOLOAD=0: no node -> native ComfyUI bit for bit with backups ({}); node enable -> Monoload's runtime merge "
          "(== the same model with the switch on, no backups)".format(plain[2]),
          torch.equal(plain[1], ref[1]) and torch.equal(plain[0], ref[0]) and plain[2] > 0
          and torch.equal(on[1], fused[1]) and on[2] == 0)


class _Cache:
    def __init__(self, entries):
        self.cache = dict(entries)


class _Caches:
    def __init__(self, outputs):
        self.outputs = _Cache(outputs)
        self.objects = None


class _Executor:
    """What release_after_prompt reads of a PromptExecutor: caches.outputs / caches.objects."""
    def __init__(self, outputs):
        self.caches = _Caches(outputs)


def release_tests(model, clip):
    lm, lc = lora(model, clip)
    rows = []
    ok = True
    for master, kw, want in ((True, {}, True), (True, {"after_prompt": "keep"}, False), (True, {"mode": "native"}, False),
                             (True, {"mode": "native", "after_prompt": "release"}, True), (False, {}, False),
                             (False, {"mode": "enable"}, True), (False, {"mode": "enable", "after_prompt": "keep"}, False)):
        settings.set_master(master)
        try:
            free_all()
            m, c = node(lm, lc, **kw) if kw else (lm, lc)
            run(m, c)
            ex = _Executor({"k": [[m]], "plain": [[model]]})
            r = release.release_after_prompt(ex)
            loaded = [x.model for x in mm.current_loaded_models if x.model is not None and x.model.model is model.model]
            released = r["models"] >= 1 and "k" not in ex.caches.outputs.cache and runtime_patches(m) == 0 and len(m.backup) == 0
            kept = r["models"] == 0 and "k" in ex.caches.outputs.cache and m in loaded
            ok = ok and (released if want else kept) and "plain" in ex.caches.outputs.cache
            rows.append("{}{} -> {}".format("" if master else "MONOLOAD=0 ", kw or "no node", "released" if released else "kept" if kept else
                        "? {} cache {} rp {} bk {} loaded {}".format(r, list(ex.caches.outputs.cache), runtime_patches(m), len(m.backup), m in loaded)))
        finally:
            settings.set_master(True)
    free_all()
    check("after the prompt, per model: " + "; ".join(rows), ok)


def disable_tests(model, clip):
    from monoload.nodes import lora_settings as nl
    os.environ["MONOLOAD_DISABLE"] = "1"
    nl._NOTED.clear()
    try:
        m, c = node(model, clip, mode="enable", merge="exact")
        m2, c2 = node(model, None)
    finally:
        os.environ.pop("MONOLOAD_DISABLE")
        nl._NOTED.clear()
    check("MONOLOAD_DISABLE=1: the node returns its inputs themselves", m is model and c is clip and m2 is model and c2 is None)


def main():
    have = CKPT in folder_paths.get_filename_list("checkpoints") and LORA in folder_paths.get_filename_list("loras")
    if not check("synthetic SD1.5 checkpoint and LoRA present (tests/make_synthetic_checkpoint.py)", have):
        finish()
    if not hotpatch.is_installed():
        hotpatch.install()
    if not release.is_installed():
        release.install()
    interface_tests()
    model, clip = nodes.CheckpointLoaderSimple().load_checkpoint(CKPT)[:2]
    priority_tests(model)
    clone_tests(model, clip)
    ref = results_tests(model, clip)
    per_patcher_tests(model, clip, ref)
    master_off_tests(model, clip, ref)
    release_tests(model, clip)
    disable_tests(model, clip)
    finish()


if __name__ == "__main__":
    main()
