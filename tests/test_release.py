"""Per-prompt LoRA release, through ComfyUI's real PromptExecutor.

Workflows (API format, SD1.5 checkpoint):
  lora  : CheckpointLoaderSimple -> LoraLoader -> CLIPTextEncode x2 -> KSampler -> SaveLatent
  plain : the same without the LoRA
  hook  : CreateHookLora -> SetClipHooks -> CLIPTextEncode (hooked conds) -> KSampler
  unet_only: LoraLoader with a LoRA that has no text-encoder keys (the CLIP
          clone LoraLoader returns carries no patches but a new patches_uuid)
  bypass: LoraLoaderBypass (bypass-LoRA injections instead of weight patches)

Sequence: lora -> plain -> unet_only -> plain -> hook -> plain -> bypass -> plain -> lora (seed 8).
Checked after each LoRA prompt:
  * every LoRA tensor (loaded from models/loras) and every patcher carrying
    LoRA patches is garbage (weakrefs dead) -- i.e. freed on CPU and, since
    device copies only live on those objects, on the GPU as well
  * no runtime patches left on any module, no device-side LoRA cache
and for the following plain prompt:
  * the checkpoint loader node is not executed again, ModelPatcher.load() is
    not called at all (base UNet / CLIP stay loaded as they are)
  * the output is bit-identical to a process that never saw a LoRA
    (--save-reference writes that output, --reference reads it)

MONOLOAD_KEEP_LORA=1: expects the LoRA to be kept instead.

    python tests/test_release.py --save-reference /out/ref.pt
    python tests/test_release.py --reference /out/ref.pt [--cache ram_pressure|classic|lru]
"""

import argparse
import asyncio
import gc
import os
import uuid
import weakref

import torch

from common import check, finish
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import execution
import folder_paths
import nodes

CKPT = "v1-5-pruned-emaonly-fp16.safetensors"
LORA = "rubber_duck.safetensors"
UNET_ONLY_LORA = "synthetic_unet_only_sd15.safetensors"  # no text-encoder keys: the CLIP clone carries no patches
POS = "a photo of a yellow rubber duck on a wooden table"
NEG = "blurry"
KEEP = os.environ.get("MONOLOAD_KEEP_LORA", "") == "1"


class Server:
    client_id = None
    last_node_id = None
    last_prompt_id = None

    def send_sync(self, *a, **k):
        pass

    def queue_updated(self):
        pass


def workflow(kind, ckpt=CKPT, lora=LORA, seed=7):
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 128, "height": 128, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0],
                                                   "seed": seed, "steps": 2, "cfg": 5.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "7": {"class_type": "SaveLatent", "inputs": {"samples": ["6", 0], "filename_prefix": "monoload_test"}},
    }
    clip = ["1", 1]
    if kind == "unet_only":
        kind, lora = "lora", UNET_ONLY_LORA
    if kind == "lora":
        wf["2"] = {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": lora,
                                                          "strength_model": 0.8, "strength_clip": 0.8}}
        wf["6"]["inputs"]["model"] = ["2", 0]
        clip = ["2", 1]
    elif kind == "bypass":
        wf["2"] = {"class_type": "LoraLoaderBypass", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": lora,
                                                                "strength_model": 0.8, "strength_clip": 0.8}}
        wf["6"]["inputs"]["model"] = ["2", 0]
        clip = ["2", 1]
    elif kind == "hook":
        wf["8"] = {"class_type": "CreateHookLora", "inputs": {"lora_name": lora, "strength_model": 0.9, "strength_clip": 0.8}}
        wf["9"] = {"class_type": "SetClipHooks", "inputs": {"clip": ["1", 1], "apply_to_conds": True, "schedule_clip": False, "hooks": ["8", 0]}}
        clip = ["9", 0]
    wf["3"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": POS}}
    wf["4"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": NEG}}
    return wf


# ---- instrumentation -------------------------------------------------------

LORA_REFS = []      # weakrefs to every tensor loaded from models/loras
PATCHER_REFS = []   # weakrefs to patchers that received weight / hook patches
COUNT = {"ckpt": 0, "load": 0}


def instrument():
    lora_dirs = [os.path.realpath(p) for p in folder_paths.get_folder_paths("loras")]
    orig_ltf = comfy.utils.load_torch_file

    def load_torch_file(ckpt, *a, **k):
        out = orig_ltf(ckpt, *a, **k)
        if any(os.path.realpath(ckpt).startswith(d) for d in lora_dirs):
            sd = out[0] if isinstance(out, tuple) else out
            LORA_REFS.extend(weakref.ref(t) for t in sd.values() if isinstance(t, torch.Tensor))
        return out
    comfy.utils.load_torch_file = load_torch_file

    MP = comfy.model_patcher.ModelPatcher
    for name in ("add_patches", "add_hook_patches", "set_injections"):
        orig = MP.__dict__[name]

        def wrapper(self, *a, _orig=orig, **k):
            PATCHER_REFS.append(weakref.ref(self))
            return _orig(self, *a, **k)
        setattr(MP, name, wrapper)

    orig_load = MP.__dict__["load"]

    def load(self, *a, **k):
        COUNT["load"] += 1
        return orig_load(self, *a, **k)
    MP.load = load

    cls = nodes.NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]
    orig_ck = cls.load_checkpoint

    def load_checkpoint(self, *a, **k):
        COUNT["ckpt"] += 1
        return orig_ck(self, *a, **k)
    cls.load_checkpoint = load_checkpoint


def rss_gib():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 30


def alive(refs):
    gc.collect()
    return sum(1 for r in refs if r() is not None)


def alive_patched(refs):
    """Patchers still alive AND still carrying weight / hook patches (a base
    patcher that only held hook patches during sampling is fine)."""
    gc.collect()
    n = 0
    for r in refs:
        p = r()
        if p is not None and (len(p.patches) > 0 or len(p.hook_patches) > 0 or "bypass_lora" in p.injections):
            n += 1
    return n


def runtime_patch_count():
    n = 0
    for lm in comfy.model_management.current_loaded_models:
        p = lm.model
        if p is None:
            continue
        for m in p.model.modules():
            for attr in ("weight_function", "bias_function"):
                n += sum(1 for f in m.__dict__.get(attr, None) or () if getattr(f, "is_monoload_patch", False))
    return n


def device_cache_entries():
    n = 0
    for obj in gc.get_objects():
        if type(obj).__name__ == "_State" and hasattr(obj, "device_cache"):
            n += len(obj.device_cache)
    return n


def loaded_lora_caches(e):
    n = 0
    c = e.caches.objects
    for obj in getattr(c, "cache", {}).values():
        if getattr(obj, "loaded_lora", None) is not None:
            n += 1
    return n


def run(e, wf):
    pid = str(uuid.uuid4())
    e.execute(wf, pid, {}, ["7"])
    assert e.success, "prompt failed: {}".format(e.status_messages[-1:])
    entry = asyncio.run(e.caches.outputs.get("6"))
    return entry.outputs[0][0]["samples"].clone()


def base_model_id(e):
    entry = asyncio.run(e.caches.outputs.get("1"))
    return None if entry is None else id(entry.outputs[0][0].model)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save-reference")
    p.add_argument("--reference")
    p.add_argument("--cache", default="ram_pressure", choices=["ram_pressure", "classic", "lru"])
    a = p.parse_args()

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    ok = asyncio.run(nodes.load_custom_node("/opt/ComfyUI/custom_nodes/monoload"))
    check("Monoload plugin loaded", ok)
    instrument()
    ctype = {"ram_pressure": execution.CacheType.RAM_PRESSURE, "classic": execution.CacheType.CLASSIC, "lru": execution.CacheType.LRU}[a.cache]
    e = execution.PromptExecutor(Server(), cache_type=ctype, cache_args={"lru": 20, "ram": 2.0, "ram_inactive": 16.0})

    if a.save_reference:
        out = run(e, workflow("plain"))
        check("reference: plain output from a process that never saw a LoRA", COUNT["ckpt"] == 1 and not LORA_REFS)
        lora8 = run(e, workflow("lora", seed=8))
        torch.save({"plain": out, "lora8": lora8}, a.save_reference)
        finish()

    refs = torch.load(a.reference)
    ref = refs["plain"]
    rss0 = rss_gib()
    summary = {"cache": a.cache, "keep": KEEP}
    for lora_kind in ("lora", "unet_only", "hook", "bypass"):
        before = dict(COUNT)
        out_l = run(e, workflow(lora_kind))
        rss_l = rss_gib()
        n_tensors = len(LORA_REFS)
        n_alive, n_patchers = alive(LORA_REFS), alive_patched(PATCHER_REFS)
        tag = "{} [{}]".format(lora_kind, a.cache)
        check("{}: LoRA changes the output".format(tag), not torch.equal(out_l, ref))
        if KEEP:
            check("{}: MONOLOAD_KEEP_LORA=1 keeps the LoRA ({} of {} LoRA tensors alive)".format(tag, n_alive, n_tensors), n_alive > 0)
        else:
            check("{}: all {} LoRA tensors freed after the prompt".format(tag, n_tensors), n_tensors > 0 and n_alive == 0, "{} still alive".format(n_alive))
            check("{}: no patcher carrying LoRA patches left".format(tag), n_patchers == 0, "{} alive".format(n_patchers))
            check("{}: no runtime patches left on loaded models".format(tag), runtime_patch_count() == 0, str(runtime_patch_count()))
            check("{}: no device-side LoRA cache left".format(tag), device_cache_entries() == 0)
            check("{}: no node-level LoRA cache (loaded_lora) left".format(tag), loaded_lora_caches(e) == 0)
        bid = base_model_id(e)

        mid = dict(COUNT)
        out_p = run(e, workflow("plain"))
        rss_p = rss_gib()
        check("{} -> plain: checkpoint loader not executed again (total {})".format(tag, COUNT["ckpt"]), COUNT["ckpt"] == 1)
        check("{} -> plain: base model object reused".format(tag), bid is not None and base_model_id(e) == bid)
        if not KEEP:
            check("{} -> plain: ModelPatcher.load() not called (base UNet/CLIP stay loaded)".format(tag), COUNT["load"] == mid["load"],
                  "{} load() calls".format(COUNT["load"] - mid["load"]))
        d = float((out_p.float() - ref.float()).abs().max())
        check("{} -> plain: output bit-identical to never-LoRA reference".format(tag), torch.equal(out_p, ref), "max_abs {}".format(d))
        summary[lora_kind] = {"lora_tensors": n_tensors, "alive_after_prompt": n_alive, "rss_after_lora_gib": round(rss_l, 3),
                              "rss_after_plain_gib": round(rss_p, 3), "load_calls_lora_prompt": mid["load"] - before["load"]}
        print("[INFO] {}: RSS start {:.3f} GiB, after LoRA prompt {:.3f} GiB, after plain prompt {:.3f} GiB".format(tag, rss0, rss_l, rss_p))
    # LoRA used again after being released: must be re-applied correctly
    out8 = run(e, workflow("lora", seed=8))
    d = float((out8.float() - refs["lora8"].float()).abs().max())
    check("LoRA used again (seed 8) == fresh process", torch.equal(out8, refs["lora8"]), "max_abs {}".format(d))
    finish(summary)


if __name__ == "__main__":
    main()
