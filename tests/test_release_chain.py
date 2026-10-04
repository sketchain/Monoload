"""Per-prompt release with a chain of patch-free clones (monoload/release.py
_repoint_orphans): the "memory leak with model SDXLClipModel" warning seen
on CT 700.

Situation: Checkpoint -> LoraLoader (a LoRA without text-encoder keys: the
CLIP clone carries no patches) -> Monoload LoRA Settings (another
patch-free CLIP clone) -> CLIPTextEncode; the loaded CLIP's patcher is the
second clone. The release drops both nodes' cached outputs; when something
else (on CT 700: an error's traceback cycle in the executor) keeps both
clones alive until the release's gc.collect, both die in the same
collection, ComfyUI's finalizer (one level: patcher -> parent) finds the
parent dead too, and the LoadedModel is left without a patcher while its
model lives on in the base -> ComfyUI's cleanup_models_gc warns on every
load. Monoload must point it at the nearest living ancestor.

No model files: a two-layer comfy.ops model stands in for the CLIP and the
UNet; ComfyUI's own LoadedModel / load_models_gpu / cleanup_models_gc.

    python tests/test_release_chain.py
"""

import gc
import logging

import torch

from common import check, finish
import comfy.model_management as mm
import comfy.model_patcher
from monoload import release, settings
from test_master_switch import make_lora, make_net
from test_vae_node import LogCapture

CPU = torch.device("cpu")


class _Cache:
    def __init__(self, entries):
        self.cache = dict(entries)


class _Executor:
    def __init__(self, entries):
        self.caches = type("C", (), {})()
        self.caches.outputs = _Cache(entries)
        self.caches.objects = None


def scenario(cycle):
    """(LoadedModel of the clip-like patcher after the release, base patcher, release result, warnings)."""
    mm.unload_all_models()
    gc.collect()
    unet0 = comfy.model_patcher.ModelPatcher(make_net(), CPU, CPU)
    clip0 = comfy.model_patcher.ModelPatcher(make_net(), CPU, CPU)
    unet1 = unet0.clone()
    unet1.add_patches(make_lora(5), 0.8)          # LoraLoader: the UNet clone carries the LoRA
    clip1 = clip0.clone()                         # ... the CLIP clone carries nothing (no text-encoder keys)
    unet2, clip2 = unet1.clone(), clip1.clone()   # Monoload LoRA Settings: one more clone level each
    mm.load_models_gpu([clip2])
    mm.load_models_gpu([unet2])
    lm = next(x for x in mm.current_loaded_models if x.model is clip2)
    ex = _Executor({"lora": [[unet1, clip1]], "settings": [[unet2, clip2]], "ckpt": [[unet0, clip0]]})
    if cycle:   # keep both clip clones alive until a collection, as the executor's traceback does
        holder = [clip1, clip2]
        holder.append(holder)
        del holder
    del unet1, unet2, clip1, clip2
    r = release.release_after_prompt(ex)
    with LogCapture() as cap:
        logging.getLogger().setLevel(logging.INFO)
        mm.cleanup_models_gc()
    warned = [line for line in cap.lines if "memory leak" in line]
    return lm, clip0, r, warned


def main():
    from monoload import hotpatch
    if not hotpatch.is_installed():
        hotpatch.install()
    settings.set_master(True)
    import os
    if os.environ.get("WITHOUT_FIX"):   # show the failure the fix removes
        release._repoint_orphans = lambda chains: 0
    rows = []
    ok = True
    for cycle in (False, True):
        lm, clip0, r, warned = scenario(cycle)
        good = lm.model is clip0 and not lm.is_dead() and not warned and lm.model.model.current_weight_patches_uuid == clip0.patches_uuid
        ok = ok and good
        rows.append("{}: loaded CLIP -> {}, re-pointed {}, warnings {}".format(
            "clones die in the collection (traceback cycle)" if cycle else "clones die at once", "base" if lm.model is clip0 else lm.model,
            r.get("repointed"), len(warned)))
        if cycle and not r.get("repointed"):
            ok = False
    check("chain of two patch-free CLIP clones released: the loaded entry ends on the base, no 'memory leak' warning\n  "
          + "\n  ".join(rows), ok)
    finish()


if __name__ == "__main__":
    main()
