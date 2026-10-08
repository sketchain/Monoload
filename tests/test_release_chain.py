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

Second situation (review 2026-10 item 02, README §13's CheckpointSave set-up):
Checkpoint -> Monoload LoRA Settings (MODEL only, mode native) -> LoraLoader
(MODEL + CLIP). The MODEL is native: its LoRA is baked into the weights, the
originals in the backup that every clone shares. The CLIP is Monoload's and is
released, which drops the LoraLoader's cached output; the native MODEL clone
dies, ComfyUI's finalizer points the LoadedModel at the Settings clone (no
patches, but the shared backup and the baked weights). Its uuid must not be
synced to the Settings clone's, so the next prompt that uses the Settings
output directly restores the backup instead of running with the dead clone's
LoRA.

No model files: a two-layer comfy.ops model stands in for the CLIP and the
UNet; ComfyUI's own LoadedModel / load_models_gpu / cleanup_models_gc.

    python tests/test_release_chain.py
"""

import gc
import logging
import os

import torch

from common import check, finish
import comfy.model_management as mm
import comfy.model_patcher
from monoload import lora_overrides, release, settings
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


def native_model_scenario():
    """(loaded UNet entry on the Settings clone, its uuid left unsynced while the backup is there, after loading the
    Settings clone: weights == base, no backup, output == base output; release result)."""
    mm.unload_all_models()
    gc.collect()
    x = torch.randn(4, 96, generator=torch.Generator().manual_seed(3))
    net = make_net()
    w0 = {k: v.clone() for k, v in net.state_dict().items()}
    y0 = net(x)
    unet0 = comfy.model_patcher.ModelPatcher(net, CPU, CPU)
    clip0 = comfy.model_patcher.ModelPatcher(make_net(), CPU, CPU)
    unet_s = lora_overrides.with_settings(unet0, mode="native")   # Monoload LoRA Settings, MODEL only
    unet_l = unet_s.clone()                                        # LoraLoader: the MODEL clone is native too
    unet_l.add_patches(make_lora(5), 0.8)
    clip_l = clip0.clone()                                         # ... the CLIP clone is Monoload's (released)
    clip_l.add_patches(make_lora(6), 0.8)
    mm.load_models_gpu([clip_l])
    mm.load_models_gpu([unet_l])
    baked = len(unet_l.backup) > 0 and not torch.equal(net.state_dict()["a.weight"], w0["a.weight"])
    lm = next(m for m in mm.current_loaded_models if m.model is unet_l)
    ex = _Executor({"ckpt": [[unet0, clip0]], "settings": [[unet_s]], "lora": [[unet_l, clip_l]]})
    del unet_l, clip_l
    r = release.release_after_prompt(ex)
    on_settings = lm.model is unet_s and len(unet_s.backup) > 0
    unsynced = net.current_weight_patches_uuid != unet_s.patches_uuid
    mm.load_models_gpu([unet_s])   # the next prompt: the Settings output straight into the sampler
    restored = all(torch.equal(v, w0[k]) for k, v in net.state_dict().items()) and len(unet_s.backup) == 0 and torch.equal(net(x), y0)
    mm.unload_all_models()
    return baked, on_settings, unsynced, restored, r


def main():
    from monoload import hotpatch
    if not hotpatch.is_installed():
        hotpatch.install()
    settings.set_master(True)
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
    if os.environ.get("WITHOUT_FIX"):   # the sync without the backup check
        orig_sync = release._sync_clean_loaded_models

        def sync_ignoring_backup():
            n = 0
            for lm in mm.current_loaded_models:
                p = lm.model
                if p is not None and not release.patcher_has_weight_patches(p) and p.model.current_weight_patches_uuid != p.patches_uuid:
                    p.model.current_weight_patches_uuid = p.patches_uuid
                    n += 1
            return n
        release._sync_clean_loaded_models = sync_ignoring_backup
    baked, on_settings, unsynced, restored, r = native_model_scenario()
    check("native MODEL (Settings mode native before LoraLoader) + Monoload CLIP released: the loaded MODEL ends on the Settings "
          "clone with the shared backup (LoRA baked: {}, on it: {}), its uuid is not synced ({}; synced {}), the next load of the "
          "Settings clone restores the base: weights, no backup, output ({})".format(baked, on_settings, unsynced, r.get("synced"), restored),
          baked and on_settings and unsynced and restored)
    finish()


if __name__ == "__main__":
    main()
