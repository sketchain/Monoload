"""Runtime LoRA merge vs native ComfyUI, through the native loader nodes.

Pipelines:
  checkpoint : CheckpointLoaderSimple  (MODEL + CLIP from one file)
  unet+clip  : UNETLoader + CLIPLoader (separately loaded diffusion model / text encoder)

For every pipeline the same sequence of LoRA combinations (LoraLoader, model
and CLIP strengths) is run natively (Monoload uninstalled) and with Monoload,
switching combinations without unloading in between. Checked:
  * text-encoder output and sampled latent bit-identical to native
  * no backups (ModelPatcher.backup / hook_backup / cached_hook_patches) on MODEL and CLIP
  * weights never change: equal to the state right after loading, during and after LoRA
  * hook LoRA (keyframed, SetClipHooks + conditioning hooks) bit-identical
  * refusals: force_patch_weights, non-comfy.ops param, shape change, DynamicVRAM

    python tests/test_lora_hot.py --checkpoint SD.safetensors --unet SD.safetensors --clip clip_l.safetensors
"""

import argparse

import torch

from common import (MonoloadUnsupportedError, apply_loras, check, cond_equal, diff_stats, digests_diff, encode,
                    expect_raises, finish, free_all, load_checkpoint, load_clip, load_unet, sample, set_runtime,
                    weight_digests)
import comfy.hooks
import comfy.memory_management
import comfy.model_management
import comfy.utils
import folder_paths
from comfy_extras.nodes_hooks import SetClipHooks

DUCK = "rubber_duck.safetensors"
LOCON = "lycoris_annalise.safetensors"
LOKR = "synthetic_lokr_sd15.safetensors"
LOHA = "synthetic_loha_sd15.safetensors"

COMBOS = {
    "none": [],
    "duck": [(DUCK, 0.8, 0.8)],
    "locon": [(LOCON, 0.7, 0.9)],
    "duck+locon": [(DUCK, 0.6, 0.6), (LOCON, 0.5, 0.7)],
    "lokr": [(LOKR, 0.8, 0.8)],
    "loha": [(LOHA, 0.8, 0.8)],
}
# switching order used in Monoload mode (native references are per combo)
SEQUENCE = ["duck", "locon", "duck+locon", "duck", "none", "lokr", "loha", "duck+locon"]
POS = "a photo of a yellow rubber duck on a wooden table, studio light"
NEG = "blurry, lowres"


def load_pipeline(kind, a):
    if kind == "checkpoint":
        return load_checkpoint(a.checkpoint)
    return load_unet(a.unet), load_clip(a.clip)


def run_combo(model, clip, combo, latent, steps):
    m, c = apply_loras(model, clip, COMBOS[combo])
    pos = encode(c, POS)
    neg = encode(c, NEG)
    out = sample(m, pos, neg, latent, steps=steps)
    return m, c, pos, neg, out


def hooked_inputs(clip, lora_sd):
    hooks = comfy.hooks.create_hook_lora(lora_sd, strength_model=0.9, strength_clip=0.8)
    kf = comfy.hooks.HookKeyframeGroup()
    kf.add(comfy.hooks.HookKeyframe(strength=1.0, start_percent=0.0))
    kf.add(comfy.hooks.HookKeyframe(strength=0.4, start_percent=0.5))
    hooks.set_keyframes_on_hooks(kf)
    hclip = SetClipHooks().apply_hooks(clip, schedule_clip=False, apply_to_conds=True, hooks=hooks)[0]
    return hclip, encode(hclip, POS), encode(hclip, NEG)


def backups(*patchers):
    return sum(len(p.backup) + len(p.hook_backup) + len(p.cached_hook_patches) for p in patchers)


def runtime_functions_left(*modules):
    n = 0
    for mod in modules:
        for m in mod.modules():
            if m.__dict__.get("weight_function") or m.__dict__.get("bias_function"):
                n += 1
    return n


def test_pipeline(kind, a, summary):
    latent = torch.zeros(1, 4, 32, 32)
    hook_sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", DUCK), safe_load=True)

    # ---- native references ------------------------------------------------
    set_runtime(False)
    model, clip = load_pipeline(kind, a)
    ref = {}
    native_backups = {}
    for combo in COMBOS:
        m, c, pos, neg, out = run_combo(model, clip, combo, latent, a.steps)
        ref[combo] = (pos, neg, out)
        native_backups[combo] = (len(m.backup), len(c.patcher.backup))
    hclip, hpos, hneg = hooked_inputs(clip, hook_sd)
    ref["hook"] = (hpos, hneg, sample(model, hpos, hneg, latent, steps=max(a.steps, 3)))
    del model, clip, m, c, hclip
    free_all()

    # ---- Monoload ----------------------------------------------------------
    set_runtime(True)
    model, clip = load_pipeline(kind, a)
    d_model = weight_digests(model.model)
    d_clip = weight_digests(clip.patcher.model)
    for i, combo in enumerate(SEQUENCE):
        m, c, pos, neg, out = run_combo(model, clip, combo, latent, a.steps)
        rpos, rneg, rout = ref[combo]
        d = diff_stats(rout, out)
        te_ok = cond_equal(rpos, pos) and cond_equal(rneg, neg)
        n_te = len(c.patcher.patches)
        n_unet = len(m.patches)
        check("{} #{} {}: TE output identical ({} TE keys patched)".format(kind, i, combo, n_te), te_ok)
        check("{} #{} {}: sampled latent identical ({} UNet keys patched)".format(kind, i, combo, n_unet), d["bit_exact"], str(d))
        check("{} #{} {}: no backups on MODEL/CLIP (native had {})".format(kind, i, combo, native_backups[combo]), backups(m, c.patcher) == 0)
        summary.setdefault(kind, {})[combo] = {"unet_keys": n_unet, "te_keys": n_te, "native_backups": native_backups[combo], "vs_native": d}
    lora_effect = diff_stats(ref["none"][2], ref["duck"][2])["max_abs"]
    check("{}: LoRA changes the output (duck vs none max_abs {:.3f})".format(kind, lora_effect), lora_effect > 0)
    check("{}: while switching LoRA, MODEL weights equal the loaded state".format(kind), not digests_diff(d_model, weight_digests(model.model)))
    check("{}: while switching LoRA, CLIP weights equal the loaded state".format(kind), not digests_diff(d_clip, weight_digests(clip.patcher.model)))

    # hook LoRA
    hclip, hpos, hneg = hooked_inputs(clip, hook_sd)
    out = sample(model, hpos, hneg, latent, steps=max(a.steps, 3))
    rpos, rneg, rout = ref["hook"]
    check("{} hook LoRA: TE output identical (SetClipHooks)".format(kind), cond_equal(rpos, hpos) and cond_equal(rneg, hneg))
    d = diff_stats(rout, out)
    check("{} hook LoRA (keyframed): sampled latent identical".format(kind), d["bit_exact"], str(d))
    check("{} hook LoRA: hook differs from plain".format(kind), not torch.equal(rout, ref["none"][2]))
    check("{} hook LoRA: no hook_backup / cached_hook_patches".format(kind), backups(model, hclip.patcher, clip.patcher) == 0)
    summary.setdefault(kind, {})["hook"] = {"vs_native": d}

    # LoRA removed
    free_all()
    m, c, pos, neg, out = run_combo(model, clip, "none", latent, a.steps)
    check("{}: after removing LoRA, output equals the no-LoRA reference".format(kind),
          torch.equal(out, ref["none"][2]) and cond_equal(pos, ref["none"][0]))
    free_all()
    check("{}: after removing LoRA, MODEL weights byte-identical to load time".format(kind), not digests_diff(d_model, weight_digests(model.model)))
    check("{}: after removing LoRA, CLIP weights byte-identical to load time".format(kind), not digests_diff(d_clip, weight_digests(clip.patcher.model)))
    left = runtime_functions_left(model.model, clip.patcher.model)
    check("{}: no weight functions left on modules".format(kind), left == 0, str(left))
    return model, clip


def test_refusals(model, clip):
    m, c = apply_loras(model, clip, COMBOS["duck"])
    expect_raises("refuse force_patch_weights (MODEL)", MonoloadUnsupportedError,
                  lambda: comfy.model_management.load_models_gpu([m], force_patch_weights=True), "force_patch_weights", "key=")
    free_all()
    expect_raises("refuse force_patch_weights (CLIP)", MonoloadUnsupportedError,
                  lambda: comfy.model_management.load_models_gpu([c.patcher], force_patch_weights=True), "force_patch_weights", "key=")
    free_all()

    key = sorted(k for k in m.patches if k.endswith(".weight"))[0]
    module = comfy.utils.get_attr(model.model, key.rsplit(".", 1)[0])
    orig_cls = module.__class__
    module.__class__ = torch.nn.Conv2d if isinstance(module, torch.nn.Conv2d) else torch.nn.Linear
    try:
        m2, _ = apply_loras(model, clip, COMBOS["duck"])
        expect_raises("refuse LoRA on a non-comfy.ops parameter", MonoloadUnsupportedError,
                      lambda: comfy.model_management.load_models_gpu([m2]), "lora_non_comfy_ops_param", key)
    finally:
        module.__class__ = orig_cls
        free_all()

    m3 = model.clone()
    w = comfy.utils.get_attr(model.model, key)
    bigger = torch.zeros([w.shape[0] + 8] + list(w.shape[1:]), dtype=w.dtype)
    m3.add_patches({key: ("diff", (bigger, {"pad_weight": True}))}, 1.0)
    expect_raises("refuse shape-changing patch", MonoloadUnsupportedError,
                  lambda: comfy.model_management.load_models_gpu([m3]), "lora_shape_change", key)
    free_all()

    m4, _ = apply_loras(model, clip, COMBOS["duck"])  # read the LoRA before faking aimdo (its loader would use aimdo mmap)
    comfy.memory_management.aimdo_enabled = True
    try:
        expect_raises("refuse LoRA under DynamicVRAM", MonoloadUnsupportedError,
                      lambda: comfy.model_management.load_models_gpu([m4]), "dynamic_vram")
    finally:
        comfy.memory_management.aimdo_enabled = False
        free_all()
    check("no backups left after refusals", backups(model, clip.patcher, m, c.patcher, m3) == 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--unet", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--only", choices=["checkpoint", "unet+clip"])
    a = p.parse_args()
    summary = {}
    last = None
    for kind in ("checkpoint", "unet+clip"):
        if a.only and kind != a.only:
            continue
        last = test_pipeline(kind, a, summary)
    test_refusals(*last)
    finish(summary)


if __name__ == "__main__":
    main()
