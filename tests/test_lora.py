"""LoRA on Monoload models: runtime per-layer merge vs native (baked) LoRA.

    python tests/test_lora.py --source SRC --converted CONV --loras a.safetensors,b.safetensors [--hook-lora x.safetensors]
"""

import argparse
import os

import torch

from common import *  # noqa: F401,F403
from common import (MODELS, MonoloadUnsupportedError, check, compare_with_file, diff_stats, expect_raises,
                    family_inputs, finish, free_all, load_monoload, load_native, sample)
import comfy.hooks
import comfy.model_management
import comfy.utils
import folder_paths
import nodes


def lora_model(patcher, name, strength=0.8):
    return nodes.LoraLoaderModelOnly().load_lora_model_only(patcher, name, strength)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="sd15")
    p.add_argument("--source", required=True)
    p.add_argument("--converted", required=True)
    p.add_argument("--loras", required=True)
    p.add_argument("--hook-lora")
    p.add_argument("--steps", type=int, default=2)
    a = p.parse_args()
    conv_path = os.path.join(MODELS, "monoload", a.converted)

    native = load_native(a.source)
    mono = load_monoload(a.converted)
    pos, neg, latent = family_inputs(a.family)

    base_m = sample(mono, pos, neg, latent, steps=a.steps)
    free_all()
    summary = {}

    for name in a.loras.split(","):
        ln = lora_model(native, name)
        lm = lora_model(mono, name)
        out_n = sample(ln, pos, neg, latent, steps=a.steps)
        native_backup = len(ln.backup)
        free_all()
        out_m = sample(lm, pos, neg, latent, steps=a.steps)
        d = diff_stats(out_n, out_m)
        dbase = diff_stats(base_m, out_m)
        summary[name] = {"vs_native": d, "effect_vs_base_max_abs": dbase["max_abs"], "patched_keys": len(lm.patches)}
        check("{}: {} keys patched, output identical to native LoRA".format(name, len(lm.patches)), d["bit_exact"],
              "{} (LoRA effect vs no-LoRA: max_abs {:.4f})".format(d, dbase["max_abs"]))
        check("{}: LoRA actually changes the output".format(name), dbase["max_abs"] > 0)
        check("{}: Monoload backup empty (native had {} backups)".format(name, native_backup), len(lm.backup) == 0 and len(lm.hook_backup) == 0)
        n, problems = compare_with_file(mono.model, conv_path)
        check("{}: while LoRA is loaded, all {} weights still equal the file".format(name, n), not problems, "; ".join(problems[:3]))
        free_all()
        del ln, lm

    # LoRA removed -> base model behaves exactly as before
    base_m2 = sample(mono, pos, neg, latent, steps=a.steps)
    check("after removing LoRA, output equals pre-LoRA output", torch.equal(base_m, base_m2), str(diff_stats(base_m, base_m2)))
    n, problems = compare_with_file(mono.model, conv_path)
    check("after removing LoRA, all {} weights equal the file".format(n), not problems, "; ".join(problems[:3]))
    has_fn = [m for m in mono.model.modules() if getattr(m, "weight_function", None) or getattr(m, "bias_function", None)]
    check("after removing LoRA, no weight functions left on modules", not has_fn, str(len(has_fn)))
    free_all()

    # Hook LoRA (scheduled strength on the positive cond only -> hook groups switch every step)
    if a.hook_lora:
        lora_sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", a.hook_lora), safe_load=True)

        def hooked(cond):
            hooks = comfy.hooks.create_hook_lora(lora_sd, strength_model=0.9, strength_clip=0.0)
            kf = comfy.hooks.HookKeyframeGroup()
            kf.add(comfy.hooks.HookKeyframe(strength=1.0, start_percent=0.0))
            kf.add(comfy.hooks.HookKeyframe(strength=0.4, start_percent=0.5))
            hooks.set_keyframes_on_hooks(kf)
            return comfy.hooks.set_hooks_for_conditioning(cond, hooks)

        hpos = hooked(pos)
        out_n = sample(native, hpos, neg, latent, steps=max(a.steps, 3))
        free_all()
        hpos = hooked(pos)
        out_m = sample(mono, hpos, neg, latent, steps=max(a.steps, 3))
        d = diff_stats(out_n, out_m)
        summary["hook:" + a.hook_lora] = {"vs_native": d}
        check("hook LoRA (keyframed): output identical to native", d["bit_exact"], str(d))
        free_all()
        plain = sample(mono, pos, neg, latent, steps=max(a.steps, 3))
        check("hook LoRA actually changes the output", not torch.equal(plain, out_m), "max_abs vs no hook {:.4f}".format(diff_stats(plain, out_m)["max_abs"]))
        check("hook LoRA: no hook_backup / cached_hook_patches", len(mono.hook_backup) == 0 and len(mono.cached_hook_patches) == 0,
              "hook_backup={} cached={}".format(len(mono.hook_backup), len(mono.cached_hook_patches)))
        free_all()
        n, problems = compare_with_file(mono.model, conv_path)
        check("after hook LoRA, all {} weights equal the file".format(n), not problems, "; ".join(problems[:3]))

    # Refusals: never fall back to modifying + backing up weights
    first = a.loras.split(",")[0]
    lm = lora_model(mono, first)
    expect_raises("force_patch_weights is refused", MonoloadUnsupportedError,
                  lambda: comfy.model_management.load_models_gpu([lm], force_patch_weights=True), "force_patch_weights", "key=")
    free_all()

    key = sorted(k for k in lm.patches if k.endswith(".weight"))[0]
    module = comfy.utils.get_attr(mono.model, key.rsplit(".", 1)[0])
    orig_cls = module.__class__
    plain = torch.nn.Conv2d if isinstance(module, torch.nn.Conv2d) else torch.nn.Linear
    module.__class__ = plain
    try:
        lm2 = lora_model(mono, first)
        expect_raises("LoRA on a non-comfy.ops parameter is refused", MonoloadUnsupportedError,
                      lambda: comfy.model_management.load_models_gpu([lm2]), "lora_non_comfy_ops_param", key)
    finally:
        module.__class__ = orig_cls
        free_all()

    lm3 = mono.clone()
    w = comfy.utils.get_attr(mono.model, key)
    bigger = torch.zeros([w.shape[0] + 8] + list(w.shape[1:]), dtype=w.dtype)
    lm3.add_patches({key: ("diff", (bigger, {"pad_weight": True}))}, 1.0)
    expect_raises("shape-changing patch is refused", MonoloadUnsupportedError,
                  lambda: comfy.model_management.load_models_gpu([lm3]), "lora_shape_change", key)
    free_all()
    check("no backups left anywhere", len(mono.backup) == 0 and len(lm3.backup) == 0)
    finish(summary)


if __name__ == "__main__":
    main()
