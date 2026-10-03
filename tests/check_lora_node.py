"""CT 700 check of the Monoload LoRA Settings node with a real model and LoRA.

Loads the checkpoint (default waiIllustriousSDXL_v170.safetensors) and the
LoRA (--lora, through ComfyUI's LoraLoader), samples the same seed natively
(Monoload's hooks uninstalled) as the reference, then once per row below, and
prints for each row: which path the LoRA actually took (Monoload's runtime
merge, or native: baked into the weights with backups), the settings and
where each came from, the per-step time, the GTT after sampling, the
difference to native (text-encoder output and latent), and what the
per-prompt release does with it.

    docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/check_lora_node.py --lora <file in models/loras>

Without --lora (or with a file that is not there) it lists models/loras and
models/checkpoints and exits. Same launch arguments as tests/bench_lora.py
($COMFY_ARGS, else those of the container's main process).

Rows (expected result in brackets):
  no node                    global default [Monoload, fused: small diff, no backups, released]
  node enable                [same as no node, bit for bit]
  node exact                 [Monoload, == native bit for bit, no backups]
  node native                [native: == native bit for bit, backups, kept]
  node keep                  [Monoload, fused, kept after the prompt]
  node exact before LoRA     node between the checkpoint and LoraLoader [== native bit for bit]
  MONOLOAD=0, no node        [native: == native bit for bit, backups, kept]
  MONOLOAD=0, node enable    [Monoload, == "no node" bit for bit, released]
"""

import argparse
import os
import shlex
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if "COMFY_ARGS" not in os.environ:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from monoload import comfy_env as _ce
    _auto = _ce.pid1_comfy_args()
    os.environ["COMFY_ARGS"] = shlex.join(_auto) if _auto is not None else "--cpu"

import torch  # noqa: E402

from common import COMFY_ARGS, free_all, set_runtime  # noqa: E402
from bench_lora import _gtt_gib, _sync  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.sample  # noqa: E402
import folder_paths  # noqa: E402
import nodes  # noqa: E402
from monoload import lora_overrides as lo, release, settings  # noqa: E402
from monoload.nodes import NODE_CLASS_MAPPINGS  # noqa: E402


def node(model, clip, **kw):
    cls = NODE_CLASS_MAPPINGS["MonoloadLoRASettings"]
    args = {"mode": "default", "merge": "default", "after_prompt": "default"}
    args.update(kw)
    return getattr(cls(), cls.FUNCTION)(model=model, clip=clip, **args)


def runtime_patches(model):
    return sum(1 for m in model.model.modules() for a in ("weight_function", "bias_function")
               for f in (m.__dict__.get(a) or ()) if getattr(f, "is_monoload_patch", False))


def sample(model, clip, a):
    pos = nodes.CLIPTextEncode().encode(clip, a.prompt)[0]
    neg = nodes.CLIPTextEncode().encode(clip, a.negative)[0]
    latent = comfy.sample.fix_empty_latent_channels(model, torch.zeros(1, 4, a.height // 8, a.width // 8))
    noise = comfy.sample.prepare_noise(latent, a.seed, None)
    mm.load_models_gpu([model])
    stamps = []

    def cb(step, x0, x, total):
        _sync()
        stamps.append(time.perf_counter())

    _sync()
    t0 = time.perf_counter()
    out = comfy.sample.sample(model, noise, a.steps, a.cfg, a.sampler, a.scheduler, pos, neg, latent,
                              denoise=1.0, callback=cb, disable_pbar=True, seed=a.seed)
    _sync()
    steps = [stamps[0] - t0] + [b - x for x, b in zip(stamps, stamps[1:])]
    return {"cond": pos[0][0].float().cpu(), "latent": out.float().cpu(), "step1": steps[0],
            "step": statistics.median(steps[1:]) if len(steps) > 1 else steps[0],
            "backups": len(model.backup) + len(clip.patcher.backup), "runtime": runtime_patches(model) + runtime_patches(clip.patcher),
            "gtt": _gtt_gib()}


def listing():
    for kind in ("loras", "checkpoints"):
        names = folder_paths.get_filename_list(kind)
        print("models/{} ({}):".format(kind, len(names)))
        for n in names:
            print("  " + n)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="waiIllustriousSDXL_v170.safetensors", help="models/checkpoints file")
    p.add_argument("--lora", help="models/loras file (omit to list the files)")
    p.add_argument("--strength", type=float, default=1.0, help="LoRA strength (model and CLIP)")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=6.0)
    p.add_argument("--sampler", default="euler")
    p.add_argument("--scheduler", default="normal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default="1girl, red fox ears, sitting in fresh snow, golden hour, detailed")
    p.add_argument("--negative", default="blurry, lowres")
    a = p.parse_args()
    if not a.lora or a.lora not in folder_paths.get_filename_list("loras") or a.checkpoint not in folder_paths.get_filename_list("checkpoints"):
        if a.lora:
            print("not found: --lora {} or --checkpoint {}".format(a.lora, a.checkpoint))
        else:
            print("pass --lora <file>; available:")
        listing()
        return
    print("ComfyUI args {}; master switch MONOLOAD {}; global merge {}; global after prompt {}".format(
        " ".join(COMFY_ARGS), "on" if settings.master() else "off", "exact" if settings.exact() else "fused",
        "keep" if settings.keep() else "release"), flush=True)
    base_model, base_clip = nodes.CheckpointLoaderSimple().load_checkpoint(a.checkpoint)[:2]
    lm, lc = nodes.LoraLoader().load_lora(base_model, base_clip, a.lora, a.strength, a.strength)
    print("checkpoint {}, LoRA {} (strength {}): {} UNet / {} text-encoder keys patched; {}x{}, {} steps, cfg {}, {} / {}, seed {}".format(
        a.checkpoint, a.lora, a.strength, len(lm.patches), len(lc.patcher.patches), a.width, a.height, a.steps, a.cfg,
        a.sampler, a.scheduler, a.seed), flush=True)

    set_runtime(False)
    try:
        ref = sample(lm, lc, a)
    finally:
        set_runtime(True)
    print("{:26s} native ComfyUI (hooks uninstalled): {} backups | step1 {:.3f}s step {:.3f}s | GTT {}".format(
        "reference", ref["backups"], ref["step1"], ref["step"], "{:.2f} GiB".format(ref["gtt"]) if ref["gtt"] is not None else "n/a"), flush=True)

    def before_lora(**kw):
        m, c = node(base_model, base_clip, **kw)
        return nodes.LoraLoader().load_lora(m, c, a.lora, a.strength, a.strength)

    rows = [
        ("no node", True, lambda: (lm, lc)),
        ("node enable", True, lambda: node(lm, lc, mode="enable")),
        ("node exact", True, lambda: node(lm, lc, merge="exact")),
        ("node native", True, lambda: node(lm, lc, mode="native")),
        ("node keep", True, lambda: node(lm, lc, after_prompt="keep")),
        ("node exact before LoRA", True, lambda: before_lora(merge="exact")),
        ("MONOLOAD=0, no node", False, lambda: (lm, lc)),
        ("MONOLOAD=0, node enable", False, lambda: node(lm, lc, mode="enable")),
    ]
    results = {}
    saved_master = settings.master()
    for label, master, make in rows:
        settings.set_master(master)
        try:
            free_all()
            m, c = make()
            r = sample(m, c, a)
            eff, src = lo.resolve(m)
            rel = release.release_after_prompt(None)
            r["released"] = rel["models"]
        finally:
            settings.set_master(saved_master)
        results[label] = r
        path = "Monoload runtime merge ({} weight functions, {} backups)".format(r["runtime"], r["backups"]) if r["runtime"] else \
            "native: LoRA baked into the weights ({} backups)".format(r["backups"])
        dl = float((r["latent"] - ref["latent"]).abs().max())
        dc = float((r["cond"] - ref["cond"]).abs().max())
        same = dl == 0 and dc == 0
        print("{:26s} {}\n{:26s} {}\n{:26s} step1 {:.3f}s step {:.3f}s | GTT {} | vs native: {} (latent max|Δ| {:.3g}, TE max|Δ| {:.3g}) "
              "| after the prompt: {}".format(
                  label, path, "", lo.note(eff, src), "", r["step1"], r["step"],
                  "{:.2f} GiB".format(r["gtt"]) if r["gtt"] is not None else "n/a",
                  "IDENTICAL" if same else "differs", dl, dc,
                  "released ({} model(s) back to base)".format(r["released"]) if r["released"] else "kept"), flush=True)
    free_all()
    same_fused = torch.equal(results["no node"]["latent"], results["node enable"]["latent"]) and \
        torch.equal(results["no node"]["latent"], results["MONOLOAD=0, node enable"]["latent"])
    print("\n'no node' == 'node enable' == 'MONOLOAD=0, node enable' bit for bit: {}".format(same_fused))
    print("expected: node exact / node native / node exact before LoRA / MONOLOAD=0 no node IDENTICAL to native; native rows "
          "baked with backups and kept; Monoload rows without backups; no node / node enable / node keep the fused default "
          "(small difference, faster steps than exact); released except node keep, node native and MONOLOAD=0 no node.")


if __name__ == "__main__":
    main()
