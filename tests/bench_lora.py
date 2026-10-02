"""Benchmark: native LoRA (baked + backup) vs Monoload runtime merge.

Runs the same LoRA sequence in one process, on the same loaded models, in
several modes (--modes, run in order):
  native*          native ComfyUI (Monoload uninstalled)
  monoload         Monoload, default merge path (fused fp16 addmm / relaxed)
  monoload-exact*  Monoload with the bit-exact path (= MONOLOAD_EXACT=1)
  monoload-relaxed* diagnostic: the default path with fusion off (relaxed merge,
                   native lowvram numerics, for every patch)
  bypass*          ComfyUI's own bypass LoRA (comfy.sd.load_bypass_lora_for_models,
                   as the LoraLoaderBypass node does; comparison data only)
Reports per combination:
  lora   : LoraLoader node calls (reads the LoRA files, builds patches)
  encode : CLIPTextEncode of positive + negative (includes loading/patching the TE)
  patch  : load_models_gpu of the diffusion model (native: restore previous
           backups + bake the new combination; Monoload: attach weight functions)
  step   : median seconds per sampling step (steps 2..N; step 1 reported separately)
plus GTT / RSS after the combination and the max difference of the final latent
between the two modes.

Examples (inside the ComfyUI container; free the server's models first):
  curl -X POST http://127.0.0.1:8188/free -d '{"unload_models":true,"free_memory":true}'
  cd /opt/ComfyUI/custom_nodes/monoload
  # SDXL checkpoint
  python tests/bench_lora.py --checkpoint sd_xl_base_1.0.safetensors --width 1024 --height 1024 --steps 20 --cfg 6 \
      --combo a.safetensors --combo b.safetensors --combo a.safetensors+b.safetensors --combo none
  # Krea 2 (UNET + CLIP loaded separately)
  python tests/bench_lora.py --unet krea2_turbo_bf16.safetensors --clip qwen3vl_4b_bf16.safetensors --clip-type krea2 \
      --width 1024 --height 1024 --steps 8 --cfg 1 --sampler euler --scheduler simple --combo x.safetensors --combo none

ComfyUI launch args: $COMFY_ARGS if set, otherwise those of the container's
main process (PID 1), otherwise "--cpu".
"""

import argparse
import logging
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

from common import (COMFY_ARGS, UNIT_ROUNDOFF, compare_dtype, error_bound, free_all, hotpatch, load_checkpoint,  # noqa: E402
                    load_clip, load_unet, set_runtime)
import comfy.model_management  # noqa: E402
import comfy.sample  # noqa: E402
import nodes  # noqa: E402


def _gtt_gib():
    import glob
    import re
    seen = {}
    for d in glob.glob("/sys/class/drm/card*"):
        if re.search(r"/card\d+$", d):
            p = os.path.join(d, "device", "mem_info_gtt_used")
            if os.path.exists(p):
                seen.setdefault(os.path.realpath(os.path.join(d, "device")), p)
    if not seen:
        return None
    return sum(int(open(p).read()) for p in seen.values()) / 2 ** 30


def _rss_gib():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 30


def _sync():
    dev = comfy.model_management.get_torch_device()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def parse_combo(s, default_strength):
    if s == "none":
        return []
    out = []
    for part in s.split("+"):
        bits = part.split(":")
        name = bits[0]
        sm = float(bits[1]) if len(bits) > 1 else default_strength
        sc = float(bits[2]) if len(bits) > 2 else sm
        out.append((name, sm, sc))
    return out


_BYPASS_LORA_CACHE = {}
BYPASS_NOISE = {"suppressed": 0}


class _QuietBypassWarnings(logging.Filter):
    """load_bypass_lora_for_models walks UNet and text-encoder adapters in one
    dict and warns for every text-encoder key while attaching to the UNet
    (they are attached to the CLIP right after). Count instead of print."""

    def filter(self, record):
        if "[BypassLoRA] Adapter key not in model state_dict" in record.getMessage():
            BYPASS_NOISE["suppressed"] += 1
            return False
        return True


def _bypass_lora(name):
    import comfy.utils
    import folder_paths
    lora = _BYPASS_LORA_CACHE.get(name)
    if lora is None:
        lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", name), safe_load=True)
        _BYPASS_LORA_CACHE[name] = lora
    return lora


def _combined_injection(injections):
    """One PatcherInjection running several bypass injections: inject in order,
    eject in reverse, so stacked hooks on the same module unwind correctly
    (ModelPatcher.eject_model ejects in insertion order)."""
    from comfy.patcher_extension import PatcherInjection

    def inject(mp):
        for inj in injections:
            inj.inject(mp)

    def eject(mp):
        for inj in reversed(injections):
            inj.eject(mp)
    return PatcherInjection(inject=inject, eject=eject)


def apply_bypass(model, clip, loras):
    """ComfyUI's bypass LoRA for one or several LoRAs. Calling
    load_bypass_lora_for_models repeatedly does not stack: every call stores
    its injection under the same key "bypass_lora", replacing the previous
    LoRA. So each LoRA is loaded against the base, and the injections (and any
    regular, non-adapter patches) are combined on one clone."""
    import uuid as _uuid
    import comfy.sd
    m, c = model.clone(), clip.clone()
    inj_m, inj_c = [], []
    root = logging.getLogger()
    flt = _QuietBypassWarnings()
    root.addFilter(flt)
    try:
        for name, sm, sc in loras:
            mi, ci = comfy.sd.load_bypass_lora_for_models(model, clip, _bypass_lora(name), sm, sc)
            inj_m += mi.injections.get("bypass_lora", [])
            inj_c += ci.patcher.injections.get("bypass_lora", [])
            for src, dst in ((mi, m), (ci.patcher, c.patcher)):
                for k, v in src.patches.items():
                    dst.patches.setdefault(k, []).extend(v)
                    dst.patches_uuid = _uuid.uuid4()
    finally:
        root.removeFilter(flt)
    if inj_m:
        m.set_injections("bypass_lora", [_combined_injection(inj_m)])
    if inj_c:
        c.patcher.set_injections("bypass_lora", [_combined_injection(inj_c)])
    return m, c


def bypass_hooks(module):
    """Modules whose forward is currently a bypass hook (an ejected hook leaves
    the original bound method behind in __dict__, so check the owner)."""
    from comfy.weight_adapter.bypass import BypassForwardHook
    n = 0
    for mm in module.modules():
        f = mm.__dict__.get("forward")
        if f is not None and isinstance(getattr(f, "__self__", None), BypassForwardHook):
            n += 1
    return n


def apply_mode_loras(model, clip, loras, mode):
    """native*/monoload*: the LoraLoader node. bypass*: ComfyUI's own bypass
    LoRA (comfy.sd.load_bypass_lora_for_models, what the LoraLoaderBypass node
    calls) -- comparison data only, Monoload is uninstalled for it."""
    if mode.startswith("bypass"):
        return apply_bypass(model, clip, loras) if loras else (model, clip)
    m, c = model, clip
    for name, sm, sc in loras:
        m, c = nodes.LoraLoader().load_lora(m, c, name, sm, sc)
    return m, c


def run_sequence(model, clip, combos, a, results, mode):
    latent0 = torch.zeros(1, 4, a.height // 8, a.width // 8)
    for rep in range(a.repeat):
        for label, loras in combos:
            _sync()
            t0 = time.perf_counter()
            m, c = apply_mode_loras(model, clip, loras, mode)
            t_lora = time.perf_counter() - t0

            _sync()
            t0 = time.perf_counter()
            pos = nodes.CLIPTextEncode().encode(c, a.prompt)[0]
            neg = nodes.CLIPTextEncode().encode(c, a.negative)[0]
            _sync()
            t_encode = time.perf_counter() - t0
            te_keys = bypass_hooks(c.patcher.model) if mode.startswith("bypass") else len(c.patcher.patches)

            t0 = time.perf_counter()
            comfy.model_management.load_models_gpu([m])
            _sync()
            t_patch = time.perf_counter() - t0
            unet_keys = bypass_hooks(m.model) if mode.startswith("bypass") else len(m.patches)

            latent = comfy.sample.fix_empty_latent_channels(m, latent0)
            noise = comfy.sample.prepare_noise(latent, a.seed, None)
            stamps = []

            def cb(step, x0, x, total):
                _sync()
                stamps.append(time.perf_counter())

            _sync()
            t0 = time.perf_counter()
            out = comfy.sample.sample(m, noise, a.steps, a.cfg, a.sampler, a.scheduler, pos, neg, latent,
                                      denoise=1.0, callback=cb, disable_pbar=True, seed=a.seed)
            _sync()
            t_total = time.perf_counter() - t0
            steps = [stamps[0] - t0] + [b - x for x, b in zip(stamps, stamps[1:])]
            row = {
                "mode": mode, "rep": rep, "combo": label,
                "unet_keys": unet_keys, "te_keys": te_keys,
                "backups": len(m.backup) + len(c.patcher.backup),
                "lora": t_lora, "encode": t_encode, "patch": t_patch,
                "step1": steps[0], "step": statistics.median(steps[1:]) if len(steps) > 1 else steps[0],
                "sample": t_total, "gtt": _gtt_gib(), "rss": _rss_gib(),
            }
            results.append((row, out.float().cpu()))
            print("{mode:8s} rep{rep} {combo:28s} keys {unet_keys:4d}/{te_keys:4d} backups {backups:4d} | lora {lora:6.2f}s encode {encode:6.2f}s "
                  "patch {patch:6.2f}s | step1 {step1:6.3f}s step {step:6.3f}s total {sample:7.2f}s | GTT {g} RSS {rss:5.2f} GiB".format(
                      g="{:6.2f}".format(row["gtt"]) if row["gtt"] is not None else "   n/a", **row), flush=True)
            del m, c, pos, neg


def layer_probe(model, clip, loras, reps=3):
    """Per-model-call cost of the weight functions alone, summed over all
    patched layers of the diffusion model:
      copy    : the temporary copy cast_bias_weight makes (paid by every variant)
      exact   : Monoload's bit-exact path (MONOLOAD_EXACT=1)
      default : Monoload's default path (fused addmm_ for plain LoRA/LoCon, relaxed for the rest)
      relaxed : calculate_weight in the compute dtype for every key (native lowvram numerics)
      mm-add  : diagnostic only: plain LoRA/LoCon as torch.mm(up, down) then
                w.add_(delta, alpha=scale) (two kernels, delta rounded once), the rest relaxed
    plus the weight difference of default / relaxed against native's merge
    (= exact), with the tolerance of docs/DESIGN.md §5.5."""
    import comfy.lora
    import comfy.utils
    from monoload.hotpatch import _to_device, _is_runtime_patch, fused_factors
    set_runtime(True)
    m, c = model, clip
    for name, sm, sc in loras:
        m, c = nodes.LoraLoader().load_lora(m, c, name, sm, sc)
    comfy.model_management.load_models_gpu([m])
    dev = comfy.model_management.get_torch_device()
    cdt = m.model.get_dtype_inference() if hasattr(m.model, "get_dtype_inference") else m.model.get_dtype()
    ldt = comfy.model_management.lora_compute_dtype(dev)
    items = []
    cache = {}
    for key in m.patches:
        mod = comfy.utils.get_attr(m.model, key.rsplit(".", 1)[0])
        attr = key.rsplit(".", 1)[1]
        f = next((x for x in mod.__dict__.get(attr + "_function", []) if _is_runtime_patch(x)), None)
        if f is None:
            continue
        param = getattr(mod, attr)
        moved = _to_device(list(m.patches[key]), dev, cache, param.numel(), {"transient": False})
        items.append((key, param, f, moved, all(fused_factors(p) is not None for p in moved)))

    def variant(kind, key, f, moved, w):
        if kind in ("exact", "default"):
            hotpatch.set_exact(kind == "exact")
            return f(w)
        if kind == "relaxed":
            return comfy.lora.calculate_weight(moved, w, key, intermediate_dtype=cdt)
        if kind == "mm-add":
            for p in moved:
                ff = fused_factors(p)
                if ff is None:
                    w = comfy.lora.calculate_weight([p], w, key, intermediate_dtype=cdt)
                else:
                    up, down, scale = ff
                    w.view(w.shape[0], -1).add_(torch.mm(up.flatten(1).to(cdt), down.flatten(1).to(cdt)), alpha=scale)
            return w
        return w

    def run(kind):
        _sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            for key, param, f, moved, _fused in items:
                w = comfy.model_management.cast_to_device(param, dev, None, copy=True).to(cdt)
                w = variant(kind, key, f, moved, w)
                del w
        _sync()
        return (time.perf_counter() - t0) / reps

    prev = hotpatch.is_exact()
    try:
        # accuracy against the bit-exact merge (= native), relative to the LoRA's own change
        if not items:
            print("\n=== layer probe: no patched layers ===")
            return
        cmp = compare_dtype(items[0][1].dtype, cdt, ldt)
        err = {k: {"d_sq": 0.0, "max": 0.0} for k in ("default", "relaxed", "mm-add")}
        lora_sq = w_sq = bound_sq = 0.0
        for key, param, f, moved, _fused in items:
            base = comfy.model_management.cast_to_device(param, dev, None, copy=True).to(cdt)
            exact = variant("exact", key, f, moved, base.clone()).to(cmp).double()
            ch = float(((exact - base.to(cmp).double()) ** 2).sum())
            wn = float(exact.norm())
            lora_sq += ch
            w_sq += wn ** 2
            bound_sq += error_bound(cmp, wn, ch ** 0.5) ** 2
            for kind in err:
                d = variant(kind, key, f, moved, base.clone()).to(cmp).double() - exact
                err[kind]["d_sq"] += float((d ** 2).sum())
                err[kind]["max"] = max(err[kind]["max"], float(d.abs().max()))
            del base, exact
        n_fused = sum(1 for it in items if it[4])

        timings = {}
        run("exact")  # warm-up
        for kind in ("copy", "exact", "default", "relaxed", "mm-add"):
            timings[kind] = run(kind)
    finally:
        hotpatch.set_exact(prev)

    nbytes = sum(it[1].numel() * it[1].element_size() for it in items)
    t_copy = timings["copy"]
    print("\n=== layer probe: {} patched layers ({} fused), {:.2f} GiB of patched weights, compute dtype {}, lora_compute_dtype {} ===".format(
        len(items), n_fused, nbytes / 2 ** 30, cdt, ldt))
    print("  per model call: temporary copy {:.3f}s | bit-exact {:.3f}s (+copy) | default (fused/relaxed) {:.3f}s (+copy) | "
          "relaxed only {:.3f}s (+copy) | mm-add {:.3f}s (+copy, diagnostic)".format(
              t_copy, timings["exact"] - t_copy, timings["default"] - t_copy, timings["relaxed"] - t_copy, timings["mm-add"] - t_copy))
    tol = (bound_sq / lora_sq) ** 0.5 if lora_sq else float("nan")
    u = UNIT_ROUNDOFF[cmp]
    print("  weight difference from bit-exact (= native merge), compared in {}; ||W|| / ||ΔW_lora|| = {:.3g}; tolerance (DESIGN §5.5) {:.3g}".format(
        cmp, (w_sq / lora_sq) ** 0.5 if lora_sq else float("nan"), tol))
    for kind, e in err.items():
        rel = (e["d_sq"] / lora_sq) ** 0.5 if lora_sq else float("nan")
        print("    {:16s}: ||Δw|| / ||ΔW_lora|| = {:.3g} ({}), ||Δw|| = {:.3g} u·||W||, max |Δw| {:.3g}".format(
            kind, rel, "within tolerance" if rel <= tol else "OVER TOLERANCE", (e["d_sq"] / w_sq) ** 0.5 / u if w_sq else float("nan"), e["max"]))
    print("  (one sampling step = 1 model call when cond/uncond are batched; the TE is separate)")
    free_all()


def _mode_desc(mode):
    if mode.startswith("monoload-relaxed"):
        return "Monoload, default path with fusion off: relaxed merge only (diagnostic)"
    if mode.startswith("monoload-exact"):
        return "Monoload, bit-exact path = MONOLOAD_EXACT=1"
    if mode.startswith("monoload"):
        return "Monoload, default path: fused fp16 addmm / relaxed"
    if mode.startswith("bypass"):
        return "ComfyUI bypass LoRA, comparison only"
    return "native ComfyUI"


def main():
    p = argparse.ArgumentParser(description="native vs Monoload LoRA benchmark")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="CheckpointLoaderSimple file (e.g. SDXL)")
    src.add_argument("--unet", help="UNETLoader file (use with --clip)")
    p.add_argument("--clip", help="CLIPLoader file")
    p.add_argument("--clip-type", default="stable_diffusion")
    p.add_argument("--combo", action="append", required=True,
                   help="LoRA combination: 'none' or 'a.safetensors[:model[:clip]]+b.safetensors...'; repeatable, run in order")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--sampler", default="euler")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeat", type=int, default=2, help="run the combo sequence this many times per mode (first pass includes warm-up)")
    p.add_argument("--modes", default="native,monoload,monoload-exact,bypass",
                   help="comma list run in order: 'monoload' = Monoload, default path (fused/relaxed); 'monoload-exact*' = "
                        "Monoload, bit-exact path (MONOLOAD_EXACT=1); 'monoload-relaxed*' = default path without fusion (diagnostic); "
                        "'bypass*' = ComfyUI's bypass LoRA "
                        "(load_bypass_lora_for_models, Monoload uninstalled, comparison only); anything else = native")
    p.add_argument("--prompt", default="a photo of a red fox sitting in fresh snow, golden hour, detailed fur")
    p.add_argument("--negative", default="blurry, lowres")
    p.add_argument("--no-probe", action="store_true", help="skip the per-layer weight-function probe")
    a = p.parse_args()
    combos = [(c, parse_combo(c, a.strength)) for c in a.combo]
    print("ComfyUI args: {}".format(" ".join(COMFY_ARGS)))
    print("device: {}  lora_compute_dtype: {}".format(comfy.model_management.get_torch_device(),
                                                       comfy.model_management.lora_compute_dtype(comfy.model_management.get_torch_device())))

    if a.checkpoint:
        model, clip = load_checkpoint(a.checkpoint)
    else:
        if not a.clip:
            p.error("--unet needs --clip")
        model, clip = load_unet(a.unet), load_clip(a.clip, a.clip_type)
    print("model dtype {}  manual_cast {}".format(model.model.get_dtype(), model.model.manual_cast_dtype))

    results = []
    for mode in a.modes.split(","):
        set_runtime(mode.startswith("monoload"))
        hotpatch.set_exact(mode.startswith("monoload-exact"))
        print("\n=== {} ({}) ===".format(mode, _mode_desc(mode)))
        if mode.startswith("monoload-relaxed"):
            # diagnostic: the default path with fusion turned off (every patch through calculate_weight in the compute dtype)
            orig_ff = hotpatch.fused_factors
            hotpatch.fused_factors = lambda patch: None
            try:
                run_sequence(model, clip, combos, a, results, mode)
            finally:
                hotpatch.fused_factors = orig_ff
        else:
            run_sequence(model, clip, combos, a, results, mode)
    free_all()

    # summary on the last repetition: every mode against the first one
    last = a.repeat - 1
    modes = a.modes.split(",")
    by = {(r["mode"], r["combo"]): (r, out) for r, out in results if r["rep"] == last}
    ref_mode = modes[0]

    def effect(mode, label):
        """What the LoRA does in this mode: max/mean |combo - none| of the same mode."""
        cur, none = by.get((mode, label)), by.get((mode, "none"))
        if cur is None or none is None or label == "none":
            return None
        d = (cur[1] - none[1]).abs()
        return float(d.max()), float(d.mean())

    for other in modes[1:]:
        print("\n=== summary (repetition {}): {} vs {} ===".format(last + 1, other, ref_mode))
        print("{:28s} | {:>9s} {:>9s} {:>6s} | {:>8s} {:>8s} | {:>8s} {:>8s} | {:>9s} {:>9s} | {:>17s} {:>17s}".format(
            "combo", "step " + ref_mode[:4], "step " + other[:4], "ratio", "patch " + ref_mode[:2], "patch " + other[:2],
            "enc " + ref_mode[:4], "enc " + other[:4], "max|Δ|", "mean|Δ|",
            "effect " + ref_mode[:6] + " max/mean", "effect " + other[:6] + " max/mean"))
        for label, _ in combos:
            n = by.get((ref_mode, label))
            m = by.get((other, label))
            if not n or not m:
                continue
            rn, on = n
            rm, om = m
            d = (on - om).abs()
            en, eo = effect(ref_mode, label), effect(other, label)
            fmt_e = lambda e: "n/a" if e is None else "{:.3g}/{:.3g}".format(*e)
            print("{:28s} | {:8.3f}s {:8.3f}s {:5.2f}x | {:7.2f}s {:7.2f}s | {:7.2f}s {:7.2f}s | {:9.3g} {:9.3g} | {:>17s} {:>17s}".format(
                label[:28], rn["step"], rm["step"], rm["step"] / rn["step"] if rn["step"] else float("nan"),
                rn["patch"], rm["patch"], rn["encode"], rm["encode"], float(d.max()), float(d.mean()), fmt_e(en), fmt_e(eo)))
    print("\n=== seconds per step (repetition {}), ratio to {} ===".format(last + 1, ref_mode))
    print("{:28s} | ".format("combo") + " | ".join("{:>17s}".format(m_[:17]) for m_ in modes))
    for label, _ in combos:
        r0 = by.get((ref_mode, label))
        cells = []
        for m_ in modes:
            r = by.get((m_, label))
            if r is None:
                cells.append("{:>17s}".format("n/a"))
            else:
                ratio = r[0]["step"] / r0[0]["step"] if r0 and r0[0]["step"] else float("nan")
                cells.append("{:>9.3f}s {:5.2f}x".format(r[0]["step"], ratio))
        print("{:28s} | ".format(label[:28]) + " | ".join(cells))
    print("\nmax|Δ| / mean|Δ| = difference of the final latent between the two modes (0 = bit-identical).")
    print("effect = what the LoRA changes in that mode (combo vs the same mode's 'none'); compare mean|Δ| with it.")
    print("Add native2 to --modes (e.g. native,monoload,monoload-exact,native2) to see how much native differs from itself on this GPU.")
    print("bypass is not bit-identical to merging by design (it adds up(down(x)) to the layer output).")
    if BYPASS_NOISE["suppressed"]:
        print("({} '[BypassLoRA] Adapter key not in model state_dict' warnings suppressed: text-encoder keys seen while "
              "attaching to the UNet; they are attached to the CLIP separately)".format(BYPASS_NOISE["suppressed"]))
    probe = next((l for _, l in combos if l), None)
    if probe and not a.no_probe:
        layer_probe(model, clip, probe)


if __name__ == "__main__":
    main()
