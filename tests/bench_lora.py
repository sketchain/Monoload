"""Benchmark: native LoRA (baked + backup) vs Monoload runtime merge.

Runs the same LoRA sequence in one process, on the same loaded models, in
several modes: native ComfyUI (Monoload uninstalled), Monoload installed, and
ComfyUI's own bypass LoRA (comfy.sd.load_bypass_lora_for_models, as the
LoraLoaderBypass node does; comparison data only). Reports per combination:
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

from common import COMFY_ARGS, free_all, hotpatch, load_checkpoint, load_clip, load_unet, set_runtime  # noqa: E402
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


def apply_mode_loras(model, clip, loras, mode):
    """native*/monoload*: the LoraLoader node. bypass*: ComfyUI's own bypass
    LoRA (comfy.sd.load_bypass_lora_for_models, what the LoraLoaderBypass node
    calls) -- comparison data only, Monoload is uninstalled for it."""
    m, c = model, clip
    for name, sm, sc in loras:
        if mode.startswith("bypass"):
            import comfy.sd
            import comfy.utils
            import folder_paths
            lora = _BYPASS_LORA_CACHE.get(name)
            if lora is None:
                lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", name), safe_load=True)
                _BYPASS_LORA_CACHE.clear()
                _BYPASS_LORA_CACHE[name] = lora
            m, c = comfy.sd.load_bypass_lora_for_models(m, c, lora, sm, sc)
        else:
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

            t0 = time.perf_counter()
            comfy.model_management.load_models_gpu([m])
            _sync()
            t_patch = time.perf_counter() - t0

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
                "unet_keys": len(m.patches), "te_keys": len(c.patcher.patches),
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
    """Per-step cost of the weight functions alone, summed over all patched
    layers of the diffusion model: the temporary copy cast_bias_weight makes,
    Monoload's bit-exact merge, and a relaxed merge (native lowvram numerics:
    calculate_weight directly in the compute dtype, no lora dtype round trip,
    no rounding). Measurement only; the relaxed variant is not used anywhere."""
    import comfy.lora
    import comfy.utils
    from monoload.hotpatch import _to_device, _is_runtime_patch
    set_runtime(True)
    m, c = model, clip
    for name, sm, sc in loras:
        m, c = nodes.LoraLoader().load_lora(m, c, name, sm, sc)
    comfy.model_management.load_models_gpu([m])
    dev = comfy.model_management.get_torch_device()
    cdt = m.model.get_dtype_inference() if hasattr(m.model, "get_dtype_inference") else m.model.get_dtype()
    items = []
    cache = {}
    for key in m.patches:
        mod = comfy.utils.get_attr(m.model, key.rsplit(".", 1)[0])
        attr = key.rsplit(".", 1)[1]
        f = next((x for x in mod.__dict__.get(attr + "_function", []) if _is_runtime_patch(x)), None)
        if f is None:
            continue
        param = getattr(mod, attr)
        items.append((key, param, f, _to_device(list(m.patches[key]), dev, cache, param.numel(), {"transient": False})))

    def run(kind):
        _sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            for key, param, f, moved in items:
                w = comfy.model_management.cast_to_device(param, dev, None, copy=True).to(cdt)
                if kind == "exact":
                    w = f(w)
                elif kind == "relaxed":
                    w = comfy.lora.calculate_weight(moved, w, key, intermediate_dtype=cdt)
                del w
        _sync()
        return (time.perf_counter() - t0) / reps

    run("exact")  # warm-up
    t_copy, t_exact, t_relaxed = run("copy"), run("exact"), run("relaxed")
    nbytes = sum(p.numel() * p.element_size() for _, p, _, _ in items)
    print("\n=== layer probe: {} patched layers, {:.2f} GiB of patched weights, compute dtype {}, lora dtype {} ===".format(
        len(items), nbytes / 2 ** 30, cdt, comfy.model_management.lora_compute_dtype(dev)))
    print("  per model call: temporary copy {:.3f}s | bit-exact merge {:.3f}s (+copy) | relaxed merge {:.3f}s (+copy)".format(
        t_copy, t_exact - t_copy, t_relaxed - t_copy))
    print("  (one sampling step = 1 model call without CFG batching, the TE is separate)")
    free_all()


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
    p.add_argument("--modes", default="native,monoload,bypass",
                   help="comma list run in order: 'monoload*' = Monoload installed; 'bypass*' = ComfyUI's bypass LoRA "
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
        print("\n=== {} ===".format(mode))
        run_sequence(model, clip, combos, a, results, mode)
    free_all()

    # summary on the last repetition: every mode against the first one
    last = a.repeat - 1
    modes = a.modes.split(",")
    by = {(r["mode"], r["combo"]): (r, out) for r, out in results if r["rep"] == last}
    ref_mode = modes[0]
    for other in modes[1:]:
        print("\n=== summary (repetition {}): {} vs {} ===".format(last + 1, other, ref_mode))
        print("{:28s} | {:>12s} {:>12s} {:>8s} | {:>12s} {:>12s} | {:>12s} {:>12s} | {:>10s}".format(
            "combo", "step " + ref_mode[:6], "step " + other[:6], "ratio", "patch " + ref_mode[:5], "patch " + other[:5],
            "enc " + ref_mode[:6], "enc " + other[:6], "max|Δ|"))
        for label, _ in combos:
            n = by.get((ref_mode, label))
            m = by.get((other, label))
            if not n or not m:
                continue
            rn, on = n
            rm, om = m
            delta = float((on - om).abs().max())
            print("{:28s} | {:11.3f}s {:11.3f}s {:7.2f}x | {:11.2f}s {:11.2f}s | {:11.2f}s {:11.2f}s | {:10.3g}".format(
                label, rn["step"], rm["step"], rm["step"] / rn["step"] if rn["step"] else float("nan"),
                rn["patch"], rm["patch"], rn["encode"], rm["encode"], delta))
    print("\nmax|Δ| = max abs difference of the final latent between the two modes (0 = bit-identical).")
    print("Run with --modes native,monoload,bypass,native2 to see how much native differs from itself on this GPU.")
    print("bypass is not bit-identical to merging by design (it adds up(down(x)) to the layer output); its max|Δ| is for reference.")
    probe = next((l for _, l in combos if l), None)
    if probe and not a.no_probe:
        layer_probe(model, clip, probe)


if __name__ == "__main__":
    main()
