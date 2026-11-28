"""Flux 2 dev text-to-image, the whole pipeline on CT 700: Monoload's default vs MONOLOAD=0 (DESIGN §9.33).

The workflow is ComfyUI's own blueprint "Text to Image (Flux.2 Dev)" (62b3c94, blueprints/), run by calling its nodes
in this process in the order the executor runs them:

    CLIPLoader (type flux2) -> CLIPTextEncode -> FluxGuidance 4                        stage "text"
    UNETLoader -> EmptyFlux2LatentImage, Flux2Scheduler (20 steps), RandomNoise,
      KSamplerSelect euler, BasicGuider -> SamplerCustomAdvanced                      stage "sample"
    VAELoader -> VAEDecode                                                             stage "vae"

once per --res (default 1024x1024, then 2048x2048; the models stay loaded between them, as in a server). Each mode
runs in a child process of its own, with the plugin loaded by ComfyUI's custom-node loader: "monoload" (Monoload's
default: the master switch on, no Monoload node) and "native" (MONOLOAD=0: everything native). Per stage: wall
seconds (synchronised), GTT used (amdgpu sysfs, device-wide: every process) before / peak / after, the allocator's
reserved peak of this process (torch.cuda.max_memory_reserved, reset at the stage's start); the VAE stage also says
what decoded (Monoload's record: layer, plan, estimate; native: whether it ran out of memory and fell back to tiled).
The parent then compares the two modes per resolution: the final latents (max|Δ|: Monoload does nothing to a model
without LoRA, so they should be identical) and the images (PSNR, the VAE's difference).

Inside the ComfyUI container, with the server's models freed first (its process holds GTT too):

    curl -X POST http://127.0.0.1:8188/free -d '{"unload_models":true,"free_memory":true}'
    cd /opt/ComfyUI/custom_nodes/monoload
    python tests/run_flux2_dev.py [--res 1024x1024,2048x2048] [--modes monoload,native] [--steps 20] [--seed 42]
                                  [--out /opt/ComfyUI/output/flux2_dev] [--json report.json]

The model files (ComfyUI folder names): --unet flux2_dev_fp8mixed.safetensors (diffusion_models), --clip
mistral_3_small_flux2_fp8.safetensors (text_encoders), --vae flux2-vae.safetensors (vae).
ComfyUI launch args: $COMFY_ARGS if set, otherwise those of the container's main process (PID 1), otherwise "--cpu".
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GIB = 1024 ** 3
PROMPT = ("a red fox curled up asleep on a mossy rock in an autumn forest at dawn, soft golden light through the mist, "
          "detailed fur, shallow depth of field, photograph")


# ---------------------------------------------------------------------------
# child: one mode, every resolution
# ---------------------------------------------------------------------------

def child(a):
    sys.path.insert(0, HERE)
    if "COMFY_ARGS" not in os.environ:
        sys.path.insert(0, REPO)
        from monoload import comfy_env as _ce
        _auto = _ce.pid1_comfy_args()
        os.environ["COMFY_ARGS"] = shlex.join(_auto) if _auto is not None else "--cpu"
    import asyncio
    import logging

    import bench_vae as B   # ComfyUI environment (common), the GTT sampler
    import torch
    import comfy.model_management as mm
    import comfy.sd
    import nodes
    from comfy_extras import nodes_custom_sampler as ncs, nodes_flux as nf

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    if not asyncio.run(nodes.load_custom_node(REPO)):
        raise SystemExit("the Monoload plugin did not load from {}".format(REPO))
    # the plugin as ComfyUI loaded it (its modules are named after the path, not the top-level "monoload" the test helpers import)
    mvae = sys.modules.get(getattr(comfy.sd.VAE.decode, "__module__", ""))
    if mvae is None or not hasattr(mvae, "last_decode"):
        raise SystemExit("Monoload's VAE wrapper is not installed (MONOLOAD_DISABLE set?)")
    settings = sys.modules[mvae.__name__.rpartition(".")[0] + ".settings"]
    cuda = mm.get_torch_device().type == "cuda"
    report = {"mode": a.child, "MONOLOAD": os.environ.get("MONOLOAD", ""), "master": settings.master(),
              "comfy_args": os.environ["COMFY_ARGS"], "torch": torch.__version__, "steps": a.steps, "seed": a.seed, "runs": []}
    print("== mode {} (MONOLOAD={!r}, master switch {}), COMFY_ARGS {}".format(a.child, report["MONOLOAD"], "on" if report["master"] else "off",
                                                                         report["comfy_args"]), flush=True)
    tiled = {"n": 0}
    orig_tiled = comfy.sd.VAE.decode_tiled_

    def decode_tiled_(self, *x, **k):    # native's OOM -> tiled fallback (counted, not prevented)
        tiled["n"] += 1
        return orig_tiled(self, *x, **k)
    comfy.sd.VAE.decode_tiled_ = decode_tiled_
    oom_lines = []

    class _Warn(logging.Handler):
        def emit(self, record):
            m = record.getMessage()
            if "Ran out of memory" in m or "out of memory" in m.lower():
                oom_lines.append(m[:200])
    logging.getLogger().addHandler(_Warn())

    def stage(name, fn):
        B.sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            r0 = torch.cuda.memory_reserved()
        s = B.Sampler(a.sample_ms / 1000.0)
        t0 = time.perf_counter()
        with s:
            out = fn()
            B.sync()
        dt = time.perf_counter() - t0
        row = {"stage": name, "seconds": dt, "gtt_before": s.base["gtt"], "gtt_peak": s.peak["gtt"], "gtt_after": s.read()["gtt"]}
        if cuda:
            row.update(reserved_before=r0, reserved_peak=torch.cuda.max_memory_reserved(), reserved_after=torch.cuda.memory_reserved())
        return out, row

    clip = unet = vae = None
    for res in a.res.split(","):
        w, h = (int(x) for x in res.lower().split("x"))
        run = {"res": res, "stages": []}
        print("-- {}x{}".format(w, h), flush=True)

        def text():
            nonlocal clip
            if clip is None:
                clip = nodes.CLIPLoader().load_clip(a.clip, "flux2", "default")[0]
            cond = nodes.CLIPTextEncode().encode(clip, a.prompt)[0]
            return nf.FluxGuidance.execute(cond, a.guidance).result[0]
        cond, row = stage("text", text)
        run["stages"].append(row)

        def sample():
            nonlocal unet
            if unet is None:
                unet = nodes.UNETLoader().load_unet(a.unet, "default")[0]
            latent = nf.EmptyFlux2LatentImage.execute(w, h, 1).result[0]
            sigmas = nf.Flux2Scheduler.execute(a.steps, w, h).result[0]
            noise = ncs.RandomNoise.execute(a.seed).result[0]
            sampler = ncs.KSamplerSelect.execute("euler").result[0]
            guider = ncs.BasicGuider.execute(unet, cond).result[0]
            return ncs.SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent).result[0]
        samples, row = stage("sample", sample)
        run["stages"].append(row)

        def decode():
            nonlocal vae
            if vae is None:
                vae = nodes.VAELoader().load_vae(a.vae)[0]
            tiled["n"] = 0
            del oom_lines[:]
            return nodes.VAEDecode().decode(vae, samples)[0]
        image, row = stage("vae", decode)
        rec = mvae.last_decode() if report["master"] else {}
        row.update(tiled=tiled["n"], oom=list(oom_lines), strategy=rec.get("strategy"), adapter=rec.get("adapter"),
                   stripes=rec.get("stripes"), rows=rec.get("rows"), gn_scheme=rec.get("gn_scheme"),
                   estimate=(rec.get("estimate") or {}).get("total"))
        run["stages"].append(row)
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, "{}_{}x{}.pt".format(a.child, w, h))
        torch.save({"latent": samples["samples"].float().cpu(), "image": image.float().cpu()}, path)
        try:
            from PIL import Image
            Image.fromarray((image[0].float().clamp(0, 1).cpu().numpy() * 255).round().astype("uint8")).save(path[:-3] + ".png")
        except Exception as e:   # the comparison does not need the PNG
            print("   (PNG not written: {})".format(e))
        run["file"] = path
        for r in run["stages"]:
            print("   {:6s} {:8.1f} s | GTT before {} peak {} after {} GiB (peak - before {}){}{}".format(
                r["stage"], r["seconds"], B.gib(r["gtt_before"]), B.gib(r["gtt_peak"]), B.gib(r["gtt_after"]),
                B.gib(r["gtt_peak"] - r["gtt_before"]) if r["gtt_peak"] is not None and r["gtt_before"] is not None else "   n/a",
                " | reserved peak {} GiB".format(B.gib(r["reserved_peak"])) if "reserved_peak" in r else "",
                " | {}".format(describe_vae(r)) if r["stage"] == "vae" else ""), flush=True)
        report["runs"].append(run)
    with open(os.path.join(a.out, "{}.json".format(a.child)), "w") as f:
        json.dump(report, f, indent=1)


def describe_vae(r):
    if r.get("strategy"):
        d = "Monoload {}".format(r["strategy"])
        if r.get("adapter"):
            d += " ({}, {} x {} rows)".format(r["adapter"], r.get("stripes"), r.get("rows"))
        if r.get("estimate"):
            d += ", estimate {:.2f} GiB".format(r["estimate"] / GIB)
        return d
    if r.get("tiled"):
        return "native ran out of memory -> tiled ({}x)".format(r["tiled"])
    return "native"


# ---------------------------------------------------------------------------
# parent: the modes in child processes, then the comparison
# ---------------------------------------------------------------------------

def psnr(a, b):
    d = (a.double() - b.double())
    rmse = float(d.pow(2).mean().sqrt())
    return float("inf") if rmse == 0 else 20 * math.log10(1.0 / rmse)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--res", default="1024x1024,2048x2048")
    p.add_argument("--modes", default="monoload,native", help="monoload (default switches), native (MONOLOAD=0); each in its own process")
    p.add_argument("--steps", type=int, default=20, help="the blueprint's 20")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance", type=float, default=4.0, help="the blueprint's FluxGuidance 4")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--unet", default="flux2_dev_fp8mixed.safetensors")
    p.add_argument("--clip", default="mistral_3_small_flux2_fp8.safetensors")
    p.add_argument("--vae", default="flux2-vae.safetensors")
    p.add_argument("--out", default="/opt/ComfyUI/output/flux2_dev")
    p.add_argument("--sample-ms", type=float, default=20.0, help="GTT sampling interval")
    p.add_argument("--json", help="also write the combined report here")
    p.add_argument("--child", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.child:
        return child(a)
    reports = {}
    for mode in a.modes.split(","):
        env = dict(os.environ)
        env.pop("MONOLOAD", None)
        if mode == "native":
            env["MONOLOAD"] = "0"
        elif mode != "monoload":
            raise SystemExit("unknown mode {!r}".format(mode))
        args = [sys.executable, os.path.abspath(__file__), "--child", mode, "--res", a.res, "--steps", str(a.steps), "--seed", str(a.seed),
                "--guidance", str(a.guidance), "--prompt", a.prompt, "--unet", a.unet, "--clip", a.clip, "--vae", a.vae, "--out", a.out,
                "--sample-ms", str(a.sample_ms)]
        t0 = time.perf_counter()
        rc = subprocess.call(args, env=env)
        print("== mode {} finished: exit code {}, {:.0f} s".format(mode, rc, time.perf_counter() - t0), flush=True)
        path = os.path.join(a.out, "{}.json".format(mode))
        if rc == 0 and os.path.exists(path):
            with open(path) as f:
                reports[mode] = json.load(f)
    print("\n== summary (GTT: device-wide GiB; peak - before = what the stage added at its highest)")
    print("{:9s} {:10s} {:6s} {:>9s} {:>8s} {:>8s} {:>8s}  {}".format("mode", "res", "stage", "seconds", "before", "peak", "+peak", "VAE"))
    for mode, rep in reports.items():
        for run in rep["runs"]:
            for r in run["stages"]:
                up = (r["gtt_peak"] - r["gtt_before"]) / GIB if r["gtt_peak"] is not None and r["gtt_before"] is not None else float("nan")
                print("{:9s} {:10s} {:6s} {:9.1f} {:8.2f} {:8.2f} {:8.2f}  {}".format(
                    mode, run["res"], r["stage"], r["seconds"], (r["gtt_before"] or 0) / GIB, (r["gtt_peak"] or 0) / GIB, up,
                    describe_vae(r) if r["stage"] == "vae" else ""))
    combined = {"reports": reports, "compare": []}
    if "monoload" in reports and "native" in reports:
        import torch
        print("\n== monoload vs native")
        for res in a.res.split(","):
            w, h = (int(x) for x in res.lower().split("x"))
            try:
                m = torch.load(os.path.join(a.out, "monoload_{}x{}.pt".format(w, h)))
                n = torch.load(os.path.join(a.out, "native_{}x{}.pt".format(w, h)))
            except OSError as e:
                print("  {}: not compared ({})".format(res, e))
                continue
            dl = float((m["latent"] - n["latent"]).abs().max())
            ps = psnr(m["image"].clamp(0, 1), n["image"].clamp(0, 1))
            print("  {}: latent max|Δ| {:.3g} (0: the sampling is identical); image PSNR {:.2f} dB, max|Δ| {:.3g}".format(
                res, dl, ps, float((m["image"] - n["image"]).abs().max())))
            combined["compare"].append({"res": res, "latent_max_abs": dl, "image_psnr": ps})
    if a.json:
        with open(a.json, "w") as f:
            json.dump(combined, f, indent=1)


if __name__ == "__main__":
    main()
