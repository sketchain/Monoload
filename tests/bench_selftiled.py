"""Native-peak measurement for the VAEs that tile / chunk on their own (G group of docs/PLAN-video.md):
MiniMax H3 video VAE (stock ViT3DDecoder, Sol NonCausalViT3DDecoder, int8), SeedVR2, RelayVSR Flash.

For every --res x --frames and every --modes entry (native = Monoload's VAE wrapper uninstalled,
monoload = installed, as the plugin runs it) one VAE.decode of a random latent, measured:
  device    torch max_memory_allocated / max_memory_reserved, absolute and as increase over the value
            before the decode (the VAE weights are loaded by a warm-up decode first, so the increase is
            the decode's own working memory plus the output buffer);
  host      GTT / VRAM peak (amdgpu sysfs sampler), cgroup memory.peak increase (pinned host memory
            counts here), SeedVR2's cache plan (offload to pinned host memory / compiled / on device);
  native    the VAE's own memory_used_decode estimate, the output buffer size, the weights size;
  fallback  native's OOM -> tiled fallback is aborted and reported (never counted as native);
  equality  monoload vs native: bitwise (sha256 of the output bytes).

GGUF VAE files: no loader in ComfyUI v0.46.3 or ComfyUI-GGUF loads one as a VAE (VAELoader lists no
.gguf, ComfyUI-GGUF only has UNet / CLIP loaders). --gguf reads the file with the gguf package,
dequantizes every tensor to fp16 and builds comfy.sd.VAE from that state dict: the decode is then the
fp16 VAE's (the GGUF only saves disk).

Run inside the ComfyUI container, in a separate process, under the GPU lock, after freeing the server:
  curl -s -X POST http://127.0.0.1:8190/free -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}'
  python tests/bench_selftiled.py --vae minimax_h3_video_vae_fp16.safetensors --res 1280x720,1920x1088 --frames 33,81,121
  python tests/bench_selftiled.py --gguf seedvr2_ema_vae_f16.gguf --res 1280x720,1920x1080 --frames 33,81
  python tests/bench_selftiled.py --vae relayvsr_flash_vae_x4_fp32.safetensors --relay --res 1920x1080 --frames 31,61
(--res is the decoded size; for --relay the low-quality input is --res / the upscale factor.)
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_vae as bv  # noqa: E402  (sets COMFY_ARGS from PID 1, imports ComfyUI)

import torch  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402
import folder_paths  # noqa: E402
import nodes  # noqa: E402
from monoload import vae as mvae  # noqa: E402

GIB = 1024 ** 3


def load_gguf_vae(name):
    import gguf
    import gguf.quants
    import numpy as np
    path = name if os.path.isfile(name) else folder_paths.get_full_path_or_raise("vae", name)
    reader = gguf.GGUFReader(path)
    sd, types = {}, {}
    for t in reader.tensors:
        shape = tuple(int(x) for x in reversed(t.shape.tolist()))
        qt = gguf.GGMLQuantizationType(t.tensor_type)
        types[qt.name] = types.get(qt.name, 0) + 1
        if qt in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.BF16):
            arr = np.array(t.data)
            ten = torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16) if qt == gguf.GGMLQuantizationType.BF16 else torch.from_numpy(arr)
        else:
            ten = torch.from_numpy(np.array(gguf.quants.dequantize(t.data, qt)))
        sd[t.name] = ten.reshape(shape).to(torch.float16 if ten.dtype != torch.float32 or ten.numel() > 64 else torch.float32)
    meta = {}
    for k, f in reader.fields.items():
        if k.startswith("general.") or k.startswith("comfy.") or k == "config":
            try:
                meta[k] = bytes(f.parts[f.data[0]]).decode("utf-8")
            except Exception:
                pass
    print("gguf: {} tensors, types {}".format(len(sd), types), flush=True)
    return comfy.sd.VAE(sd=sd, metadata=meta or None), "gguf(dequantized fp16) " + os.path.basename(path)


def stock_gguf_check(name):
    listed = name in folder_paths.get_filename_list("vae")
    gguf_vae_nodes = [k for k in nodes.NODE_CLASS_MAPPINGS if "gguf" in k.lower() and "vae" in k.lower()]
    print("gguf check: VAELoader lists {}: {}; GGUF VAE loader nodes: {}".format(name, listed, gguf_vae_nodes or "none"), flush=True)
    return listed, gguf_vae_nodes


def latent_for(vae, w, h, frames, relay, seed):
    """(latent on the VAE's device dtype-agnostic fp32 CPU, vae_options, real pixel frames)."""
    fsm = vae.first_stage_model
    g = torch.Generator().manual_seed(seed)
    if relay:
        import comfy.ldm.wan.model_vsr as vsr
        f = fsm.upscale_factor
        lw, lh = w // f, h // f
        kf = vsr.relay_keyframes(frames)
        size = vsr.relay_latent_size(lh, lw, fsm.latent_stride)
        lat = torch.randn([1, vae.latent_channels, len(kf), *size], generator=g)
        lq = torch.rand([frames, lh, lw, 3], generator=g).to(mm.intermediate_device())
        return lat, {"lq_images": lq, "detail": 1.0}, frames
    r = vae.spacial_compression_decode()
    t = vae.downscale_ratio[0](frames) if isinstance(vae.downscale_ratio, tuple) and callable(vae.downscale_ratio[0]) else 1
    t = max(1, t)
    real = vae.upscale_ratio[0](t) if isinstance(vae.upscale_ratio, tuple) and callable(vae.upscale_ratio[0]) else 1
    lat = torch.randn([1, vae.latent_channels, t, h // r, w // r], generator=g)
    return lat, {}, real


class SeedVRPlan:
    """Records SeedVR2's cache plans (offload to pinned host memory / compiled / on device)."""

    def __enter__(self):
        self.plans = []
        try:
            import comfy.ldm.seedvr.vae as sv
        except Exception:
            self.cls = None
            return self
        self.cls = sv.CausalMemoryCache
        self.orig = self.cls.__init__
        me = self

        def init(obj, *a, **kw):
            me.orig(obj, *a, **kw)
            me.plans.append("offload" if getattr(obj, "offload", False) else "compiled" if getattr(obj, "compiled", False) else "device")
        self.cls.__init__ = init
        return self

    def __exit__(self, *exc):
        if self.cls is not None:
            self.cls.__init__ = self.orig
        return False

    def summary(self):
        out = {}
        for p in self.plans:
            out[p] = out.get(p, 0) + 1
        return out


def rss():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class Sampler(bv.Sampler):
    """bench_vae's sampler plus this process's RSS (pinned host memory shows here; the cgroup is the
    whole LXC, other jobs' page cache moves it)."""

    def read(self):
        d = super().read()
        d["rss"] = rss()
        return d


def digest(t):
    # VAE.decode returns pixel_samples.movedim(1, -1) (not contiguous): copy to the host first, a device-side
    # .contiguous() would add an output-sized allocation to the measured peak
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def one(vae, lat, opts, mode, sample_ms):
    bv.configure(mode)
    lat_dev = lat.clone()
    gc_collect()
    bv.sync()
    alloc0, res0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    row = {"mode": mode}
    t0 = time.perf_counter()
    try:
        with Sampler(sample_ms / 1000.0) as smp, bv.UnloadCounter() as uc, SeedVRPlan() as plan:
            if mode == "native":
                with bv.NativeGuard():
                    out = vae.decode(lat_dev, vae_options=opts)
            else:
                out = vae.decode(lat_dev, vae_options=opts)
            bv.sync()
        row["time"] = time.perf_counter() - t0
        row["shape"] = list(out.shape)
        row["out_bytes"] = out.numel() * out.element_size()
        row["out_device"] = str(out.device)
        row["alloc_peak"] = torch.cuda.max_memory_allocated()
        row["res_peak"] = torch.cuda.max_memory_reserved()
        row["sha"] = digest(out)
        del out
    except bv.TiledFallback as e:
        row["error"] = "TILED_FALLBACK " + str(e)
        smp = plan = uc = None
    except torch.OutOfMemoryError as e:
        row["error"] = "OOM " + str(e).splitlines()[0][:160]
        smp = plan = uc = None
    row.setdefault("alloc_peak", torch.cuda.max_memory_allocated())
    row.setdefault("res_peak", torch.cuda.max_memory_reserved())
    row["alloc_inc"] = row["alloc_peak"] - alloc0
    row["res_inc"] = row["res_peak"] - res0
    if smp is not None:
        row["gtt_inc"] = None if smp.peak["gtt"] is None else smp.peak["gtt"] - smp.base["gtt"]
        row["cg_inc"] = None if smp.peak["cg"] is None else smp.peak["cg"] - smp.base["cg"]
        row["cg_peak_inc"] = None if smp.cg_peak_value is None or smp.base["cg"] is None else smp.cg_peak_value - smp.base["cg"]
        row["rss_inc"] = None if smp.peak["rss"] is None else smp.peak["rss"] - smp.base["rss"]
        row["unloads"] = uc.n
        row["seedvr_plan"] = plan.summary()
    gc_collect()
    return row


def gc_collect():
    import gc
    gc.collect()
    mm.soft_empty_cache()
    torch.cuda.empty_cache()


def g(n):
    return "  n/a" if n is None else "{:6.2f}".format(n / GIB)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", help="VAE file in models/vae (VAELoader)")
    p.add_argument("--vae-path", help="VAE safetensors by path (comfy.sd.VAE with the file's metadata), for files not in models/vae")
    p.add_argument("--gguf", help="GGUF VAE file (models/vae name or path), dequantized to fp16")
    p.add_argument("--res", default="1280x720", help="decoded WxH list")
    p.add_argument("--frames", default="33", help="pixel frame counts")
    p.add_argument("--modes", default="native,monoload")
    p.add_argument("--relay", action="store_true", help="RelayVSR Flash: random low-quality frames as lq_images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-ms", type=float, default=5)
    p.add_argument("--json", help="write the rows here")
    a = p.parse_args()

    print("COMFY_ARGS:", bv.COMFY_ARGS, "| torch", torch.__version__, "| device", mm.get_torch_device(),
          "| intermediate", mm.intermediate_device(), "| vram_state", mm.vram_state, flush=True)
    if a.gguf:
        stock_gguf_check(os.path.basename(a.gguf))
        vae, src = load_gguf_vae(a.gguf)
    elif a.vae_path:
        import comfy.utils
        sd, meta = comfy.utils.load_torch_file(a.vae_path, return_metadata=True)
        vae, src = comfy.sd.VAE(sd=sd, metadata=meta), "path " + os.path.basename(a.vae_path)
        vae.throw_exception_if_invalid()
    else:
        vae, src = nodes.VAELoader().load_vae(a.vae)[0], "VAELoader " + a.vae
    fsm = vae.first_stage_model
    wbytes = sum(t.numel() * t.element_size() for t in fsm.state_dict().values())
    print("VAE {} | class {} | dtype {} | weights {:.2f} GiB | chunked_io {} | handles_tiling {} | native reason {}".format(
        src, type(fsm).__name__, vae.vae_dtype, wbytes / GIB, getattr(fsm, "comfy_has_chunked_io", False),
        getattr(vae, "handles_tiling", False), "per decode"), flush=True)

    rows = []
    modes = [m for m in a.modes.split(",") if m]
    warmed = False
    for res in a.res.split(","):
        w, h = (int(x) for x in res.lower().split("x"))
        for fr in (int(x) for x in a.frames.split(",")):
            lat, opts, real = latent_for(vae, w, h, fr, a.relay, a.seed)
            est = vae.memory_used_decode(lat.shape, vae.vae_dtype)
            if not warmed:  # load the weights, so the measured decodes start with the VAE resident
                wl, wo, _ = latent_for(vae, 512, 512, 16, True, 1) if a.relay else latent_for(vae, 256, 256, 17, False, 1)
                bv.configure("native")
                vae.decode(wl, vae_options=wo)
                gc_collect()
                warmed = True
            print("\n== {}x{} frames {} (latent {}) | native estimate {:.2f} GiB | Monoload routing: {}".format(
                w, h, real, list(lat.shape), est / GIB, _native_reason(vae, lat) or "managed"), flush=True)
            shas = {}
            for m in modes:
                r = one(vae, lat, opts, m, a.sample_ms)
                r.update({"res": res, "frames": real, "latent": list(lat.shape), "estimate": est, "weights": wbytes})
                rows.append(r)
                shas[m] = r.get("sha")
                if "error" in r:
                    print("  {:9s} ERROR {} | reserved peak {} inc {}".format(m, r["error"], g(r["res_peak"]), g(r["res_inc"])), flush=True)
                    continue
                print("  {:9s} {:7.2f}s | alloc peak {} inc {} | reserved peak {} inc {} | out {} ({}) | GTT inc {} | RSS inc {} | cgroup inc {} | unloads {} | seedvr {} | sha {}".format(
                    m, r["time"], g(r["alloc_peak"]), g(r["alloc_inc"]), g(r["res_peak"]), g(r["res_inc"]), g(r["out_bytes"]), r["out_device"],
                    g(r.get("gtt_inc")), g(r.get("rss_inc")), g(r.get("cg_inc")), r.get("unloads"), r.get("seedvr_plan") or "-", r["sha"]), flush=True)
            if "native" in shas and "monoload" in shas and shas["native"] and shas["monoload"]:
                print("  bitwise native == monoload:", shas["native"] == shas["monoload"], flush=True)
            del lat, opts
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"vae": src, "weights": wbytes, "rows": rows}, f, indent=1, default=str)
    bv.configure("native")


def _native_reason(vae, lat):
    try:
        return mvae._native_reason(vae, lat)
    except Exception as e:
        return "? ({}: {})".format(type(e).__name__, e)


if __name__ == "__main__":
    main()
