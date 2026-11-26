"""Benchmark: LTX video VAE decodes (multi-frame) -- native vs Monoload, memory and time.

The LTX video VAEs (ComfyUI v0.46.3, comfy/ldm/lightricks/vae):
  * the conv VAE (causal_video_autoencoder.VideoVAE, `comfy_has_chunked_io`):
    LTX-Video 0.9.x (version 0 / 1), 0.9.5+ / LTX-2 / LTX-2.3 / the LTX-2.5 "conv"
    VAE (version 2). Native splits every up block's output along T into chunks
    of <= get_max_chunk_size() bytes (32-128 MiB by total VRAM) and writes into
    the preallocated output buffer.
  * the diffusion decoder (na_diffusion_decoder.CausalDiffusionVAE, called
    "LTX 2.4" in sd.py; added by "Add support for LTX 2.5"): the official
    LTX-2.5 video VAE. Neighbourhood attention + SwiGLU, one x0 step on a
    pixel-sized noise.

Sources: --vae FILE (models/vae), --checkpoint FILE (built-in VAE), or
--random ltx2|ltx09|ltx25dd: the VAE built from its default config with random
weights (peak memory and time do not depend on weight values; for VAEs whose
weights are not on the machine).

Random latents (1, 128, T, H/32, W/32) with T = (frames - 1) / 8 + 1. Every
decode is seeded (torch.manual_seed(--seed)) so the decode noise of the
timestep-conditioned VAEs and the diffusion decoder is the same for every mode;
--compare prints max|d| of each mode against the first.

  --chunk-mib N   (native experiment) override the conv VAE's max chunk size.
  mode l2x[-w<MiB>]  (experiment, no Monoload routing involved) the decoder's own decode under layer 2's
                  OpChunking (conv row blocks, workspace MiB, default the global workspace), the output buffer
                  preallocated on the output device and passed in like native does: what layer 2 with the
                  output-buffer hook (PLAN H3) will measure for a chunked-io VAE.
  --profile       one decode of --profile-mode under torch.profiler (bench_vae.profile_decode).

Run inside the container, the GPU lock held:
  flock /srv/comfy/gpu.lock docker exec -w <repo> comfyui-latest python tests/bench_ltx.py \
      --vae LTX2_video_vae_bf16.safetensors --res 768x512,1280x704 --frames 97 --modes native
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_vae as bv  # noqa: E402  (sets COMFY_ARGS from PID 1, boots ComfyUI)

import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.ldm.lightricks.vae.causal_video_autoencoder as cva  # noqa: E402

GIB = 1024 ** 3


def random_vae(kind):
    """A comfy.sd.VAE of `kind` with random weights (std 0.02, norms 1, biases 0), built from the default config."""
    if kind == "ltx25dd":
        from comfy.ldm.lightricks.vae import na_diffusion_decoder as nd
        with torch.device("meta"):
            m = nd.CausalDiffusionVAE()
    elif kind in ("ltx2", "ltx09"):
        with torch.device("meta"):
            m = cva.VideoVAE(version=2 if kind == "ltx2" else 0)
    else:
        raise SystemExit("unknown --random kind " + kind)
    sd = {}
    g = torch.Generator().manual_seed(1234)
    for k, v in m.state_dict().items():
        if v.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            continue
        if k.endswith(".bias") or "scale_shift_table" in k or "mean-of-means" in k:
            t = torch.zeros(v.shape, dtype=torch.bfloat16)
        elif "norm" in k.split(".")[-2] or "std-of-means" in k or k.endswith("timestep_scale_multiplier"):
            t = torch.ones(v.shape, dtype=torch.bfloat16)
            if k.endswith("timestep_scale_multiplier"):
                t = t * 1000.0
        else:
            t = (torch.randn(v.shape, generator=g) * 0.02).to(torch.bfloat16)
        sd[k] = t
    del m
    v = comfy.sd.VAE(sd=sd)
    del sd
    return v, "random " + kind


def latent_for(vae, w, h, frames, seed):
    r = vae.spacial_compression_decode()
    if w % r or h % r:
        raise SystemExit("resolution {}x{} is not a multiple of {}".format(w, h, r))
    if (frames - 1) % 8:
        raise SystemExit("frames must be 8k+1 (got {})".format(frames))
    t = (frames - 1) // 8 + 1
    g = torch.Generator().manual_seed(seed)
    return torch.randn([1, vae.latent_channels, t, h // r, w // r], generator=g)


class _Shim:
    """Stands in for the VAE in bench_vae.decode_once: .decode runs `fn`, .process_output is the VAE's."""

    def __init__(self, vae, fn):
        self.vae, self.fn = vae, fn
        self.process_output = vae.process_output

    def decode(self, latent):
        return self.fn(self, latent)


def l2x_decode(ws):
    from monoload.vae_ops import OpChunking, OpStats

    def run(shim, latent):
        vae = shim.vae
        fsm = vae.first_stage_model
        z = latent.to(device=vae.device, dtype=vae.vae_dtype)
        stats = OpStats()
        if getattr(fsm, "comfy_has_chunked_io", False):
            out = torch.empty(fsm.decode_output_shape(z.shape), device=vae.output_device, dtype=vae.vae_output_dtype())
            with OpChunking(fsm, ws, stats):
                fsm.decode(z, output_buffer=out)
        else:
            with OpChunking(fsm, ws, stats):
                out = fsm.decode(z).to(device=vae.output_device, dtype=vae.vae_output_dtype())
        del z
        shim.process_output(out)
        run.stats = stats
        return out
    return run


def fmt(row):
    s = "{:>11}".format(row["status"])
    s += " {:7.1f}s".format(row.get("seconds", 0.0))
    for k in ("alloc_delta", "res_delta", "gtt_delta"):
        v = row.get(k)
        s += "  {} {}".format(k.split("_")[0], bv.gib(v))
    return s


def main():
    p = argparse.ArgumentParser(description="LTX video VAE decode benchmark")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--vae")
    src.add_argument("--checkpoint")
    src.add_argument("--random", choices=["ltx2", "ltx09", "ltx25dd"])
    p.add_argument("--res", default="768x512,1280x704")
    p.add_argument("--frames", default="97")
    p.add_argument("--modes", default="native")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1, help="decodes per (res, frames, mode); the table shows each")
    p.add_argument("--chunk-mib", type=int, default=0)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--sample-ms", type=float, default=5.0)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-mode", default="native")
    p.add_argument("--profile-top", type=int, default=25)
    a = p.parse_args()

    if a.chunk_mib:
        cva.get_max_chunk_size = lambda device: a.chunk_mib * 1024 * 1024
    if a.random:
        vae, source = random_vae(a.random)
    else:
        vae, source = bv.load_vae(argparse.Namespace(vae=a.vae, checkpoint=a.checkpoint))
    bv.env_header(vae, source, argparse.Namespace())
    fsm = vae.first_stage_model
    dec = getattr(fsm, "decoder", None)
    print("first_stage_model {} (chunked io {}), decoder {}, version/config: timestep_conditioning {}, causal {}, "
          "max chunk {} MiB".format(type(fsm).__name__, getattr(fsm, "comfy_has_chunked_io", False), type(dec).__name__,
                                    getattr(fsm, "timestep_conditioning", None), getattr(dec, "causal", None),
                                    cva.get_max_chunk_size(mm.get_torch_device()) // (1024 * 1024)), flush=True)
    mm.load_models_gpu([vae.patcher], force_full_load=True)
    print("weights loaded: {} GiB allocated".format(bv.gib(torch.cuda.memory_allocated() if bv.is_cuda() else 0)), flush=True)

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    res = [tuple(int(x) for x in r.split("x")) for r in a.res.split(",") if r.strip()]
    frames = [int(f) for f in a.frames.split(",") if f.strip()]
    summary = []
    for (w, h) in res:
        for f in frames:
            lat = latent_for(vae, w, h, f, a.seed)
            ref = None
            for mode in modes:
                target = vae
                if mode.startswith("l2x"):
                    import re
                    from monoload import vae as mvae
                    mw = re.search(r"-w(\d+)$", mode)
                    bv.configure("native")
                    target = _Shim(vae, l2x_decode(int(mw.group(1)) * 1024 * 1024 if mw else mvae.workspace()))
                else:
                    bv.configure(mode)
                for rep in range(a.repeat):
                    torch.manual_seed(a.seed)
                    row, raw, px = bv.decode_once(target, lat, a.compare, a.sample_ms / 1000.0)
                    line = "{}x{}x{} {:<14} #{} {}".format(w, h, f, mode, rep, fmt(row))
                    if a.compare and px is not None:
                        if ref is None:
                            ref = px
                        else:
                            line += "  max|d| vs {} {:.3g}".format(modes[0], (px - ref).abs().max().item())
                    print(line, flush=True)
                    summary.append(line)
                    del raw, px
                    gc.collect()
            if a.profile:
                torch.manual_seed(a.seed)
                bv.profile_decode(vae, lat, "{}x{}x{}".format(w, h, f), a.profile_top, a.profile_mode, False)
    print("\n== summary (delta over the value before each decode; GiB)")
    for line in summary:
        print(line)


if __name__ == "__main__":
    main()
