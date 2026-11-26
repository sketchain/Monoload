"""Real-machine check: monoload/vae_ltx.py's chunked LTX-2.5 diffusion decoder against the native decoder.

For each latent size: the decoder (an fp32 copy, and the loaded bf16 model) decodes the same latent with the same
seeded noise natively and under vae_ltx.chunking at a few workspaces (a tiny one splits every neighbourhood attention
into one-frame query blocks and every SwiGLU into small token chunks). Prints max|chunked - native| /
max(1, max|native|) (fp32 must be <= vae_engine.SELFTEST_TOL, 1e-4), what was split, and the peak memory of each run
over the weights (torch.cuda.max_memory_allocated) next to vae_ltx.plan()'s estimate.

  python tests/check_ltx_dd.py --vae ltx-2.5-video-vae-bf16.safetensors        (or --random)
      [--sizes 256x256x25,384x256x49] [--workspaces 1,64,0] (MiB; 0 = whole, no split) [--no-fp32] [--no-bf16]
"""

import argparse
import copy
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_vae as bv  # noqa: E402  (boots ComfyUI)

import torch  # noqa: E402
import comfy.model_management as mm  # noqa: E402

from monoload import vae_ltx  # noqa: E402
from monoload.vae_engine import SELFTEST_TOL  # noqa: E402
from monoload.vae_ops import MIB, OpStats  # noqa: E402

GIB = 1024 ** 3


def decode(fsm, z, ws=None, stats=None):
    """fsm.decode(z) (fixed generator inside), natively or under chunking(ws)."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        if ws is None:
            out = fsm.decode(z)
        else:
            with vae_ltx.chunking(fsm, ws, stats):
                out = fsm.decode(z)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    return out.float().cpu(), peak


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--vae")
    src.add_argument("--random", action="store_true")
    p.add_argument("--sizes", default="256x256x25,384x256x49")
    p.add_argument("--workspaces", default="1,64,0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-fp32", action="store_true")
    p.add_argument("--no-bf16", action="store_true")
    a = p.parse_args()

    if a.random:
        import bench_ltx
        vae, source = bench_ltx.random_vae("ltx25dd")
    else:
        vae, source = bv.load_vae(argparse.Namespace(vae=a.vae, checkpoint=None))
    fsm = vae.first_stage_model
    if vae_ltx.kind(fsm) != "diffusion":
        raise SystemExit("not an LTX-2.5 diffusion VAE: {}".format(type(fsm).__name__))
    mm.load_models_gpu([vae.patcher], force_full_load=True)
    dev = vae.device
    print("source {}; device {}; vae dtype {}".format(source, dev, vae.vae_dtype), flush=True)
    models = []
    if not a.no_fp32:
        f32 = copy.deepcopy(fsm).to(device=dev, dtype=torch.float32)
        models.append(("fp32", f32, torch.float32))
    if not a.no_bf16:
        models.append((str(vae.vae_dtype).replace("torch.", ""), fsm, vae.vae_dtype))

    wss = [int(x) for x in a.workspaces.split(",") if x.strip()]
    failed = False
    for size in [s for s in a.sizes.split(",") if s.strip()]:
        w, h, f = (int(x) for x in size.split("x"))
        g = torch.Generator().manual_seed(a.seed)
        lat = torch.randn([1, 128, (f - 1) // 8 + 1, h // 32, w // 32], generator=g)
        for label, m, dt in models:
            z = lat.to(device=dev, dtype=dt)
            ref, peak0 = decode(m, z)
            scale = max(1.0, ref.abs().max().item())
            print("{} {:<9} native               peak {:6.2f} GiB  out {}".format(size, label, peak0 / GIB, tuple(ref.shape)), flush=True)
            for ws_mib in wss:
                ws = ws_mib * MIB if ws_mib else 1 << 62
                st = OpStats()
                out, peak = decode(m, z, ws, st)
                err = (out - ref).abs().max().item() / scale
                est = vae_ltx.plan(m, tuple(z.shape), z.element_size(), ws)
                ok = bool(torch.isfinite(out).all()) and (label != "fp32" or err <= SELFTEST_TOL)
                failed |= not ok
                print("{} {:<9} ws {:>5} MiB  {}  err {:.3g}  peak {:6.2f} GiB  est {:6.2f} GiB ({})  "
                      "na split {}/{} slabs {} block_min {}  mlp split {}".format(
                          size, label, ws_mib or "whole", "ok  " if ok else "FAIL", err, peak / GIB, est["total"] / GIB,
                          est["phase"], getattr(st, "ltx_na_split", 0), getattr(st, "ltx_na_calls", 0),
                          getattr(st, "ltx_na_slabs", 0), getattr(st, "ltx_na_block_min", None),
                          getattr(st, "ltx_mlp_split", 0)), flush=True)
                if est["total"] < peak:
                    print("   !! estimate below the measured peak", flush=True)
                del out
            del ref, z
    print("RESULT", "FAIL" if failed else "PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
