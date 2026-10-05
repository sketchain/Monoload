"""CT 700 diagnostic: what the first-use layer-1 self-test costs, apart from
what the first GPU work of a process costs (DESIGN §9.19).

In a fresh process, each step measured on its own (allocator cache emptied
before it; reserved / GTT peak increase, and what stays after the cache is
emptied again):

  warm-up    (--warmup) one bf16 and one fp32 matmul and 3x3 conv on small
             tensors: the one-time cost of the first GPU work (BLAS handles and
             workspaces, kernels loaded); in a ComfyUI server the sampler has
             paid it long before the VAE decode
  self-test  vae_engine.self_test alone (fp32 copy of the decoder, whole-image
             fp32 reference decode of a 24 x 24 latent, forced small stripes),
             of the layer-1 variant the decodes will use: with --budget the
             budget's choice (which can be another GroupNorm scheme than the
             default), chosen beforehand without a self-test; the shape probe
             of that choice is dropped again, so decode 1 runs it as any first
             decode with a budget does
  decode 1   the decode (default policy, or --budget); its self-test is cached
             by now, so this is the decode alone (each decode line says whether
             a self-test ran in it after all)
  decode 2   the same again

    docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/check_selftest_mem.py \\
        --vae flux2-vae.safetensors --res 3840x2160 --budget 3
    docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/check_selftest_mem.py \\
        --vae flux2-vae.safetensors --res 3840x2160 --budget 3 --warmup

Same launch arguments as tests/bench_vae.py ($COMFY_ARGS, else those of the
container's main process).
"""

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_vae as B  # noqa: E402  (ComfyUI environment, measuring helpers)
from bench_vae import gib, mm, mvae, torch  # noqa: E402
from monoload import vae_engine  # noqa: E402


def measure(label, fn):
    gc.collect()
    B.sync()
    torch.cuda.empty_cache()
    base_res = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    with B.Sampler(0.01) as smp:
        out = fn()
        B.sync()
    peak = torch.cuda.max_memory_reserved() - base_res
    gc.collect()
    torch.cuda.empty_cache()
    stays = torch.cuda.memory_reserved() - base_res
    g0, gp = smp.base["gtt"], smp.peak["gtt"]
    gtt_after = B.read_sum(smp.gtt) if g0 is not None else None
    print("{:10s} reserved peak +{} | stays reserved after empty_cache +{} | GTT peak +{} | GTT after +{} GiB".format(
        label, gib(peak).strip(), gib(stays).strip(), gib(gp - g0).strip() if gp is not None else "n/a",
        gib(gtt_after - g0).strip() if gtt_after is not None else "n/a"), flush=True)
    return out


def warmup():
    dev = torch.device("cuda")
    for dt in (torch.bfloat16, torch.float32):
        a = torch.randn(1024, 1024, device=dev, dtype=dt)
        (a @ a).sum().item()
        x = torch.randn(1, 64, 128, 128, device=dev, dtype=dt)
        w = torch.randn(64, 64, 3, 3, device=dev, dtype=dt)
        torch.nn.functional.conv2d(x, w, padding=1).sum().item()


def selected_bound(vae, lat, budget):
    """(bound, why) of the layer-1 variant a decode of `lat` uses with the
    current settings: with a budget (set beforehand with mvae.set_budget) the
    budget's choice, made without running a self-test; the shape probe it ran
    is dropped (decode 1 runs it again, as a first decode with a budget does).
    bound None: layer 1 is not used (why says so)."""
    bound, why = mvae._select_layer1(vae, lat, {})
    if bound is None or not budget:
        return bound, why
    d = mvae.choose_budget(vae, lat, {}, budget, selftest=lambda b: (True, "not run while choosing"))
    mvae._PROBES.pop(vae.first_stage_model, None)
    if d["layer"] != 1:
        return None, "the budget chooses layer 2"
    return d["bound"], why


def selftest_in_decode(m):
    """Bytes of the self-test bound if the last decode ran a self-test first, else 0."""
    return (m.get("estimate") or {}).get("selftest") or 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="checkpoint with a built-in VAE (models/checkpoints)")
    src.add_argument("--vae", help="VAE file (models/vae)")
    p.add_argument("--res", default="3840x2160", help="output resolution WxH (random latent)")
    p.add_argument("--budget", type=float, default=0.0, help="MONOLOAD_VAE_BUDGET for the decodes in GiB (0: the default policy)")
    p.add_argument("--warmup", action="store_true", help="do the first GPU work before measuring the self-test")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if not B.is_cuda():
        raise SystemExit("needs the GPU")
    mvae.install()
    B.configure("monoload")
    vae, what = B.load_vae(a)
    w, h = (int(x) for x in a.res.split("x"))
    lat = B.random_latent(vae, w, h, a.seed, 1.0)
    mm.load_models_gpu([vae.patcher])
    print("VAE: {} ({}), latent {}; budget {}; warm-up {}".format(what, type(vae.first_stage_model).__name__, list(lat.shape),
                                                                  "{} GiB".format(a.budget) if a.budget else "none", a.warmup), flush=True)
    if a.warmup:
        measure("warm-up", warmup)
    budget = int(a.budget * (1 << 30)) if a.budget else 0
    if budget:
        mvae.set_budget(budget)
    bound, why = selected_bound(vae, lat, budget)
    if bound is None:
        raise SystemExit("layer 1 does not apply: " + why)
    vae_engine._SELFTEST.clear()
    mm.load_models_gpu([vae.patcher], memory_required=bound.selftest_memory())
    ok, detail = measure("self-test", lambda: vae_engine.self_test(bound, vae))
    print("           {} ({}): {}; passed to load_models_gpu for it: {} GiB".format(bound.name, "ok" if ok else "FAILED", detail,
                                                                                    gib(bound.selftest_memory()).strip()), flush=True)
    for label in ("decode 1", "decode 2"):
        measure(label, lambda: vae.decode(lat))
        m = mvae.last_decode()
        st = selftest_in_decode(m)
        print("           {}, estimate {} GiB; {}".format(
            m.get("adapter") or m.get("strategy"), gib((m.get("estimate") or {}).get("total")).strip(),
            "a self-test ran in this decode too (up to {} GiB): NOT the decode alone".format(gib(st).strip()) if st
            else "no self-test in this decode"), flush=True)
    print("\nexpected (Flux 2 / SDXL VAE, README §9.13): self-test reserved peak ~0.40 GiB (~0.47 without --warmup: the first GPU work "
          "of the process happens inside it, ~0.07 stays); decode 1 == decode 2 (4K budget 3G: ~2.44), after each only the output "
          "stays (4K: ~0.09 GiB).")


if __name__ == "__main__":
    main()
