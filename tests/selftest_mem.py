"""What a layer-1 adapter's first-use self-test reserves on the GPU, against bound.selftest_memory() (vae_engine):
the reserved / allocated growth, the bound's terms, and from the allocator's memory history the segments at the
reserved peak (each with the request that reserved it) and the live blocks there, grouped by allocating frame.

  python tests/selftest_mem.py --vae hunyuanvideo15_vae_fp16.safetensors --adapter vae_hunyuan15 --latent 1,32,21,30,52

Real machine only (needs a CUDA/HIP device). The adapter is registered in-process if vae.py does not list it."""
import argparse
import gc
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_vae  # noqa: E402  (bootstraps ComfyUI)
import torch  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import nodes  # noqa: E402
from monoload import vae as mvae, vae_engine as eng  # noqa: E402

MIB = 1 << 20


def fm(n):
    return "{:.1f} MiB".format(n / MIB)


def frame_of(ev):
    for fr in ev.get("frames") or []:
        f = fr.get("filename", "")
        if ("monoload" in f or "/comfy" in f) and "selftest_mem" not in f:
            return "{}:{} {}".format(os.path.basename(f), fr.get("line"), fr.get("name"))
    fr = (ev.get("frames") or [{}])[0]
    return "{}:{} {}".format(os.path.basename(fr.get("filename", "?")), fr.get("line"), fr.get("name"))


def bound_terms(b):
    """selftest_memory()'s terms, recomputed as vae_engine does."""
    n = b.selftest_size
    plan = eng.Plan(b, n, n, eng.SELFTEST_ROWS, eng.SELFTEST_WORKSPACE, 4, 0, 0, b.time_chunks(tuple(b.selftest_latent(n))))
    act = max(c * h * w * p for c, h, w, p in zip(plan.channels, plan.heights, plan.widths, plan.planes))
    act = max(act, max(b.prefix_out_channels(m) for _, m in b.prefix) * n * n * plan.planes[0]) * 4
    out = b.out_channels * plan.h_out * plan.w_out * (max(c.ot1 for c in plan.chunks) if plan.chunks else 1) * 4
    ret = b.decode_retained(n, n, 4)
    ref = b.activation_copies * act + 2 * eng.SELFTEST_REF_WORKSPACE + out + ret
    w = b.selftest_params() * 4
    print("bound terms: weights fp32 {}; reference = {} x act {} + 2 x ref workspace {} + out {} + retained {} = {}; "
          "stripe pass = plan estimate {} + out; margin {}".format(
              fm(w), b.activation_copies, fm(act), fm(eng.SELFTEST_REF_WORKSPACE), fm(out), fm(ret), fm(ref),
              fm(plan.estimate), fm(eng.SELFTEST_MARGIN)))
    print("  plan: levels channels {} heights {} widths {} planes {}; chunks {}".format(
        plan.channels, plan.heights, plan.widths, plan.planes, len(plan.chunks) if plan.chunks else 0))


def report(top):
    tr = torch.cuda.memory._snapshot()["device_traces"][torch.cuda.current_device()]
    free_act = "free_requested" if any(ev.get("action") == "free_requested" for ev in tr) else "free_completed"
    live, cur, apeak, apeak_i = {}, 0, 0, -1
    segs, scur, speak, speak_i = {}, 0, 0, -1
    for i, ev in enumerate(tr):
        a = ev["action"]
        if a == "alloc":
            live[ev["addr"]] = ev["size"]
            cur += ev["size"]
            if cur > apeak:
                apeak, apeak_i = cur, i
        elif a == free_act:
            s = live.pop(ev["addr"], None)
            if s is not None:
                cur -= s
        elif a in ("segment_alloc", "segment_map"):
            segs[ev["addr"]] = ev["size"]
            scur += ev["size"]
            if scur > speak:
                speak, speak_i = scur, i
        elif a in ("segment_free", "segment_unmap"):
            scur -= segs.pop(ev["addr"], 0)
    print("memory history: allocated peak +{} (event {}), reserved peak +{} (event {} of {})".format(
        fm(apeak), apeak_i, fm(speak), speak_i, len(tr)))
    for name, idx in (("allocated peak", apeak_i), ("reserved peak", speak_i)):
        live, segs, order = {}, {}, []
        last_alloc = {}
        for ev in tr[:idx + 1]:
            a = ev["action"]
            if a == "alloc":
                live[ev["addr"]] = ev
            elif a == free_act:
                live.pop(ev["addr"], None)
            elif a in ("segment_alloc", "segment_map"):
                segs[ev["addr"]] = ev
                order.append(ev["addr"])
            elif a in ("segment_free", "segment_unmap"):
                segs.pop(ev["addr"], None)
        groups = {}
        for ev in live.values():
            k = frame_of(ev)
            n, b = groups.get(k, (0, 0))
            groups[k] = (n + 1, b + ev["size"])
        print("-- at the {}: {} live ({} blocks), {} reserved in {} segments; live by frame:".format(
            name, fm(sum(e["size"] for e in live.values())), len(live), fm(sum(s["size"] for s in segs.values())), len(segs)))
        for k, (n, b) in sorted(groups.items(), key=lambda kv: -kv[1][1])[:top]:
            print("   {:>11} x{:<4} {}".format(fm(b), n, k))
        if name == "reserved peak":
            print("   segments (in order reserved; size, live in it, the request that reserved it):")
            for addr in order:
                s = segs.get(addr)
                if s is None:
                    continue
                inside = sum(e["size"] for a2, e in live.items() if addr <= a2 < addr + s["size"])
                print("   {:>11} live {:>11}  {}".format(fm(s["size"]), fm(inside), frame_of(s)))
        del live, segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--adapter", default="vae_hunyuan15")
    ap.add_argument("--latent", required=True, help="B,C,T,H,W of the decode the bound is matched for")
    ap.add_argument("--top", type=int, default=16)
    a = ap.parse_args()
    mod = importlib.import_module("monoload." + a.adapter)
    if mod not in mvae.STRIPE_ADAPTERS:
        mvae.STRIPE_ADAPTERS.append(mod)
    v = nodes.VAELoader().load_vae(a.vae)[0]
    mm.load_models_gpu([v.patcher])
    shape = tuple(int(x) for x in a.latent.split(","))
    lat = torch.zeros(shape)
    b, why = mod.match(v, lat, {})
    if b is None:
        sys.exit("not matched: {}".format(why))
    print("{}: key {}; selftest_memory {}".format(b.name, b.key, fm(b.selftest_memory())))
    bound_terms(b)
    eng._SELFTEST.pop(b.key, None)
    gc.collect()
    mm.soft_empty_cache(True)
    torch.cuda.synchronize()
    r0, a0 = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=2000000, stacks="python")
    try:
        ok, detail = eng.self_test(b, v)
        torch.cuda.synchronize()
        print("self-test: ok {} ({})".format(ok, detail))
        print("growth: reserved +{} allocated +{} against selftest_memory {}".format(
            fm(torch.cuda.max_memory_reserved() - r0), fm(torch.cuda.max_memory_allocated() - a0), fm(b.selftest_memory())))
        report(a.top)
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
