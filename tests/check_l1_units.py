"""CT 700 diagnosis of a layer-1 (stripe) decode, call by call (DESIGN §9.26).

Wraps every prefix module and stripe unit of the bound adapter for the
duration of each decode and records, for each call: the rows it got, what was
allocated before it (relative to the start of the decode), its own peak above
that (max_memory_allocated, reset before the call), the adapter's model of
that peak (unit_peak / prefix_peak minus the input's storage, which the caller
holds), and whether the caching allocator reserved new segments during it
(their sizes, from memory_snapshot). The self-test and the first GPU work run
first, in a warm-up decode that is not measured.

    docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/check_l1_units.py \\
        --vae wan2.2_vae.safetensors --res 3840x2160 --rows 64,0,256 --json /opt/ComfyUI/output/l1_units.json

--rows 0 is the default stripe policy. The measurement synchronizes the device
around every call, so the times are not representative. Same launch arguments
as tests/bench_vae.py ($COMFY_ARGS, else those of the container's main process).

--sim MODEL prints the same table for tests/alloc_sim.py's replay of the decode
(no GPU; MODEL as alloc_sim's --model, e.g. wan22), to set against the GPU's:

    python tests/check_l1_units.py --sim wan22 --res 3840x2160 --rows 64,0,256
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SIM = "--sim" in sys.argv
if SIM:
    import common  # noqa: E402,F401  (ComfyUI environment, CPU)
    import alloc_sim  # noqa: E402
    import torch  # noqa: E402
    from monoload import vae as mvae, vae_engine as eng  # noqa: E402
else:
    import bench_vae as B  # noqa: E402  (ComfyUI environment, measuring helpers)
    from bench_vae import mm, mvae, torch, vae_engine as eng  # noqa: E402

MIB = 1 << 20


def mib(n):
    return "{:8.1f}".format(n / MIB) if n is not None else "     n/a"


class Meter:
    """Patches StripeAdapter.run: while a decode runs, its prefix modules and units are wrapped in probes."""

    def __init__(self):
        self.runs = []
        self._orig = None

    def __enter__(self):
        self._orig = eng.StripeAdapter.run
        meter = self

        def run(bound, vae, samples_in, plan, budget_ws, stats):
            rec = meter.begin(bound, plan, budget_ws)
            try:
                return meter._orig(bound, vae, samples_in, plan, budget_ws, stats)
            finally:
                meter.end(bound, rec)

        eng.StripeAdapter.run = run
        return self

    def __exit__(self, *exc):
        eng.StripeAdapter.run = self._orig

    def begin(self, bound, plan, ws):
        torch.cuda.synchronize()
        ws = getattr(plan, "workspace", None) or ws
        rec = {"adapter": bound.name, "rows": [o1 - o0 for o0, o1 in plan.stripes], "workspace": ws,
               "estimate": plan.estimate, "arena": plan.arena, "live_peak": plan.live_peak,
               "out_segment": getattr(plan, "out_segment", 0),
               "base_alloc": torch.cuda.memory_allocated(), "base_res": torch.cuda.memory_reserved(),
               "segments": None, "calls": [],
               "saved": (list(bound.prefix), [u.module for u in bound.units])}
        hdim = bound.hdim
        bound.prefix = [(name, _Probe(self, rec, bound, name, m, None, m, hdim, ws)) for name, m in bound.prefix]
        for u in bound.units:
            u.module = _Probe(self, rec, bound, u.name, u.module, u, None, hdim, ws)
        self.runs.append(rec)
        return rec

    def end(self, bound, rec):
        prefix, mods = rec.pop("saved")
        bound.prefix = prefix
        for u, m in zip(bound.units, mods):
            u.module = m
        rec.pop("segments")


class _Probe:
    __slots__ = ("meter", "rec", "bound", "name", "fn", "unit", "pmod", "hdim", "ws")

    def __init__(self, meter, rec, bound, name, fn, unit, pmod, hdim, ws):
        self.meter, self.rec, self.bound, self.name, self.fn = meter, rec, bound, name, fn
        self.unit, self.pmod, self.hdim, self.ws = unit, pmod, hdim, ws

    def __call__(self, x):
        rec, b = self.rec, self.bound
        torch.cuda.synchronize()
        if rec["segments"] is None:   # the output's segment and the arena are there by the first call
            rec["segments"] = {seg["address"] for seg in torch.cuda.memory_snapshot()}
        a0 = torch.cuda.memory_allocated()
        r0 = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        y = self.fn(x)
        torch.cuda.synchronize()
        pk = torch.cuda.max_memory_allocated()
        rp = torch.cuda.max_memory_reserved()
        s = x.untyped_storage().nbytes()
        r, w, e = int(x.shape[self.hdim]), int(x.shape[-1]), x.element_size()
        if self.unit is not None:
            model = b.unit_peak(self.unit, r, w, e, self.ws, s) - s
            largest = b.unit_largest(self.unit, r, w, e, self.ws)
        else:
            model = b.prefix_peak(self.pmod, r, w, e, self.ws, s) - s
            largest = b.prefix_largest(self.pmod, r, w, e, self.ws)
        c = {"name": self.name, "rows": r, "width": w, "contiguous": bool(x.is_contiguous()), "in_storage": s,
             "out_storage": y.untyped_storage().nbytes(), "before": a0 - rec["base_alloc"], "real": pk - a0, "model": model,
             "model_largest": largest, "res_before": r0 - rec["base_res"], "res_peak": rp - rec["base_res"], "new_segments": []}
        if rp > r0:
            for seg in torch.cuda.memory_snapshot():
                if seg["address"] not in rec["segments"]:
                    rec["segments"].add(seg["address"])
                    blocks = [blk["size"] for blk in seg.get("blocks", []) if blk.get("state") == "active_allocated"]
                    c["new_segments"].append({"size": seg["total_size"], "allocated_now": seg["allocated_size"],
                                              "largest_block_now": max(blocks) if blocks else 0})
        rec["calls"].append(c)
        return y


def report(rec, top):
    calls = rec["calls"]
    print("  stripes {} (rows {}), workspace {} MiB; plan: live peak {} MiB, arena {} MiB, output segment {} MiB, estimate {} MiB".format(
        len(rec["rows"]), sorted(set(rec["rows"])), rec["workspace"] // MIB, mib(rec["live_peak"]).strip(), mib(rec["arena"]).strip(),
        mib(rec["out_segment"]).strip(), mib(rec["estimate"]).strip()))
    if not calls:
        print("  no calls recorded")
        return
    peak = max(calls, key=lambda c: c["before"] + c["real"])
    res = max(c["res_peak"] for c in calls)
    print("  measured (inside the calls): alloc peak {} MiB above the decode's start, reserved peak {} MiB; "
          "model: output segment + live peak = {} MiB, estimate {} MiB".format(
              mib(peak["before"] + peak["real"]).strip(), mib(res).strip(), mib(rec["out_segment"] + rec["live_peak"]).strip(),
              mib(rec["estimate"]).strip()))
    print("  per module / unit (MiB; real = its own peak above what was allocated before it, model = the adapter's count of the same;"
          " before = allocated before it, relative to the decode's start):")
    print("    {:34s} {:>5s} {:>9s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>5s}".format(
        "name", "calls", "rows", "real", "model", "excess", "before", "in", "resv+", "segs"))
    order = []
    for c in calls:
        if c["name"] not in order:
            order.append(c["name"])
    for name in order:
        cs = [c for c in calls if c["name"] == name]
        worst = max(cs, key=lambda c: c["real"] - c["model"])
        grew = sum(len(c["new_segments"]) for c in cs)
        resv = sum(seg["size"] for c in cs for seg in c["new_segments"])
        rows = sorted({c["rows"] for c in cs})
        print("    {:34s} {:5d} {:>9s} {} {} {} {} {} {} {:5d}".format(
            name[:34], len(cs), "{}-{}".format(rows[0], rows[-1]) if len(rows) > 1 else str(rows[0]), mib(worst["real"]), mib(worst["model"]),
            mib(worst["real"] - worst["model"]), mib(max(c["before"] for c in cs)), mib(worst["in_storage"]), mib(resv), grew))
    print("  the call at the alloc peak: {} on {} rows x {} (input {}contiguous), before {} MiB, real {} MiB, model {} MiB".format(
        peak["name"], peak["rows"], peak["width"], "" if peak["contiguous"] else "non-", mib(peak["before"]).strip(),
        mib(peak["real"]).strip(), mib(peak["model"]).strip()))
    grows = [c for c in calls if c["new_segments"]]
    print("  calls during which the allocator reserved new segments: {} (largest first, up to {}):".format(len(grows), top))
    for c in sorted(grows, key=lambda c: -sum(s["size"] for s in c["new_segments"]))[:top]:
        print("    {:34s} rows {:4d}: reserved {} -> {} MiB, new segments {}".format(
            c["name"][:34], c["rows"], mib(c["res_before"]).strip(), mib(c["res_peak"]).strip(),
            ", ".join("{} MiB (allocated after the call {}, largest block {})".format(
                mib(s["size"]).strip(), mib(s["allocated_now"]).strip(), mib(s["largest_block_now"]).strip()) for s in c["new_segments"])))


class _SimCuda:
    """torch.cuda's memory functions answered by alloc_sim's allocator model (the tracer frees dead storages on poll)."""
    NAMES = ("synchronize", "memory_allocated", "memory_reserved", "reset_peak_memory_stats", "max_memory_allocated",
             "max_memory_reserved", "memory_snapshot")

    def __init__(self):
        self.sim = self.tracer = None
        self._saved = {}

    def __enter__(self):
        orig = alloc_sim.make_tracer

        def make_tracer(sim):
            self.sim, self.tracer = sim, orig(sim)
            return self.tracer
        self._saved["make_tracer"] = (alloc_sim, orig)
        alloc_sim.make_tracer = make_tracer
        for n in self.NAMES:
            self._saved[n] = (torch.cuda, getattr(torch.cuda, n))
            setattr(torch.cuda, n, getattr(self, n))
        return self

    def __exit__(self, *exc):
        for n, (mod, f) in self._saved.items():
            setattr(mod, n, f)

    def _poll(self):
        if self.tracer is not None:
            self.tracer.poll()

    def synchronize(self, *a):
        self._poll()

    def memory_allocated(self, *a):
        self._poll()
        return self.sim.allocated if self.sim else 0

    def memory_reserved(self, *a):
        return self.sim.reserved if self.sim else 0

    def reset_peak_memory_stats(self, *a):
        self._poll()
        if self.sim:
            self.sim.reset_peak()

    def max_memory_allocated(self, *a):
        self._poll()
        return self.sim.peak_allocated if self.sim else 0

    def max_memory_reserved(self, *a):
        return self.sim.peak_reserved if self.sim else 0

    def memory_snapshot(self):
        out = []
        for head in (self.sim.heads if self.sim else []):
            blocks, b = [], head
            while b is not None:
                blocks.append({"size": b.size, "state": "inactive" if b.free else "active_allocated"})
                b = b.next
            out.append({"address": head.addr, "total_size": sum(x["size"] for x in blocks),
                        "allocated_size": sum(x["size"] for x in blocks if x["state"] != "inactive"), "blocks": blocks})
        return out


def main_sim(a):
    w, h = (int(x) for x in a.res.split("x"))
    dtype = "fp32" if a.fp32 else "bf16"
    print("tests/alloc_sim.py: model {}, {}, {} (the allocator model; GPU kernels' own buffers added by hand)".format(a.sim, a.res, dtype))
    out = []
    for rows in [int(x) for x in a.rows.split(",")]:
        with _SimCuda(), Meter() as meter:
            info = alloc_sim.decode_trace(w, h, dtype, rows or None, model=a.sim)
        print("\n=== rows {}: {} stripes of {} rows ===".format(rows or "default", info["stripes"], info["rows"]))
        for rec in meter.runs:
            report(rec, a.top)
            out.append(dict(rec, rows_arg=rows, res=a.res, dtype=dtype, sim=a.sim))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f)
        print("\nresults written to {}".format(a.json))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=not SIM)
    src.add_argument("--checkpoint", help="checkpoint with a built-in VAE (models/checkpoints)")
    src.add_argument("--vae", help="VAE file (models/vae)")
    p.add_argument("--res", default="3840x2160", help="output resolution WxH (random latent)")
    p.add_argument("--rows", default="0", help="comma-separated stripe heights to measure (0 = the default policy)")
    p.add_argument("--fp32", action="store_true", help="load the VAE in fp32 (as bench_vae.py --fp32-ref)")
    p.add_argument("--top", type=int, default=12, help="segment-growth calls to list")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="also write every call to this JSON file")
    p.add_argument("--sim", metavar="MODEL", help="no GPU: the same table from tests/alloc_sim.py's replay (qwen / wan22 / hy21 / sdxl / flux / flux2)")
    a = p.parse_args()
    if a.sim:
        return main_sim(a)
    if not B.is_cuda():
        raise SystemExit("needs a CUDA / HIP device")

    mvae.install()
    B.configure("monoload")
    vae = B.load_vae_fp32(a) if a.fp32 else B.load_vae(a)[0]
    mm.load_models_gpu([vae.patcher])
    w, h = (int(x) for x in a.res.split("x"))
    lat = B.random_latent(vae, w, h, a.seed, 1.0)
    print("VAE {} ({}, {}), latent {}; env PYTORCH_*ALLOC_CONF={} TORCH_BLAS_PREFER_HIPBLASLT={}".format(
        a.vae or a.checkpoint, type(vae.first_stage_model).__name__, vae.vae_dtype, list(lat.shape),
        os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_HIP_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT")), flush=True)
    row, _, _ = B.decode_once(vae, lat, False, 0.02)
    print("warm-up decode (self-test, first GPU work; not measured): {}, {:.2f} s".format(row["status"], row["seconds"]), flush=True)

    out = []
    for rows in [int(x) for x in a.rows.split(",")]:
        mvae.set_stripe_rows(rows or None)
        with Meter() as meter:
            row, _, _ = B.decode_once(vae, lat, False, 0.02)
        m = mvae.last_decode()
        print("\n=== rows {} -> {}: {} ===".format(rows or "default", row["status"], B.mono_line(m) if m else "?"), flush=True)
        if m.get("strategy") != "layer1" or not meter.runs:
            print("  not a layer-1 decode: nothing to meter")
            continue
        for rec in meter.runs:
            report(rec, a.top)
            out.append(dict(rec, rows_arg=rows, res=a.res, dtype=str(vae.vae_dtype)))
    mvae.set_stripe_rows(None)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f)
        print("\nresults written to {}".format(a.json))


if __name__ == "__main__":
    main()
