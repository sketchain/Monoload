"""CT 700 check: the scratch buffer of F.normalize(x, dim=1) on ROCm (DESIGN §9.28).

F.normalize -> Tensor.norm -> torch.linalg.vector_norm -> the generic reduction
kernel (aten/src/ATen/native/cuda/Reduce.cuh, gpu_reduce_kernel). When its
launch configuration (setReduceConfig) splits each output's reduction over
several thread blocks ("global reduce"), it allocates a scratch buffer from the
caching allocator for the duration of the kernel:

    4 B x outputs x ctas_per_output x block.x x output_vec_size

and on ROCm ctas_per_output is raised to at least 64. For a reduction over the
channels of an NC[T]HW tensor this is 32768 B per position on gfx1151 (block.x
32, vec 4) whatever the channel count -- 1.23 GiB for 40320 positions.

This script measures one F.normalize(x, dim=1) call per configuration -- bf16 /
fp32, channels 384 / 512 / 1024, 5D [1, C, 1, H, W] and 4D [1, C, H, W],
contiguous and a row slice of a taller tensor (the stripes' inputs), a sweep of
positions plus the sizes the Wan 2.2 decode uses -- and prints the allocation
above the call's inputs and output (max_memory_allocated), the segments it made
the allocator reserve (cache emptied before each call), and the buffer the
setReduceConfig model of this file predicts from the device properties. Then
the same for the grouped candidate (the norm over channel groups of at most 510
in fp32, combined: no global reduce, DESIGN §9.28): its scratch, its difference
from F.normalize and from an fp64 reference, and the time of both.

    docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/check_norm_reduce.py \\
        --json /opt/ComfyUI/output/norm_reduce.json 2>&1 | tee norm_reduce.txt

--predict prints the model's table without a GPU (--mp / --threads-per-mp / --warp
for the device). Plain PyTorch, no ComfyUI, no Monoload.
"""

import argparse
import json
import math
import sys
import time

import torch
import torch.nn.functional as F

MIB = 1 << 20
NOISE = 16 * MIB   # small allocations (norm results, clamp, small-pool rounding: <= 3 MiB measured) are below this; the
                   # smallest scratch the sweep can trigger is 7680 positions x 32 KiB = 240 MiB
ROWS_SWEEP = list(range(8, 86, 2))          # x 960 columns: 7680 .. 80640 positions
WAN22 = [  # (label, rows, cols, channels) of the Wan 2.2 calls in check_l1_units (4K / 2688, layer 1, prefix)
    ("4K prefix middle", 135, 240, 1024), ("4K r128 upsamples.2", 42, 960, 1024), ("4K r128 upsamples.2 last", 53, 960, 1024),
    ("4K r64 upsamples.2", 37, 960, 1024), ("4K r256 upsamples.2", 80, 960, 1024), ("4K r128 upsamples.1", 40, 480, 1024),
    ("2688 upsamples.2", 52, 672, 1024), ("2688 prefix middle", 96, 168, 1024), ("4K r128 upsamples.3", 78, 1920, 512),
    ("Wan 2.1 4K prefix", 135, 240, 384), ("Wan 2.1 4K r128 H/4", 40, 960, 384),
]


# ---------------------------------------------------------------------------
# the launch configuration of gpu_reduce_kernel (Reduce.cuh at 3f8a4a1862), for
# a reduction over dim 1 of an N C [T] H W tensor whose other dims coalesce into
# one output dim of `n` positions (contiguous, or a row slice: both coalesce)
# ---------------------------------------------------------------------------

def last_pow2(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return max(1, p)


def predict(n, c, c_stride, base_elems, mp, threads_per_mp, warp, rocm=True):
    """(scratch bytes, details) of one vector_norm over the c channels (stride c_stride elements) for n output
    positions, the input's first element base_elems elements into an aligned allocation."""
    vec = 4                                   # get_output_vec_size: halved until it divides all of these
    for v in (base_elems, n, c_stride):
        while v % vec:
            vec //= 2
    max_threads = 512 // vec                  # mnt_wrapper MAX_NUM_THREADS / output_vec_size
    dim0, dim1 = n // vec, c                  # "vectorize along output": block.x over outputs, block.y over channels
    d0 = last_pow2(dim0) if dim0 < max_threads else max_threads
    d1 = last_pow2(dim1) if dim1 < max_threads else max_threads
    bw = min(d0, warp)
    bh = min(d1, max_threads // bw)
    bw = min(d0, max_threads // bh)
    threads = bw * bh
    step_output, step_input = bw, 1           # output_mult[0] = split_output(block_width)
    vpt = c
    split = vpt >= min(bh * 16, 256)          # split_across_warps
    if split:
        step_input *= bh
    else:
        step_output *= bh
    vpt = -(-c // step_input)
    grid = -(-(n // vec) // step_output)
    target = mp * (threads_per_mp // threads)
    ctas = 1
    if split and vpt >= 256 and grid <= target:
        c1, c2, c3 = -(-target // grid), -(-vpt // (128 if rocm else 16)), -(-vpt // 256)
        ctas = min(max(c1, c3), c2)           # std::clamp(c1, c3, c2)
        if ctas > 1 and rocm:
            ctas = max(ctas, 64)
    size = 4 * n * ctas * bw * vec if ctas > 1 else 0
    return size, {"vec": vec, "block": (bw, bh), "values_per_thread": vpt, "grid": grid, "target_grid": target, "ctas_per_output": ctas}


# ---------------------------------------------------------------------------
# the candidate: the same normalization without the global reduce
# ---------------------------------------------------------------------------

GROUP = 510   # channels per group: values_per_thread <= ceil(510 / block_height >= 2) = 255 < 256: never a global reduce


def group_size(c, limit=GROUP):
    if c <= limit:
        return c
    return max(d for d in range(1, limit + 1) if c % d == 0)


def grouped_normalize(x, dim=1, eps=1e-12):
    """F.normalize(x, p=2, dim=1, eps): the norm as the fp32 norms of channel groups of <= GROUP channels combined in
    fp32, cast to x's dtype like torch.norm's result; then the same clamp_min / expand / divide as F.normalize."""
    c = x.shape[dim]
    g = group_size(c)
    if g == c:
        denom = x.norm(2, dim, keepdim=True)
    else:
        shape = list(x.shape)
        shape[dim:dim + 1] = [c // g, g]
        part = torch.linalg.vector_norm(x.view(shape), 2, dim + 1, keepdim=False, dtype=torch.float32)
        denom = torch.linalg.vector_norm(part, 2, dim, keepdim=True).to(x.dtype)
    return x / denom.clamp_min(eps).expand_as(x)


# ---------------------------------------------------------------------------
# measuring
# ---------------------------------------------------------------------------

def make(c, rows, cols, dtype, nd, sliced, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, c, 1, rows + 8 if sliced else rows, cols) if nd == 5 else (1, c, rows + 8 if sliced else rows, cols)
    base = torch.randn(shape, generator=g, device="cuda", dtype=torch.float32).to(dtype)
    if sliced:   # rows 4 .. 4 + rows of a taller tensor, as a stripe's input (offset 4 x cols elements)
        return base.narrow(nd - 2, 4, rows), base
    return base, base


def measure(fn, x):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    a0, r0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    s0 = torch.cuda.memory_stats().get("segment.all.current", 0)
    torch.cuda.reset_peak_memory_stats()
    y = fn(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - a0     # above the inputs, the output included
    res = torch.cuda.memory_reserved() - r0
    segs = torch.cuda.memory_stats().get("segment.all.current", 0) - s0
    return y, peak, res, segs


def timed(fn, x, n=5):
    fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def cases(quick=False):
    out = []
    for label, rows, cols, c in WAN22:
        out.append((label, rows, cols, c))
    for c in (384, 512, 1024):
        for r in ROWS_SWEEP[::4] if quick else ROWS_SWEEP:
            out.append(("sweep", r, 960, c))
        out.append(("sweep, positions not a multiple of 4", 41, 239, c))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", help="also write every row to this JSON file")
    p.add_argument("--quick", action="store_true", help="every fourth row count of the sweep")
    p.add_argument("--predict", action="store_true", help="no GPU: the model's table only")
    p.add_argument("--mp", type=int, default=None, help="--predict: multiProcessorCount")
    p.add_argument("--threads-per-mp", type=int, default=2048, help="--predict: maxThreadsPerMultiProcessor")
    p.add_argument("--warp", type=int, default=32, help="--predict: warp size")
    a = p.parse_args()

    if a.predict:
        mp, tpm, warp, rocm, gpu = a.mp or 20, a.threads_per_mp, a.warp, True, False
        print("model only (no GPU): multiProcessorCount {}, maxThreadsPerMultiProcessor {}, warp {}, ROCm".format(mp, tpm, warp))
    else:
        if not torch.cuda.is_available():
            raise SystemExit("needs a CUDA / HIP device (or --predict)")
        pr = torch.cuda.get_device_properties(0)
        mp = pr.multi_processor_count
        tpm = getattr(pr, "max_threads_per_multi_processor", None)
        warp = getattr(pr, "warp_size", None)
        rocm, gpu = torch.version.hip is not None, True
        print("device {} ({}), multiProcessorCount {}, maxThreadsPerMultiProcessor {}, warp {}; torch {} hip {}".format(
            pr.name, getattr(pr, "gcnArchName", "?"), mp, tpm, warp, torch.__version__, torch.version.hip), flush=True)
        if tpm is None or warp is None:
            print("(this torch does not expose maxThreadsPerMultiProcessor / warp_size: the prediction uses 2048 / 32)")
            tpm, warp = tpm or 2048, warp or 32

    rows_out = []
    print("predicted = the scratch the launch configuration allocates; peak+ = max_memory_allocated above the inputs during the "
          "call, the output included (the scratch is freed before the output is allocated, so a call with a scratch has "
          "peak+ = scratch + its small results); out = the output; g scratch = the grouped candidate's peak+ minus its output")
    hdr = "{:38s} {:>5s} {:>6s} {:>5s} {:>2s} {:>6s} {:>8s} | {:>9s} {:>9s} {:>8s} {:>9s} {:>4s} | {:>9s} {:>9s} {:>9s} {:>11s} | {}"
    print(hdr.format("case", "C", "rows", "cols", "nd", "layout", "positions", "predicted", "peak+", "out", "resv+", "segs",
                     "g scratch", "max|Δ|", "rel fp64", "ms n/g", "config (vec, block, values/thread, grid / target, ctas)"))
    for dtype in (torch.bfloat16, torch.float32):
        for label, rows, cols, c in cases(a.quick):
            for nd in (5, 4):
                for sliced in (False, True):
                    n = rows * cols
                    full_rows = rows + 8 if sliced else rows
                    pred, cfg = predict(n, c, full_rows * cols, 4 * cols if sliced else 0, mp, tpm, warp, rocm)
                    row = {"case": label, "dtype": str(dtype), "C": c, "rows": rows, "cols": cols, "nd": nd, "sliced": sliced,
                           "positions": n, "predicted": pred, "config": cfg}
                    if gpu:
                        x, base = make(c, rows, cols, dtype, nd, sliced, rows * 7 + c)
                        y, peak, res, segs = measure(lambda t: F.normalize(t, dim=1), x)
                        out_b = y.untyped_storage().nbytes()
                        yg, gpeak, _, _ = measure(grouped_normalize, x)
                        ref = F.normalize(x.double(), dim=1)
                        d = float((yg.float() - y.float()).abs().max())
                        rel = float(((yg.double() - ref).norm() / ref.norm()))
                        rel_native = float(((y.double() - ref).norm() / ref.norm()))
                        tn, tg = timed(lambda t: F.normalize(t, dim=1), x), timed(grouped_normalize, x)
                        row.update(peak=peak, out=out_b, triggered=peak > out_b + NOISE, reserved=res, segments=segs,
                                   grouped_scratch=gpeak - out_b, max_diff=d, rel_fp64=rel,
                                   rel_fp64_native=rel_native, ms_native=tn, ms_grouped=tg,
                                   differing=float((yg != y).float().mean()))
                        del x, base, y, yg, ref
                    print("{:38s} {:5d} {:6d} {:5d} {:2d} {:>6s} {:8d} | {:>9s} {:>9s} {:>8s} {:>9s} {:>4s} | {:>9s} {:>9s} {:>9s} {:>11s} | {}".format(
                        label[:38] + (" " + str(dtype).replace("torch.", "") if label == "sweep" else ""), c, rows, cols, nd,
                        "slice" if sliced else "contig", n, "{:.1f}".format(pred / MIB),
                        "{:.1f}".format(row["peak"] / MIB) if gpu else "-", "{:.1f}".format(row["out"] / MIB) if gpu else "-",
                        "{:.1f}".format(row["reserved"] / MIB) if gpu else "-",
                        str(row["segments"]) if gpu else "-", "{:.2f}".format(row["grouped_scratch"] / MIB) if gpu else "-",
                        "{:.2g}".format(row["max_diff"]) if gpu else "-", "{:.2g}".format(row["rel_fp64"]) if gpu else "-",
                        "{:.2f}/{:.2f}".format(row["ms_native"], row["ms_grouped"]) if gpu else "-",
                        "{} {}x{} {} {}/{} {}".format(cfg["vec"], cfg["block"][0], cfg["block"][1], cfg["values_per_thread"], cfg["grid"],
                                                       cfg["target_grid"], cfg["ctas_per_output"])), flush=True)
                    rows_out.append(row)
        print()

    # the trigger range per (dtype, channels, layout): positions whose call allocated (or is predicted to allocate) a scratch
    print("=== where the scratch appears (measured: peak+ more than {} MiB above the output; predicted where not measured) ===".format(NOISE // MIB))
    groups = {}
    for r in rows_out:
        if r["case"].startswith("sweep"):
            groups.setdefault((r["dtype"], r["C"], r["nd"], r["sliced"]), []).append(r)
    for (dt, c, nd, sl), rs in sorted(groups.items()):
        trig = (lambda r: r["triggered"]) if gpu else (lambda r: r["predicted"] > 0)
        size = (lambda r: r["peak"]) if gpu else (lambda r: r["predicted"])
        hit = [r for r in rs if trig(r)]
        miss = [r["positions"] for r in rs if not trig(r)]
        agree = None
        if gpu:   # the same rows trigger, and where they do the peak is the predicted scratch (plus results of a few MiB)
            off = max((abs(r["peak"] - r["predicted"]) for r in hit), default=0)
            agree = all(trig(r) == (r["predicted"] > 0) for r in rs) and off <= NOISE
        print("{:15s} C {:4d} {}D {:6s}: scratch for {} positions{}; none for {} positions{}".format(
            dt.replace("torch.", ""), c, nd, "slice" if sl else "contig",
            "{} .. {} ({} sizes)".format(min(r["positions"] for r in hit), max(r["positions"] for r in hit), len(hit)) if hit else "no",
            ", {:.0f} B per position".format(statistics_mean([size(r) / r["positions"] for r in hit])) if hit else "",
            "{} .. {} ({} sizes)".format(min(miss), max(miss), len(miss)) if miss else "no",
            "" if agree is None else ("; prediction agrees (peak - predicted <= {:.1f} MiB)".format(off / MIB) if agree
                                      else "; PREDICTION DIFFERS (peak - predicted up to {:.1f} MiB)".format(off / MIB))))
    if gpu:
        worst = max(rows_out, key=lambda r: r["grouped_scratch"])
        print("grouped candidate: largest scratch {:.2f} MiB ({}); max|Δ| vs F.normalize {:.3g}; relative error vs fp64 {:.3g} "
              "(F.normalize {:.3g}); time {:.2f} vs {:.2f} ms summed over all rows".format(
                  worst["grouped_scratch"] / MIB, worst["case"], max(r["max_diff"] for r in rows_out),
                  max(r["rel_fp64"] for r in rows_out), max(r["rel_fp64_native"] for r in rows_out),
                  sum(r["ms_grouped"] for r in rows_out), sum(r["ms_native"] for r in rows_out)))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows_out, f)
        print("results written to {}".format(a.json))


def statistics_mean(v):
    return sum(v) / len(v) if v else math.nan


if __name__ == "__main__":
    sys.exit(main())
