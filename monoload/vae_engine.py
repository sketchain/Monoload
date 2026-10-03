"""Layer 1 engine: stripe decoding of a recognized decoder (docs/DESIGN.md §9.13).

What is the same for every decoder lives here; what depends on the decoder's
structure lives in its adapter (monoload/vae_wan.py: the Wan 2.1 VAE single
frame). Adding a decoder means writing an adapter.

A decoder the engine handles is a whole-image prefix (low resolution, global
attention included) followed by a chain of units that are local in the row
dimension. The prefix runs on the whole image; its output is the checkpoint.
The output is then produced in stripes of rows: for a stripe [o0, o1) the rows
each unit needs are worked out backwards (need_in), then the units run forwards
on exactly those rows of the checkpoint.

Each unit is the original module instance, called unchanged on a slice of
rows (comfy.ops cast / weight_function paths and the op-level chunking of
layer 2 apply as usual). A slice that does not start at the real top edge of
the image is zero-padded by the module like any input; the output rows that
depend on that padding are exactly the first `halo` rows (and likewise at the
bottom), and they are dropped: valid_out() computes which rows of a unit's
output are exact from the global row range of its input, and a stripe only
continues with rows that are both needed and valid (checked on every unit, a
violation is an internal error). So zero padding only reaches the result at the
real top and bottom edges of the image, inner stripe boundaries use the real
neighbouring rows, and each stripe writes only its own core rows into the
preallocated output.

The adapter interface (StripeAdapter):
  name, key            label for logs, structure signature (self-test cache)
  module               the nn.Module the op-level chunking is installed on
  prefix               [(name, module)] run on the whole image
  units                [Unit] the stripe part
  ckpt_channels        channels of the checkpoint
  hdim                 the row dimension of the tensors (5D Wan: 3, 4D: 2)
  scale                output rows per latent row
  out_channels         channels of the decoder output
  output_shape(samples)            the native-layout shape of the decoded batch
  cost model           prefix_peak / prefix_largest / prefix_out_channels /
                       prefix_macs (per prefix module), unit_peak / unit_largest
                       (per unit; the work of a unit is Unit.macs_row)
  fp32_copy(device)    an adapter bound to an fp32 copy of the decoder (self-test)
  reference_decode(z)  the decoder's own whole-image decode (on such a copy)
  selftest_latent(n)   the latent shape of the self-test
  selftest_params()    parameter count of what fp32_copy copies
"""

import gc
import os
import time

import torch

import comfy.model_management as mm

from .errors import MonoloadError
from .vae_ops import MIB, OpChunking, OpStats, fmt_bytes

POINT, CONV, RES, UP = "point", "conv", "res", "up"
MIN_ROWS = 8                 # smallest automatic / OOM-retry stripe core height (output rows)
SELFTEST_LATENT = 24         # latent rows/cols of the self-test
SELFTEST_ROWS = 40           # forced stripe core height in the self-test (192 output rows -> 5 stripes, the last one shorter)
SELFTEST_WORKSPACE = 8 * MIB # small workspace: the big convs inside the stripes also run in layer-2 row blocks
SELFTEST_TOL = 1e-4          # max|stripes - whole| / max(1, max|whole|), fp32


class StripeError(MonoloadError):
    """Internal inconsistency of the stripe plan (a stripe would use rows that are not exact)."""


# ---------------------------------------------------------------------------
# intervals (pure functions; rows are global, half-open)
# ---------------------------------------------------------------------------

class Unit:
    """One step of the stripe part: kind, halo (rows on each side a unit
    needs, at its output level), scale (output rows per input row), channels."""
    __slots__ = ("kind", "module", "halo", "scale", "cin", "cout", "name", "macs_row", "shortcut")

    def __init__(self, kind, module, halo, scale, cin, cout, name, macs_row=0, shortcut=False):
        self.kind, self.module, self.halo, self.scale = kind, module, halo, scale
        self.cin, self.cout, self.name, self.macs_row = cin, cout, name, macs_row
        self.shortcut = shortcut   # residual block with a 1x1 conv shortcut


def need_in(unit, a, b, h_in, h_out):
    """Input rows unit needs to produce output rows [a, b), within the image."""
    if unit.kind == UP:  # nearest x2, then a conv with `halo` rows on each side at the output level
        c0, c1 = max(0, a - unit.halo), min(h_out, b + unit.halo)
        return c0 // 2, min(h_in, -(-c1 // 2))
    return max(0, a - unit.halo), min(h_in, b + unit.halo)


def valid_out(unit, xa, xb, h_out):
    """Output rows of unit that are exact when it runs on input rows [xa, xb):
    the rows next to a slice edge that is not a real image edge depend on the
    zero padding the module adds there."""
    oa, ob = xa * unit.scale, xb * unit.scale
    return oa + (unit.halo if oa > 0 else 0), ob - (unit.halo if ob < h_out else 0)


def stripe_needs(units, heights, o0, o1):
    """needs[i] = input rows of unit i (needs[-1] = the stripe's output rows)."""
    needs = [None] * (len(units) + 1)
    needs[-1] = (o0, o1)
    for i in reversed(range(len(units))):
        needs[i] = need_in(units[i], needs[i + 1][0], needs[i + 1][1], heights[i], heights[i + 1])
    return needs


def split_rows(h, rows):
    """Balanced stripes of at most `rows` output rows."""
    rows = max(1, min(int(rows), h))
    n = -(-h // rows)
    base, extra = divmod(h, n)
    out, o = [], 0
    for i in range(n):
        r = base + (1 if i < extra else 0)
        out.append((o, o + r))
        o += r
    return out


def hooked_module(model):
    """Name of a module whose call would not be its class' forward (forward
    hooks, instance-level forward replacements such as bypass injections),
    or None."""
    for name, m in model.named_modules():
        if "forward" in m.__dict__ or m._forward_hooks or m._forward_pre_hooks:
            return name or "<root>"
    if torch.nn.modules.module._global_forward_hooks or torch.nn.modules.module._global_forward_pre_hooks:
        return "<global module hooks>"
    return None


def conv_io(m):
    w = m.weight
    return int(w.shape[1]), int(w.shape[0])


# ---------------------------------------------------------------------------
# memory model (bytes)
# ---------------------------------------------------------------------------
# The adapter counts the live tensors of its modules from their forward code
# (DESIGN §9.13.4); conv_extra is how the layer-2 conv wrapper runs a conv.
#
# The caching allocator then decides what is reserved: the decode runs in one
# arena (reserve_arena) of the live peak plus room for fragmentation, sized with
# tests/alloc_sim.py (DESIGN §9.13.4: stripes needed at most live + 2.2 %, a
# single whole-image stripe up to live + 12 %); the estimate is the arena plus
# the largest single allocation plus ESTIMATE_PAD for the small-block pool.

ARENA_DIV = 32               # arena = live peak + live peak / ARENA_DIV + ARENA_PAD, rounded up to 2 MiB
ARENA_DIV_SINGLE = 8         # ... with one stripe (whole-image planes of 1-2 GiB fragment more)
ARENA_PAD = 64 * MIB
ESTIMATE_PAD = 16 * MIB      # small-block pool (<= 1 MiB requests come from their own 2 MiB segments; 2-6 MiB seen)
CONTIGUOUS_INPUT = (UP,)     # units whose row slice is made contiguous before the call
ARENA_ENABLED = True         # bench --no-arena: off, to measure the tensors' own peak (the arena block counts as allocated)


def conv_extra(cin, cout, k, r_in, r_out, w_in, w_out, e, ws, contiguous=True):
    """Bytes a stride-1 k x k conv needs besides its input and output, as the
    layer-2 conv wrapper runs it: unblocked (columns <= ws) the columns and a
    contiguous copy of a sliced input; in row blocks one block of columns, the
    block's input copy and output."""
    per_row = k * k * cin * w_out * e
    full = per_row * r_out
    wcopy = cout * cin * k * k * e
    if full <= ws:
        return full + (0 if contiguous else r_in * w_in * cin * e) + wcopy
    rb = max(1, ws // per_row)
    return rb * per_row + (rb + k - 1) * w_in * cin * e + rb * w_out * cout * e + wcopy


def arena_bytes(live, stripes=2):
    a = live + live // (ARENA_DIV if stripes > 1 else ARENA_DIV_SINGLE) + ARENA_PAD
    return -(-a // (2 * MIB)) * (2 * MIB)


class Plan:
    """Stripes, intervals and the memory / work estimate of one decode."""

    def __init__(self, bound, h8, w8, rows, ws, elem, out_bytes, lat_bytes):
        self.rows = rows
        self.workspace = ws
        self.hdim = bound.hdim
        units = bound.units
        heights = [h8]
        widths = [w8]
        for u in units:
            heights.append(heights[-1] * u.scale)
            widths.append(widths[-1] * u.scale)
        self.heights, self.widths = heights, widths
        self.h_out, self.w_out = heights[-1], widths[-1]
        self.stripes = split_rows(self.h_out, rows)
        self.needs = [stripe_needs(units, heights, a, b) for a, b in self.stripes]
        # the stripe with the largest slices runs first: the blocks it leaves in the allocator's cache fit every later stripe
        size = [sum(n[1] - n[0] for n in needs) for needs in self.needs]
        first = max(range(len(size)), key=lambda i: (size[i], -i))
        self.order = [first] + [i for i in range(len(size)) if i != first]
        self.ckpt_bytes = bound.ckpt_channels * h8 * w8 * elem
        prefix_live = s = 0
        largest = max(out_bytes, lat_bytes)
        for _, m in bound.prefix:
            prefix_live = max(prefix_live, bound.prefix_peak(m, h8, w8, elem, ws, s))
            largest = max(largest, bound.prefix_largest(m, h8, w8, elem, ws))
            s = bound.prefix_out_channels(m) * h8 * w8 * elem
        strip_live = 0
        work_stripes = 0
        for needs in self.needs:
            s = 0   # the first unit reads a slice of the checkpoint (counted on its own)
            for i, u in enumerate(units):
                r = needs[i][1] - needs[i][0]
                strip_live = max(strip_live, bound.unit_peak(u, r, widths[i], elem, ws, s))
                largest = max(largest, bound.unit_largest(u, r, widths[i], elem, ws))
                s = r * u.scale * widths[i + 1] * u.cout * elem
                work_stripes += r * u.scale * widths[i + 1] * u.macs_row
        self.prefix_live, self.stripe_live = prefix_live, strip_live
        self.prefix_bytes = prefix_live
        self.stripe_bytes = self.ckpt_bytes + strip_live
        self.persistent = out_bytes + lat_bytes
        self.live_peak = int(self.persistent + max(self.prefix_bytes, self.stripe_bytes))
        self.arena = arena_bytes(self.live_peak, len(self.stripes))
        self.largest = int(largest)
        # reserved = the arena, unless fragmentation strands one request outside it (then that request's own segment)
        self.estimate = self.arena + self.largest + ESTIMATE_PAD
        work_prefix = sum(bound.prefix_macs(m) for _, m in bound.prefix) * h8 * w8
        work_whole = work_prefix + sum(heights[i + 1] * widths[i + 1] * u.macs_row for i, u in enumerate(units))
        self.recompute = (work_prefix + work_stripes) / max(1, work_whole)

    def describe(self):
        return "{} stripes of {} rows (core), recompute {:.2f}x, checkpoint {}".format(
            len(self.stripes), max(b - a for a, b in self.stripes), self.recompute, fmt_bytes(self.ckpt_bytes))


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def run_prefix(prefix, z):
    x = z
    for _, m in prefix:
        x = m(x)
    return x


def run_stripes(units, plan, ckpt, write):
    """Run every stripe from the checkpoint; write(o0, o1, rows) receives the
    stripe's core rows (rows along plan.hdim)."""
    heights = plan.heights
    hdim = plan.hdim
    for k in plan.order:
        (o0, o1), needs = plan.stripes[k], plan.needs[k]
        xa, xb = needs[0]
        x = ckpt.narrow(hdim, xa, xb - xa)
        for i, u in enumerate(units):
            y = u.module(x)
            del x
            oa = xa * u.scale
            if y.shape[hdim] != (xb - xa) * u.scale:
                raise StripeError("[Monoload] internal error: {} returned {} rows for {} input rows".format(u.name, y.shape[hdim], xb - xa))
            va, vb = valid_out(u, xa, xb, heights[i + 1])
            ta, tb = needs[i + 1]
            if ta < va or tb > vb:
                raise StripeError("[Monoload] internal error: stripe [{}, {}) needs rows [{}, {}) of {} but only [{}, {}) are exact".format(
                    o0, o1, ta, tb, u.name, va, vb))
            x = y.narrow(hdim, ta - oa, tb - ta)
            del y
            if i + 1 < len(units) and units[i + 1].kind in CONTIGUOUS_INPUT and not x.is_contiguous():
                x = x.contiguous()   # the slice's own copy instead of the one upsampling would make inside (DESIGN §9.13.4)
            xa, xb = ta, tb
        write(o0, o1, x)
        del x


def arena_supported(device):
    """True when the decode can run in one reserved segment of the caching
    allocator (DESIGN §9.13.4): a CUDA / HIP device with PyTorch's native
    allocator in its default mode -- blocks of a segment are split and merged,
    no max_split_size_mb (that would keep the arena from being split), no
    expandable segments (they do not fragment like this in the first place)."""
    if not ARENA_ENABLED or device.type != "cuda":
        return False
    try:
        if torch.cuda.memory.get_allocator_backend() != "native":
            return False
    except Exception:
        return False
    conf = ",".join(os.environ.get(k, "") for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"))
    conf = conf.replace(" ", "").lower()
    return "max_split_size_mb" not in conf and "expandable_segments:true" not in conf


def reserve_arena(device, nbytes):
    """Allocate nbytes in one block and free it at once: the caching allocator
    keeps it as one free segment, and the decode's tensors are then carved
    out of it (best fit, split, merged again when freed) instead of each new
    size getting a segment of its own that only that size can reuse."""
    t = torch.empty(int(nbytes), dtype=torch.uint8, device=device)
    del t


class StripeAdapter:
    """What every layer-1 adapter shares: planning and running a decode. A
    subclass binds to one decoder instance and provides the structure and the
    cost model (see the module docstring)."""

    name = "?"
    hdim = 2
    scale = 8

    def plan(self, vae, samples, budget, ws, rows=None, out_bytes=0, measure="estimate"):
        """rows=None: the largest stripe height whose `measure` ("estimate", the
        bound handed to load_models_gpu, or "arena", what is reserved) fits the
        budget (None if even MIN_ROWS does not fit); else exactly `rows`."""
        h8, w8 = int(samples.shape[-2]), int(samples.shape[-1])
        elem = mm.dtype_size(vae.vae_dtype)
        lat = samples[0:1].numel() * elem
        if rows is not None:
            return Plan(self, h8, w8, rows, ws, elem, out_bytes, lat)
        h_out = h8 * self.scale
        best = None
        lo, hi = min(MIN_ROWS, h_out), h_out
        if getattr(Plan(self, h8, w8, lo, ws, elem, out_bytes, lat), measure) > budget:
            return None
        while lo <= hi:
            mid = (lo + hi) // 2
            p = Plan(self, h8, w8, mid, ws, elem, out_bytes, lat)
            if getattr(p, measure) <= budget:
                best, lo = p, mid + 1
            else:
                hi = mid - 1
        return best

    def prefix_pass(self, z):
        """The prefix on the whole image -> the checkpoint."""
        return run_prefix(self.prefix, z)

    def stripe_pass(self, ckpt, plan, write):
        """The stripes from the checkpoint."""
        run_stripes(self.units, plan, ckpt, write)

    def run(self, vae, samples_in, plan, budget_ws, stats):
        """Decode every sample with the stripe plan; returns pixel_samples
        (process_output applied) shaped like native's output."""
        n = samples_in.shape[0]
        hdim = self.hdim
        if plan.arena and arena_supported(vae.device):
            reserve_arena(vae.device, plan.arena)
            stats.arena = plan.arena
        pixel_samples = None
        with OpChunking(self.module, budget_ws, stats):
            for i in range(n):
                z = samples_in[i:i + 1].to(device=vae.device, dtype=vae.vae_dtype)
                ckpt = self.prefix_pass(z)
                del z
                if pixel_samples is None:
                    pixel_samples = torch.empty(self.output_shape(samples_in), device=vae.output_device, dtype=vae.vae_output_dtype())
                dst = pixel_samples[i:i + 1]

                def write(o0, o1, rows, dst=dst):
                    dst.narrow(hdim, o0, o1 - o0).copy_(rows)  # = native's .to(output dtype) + copy_, per stripe
                self.stripe_pass(ckpt, plan, write)
                del ckpt
                vae.process_output(pixel_samples[i:i + 1])
        return pixel_samples

    def output_bytes(self, vae, samples):
        """The output buffer, when it lives on the VAE's device (--gpu-only), else 0."""
        a, b = torch.device(vae.output_device), torch.device(vae.device)
        if a.type != b.type or (a.index or 0) != (b.index or 0):
            return 0
        n = 1
        for d in self.output_shape(samples):
            n *= int(d)
        return n * mm.dtype_size(vae.vae_output_dtype())

    def selftest_memory(self):
        """What the self-test needs on the device (load_models_gpu): the fp32
        copy of the weights, the workspace, a margin."""
        return int(self.selftest_params() * 4 + 2 * SELFTEST_WORKSPACE + 256 * MIB)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

_SELFTEST = {}   # structure key -> (ok, detail)


def self_test(bound, vae):
    """First use of a structure in this process: decode a small latent with
    forced small stripes (several inner boundaries, layer-2 conv blocks inside)
    and with the decoder's own whole-image decode, both on an fp32 copy of the
    decoder; pass if they agree to SELFTEST_TOL. Cached per structure; the RNG
    state is preserved. Afterwards the fp32 copy and every tensor of the test
    are freed and the allocator's cache is emptied, so the decode that follows
    starts from the same memory state as any later one. Returns (ok, detail)."""
    hit = _SELFTEST.get(bound.key)
    if hit is not None:
        return hit
    oom = False
    try:
        ok, detail = _self_test_run(bound, vae)
    except Exception as e:
        if not mm.is_oom(e):
            ok, detail = False, "{}: {}".format(type(e).__name__, str(e).splitlines()[0] if str(e) else "")
        else:
            oom = True
    # out of the except block: the traceback (and the tensors its frames hold) is gone
    gc.collect()
    mm.soft_empty_cache(True)
    if oom:
        raise mm.OOM_EXCEPTION("[Monoload] out of memory in the VAE layer-1 self-test")
    _SELFTEST[bound.key] = (ok, detail)
    return ok, detail


def _self_test_run(bound, vae):
    t0 = time.perf_counter()
    dev = vae.device
    devs = [dev.index or 0] if dev.type == "cuda" else []
    with torch.inference_mode(), torch.random.fork_rng(devices=devs):
        tb = bound.fp32_copy(dev)
        if tb.key != bound.key:
            raise StripeError("fp32 copy has structure {} instead of {}".format(tb.key, bound.key))
        g = torch.Generator().manual_seed(0)
        z = torch.randn(bound.selftest_latent(SELFTEST_LATENT), generator=g).to(dev)
        ref = tb.reference_decode(z)
        plan = Plan(tb, SELFTEST_LATENT, SELFTEST_LATENT, SELFTEST_ROWS, SELFTEST_WORKSPACE, 4, 0, 0)
        out = torch.empty_like(ref)
        stats = OpStats()
        hdim = tb.hdim

        def write(o0, o1, rows):
            out.narrow(hdim, o0, o1 - o0).copy_(rows)
        with OpChunking(tb.module, SELFTEST_WORKSPACE, stats):
            ckpt = tb.prefix_pass(z)
            tb.stripe_pass(ckpt, plan, write)
        scale = max(1.0, float(ref.abs().max()))
        err = float((out - ref).abs().max()) / scale
        ok = bool(torch.isfinite(out).all()) and err <= SELFTEST_TOL
        detail = "max|stripes - whole| / max(1, max|whole|) = {:.2g} (tolerance {:g}), {} stripes, {} conv calls in row blocks, {:.2f}s".format(
            err, SELFTEST_TOL, len(plan.stripes), stats.conv_chunked, time.perf_counter() - t0)
    return ok, detail
