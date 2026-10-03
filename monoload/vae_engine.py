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
import comfy.ops

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
    __slots__ = ("kind", "module", "halo", "scale", "cin", "cout", "name", "macs_row", "shortcut", "norms", "contiguous")

    def __init__(self, kind, module, halo, scale, cin, cout, name, macs_row=0, shortcut=False, norms=(), contiguous=False):
        self.kind, self.module, self.halo, self.scale = kind, module, halo, scale
        self.cin, self.cout, self.name, self.macs_row = cin, cout, name, macs_row
        self.shortcut = shortcut   # residual block with a 1x1 conv shortcut
        self.norms = tuple(norms)  # NormRef: norms in this unit that need whole-image statistics, in call order
        self.contiguous = contiguous   # its row slice is made contiguous before the call (as for the CONTIGUOUS_INPUT kinds)


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
ARENA_DIV_SAVES = 16         # ... with GroupNorm saves (the pool splits the arena: tests/alloc_sim.py measured up to live + 9.5 %)
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


def arena_bytes(live, stripes=2, saves=False):
    a = live + live // (ARENA_DIV_SINGLE if stripes <= 1 else ARENA_DIV_SAVES if saves else ARENA_DIV) + ARENA_PAD
    return -(-a // (2 * MIB)) * (2 * MIB)


class Pass:
    """One statistics pass (DESIGN §9.14): from the save at `start`, run `chain`
    (units[start:unit], plus the norm's partial unit when the norm sits inside
    unit `unit`) on stripes of rows at the norm's input level, and accumulate
    the norm's statistics over each stripe's core rows. On the way, the saves
    in `builds` (positions start < p <= unit) are filled with the core rows."""
    __slots__ = ("start", "unit", "norm", "chain", "heights", "widths", "rows", "stripes", "needs", "order", "builds",
                 "alive", "live", "largest", "work")


def _chain_cost(bound, chain, heights, widths, stripes, elem, ws, tail=None):
    """(needs, order, live, largest, work) of running `chain` on `stripes` (rows at
    the chain's output level). live: the largest unit peak (the first unit reads a
    slice of a save, counted on its own); tail(rows, s) -> the peak after the chain
    (the statistics of a pass). The peaks depend only on the rows each unit runs
    on, so each distinct pattern of rows is evaluated once."""
    needs_all = [stripe_needs(chain, heights, a, b) for a, b in stripes]
    # the stripe with the largest slices runs first: the blocks it leaves in the allocator's cache fit every later stripe
    size = [sum(n[1] - n[0] for n in needs) for needs in needs_all]
    first = max(range(len(size)), key=lambda i: (size[i], -i))
    order = [first] + [i for i in range(len(size)) if i != first]
    patterns = {}
    for needs in needs_all:
        key = tuple(n[1] - n[0] for n in needs)
        patterns[key] = patterns.get(key, 0) + 1
    live = largest = work = 0
    for rows, count in patterns.items():
        s = 0
        for i, u in enumerate(chain):
            r = rows[i]
            live = max(live, bound.unit_peak(u, r, widths[i], elem, ws, s))
            largest = max(largest, bound.unit_largest(u, r, widths[i], elem, ws))
            s = r * u.scale * widths[i + 1] * u.cout * elem
            work += count * r * u.scale * widths[i + 1] * u.macs_row
        if tail is not None:
            live = max(live, tail(rows[-1], s))
    return needs_all, order, live, largest, work


def moments_chunk_rows(c, w, ws):
    """Rows per chunk of the fp32 copies a frozen-statistics norm and the
    statistics accumulation make (DESIGN §9.14): at most ws bytes, >= 1 row."""
    return max(1, ws // max(1, c * w * 4))


def norm_temp(c, r, w, ws):
    """Bytes of one fp32 chunk of r rows (see moments_chunk_rows)."""
    return min(r, moments_chunk_rows(c, w, ws)) * c * w * 4


_PASS_COST = {}   # (structure key, geometry, pass, rows) -> (stripes, _chain_cost): pass costs do not depend on the final stripes


class Plan:
    """Stripes, intervals and the memory / work estimate of one decode.

    Without global norms (Wan): the prefix, then one pass of stripes from the
    checkpoint. With them (GroupNorm, DESIGN §9.14): the prefix, a statistics
    pass per norm of the stripe part (in chain order, each from the latest save),
    then the final pass of output stripes from the last save."""

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
        self.ckpt_bytes = bound.ckpt_channels * h8 * w8 * elem
        self.channels = [bound.ckpt_channels] + [u.cout for u in units]
        targets = [(i, nr) for i, u in enumerate(units) for nr in u.norms]
        saves = sorted(p for p in set(bound.save_positions()) if 0 < p <= targets[-1][0]) if targets else []
        self.saves = saves

        def save_bytes(p):
            return self.ckpt_bytes if p == 0 else units[p - 1].cout * heights[p] * widths[p] * elem
        self.save_bytes = {p: save_bytes(p) for p in [0] + saves}

        # the final pass: output stripes from the last save
        sf = saves[-1] if saves else 0
        self.final_start = sf
        self.final_units = units[sf:]
        self.final_heights = heights[sf:]
        self.stripes = split_rows(self.h_out, rows)
        prefix_live = s = 0
        largest = max(out_bytes, lat_bytes)
        for _, m in bound.prefix:
            prefix_live = max(prefix_live, bound.prefix_peak(m, h8, w8, elem, ws, s))
            largest = max(largest, bound.prefix_largest(m, h8, w8, elem, ws))
            s = bound.prefix_out_channels(m) * h8 * w8 * elem
        self.needs, self.order, strip_live, lg, work_stripes = _chain_cost(
            bound, self.final_units, self.final_heights, widths[sf:], self.stripes, elem, ws)
        largest = max(largest, lg)
        self.prefix_live, self.stripe_live = prefix_live, strip_live
        self.prefix_bytes = prefix_live
        self.stripe_bytes = strip_live + self.ckpt_bytes if sf == 0 else None   # with saves: set by the save layout below

        # statistics passes: the peak none of them can avoid is the saves they hold plus their smallest stripes;
        # within the larger of that and the prefix / final pass, each takes the tallest stripes that fit
        self.passes = []
        cur, built = 0, {0}
        for i, nr in targets:
            ps = Pass()
            ps.start, ps.unit, ps.norm = cur, i, nr
            ps.builds = [p for p in saves if cur < p <= i and p not in built]
            ps.chain = units[cur:i] + ([nr.partial] if nr.partial is not None else [])
            ps.heights = heights[cur:i + 1] + ([heights[i] * nr.partial.scale] if nr.partial is not None else [])
            ps.widths = widths[cur:i + 1] + ([widths[i] * nr.partial.scale] if nr.partial is not None else [])
            self.passes.append(ps)
            built.update(ps.builds)
            if ps.builds:
                cur = max(ps.builds)
        # Where the saves live (DESIGN §9.14), two layouts, the one with the smaller arena is used:
        #   separate  each save its own allocation, freed when nothing starts from it any more; a later, larger save
        #             cannot use the holes of dead smaller ones, so the arena gets that much slack
        #   pool      one allocation of slots (a save shares a slot with an earlier one that is dead by then: 2 slots
        #             when each pass builds one save), from the first save on: no holes, but held whole to the end
        n_pass = len(self.passes)
        dies = {}                                   # save -> index of the pass after which nothing starts from it
        for k, ps in enumerate(self.passes):
            if ps.builds:
                dies[ps.start] = k
        self.slots, self.save_slot = [], {}
        slack = 0
        for k, ps in enumerate(self.passes):
            for p in ps.builds:
                dead = [q for q in [0] + saves if q in dies and dies[q] < k]
                slack = max(slack, sum(self.save_bytes[q] for q in dead if self.save_bytes[q] < self.save_bytes[p]))
                free = [j for j, (size, last) in enumerate(self.slots) if last is not None and dies.get(last, n_pass) < k]
                j = min(free, key=lambda j: abs(self.slots[j][0] - self.save_bytes[p])) if free else len(self.slots)
                if j == len(self.slots):
                    self.slots.append([0, None])
                self.slots[j] = [max(self.slots[j][0], self.save_bytes[p]), p]
                self.save_slot[p] = j
        self.slot_offsets = []
        off = 0
        for size, _ in self.slots:
            self.slot_offsets.append(off)
            off += -(-size // 512) * 512
        self.pool_bytes = off
        first_build = next((k for k, ps in enumerate(self.passes) if ps.builds), None)

        # a pass's cost depends on the geometry and its own stripes, not on the final stripes: cached across plans
        cache = _PASS_COST
        if len(cache) > 50000:
            cache.clear()

        def cost(ps, r):
            key = (bound.key, h8, w8, ws, elem, ps.start, ps.unit, ps.norm.name, r)
            hit = cache.get(key)
            if hit is None:
                st = split_rows(ps.heights[-1], r)
                c_t, w_t = ps.norm.channels, ps.widths[-1]
                hit = cache[key] = (st, _chain_cost(bound, ps.chain, ps.heights, ps.widths, st, elem, ws,
                                                    tail=lambda rr, s_: s_ + norm_temp(c_t, rr, w_t, ws)))
            return hit

        def layout(alive, final_bytes):
            """Statistics passes' stripes for these held bytes: the peak none of them can avoid is what they hold plus
            their smallest stripes; within the larger of that and the prefix / final pass, each takes the tallest
            stripes that fit. -> (final bytes, passes' bytes, [(rows, cost)])"""
            floor = [cost(ps, 1) for ps in self.passes]
            ref = max([self.prefix_bytes, final_bytes] + [a + c[1][2] for a, c in zip(alive, floor)])
            out = []
            for ps, a, c1 in zip(self.passes, alive, floor):
                lo, hi, best = 2, ps.heights[-1], (1, c1)
                while lo <= hi:
                    mid = (lo + hi) // 2
                    c = cost(ps, mid)
                    if a + c[1][2] <= ref:
                        best, lo = (mid, c), mid + 1
                    else:
                        hi = mid - 1
                out.append(best)
            return final_bytes, max([0] + [a + b[1][1][2] for a, b in zip(alive, out)]), out

        persistent = out_bytes + lat_bytes
        variants = [("separate", [self.save_bytes[ps.start] + sum(self.save_bytes[p] for p in ps.builds) for ps in self.passes],
                     self.save_bytes[sf] + strip_live, slack, max([0] + [self.save_bytes[p] for p in saves]))]
        if len(saves) > 1:
            variants.append(("pool", [(self.ckpt_bytes if ps.start == 0 else 0) + (self.pool_bytes if k >= first_build else 0)
                                      for k, ps in enumerate(self.passes)], self.pool_bytes + strip_live, 0, self.pool_bytes))
        best_v = None
        for name, alive, final_bytes, extra, save_alloc in variants:
            fb, pb, rows = layout(alive, final_bytes)
            live = int(persistent + max(self.prefix_bytes, fb, pb))
            arena = arena_bytes(live, len(self.stripes), bool(saves)) + extra
            if best_v is None or arena < best_v[0]:
                best_v = (arena, name, alive, fb, pb, rows, live, save_alloc, extra)
        self.arena, self.save_layout, alive, self.stripe_bytes, self.pass_bytes, rows, self.live_peak, save_alloc, self.save_slack = best_v
        for ps, a, (r, (st, (needs, order, live, plg, work))) in zip(self.passes, alive, rows):
            ps.alive, ps.rows, ps.stripes, ps.needs, ps.order, ps.live, ps.work = a, r, st, needs, order, live, work
            largest = max(largest, plg, norm_temp(ps.norm.channels, max(b - a_ for a_, b in st), ps.widths[-1], ws))
        largest = max(largest, save_alloc)
        self.persistent = persistent
        if not self.passes:
            self.live_peak = int(self.persistent + max(self.prefix_bytes, self.stripe_bytes))
            self.arena = arena_bytes(self.live_peak, len(self.stripes))
        self.largest = int(largest)
        # reserved = the arena, unless fragmentation strands one request outside it (then that request's own segment)
        self.estimate = self.arena + self.largest + ESTIMATE_PAD
        work_prefix = sum(bound.prefix_macs(m) for _, m in bound.prefix) * h8 * w8
        work_whole = work_prefix + sum(heights[i + 1] * widths[i + 1] * u.macs_row for i, u in enumerate(units))
        self.work_passes = sum(ps.work for ps in self.passes)
        self.recompute = (work_prefix + work_stripes + self.work_passes) / max(1, work_whole)

    def describe(self):
        d = "{} stripes of {} rows (core), recompute {:.2f}x, checkpoint {}".format(
            len(self.stripes), max(b - a for a, b in self.stripes), self.recompute, fmt_bytes(self.ckpt_bytes))
        if self.passes:
            d += "; {} statistics passes, saves {}".format(
                len(self.passes), "+".join(fmt_bytes(self.save_bytes[p]) for p in self.saves) or "none")
        return d


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def run_prefix(prefix, z):
    x = z
    for _, m in prefix:
        x = m(x)
    return x


def run_chain(chain, heights, src, needs, hdim, label, after=None):
    """Run the units of `chain` on rows needs[0] of src (global rows of src's
    level); returns the rows needs[-1] of the last unit's output. Every unit is
    checked to have produced the next needed rows exactly. after(k, x, a, b):
    called with the rows [a, b) of the output of chain[k - 1]."""
    xa, xb = needs[0]
    x = src.narrow(hdim, xa, xb - xa)
    for i, u in enumerate(chain):
        y = u.module(x)
        del x
        oa = xa * u.scale
        if y.shape[hdim] != (xb - xa) * u.scale:
            raise StripeError("[Monoload] internal error: {} returned {} rows for {} input rows".format(u.name, y.shape[hdim], xb - xa))
        va, vb = valid_out(u, xa, xb, heights[i + 1])
        ta, tb = needs[i + 1]
        if ta < va or tb > vb:
            raise StripeError("[Monoload] internal error: {} needs rows [{}, {}) of {} but only [{}, {}) are exact".format(
                label, ta, tb, u.name, va, vb))
        x = y.narrow(hdim, ta - oa, tb - ta)
        del y
        if i + 1 < len(chain) and (chain[i + 1].kind in CONTIGUOUS_INPUT or chain[i + 1].contiguous) and not x.is_contiguous():
            x = x.contiguous()   # the slice's own copy instead of the one upsampling would make inside (DESIGN §9.13.4)
        xa, xb = ta, tb
        if after is not None:
            after(i + 1, x, xa, xb)
    return x


def run_stripes(units, plan, ckpt, write):
    """Run every output stripe from the save `ckpt` (units = plan.final_units);
    write(o0, o1, rows) receives the stripe's core rows (rows along plan.hdim)."""
    for k in plan.order:
        (o0, o1), needs = plan.stripes[k], plan.needs[k]
        x = run_chain(units, plan.final_heights, ckpt, needs, plan.hdim, "stripe [{}, {})".format(o0, o1))
        write(o0, o1, x)
        del x


# ---------------------------------------------------------------------------
# global statistics of GroupNorm across stripes (DESIGN §9.14)
# ---------------------------------------------------------------------------

class NormRef:
    """A norm in a unit whose statistics are taken over the whole image
    (torch.nn.GroupNorm). partial None: its input is the unit's input; else a
    Unit computing its input from the unit's input (the start of the unit)."""
    __slots__ = ("module", "partial", "name", "groups", "channels", "eps")

    def __init__(self, module, name, partial=None):
        self.module, self.name, self.partial = module, name, partial
        self.groups, self.channels, self.eps = int(module.num_groups), int(module.num_channels), float(module.eps)


class Moments:
    """Per-group count / mean / M2 of a norm's input, fp32 (DESIGN §9.14).

    Shifted data: every value is taken relative to a per-group shift K (the
    first chunk's mean), so the sums stay small even where the mean is large
    against the spread. Each chunk's mean and M2 come from torch.var_mean of
    its fp32 copy minus K (computed in place), and chunks / stripes are merged
    with Chan et al.'s pairwise update
        d = mean_b - mean_a, n = n_a + n_b
        mean = mean_a + d * n_b / n,   M2 = M2_a + M2_b + d^2 * n_a * n_b / n
    which never subtracts two large sums (unlike E[x^2] - E[x]^2)."""

    def __init__(self, groups):
        self.groups = groups
        self.n = 0
        self.shift = self.mean = self.m2 = None

    def add(self, x, hdim, ws):
        g = self.groups
        c = x.shape[1]
        h = x.shape[hdim]
        w = x.numel() // max(1, c * h)
        rows = moments_chunk_rows(c, w, ws)
        for r0 in range(0, h, rows):
            xc = x.narrow(hdim, r0, min(rows, h - r0)).to(torch.float32, copy=True).contiguous()
            n_b = xc.numel() // g
            xg = xc.view(g, n_b)
            if self.shift is None:
                self.shift = xg.mean(dim=1)
            xg.sub_(self.shift.unsqueeze(1))
            var, mean_b = torch.var_mean(xg, dim=1, correction=0)
            del xc, xg
            m2_b = var * n_b
            if self.n == 0:
                self.mean, self.m2 = mean_b, m2_b
            else:
                n = self.n + n_b
                d = mean_b - self.mean
                self.mean = self.mean + d * (n_b / n)
                self.m2 = self.m2 + m2_b + d * d * (self.n * n_b / n)
            self.n += n_b

    def freeze(self, eps):
        """(mean, rstd) per group, fp32: var = M2 / n (biased, as GroupNorm)."""
        return self.shift + self.mean, torch.rsqrt(self.m2 / self.n + eps)


def group_norm_frozen(x, weight, bias, mean, rstd, hdim, ws):
    """GroupNorm of x with given per-group mean / rstd: y = x * a + b with
    a = rstd * weight, b = bias - mean * a per channel, in fp32, rounded to x's
    dtype (what ATen's GroupNorm kernel computes once it has the statistics);
    row chunks of at most ws bytes in fp32."""
    c = x.shape[1]
    rep = c // mean.shape[0]
    a = rstd.repeat_interleave(rep)
    if weight is not None:
        a = a * weight.float()
    b = -mean.repeat_interleave(rep) * a
    if bias is not None:
        b = b + bias.float()
    shape = [1, c] + [1] * (x.ndim - 2)
    a, b = a.view(shape), b.view(shape)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    h = x.shape[hdim]
    w = x.numel() // max(1, c * h * x.shape[0])
    rows = moments_chunk_rows(c, w * x.shape[0], ws)
    for r0 in range(0, h, rows):
        n = min(rows, h - r0)
        out.narrow(hdim, r0, n).copy_(torch.addcmul(b, x.narrow(hdim, r0, n), a))
    return out


class _FrozenGroupNorm:
    """Instance-level `forward` of a GroupNorm during a managed layer-1 decode:
    the module's own weight path (comfy.ops cast_bias_weight with its
    weight_function / bias_function, or the plain parameters), then the norm
    with the whole-image statistics frozen for it (DESIGN §9.14)."""
    __slots__ = ("mod", "store", "hdim", "ws")

    def __init__(self, mod, store, hdim, ws):
        self.mod, self.store, self.hdim, self.ws = mod, store, hdim, ws

    def __call__(self, x):
        mod = self.mod
        st = self.store.get(mod)
        if st is None:
            raise StripeError("[Monoload] internal error: GroupNorm called before its whole-image statistics are known")
        mean, rstd = st
        if isinstance(mod, comfy.ops.CastWeightBiasOp):
            comfy.ops.run_every_op()
            if mod.comfy_cast_weights or len(mod.weight_function) > 0 or len(mod.bias_function) > 0:
                with comfy.ops.CastBiasWeightContext(mod, x, offloadable=True) as (weight, bias):
                    return group_norm_frozen(x, weight, bias, mean, rstd, self.hdim, self.ws)
        return group_norm_frozen(x, mod.weight, mod.bias, mean, rstd, self.hdim, self.ws)


class GlobalNorms:
    """Within the `with` block, the norms of `refs` (instances) run with frozen
    whole-image statistics (`store`: module -> (mean, rstd)); restored on exit,
    also on errors. This replaces the norms' forward (a deliberate exception to
    "modules are called unchanged": GroupNorm on a stripe would use the stripe's
    own statistics); the fp32 self-test checks the result against the whole-image
    decode."""

    def __init__(self, refs, hdim, ws):
        self.refs = refs
        self.hdim, self.ws = hdim, ws
        self.store = {}
        self._saved = []

    def __enter__(self):
        try:
            for nr in self.refs:
                m = nr.module
                if "forward" in m.__dict__:
                    raise StripeError("[Monoload] internal error: {} already has an instance-level forward".format(nr.name))
                self._saved.append(m)
                m.__dict__["forward"] = _FrozenGroupNorm(m, self.store, self.hdim, self.ws)
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self):
        while self._saved:
            self._saved.pop().__dict__.pop("forward", None)

    def __exit__(self, *exc):
        self._restore()
        return False


def run_passes(plan, saves, hdim, ws, norms):
    """The statistics passes of the plan; saves: {position: tensor} (0: the
    prefix checkpoint), filled and pruned as the passes go; the saves are their
    own tensors or views of one pool (Plan.save_layout). Returns the save the
    final pass starts from."""
    pool = None
    for ps in plan.passes:
        src = saves[ps.start]
        elem = src.element_size()
        if ps.builds and pool is None and plan.save_layout == "pool":
            pool = torch.empty(plan.pool_bytes // elem, dtype=src.dtype, device=src.device)
        for p in ps.builds:
            shape = list(src.shape)
            shape[1] = plan.channels[p]
            shape[hdim], shape[hdim + 1] = plan.heights[p], plan.widths[p]
            if pool is None:
                saves[p] = torch.empty(shape, dtype=src.dtype, device=src.device)
                continue
            n = 1
            for d in shape:
                n *= d
            saves[p] = pool.narrow(0, plan.slot_offsets[plan.save_slot[p]] // elem, n).view(shape)
        acc = Moments(ps.norm.groups)
        h_t = ps.heights[-1]
        for k in ps.order:
            (t0, t1), needs = ps.stripes[k], ps.needs[k]

            def after(j, x, a, b, t0=t0, t1=t1):
                p = ps.start + j
                if p in ps.builds:
                    hp = plan.heights[p]
                    c0, c1 = t0 * hp // h_t, t1 * hp // h_t
                    if c0 < a or c1 > b:
                        raise StripeError("[Monoload] internal error: save {} needs rows [{}, {}) but the pass has [{}, {})".format(p, c0, c1, a, b))
                    if c1 > c0:
                        saves[p].narrow(hdim, c0, c1 - c0).copy_(x.narrow(hdim, c0 - a, c1 - c0))
            x = run_chain(ps.chain, ps.heights, src, needs, hdim, "statistics of {} [{}, {})".format(ps.norm.name, t0, t1),
                          after if ps.builds else None)
            acc.add(x, hdim, ws)
            del x
        del src
        norms.store[ps.norm.module] = acc.freeze(ps.norm.eps)
        if ps.builds:
            for p in [q for q in saves if q < max(ps.builds)]:
                del saves[p]
    return saves[plan.final_start]


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

    def save_positions(self):
        """Unit indices p whose input is kept whole (a save) for the statistics passes."""
        return ()

    def norm_refs(self):
        return [nr for u in self.units for nr in u.norms]

    def prefix_pass(self, z):
        """The prefix on the whole image -> the checkpoint."""
        return run_prefix(self.prefix, z)

    def stripe_pass(self, saves, plan, write):
        """The statistics passes (if any) and the output stripes; saves =
        {0: the checkpoint}, emptied as the saves are no longer needed."""
        if not plan.passes:
            run_stripes(self.units, plan, saves.pop(0), write)
            return
        with GlobalNorms(self.norm_refs(), self.hdim, plan.workspace) as norms:
            src = run_passes(plan, saves, self.hdim, plan.workspace, norms)
            saves.clear()
            run_stripes(plan.final_units, plan, src, write)

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
                saves = {0: ckpt}
                del ckpt
                self.stripe_pass(saves, plan, write)
                del saves
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
            tb.stripe_pass({0: tb.prefix_pass(z)}, plan, write)
        scale = max(1.0, float(ref.abs().max()))
        err = float((out - ref).abs().max()) / scale
        ok = bool(torch.isfinite(out).all()) and err <= SELFTEST_TOL
        detail = "max|stripes - whole| / max(1, max|whole|) = {:.2g} (tolerance {:g}), {} stripes, {} conv calls in row blocks, {:.2f}s".format(
            err, SELFTEST_TOL, len(plan.stripes), stats.conv_chunked, time.perf_counter() - t0)
    return ok, detail
