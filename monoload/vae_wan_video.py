"""Layer-1 adapter: the Wan 2.1 / Wan 2.2 VAEs decoding video (several latent frames; docs/PLAN-video.md group A).

The single-frame adapters (monoload/vae_wan.py, monoload/vae_wan22.py) cover T = 1. For T > 1 both
WanVAE.decode loops run the decoder chunk by chunk along time, causally (ComfyUI v0.46.3):

  Wan 2.1 (comfy.ldm.wan.vae)     x = conv2(z); chunks of latent frames [0], [1, 3), [3, 5), ...; each chunk
                                  runs decoder.conv1, middle (per-frame attention), upsamples, head with one
                                  feat_cache list shared by all chunks; Decoder3d.run_up splits the frames after a
                                  temporal Resample ("upsample3d") into pieces of 2 (feat_idx copied per piece);
                                  the output pieces are collected in a list and concatenated.
  Wan 2.2 (comfy.ldm.wan.vae2_2)  x = conv2(z); one latent frame per chunk (first_chunk=True on the first);
                                  each chunk's output is concatenated onto the previous ones; unpatchify at the end.

Between chunks only feat_cache travels: for every CausalConv3d the last (up to) 2 frames of its input
(Wan 2.2: cloned, padded to 2 with the previous chunk's last frame), and a temporal Resample's "Rep" marker on
the first chunk. Every one of those cached tensors is local in the rows the way the conv's input is (a 3x3
conv's input rows, a 1x1 time_conv's input rows), and the per-frame attention sits in the middle, which the
single-frame adapters already put into the whole-image prefix.

So the decode runs stripe-major (docs/PLAN-video.md H2):

  1. prefix: conv2 on the whole latent, then for each time chunk the prefix modules of the single-frame adapter
     (decoder.conv1, middle, Wan 2.1 also the ResidualBlocks before the first Resample) with the prefix's own
     feat_cache, at latent resolution; each chunk's output goes straight into the checkpoint
     [1, C, T, h, w] (T latent frames: ~100 MB at 832x480x81 for Wan 2.1, bf16).
  2. stripes: for each output row stripe (the single-frame adapter's units, halos and stripe plan: the time
     kernels do not change the spatial halo), a feat_cache list of the stripe's own; all time chunks run in order
     through the units on exactly the stripe's needed rows of the checkpoint, the module instances called with
     the cache exactly as the native decoder calls them (Wan 2.1's 2-frame split included). A unit always runs on
     the same row range, so its cache entries cover the same rows in every chunk: the rows a stripe keeps are
     exact for the same reason as in the single-frame case (valid_out checks every unit), the inexact halo rows
     next to an inner stripe edge are dropped.
  3. each output piece (frames x the stripe's core rows) is written straight into the preallocated 5D output
     (H3: no list, no torch.cat).

Peak = checkpoint + the largest unit of one stripe on one chunk + that stripe's caches (2 frames of every cached
conv input at the stripe's rows) + the output (its own segment, as for images). Only the checkpoint and the
output grow with the number of frames.

The adapter subclasses the engine's StripeAdapter: the self-test (vae_engine.self_test) builds the engine's
image Plan from the units for the geometry and calls prefix_pass / stripe_pass, which are the video ones here;
plan() returns a VideoPlan (the video memory model), run() writes the 5D output directly.
"""

import torch

import comfy.ldm.wan.vae as wan
import comfy.ldm.wan.vae2_2 as wan22
import comfy.model_management as mm

from . import vae_engine as eng
from . import vae_ops
from . import vae_wan
from . import vae_wan22
from .vae_engine import CONV, POINT, RES, UP, OpChunking, conv_io
from .vae_ops import MIB, fmt_bytes, grouped_norm_bytes
from .vae_wan22 import DUPUP, UNPATCH

CACHE_T = 2
SELFTEST_FRAMES = {"2.1": 5, "2.2": 3}   # latent frames of the self-test: 3 time chunks each (the caches cross 2 chunk edges)


# ---------------------------------------------------------------------------
# structure (on top of the single-frame checks)
# ---------------------------------------------------------------------------

def _time_conv(rs, cin, name):
    """A temporal Resample's time_conv: CausalConv3d (kt, 1, 1), stride 1, cin -> 2 cin, no spatial padding."""
    tc = getattr(rs, "time_conv", None)
    if type(tc) is not wan.CausalConv3d:
        return "{}.time_conv is {}".format(name, type(tc).__name__)
    ks = tuple(tc.kernel_size)
    if ks[1:] != (1, 1) or tuple(tc.stride) != (1, 1, 1) or tuple(tc.dilation) != (1, 1, 1) or tc.groups != 1:
        return "{}.time_conv kernel {} stride {}".format(name, ks, tuple(tc.stride))
    if tuple(tc.padding) != (0, 0, 0) or tc.padding_mode != "zeros":
        return "{}.time_conv padding {} {}".format(name, tuple(tc.padding), tc.padding_mode)
    if conv_io(tc) != (cin, 2 * cin):
        return "{}.time_conv channels {} on {}".format(name, conv_io(tc), cin)
    return None


def _check_video_21(fsm, units):
    for u in units:
        if u.kind == UP and u.module.mode == "upsample3d":
            e = _time_conv(u.module, u.cin, u.name)
            if e:
                return e
    if not any(u.kind == UP and u.module.mode == "upsample3d" for u in units):
        return "no temporal Resample after the prefix"
    return None


def _check_video_22(fsm, units):
    for u in units:
        if u.kind == DUPUP:
            blk = u.module.blk
            rs = list(blk.upsamples)[-1]
            if rs.mode == "upsample3d":
                e = _time_conv(rs, u.cout, u.name)
                if e:
                    return e
                if blk.avg_shortcut.factor_t != 2:
                    return "{}: temporal Resample with DupUp3D factor_t {}".format(u.name, blk.avg_shortcut.factor_t)
            elif blk.avg_shortcut.factor_t != 1:
                return "{}: spatial Resample with DupUp3D factor_t {}".format(u.name, blk.avg_shortcut.factor_t)
    return None


def _temporal(u):
    """Whether unit u doubles the frames (on every chunk but the first)."""
    if u.kind == UP:
        return u.module.mode == "upsample3d"
    if u.kind == DUPUP:
        return list(u.module.blk.upsamples)[-1].mode == "upsample3d"
    return False


def _time_kernel(fsm):
    return int(fsm.decoder.conv1.weight.shape[2])


# ---------------------------------------------------------------------------
# memory model (bytes)
# ---------------------------------------------------------------------------
# Counted from the v0.46.3 forward code as in vae_wan / vae_wan22, with f frames per call. A, B: input / output
# channels on the rows a module runs on, all frames of the call; e element size; s the storage of the unit's
# input (held by the caller). The cached conv inputs are not in the phases: every slot of the stripe's (or the
# prefix's) cache is counted on its own, at 2 frames (cache_bytes), for the whole time loop.
#   causal conv (kt > 1)  torch.cat([zeros, cache, x]) -> (f + kt - 1) frames, then the conv: vol2col columns
#                         cin * kt * kh * kw per output position (layer-2 row blocks when above the workspace:
#                         one block of columns, its input and output), the output; Wan 2.2 first clones the new
#                         cache (2 frames).

def conv3_extra(cin, cout, kt, k, r, w, f, e, ws, clone=False):
    """Bytes a causal Conv3d (time kernel kt, spatial k x k) on f frames of r rows needs besides input and output."""
    ft = f + kt - 1 if kt > 1 else f
    cat = ft * r * w * cin * e if kt > 1 else 0
    per_row = cin * kt * k * k * w * f * e
    wcopy = cout * cin * kt * k * k * e
    full = per_row * r
    if full <= ws:
        cols = full
    else:
        rb = max(1, ws // per_row)
        cols = rb * per_row + (rb + k - 1) * w * ft * cin * e + rb * w * f * cout * e
    return cat + cols + wcopy + (CACHE_T * r * w * cin * e if clone else 0)


def res_peak_v(cin, cout, shortcut, kt, r, w, f, e, ws, s, clone):
    a, b = f * r * w * cin * e, f * r * w * cout * e
    n = r * w * f
    phases = [2 * a + grouped_norm_bytes(cin, n, e),
              a + b + conv3_extra(cin, cout, kt, 3, r, w, f, e, ws, clone),
              3 * b + grouped_norm_bytes(cout, n, e),
              2 * b + conv3_extra(cout, cout, kt, 3, r, w, f, e, ws, clone),
              2 * b]
    if shortcut:
        phases += [2 * b + vae_ops_conv1(cin, cout, r, w, f, e, ws), 3 * b]
    return s + max(phases)


def vae_ops_conv1(cin, cout, r, w, f, e, ws):
    """A 1x1x1 Conv3d shortcut (no time padding): its input copy, columns, output."""
    return f * r * w * cin * e * 2 + cin * cout * e


def resample_peak_v(cin, cout, temporal, kt, r, w, f, e, ws, s, clone):
    """A Resample (nearest x2 + 3x3 conv cin -> cout) on f frames; temporal: time_conv first (cin -> 2 cin, then
    the frames interleaved: 2 f frames)."""
    a1 = r * w * cin * e
    a = f * a1
    phases = []
    f2 = f
    if temporal:
        phases.append(a + conv3_extra(cin, 2 * cin, kt, 1, r, w, f, e, ws, clone) + 2 * a)
        phases.append(a + 4 * a)                  # time_conv out + the stacked copy
        f2 = 2 * f
    a2 = f2 * a1
    b2 = 4 * f2 * r * w * cout * e
    per_row = 9 * cin * 2 * w * f2 * e
    full = per_row * 2 * r
    cols = full if full <= ws else (max(1, ws // per_row) * per_row + (max(1, ws // per_row) + 2) * 2 * w * f2 * cin * e
                                    + max(1, ws // per_row) * 2 * w * f2 * cout * e)
    phases += [2 * a2, a2 + 4 * a2, 4 * a2 + b2 + cols + 9 * cin * cout * e, 2 * b2]
    return s + max(phases)


def dupup_peak_v(u, kt, r, w, f, e, ws, s, first, clone):
    """An Up_ResidualBlock (k ResidualBlocks, the Resample, the DupUp3D shortcut on its input x, which stays alive)."""
    blocks, ft = u.extra["blocks"], u.extra["ft"]
    temporal = ft == 2 and not first
    phases = []
    b = f * r * w * u.cout * e
    for j, (ci, co, sc) in enumerate(blocks):
        phases.append(res_peak_v(ci, co, sc, kt, r, w, f, e, ws, s if j == 0 else s + b, clone))
    phases.append(s + resample_peak_v(u.cout, u.cout, temporal, kt, r, w, f, e, ws, b, clone))
    f2 = 2 * f if temporal else f
    b1 = r * w * u.cout * e
    main = 4 * f2 * b1
    sc_rep = 4 * ft * f * b1           # repeat_interleave (all factor_t slices), its permuted copy
    phases.append(s + main + 2 * sc_rep)
    phases.append(s + 2 * main + sc_rep)
    return max(phases)


def unit_peak_v(u, kt, r, w, f, e, ws, s, first, clone):
    """Peak bytes while unit u runs on f frames of r input rows of width w, input storage s (caches not included)."""
    if u.kind == RES:
        return res_peak_v(u.cin, u.cout, u.shortcut, kt, r, w, f, e, ws, s, clone)
    if u.kind == UP:
        return resample_peak_v(u.cin, u.cout, _temporal(u) and not first, kt, r, w, f, e, ws, s, clone)
    if u.kind == DUPUP:
        return dupup_peak_v(u, kt, r, w, f, e, ws, s, first, clone)
    if u.kind == CONV:
        return s + f * r * w * u.cout * e + f * r * w * u.cin * e + conv3_extra(u.cin, u.cout, kt, 3, r, w, f, e, ws, clone)
    if u.kind == UNPATCH:
        return s + f * r * w * u.cin * e
    return s + 2 * f * r * w * u.cin * e + (grouped_norm_bytes(u.cin, r * w * f, e) if isinstance(u.module, wan.RMS_norm) else 0)


def unit_out_frames(u, f, first):
    return 2 * f if (_temporal(u) and not first) else f


def unit_largest_v(u, kt, r, w, f, e, ws, first):
    """Largest single allocation while unit u runs on f frames of r input rows of width w: the image adapters'
    bound with the frames as extra width (output frames for an upsampling unit), the time-padded copy of a
    causal conv's input, a block of conv columns."""
    fo = unit_out_frames(u, f, first)
    c = max(u.cin, u.cout)
    return max(4 * fo * r * w * c * e, (f + kt - 1) * r * w * c * e, min(ws, 9 * kt * c * r * w * f * e))


def unit_cache_channels(u):
    """Channels of the conv inputs unit u caches (one slot each; 2 frames at the unit's input rows / resolution)."""
    if u.kind == RES:
        return u.cin + u.cout
    if u.kind == UP:
        return u.cin if u.module.mode == "upsample3d" else 0
    if u.kind == DUPUP:
        n = sum(ci + co for ci, co, _ in u.extra["blocks"])
        return n + (u.cout if u.extra["ft"] == 2 else 0)
    if u.kind == CONV:
        return u.cin
    return 0


def prefix_cache_channels(m):
    if isinstance(m, (wan.ResidualBlock, wan22.ResidualBlock)):
        return m.in_dim + m.out_dim
    if isinstance(m, wan.CausalConv3d):
        return conv_io(m)[0] if int(m.kernel_size[0]) > 1 else 0
    return 0


def prefix_peak_v(m, kt, h, w, f, e, ws, s, clone):
    if isinstance(m, wan.AttentionBlock):
        return vae_wan.attention_peak(int(m.norm.gamma.shape[0]), h, w * f, e, ws, s, vae_wan.attn_split(m))
    if isinstance(m, (wan.ResidualBlock, wan22.ResidualBlock)):
        return res_peak_v(m.in_dim, m.out_dim, not isinstance(m.shortcut, torch.nn.Identity), kt, h, w, f, e, ws, s, clone)
    ci, co = conv_io(m)
    k = int(m.kernel_size[-1])
    return s + f * h * w * co * e + conv3_extra(ci, co, int(m.kernel_size[0]), k, h, w, f, e, ws, clone)


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------

class VideoPlan:
    """A stripe plan of a video decode: the output row stripes (engine.split_rows / stripe_needs on the units,
    exactly as for one frame) and the memory model over the time chunks. Has what vae.py reads from a plan:
    estimate, arena, stripes, recompute, ckpt_bytes, prefix_bytes, stripe_bytes, describe()."""

    def __init__(self, bound, t, h8, w8, rows, ws, elem, out_bytes):
        self.bound, self.workspace, self.elem, self.hdim = bound, ws, elem, bound.hdim
        units = bound.units
        self.final_units = units
        heights, widths = [h8], [w8]
        for u in units:
            heights.append(heights[-1] * u.scale)
            widths.append(widths[-1] * u.scale)
        self.final_heights, self.widths = heights, widths
        self.h_out, self.w_out = heights[-1], widths[-1]
        self.t = t
        self.stripes = eng.split_rows(self.h_out, rows)
        self.needs = [eng.stripe_needs(units, heights, a, b) for a, b in self.stripes]
        self.order = list(range(len(self.stripes)))
        self.rows = max(b - a for a, b in self.stripes)
        e = elem
        kt, clone = bound.kt, bound.clone_cache
        frames = bound.unit_frames()
        # the checkpoint: T latent frames at h8 x w8
        self.ckpt_bytes = bound.ckpt_channels * t * h8 * w8 * e
        # prefix: the latent and conv2's output (whole, all frames), the checkpoint, the prefix's caches, the
        # largest prefix module on one chunk (its input counted as a whole chunk of its own on top)
        zc = int(bound.fsm.conv2.weight.shape[0])
        pf = bound.prefix_frames
        pcache = sum(prefix_cache_channels(m) for _, m in bound.prefix) * CACHE_T * h8 * w8 * e
        s = pf * h8 * w8 * zc * e
        ppeak = plargest = 0
        for _, m in bound.prefix:
            ppeak = max(ppeak, prefix_peak_v(m, kt, h8, w8, pf, e, ws, s, clone))
            plargest = max(plargest, bound.prefix_largest(m, h8, w8 * (pf + kt - 1), e, ws))
            s = pf * h8 * w8 * bound.prefix_out_channels(m) * e
        self.prefix_bytes = 2 * t * h8 * w8 * zc * e + self.ckpt_bytes + pcache + ppeak
        # stripes: the largest over stripes of (caches of the stripe + the largest unit on one chunk)
        live, largest, rows_total = 0, plargest, 0
        for needs in self.needs:
            cache = peak = 0
            for i, u in enumerate(units):
                ra, rb = needs[i]
                r = rb - ra
                w = widths[i]
                cache += unit_cache_channels(u) * CACHE_T * r * w * e
                for f, first, fs in frames[i]:
                    if i == 0:
                        s = 0       # a slice of the checkpoint (counted on its own)
                    else:
                        pa, pb = needs[i - 1]
                        s = fs * (pb - pa) * units[i - 1].scale * w * units[i - 1].cout * e
                    peak = max(peak, unit_peak_v(u, kt, r, w, f, e, ws, s, first, clone))
                    largest = max(largest, unit_largest_v(u, kt, r, w, f, e, ws, first))
            live = max(live, cache + peak)
            rows_total += needs[0][1] - needs[0][0]
        self.stripe_bytes = live
        self.live = max(self.prefix_bytes, self.ckpt_bytes + live)
        self.recompute = rows_total / float(h8) - 1.0
        self.arena = eng.arena_bytes(self.live, stripes=len(self.stripes))
        self.largest = largest
        self.out_bytes = out_bytes
        self.estimate = int(eng.out_segment(out_bytes) + self.arena + largest + eng.ESTIMATE_PAD)
        self.passes = ()

    def describe(self):
        from .messages import msg
        return msg("vae.plan", n=len(self.stripes), rows=self.rows, rec=self.recompute, ckpt=fmt_bytes(self.ckpt_bytes)) + \
            " x {} latent frames".format(self.t)


# ---------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------

class WanVideoStripe(eng.StripeAdapter):
    """Video layer-1 adapter bound to one WanVAE (2.1 or 2.2); `base` is the single-frame adapter (units, prefix,
    the cost model the engine's image Plan uses in the self-test)."""

    hdim = 3

    def __init__(self, base, fsm, module=None):
        self.base = base
        self.fsm = fsm
        self.module = fsm if module is None else module
        self.prefix, self.units = base.prefix, base.units
        self.scale = base.scale
        self.ckpt_channels = base.ckpt_channels
        self.out_channels = base.out_channels
        self.kt = _time_kernel(fsm)
        self.ncache = 1 + sum(1 for m in fsm.decoder.modules() if isinstance(m, wan.CausalConv3d))
        self.key = ("video",) + tuple(base.key) + (self.kt,)
        self.unit_peak = base.unit_peak
        self.unit_largest = base.unit_largest
        self.prefix_peak = base.prefix_peak
        self.prefix_largest = base.prefix_largest
        self.prefix_out_channels = base.prefix_out_channels
        self.prefix_macs = base.prefix_macs
        self.activation_copies = base.activation_copies
        self.selftest_size = base.selftest_size

    # --- time schedule

    def chunks(self, t):
        """[(first latent frame, frames)] of the native time loop."""
        raise NotImplementedError

    split_temporal = False   # Wan 2.1: Decoder3d.run_up splits the frames after a temporal Resample into pieces of 2

    def unit_frames(self):
        """Per unit, [(input frames, first chunk, frames of the storage the input is a slice of)]: a chunk after the
        first at its largest, and the first chunk."""
        out = []
        cur = [(self.chunk_frames, False), (1, True)]
        store = [self.chunk_frames, 1]
        for u in self.units:
            out.append([(f, first, fs) for (f, first), fs in zip(cur, store)])
            nxt = [unit_out_frames(u, f, first) for f, first in cur]
            store = list(nxt)
            if self.split_temporal and u.kind == UP and u.module.mode == "upsample3d":
                nxt = [min(n, 2) for n in nxt]
            cur = [(n, first) for n, (_, first) in zip(nxt, cur)]
        return out

    def out_frames(self, t):
        n = 0
        for c0, k in self.chunks(t):
            f = k
            for u in self.units:
                f = unit_out_frames(u, f, c0 == 0)
            n += f
        return n

    def output_shape(self, samples):
        return (samples.shape[0], self.out_channels, self.out_frames(int(samples.shape[2])),
                int(samples.shape[-2]) * self.scale, int(samples.shape[-1]) * self.scale)

    # --- planning

    def plan(self, vae, samples, budget, ws, rows=None, out_bytes=0, measure="estimate"):
        h8, w8 = self.decoder_hw(samples)
        t = int(samples.shape[2])
        elem = mm.dtype_size(vae.vae_dtype)
        if rows is not None:
            return VideoPlan(self, t, h8, w8, rows, ws, elem, out_bytes)
        h_out = h8 * self.scale
        lo = min(eng.MIN_ROWS, h_out)
        while True:
            best = VideoPlan(self, t, h8, w8, lo, ws, elem, out_bytes)
            if getattr(best, measure) <= budget:
                break
            if lo >= h_out:
                return None
            lo = min(h_out, lo * 2)
        lo, hi = lo + 1, h_out
        while lo <= hi:
            mid = (lo + hi) // 2
            p = VideoPlan(self, t, h8, w8, mid, ws, elem, out_bytes)
            if getattr(p, measure) <= budget:
                best, lo = p, mid + 1
            else:
                hi = mid - 1
        return best

    # --- running

    def prefix_pass(self, z):
        """conv2 on the whole latent, then the prefix chunk by chunk with its own feat_cache -> the checkpoint
        [1, C, T, h, w] (each chunk written into it)."""
        x = self.module_conv2()(z)
        t = int(z.shape[2])
        ckpt = torch.empty((z.shape[0], self.ckpt_channels, t) + tuple(z.shape[-2:]), dtype=x.dtype, device=x.device)
        cache = [None] * self.ncache
        for c0, k in self.chunks(t):
            y = self.prefix_chunk(x[:, :, c0:c0 + k], cache, c0 == 0)
            ckpt[:, :, c0:c0 + k].copy_(y)
            del y
        del x, cache
        return ckpt

    def module_conv2(self):
        return self.fsm.conv2

    def stripe_pass(self, saves, plan, write):
        """The self-test's stripes (engine Plan): each stripe's frames into a buffer, then write(o0, o1, rows)."""
        ckpt = saves.pop(0)
        t = int(ckpt.shape[2])
        tout = self.out_frames(t)
        for k in plan.order:
            (o0, o1), needs = plan.stripes[k], plan.needs[k]
            buf = None

            def emit(t0, x):
                nonlocal buf
                if buf is None:
                    buf = torch.empty(x.shape[:2] + (tout,) + x.shape[3:], dtype=x.dtype, device=x.device)
                buf.narrow(2, t0, x.shape[2]).copy_(x)
            self.stripe(ckpt, needs, plan.final_heights, emit, "stripe [{}, {})".format(o0, o1))
            write(o0, o1, buf)
            del buf

    def stripe(self, ckpt, needs, heights, emit, label):
        """All time chunks of one stripe; emit(t0, x) gets each output piece (frames from t0, the stripe's core rows)."""
        cache = [None] * self.ncache
        tpos = [0]

        def out(x):
            emit(tpos[0], x)
            tpos[0] += int(x.shape[2])
        xa, xb = needs[0]
        for c0, k in self.chunks(int(ckpt.shape[2])):
            x = ckpt.narrow(2, c0, k).narrow(self.hdim, xa, xb - xa)
            self.run_units(0, x, needs, heights, cache, [0], c0 == 0, out, label)
            del x

    def _step(self, i, y, needs, heights, xa, xb, label):
        """Unit i's output rows -> the next needed rows (checked exact), contiguous before an upsampling unit."""
        u = self.units[i]
        hd = self.hdim
        if y.shape[hd] != (xb - xa) * u.scale:
            raise eng.StripeError("[Monoload] internal error: {} returned {} rows for {} input rows".format(u.name, y.shape[hd], xb - xa))
        va, vb = eng.valid_out(u, xa, xb, heights[i + 1])
        ta, tb = needs[i + 1]
        if ta < va or tb > vb:
            raise eng.StripeError("[Monoload] internal error: {} needs rows [{}, {}) of {} but only [{}, {}) are exact".format(
                label, ta, tb, u.name, va, vb))
        x = y.narrow(hd, ta - xa * u.scale, tb - ta)
        if i + 1 < len(self.units) and (self.units[i + 1].kind in eng.CONTIGUOUS_INPUT or self.units[i + 1].contiguous) \
                and not x.is_contiguous():
            x = x.contiguous()
        return x

    def run_units(self, i, x, needs, heights, cache, idx, first, out, label):
        raise NotImplementedError

    # --- the decode

    def run(self, vae, samples_in, plan, budget_ws, stats):
        """Decode every sample (the engine's memory layout: cache emptied, the output in its own segment, the arena);
        the stripes write their pieces straight into the output."""
        n = samples_in.shape[0]
        hd = self.hdim
        mm.soft_empty_cache()
        with vae_ops.malloc_graph_paused():
            pixel_samples = torch.empty(self.output_shape(samples_in), device=vae.output_device, dtype=vae.vae_output_dtype())
        arena = bool(plan.arena) and eng.arena_supported(vae.device)
        if arena:
            eng.reserve_arena(vae.device, plan.arena)
            stats.arena = plan.arena
        with OpChunking(self.module, budget_ws, stats):
            for i in range(n):
                z = samples_in[i:i + 1].to(device=vae.device, dtype=vae.vae_dtype)
                ckpt = self.prefix_pass(z)
                del z
                if arena and eng.CKPT_LOW:
                    ckpt = eng.move_low(ckpt, vae.device)
                dst = pixel_samples[i:i + 1]
                for k in plan.order:
                    (o0, o1), needs = plan.stripes[k], plan.needs[k]

                    def emit(t0, x, o0=o0, o1=o1, dst=dst):
                        dst.narrow(2, t0, x.shape[2]).narrow(hd, o0, o1 - o0).copy_(x)
                    self.stripe(ckpt, needs, plan.final_heights, emit, "stripe [{}, {})".format(o0, o1))
                del ckpt
                vae.process_output(pixel_samples[i:i + 1])
        mm.soft_empty_cache()
        return pixel_samples

    # --- self-test

    def selftest_latent(self, n):
        return (1, int(self.fsm.conv2.weight.shape[1]), SELFTEST_FRAMES[self.version], n, n)

    def selftest_params(self):
        return sum(p.numel() for p in self.fsm.decoder.parameters()) + sum(p.numel() for p in self.fsm.conv2.parameters())

    def selftest_memory(self):
        """Upper bound of the self-test (fp32, small, SELFTEST_FRAMES latent frames): the weights in fp32, then the
        larger of the reference (the native video decode under layer-2 chunking: every cached conv input of the
        whole image, activation_copies x the largest activation of a chunk, 2 x the workspace, the prefix, its
        output 3 x: the pieces, torch.cat, Wan 2.2's unpatchify) and the stripes (the reference's output, the
        stripes' output, one stripe's buffer, their plan's estimate), a margin."""
        key = ("selftest_memory", self.key)
        hit = eng._PASS_COST.get(key)
        if hit is not None:
            return hit
        n = self.selftest_size
        t = SELFTEST_FRAMES[self.version]
        plan = VideoPlan(self, t, n, n, eng.SELFTEST_ROWS, eng.SELFTEST_WORKSPACE, 4, 0)
        out = self.out_channels * self.out_frames(t) * plan.h_out * plan.w_out * 4
        cache = act = 0
        for i, (u, fr) in enumerate(zip(self.units, self.unit_frames())):
            h, w = plan.final_heights[i], plan.widths[i]
            cache += unit_cache_channels(u) * CACHE_T * h * w * 4
            f = max(max(unit_out_frames(u, fi, first), fs) for fi, first, fs in fr)
            act = max(act, 4 * f * h * w * max(u.cin, u.cout) * 4)
        ref = self.activation_copies * act + cache + 3 * out + 2 * eng.SELFTEST_REF_WORKSPACE + plan.prefix_bytes
        stripes = 3 * out + plan.estimate
        need = int(self.selftest_params() * 4 + max(ref, stripes) + eng.SELFTEST_MARGIN)
        eng._PASS_COST[key] = need
        return need

    def fp32_copy(self, device):
        raise NotImplementedError

    def reference_decode(self, z):
        raise NotImplementedError


class Wan21Video(WanVideoStripe):
    name = "Wan 2.1 video stripes"
    version = "2.1"
    clone_cache = False
    split_temporal = True
    chunk_frames = 2       # latent frames of a chunk after the first
    prefix_frames = 2

    def chunks(self, t):
        return [(0, 1)] + [(c, min(2, t - c)) for c in range(1, t, 2)]

    def prefix_chunk(self, x, cache, first):
        """Decoder3d.forward's conv1 and middle, then run_up's layers before the first Resample, with the cache."""
        dec = self.fsm.decoder
        idx = [0]
        cache_x = x[:, :, -CACHE_T:, :, :]
        x = dec.conv1(x, cache[idx[0]])
        cache[idx[0]] = cache_x
        idx[0] += 1
        del cache_x
        for _, m in self.prefix[2:]:
            x = m(x, cache, idx)
        return x

    def call(self, u, x, cache, idx, first):
        if u.kind in (RES, UP):
            return u.module(x, cache, idx)
        if u.kind == CONV:
            cache_x = x[:, :, -CACHE_T:, :, :]
            y = u.module(x, cache[idx[0]])
            cache[idx[0]] = cache_x
            idx[0] += 1
            return y
        return u.module(x)

    def run_units(self, i, x, needs, heights, cache, idx, first, out, label):
        while i < len(self.units):
            u = self.units[i]
            xa, xb = needs[i]
            y = self.call(u, x, cache, idx, first)
            del x
            x = self._step(i, y, needs, heights, xa, xb, label)
            del y
            if u.kind == UP and u.module.mode == "upsample3d" and x.shape[2] > 2:
                # Decoder3d.run_up: pieces of 2 frames, each with its own copy of feat_idx
                for f0 in range(0, x.shape[2], 2):
                    piece = x[:, :, f0:f0 + 2]
                    if i + 1 < len(self.units) and self.units[i + 1].kind in eng.CONTIGUOUS_INPUT:
                        piece = piece.contiguous()
                    self.run_units(i + 1, piece, needs, heights, cache, list(idx), first, out, label)
                    del piece
                return
            i += 1
        out(x)

    def fp32_copy(self, device):
        """conv2 + decoder rebuilt in fp32 with the model's weights (full time kernels: video reads them all)."""
        fsm = self.fsm
        dec = fsm.decoder
        holder = torch.nn.Module()
        z = int(fsm.conv2.weight.shape[0])
        out_c = int(dec.head[2].weight.shape[0])
        with eng.skeleton():
            holder.conv2 = wan.CausalConv3d(z, z, 1)
            holder.decoder = wan.Decoder3d(dec.dim, dec.z_dim, out_c, list(dec.dim_mult), dec.num_res_blocks,
                                           list(dec.attn_scales), list(dec.temperal_upsample), 0.0)
        holder.to_empty(device=device)
        holder.to(dtype=torch.float32)
        holder.conv2.load_state_dict(fsm.conv2.state_dict(), strict=True)
        holder.decoder.load_state_dict(dec.state_dict(), strict=True)
        holder.eval()
        shim = vae_wan._Shim(holder)
        reason, info = vae_wan.wan_structure(shim)
        if reason:
            raise eng.StripeError("fp32 copy does not match: " + reason)
        return Wan21Video(vae_wan.WanStripe(shim, info["first_resample"], module=holder), shim, module=holder)

    def reference_decode(self, z):
        return wan.WanVAE.decode(self.module, z)


class Wan22Video(WanVideoStripe):
    name = "Wan 2.2 video stripes"
    version = "2.2"
    clone_cache = True
    chunk_frames = 1
    prefix_frames = 1

    def chunks(self, t):
        return [(c, 1) for c in range(t)]

    @staticmethod
    def _cached_conv(conv, x, cache, idx):
        """vae2_2's pattern for a CausalConv3d with the cache (decoder.conv1, the head conv): the new cache is the
        input's last 2 frames, cloned, padded in front with the previous cache's last frame."""
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and cache[idx[0]] is not None:
            cache_x = torch.cat([cache[idx[0]][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        y = conv(x, cache[idx[0]])
        cache[idx[0]] = cache_x
        idx[0] += 1
        return y

    def prefix_chunk(self, x, cache, first):
        dec = self.fsm.decoder
        idx = [0]
        x = self._cached_conv(dec.conv1, x, cache, idx)
        for m in dec.middle:
            x = m(x, cache, idx) if isinstance(m, wan22.ResidualBlock) else m(x)
        return x

    def call(self, u, x, cache, idx, first):
        if u.kind == DUPUP:
            return u.module.blk(x, cache, idx, first)
        if u.kind == RES:
            return u.module(x, cache, idx)
        if u.kind == CONV:
            return self._cached_conv(u.module, x, cache, idx)
        return u.module(x)

    def run_units(self, i, x, needs, heights, cache, idx, first, out, label):
        for j in range(i, len(self.units)):
            xa, xb = needs[j]
            y = self.call(self.units[j], x, cache, idx, first)
            del x
            x = self._step(j, y, needs, heights, xa, xb, label)
            del y
        out(x)

    def fp32_copy(self, device):
        """conv2 + decoder rebuilt in fp32 with the model's weights (full time kernels)."""
        fsm = self.fsm
        dec = fsm.decoder
        holder = torch.nn.Module()
        holder.patch_size = vae_wan22.patch_size(fsm)
        z = int(fsm.conv2.weight.shape[0])
        with eng.skeleton():
            holder.conv2 = wan.CausalConv3d(z, z, 1)
            holder.decoder = wan22.Decoder3d(dec.dim, dec.z_dim, list(dec.dim_mult), dec.num_res_blocks, list(dec.attn_scales),
                                             list(dec.temperal_upsample), 0.0, out_channels=conv_io(dec.head[2])[1],
                                             temporal_kernel=int(dec.conv1.weight.shape[2]))
        holder.to_empty(device=device)
        holder.to(dtype=torch.float32)
        holder.conv2.load_state_dict(fsm.conv2.state_dict(), strict=True)
        holder.decoder.load_state_dict(dec.state_dict(), strict=True)
        holder.eval()
        shim = vae_wan22._Shim(holder)
        reason, _ = vae_wan22.wan22_structure(shim)
        if reason:
            raise eng.StripeError("fp32 copy does not match: " + reason)
        return Wan22Video(vae_wan22.Wan22Stripe(shim, module=holder), shim, module=holder)

    def reference_decode(self, z):
        return wan22.WanVAE.decode(self.module, z)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def match(vae, samples, vae_options):
    """(bound adapter, None) or (None, reason): a Wan 2.1 / 2.2 VAE decoding a latent of several frames."""
    if samples.ndim != 5 or samples.shape[2] < 2:
        return None, "video: latent is not several frames ({})".format(list(samples.shape))
    fsm = vae.first_stage_model
    if vae_options:
        return None, "video: vae_options {} given".format(sorted(vae_options))
    reason21, info = vae_wan.wan_structure(fsm)
    if reason21 is None:
        base = vae_wan.WanStripe(fsm, info["first_resample"])
        e = _check_video_21(fsm, base.units)
        if e:
            return None, "video (Wan 2.1): " + e
        cls = Wan21Video
    else:
        reason22, _ = vae_wan22.wan22_structure(fsm)
        if reason22 is not None:
            return None, "video: {}; {}".format(reason21, reason22)
        base = vae_wan22.Wan22Stripe(fsm)
        e = _check_video_22(fsm, base.units)
        if e:
            return None, "video (Wan 2.2): " + e
        cls = Wan22Video
    if samples.shape[1] != fsm.conv2.weight.shape[1]:
        return None, "video: latent has {} channels, conv2 expects {}".format(samples.shape[1], fsm.conv2.weight.shape[1])
    return cls(base, fsm), None


__all__ = ["match", "Wan21Video", "Wan22Video", "VideoPlan", "MIB"]
