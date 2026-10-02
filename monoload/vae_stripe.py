"""Layer 1: stripe decoding of a recognized decoder (docs/DESIGN.md §9.13).

Phase 2 covers one structure: the Wan 2.1 VAE (comfy.ldm.wan.vae.WanVAE with
Decoder3d; qwen_image_vae) decoding a single frame. In 62b3c94 a T=1 decode is
a plain sequence of module calls (feat_map is None, CausalConv3d takes its
autopad="causal_zero" fast path, run_up never splits frames):

    conv2 -> decoder.conv1 -> decoder.middle[*] -> decoder.upsamples[*] -> decoder.head[*]

The modules up to the last one before the first Resample (conv1, middle with
the global attention, the ResidualBlocks of the lowest resolution) run on the
whole image; their output is the checkpoint (H/8, 384 channels: ~0.1 GiB at
4K). Everything after it is local in H -- ResidualBlocks (two 3x3 convs, RMS
norm per position), Resample (nearest x2 + 3x3 conv), RMS/SiLU/3x3 conv of the
head -- so the output is produced in stripes of rows. For a stripe [o0, o1) the
rows each unit needs are worked out backwards (need_in), then the units run
forwards on exactly those rows of the checkpoint.

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
"""

import gc
import time

import torch

import comfy.model_management as mm
import comfy.ldm.wan.vae as wan
from comfy.ldm.modules.diffusionmodules import model as ldm_model

from .errors import MonoloadError
from .vae_ops import MIB, OpChunking, OpStats, fmt_bytes

HDIM = 3                     # rows in [B, C, T, H, W]
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
        self.shortcut = shortcut   # ResidualBlock with a 1x1 conv shortcut


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


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def _t3(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v, v)


def _causal_conv(m, k):
    """A CausalConv3d with spatial kernel k (3 or 1), stride 1, the padding its
    constructor gives (spatial k//2, time handled by autopad), zeros mode."""
    if type(m) is not wan.CausalConv3d:
        return "not a CausalConv3d ({})".format(type(m).__name__)
    ks = tuple(m.kernel_size)
    if ks[1:] != (k, k) or ks[0] not in (1, k):
        return "kernel {} (expected {}x{})".format(ks, k, k)
    if tuple(m.stride) != (1, 1, 1) or tuple(m.dilation) != (1, 1, 1) or m.groups != 1 or m.padding_mode != "zeros":
        return "stride {} dilation {} groups {} padding_mode {}".format(tuple(m.stride), tuple(m.dilation), m.groups, m.padding_mode)
    if tuple(m.padding) != (0, k // 2, k // 2):
        return "padding {} (expected (0, {}, {}))".format(tuple(m.padding), k // 2, k // 2)
    return None


def _rms(m):
    if type(m) is not wan.RMS_norm or not m.channel_first:
        return "not a channel-first RMS_norm ({})".format(type(m).__name__)
    return None


def _no_dropout(m):
    if type(m) is not torch.nn.Dropout:
        return "not a Dropout ({})".format(type(m).__name__)
    if m.training and m.p > 0:
        return "Dropout p={} in training mode".format(m.p)
    return None


def _check_residual(rb, name):
    if type(rb) is not wan.ResidualBlock:
        return "{} is {}, not a ResidualBlock".format(name, type(rb).__name__)
    r = list(rb.residual)
    kinds = [type(x) for x in r]
    want = [wan.RMS_norm, torch.nn.SiLU, wan.CausalConv3d, wan.RMS_norm, torch.nn.SiLU, torch.nn.Dropout, wan.CausalConv3d]
    if kinds != want:
        return "{}.residual is [{}]".format(name, ", ".join(k.__name__ for k in kinds))
    for i, chk in ((0, _rms), (3, _rms), (5, _no_dropout)):
        e = chk(r[i])
        if e:
            return "{}.residual[{}]: {}".format(name, i, e)
    for i in (2, 6):
        e = _causal_conv(r[i], 3)
        if e:
            return "{}.residual[{}]: {}".format(name, i, e)
    if isinstance(rb.shortcut, torch.nn.Identity):
        if rb.in_dim != rb.out_dim:
            return "{}: identity shortcut with {} -> {} channels".format(name, rb.in_dim, rb.out_dim)
    else:
        e = _causal_conv(rb.shortcut, 1)
        if e:
            return "{}.shortcut: {}".format(name, e)
    return None


def _check_resample(rs, name):
    if type(rs) is not wan.Resample:
        return "{} is {}".format(name, type(rs).__name__)
    if rs.mode not in ("upsample2d", "upsample3d"):
        return "{} mode {}".format(name, rs.mode)
    seq = list(rs.resample)
    if len(seq) != 2 or type(seq[0]) is not torch.nn.Upsample or not isinstance(seq[1], torch.nn.Conv2d):
        return "{}.resample is [{}]".format(name, ", ".join(type(x).__name__ for x in seq))
    up, conv = seq
    sf = up.scale_factor if isinstance(up.scale_factor, tuple) else (up.scale_factor, up.scale_factor)
    if up.mode not in ("nearest-exact", "nearest") or tuple(float(s) for s in sf) != (2.0, 2.0) or up.size is not None:
        return "{} upsample mode {} scale {}".format(name, up.mode, up.scale_factor)
    if (tuple(conv.kernel_size) != (3, 3) or tuple(conv.stride) != (1, 1) or tuple(conv.padding) != (1, 1)
            or tuple(conv.dilation) != (1, 1) or conv.groups != 1 or conv.padding_mode != "zeros"):
        return "{} conv kernel {} stride {} padding {}".format(name, conv.kernel_size, conv.stride, conv.padding)
    return None


def _hooks(model):
    """Modules whose call would not be their class' forward (forward hooks,
    instance-level forward replacements such as bypass injections)."""
    for name, m in model.named_modules():
        if "forward" in m.__dict__ or m._forward_hooks or m._forward_pre_hooks:
            return name or "<root>"
    if torch.nn.modules.module._global_forward_hooks or torch.nn.modules.module._global_forward_pre_hooks:
        return "<global module hooks>"
    return None


def wan_structure(fsm):
    """(None, info) if fsm is a Wan 2.1 VAE the stripe decoder handles, else (reason, None)."""
    if not _is_wan(fsm):
        return "first-stage model is {}, not comfy.ldm.wan.vae.WanVAE".format(type(fsm).__name__), None
    dec = getattr(fsm, "decoder", None)
    if type(dec) is not wan.Decoder3d:
        return "decoder is {}, not Decoder3d".format(type(dec).__name__), None
    e = _causal_conv(fsm.conv2, 1) or _causal_conv(dec.conv1, 3)
    if e:
        return "conv2 / decoder.conv1: " + e, None
    ups = list(dec.upsamples)
    first = next((i for i, m in enumerate(ups) if isinstance(m, wan.Resample)), None)
    if first is None:
        return "decoder.upsamples has no Resample", None
    for i, m in enumerate(ups):
        if isinstance(m, wan.AttentionBlock):
            return "decoder.upsamples[{}] is an AttentionBlock".format(i), None
        e = _check_resample(m, "upsamples[{}]".format(i)) if isinstance(m, wan.Resample) else _check_residual(m, "upsamples[{}]".format(i))
        if e:
            return e, None
    head = list(dec.head)
    if [type(x) for x in head] != [wan.RMS_norm, torch.nn.SiLU, wan.CausalConv3d]:
        return "decoder.head is [{}]".format(", ".join(type(x).__name__ for x in head)), None
    e = _rms(head[0]) or _causal_conv(head[2], 3)
    if e:
        return "decoder.head: " + e, None
    for i, m in enumerate(dec.middle):
        if not isinstance(m, (wan.ResidualBlock, wan.AttentionBlock)):
            return "decoder.middle[{}] is {}".format(i, type(m).__name__), None
    h = _hooks(fsm)
    if h:
        return "module {} has a forward hook or an instance-level forward".format(h), None
    return None, {"first_resample": first}


def _conv_io(m):
    w = m.weight
    return int(w.shape[1]), int(w.shape[0])


def build_units(fsm, first_resample):
    """(prefix modules, stripe units)."""
    dec = fsm.decoder
    ups = list(dec.upsamples)
    prefix = [("conv2", fsm.conv2), ("decoder.conv1", dec.conv1)]
    prefix += [("decoder.middle.{}".format(i), m) for i, m in enumerate(dec.middle)]
    prefix += [("decoder.upsamples.{}".format(i), m) for i, m in enumerate(ups[:first_resample])]
    units = []
    for i, m in enumerate(ups[first_resample:], first_resample):
        name = "decoder.upsamples.{}".format(i)
        if isinstance(m, wan.Resample):
            cin, cout = _conv_io(m.resample[1])
            units.append(Unit(UP, m, 1, 2, cin, cout, name, 9 * cin * cout))
        else:
            c1i, c1o = _conv_io(m.residual[2])
            c2i, c2o = _conv_io(m.residual[6])
            sc = not isinstance(m.shortcut, torch.nn.Identity)
            macs = 9 * (c1i * c1o + c2i * c2o) + (c1i * c2o if sc else 0)
            units.append(Unit(RES, m, 2, 1, c1i, c2o, name, macs, sc))
    rms, silu, conv = list(dec.head)
    c = int(rms.gamma.shape[0])
    cin, cout = _conv_io(conv)
    units.append(Unit(POINT, rms, 0, 1, c, c, "decoder.head.0"))
    units.append(Unit(POINT, silu, 0, 1, c, c, "decoder.head.1"))
    units.append(Unit(CONV, conv, 1, 1, cin, cout, "decoder.head.2", 9 * cin * cout))
    return prefix, units


def signature(fsm, units):
    """Structure key (self-test cache)."""
    dec = fsm.decoder
    return ("WanVAE", int(fsm.conv2.weight.shape[0]), tuple(dec.dim_mult), int(dec.dim), int(dec.num_res_blocks),
            tuple((u.kind, u.cin, u.cout) for u in units))


# ---------------------------------------------------------------------------
# memory model (bytes), conv work model
# ---------------------------------------------------------------------------
# Live tensors, counted from the 62b3c94 forward code (DESIGN §9.13.4). S is
# the storage of a module's input (held by the caller during the call: in the
# stripes the previous unit's whole output, of which the input is a row
# slice), A / B one plane of input / output channels on the rows the module
# runs on, e = element size.
#   RMS_norm       F.normalize(x) * scale * gamma + 0: two temporaries at a time   S + 2A
#   ResidualBlock  RMS 2A | SiLU out + conv1 out + conv1 extra | RMS on conv1 out 3B
#                  | conv2 in + out + extra | x + shortcut(old_x) 2B (3B and the
#                  1x1 conv's input copy + columns with a shortcut conv)        S + max(...)
#   Resample       contiguous copy A + upsampled 4A | 4A + conv out 4B + extra   S + max(...)
#   head conv      output + extra (its input is a row slice: copied)            S + B + extra
#   AttentionBlock (prefix) norm out P + qkv 3P + attention out P + scores
#                  (split attention; q/k/v/out copies of SDPA / xformers: +3P)  S + 5P + scores
# "extra" of a conv: the im2col / vol2col columns (bounded by the workspace when
# the conv runs in layer-2 row blocks), the block input copy and block output,
# the weight copy. Phase peaks get the caching allocator's slack on top
# (ALLOC_SLACK_DIV / ALLOC_SLACK; the stripe phase starts from an emptied cache).

ALLOC_SLACK_DIV = 6          # reserved <= live + live / 6 + 64 MiB per phase (measured +12..16 %, DESIGN §9.13.4)
ALLOC_SLACK = 64 * MIB


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


def res_peak(cin, cout, shortcut, r, w, e, ws, s):
    a, b = r * w * cin * e, r * w * cout * e
    phases = [2 * a,
              a + b + conv_extra(cin, cout, 3, r, r, w, w, e, ws),
              3 * b,
              2 * b + conv_extra(cout, cout, 3, r, r, w, w, e, ws),
              2 * b]
    if shortcut:
        phases += [2 * b + conv_extra(cin, cout, 1, r, r, w, w, e, ws, contiguous=False), 3 * b]
    return s + max(phases)


def up_peak(cin, cout, r, w, e, ws, s):
    a = r * w * cin * e
    return s + max(5 * a, 4 * a + 4 * r * w * cout * e + conv_extra(cin, cout, 3, 2 * r, 2 * r, 2 * w, 2 * w, e, ws))


def unit_peak(u, r, w, e, ws, s):
    """Peak bytes while unit u runs on r input rows of width w, input storage s."""
    if u.kind == RES:
        return res_peak(u.cin, u.cout, u.shortcut, r, w, e, ws, s)
    if u.kind == UP:
        return up_peak(u.cin, u.cout, r, w, e, ws, s)
    if u.kind == CONV:
        return s + r * w * u.cout * e + conv_extra(u.cin, u.cout, 3, r, r, w, w, e, ws, contiguous=False)
    return s + 2 * r * w * u.cin * e


def _attn_split(m):
    return m.__dict__.get("optimized_attention") is ldm_model.normal_attention


def prefix_peak(m, h, w, e, ws, s):
    """Peak bytes while prefix module m runs on the whole h x w image, input storage s."""
    if isinstance(m, wan.AttentionBlock):
        c = int(m.norm.gamma.shape[0])
        p, n = h * w * c * e, h * w
        rows = min(n, max(1, ws // (2 * n * e)))
        scores = 2 * rows * n * e + 2 * rows * c * e
        return s + (5 if _attn_split(m) else 8) * p + scores
    if isinstance(m, wan.ResidualBlock):
        ci, co = _conv_io(m.residual[2])[0], _conv_io(m.residual[6])[1]
        return res_peak(ci, co, not isinstance(m.shortcut, torch.nn.Identity), h, w, e, ws, s)
    ci, co = _conv_io(m)
    k = int(m.kernel_size[-1])
    return s + h * w * co * e + conv_extra(ci, co, k, h, h, w, w, e, ws)


def prefix_out_channels(m):
    if isinstance(m, wan.AttentionBlock):
        return int(m.norm.gamma.shape[0])
    if isinstance(m, wan.ResidualBlock):
        return _conv_io(m.residual[6])[1]
    return _conv_io(m)[1]


def _macs_prefix_module(m):
    if isinstance(m, wan.AttentionBlock):
        c = int(m.norm.gamma.shape[0])
        return 4 * c * c          # qkv + proj (the attention itself is the same in both)
    if isinstance(m, wan.ResidualBlock):
        a, b = _conv_io(m.residual[2]), _conv_io(m.residual[6])
        return 9 * (a[0] * a[1] + b[0] * b[1]) + (a[0] * b[1] if not isinstance(m.shortcut, torch.nn.Identity) else 0)
    ci, co = _conv_io(m)
    k = int(m.kernel_size[-1])
    return k * k * ci * co


def with_slack(live):
    return int(live + live // ALLOC_SLACK_DIV + ALLOC_SLACK)


class Plan:
    """Stripes, intervals and the memory / work estimate of one decode."""

    def __init__(self, bound, h8, w8, rows, ws, elem, out_bytes, lat_bytes):
        self.rows = rows
        self.workspace = ws
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
        for _, m in bound.prefix:
            prefix_live = max(prefix_live, prefix_peak(m, h8, w8, elem, ws, s))
            s = prefix_out_channels(m) * h8 * w8 * elem
        strip_live = 0
        work_stripes = 0
        for needs in self.needs:
            s = 0   # the first unit reads a slice of the checkpoint (counted on its own)
            for i, u in enumerate(units):
                r = needs[i][1] - needs[i][0]
                strip_live = max(strip_live, unit_peak(u, r, widths[i], elem, ws, s))
                s = r * u.scale * widths[i + 1] * u.cout * elem
                work_stripes += r * u.scale * widths[i + 1] * u.macs_row
        self.prefix_live, self.stripe_live = prefix_live, strip_live
        self.prefix_bytes = with_slack(prefix_live)
        self.stripe_bytes = self.ckpt_bytes + with_slack(strip_live)
        self.persistent = out_bytes + lat_bytes
        self.estimate = int(self.persistent + max(self.prefix_bytes, self.stripe_bytes))
        work_prefix = sum(_macs_prefix_module(m) for _, m in bound.prefix) * h8 * w8
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
    stripe's core rows [B, C, T, o1-o0, W]."""
    heights = plan.heights
    for k in plan.order:
        (o0, o1), needs = plan.stripes[k], plan.needs[k]
        xa, xb = needs[0]
        x = ckpt.narrow(HDIM, xa, xb - xa)
        for i, u in enumerate(units):
            y = u.module(x)
            del x
            oa = xa * u.scale
            if y.shape[HDIM] != (xb - xa) * u.scale:
                raise StripeError("[Monoload] internal error: {} returned {} rows for {} input rows".format(u.name, y.shape[HDIM], xb - xa))
            va, vb = valid_out(u, xa, xb, heights[i + 1])
            ta, tb = needs[i + 1]
            if ta < va or tb > vb:
                raise StripeError("[Monoload] internal error: stripe [{}, {}) needs rows [{}, {}) of {} but only [{}, {}) are exact".format(
                    o0, o1, ta, tb, u.name, va, vb))
            x = y.narrow(HDIM, ta - oa, tb - ta)
            del y
            xa, xb = ta, tb
        write(o0, o1, x)
        del x


class WanStripe:
    """Layer-1 adapter bound to one WanVAE."""

    name = "Wan 2.1 stripes"

    def __init__(self, fsm, first_resample):
        self.fsm = fsm
        self.prefix, self.units = build_units(fsm, first_resample)
        last = self.prefix[-1][1]
        self.ckpt_channels = _conv_io(last.residual[6])[1] if isinstance(last, wan.ResidualBlock) else (
            int(last.norm.gamma.shape[0]) if isinstance(last, wan.AttentionBlock) else _conv_io(last)[1])
        self.key = signature(fsm, self.units)

    def plan(self, vae, samples, budget, ws, rows=None, out_bytes=0):
        """rows=None: the largest stripe height whose estimate fits the budget
        (None if even MIN_ROWS does not fit); else exactly `rows`."""
        h8, w8 = int(samples.shape[-2]), int(samples.shape[-1])
        elem = mm.dtype_size(vae.vae_dtype)
        lat = samples[0:1].numel() * elem
        if rows is not None:
            return Plan(self, h8, w8, rows, ws, elem, out_bytes, lat)
        h_out = h8 * 8
        best = None
        lo, hi = min(MIN_ROWS, h_out), h_out
        if Plan(self, h8, w8, lo, ws, elem, out_bytes, lat).estimate > budget:
            return None
        while lo <= hi:
            mid = (lo + hi) // 2
            p = Plan(self, h8, w8, mid, ws, elem, out_bytes, lat)
            if p.estimate <= budget:
                best, lo = p, mid + 1
            else:
                hi = mid - 1
        return best

    def run(self, vae, samples_in, plan, budget_ws, stats):
        """Decode every sample with the stripe plan; returns pixel_samples
        (process_output applied) shaped like native's [B, C, T, H, W]."""
        fsm = self.fsm
        n = samples_in.shape[0]
        pixel_samples = None
        with OpChunking(fsm, budget_ws, stats):
            for i in range(n):
                z = samples_in[i:i + 1].to(device=vae.device, dtype=vae.vae_dtype)
                ckpt = run_prefix(self.prefix, z)
                del z
                # give the prefix's freed blocks (attention scores, qkv, columns of whole-image convs) back: the
                # stripes allocate other sizes and would otherwise stack their blocks on top of them (DESIGN §9.13.4)
                mm.soft_empty_cache(True)
                stats.cache_releases += 1
                if pixel_samples is None:
                    shape = (n, self.units[-1].cout, ckpt.shape[2], plan.h_out, plan.w_out)
                    pixel_samples = torch.empty(shape, device=vae.output_device, dtype=vae.vae_output_dtype())
                dst = pixel_samples[i:i + 1]

                def write(o0, o1, rows, dst=dst):
                    dst.narrow(HDIM, o0, o1 - o0).copy_(rows)  # = native's .to(output dtype) + copy_, per stripe
                run_stripes(self.units, plan, ckpt, write)
                del ckpt
                vae.process_output(pixel_samples[i:i + 1])
        return pixel_samples


def match(vae, samples, vae_options):
    """(bound adapter, None) or (None, reason)."""
    fsm = vae.first_stage_model
    reason, info = wan_structure(fsm)
    if reason:
        return None, reason
    if vae_options:
        return None, "vae_options {} given".format(sorted(vae_options))
    if samples.ndim != 5 or samples.shape[2] != 1:
        return None, "latent is not a single frame ({})".format(list(samples.shape))
    if samples.shape[1] != fsm.conv2.weight.shape[1]:
        return None, "latent has {} channels, conv2 expects {}".format(samples.shape[1], fsm.conv2.weight.shape[1])
    return WanStripe(fsm, info["first_resample"]), None


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

_SELFTEST = {}   # signature -> (ok, detail)


def _fp32_copy(fsm, device):
    """conv2 + decoder rebuilt from the model's configuration in fp32 with the
    model's weights (the encoder is not copied); freed after the self-test."""
    dec = fsm.decoder
    holder = torch.nn.Module()
    z = int(fsm.conv2.weight.shape[0])
    out_c = int(dec.head[2].weight.shape[0])
    with torch.device(device):
        holder.conv2 = wan.CausalConv3d(z, z, 1)
        holder.decoder = wan.Decoder3d(dec.dim, dec.z_dim, out_c, list(dec.dim_mult), dec.num_res_blocks,
                                       list(dec.attn_scales), list(dec.temperal_upsample), 0.0)
    holder.to(device=device, dtype=torch.float32)
    holder.conv2.load_state_dict(fsm.conv2.state_dict(), strict=True)
    holder.decoder.load_state_dict(dec.state_dict(), strict=True)
    holder.eval()
    return holder


def self_test(bound, vae):
    """First use of a structure in this process: decode a small latent with
    forced small stripes (several inner boundaries, layer-2 conv blocks inside)
    and with the whole-image WanVAE.decode, both on an fp32 copy of the decoder;
    pass if they agree to SELFTEST_TOL. Cached per structure; the RNG state is
    preserved. Afterwards the fp32 copy and every tensor of the test are freed
    and the allocator's cache is emptied, so the decode that follows starts
    from the same memory state as any later one. Returns (ok, detail)."""
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
        holder = _fp32_copy(bound.fsm, dev)
        reason, info = wan_structure(_Shim(holder))
        if reason:
            raise StripeError("fp32 copy does not match: " + reason)
        tb = WanStripe(_Shim(holder), info["first_resample"])
        if tb.key != bound.key:
            raise StripeError("fp32 copy has structure {} instead of {}".format(tb.key, bound.key))
        g = torch.Generator().manual_seed(0)
        z = torch.randn(1, int(holder.conv2.weight.shape[1]), 1, SELFTEST_LATENT, SELFTEST_LATENT, generator=g).to(dev)
        ref = wan.WanVAE.decode(holder, z)
        plan = Plan(tb, SELFTEST_LATENT, SELFTEST_LATENT, SELFTEST_ROWS, SELFTEST_WORKSPACE, 4, 0, 0)
        out = torch.empty_like(ref)
        stats = OpStats()

        def write(o0, o1, rows):
            out.narrow(HDIM, o0, o1 - o0).copy_(rows)
        with OpChunking(holder, SELFTEST_WORKSPACE, stats):
            ckpt = run_prefix(tb.prefix, z)
            run_stripes(tb.units, plan, ckpt, write)
        scale = max(1.0, float(ref.abs().max()))
        err = float((out - ref).abs().max()) / scale
        ok = bool(torch.isfinite(out).all()) and err <= SELFTEST_TOL
        detail = "max|stripes - whole| / max(1, max|whole|) = {:.2g} (tolerance {:g}), {} stripes, {} conv calls in row blocks, {:.2f}s".format(
            err, SELFTEST_TOL, len(plan.stripes), stats.conv_chunked, time.perf_counter() - t0)
    return ok, detail


class _Shim:
    """Duck-types the attributes wan_structure / WanStripe read from a WanVAE."""

    def __init__(self, holder):
        self.conv2 = holder.conv2
        self.decoder = holder.decoder
        self._holder = holder

    def named_modules(self):
        return self._holder.named_modules()


def _is_wan(fsm):
    return type(fsm) is wan.WanVAE or isinstance(fsm, _Shim)
