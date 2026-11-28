"""CogVideoX VAE (comfy.ldm.cogvideo.vae.AutoencoderKLCogVideoX, ComfyUI v0.46.3): the multi-frame decode as
pieces of output frames, with the last temporal upsampling streamed (group C, docs/PLAN-video.md; DESIGN §9.46).

What native decode does (AutoencoderKLCogVideoX._decode_rolling, v0.46.3). With split = log2(temporal ratio 4) = 2:

  phase 1, the whole video:  conv_in -> mid_block -> up_blocks[0] (4 resnets at H/8, Upsample3D: time 2T-1, x2)
                             -> up_blocks[1] (4 resnets at H/4, Upsample3D: time 2(2T-1)-1 = 4T-3, x2)
  phase 2, 4 frames at a time with conv caches carried between chunks:
                             up_blocks[2] (4 resnets at H/2, Upsample3D x2 spatial) -> up_blocks[3] (4 resnets at H)
                             -> norm_out -> SiLU -> conv_out; each chunk .cpu(), torch.cat at the end.

Every GroupNorm (SpatialNorm3D) normalises over the tensor it gets: in phase 1 the whole video, in phase 2 one
chunk of 4 output frames (native's own chunk statistics). What dominates native's peak is the output of
up_blocks[1]'s Upsample3D: the whole video at H/2 with 256 channels (720p x 81 frames, bf16: 9.6 GB) and the
interpolation / conv copies around it, alive through phase 2 (DESIGN §9.32: the simulated 720p x 81 decode needs
121 GiB natively, 45 GiB under layer 2 alone).

Here (chunks()) phase 1 stops before that upsampling: up_blocks[1]'s resnets run on the whole video as native, and
its Upsample3D runs per phase-2 chunk on the source frames that chunk needs. Upsample3D is per frame apart from the
nearest-neighbour time interpolation (which frames are copied where): spatial nearest x2 then a 3x3 Conv2d on each
frame. So a chunk's frames are the frames native slices out of the whole upsampled video; the conv runs once per
distinct source frame (half native's work) and the copies are taken from its output, in native's memory layout
((b t) c h w -> b c t h w). Phase 2 is native's loop, the same modules, caches and chunk boundaries, each chunk
handed over as it is made (vae.py copies it into the output buffer; no .cpu(), no torch.cat).

The frames out of t latent frames (output_frames): each compress_time Upsample3D maps T -> 2T - 1 (T odd > 1),
2T (T even), 1 (T = 1); sd.py's upscale_ratio max(0, 4t - 3) is that rule for odd t only.

Values: the same module calls as native on the same frames. Native convolves the duplicated frames again (same
input); here they are copies of one result. Kernels chosen for another batch size may round differently: fp-close,
not bit-identical (the real machine measures it, PLAN-video C).
"""

import math
import time

import torch
import torch.nn.functional as F

import comfy.ldm.cogvideo.vae as cog
import comfy.model_management as mm
import comfy.ops

from . import vae_engine as eng
from .vae_ops import MIB, OpChunking, OpStats, fmt_bytes, malloc_graph_paused

CHUNK = 4   # pixel frames per phase-2 chunk (_decode_rolling's chunk_size, v0.46.3)


def structure(fsm):
    """(reason, split) — reason None when fsm is the CogVideoX VAE this module streams, split the number of
    up_blocks that upsample in time (phase 1)."""
    if type(fsm) is not cog.AutoencoderKLCogVideoX:
        return "not AutoencoderKLCogVideoX ({})".format(type(fsm).__name__), None
    dec = getattr(fsm, "decoder", None)
    if type(dec) is not cog.Decoder3D:
        return "decoder is {}, not Decoder3D".format(type(dec).__name__), None
    ratio = int(getattr(fsm, "temporal_compression_ratio", 0) or 0)
    if ratio < 2 or ratio & (ratio - 1):
        return "temporal_compression_ratio {} is not a power of 2".format(ratio), None
    split = int(math.log2(ratio))
    ups = dec.up_blocks
    if len(ups) <= split:
        return "{} up_blocks for a time split at {}".format(len(ups), split), None
    for i, b in enumerate(ups):
        if type(b) is not cog.UpBlock3D:
            return "up_blocks[{}] is {}".format(i, type(b).__name__), None
        us = list(b.upsamplers) if b.upsamplers is not None else []
        if i < split and (len(us) != 1 or type(us[0]) is not cog.Upsample3D or not us[0].compress_time):
            return "up_blocks[{}] has no single time-upsampling Upsample3D".format(i), None
        if i >= split and any(type(u) is not cog.Upsample3D or u.compress_time for u in us):
            return "up_blocks[{}] upsamples in time".format(i), None
    for name in ("conv_in", "conv_out"):
        if type(getattr(dec, name, None)) is not cog.CausalConv3d:
            return "decoder.{} is not CausalConv3d".format(name), None
    if type(getattr(dec, "norm_out", None)) is not cog.SpatialNorm3D:
        return "decoder.norm_out is not SpatialNorm3D", None
    return None, split


def up_frames(t):
    """Frames out of a compress_time Upsample3D with t frames in (Upsample3D.forward)."""
    if t > 1:
        return 2 * t - 1 if t % 2 else 2 * t
    return t


def up_source(t, k):
    """Input frame of a compress_time Upsample3D (t frames in) that its output frame k is copied from (nearest)."""
    if t > 1 and t % 2:
        return 0 if k == 0 else 1 + (k - 1) // 2
    if t > 1:
        return k // 2
    return k


def _frames(t, split):
    for _ in range(split):
        t = up_frames(t)
    return t


def output_frames(fsm, t):
    """Frames native decode makes from t latent frames (None if fsm is not a CogVideoX VAE this module knows)."""
    reason, split = structure(fsm)
    if reason:
        return None
    return _frames(t, split)


def upsample_frames(us, x, k0, k1):
    """Frames [k0, k1) of the time-upsampling Upsample3D `us` applied to the whole x ([b, c, t, h, w]), in native's
    layout: the source frames' spatial nearest x2 and conv once each, the output frames copies of them."""
    t = int(x.shape[2])
    src = [up_source(t, k) for k in range(k0, k1)]
    s0, s1 = src[0], src[-1] + 1
    b, c = int(x.shape[0]), int(x.shape[1])
    xs = x[:, :, s0:s1].permute(0, 2, 1, 3, 4).reshape(b * (s1 - s0), c, x.shape[3], x.shape[4])
    xs = F.interpolate(xs, scale_factor=2.0)
    y = us.conv(xs)
    del xs
    y = y.reshape(b, s1 - s0, *y.shape[1:])
    if src == list(range(s0, s1)):
        out = y
    else:
        out = y.index_select(1, torch.tensor([s - s0 for s in src], device=y.device))
    del y
    return out.permute(0, 2, 1, 3, 4)


def chunks(fsm, z):
    """Pieces of fsm.decode(z) (z [1, C, T, h, w] on the device), consecutive along the frames (module docstring)."""
    reason, split = structure(fsm)
    if reason:
        yield fsm.decode(z)
        return
    dec = fsm.decoder
    x, _ = dec.conv_in(z)
    x, _ = dec.mid_block(x, None, z)
    for i in range(split - 1):
        x, _ = dec.up_blocks[i](x, None, z)
    last = dec.up_blocks[split - 1]
    for resnet in last.resnets:
        x, _ = resnet(x, None, z)
    us = last.upsamplers[0]
    remaining = list(range(split, len(dec.up_blocks)))
    t_exp = up_frames(int(x.shape[2]))
    if t_exp <= CHUNK or not remaining:
        # native's "small enough" branch: everything on the whole tensor
        x = us(x)
        for i in remaining:
            x, _ = dec.up_blocks[i](x, None, z)
        x, _ = dec.norm_out(x, z)
        x = dec.conv_act(x)
        x, _ = dec.conv_out(x)
        yield x
        return
    z_te = cog._interpolate_zq(z, (t_exp, z.shape[3], z.shape[4]))
    caches = {}
    for c0 in range(0, t_exp, CHUNK):
        c1 = min(c0 + CHUNK, t_exp)
        xc = upsample_frames(us, x, c0, c1)
        zc = z_te[:, :, c0:c1]
        zs = {}
        for i in remaining:
            hw = (xc.shape[3], xc.shape[4])
            if hw not in zs:
                zs[hw] = zc if (zc.shape[3], zc.shape[4]) == hw else F.interpolate(zc, size=(zc.shape[2], hw[0], hw[1]))
            key = "up_block_{}".format(i)
            xc, caches[key] = dec.up_blocks[i](xc, None, zs[hw], conv_cache=caches.get(key))
        hw = (xc.shape[3], xc.shape[4])
        if hw not in zs:
            zs[hw] = F.interpolate(zc, size=(zc.shape[2], hw[0], hw[1]))
        xc, caches["norm_out"] = dec.norm_out(xc, zs[hw], conv_cache=caches.get("norm_out"))
        del zs
        xc = dec.conv_act(xc)
        xc, caches["conv_out"] = dec.conv_out(xc, conv_cache=caches.get("conv_out"))
        yield xc
        del xc
    del x, z_te, caches


# ---------------------------------------------------------------------------
# layer 1: the same decode with every full-size temporary replaced by row slices (DESIGN §9.46)
# ---------------------------------------------------------------------------
#
# The order of work is native's (phase 1 on the whole video, phase 2 in native's 4-frame chunks with the conv caches
# carried between them, up_blocks[1]'s Upsample3D streamed per chunk as in chunks()). What changes is how each
# module runs: no module's full-size temporaries exist. Per ResnetBlock3D:
#
#   statistics  norm1's GroupNorm statistics over the whole input (the whole video in phase 1, the chunk in phase
#               2: exactly the tensor native normalises), fp32, by row slices (vae_engine.Moments);
#   conv1       output rows [a, b) from norm1 -> SiLU of input rows [a - 1, b + 1) only, computed on the fly
#               (GroupNorm with the frozen statistics, SpatialNorm3D's conv_y / conv_b of zq on those rows), the
#               causal time padding / conv cache concatenated per slice, the module's own conv (its zero padding at
#               the slice's edges falls on the halo rows, which are cropped); the new conv cache (the last
#               kt - 1 frames of the padded input, native's clone) filled row by row;
#   norm2       statistics of conv1's output (stored whole: the statistics need all of it), conv2 the same way;
#   residual    conv2's rows + the shortcut (1x1 conv, per row) or the input itself: written into the input's
#               storage when the channels match (conv2 reads conv1's output, never the input), else into a new
#               tensor.
#
# Live at once: the block's input, conv1's output, (the output when the channels change), the caches, row slices.
# Native: the input plus three to five full-size copies around each SpatialNorm3D (GroupNorm, conv_y, conv_b, the
# product, the sum). Upsample3D runs by output row slices (source rows +-1 for the 3x3 conv); the time
# interpolation is the frame map of chunks(). norm_out -> SiLU -> conv_out writes its rows straight into the output
# buffer. The values are native's up to the rounding of the GroupNorm statistics (fp32 sums in another order) and of
# the normalisation (one fp32 multiply-add, rounded once, as vae_engine.group_norm_frozen); the self-test compares
# the decode with chunks() (native's module calls) on an fp32 copy.

SLICE_TEMPS = 6          # row-slice temporaries bounded by this many workspaces (fp32 normalisation, the bf16 products,
                         # the padded conv input, the conv output, the conv's own layer-2 blocks)
SELFTEST_N = 16          # self-test latent rows / cols (128 x 128 output)
SELFTEST_T = 4           # self-test latent frames (even: 4t = 16 output frames, 4 phase-2 chunks)
SELFTEST_ROWS = 24       # self-test output rows per slice (3 / 6 / 12 / 24 rows at the four levels)
MARGIN = 256 * MIB       # allocator rounding on top of the counted tensors. Real machine (PLAN-video C), conv caches in place:
                         # 720p x 81 reserved 12.05 GiB at live peak 13.04, 720p x 77 11.86 at 13.01 (before, a new cache
                         # per chunk: 14.62 reserved at 13.04)


def _conv_ok(cc):
    """A CausalConv3d this layer-1 decode runs by row slices: stride 1, dilation 1, odd kernel, padding (0, kh//2, kw//2)."""
    c = cc.conv
    k = tuple(c.kernel_size)
    return (tuple(c.stride) == (1, 1, 1) and tuple(c.dilation) == (1, 1, 1) and k[1] % 2 == 1 and k[2] % 2 == 1
            and tuple(c.padding) == (0, k[1] // 2, k[2] // 2) and c.groups == 1 and cc.pad_mode != "replicate"
            and cc.time_kernel_size == k[0])


def layer1_structure(fsm):
    """Reason the layer-1 decode does not take fsm, or None (on top of structure())."""
    reason, split = structure(fsm)
    if reason:
        return reason
    dec = fsm.decoder
    blocks = list(dec.mid_block.resnets) + [r for b in dec.up_blocks for r in b.resnets]
    for i, rb in enumerate(blocks):
        if type(rb) is not cog.ResnetBlock3D:
            return "resnet {} is {}".format(i, type(rb).__name__)
        if type(rb.norm1) is not cog.SpatialNorm3D or type(rb.norm2) is not cog.SpatialNorm3D:
            return "resnet {} has no SpatialNorm3D".format(i)
        for cc in (rb.conv1, rb.conv2, rb.norm1.conv_y, rb.norm1.conv_b, rb.norm2.conv_y, rb.norm2.conv_b):
            if type(cc) is not cog.CausalConv3d or not _conv_ok(cc):
                return "resnet {} has a conv this decode does not slice".format(i)
        if rb.conv_shortcut is not None and tuple(rb.conv_shortcut.kernel_size) != (1, 1, 1):
            return "resnet {} has a shortcut that is not 1x1x1".format(i)
        if rb.conv_shortcut is None and rb.in_channels != rb.out_channels:
            return "resnet {} changes channels without a shortcut".format(i)
    for b in dec.up_blocks:
        for us in (b.upsamplers or []):
            c = us.conv
            if tuple(c.kernel_size) != (3, 3) or tuple(c.stride) != (1, 1) or tuple(c.padding) != (1, 1) or tuple(c.dilation) != (1, 1):
                return "an Upsample3D conv is not 3x3 / stride 1 / padding 1"
    if not _conv_ok(dec.conv_out) or not _conv_ok(dec.norm_out.conv_y) or not _conv_ok(dec.norm_out.conv_b):
        return "norm_out / conv_out has a conv this decode does not slice"
    if type(dec.conv_act) is not torch.nn.SiLU:
        return "conv_act is not SiLU"
    return None


def structure_key(fsm):
    """What the decode and its self-test depend on: channels, groups, kernels, blocks."""
    dec = fsm.decoder
    res = tuple((r.in_channels, r.out_channels, r.norm1.norm_layer.num_groups, float(r.norm1.norm_layer.eps),
                 float(r.norm2.norm_layer.eps), tuple(r.conv1.conv.kernel_size))
                for r in list(dec.mid_block.resnets) + [r for b in dec.up_blocks for r in b.resnets])
    ups = tuple((len(b.resnets), tuple((u.compress_time, u.conv.in_channels, u.conv.out_channels) for u in (b.upsamplers or [])))
                for b in dec.up_blocks)
    return ("cogvideox", int(fsm.temporal_compression_ratio), int(dec.conv_in.conv.in_channels), int(dec.conv_out.conv.out_channels),
            res, ups, int(dec.norm_out.norm_layer.num_groups))


# --- the row-sliced modules ---------------------------------------------------

def _rows(per_row, cap, ws):
    """Rows per slice: at most cap, at most ws bytes of per_row bytes each, at least 1."""
    return max(1, min(cap, ws // max(1, per_row)))


def _gn_affine(gn, x, ws):
    """(a, b) fp32 [1, C, 1, 1, 1]: GroupNorm gn of x is x * a + b, with x's whole statistics (vae_engine.Moments, as
    vae_engine.group_norm_frozen; the weights through the module's own cast path, as vae_engine._FrozenGroupNorm)."""
    acc = eng.Moments(gn.num_groups)
    acc.add(x, 3, ws)
    mean, rstd = acc.freeze(gn.eps)
    c = x.shape[1]
    rep = c // gn.num_groups

    def affine(weight, bias):
        a = rstd.repeat_interleave(rep)
        if weight is not None:
            a = a * weight.float()
        b = -mean.repeat_interleave(rep) * a
        if bias is not None:
            b = b + bias.float()
        return a.view(1, c, 1, 1, 1), b.view(1, c, 1, 1, 1)
    if isinstance(gn, comfy.ops.CastWeightBiasOp):
        comfy.ops.run_every_op()
        if gn.comfy_cast_weights or len(gn.weight_function) > 0 or len(gn.bias_function) > 0:
            with comfy.ops.CastBiasWeightContext(gn, x, offloadable=True) as (weight, bias):
                return affine(weight, bias)
    return affine(gn.weight, gn.bias)


def _snorm_rows(sn, ab, x, zq, act):
    """SpatialNorm3D sn on the row slice x (zq: its rows of zq at x's frames / size), then act (SiLU or None).
    Rounding as native: the GroupNorm rounded to x's dtype, then * conv_y + conv_b in that dtype."""
    a, b = ab
    n = torch.addcmul(b, x.float(), a).to(x.dtype)
    cy, _ = sn.conv_y(zq)
    n = n * cy
    del cy
    cb, _ = sn.conv_b(zq)
    n = n + cb
    del cb
    return act(n) if act is not None else n


def _causal_rows(cc, src, shape, cache, write, rows, ws, want_cache, like):
    """CausalConv3d cc on an input given by rows (src(r0, r1) -> [1, ci, T, r1 - r0, W]) of shape (ci, T, H, W):
    output rows [a, b) written by write(a, b, rows); returns the new conv cache (native's clone of the last kt - 1
    padded input frames) or None. The cache-None padding is native's: the first frame kt - 1 times (T > 1; a single
    frame without a cache takes native's summed-kernel path, which this decode does not run: match()). like: a tensor
    of the input's dtype / device."""
    ci, T, H, W = shape
    kt = cc.time_kernel_size
    hp = cc.conv.kernel_size[1] // 2
    r = _rows((T + kt - 1) * ci * W * 4, rows, ws)
    new = None
    if kt > 1 and want_cache:
        cshape = (1, ci, kt - 1, H, W)
        if cache is not None and tuple(cache.shape) == cshape and cache.dtype == like.dtype and r >= hp:
            new = cache   # written in place (the same buffer every chunk: no allocator churn, PLAN-video C)
        else:
            new = torch.empty(cshape, dtype=like.dtype, device=like.device)
    kept = None   # the old cache's rows [a - hp, a), overwritten by the previous slice and read by this one's halo
    for a in range(0, H, r):
        b = min(H, a + r)
        r0, r1 = max(0, a - hp), min(H, b + hp)
        s = src(r0, r1)
        if kt > 1:
            if cache is not None:
                old = cache[:, :, :, r0:r1] if kept is None else torch.cat([kept, cache[:, :, :, a:r1]], dim=3)
                s = torch.cat([old, s], dim=2)
                del old
            else:
                s = torch.cat([s[:, :, :1]] * (kt - 1) + [s], dim=2)
            if new is not None:
                if new is cache:
                    kept = cache[:, :, :, max(a, b - hp):b].clone() if hp and b < H else None
                new[:, :, :, a:b].copy_(s[:, :, -(kt - 1):, a - r0:b - r0])
        o = cc.conv(s)
        del s
        write(a, b, o[:, :, :, a - r0:b - r0])
        del o
    return new


def _zq_for(zq, x):
    return zq if tuple(zq.shape[-3:]) == tuple(x.shape[-3:]) else cog._interpolate_zq(zq, tuple(x.shape[-3:]))


def resnet_rows(rb, x, zq, cache, rows, ws, want_cache):
    """ResnetBlock3D rb (temb None) on x [1, ci, T, H, W] with conv caches `cache` (dict or None) -> (out, new cache).
    x is consumed: when the channels match the output is written into it."""
    _, ci, T, H, W = x.shape
    co = rb.out_channels
    cache = cache or {}
    zf = _zq_for(zq, x)
    act = rb.nonlinearity
    ab1 = _gn_affine(rb.norm1.norm_layer, x, ws)
    h1 = torch.empty((1, co, T, H, W), dtype=x.dtype, device=x.device)

    def w1(a, b, o):
        h1[:, :, :, a:b].copy_(o)
    c1 = _causal_rows(rb.conv1, lambda r0, r1: _snorm_rows(rb.norm1, ab1, x[:, :, :, r0:r1], zf[:, :, :, r0:r1], act),
                      (ci, T, H, W), cache.get("conv1"), w1, rows, ws, want_cache, x)
    ab2 = _gn_affine(rb.norm2.norm_layer, h1, ws)
    if rb.conv_shortcut is None:
        out = x

        def w2(a, b, o):
            out[:, :, :, a:b].add_(o)
    else:
        out = torch.empty((1, co, T, H, W), dtype=x.dtype, device=x.device)

        def w2(a, b, o):
            out[:, :, :, a:b].copy_(o + rb.conv_shortcut(x[:, :, :, a:b]))
    c2 = _causal_rows(rb.conv2, lambda r0, r1: _snorm_rows(rb.norm2, ab2, h1[:, :, :, r0:r1], zf[:, :, :, r0:r1], act),
                      (co, T, H, W), cache.get("conv2"), w2, rows, ws, want_cache, x)
    del h1, x, zf
    return out, ({"conv1": c1, "conv2": c2} if want_cache else None)


def upsample_rows(us, x, k0, k1, rows, ws):
    """Frames [k0, k1) of Upsample3D us on x [1, ci, t, H, W] -> [1, co, k1 - k0, 2H, 2W], by output row slices:
    source rows +-1 output row for the 3x3 conv (its zero padding at the slice's edges falls on cropped rows), the
    source frames once each (the time interpolation is nearest: copies, chunks()'s frame map)."""
    _, ci, t, H, W = x.shape
    src = [up_source(t, k) for k in range(k0, k1)] if us.compress_time else list(range(k0, k1))
    s0f, s1f = src[0], src[-1] + 1
    tf = s1f - s0f
    co = us.conv.out_channels
    out = torch.empty((1, co, k1 - k0, 2 * H, 2 * W), dtype=x.dtype, device=x.device)
    sel = None if src == list(range(s0f, s1f)) else torch.tensor([s - s0f for s in src], device=x.device)
    r = _rows(tf * max(ci, co) * 2 * W * x.element_size() * 3, rows, ws)
    for a in range(0, 2 * H, r):
        b = min(2 * H, a + r)
        s0, s1 = max(0, (a - 1) // 2), min(H, b // 2 + 1)
        xs = x[:, :, s0f:s1f, s0:s1].permute(0, 2, 1, 3, 4).reshape(tf, ci, s1 - s0, W)
        xs = F.interpolate(xs, scale_factor=2.0)
        y = us.conv(xs)
        del xs
        y = y[:, :, a - 2 * s0:b - 2 * s0].reshape(1, tf, co, b - a, 2 * W)
        if sel is not None:
            y = y.index_select(1, sel)
        out[:, :, :, a:b].copy_(y.permute(0, 2, 1, 3, 4))
        del y
    return out


def decode_rows(fsm, z, dest, rows, ws):
    """The layer-1 decode of one sample z [1, C, T, h, w] (T >= 2) into dest [1, 3, frames, H, W] (any device and
    dtype: each row slice of conv_out is copied in, as native's .to(output dtype) + copy_). rows: row-slice height at
    the output resolution (scaled to each level), ws: bytes of one slice's fp32 temporaries."""
    _, split = structure(fsm)
    dec = fsm.decoder
    h_out = int(dest.shape[-2])

    def lr(h):
        return max(1, rows * h // h_out)
    x, _ = dec.conv_in(z)
    for rb in dec.mid_block.resnets:
        x, _ = resnet_rows(rb, x, z, None, lr(x.shape[3]), ws, False)
    for i in range(split):
        blk = dec.up_blocks[i]
        for rb in blk.resnets:
            x, _ = resnet_rows(rb, x, z, None, lr(x.shape[3]), ws, False)
        if i < split - 1:
            x = upsample_rows(blk.upsamplers[0], x, 0, up_frames(int(x.shape[2])), lr(2 * x.shape[3]), ws)
    last_us = dec.up_blocks[split - 1].upsamplers[0]
    remaining = range(split, len(dec.up_blocks))
    t_exp = up_frames(int(x.shape[2]))
    if t_exp != dest.shape[2]:
        raise eng.StripeError("CogVideoX layer 1: {} frames decoded, the output has {}".format(t_exp, dest.shape[2]))
    z_te = cog._interpolate_zq(z, (t_exp, z.shape[3], z.shape[4]))
    mm.soft_empty_cache()   # phase 1's blocks go back; x (up_blocks[split - 1]'s output) stays
    caches = {}
    for c0 in range(0, t_exp, CHUNK):
        c1 = min(c0 + CHUNK, t_exp)
        zc = z_te[:, :, c0:c1]
        zs = {}

        def zq_at(t):
            hw = (int(t.shape[3]), int(t.shape[4]))
            if hw not in zs:
                zs[hw] = zc if (zc.shape[3], zc.shape[4]) == hw else F.interpolate(zc, size=(zc.shape[2], hw[0], hw[1]))
            return zs[hw]
        xc = upsample_rows(last_us, x, c0, c1, lr(2 * x.shape[3]), ws)
        for i in remaining:
            blk = dec.up_blocks[i]
            for j, rb in enumerate(blk.resnets):
                xc, caches[(i, j)] = resnet_rows(rb, xc, zq_at(xc), caches.get((i, j)), lr(xc.shape[3]), ws, True)
            for us in (blk.upsamplers or []):
                xc = upsample_rows(us, xc, 0, int(xc.shape[2]), lr(2 * xc.shape[3]), ws)
        zq = zq_at(xc)
        ab = _gn_affine(dec.norm_out.norm_layer, xc, ws)
        d = dest[:, :, c0:c1]

        def wout(a, b, o, d=d):
            d[:, :, :, a:b].copy_(o)
        _, c, f, hh, ww = xc.shape
        caches["conv_out"] = _causal_rows(dec.conv_out, lambda r0, r1: _snorm_rows(dec.norm_out, ab, xc[:, :, :, r0:r1], zq[:, :, :, r0:r1], dec.conv_act),
                                          (c, f, hh, ww), caches.get("conv_out"), wout, lr(hh), ws, True, xc)
        del xc, zs, zq, d
    del x, z_te, caches


class _Geom:
    """The decoder's blocks as the cost model sees them: per stage (level, resnets [(ci, co)], upsampler
    (ci, co, compress_time) or None)."""

    def __init__(self, fsm):
        dec = fsm.decoder
        self.split = structure(fsm)[1]
        self.zc = int(dec.conv_in.conv.in_channels)
        self.c_in = int(dec.conv_in.conv.out_channels)
        self.kt = int(dec.conv_in.time_kernel_size)
        self.mid = [(int(r.in_channels), int(r.out_channels)) for r in dec.mid_block.resnets]
        self.ups = [([(int(r.in_channels), int(r.out_channels)) for r in b.resnets],
                     (int(b.upsamplers[0].conv.in_channels), int(b.upsamplers[0].conv.out_channels), bool(b.upsamplers[0].compress_time))
                     if b.upsamplers else None) for b in dec.up_blocks]
        self.c_last = int(dec.conv_out.conv.in_channels)
        self.out_c = int(dec.conv_out.conv.out_channels)
        self.scale = 2 ** sum(1 for _, u in self.ups if u is not None)


def _resnet_bytes(ci, co, px, e, zq_px):
    """Full-size tensors live in resnet_rows: input, conv1's output, the output when the channels change, zq."""
    return (ci + co + (co if ci != co else 0)) * px * e + zq_px


def cost(g, h, w, t, e, rows, ws):
    """(phase-1 peak, phase-2 peak, archive, caches, slice temporaries) in bytes for a latent of t frames, h x w,
    element size e, row-slice height rows (output resolution) and workspace ws (see decode_rows)."""
    h_out = h * g.scale
    kt = g.kt
    temps = [0]

    def slice_temp(c, tt, hh, ww):
        r = max(1, rows * hh // h_out)
        per_row = (tt + kt - 1) * c * ww * 4
        temps[0] = max(temps[0], min(ws, (r + 2) * per_row))

    # phase 1
    px = t * h * w
    peak1 = (t + kt - 1) * g.zc * h * w * e + g.c_in * px * e
    zq = 0
    for ci, co in g.mid:
        peak1 = max(peak1, _resnet_bytes(ci, co, px, e, zq))
        slice_temp(ci, t, h, w)
    hh, ww, tt = h, w, t
    archive = 0
    for i in range(g.split):
        res, up = g.ups[i]
        px = tt * hh * ww
        zq = 0 if i == 0 else g.zc * px * e
        for ci, co in res:
            peak1 = max(peak1, _resnet_bytes(ci, co, px, e, zq))
            slice_temp(ci, tt, hh, ww)
        c_now = res[-1][1] if res else up[0]
        if i < g.split - 1:
            t2 = up_frames(tt)
            peak1 = max(peak1, c_now * px * e + up[1] * t2 * 4 * hh * ww * e)
            tt, hh, ww = t2, 2 * hh, 2 * ww
        else:
            archive = c_now * px * e
    # phase 2
    t_exp = up_frames(tt)
    f = min(CHUNK, t_exp)
    up1 = g.ups[g.split - 1][1]
    hh, ww = 2 * hh, 2 * ww
    caches = 0
    zq_all = g.zc * t_exp * h * w * e
    steps = [up1[1] * f * hh * ww * e]
    cur = up1[1]
    for i in range(g.split, len(g.ups)):
        res, up = g.ups[i]
        px = f * hh * ww
        zq_all += g.zc * px * e
        for ci, co in res:
            cb = (ci + co) * (kt - 1) * hh * ww * e
            caches += cb
            steps.append(_resnet_bytes(ci, co, px, e, 0) + cb)
            slice_temp(ci, f, hh, ww)
            cur = co
        if up is not None:
            steps.append(cur * px * e + up[1] * 4 * px * e)
            hh, ww = 2 * hh, 2 * ww
            cur = up[1]
    px = f * hh * ww
    cb = g.c_last * (kt - 1) * hh * ww * e
    caches += cb
    steps.append(g.c_last * px * e + cb)
    slice_temp(g.c_last, f, hh, ww)
    peak2 = archive + caches + zq_all + max(steps)
    return peak1, peak2, archive, caches, SLICE_TEMPS * temps[0]


class CogPlan:
    """A layer-1 plan, with the fields vae.py's records and messages read (vae_engine.Plan's names)."""

    def __init__(self, bound, h, w, t, rows, ws, elem, out_bytes, lat_bytes):
        g = bound.geom
        self.rows, self.workspace = int(rows), int(ws)
        self.h_out, self.w_out = h * g.scale, w * g.scale
        self.frames = _frames(t, g.split)
        p1, p2, arch, caches, temps = cost(g, h, w, t, elem, rows, ws)
        self.prefix_bytes, self.stripe_bytes, self.ckpt_bytes, self.persistent = p1, p2, arch, caches
        self.out_segment = out_bytes
        self.live_peak = lat_bytes + out_bytes + max(p1, p2) + temps
        self.estimate = self.live_peak + MARGIN
        self.arena = self.estimate          # no arena: what is reserved is the estimate (the default policy's measure)
        self.stripes = [(a, min(self.h_out, a + self.rows)) for a in range(0, self.h_out, self.rows)]
        self.order = list(range(len(self.stripes)))
        self.recompute = 0.0
        self.passes, self.saves, self.save_bytes = [], [], {}
        self.chunks = None

    def describe(self):
        return "{}-row slices, phase 1 {} / phase 2 {} (archive {}, caches {})".format(
            self.rows, fmt_bytes(self.prefix_bytes), fmt_bytes(self.stripe_bytes), fmt_bytes(self.ckpt_bytes), fmt_bytes(self.persistent))


class CogStripe:
    """The layer-1 bound of one CogVideoX VAE (the interface vae.py uses of a vae_engine.StripeAdapter)."""

    name = "cogvideox"
    # layer 2 when its estimate fits, layer 1 only when it does not (read by vae.py's default policy): layer 2 runs
    # every conv once and is ~45% faster (real machine 720p x 81: 163 s / 26.8 GiB reserved vs 236 s / 12.1 GiB)
    prefer_layer2 = True
    schemes = ()
    gn_scheme = None
    units = ()

    def __init__(self, fsm):
        self.fsm = fsm
        self.module = fsm.decoder
        self.geom = _Geom(fsm)
        self.key = structure_key(fsm)

    def variants(self, scheme=None):
        return [self]

    def predict_seconds(self, plan):
        return None

    def output_shape(self, samples):
        g = self.geom
        return (int(samples.shape[0]), g.out_c, _frames(int(samples.shape[2]), g.split),
                int(samples.shape[-2]) * g.scale, int(samples.shape[-1]) * g.scale)

    def output_bytes(self, vae, samples):
        a, b = torch.device(vae.output_device), torch.device(vae.device)
        if a.type != b.type or (a.index or 0) != (b.index or 0):
            return 0
        n = 1
        for d in self.output_shape(samples):
            n *= int(d)
        return n * mm.dtype_size(vae.vae_output_dtype())

    def plan(self, vae, samples, budget, ws, rows=None, out_bytes=0, measure="estimate"):
        h, w, t = int(samples.shape[-2]), int(samples.shape[-1]), int(samples.shape[2])
        elem = mm.dtype_size(vae.vae_dtype)
        lat = samples[0:1].numel() * elem
        h_out = h * self.geom.scale
        if rows is not None:
            return CogPlan(self, h, w, t, rows, ws, elem, out_bytes, lat)
        lo = min(eng.MIN_ROWS, h_out)
        best = CogPlan(self, h, w, t, lo, ws, elem, out_bytes, lat)
        if getattr(best, measure) > budget:
            return None
        lo, hi = lo + 1, h_out
        while lo <= hi:
            mid = (lo + hi) // 2
            p = CogPlan(self, h, w, t, mid, ws, elem, out_bytes, lat)
            if getattr(p, measure) <= budget:
                best, lo = p, mid + 1
            else:
                hi = mid - 1
        return best

    def smallest_plan(self, vae, samples, ws, out_bytes=0):
        return self.plan(vae, samples, 0, ws, rows=min(eng.MIN_ROWS, int(samples.shape[-2]) * self.geom.scale), out_bytes=out_bytes)

    def run(self, vae, samples_in, plan, budget_ws, stats):
        mm.soft_empty_cache()
        with malloc_graph_paused():
            pixel_samples = torch.empty(self.output_shape(samples_in), device=vae.output_device, dtype=vae.vae_output_dtype())
        with OpChunking(self.module, budget_ws, stats):
            for i in range(samples_in.shape[0]):
                z = samples_in[i:i + 1].to(device=vae.device, dtype=vae.vae_dtype)
                decode_rows(self.fsm, z, pixel_samples[i:i + 1], plan.rows, plan.workspace)
                del z
                vae.process_output(pixel_samples[i:i + 1])
        mm.soft_empty_cache()
        return pixel_samples

    # --- self-test -------------------------------------------------------------

    def selftest_params(self):
        return sum(p.numel() for p in self.module.parameters())

    def selftest_memory(self):
        """fp32 decoder copy + the larger of the reference (chunks(), layer-2 blocks of SELFTEST_REF_WORKSPACE: six
        copies of its largest activation, two workspaces) and this decode (its estimate in fp32) + both outputs + a margin."""
        g = self.geom
        n, t = SELFTEST_N, SELFTEST_T
        p1, p2, _, _, temps = cost(g, n, n, t, 4, SELFTEST_ROWS, eng.SELFTEST_WORKSPACE)
        frames = _frames(t, g.split)
        out = g.out_c * frames * (n * g.scale) ** 2 * 4
        big = max(max(ci, co) for res, _ in g.ups for ci, co in res) * min(CHUNK, frames) * (n * g.scale) ** 2 * 4
        big = max(big, g.c_in * up_frames(t) * (2 * n) ** 2 * 4)
        ref = 6 * big + 2 * eng.SELFTEST_REF_WORKSPACE
        return int(self.selftest_params() * 4 + max(ref, max(p1, p2) + temps) + 2 * out + eng.SELFTEST_MARGIN + MARGIN)

    def fp32_copy(self, device):
        """An AutoencoderKLCogVideoX holding only an fp32 copy of the decoder (rebuilt from the structure, the
        model's weights loaded; no encoder)."""
        dec = self.module
        g = self.geom
        chans = [res[0][1] for res, _ in g.ups]   # reversed block_out_channels
        with torch.device(device):
            d = cog.Decoder3D(in_channels=g.zc, out_channels=g.out_c, block_out_channels=tuple(reversed(chans)),
                              layers_per_block=len(dec.up_blocks[0].resnets) - 1, groups=dec.norm_out.norm_layer.num_groups,
                              eps=float(dec.mid_block.resnets[0].norm1.norm_layer.eps),
                              temporal_compression_ratio=int(self.fsm.temporal_compression_ratio))
        d.to(device=device, dtype=torch.float32)
        d.load_state_dict(dec.state_dict(), strict=True)
        d.eval()
        cp = cog.AutoencoderKLCogVideoX.__new__(cog.AutoencoderKLCogVideoX)
        torch.nn.Module.__init__(cp)
        cp.decoder = d
        cp.temporal_compression_ratio = self.fsm.temporal_compression_ratio
        cp.latent_channels = getattr(self.fsm, "latent_channels", g.zc)
        if structure_key(cp) != self.key:
            raise eng.StripeError("fp32 copy has another structure")
        return cp

    def self_test_run(self, vae):
        """vae_engine.self_test's run for this bound: chunks() (native's module calls, layer-2 blocks) against
        decode_rows (forced small slices and workspace: several slices at every level, 4 phase-2 chunks with the
        caches carried) on an fp32 copy; (ok, detail)."""
        t0 = time.perf_counter()
        dev = vae.device
        devs = [dev.index or 0] if dev.type == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=devs):
            cp = self.fp32_copy(dev)
            g = torch.Generator().manual_seed(0)
            z = torch.randn((1, self.geom.zc, SELFTEST_T, SELFTEST_N, SELFTEST_N), generator=g).to(dev)
            with OpChunking(cp.decoder, eng.SELFTEST_REF_WORKSPACE, OpStats()):
                ref = torch.cat(list(chunks(cp, z)), dim=2)
            out = torch.empty_like(ref)
            stats = OpStats()
            with OpChunking(cp.decoder, eng.SELFTEST_WORKSPACE, stats):
                decode_rows(cp, z, out, SELFTEST_ROWS, eng.SELFTEST_WORKSPACE)
            scale = max(1.0, float(ref.abs().max()))
            err = float((out - ref).abs().max()) / scale
            ok = bool(torch.isfinite(out).all()) and err <= eng.SELFTEST_TOL
            n_chunks = -(-int(ref.shape[2]) // CHUNK)
            detail = "max|rows - chunks()| / max(1, max|chunks()|) = {:.2g} (tolerance {:g}), {}-row slices x {} chunks, {:.2f}s".format(
                err, eng.SELFTEST_TOL, SELFTEST_ROWS, n_chunks, time.perf_counter() - t0)
            del cp, z, ref, out
        return ok, detail


def match(vae, samples, vae_options):
    """(bound, None) or (None, reason): the layer-1 decode takes a CogVideoX VAE with a multi-frame latent."""
    fsm = vae.first_stage_model
    reason = layer1_structure(fsm)
    if reason:
        return None, "CogVideoX layer 1: " + reason
    if vae_options:
        return None, "vae_options {} given".format(sorted(vae_options))
    if samples.ndim != 5 or samples.shape[2] < 2:
        return None, "CogVideoX layer 1 needs a latent with 2+ frames ({}); a single frame runs under layer 2".format(list(samples.shape))
    if samples.shape[1] != fsm.decoder.conv_in.conv.in_channels:
        return None, "latent has {} channels, conv_in expects {}".format(samples.shape[1], fsm.decoder.conv_in.conv.in_channels)
    return CogStripe(fsm), None
