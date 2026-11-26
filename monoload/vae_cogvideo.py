"""CogVideoX VAE (comfy.ldm.cogvideo.vae.AutoencoderKLCogVideoX, ComfyUI v0.46.3): the multi-frame decode as
pieces of output frames, with the last temporal upsampling streamed (group C, docs/PLAN-video.md; DESIGN §9.38).

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

import torch
import torch.nn.functional as F

import comfy.ldm.cogvideo.vae as cog

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


def output_frames(fsm, t):
    """Frames native decode makes from t latent frames (None if fsm is not a CogVideoX VAE this module knows)."""
    reason, split = structure(fsm)
    if reason:
        return None
    for _ in range(split):
        t = up_frames(t)
    return t


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
