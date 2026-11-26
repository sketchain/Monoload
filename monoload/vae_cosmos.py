"""Cosmos 1.0 VAE (comfy.ldm.cosmos.vae.CausalContinuousVideoTokenizer, the CV8x8x8 tokenizer, ComfyUI v0.46.3): the
multi-frame decode with everything after the first time upsampling streamed one frame at a time, causal-conv caches
carried between pieces (group D, docs/PLAN-video.md; DESIGN §9.40).

What native decode does (CausalContinuousVideoTokenizer.decode): un-normalise the latent (per latent frame index),
post_quant_conv (1x1x1), then DecoderFactorized on the whole video. With sd.py's config (channels 128, mult [2, 4, 4],
patch 4, spatial / temporal compression 8):

  latent res h x w:   conv_in (1x3x3 + 3x1x1) -> mid (resnet, spatial attention per frame + causal temporal
                      attention per pixel, resnet) -> up[2] (3 resnets, 512; its upsample is a no-op)
                      -> up[1] (3 resnets, 512) -> up[1].upsample: time x2, space x2 (CausalHybridUpsample3d)
  2h x 2w:            up[0] (3 resnets, 256) -> norm_out -> SiLU -> conv_out (1x3x3 + 3x1x1, 192 channels)
                      -> unpatcher3d: inverse haar, x4 in time and space, the first 3 frames dropped.

Everything is causal in time: CausalConv3d pads the front with copies of the first frame; CausalNormalize with one
group normalises each frame on its own; the temporal attention is masked (tril); the time upsampling repeats every
frame twice and drops the first frame of the whole video; the inverse haar maps each frame to 4 frames on its own.
Native's peak is the 2h part on the whole video (1280x704x121 simulated: 30 GiB; the whole-video decode under layer 2
alone 13 GiB, DESIGN §9.39).

Here (chunks()) the latent-resolution part up to the first time-upsampling block runs on the whole video exactly as
native (the same calls on the same tensors; it is small: 512 channels at h x w). From that block on, everything runs
depth first on one frame at a time: the upsampling gets one source frame (the first: as native, one frame out;
the others: repeated twice, as native's repeat_interleave), every later stage one frame. Each CausalConv3d there with
a time kernel > 1 keeps the last time_pad frames of its time-padded input (first call: native's first-frame copies);
the unpatcher drops its leading frames on the first piece only. Depth first means every module sees its frames in
time order. The pieces come out in output order, 4 pixel frames each (the first: 1), and vae.py copies each into
the output buffer (H3). Memory: the latent-resolution prefix for the whole video, the caches (time_pad frames of the
streamed convs' inputs, at 2h) and one frame's activations; linear in the area, as the layer-2 probe assumes.

Values: the same modules on the same frames; convs / GroupNorms over fewer frames may pick other kernels: fp-close,
not bit-identical (tests/test_vae_stream.py compares with native in fp32 on the real machine). Anything this module
does not recognise decodes whole as before.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.ldm.cosmos.vae as cvae
import comfy.ldm.cosmos.cosmos_tokenizer.layers3d as l3d
import comfy.ldm.cosmos.cosmos_tokenizer.utils as cutils

PIECE_FRAMES = 1   # frames per piece between stages (fixed: the probe's slope must not depend on the size)


def _cconv_ok(m):
    return type(m) is l3d.CausalConv3d and tuple(m.conv3d.stride)[0] == 1 and tuple(m.conv3d.dilation)[0] == 1


def _seq_ok(seq):
    return isinstance(seq, torch.nn.Sequential) and all(_cconv_ok(c) for c in seq)


def _resnet_ok(rb):
    if type(rb) is not l3d.CausalResnetBlockFactorized3d:
        return "{} block".format(type(rb).__name__)
    for n in (rb.norm1, rb.norm2):
        if type(n) is not cutils.CausalNormalize or n.num_groups != 1:
            return "a norm over all frames (num_groups != 1)"
    if not _seq_ok(rb.conv1) or not _seq_ok(rb.conv2):
        return "conv of another kind"
    if not (isinstance(rb.nin_shortcut, torch.nn.Identity) or _cconv_ok(rb.nin_shortcut)):
        return "shortcut of another kind"
    return None


def structure(fsm):
    """(reason, split): reason None when fsm is the Cosmos tokenizer this module streams; split is the up[] level
    whose upsample is the first one in time (streaming starts there)."""
    if type(fsm) is not cvae.CausalContinuousVideoTokenizer:
        return "not CausalContinuousVideoTokenizer ({})".format(type(fsm).__name__), None
    dec = getattr(fsm, "decoder", None)
    if type(dec) is not l3d.DecoderFactorized:
        return "decoder is {}".format(type(dec).__name__), None
    if type(fsm.post_quant_conv) is not l3d.CausalConv3d or fsm.post_quant_conv.time_pad != 0:
        return "post_quant_conv mixes frames", None
    split = None
    for i_level in reversed(range(dec.num_resolutions)):
        up = dec.up[i_level]
        if i_level != 0 and type(up.upsample) is l3d.CausalHybridUpsample3d and up.upsample.temporal_up:
            split = i_level
            break
    if split is None:
        return "no time upsampling", None
    us = dec.up[split].upsample
    if not (_cconv_ok(us.conv1) and _cconv_ok(us.conv2) and _cconv_ok(us.conv3)):
        return "upsampling conv of another kind", None
    for i_level in range(split):   # the streamed levels below split
        up = dec.up[i_level]
        if len(up.attn) > 0:
            return "attention at up[{}] (over all frames)".format(i_level), None
        for b in up.block:
            why = _resnet_ok(b)
            if why:
                return "up[{}]: {}".format(i_level, why), None
        if i_level != 0:
            u = up.upsample
            if type(u) is not l3d.CausalHybridUpsample3d or u.temporal_up or u.spatial_up:
                return "a second upsampling at up[{}]".format(i_level), None
    if type(dec.norm_out) is not cutils.CausalNormalize or dec.norm_out.num_groups != 1:
        return "norm_out over all frames", None
    if not _seq_ok(dec.conv_out):
        return "conv_out of another kind", None
    unp = dec.unpatcher3d
    if type(unp) is not l3d.UnPatcher3D or unp.patch_method not in ("haar", "rearrange"):
        return "unpatcher of another kind", None
    for m in dec.modules():   # an instance forward we did not put there (a user's hook): leave the decode alone
        if "forward" in m.__dict__ and not getattr(m.__dict__["forward"], "monoload_replacement", False):
            return "{} has its own forward".format(type(m).__name__), None
    return None, split


def _streamed_modules(dec, split):
    mods = [dec.up[split].upsample]
    for i_level in range(split):
        mods.append(dec.up[i_level])
    mods += [dec.norm_out, dec.conv_out]
    return mods


class _Stream:
    """Inside: every CausalConv3d with time_pad > 0 in the streamed part carries its cache (instance forwards,
    removed on exit)."""

    def __init__(self, mods):
        self.roots = mods
        self.cache = {}
        self.mods = []

    def _cconv(self, m):
        tp = m.time_pad

        def forward(x):
            k = id(m)
            prev = self.cache.pop(k, None)
            if prev is None:   # first call: native's first-frame copies (CausalConv3d._replication_pad)
                x = torch.cat([x[:, :, :1, ...].repeat(1, 1, tp, 1, 1), x], dim=2)
            else:
                x = torch.cat([prev, x], dim=2)
                del prev
            self.cache[k] = x[:, :, -tp:].clone()
            x = F.pad(x, m.spatial_pad + (0, 0), mode=m.pad_mode, value=0.0)
            return m.conv3d(x)
        forward.monoload_replacement = True
        return forward

    def __enter__(self):
        seen = set()
        for root in self.roots:
            for m in root.modules():
                if type(m) is l3d.CausalConv3d and m.time_pad > 0 and id(m) not in seen:
                    seen.add(id(m))
                    m.forward = self._cconv(m)
                    self.mods.append(m)
        return self

    def __exit__(self, *exc):
        for m in self.mods:
            m.__dict__.pop("forward", None)
        self.mods = []
        self.cache.clear()
        return False


def _upsample(us, x, first):
    """CausalHybridUpsample3d.forward on one source frame of the whole video (native repeats every frame twice and
    drops the first output frame: the first source frame gives one frame, every other two)."""
    if us.temporal_up:
        if not first:
            x = x.repeat_interleave(2, dim=2)
        x = us.conv1(x) + x
    if us.spatial_up:
        x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
        x = us.conv2(x) + x
    return us.conv3(x)


def _unpatch(unp, x, first):
    """UnPatcher3D.forward on one piece: the leading patch_size - 1 frames are the whole video's, dropped on the
    first piece only."""
    if first:
        return unp(x)
    if unp.patch_method == "haar":
        for _ in unp.range:
            x = unp._idwt(x, "haar", rescale=True)
        return x
    p = unp.patch_size
    return rearrange(x, "b (c p1 p2 p3) t h w -> b c (t p1) (h p2) (w p3)", p1=p, p2=p, p3=p)


def _stages(dec, split):
    st = []
    for i_level in reversed(range(split)):
        up = dec.up[i_level]
        st += list(up.block)
        if i_level != 0:
            st.append(up.upsample)   # a no-op (structure() checked)
    st.append(dec.norm_out)
    st.append(cutils.nonlinearity)
    st.append(dec.conv_out)
    return st


def _depth_first(stages, k, x, tail):
    """Outputs of stages[k:] + tail for the frames of x, one piece per PIECE_FRAMES frames at every stage. Iterative
    along a piece (each stage's input is dropped as soon as its output exists); a stage output with more frames is
    split and each part continues depth first."""
    while k < len(stages):
        x = stages[k](x)
        k += 1
        t = int(x.shape[2])
        if t > PIECE_FRAMES and k < len(stages):
            y, x = x, None
            for s in range(0, t, PIECE_FRAMES):
                yield from _depth_first(stages, k, y[:, :, s:s + PIECE_FRAMES], tail)
            return
    yield tail(x)


def _latent_in(fsm, z):
    """CausalContinuousVideoTokenizer.decode up to the decoder call (v0.46.3, the same lines)."""
    in_dtype = z.dtype
    latent_ch = z.shape[1]
    latent_t = z.shape[2]
    mean = fsm.latent_mean.view(latent_ch, -1)
    std = fsm.latent_std.view(latent_ch, -1)
    mean = mean.repeat(1, math.ceil(latent_t / mean.shape[-1]))[:, : latent_t].reshape([1, latent_ch, -1, 1, 1]).to(dtype=in_dtype, device=z.device)
    std = std.repeat(1, math.ceil(latent_t / std.shape[-1]))[:, : latent_t].reshape([1, latent_ch, -1, 1, 1]).to(dtype=in_dtype, device=z.device)
    z = z / fsm.sigma_data
    z = z * std + mean
    return fsm.post_quant_conv(z)


def chunks(fsm, z):
    """Pieces of fsm.decode(z) (z [1, C, T, h, w] on the device), consecutive along the frames (module docstring)."""
    reason, split = structure(fsm)
    if reason or z.ndim != 5 or z.shape[2] < 2:
        yield fsm.decode(z)
        return
    dec = fsm.decoder
    # the prefix, DecoderFactorized.forward up to up[split]'s upsample, on the whole video
    h = dec.conv_in(_latent_in(fsm, z))
    h = dec.mid.block_1(h)
    h = dec.mid.attn_1(h)
    h = dec.mid.block_2(h)
    for i_level in reversed(range(split, dec.num_resolutions)):
        for i_block in range(dec.num_res_blocks + 1):
            h = dec.up[i_level].block[i_block](h)
            if len(dec.up[i_level].attn) > 0:
                h = dec.up[i_level].attn[i_block](h)
        if i_level != split:
            h = dec.up[i_level].upsample(h)
    us = dec.up[split].upsample
    stages = _stages(dec, split)
    state = {"first_out": True}

    def tail(x):
        out = _unpatch(dec.unpatcher3d, x, state["first_out"])
        state["first_out"] = False
        return out

    calls = [0]

    def upsample(x):   # one source frame per call, in time order
        calls[0] += 1
        return _upsample(us, x, calls[0] == 1)

    stages = [upsample] + stages
    with _Stream(_streamed_modules(dec, split)):
        for i in range(int(h.shape[2])):
            for p in _depth_first(stages, 0, h[:, :, i:i + 1], tail):
                yield p
                del p
    del h
