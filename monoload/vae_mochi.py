"""Mochi VAE (comfy.ldm.genmo.vae.model.VideoVAE, ComfyUI v0.46.3): the multi-frame decode streamed one frame at a
time through the decoder, causal-conv caches carried between pieces (group D, docs/PLAN-video.md; DESIGN §9.40).

What native decode does (VideoVAE.decode -> Decoder.forward): the whole video at once through

  blocks[0]   Conv3d 1x1x1 (12 -> 768) + 3 ResBlocks at h x w
  blocks[1]   CausalUpsampleBlock: 6 ResBlocks (768) at h, Conv1x1 768 -> 512*3*4, DepthToSpaceTime (time x3, x2)
  blocks[2]   CausalUpsampleBlock: 4 ResBlocks (512) at 2h, Conv1x1, DepthToSpaceTime (time x2, x2)
  blocks[3]   CausalUpsampleBlock: 3 ResBlocks (256) at 4h, Conv1x1, DepthToSpaceTime (time x1, x2)
  blocks[4]   3 ResBlocks (128) at 8h; then SiLU, Conv1x1 128 -> 3.

No attention in the decoder (has_attention all False). Every ResBlock is GroupNormSpatial (per frame: each frame's
own statistics) -> SiLU -> PConv3d 3x3x3, twice. PConv3d is causal in time: it pads the front with copies of the
first frame (replicate) and nothing at the back, so an output frame reads its own input frame and the two before.
DepthToSpaceTime maps each input frame to texp output frames and drops the first texp - 1 frames of the whole video
(t latent frames -> 6t - 5). Nothing else looks across frames. Native keeps everything for the whole video at full
resolution (848x480x85 simulated: 362 GiB; the whole-video decode under layer 2 alone still 59 GiB, DESIGN §9.32).

Here (chunks()) the same module calls run depth first on one frame at a time: the latent's frames one by one, and
each stage's output split into single frames before the next stage, so a stage always gets one frame. Each PConv3d
with a time kernel > 1 keeps the last (kernel - 1) frames of its padded input (its first call pads with the first
frame as native does; later calls put the kept frames in front instead): its output frame is native's. Each
DepthToSpaceTime drops its texp - 1 leading frames on its first call only. Depth first means every module sees its
frames in time order. The pieces come out in output order, one pixel frame each, and vae.py copies each into the
output buffer (H3). Memory: the caches (2 frames of every causal conv's input, ~3.9 KB per output pixel in a 2-byte
dtype, most of it the 6 convs at full resolution) plus one frame's activations; linear in the area and almost
independent of the frame count, as the layer-2 probe (vae._probe_mf) assumes.

Values: the same modules on the same frames; a conv / GroupNorm over 3 (or 1) frames instead of the whole video may
pick other kernels: fp-close, not bit-identical (tests/test_vae_stream.py compares with native in fp32 on the
real machine). Anything this module does not recognise decodes whole as before.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.ldm.genmo.vae.model as genmo

PIECE_FRAMES = 1   # frames per piece between stages (fixed: the probe's slope must not depend on the size)


def _plain_conv_ok(m):
    return isinstance(m, torch.nn.Conv3d) and tuple(m.kernel_size)[0] == 1 and tuple(m.stride)[0] == 1


def _pconv_ok(m):
    return (m.causal and tuple(m.stride)[0] == 1 and tuple(m.dilation)[0] == 1 and tuple(m.padding)[0] == 0
            and m.padding_mode in ("replicate", "zeros"))


def _resblock_ok(rb):
    if type(rb) is not genmo.ResBlock or not isinstance(rb.attn_block, torch.nn.Identity):
        return "ResBlock with attention or of another type ({})".format(type(rb).__name__)
    for m in rb.stack:
        if isinstance(m, genmo.PConv3d):
            if not _pconv_ok(m):
                return "PConv3d not causal / strided in time"
        elif not isinstance(m, (genmo.GroupNormSpatial, torch.nn.SiLU)):
            return "ResBlock layer {}".format(type(m).__name__)
    return None


def _stages(dec):
    """The decoder as a flat list of callables applied in order, each either a module (a ResBlock, a 1x1x1 conv, a
    Conv1x1) or a DepthToSpaceTime; (None, reason) when it is not the decoder this module knows."""
    out = []
    for i, blk in enumerate(dec.blocks):
        if isinstance(blk, torch.nn.Sequential):
            for m in blk:
                if isinstance(m, genmo.ResBlock):
                    why = _resblock_ok(m)
                    if why:
                        return None, "blocks[{}]: {}".format(i, why)
                elif not _plain_conv_ok(m):
                    return None, "blocks[{}]: {}".format(i, type(m).__name__)
                out.append(m)
        elif type(blk) is genmo.CausalUpsampleBlock:
            for m in blk.blocks:
                why = _resblock_ok(m)
                if why:
                    return None, "blocks[{}]: {}".format(i, why)
                out.append(m)
            if not isinstance(blk.proj, genmo.Conv1x1) or type(blk.d2st) is not genmo.DepthToSpaceTime:
                return None, "blocks[{}]: upsampling of another kind".format(i)
            out.append(blk.proj)
            out.append(blk.d2st)
        else:
            return None, "blocks[{}] is {}".format(i, type(blk).__name__)
    if dec.output_nonlinearity not in ("silu", None, ""):
        return None, "output nonlinearity {}".format(dec.output_nonlinearity)
    if not isinstance(dec.output_proj, genmo.Conv1x1):
        return None, "output_proj is {}".format(type(dec.output_proj).__name__)
    return out, None


def structure(fsm):
    """(reason, stages): reason None when fsm is the Mochi VAE this module streams."""
    if type(fsm) is not genmo.VideoVAE:
        return "not genmo VideoVAE ({})".format(type(fsm).__name__), None
    dec = getattr(fsm, "decoder", None)
    if type(dec) is not genmo.Decoder:
        return "decoder is {}".format(type(dec).__name__), None
    stages, why = _stages(dec)
    if why:
        return why, None
    for m in dec.modules():   # an instance forward we did not put there (a user's hook): leave the decode alone
        if "forward" in m.__dict__ and not getattr(m.__dict__["forward"], "monoload_replacement", False):
            return "{} has its own forward".format(type(m).__name__), None
    return None, stages


class _Stream:
    """Inside: every PConv3d with a time kernel > 1 of the decoder carries its cache, every DepthToSpaceTime trims on
    its first call only (instance forwards, removed on exit)."""

    def __init__(self, dec):
        self.dec = dec
        self.cache = {}
        self.seen = set()
        self.mods = []

    def _pconv(self, m):
        ctx = m.kernel_size[0] - 1
        mode = "constant" if m.padding_mode == "zeros" else m.padding_mode
        base = super(genmo.PConv3d, m).forward

        def forward(x):
            k = id(m)
            prev = self.cache.pop(k, None)
            if prev is None:   # first call: native's front padding
                xp = F.pad(x, (0, 0, 0, 0, ctx, 0), mode=mode)
            else:
                xp = torch.cat([prev, x], dim=2)
                del prev
            self.cache[k] = xp[:, :, -ctx:].clone()
            return base(xp)
        forward.monoload_replacement = True
        return forward

    def _d2st(self, m):
        st, s = m.temporal_expansion, m.spatial_expansion
        orig = type(m).forward.__get__(m)

        def forward(x):
            k = id(m)
            if k not in self.seen:
                self.seen.add(k)
                return orig(x)
            return rearrange(x, "B (C st sh sw) T H W -> B C (T st) (H sh) (W sw)", st=st, sh=s, sw=s)
        forward.monoload_replacement = True
        return forward

    def __enter__(self):
        for m in self.dec.modules():
            if isinstance(m, genmo.PConv3d) and m.kernel_size[0] > 1:
                m.forward = self._pconv(m)
                self.mods.append(m)
            elif type(m) is genmo.DepthToSpaceTime and m.temporal_expansion > 1:
                m.forward = self._d2st(m)
                self.mods.append(m)
        return self

    def __exit__(self, *exc):
        for m in self.mods:
            m.__dict__.pop("forward", None)
        self.mods = []
        self.cache.clear()
        self.seen.clear()
        return False


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


def chunks(fsm, z):
    """Pieces of fsm.decode(z) (z [1, C, T, h, w] on the device), consecutive along the frames (module docstring)."""
    reason, stages = structure(fsm)
    if reason or z.ndim != 5:
        yield fsm.decode(z)
        return
    dec = fsm.decoder

    def tail(x):   # Decoder.forward after the blocks
        if dec.output_nonlinearity == "silu":
            x = F.silu(x, inplace=not dec.training)
        return dec.output_proj(x).contiguous()

    with _Stream(dec):
        for i in range(int(z.shape[2])):
            for p in _depth_first(stages, 0, z[:, :, i:i + 1], tail):
                yield p
                del p
