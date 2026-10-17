"""Layer-1 adapter: the HunyuanImage 2.1 VAE decoder (comfy.ldm.hunyuan_video.vae.Decoder;
z 64, 32x) decoding an image (docs/DESIGN.md §9.24).

comfy.sd.VAE builds it for a state dict whose decoder.conv_in takes 64 channels:
an AutoencodingEngine (no post_quant_conv) with hunyuan_video.vae.Decoder
(block_out_channels [128, 256, 512, 512, 1024, 1024], reversed in the decoder,
num_res_blocks 2, ffactor_spatial 32). In 62b3c94 its forward is a plain
sequence of module calls, the stages from the lowest resolution up:

    x = conv_in(z) + z.repeat_interleave(C0 // z_channels, 1)
    -> mid.block_1 -> mid.attn_1 -> mid.block_2
    -> up[0].block[*] -> up[0].upsample -> up[1].block[*] -> ... -> up[5].block[*]
    -> norm_out -> F.silu -> conv_out

The blocks are the LDM ResnetBlock / AttnBlock (comfy.ldm.modules.diffusionmodules.model,
GroupNorm 32 groups); the upsampling is PixelUnshuffle2D (despite its name a
depth-to-space x2): a 3x3 conv to 4 x C_out channels rearranged 2x2 into rows,
plus the input with its channels repeated and rearranged the same way. Output
row 2h + i depends on conv output row h, i.e. input rows h - 1 .. h + 1: an
upsampling unit with halo 2 at its output level.

So this is the LDM layer 1 (monoload/vae_ldm.py) with another first step,
another upsampling unit and six levels: the prefix runs up to the last block
of the lowest resolution (H/32, the mid attention included), the GroupNorms
of the stripe part get whole-image statistics from statistics passes, and the
GroupNorm schemes are the LDM ones, by structure: A keeps nothing, D the
output of the first stripe level (H/16), B the output of every level below
full resolution (H/16, H/8, H/4, H/2), C B and every full-resolution block's
output. Tensors are 4D [B, C, H, W].
"""

import torch
import torch.nn.functional as F

import comfy.ldm.hunyuan_video.vae as hy
import comfy.ldm.hunyuan_video.vae_refiner as hyr
from comfy.ldm.models import autoencoder as ae
from comfy.ldm.modules.diffusionmodules import model as ldm

from . import vae_engine as eng
from . import vae_ldm
from .vae_engine import CONV, POINT, RES, NormRef, Unit, conv_extra, conv_io
from .vae_ldm import PART, SCHEMES, TIME_ATTN, TIME_COEF, TIME_PER_CALL, _check_resnet, _conv2d, _group_norm, _partial, _resnet_macs
from .vae_wan import attention_peak, attn_split

PU = "pixel_unshuffle"   # hunyuan_video.vae.PixelUnshuffle2D (scale 2, halo 2)


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def _check_pu(pu, c, name):
    """(error, output channels) of a PixelUnshuffle2D on c channels."""
    if type(pu) is not hy.PixelUnshuffle2D:
        return "{} is {}, not a PixelUnshuffle2D".format(name, type(pu).__name__), None
    e = _conv2d(pu.conv, 3, name + ".conv")
    if e:
        return e, None
    ci, co4 = conv_io(pu.conv)
    if ci != c or co4 % 4 or pu.scale * c != co4:
        return "{}: conv {} -> {} channels, scale {} on {} channels".format(name, ci, co4, pu.scale, c), None
    return None, co4 // 4


def hunyuan_structure(fsm):
    """(None, info) if fsm is a HunyuanImage 2.1 VAE the stripe decoder handles, else (reason, None)."""
    if not _is_hunyuan(fsm):
        return "first-stage model is {}, not comfy.ldm.models.autoencoder.AutoencodingEngine".format(type(fsm).__name__), None
    dec = getattr(fsm, "decoder", None)
    if type(dec) is not hy.Decoder:
        return "decoder is {}.{}, not comfy.ldm.hunyuan_video.vae.Decoder".format(type(dec).__module__, type(dec).__name__), None
    e = _conv2d(dec.conv_in, 3, "decoder.conv_in")
    if e:
        return e, None
    zc, c = conv_io(dec.conv_in)
    boc = list(dec.block_out_channels)
    if zc != dec.z_channels or not boc or boc[0] != c or c % zc:
        return "decoder.conv_in {} -> {} with z_channels {}, block_out_channels {}".format(zc, c, dec.z_channels, boc), None
    for nm in ("block_1", "block_2"):
        e = _check_resnet(getattr(dec.mid, nm), "decoder.mid." + nm)
        if e:
            return e, None
        if getattr(dec.mid, nm).in_channels != c or getattr(dec.mid, nm).out_channels != c:
            return "decoder.mid.{} on {} channels".format(nm, c), None
    at = dec.mid.attn_1
    if type(at) is not ldm.AttnBlock:
        return "decoder.mid.attn_1 is {}, not an AttnBlock".format(type(at).__name__), None
    e = _group_norm(at.norm, c, "decoder.mid.attn_1.norm")
    if e:
        return e, None
    for nm in ("q", "k", "v", "proj_out"):
        e = _conv2d(getattr(at, nm), 1, "decoder.mid.attn_1." + nm)
        if e:
            return e, None
    if len(dec.up) < 2:
        return "decoder with {} up stages".format(len(dec.up)), None
    for i, stage in enumerate(dec.up):
        if len(stage.block) != dec.num_res_blocks + 1:
            return "decoder.up[{}] has {} blocks".format(i, len(stage.block)), None
        for j, rb in enumerate(stage.block):
            name = "decoder.up[{}].block[{}]".format(i, j)
            e = _check_resnet(rb, name)
            if e:
                return e, None
            if rb.in_channels != c:
                return "{} takes {} channels, gets {}".format(name, rb.in_channels, c), None
            c = rb.out_channels
        if hasattr(stage, "upsample"):
            e, c = _check_pu(stage.upsample, c, "decoder.up[{}].upsample".format(i))
            if e:
                return e, None
        if hasattr(stage, "attn"):
            return "decoder.up[{}] has attention".format(i), None
    if not hasattr(dec.up[0], "upsample"):
        return "decoder.up[0] has no upsample", None
    e = _group_norm(dec.norm_out, c, "decoder.norm_out") or _conv2d(dec.conv_out, 3, "decoder.conv_out")
    if e:
        return e, None
    if conv_io(dec.conv_out)[0] != c:
        return "decoder.conv_out takes {} channels, gets {}".format(conv_io(dec.conv_out)[0], c), None
    h = eng.hooked_module(fsm)
    if h:
        return "module {} has a forward hook or an instance-level forward".format(h), None
    return None, {}


class ConvIn:
    """The decoder's first line, on its own modules, same operations and order:
    conv_in(z) + z.repeat_interleave(C0 // z_channels, 1). A prefix module."""

    def __init__(self, dec):
        self.dec = dec
        self.zc, self.c = conv_io(dec.conv_in)

    def __call__(self, z):
        d = self.dec
        return d.conv_in(z) + z.repeat_interleave(d.block_out_channels[0] // d.z_channels, 1)


def build_units(fsm):
    """(prefix modules, stripe units)."""
    dec = fsm.decoder
    prefix = [("decoder.conv_in+repeat", ConvIn(dec)), ("decoder.mid.block_1", dec.mid.block_1),
              ("decoder.mid.attn_1", dec.mid.attn_1), ("decoder.mid.block_2", dec.mid.block_2)]
    prefix += [("decoder.up.0.block.{}".format(j), rb) for j, rb in enumerate(dec.up[0].block)]
    units = []

    def upsample(i):
        u = dec.up[i].upsample
        ci, co4 = conv_io(u.conv)
        # contiguous: the row slice is copied before the call (the conv would copy it under the layer-2 wrapper anyway)
        units.append(Unit(PU, u, 2, 2, ci, co4 // 4, "decoder.up.{}.upsample".format(i), 9 * ci * co4 // 4, contiguous=True))
    upsample(0)
    for i in range(1, len(dec.up)):
        for j, rb in enumerate(dec.up[i].block):
            name = "decoder.up.{}.block.{}".format(i, j)
            ci, co = rb.in_channels, rb.out_channels
            part = Unit(PART, _partial(rb), 1, 1, ci, co, name + "[:conv1]", 9 * ci * co)
            units.append(Unit(RES, rb, 2, 1, ci, co, name, _resnet_macs(rb), ci != co,
                              norms=(NormRef(rb.norm1, name + ".norm1"), NormRef(rb.norm2, name + ".norm2", part)), contiguous=ci != co))
        if hasattr(dec.up[i], "upsample"):
            upsample(i)
    c = dec.norm_out.num_channels
    units.append(Unit(POINT, dec.norm_out, 0, 1, c, c, "decoder.norm_out", norms=(NormRef(dec.norm_out, "decoder.norm_out"),)))
    units.append(Unit(POINT, F.silu, 0, 1, c, c, "decoder.silu"))
    cin, cout = conv_io(dec.conv_out)
    units.append(Unit(CONV, dec.conv_out, 1, 1, cin, cout, "decoder.conv_out", 9 * cin * cout))
    return prefix, units


def scheme_positions(units, name):
    """Save positions of a GroupNorm scheme (vae_ldm.scheme_positions, upsampling units by scale)."""
    ups = [i for i, u in enumerate(units) if u.scale > 1]
    level_ends = ups[1:]
    full = [i + 1 for i, u in enumerate(units) if u.kind == RES and i > ups[-1]]
    return {"A": [], "D": level_ends[:1], "B": level_ends, "C": level_ends + full}[name]


# ---------------------------------------------------------------------------
# memory model (bytes), as vae_ldm, plus:
#   PixelUnshuffle2D  (row slice made contiguous first: S + A, then A is the input's storage)
#                     conv: A + its output Y (4 x C_out channels at the input rows) + extra
#                     | the rearranged copy: A + 2Y | repeat_interleave R (= Y) and its
#                     rearranged copy, next to y: A + 3Y | y + r: A + 3Y
#   ConvIn            conv_in output + the repeated latent + the sum: 3 planes of C0 (+ the conv's extra)
# ---------------------------------------------------------------------------

def pu_peak(cin, cout, r, w, e, ws, s):
    a = r * w * cin * e
    y = r * w * 4 * cout * e
    return max(s + a, a + max(y + conv_extra(cin, 4 * cout, 3, r, r, w, w, e, ws), 3 * y))


def unit_peak(u, r, w, e, ws, s):
    if u.kind == PU:
        return pu_peak(u.cin, u.cout, r, w, e, ws, s)
    return vae_ldm.unit_peak(u, r, w, e, ws, s)


def unit_largest(u, r, w, e, ws):
    if u.kind == PU:
        return max(r * w * u.cin * e, r * w * 4 * u.cout * e, min(ws, 9 * u.cin * r * w * e))
    return vae_ldm.unit_largest(u, r, w, e, ws)


def unit_calls(u, r, w, e, ws):
    if u.kind == PU:
        return vae_ldm.conv_calls(u.cin, r, w, e, ws)
    return vae_ldm.unit_calls(u, r, w, e, ws)


def prefix_peak(m, h, w, e, ws, s):
    if isinstance(m, ConvIn):
        return s + 3 * h * w * m.c * e + conv_extra(m.zc, m.c, 3, h, h, w, w, e, ws)
    if isinstance(m, ldm.AttnBlock):
        return attention_peak(m.in_channels, h, w, e, ws, s, attn_split(m))
    return vae_ldm.prefix_peak(m, h, w, e, ws, s)


def prefix_largest(m, h, w, e, ws):
    if isinstance(m, ConvIn):
        return max(h * w * m.c * e, min(ws, 9 * m.zc * h * w * e))
    return vae_ldm.prefix_largest(m, h, w, e, ws)


def prefix_out_channels(m):
    if isinstance(m, ConvIn):
        return m.c
    return vae_ldm.prefix_out_channels(m)


def prefix_macs(m):
    if isinstance(m, ConvIn):
        return 9 * m.zc * m.c
    return vae_ldm.prefix_macs(m)


# ---------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------

class HunyuanStripe(eng.StripeAdapter):
    """Layer-1 adapter bound to one HunyuanImage 2.1 first-stage model."""

    hdim = 2
    schemes = SCHEMES
    # the self-test's decoder input: 12 x 12 (output 384 x 384, ~10 stripes of SELFTEST_ROWS with inner boundaries at
    # every level); the engine's 24 would be a 768 x 768 fp32 decode at 32x (~2.6 GiB)
    selftest_size = 12
    # the layer-2 bound (vae.estimate) and the self-test's reference decode: each PixelUnshuffle2D makes five tensors of
    # its output's size (conv output, its rearranged copy, the repeated input, its rearranged copy, the sum), up to three
    # alive next to its input, and the caching allocator keeps their segments into the full-resolution level, where the
    # ResnetBlocks need blocks of their own: tests/alloc_sim.py, 4K layer 2 reserved 11.1 GiB = 5.6 x the largest
    # activation (1.99 GiB) + the workspace; 4 copies (10.06 GiB) are not an upper bound
    activation_copies = 6

    def __init__(self, fsm, gn_scheme=None, module=None):
        self.fsm = fsm
        self.module = fsm if module is None else module
        self.prefix, self.units = build_units(fsm)
        self.scale = 1
        for u in self.units:
            self.scale *= u.scale
        self.ckpt_channels = prefix_out_channels(self.prefix[-1][1])
        self.out_channels = self.units[-1].cout
        self.gn_scheme = gn_scheme or vae_ldm.scheme()
        self.saves = scheme_positions(self.units, self.gn_scheme)
        self.name = "HunyuanImage 2.1 stripes, GroupNorm scheme {}".format(self.gn_scheme)
        dec = fsm.decoder
        self.key = ("HunyuanImage2.1", conv_io(dec.conv_in), tuple(dec.block_out_channels), int(dec.num_res_blocks),
                    tuple((u.kind, u.cin, u.cout) for u in self.units), self.gn_scheme)

    unit_peak = staticmethod(unit_peak)
    unit_calls = staticmethod(unit_calls)
    unit_largest = staticmethod(unit_largest)
    prefix_peak = staticmethod(prefix_peak)
    prefix_largest = staticmethod(prefix_largest)
    prefix_out_channels = staticmethod(prefix_out_channels)
    prefix_macs = staticmethod(prefix_macs)

    def save_positions(self):
        return self.saves

    def variants(self, scheme=None):
        names = [scheme] if scheme else [self.gn_scheme] + [n for n in SCHEMES if n != self.gn_scheme]
        return [self if n == self.gn_scheme else HunyuanStripe(self.fsm, gn_scheme=n) for n in names]

    def predict_seconds(self, plan):
        """vae_ldm's time model (fitted to SDXL on CT 700), for ranking the schemes of one decode: the levels mapped by
        their resolution relative to the output (full resolution = SDXL's level 8, H/2 = 4, H/4 and below = 2). Not
        calibrated for this decoder: absolute seconds are rough."""
        top = TIME_COEF[max(TIME_COEF)]
        f = self.scale // 8
        macs = sum(TIME_COEF.get(max(2, lv // f), top) * m for lv, m in plan.work_levels.items() if lv > 1)
        tokens = plan.heights[0] * plan.widths[0]
        return macs / 1e12 + TIME_PER_CALL * plan.conv_calls + TIME_ATTN * 2 * self.ckpt_channels * tokens * tokens / 1e12

    def output_shape(self, samples):
        return (samples.shape[0], self.out_channels, int(samples.shape[-2]) * self.scale, int(samples.shape[-1]) * self.scale)

    # self-test

    def selftest_params(self):
        return sum(p.numel() for p in self.fsm.decoder.parameters())

    def selftest_latent(self, n):
        return (1, conv_io(self.fsm.decoder.conv_in)[0], n, n)

    def fp32_copy(self, device):
        """The decoder rebuilt from the model's configuration in fp32 with the model's
        weights (the encoder is not copied); freed after the self-test."""
        dec = self.fsm.decoder
        boc = list(dec.block_out_channels)[::-1]          # the constructor's order (the decoder keeps it reversed)
        depth = sum(1 for s in dec.up if hasattr(s, "upsample"))
        holder = torch.nn.Module()
        with torch.device(device):
            holder.decoder = hy.Decoder(z_channels=dec.z_channels, out_channels=conv_io(dec.conv_out)[1], block_out_channels=boc,
                                        num_res_blocks=int(dec.num_res_blocks), ffactor_spatial=2 ** depth,
                                        upsample_match_channel=True)
        holder.to(device=device, dtype=torch.float32)
        holder.decoder.load_state_dict(dec.state_dict(), strict=True)
        holder.eval()
        shim = _Shim(holder)
        reason, _ = hunyuan_structure(shim)
        if reason:
            raise eng.StripeError("fp32 copy does not match: " + reason)
        return _CopyStripe(shim, self.gn_scheme, holder)


class _CopyStripe(HunyuanStripe):
    """The adapter on the fp32 copy; its reference is AutoencodingEngine.decode."""

    def __init__(self, shim, gn_scheme, holder):
        super().__init__(shim, gn_scheme, module=holder)

    def reference_decode(self, z):
        return ae.AutoencodingEngine.decode(self.module, z)


# The HunyuanImage 2.1 Refiner's decoder (vae_refiner.Decoder with refiner_vae=False: Conv3d, GroupNorm; an image
# decoded as 4 frames, DESIGN §9.25 / §9.30): its UpSmpl depth-to-space makes the same five output-sized tensors as
# PixelUnshuffle2D, and every level's activation is 2-8x the one below, so each level's cached segments are too small
# for the next one: tests/alloc_sim.py, layer 2 reserved up to 5.67 x the largest activation + 2 x the workspace
# (1024-3072 wide, 768-2560 high, workspace 128 MiB - 1 GiB, bf16 / fp32); 4 copies are not an upper bound
# (2048x2048: 23.4 GiB simulated, 18.05 estimated)
REFINER_ACTIVATION_COPIES = 6


def decoder_terms(vae, samples):
    """The layer-2 estimate's terms (vae._decoder_terms) for any hunyuan_video.vae.Decoder and the Refiner's
    vae_refiner.Decoder (refiner_vae=False), by their class: a forward hook or vae_options keep the stripe decoder off
    it (match), not the depth-to-space copies of its own decode. None for others (HunyuanVideo 1.5's vae_refiner.Decoder
    with refiner_vae=True included)."""
    dec = getattr(getattr(vae, "first_stage_model", None), "decoder", None)
    if type(dec) is hy.Decoder:
        return {"copies": HunyuanStripe.activation_copies}
    if type(dec) is hyr.Decoder and dec.refiner_vae is False:
        return {"copies": REFINER_ACTIVATION_COPIES}
    return None


def match(vae, samples, vae_options):
    """(bound adapter, None) or (None, reason)."""
    fsm = vae.first_stage_model
    reason, _ = hunyuan_structure(fsm)
    if reason:
        return None, reason
    if vae_options:
        return None, "vae_options {} given".format(sorted(vae_options))
    if samples.ndim != 4:
        return None, "latent has {} dims".format(samples.ndim)
    if samples.shape[1] != fsm.decoder.z_channels:
        return None, "latent has {} channels, the decoder takes {}".format(samples.shape[1], fsm.decoder.z_channels)
    return HunyuanStripe(fsm), None


class _Shim:
    """Duck-types the attributes hunyuan_structure / HunyuanStripe read from the first-stage model."""

    def __init__(self, holder):
        self.decoder = holder.decoder
        self._holder = holder

    def named_modules(self):
        return self._holder.named_modules()


def _is_hunyuan(fsm):
    return type(fsm) is ae.AutoencodingEngine or isinstance(fsm, _Shim)


def is_hunyuan(fsm):
    """Whether fsm has the HunyuanImage 2.1 decoder this adapter handles (the Info node's scheme hints)."""
    return hunyuan_structure(fsm)[0] is None if fsm is not None else False


has_schemes = is_hunyuan   # the GroupNorm schemes apply to it (vae.gn_schemes_apply)
