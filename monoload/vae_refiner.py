"""Layer-1 adapter: the HunyuanImage 2.1 Refiner VAE decoder (comfy.ldm.hunyuan_video.vae_refiner.Decoder with
refiner_vae=False; z 32, 16x) decoding an image (docs/DESIGN.md §9.25, §9.31).

comfy.sd.VAE builds it for a state dict whose decoder.conv_in is 5D with 32 input channels: an AutoencodingEngine
with vae_refiner.Decoder (block_out_channels [128, 256, 512, 1024, 1024], reversed in the decoder, num_res_blocks 2,
ffactor_spatial 16, ffactor_temporal 4, refiner_vae False), latent_dim 3. With refiner_vae False the convs are plain
ops.Conv3d (3x3x3, padding 1: zeros on both sides in time, not causal), the norms GroupNorm (32 groups; over
C/32 x T x H x W: across the frames), the mid attention the LDM AttnBlock on 5D. In 62b3c94 an image (latent T=1)
decodes as a plain sequence of module calls, the stages from the lowest resolution up:

    x = conv_in(z) + z.repeat_interleave(C0 // z_channels, 1)              H/16, T=1
    -> mid.block_1 -> mid.attn_1 -> mid.block_2 -> up[0].block[*]           T=1
    -> up[0].upsample (time x2) -> up[1].block[*] -> up[1].upsample (x2)    H/8 T=2, H/4 T=4
    -> up[2].block[*] -> up[2].upsample -> ... -> up[4].block[*]            T=4
    -> norm_out -> F.silu -> conv_out -> [:, :, -1:]                        3 x 4 frames -> the last

UpSmpl: a 3x3x3 conv to (time r1 x 2 x 2) x C_out channels, rearranged into frames / rows / columns, plus the input
with its channels repeated and rearranged the same way (no first-frame special case with refiner_vae False). Output
row 2h + i depends on conv output row h, i.e. input rows h - 1 .. h + 1: an upsampling unit with halo 2 at its
output level, like HunyuanImage 2.1's PixelUnshuffle2D.

So an image is decoded as 4 frames and the last one is kept; the time convs mix the frames and the GroupNorm
statistics span them. The stripes cut only H and carry every frame (Unit.tscale: the planes per row, 1 -> 2 -> 4
-> 1 at the last step), so the time convs and their zero padding run whole inside each stripe; GroupNorm gets
whole-image statistics (all frames) from the statistics passes, with the LDM schemes by structure: A keeps
nothing, D the output of the first stripe level (H/8), B every level below full resolution (H/8, H/4, H/2), C B
and every full-resolution block's output. Tensors are 5D [B, C, T, H, W].
"""

import fractions

import torch
import torch.nn.functional as F

import comfy.ldm.hunyuan_video.vae_refiner as hyr
from comfy.ldm.models import autoencoder as ae
from comfy.ldm.modules.diffusionmodules import model as ldm

from . import vae_engine as eng
from . import vae_ldm
from .vae_engine import CONV, POINT, RES, NormRef, Unit, conv_io, norm_temp
from .vae_ldm import PART, SCHEMES, TIME_ATTN, TIME_COEF, TIME_PER_CALL, _group_norm, _partial
from .vae_wan import attention_peak, attn_split, no_dropout

UPS = "upsmpl"       # vae_refiner.UpSmpl (scale 2, halo 2, planes x2 when it upsamples in time)
LAST = "last_frame"  # out[:, :, -1:] at the end of Decoder.forward for a one-frame latent


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def _conv3d(m, k, name, sliced=False):
    """None if m is a stride-1 k x k x k Conv3d with padding k // 2 (zeros); sliced: also its middle time slice
    (kernel 1 x k x k, no time padding: the fp32 copy's convs that only see one frame)."""
    if not isinstance(m, torch.nn.Conv3d):
        return "{} is {}, not a Conv3d".format(name, type(m).__name__)
    p = k // 2
    shapes = [((k, k, k), (p, p, p))] + ([((1, k, k), (0, p, p))] if sliced and k > 1 else [])
    if ((tuple(m.kernel_size), tuple(m.padding) if isinstance(m.padding, tuple) else m.padding) not in shapes
            or tuple(m.stride) != (1, 1, 1) or tuple(m.dilation) != (1, 1, 1) or m.groups != 1 or m.padding_mode != "zeros"):
        return "{}: kernel {} stride {} padding {} dilation {} groups {} {} (expected {}x{}x{}, stride 1, padding {}, zeros)".format(
            name, tuple(m.kernel_size), tuple(m.stride), m.padding, tuple(m.dilation), m.groups, m.padding_mode, k, k, k, p)
    return None


def _check_resnet(rb, name, sliced=False):
    if type(rb) is not ldm.ResnetBlock:
        return "{} is {}, not a ResnetBlock".format(name, type(rb).__name__)
    ci, co = rb.in_channels, rb.out_channels
    e = (_group_norm(rb.norm1, ci, name + ".norm1") or _group_norm(rb.norm2, co, name + ".norm2")
         or _conv3d(rb.conv1, 3, name + ".conv1", sliced) or _conv3d(rb.conv2, 3, name + ".conv2", sliced))
    if e:
        return e
    if conv_io(rb.conv1) != (ci, co) or conv_io(rb.conv2) != (co, co):
        return "{}: conv channels {} / {} for {} -> {}".format(name, conv_io(rb.conv1), conv_io(rb.conv2), ci, co)
    if type(rb.swish) is not torch.nn.SiLU:
        return "{}.swish is {}".format(name, type(rb.swish).__name__)
    e = no_dropout(rb.dropout)
    if e:
        return "{}.dropout: {}".format(name, e)
    if ci != co:
        if rb.use_conv_shortcut or not hasattr(rb, "nin_shortcut"):
            return "{}: 3x3x3 conv shortcut".format(name)
        e = _conv3d(rb.nin_shortcut, 1, name + ".nin_shortcut")
        if e:
            return e
    return None


def _check_upsmpl(up, c, name, sliced=False):
    """(error, output channels, planes factor) of an UpSmpl on c channels."""
    if type(up) is not hyr.UpSmpl:
        return "{} is {}, not an UpSmpl".format(name, type(up).__name__), None, None
    if up.refiner_vae is not False:
        return "{}: refiner_vae {}".format(name, up.refiner_vae), None, None
    e = _conv3d(up.conv, 3, name + ".conv", sliced)
    if e:
        return e, None, None
    ci, cf = conv_io(up.conv)
    fct = 8 if up.tus else 4
    if ci != c or cf % fct or up.rp * ci != cf:
        return "{}: conv {} -> {} channels, repeat {} on {} channels (time upsampling {})".format(name, ci, cf, up.rp, c, bool(up.tus)), None, None
    return None, cf // fct, 2 if up.tus else 1


def _is_engine(fsm):
    return type(fsm) is ae.AutoencodingEngine or isinstance(fsm, _Shim)


def refiner_structure(fsm, sliced=False):
    """(None, info) if fsm is a HunyuanImage 2.1 Refiner VAE the stripe decoder handles, else (reason, None).
    info: {"planes": planes at the end of the decoder (before the last-frame step), "t1": names of the 3x3x3 convs
    that only see one frame}. sliced: those convs may be their middle time slice (the fp32 copy)."""
    if not _is_engine(fsm):
        return "first-stage model is {}, not comfy.ldm.models.autoencoder.AutoencodingEngine".format(type(fsm).__name__), None
    dec = getattr(fsm, "decoder", None)
    if type(dec) is not hyr.Decoder:
        return "decoder is {}.{}, not comfy.ldm.hunyuan_video.vae_refiner.Decoder".format(type(dec).__module__, type(dec).__name__), None
    if dec.refiner_vae is not False:
        return "decoder.refiner_vae is {} (HunyuanVideo 1.5: causal convs, RMS norm)".format(dec.refiner_vae), None
    t1 = []
    e = _conv3d(dec.conv_in, 3, "decoder.conv_in", sliced)
    if e:
        return e, None
    t1.append("conv_in")
    zc, c = conv_io(dec.conv_in)
    boc = list(dec.block_out_channels)
    if zc != dec.z_channels or not boc or boc[0] != c or c % zc:
        return "decoder.conv_in {} -> {} with z_channels {}, block_out_channels {}".format(zc, c, dec.z_channels, boc), None
    for nm in ("block_1", "block_2"):
        rb = getattr(dec.mid, nm)
        e = _check_resnet(rb, "decoder.mid." + nm, sliced)
        if e:
            return e, None
        if rb.in_channels != c or rb.out_channels != c:
            return "decoder.mid.{} on {} channels".format(nm, c), None
        t1 += ["mid.{}.conv1".format(nm), "mid.{}.conv2".format(nm)]
    at = dec.mid.attn_1
    if type(at) is not ldm.AttnBlock:
        return "decoder.mid.attn_1 is {}, not an AttnBlock".format(type(at).__name__), None
    e = _group_norm(at.norm, c, "decoder.mid.attn_1.norm")
    if e:
        return e, None
    for nm in ("q", "k", "v", "proj_out"):
        e = _conv3d(getattr(at, nm), 1, "decoder.mid.attn_1." + nm)
        if e:
            return e, None
    if len(dec.up) < 2:
        return "decoder with {} up stages".format(len(dec.up)), None
    t = 1
    for i, stage in enumerate(dec.up):
        if len(stage.block) != dec.num_res_blocks + 1:
            return "decoder.up[{}] has {} blocks".format(i, len(stage.block)), None
        for j, rb in enumerate(stage.block):
            name = "decoder.up[{}].block[{}]".format(i, j)
            e = _check_resnet(rb, name, sliced and t == 1)
            if e:
                return e, None
            if rb.in_channels != c:
                return "{} takes {} channels, gets {}".format(name, rb.in_channels, c), None
            c = rb.out_channels
            if t == 1:
                t1 += ["up.{}.block.{}.conv1".format(i, j), "up.{}.block.{}.conv2".format(i, j)]
        if hasattr(stage, "upsample"):
            e, c, f = _check_upsmpl(stage.upsample, c, "decoder.up[{}].upsample".format(i), sliced and t == 1)
            if e:
                return e, None
            if t == 1:
                t1.append("up.{}.upsample.conv".format(i))
            t *= f
        if hasattr(stage, "attn"):
            return "decoder.up[{}] has attention".format(i), None
    if not hasattr(dec.up[0], "upsample"):
        return "decoder.up[0] has no upsample", None
    e = _group_norm(dec.norm_out, c, "decoder.norm_out") or _conv3d(dec.conv_out, 3, "decoder.conv_out", sliced and t == 1)
    if e:
        return e, None
    if conv_io(dec.conv_out)[0] != c:
        return "decoder.conv_out takes {} channels, gets {}".format(conv_io(dec.conv_out)[0], c), None
    if t == 1:
        t1.append("conv_out")
    h = eng.hooked_module(fsm)
    if h:
        return "module {} has a forward hook or an instance-level forward".format(h), None
    return None, {"planes": t, "t1": t1}


class ConvIn:
    """The decoder's first line, on its own modules, same operations and order:
    conv_in(z) + z.repeat_interleave(C0 // z_channels, 1). A prefix module."""

    def __init__(self, dec):
        self.dec = dec
        self.zc, self.c = conv_io(dec.conv_in)

    def __call__(self, z):
        d = self.dec
        return d.conv_in(z) + z.repeat_interleave(d.block_out_channels[0] // d.z_channels, 1)


class LastFrame:
    """The end of Decoder.forward for a one-frame latent (refiner_vae False): out[:, :, -1:] (a view)."""

    def __call__(self, x):
        return x[:, :, -1:]


def _resnet_macs(rb):
    ci, co = rb.in_channels, rb.out_channels
    return 27 * (ci * co + co * co) + (ci * co if ci != co else 0)


def build_units(fsm, planes):
    """(prefix modules, stripe units); planes: the decoder's planes before the last-frame step (refiner_structure)."""
    dec = fsm.decoder
    prefix = [("decoder.conv_in+repeat", ConvIn(dec)), ("decoder.mid.block_1", dec.mid.block_1),
              ("decoder.mid.attn_1", dec.mid.attn_1), ("decoder.mid.block_2", dec.mid.block_2)]
    prefix += [("decoder.up.0.block.{}".format(j), rb) for j, rb in enumerate(dec.up[0].block)]
    units = []

    def upsample(i):
        u = dec.up[i].upsample
        ci, cf = conv_io(u.conv)
        f = 2 if u.tus else 1
        co = cf // (4 * f)
        # contiguous: the row slice is copied before the call (the conv would copy it under the layer-2 wrapper anyway);
        # extra: the conv's output channels (4 x C_out, 8 x with time upsampling)
        units.append(Unit(UPS, u, 2, 2, ci, co, "decoder.up.{}.upsample".format(i), 27 * ci * co, contiguous=True, extra=cf, tscale=f))
    upsample(0)
    for i in range(1, len(dec.up)):
        for j, rb in enumerate(dec.up[i].block):
            name = "decoder.up.{}.block.{}".format(i, j)
            ci, co = rb.in_channels, rb.out_channels
            part = Unit(PART, _partial(rb), 1, 1, ci, co, name + "[:conv1]", 27 * ci * co)
            units.append(Unit(RES, rb, 2, 1, ci, co, name, _resnet_macs(rb), ci != co,
                              norms=(NormRef(rb.norm1, name + ".norm1"), NormRef(rb.norm2, name + ".norm2", part)), contiguous=ci != co))
        if hasattr(dec.up[i], "upsample"):
            upsample(i)
    c = dec.norm_out.num_channels
    units.append(Unit(POINT, dec.norm_out, 0, 1, c, c, "decoder.norm_out", norms=(NormRef(dec.norm_out, "decoder.norm_out"),)))
    units.append(Unit(POINT, F.silu, 0, 1, c, c, "decoder.silu"))
    cin, cout = conv_io(dec.conv_out)
    units.append(Unit(CONV, dec.conv_out, 1, 1, cin, cout, "decoder.conv_out", 27 * cin * cout))
    units.append(Unit(LAST, LastFrame(), 0, 1, cout, cout, "decoder.last_frame", tscale=fractions.Fraction(1, planes)))
    return prefix, units


def scheme_positions(units, name):
    """Save positions of a GroupNorm scheme (vae_ldm.scheme_positions, upsampling units by scale)."""
    ups = [i for i, u in enumerate(units) if u.scale > 1]
    level_ends = ups[1:]
    full = [i + 1 for i, u in enumerate(units) if u.kind == RES and i > ups[-1]]
    return {"A": [], "D": level_ends[:1], "B": level_ends, "C": level_ends + full}[name]


# ---------------------------------------------------------------------------
# memory model (bytes), as vae_ldm with 3x3x3 convs; w is a row's elements (width x planes, Unit.tscale):
#   Conv3d        the layer-2 conv wrapper runs it on SlowDilated3d (5D, no cuDNN / MIOpen): vol2col columns
#                 k^3 x C_in per output element, also for a 1x1x1 conv (nin_shortcut, the attention's q / k / v /
#                 proj_out: no pointwise shortcut in 3D), plus the copy of a non-contiguous input (conv3_extra)
#   ResnetBlock   vae_ldm.res_peak with conv3_extra; the nin_shortcut's phase: h + its output + its columns
#   UpSmpl        (row slice made contiguous first: S + A, then A is the input's storage)
#                 conv: A + its output Y (4 or 8 x C_out channels at the input rows) + extra
#                 | the rearranged copy: A + 2Y | repeat_interleave R (= Y) and its rearranged copy, next to h: A + 3Y
#                 | h + x: A + 3Y (as PixelUnshuffle2D, vae_hunyuan.pu_peak)
#   conv_out      output + extra (its input is a row slice: copied)               S + B + extra
#   last frame    a view                                                          S
#   ConvIn        conv_in output + the repeated latent + the sum: 3 planes of C0 (+ the conv's extra)
#   AttnBlock     (prefix) vae_wan.attention_peak + one plane for proj_out's columns (q / k / v stay alive
#                 through it: q, k, v, the attention output, proj_out's output and columns)
# ---------------------------------------------------------------------------

def conv3_extra(cin, cout, k, r_in, r_out, w_in, w_out, e, ws, contiguous=True):
    """vae_engine.conv_extra for a stride-1 k x k x k Conv3d (w: a row's elements)."""
    kv = k * k * k
    per_row = kv * cin * w_out * e
    full = per_row * r_out
    wcopy = cout * cin * kv * e
    if full <= ws:
        return full + (0 if contiguous else r_in * w_in * cin * e) + wcopy
    rb = max(1, ws // per_row)
    return rb * per_row + (rb + k - 1) * w_in * cin * e + rb * w_out * cout * e + wcopy


def res_peak(cin, cout, shortcut, r, w, e, ws, s, frozen=True, part=False, contiguous=False):
    """vae_ldm.res_peak with 3x3x3 convs (contiguous: the caller copies the row slice first)."""
    a, b = r * w * cin * e, r * w * cout * e
    g1 = norm_temp(cin, r, w, ws) if frozen else 0
    g2 = norm_temp(cout, r, w, ws) if frozen else 0
    phases = [a + g1, a + b + conv3_extra(cin, cout, 3, r, r, w, w, e, ws)]
    if not part:
        phases += [2 * b + g2, 2 * b + conv3_extra(cout, cout, 3, r, r, w, w, e, ws), 2 * b]
        if shortcut:
            phases += [2 * b + conv3_extra(cin, cout, 1, r, r, w, w, e, ws, contiguous=contiguous), 3 * b]
    if contiguous:
        return max(s + a, a + max(phases))
    return s + max(phases)


def ups_peak(cin, cf, r, w, e, ws, s):
    a = r * w * cin * e
    y = r * w * cf * e
    return max(s + a, a + max(y + conv3_extra(cin, cf, 3, r, r, w, w, e, ws), 3 * y))


def conv3_calls(cin, k, r, w, e, ws):
    """GEMM calls of a k x k x k conv on r output rows of w elements under the layer-2 conv wrapper."""
    per_row = k * k * k * cin * w * e
    if per_row * r <= ws:
        return 1
    return -(-r // max(1, ws // per_row))


def unit_peak(u, r, w, e, ws, s):
    """Peak bytes while unit u runs on r input rows of w elements (width x planes), input storage s."""
    if u.kind in (RES, PART):
        return res_peak(u.cin, u.cout, u.shortcut, r, w, e, ws, s, part=u.kind == PART, contiguous=u.contiguous)
    if u.kind == UPS:
        return ups_peak(u.cin, u.extra, r, w, e, ws, s)
    if u.kind == CONV:
        return s + r * w * u.cout * e + conv3_extra(u.cin, u.cout, 3, r, r, w, w, e, ws, contiguous=False)
    if u.kind == LAST:
        return s
    a = r * w * u.cin * e
    return s + a + (norm_temp(u.cin, r, w, ws) if u.norms else 0)


def unit_calls(u, r, w, e, ws):
    if u.kind in (RES, PART):
        n = conv3_calls(u.cin, 3, r, w, e, ws)
        if u.kind == RES:
            n += conv3_calls(u.cout, 3, r, w, e, ws) + (conv3_calls(u.cin, 1, r, w, e, ws) if u.shortcut else 0)
        return n
    if u.kind in (UPS, CONV):
        return conv3_calls(u.cin, 3, r, w, e, ws)
    return 0


def unit_largest(u, r, w, e, ws):
    """Largest single allocation while unit u runs on r input rows of w elements."""
    a, b = r * w * u.cin * e, r * w * u.cout * e
    if u.kind in (RES, PART):
        return max(a, b, min(ws, 27 * max(u.cin, u.cout) * r * w * e), norm_temp(max(u.cin, u.cout), r, w, ws))
    if u.kind == UPS:
        return max(a, r * w * u.extra * e, min(ws, 27 * u.cin * r * w * e))
    if u.kind == CONV:
        return max(a, b, min(ws, 27 * u.cin * r * w * e))
    if u.kind == LAST:
        return 0
    return max(a, norm_temp(u.cin, r, w, ws) if u.norms else 0)


def prefix_peak(m, h, w, e, ws, s):
    """Peak bytes while prefix module m runs on the whole h x w image (one plane, native GroupNorm), input storage s."""
    if isinstance(m, ConvIn):
        return s + 3 * h * w * m.c * e + conv3_extra(m.zc, m.c, 3, h, h, w, w, e, ws)
    if isinstance(m, ldm.AttnBlock):
        return attention_peak(m.in_channels, h, w, e, ws, s, attn_split(m)) + h * w * m.in_channels * e
    return res_peak(m.in_channels, m.out_channels, m.in_channels != m.out_channels, h, w, e, ws, s, frozen=False)


def prefix_largest(m, h, w, e, ws):
    if isinstance(m, ConvIn):
        return max(h * w * m.c * e, min(ws, 27 * m.zc * h * w * e))
    if isinstance(m, ldm.AttnBlock):
        return 3 * h * w * m.in_channels * e
    c = max(m.in_channels, m.out_channels)
    return max(h * w * e * c, min(ws, 27 * c * h * w * e))


def prefix_out_channels(m):
    if isinstance(m, ConvIn):
        return m.c
    if isinstance(m, ldm.AttnBlock):
        return m.in_channels
    return m.out_channels


def prefix_macs(m):
    if isinstance(m, ConvIn):
        return 27 * m.zc * m.c
    if isinstance(m, ldm.AttnBlock):
        return 4 * m.in_channels * m.in_channels
    return _resnet_macs(m)


# ---------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------

class RefinerStripe(eng.StripeAdapter):
    """Layer-1 adapter bound to one HunyuanImage 2.1 Refiner first-stage model."""

    hdim = 3              # rows in [B, C, T, H, W]
    tdim = 2              # the stripes carry every frame (planes)
    ckpt_planes = 1
    schemes = SCHEMES
    # the self-test's decoder input: 12 x 12 (output 192 x 192, 5 stripes of SELFTEST_ROWS with inner boundaries at
    # every level); the engine's 24 would be a 384 x 384 fp32 decode of 4 frames (~1.8 GiB for the reference alone)
    selftest_size = 12
    # the layer-2 bound (vae.estimate) and the self-test's reference decode (DESIGN §9.30): UpSmpl makes five
    # output-sized tensors, every level's activation is 2-8x the one below, the caching allocator keeps each level's
    # segments into the next: tests/alloc_sim.py, layer 2 reserved up to 5.67 x the largest activation + 2 x the
    # workspace (1024-3072 wide, workspace 128 MiB - 1 GiB, bf16 / fp32); 4 copies are not an upper bound
    activation_copies = 6

    def __init__(self, fsm, gn_scheme=None, module=None, info=None):
        self.fsm = fsm
        self.module = fsm if module is None else module
        if info is None:
            reason, info = refiner_structure(fsm)
            if reason:
                raise eng.StripeError(reason)
        self.info = info
        self.prefix, self.units = build_units(fsm, info["planes"])
        self.scale = 1
        for u in self.units:
            self.scale *= u.scale
        self.ckpt_channels = prefix_out_channels(self.prefix[-1][1])
        self.out_channels = self.units[-1].cout
        self.gn_scheme = gn_scheme or vae_ldm.scheme()
        self.saves = scheme_positions(self.units, self.gn_scheme)
        self.name = "HunyuanImage 2.1 Refiner stripes, GroupNorm scheme {}".format(self.gn_scheme)
        dec = fsm.decoder
        self.key = ("HunyuanImage2.1Refiner", conv_io(dec.conv_in), tuple(dec.block_out_channels), int(dec.num_res_blocks),
                    tuple((u.kind, u.cin, u.cout, u.tscale) for u in self.units), self.gn_scheme)

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
        return [self if n == self.gn_scheme else RefinerStripe(self.fsm, gn_scheme=n, info=self.info) for n in names]

    def predict_seconds(self, plan):
        """vae_ldm's time model (fitted to SDXL's 2D convs on CT 700) on this decoder's MACs (every frame counted), the
        levels mapped by their resolution relative to the output (full resolution = SDXL's level 8, H/2 = 4, below = 2).
        Not calibrated for the 5D convs (SlowDilated3d): for ranking the schemes of one decode; absolute seconds are
        rough (DESIGN §9.31)."""
        top = TIME_COEF[max(TIME_COEF)]
        f = self.scale // 8
        macs = sum(TIME_COEF.get(max(2, lv // f), top) * m for lv, m in plan.work_levels.items() if lv > 1)
        tokens = plan.heights[0] * plan.widths[0]
        return macs / 1e12 + TIME_PER_CALL * plan.conv_calls + TIME_ATTN * 2 * self.ckpt_channels * tokens * tokens / 1e12

    def output_shape(self, samples):
        return (samples.shape[0], self.out_channels, 1, int(samples.shape[-2]) * self.scale, int(samples.shape[-1]) * self.scale)

    # self-test

    def selftest_params(self):
        """What fp32_copy copies: the decoder, the 3x3x3 convs that only see one frame as their middle time slice."""
        dec = self.fsm.decoder
        mods = dict(dec.named_modules())
        sliced = sum(mods[n].weight.numel() * 2 // 3 for n in self.info["t1"] if mods[n].weight.shape[2] == 3)
        return sum(p.numel() for p in dec.parameters()) - sliced

    def selftest_latent(self, n):
        return (1, conv_io(self.fsm.decoder.conv_in)[0], 1, n, n)

    def fp32_copy(self, device):
        """The decoder rebuilt from the model's configuration in fp32 with the model's weights (the encoder is not
        copied); freed after the self-test. The 3x3x3 convs that only see one frame (refiner_structure's "t1": up to
        the first time upsampling) keep only the middle slice of their time kernel, without time padding: one frame
        padded with a zero frame on each side meets only that slice, so it is the same conv (the full kernels in fp32
        are 3.25 GiB for the real decoder, with the slices 1.98 GiB; tests/test_vae_refiner.py checks that the copy's
        own decode equals the model's)."""
        dec = self.fsm.decoder
        boc = list(dec.block_out_channels)[::-1]          # the constructor's order (the decoder keeps it reversed)
        depth = sum(1 for s in dec.up if hasattr(s, "upsample"))
        tdepth = sum(1 for s in dec.up if hasattr(s, "upsample") and s.upsample.tus)
        holder = torch.nn.Module()
        with eng.skeleton():
            holder.decoder = hyr.Decoder(z_channels=dec.z_channels, out_channels=conv_io(dec.conv_out)[1], block_out_channels=boc,
                                         num_res_blocks=int(dec.num_res_blocks), ffactor_spatial=2 ** depth, ffactor_temporal=2 ** tdepth,
                                         upsample_match_channel=True, refiner_vae=False)
            mods = dict(holder.decoder.named_modules())
            for n in self.info["t1"]:
                m = mods[n]
                if m.weight.shape[2] == 3:
                    m.weight = torch.nn.Parameter(torch.empty(m.weight.shape[:2] + (1,) + m.weight.shape[3:]), requires_grad=False)
                    m.kernel_size = (1,) + tuple(m.kernel_size[1:])
                    m.padding = (0,) + tuple(m.padding[1:])
                    m._reversed_padding_repeated_twice = [p for p in m._reversed_padding_repeated_twice[:4]] + [0, 0]
        t1 = set(self.info["t1"])
        sd = {}
        for k, v in dec.state_dict().items():
            owner = k.rpartition(".")[0]
            if owner in t1 and k.endswith("weight") and v.ndim == 5 and v.shape[2] == 3:
                v = v[:, :, 1:2]
            sd["decoder." + k] = v
        holder.to_empty(device=device)
        holder.to(dtype=torch.float32)
        holder.load_state_dict(sd, strict=True)
        holder.eval()
        shim = _Shim(holder)
        reason, info = refiner_structure(shim, sliced=True)
        if reason:
            raise eng.StripeError("fp32 copy does not match: " + reason)
        return _CopyStripe(shim, self.gn_scheme, holder, info)


class _CopyStripe(RefinerStripe):
    """The adapter on the fp32 copy; its reference is AutoencodingEngine.decode."""

    def __init__(self, shim, gn_scheme, holder, info):
        super().__init__(shim, gn_scheme, module=holder, info=info)

    def reference_decode(self, z):
        return ae.AutoencodingEngine.decode(self.module, z)


def decoder_terms(vae, samples):
    """The layer-2 estimate's terms (vae._decoder_terms) for any vae_refiner.Decoder with refiner_vae=False, by its
    class: a forward hook or vae_options keep the stripe decoder off it (match), not the copies of its own decode
    (DESIGN §9.30). None for others (HunyuanVideo 1.5's vae_refiner.Decoder with refiner_vae=True included)."""
    dec = getattr(getattr(vae, "first_stage_model", None), "decoder", None)
    if type(dec) is not hyr.Decoder or dec.refiner_vae is not False:
        return None
    return {"copies": RefinerStripe.activation_copies}


def match(vae, samples, vae_options):
    """(bound adapter, None) or (None, reason)."""
    fsm = vae.first_stage_model
    reason, info = refiner_structure(fsm)
    if reason:
        return None, reason
    if vae_options:
        return None, "vae_options {} given".format(sorted(vae_options))
    if samples.ndim != 5:
        return None, "latent has {} dims".format(samples.ndim)
    if samples.shape[2] != 1:
        return None, "latent has {} frames".format(samples.shape[2])
    if samples.shape[1] != fsm.decoder.z_channels:
        return None, "latent has {} channels, the decoder takes {}".format(samples.shape[1], fsm.decoder.z_channels)
    return RefinerStripe(fsm, info=info), None


class _Shim:
    """Duck-types the attributes refiner_structure / RefinerStripe read from the first-stage model."""

    def __init__(self, holder):
        self.decoder = holder.decoder
        self._holder = holder

    def named_modules(self):
        return self._holder.named_modules()


def is_refiner(fsm):
    """Whether fsm has the Refiner decoder this adapter handles (the Info node's scheme hints)."""
    return refiner_structure(fsm)[0] is None if fsm is not None else False


has_schemes = is_refiner   # the GroupNorm schemes apply to it (vae.gn_schemes_apply)
