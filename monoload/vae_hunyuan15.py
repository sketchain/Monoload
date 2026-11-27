"""Layer-1 adapter: HunyuanVideo 1.5's VAE decoder (comfy.ldm.hunyuan_video.vae_refiner.Decoder, refiner_vae=True; z 32,
16x spatial, 4x temporal), multi-frame and single frame; the official HunyuanImage 2.1 Refiner file builds the same
decoder and decodes one frame (docs/DESIGN.md §9.31, §9.38).

comfy.sd.VAE builds it for a state dict whose decoder.conv_in.conv.weight takes 32 channels: an AutoencodingEngine
(no post_quant_conv; decode = decoder(z)) with block_out_channels [128, 256, 512, 1024, 1024] (reversed in the
decoder), num_res_blocks 2, ffactor_spatial 16, ffactor_temporal 4. Its forward (v0.46.3):

    x = conv_carry_causal_3d([z], conv_in) + z.repeat_interleave(C0 // z_channels, 1)
    x = mid.block_2(mid.attn_1(mid.block_1(x)))                 # all latent frames, global attention over T*h*w
    for each 2-latent-frame chunk of x (torch.split), with a carry list from the previous chunk:
        up[0].block[*] -> up[0].upsample -> up[1].block[*] -> up[1].upsample -> ... -> up[4].block[*]
        -> norm_out -> F.silu -> conv_carry_causal_3d(conv_out)
    output = torch.cat(chunks, dim=2)

  * every 3x3x3 conv is a CarriedConv3d called through conv_carry_causal_3d: replicate pad (1, 1, 1, 1, 2, 0) without
    a carry, else the spatial pad and the previous chunk's last 2 padded input frames in front; its own last 2 padded
    input frames go (cloned) to the next chunk. The carries are popped / appended in call order, so each unit can keep
    its own list (call_unit). Over the whole sequence this is the same conv as one call with the front replicate pad:
    a carry is exactly the 2 frames before the chunk.
  * norms are vae_refiner.RMS_norm (per position, no statistics): no statistics passes, no schemes. Layer 2's scheme G
    (H4) replaces the ones with more than 510 channels by the grouped norm (its scratch: vae_ops.grouped_norm_bytes, in
    the cost model of every norm below).
  * UpSmpl (stages 0..3; time x2 in stages 0 and 1, `tus`): a 3x3x3 conv to fct x C_out channels, depth-to-space 2x2
    (and 2 frames), plus the input with its channels repeated and rearranged the same way. Without a carry (the first
    chunk) the first frame is spatial only: 2 latent frames -> 3 -> 5 output frames in the first chunk, 2 -> 4 -> 8 in
    the others (4T - 3 in all). Output row 2h + i depends on conv output row h: halo 2 at its output level.

Layer 1 (the engine's time axis, H2 of docs/PLAN-video.md):
  prefix      conv_in + repeat, mid (block_1, attn_1, block_2) and up[0]'s blocks on all latent frames at once (the
              blocks without a carry: the same conv as the chunked one, above); the checkpoint is up[0]'s output,
              1024 channels at the latent resolution, every frame.
  units       up[0].upsample, then per stage its blocks and upsample, norm_out, silu, conv_out; each carried unit is
              called with its own carry lists in the stripe's state (call_unit). Spatially every unit is local: the
              replicate pad at an inner stripe edge (in the input and so in the carries) only reaches halo rows.
  time chunks the native ones: 2 latent frames each (the last one 1 if T is odd); chunk 0 without carries.
  single frame (T = 1, the Refiner): no chunks, no carries (an image decode, planes 1 everywhere).

Tensors are 5D [B, C, T, H, W].
"""

import torch
import torch.nn.functional as F

import comfy.ldm.hunyuan_video.vae_refiner as hyr
from comfy.ldm.models import autoencoder as ae
from comfy.ldm.modules.diffusionmodules import model as ldm

from . import vae_engine as eng
from .vae_engine import CONV, POINT, RES, TimeChunk, Unit, conv_io
from .vae_ops import grouped_norm_bytes
from .vae_wan import attention_peak, attn_split

UPS = "upsmpl"       # vae_refiner.UpSmpl (scale 2, halo 2)
CHUNK = 2            # latent frames per time chunk (vae_refiner.Decoder.forward's torch.split)
SELFTEST_SIZE = 12   # latent rows / cols of the self-test: 192 output rows, 5 stripes of SELFTEST_ROWS
SELFTEST_FRAMES = 5  # latent frames of the multi-frame self-test: chunks of 2, 2, 1 (SELFTEST_MIN_CHUNKS = 3)

# A HunyuanVideo 1.5 video (T > 1) is layer 1 only as the fallback (prefer_layer2, DESIGN §9.41): on the GPU
# (832x480x81) layer 1 reserves 2.54 GiB against layer 2's 10.43 but takes 1.56x as long, so layer 2 when its estimate
# fits the device (vae.layer2_room), layer 1 past that. Its plan's estimate is still under the reserved growth there
# (2.33): the carries' clones and the top level's padded input fragment the arena, two requests land outside it where
# `largest` covers one (an engine fix, the vae_engine owner's). False: a video never takes layer 1 (layer 2 as before).
MULTIFRAME_ENABLED = True


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def _carried(m, k, name):
    """None if m is a CarriedConv3d whose conv is a stride-1, unpadded, undilated, ungrouped k x k x k Conv3d."""
    if type(m) is not ldm.CarriedConv3d:
        return "{} is {}, not a CarriedConv3d".format(name, type(m).__name__)
    c = m.conv
    if not isinstance(c, torch.nn.Conv3d):
        return "{}.conv is {}".format(name, type(c).__name__)
    if (tuple(c.kernel_size) != (k, k, k) or tuple(c.stride) != (1, 1, 1) or tuple(c.padding) != (0, 0, 0)
            or tuple(c.dilation) != (1, 1, 1) or c.groups != 1 or c.padding_mode != "zeros"):
        return "{}.conv kernel {} stride {} padding {} dilation {} groups {}".format(name, c.kernel_size, c.stride, c.padding, c.dilation, c.groups)
    return None


def _rms(m, c, name):
    if type(m) is not hyr.RMS_norm:
        return "{} is {}, not a vae_refiner.RMS_norm".format(name, type(m).__name__)
    if tuple(m.gamma.shape) != (c, 1, 1, 1):
        return "{}.gamma {} on {} channels".format(name, tuple(m.gamma.shape), c)
    return None


def _check_resnet(rb, ci, co, name):
    if type(rb) is not ldm.ResnetBlock:
        return "{} is {}, not an ldm ResnetBlock".format(name, type(rb).__name__)
    if rb.in_channels != ci or rb.out_channels != co or rb.use_conv_shortcut or rb.fused_norm:
        return "{}: {} -> {} channels (conv shortcut {}, fused {}) on {} -> {}".format(
            name, rb.in_channels, rb.out_channels, rb.use_conv_shortcut, rb.fused_norm, ci, co)
    e = (_rms(rb.norm1, ci, name + ".norm1") or _rms(rb.norm2, co, name + ".norm2")
         or _carried(rb.conv1, 3, name + ".conv1") or _carried(rb.conv2, 3, name + ".conv2"))
    if e:
        return e
    if conv_io(rb.conv1.conv) != (ci, co) or conv_io(rb.conv2.conv) != (co, co):
        return "{}: conv channels {} / {}".format(name, conv_io(rb.conv1.conv), conv_io(rb.conv2.conv))
    if not isinstance(rb.dropout, torch.nn.Dropout) or rb.dropout.p != 0.0:
        return "{}.dropout {}".format(name, rb.dropout)
    if ci != co:
        e = _carried(rb.nin_shortcut, 1, name + ".nin_shortcut")
        if e:
            return e
        if conv_io(rb.nin_shortcut.conv) != (ci, co):
            return "{}.nin_shortcut channels {}".format(name, conv_io(rb.nin_shortcut.conv))
    return None


def _check_up(up, ci, name):
    """(error, output channels, temporal) of an UpSmpl on ci channels."""
    if type(up) is not hyr.UpSmpl:
        return "{} is {}, not a vae_refiner.UpSmpl".format(name, type(up).__name__), None, None
    if up.refiner_vae is not True:
        return "{}: refiner_vae {}".format(name, up.refiner_vae), None, None
    e = _carried(up.conv, 3, name + ".conv")
    if e:
        return e, None, None
    fct = 8 if up.tus else 4
    a, b = conv_io(up.conv.conv)
    if a != ci or b % fct:
        return "{}: conv {} -> {} on {} channels".format(name, a, b, ci), None, None
    oc = b // fct
    if up.rp * ci != fct * oc or (up.tus and up.rp % 2):
        return "{}: rp {} for {} -> {} channels".format(name, up.rp, ci, oc), None, None
    return None, oc, bool(up.tus)


def hv15_structure(fsm):
    """(None, info) if fsm is a HunyuanVideo 1.5 VAE (vae_refiner.Decoder, refiner_vae=True) this adapter handles,
    else (reason, None)."""
    if not (type(fsm) is ae.AutoencodingEngine or isinstance(fsm, _Shim)):
        return "first-stage model is {}, not comfy.ldm.models.autoencoder.AutoencodingEngine".format(type(fsm).__name__), None
    dec = getattr(fsm, "decoder", None)
    if type(dec) is not hyr.Decoder:
        return "decoder is {}.{}, not comfy.ldm.hunyuan_video.vae_refiner.Decoder".format(type(dec).__module__, type(dec).__name__), None
    if dec.refiner_vae is not True:
        return "vae_refiner.Decoder with refiner_vae={} (decodes whole)".format(dec.refiner_vae), None
    e = _carried(dec.conv_in, 3, "decoder.conv_in")
    if e:
        return e, None
    zc, c = conv_io(dec.conv_in.conv)
    boc = list(dec.block_out_channels)
    if zc != dec.z_channels or not boc or boc[0] != c or c % zc:
        return "decoder.conv_in {} -> {} with z_channels {}, block_out_channels {}".format(zc, c, dec.z_channels, boc), None
    for nm in ("block_1", "block_2"):
        e = _check_resnet(getattr(dec.mid, nm), c, c, "decoder.mid." + nm)
        if e:
            return e, None
    at = dec.mid.attn_1
    if type(at) is not ldm.AttnBlock:
        return "decoder.mid.attn_1 is {}".format(type(at).__name__), None
    e = _rms(at.norm, c, "decoder.mid.attn_1.norm")
    if e:
        return e, None
    for nm in ("q", "k", "v", "proj_out"):
        m = getattr(at, nm)
        if not isinstance(m, torch.nn.Conv3d) or tuple(m.kernel_size) != (1, 1, 1) or conv_io(m) != (c, c) or tuple(m.padding) != (0, 0, 0):
            return "decoder.mid.attn_1.{} is not a 1x1x1 Conv3d on {} channels".format(nm, c), None
    if len(dec.up) < 1:
        return "decoder has no up stage", None
    tus = []
    for i, stage in enumerate(dec.up):
        blocks = list(stage.block)
        if len(blocks) != dec.num_res_blocks + 1:
            return "decoder.up.{} has {} blocks".format(i, len(blocks)), None
        for j, rb in enumerate(blocks):
            co = rb.out_channels
            e = _check_resnet(rb, c, co, "decoder.up.{}.block.{}".format(i, j))
            if e:
                return e, None
            c = co
        if hasattr(stage, "upsample"):
            e, c, t = _check_up(stage.upsample, c, "decoder.up.{}.upsample".format(i))
            if e:
                return e, None
            tus.append(t)
        elif i != len(dec.up) - 1:
            return "decoder.up.{} has no upsample but is not the last stage".format(i), None
    if not tus or any(not t for t in tus[:sum(tus)]) or sum(tus) > 2:
        return "upsampling stages: temporal {}".format(tus), None
    e = _rms(dec.norm_out, c, "decoder.norm_out") or _carried(dec.conv_out, 3, "decoder.conv_out")
    if e:
        return e, None
    if conv_io(dec.conv_out.conv)[0] != c:
        return "decoder.conv_out takes {} channels, gets {}".format(conv_io(dec.conv_out.conv)[0], c), None
    h = eng.hooked_module(fsm)
    if h:
        return "module {} has a forward hook or an instance-level forward".format(h), None
    return None, {"tus": tus}


# ---------------------------------------------------------------------------
# units (callables with the carry lists as optional arguments)
# ---------------------------------------------------------------------------

class _ConvIn:
    """conv_in + the channel-repeated latent (vae_refiner.Decoder.forward's first line)."""
    __slots__ = ("dec",)

    def __init__(self, dec):
        self.dec = dec

    def __call__(self, z):
        d = self.dec
        return ldm.conv_carry_causal_3d([z], d.conv_in) + z.repeat_interleave(d.block_out_channels[0] // d.z_channels, 1)


class _Res:
    __slots__ = ("blk",)

    def __init__(self, blk):
        self.blk = blk

    def __call__(self, x, carry_in=None, carry_out=None):
        return self.blk(x, None, carry_in, carry_out)


class _Up:
    __slots__ = ("up",)

    def __init__(self, up):
        self.up = up

    def __call__(self, x, carry_in=None, carry_out=None):
        return self.up(x, carry_in, carry_out)


class _Silu:
    __slots__ = ()

    def __call__(self, x):
        return F.silu(x)


class _ConvOut:
    __slots__ = ("conv",)

    def __init__(self, conv):
        self.conv = conv

    def __call__(self, x, carry_in=None, carry_out=None):
        return ldm.conv_carry_causal_3d([x], self.conv, carry_in, carry_out)


def _res_macs(rb):
    ci, co = rb.in_channels, rb.out_channels
    return 27 * (ci * co + co * co) + (ci * co if ci != co else 0)


def build_units(fsm, single):
    """(prefix modules, stripe units). single: one frame (no time upsampling counted in the planes)."""
    dec = fsm.decoder
    prefix = [("decoder.conv_in", _ConvIn(dec)), ("decoder.mid.block_1", dec.mid.block_1), ("decoder.mid.attn_1", dec.mid.attn_1),
              ("decoder.mid.block_2", dec.mid.block_2)]
    prefix += [("decoder.up.0.block.{}".format(j), rb) for j, rb in enumerate(dec.up[0].block)]
    units = []

    def add(kind, module, halo, scale, ci, co, name, macs=0, shortcut=False, tscale=1, carried=()):
        units.append(Unit(kind, module, halo, scale, ci, co, name, macs, shortcut, tscale=tscale,
                          extra={"i": len(units), "carried": tuple(carried)}))

    for i, stage in enumerate(dec.up):
        if i > 0:
            for j, rb in enumerate(stage.block):
                add(RES, _Res(rb), 2, 1, rb.in_channels, rb.out_channels, "decoder.up.{}.block.{}".format(i, j), _res_macs(rb),
                    rb.in_channels != rb.out_channels, carried=(rb.in_channels, rb.out_channels))
        if hasattr(stage, "upsample"):
            up = stage.upsample
            ci, b = conv_io(up.conv.conv)
            fct = 8 if up.tus else 4
            add(UPS, _Up(up), 2, 2, ci, b // fct, "decoder.up.{}.upsample".format(i), 27 * ci * (b // fct), tscale=2 if (up.tus and not single) else 1,
                carried=(ci,))
            units[-1].extra["fct"] = fct
            units[-1].extra["tus"] = bool(up.tus)
    c = int(dec.norm_out.gamma.shape[0])
    ci, co = conv_io(dec.conv_out.conv)
    add(POINT, dec.norm_out, 0, 1, c, c, "decoder.norm_out")
    add(POINT, _Silu(), 0, 1, c, c, "silu")
    add(CONV, _ConvOut(dec.conv_out), 1, 1, ci, co, "decoder.conv_out", 27 * ci * co, carried=(ci,))
    return prefix, units


def signature(fsm, units, single):
    dec = fsm.decoder
    return ("HunyuanVideo1.5", int(dec.z_channels), tuple(dec.block_out_channels), int(dec.num_res_blocks), bool(single),
            tuple((u.kind, u.cin, u.cout, u.tscale) for u in units))


# ---------------------------------------------------------------------------
# memory model (bytes)
# ---------------------------------------------------------------------------
# Live tensors, counted from the v0.46.3 forward code. S: the storage of the unit's input (held by the caller), e the
# element size; r rows, W columns, p planes (frames) of the unit's input; A / B one input / output-channel tensor of it,
# Ap / Bp the same padded by conv_carry_causal_3d ((r + 2) x (W + 2) x (p + 2): the spatial replicate pad and 2 frames
# in front, the time pad or the carry); gn(c): the grouped norm's scratch (H4, more than 510 channels).
#   RMS_norm      F.normalize * scale * gamma: two temporaries at a time                        S + 2A + gn
#   conv          (conv_carry_causal_3d) the pad (and with a carry the cat: the spatially padded frames + Ap), then
#                 the conv: Ap + out + the vol2col columns / layer-2 block (conv3_extra)
#   ResnetBlock   norm1 2A | its output + pad + cat A + 2Ap | conv1 Ap + B + extra | norm2 3B | B + 2Bp | conv2 Bp + B
#                 + extra | shortcut (1x1x1 Conv3d on the row slice: copy + columns) 2B + 2A | add 3B      S + max(...)
#   UpSmpl        pad / cat 2Ap | conv Ap + H + extra (H: the conv output, fct x C_out channels, as many elements as the
#                 output) | the rearranged conv output, the repeated input and its rearranged copy, the first frame's
#                 copies (first chunk), the cats and the sum: at most 4H                       S + max(...)
#   conv_out      pad / cat 2Ap | Ap + B + extra                                                S + max(...)
#   silu          F.silu (not in place)                                                         S + A
#   carries       2 padded frames per carried conv, (r + 2) x (W + 2) x C_in, kept in the stripe's state from one chunk
#                 to the next (unit_cache); the previous one is freed in the cat, before the new clone
#   prefix        as above on all frames (no carry: the front time pad), the conv_in + repeat 3P + extra, attention as
#                 vae_wan.attention_peak (tokens T x h x w) plus its 1x1x1 convs' columns


def conv3_extra(cin, cout, r_out, w_in, w_out, e, ws):
    """Bytes a 3x3x3 unpadded Conv3d needs besides its (contiguous, padded) input and output, as the layer-2 conv wrapper
    runs it: unblocked, the vol2col columns and the weight copy; in row blocks, one block of columns, the block's input
    copy (a row slice of 5D) and its output. w_in / w_out: elements of an input / output row (width x planes)."""
    per_row = 27 * cin * w_out * e
    wcopy = 27 * cin * cout * e
    if per_row * r_out <= ws:
        return per_row * r_out + wcopy
    rb = max(1, ws // per_row)
    return rb * per_row + (rb + 2) * w_in * cin * e + rb * w_out * cout * e + wcopy


def _padded(c, r, W, p, e):
    return (r + 2) * (W + 2) * (p + 2) * c * e


def res_peak(ci, co, sc, r, W, p, e, ws, s):
    n = r * W * p
    a, b = n * ci * e, n * co * e
    ap, bp = _padded(ci, r, W, p, e), _padded(co, r, W, p, e)
    wi, wo = (W + 2) * (p + 2), W * p
    phases = [2 * a + grouped_norm_bytes(ci, n, e),
              a + 2 * ap,
              ap + b + conv3_extra(ci, co, r, wi, wo, e, ws),
              3 * b + grouped_norm_bytes(co, n, e),
              b + 2 * bp,
              bp + b + conv3_extra(co, co, r, wi, wo, e, ws),
              3 * b]
    if sc:
        phases.append(2 * b + 2 * a)
    return s + max(phases)


def ups_peak(ci, co, fct, r, W, p, e, ws, s):
    ap = _padded(ci, r, W, p, e)
    h = r * W * p * co * fct * e
    return s + max(2 * ap, ap + h + conv3_extra(ci, co * fct, r, (W + 2) * (p + 2), W * p, e, ws), 4 * h)


def conv_peak(ci, co, r, W, p, e, ws, s):
    ap = _padded(ci, r, W, p, e)
    return s + max(2 * ap, ap + r * W * p * co * e + conv3_extra(ci, co, r, (W + 2) * (p + 2), W * p, e, ws))


def attn_peak(c, h, W, p, e, ws, s, split):
    """vae_wan.attention_peak over T x h x w tokens, plus the vol2col columns of a 1x1x1 Conv3d (one P)."""
    return attention_peak(c, h, W * p, e, ws, s, split) + h * W * p * c * e


def _carry_bytes(cs, r, W, e):
    return sum(2 * (r + 2) * (W + 2) * c * e for c in cs)


# ---------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------

class HV15Stripe(eng.StripeAdapter):
    """Layer-1 adapter bound to one HunyuanVideo 1.5 VAE, for a multi-frame latent or a single frame (single)."""

    hdim = 3
    tdim = 2
    selftest_size = SELFTEST_SIZE
    activation_copies = 6    # the reference (native) decode's UpSmpl: the conv output, its rearranged copy, the repeated
                             # input, its copy, the first frame's copies and the input (ups_peak)

    def __init__(self, fsm, single, module=None):
        self.fsm = fsm
        self.single = bool(single)
        self.module = fsm if module is None else module
        self.name = "HunyuanVideo 1.5 stripes" + (" (single frame)" if self.single else "")
        self.prefix, self.units = build_units(fsm, self.single)
        self.scale = 1
        for u in self.units:
            self.scale *= u.scale
        self.ckpt_channels = self.fsm.decoder.up[0].block[-1].out_channels
        self.out_channels = self.units[-1].cout
        self.key = signature(fsm, self.units, self.single)
        self._frames = 1                                   # latent frames of the latest time_chunks (the prefix's planes)
        self._planes = [1] * (len(self.units) + 1)         # planes at each level (the largest over the chunks), likewise

    @property
    def prefer_layer2(self):
        """A video: layer 2 when its estimate fits the device, layer 1 only past that (vae.py's prefer_layer2 routing;
        1.56x slower than layer 2 at 832x480x81). One frame (the Refiner): layer 1 as usual."""
        return not self.single

    # time axis

    def time_chunks(self, shape):
        t = int(shape[2])
        self._frames = t
        if self.single:
            if t != 1:
                raise eng.StripeError("[Monoload] internal error: single-frame {} on {} frames".format(self.name, t))
            self._planes = [1] * (len(self.units) + 1)
            return None
        chunks, o = [], 0
        for c0 in range(0, t, CHUNK):
            c1 = min(t, c0 + CHUNK)
            n = (4 * (c1 - c0) - 3) if c0 == 0 else 4 * (c1 - c0)
            chunks.append(TimeChunk(c0, c1, o, o + n))
            o += n
        per = [self.chunk_planes(c) for c in chunks]
        self._planes = [max(pp[i] for pp in per) for i in range(len(self.units) + 1)]
        return chunks

    def chunk_planes(self, chunk):
        p = [chunk.ct1 - chunk.ct0]
        for u in self.units:
            q = p[-1]
            if u.kind == UPS and u.extra["tus"]:
                q = 2 * q - 1 if chunk.ct0 == 0 else 2 * q
            p.append(q)
        return p

    def stripe_state(self):
        return {}

    def call_unit(self, u, x, state, k):
        if not u.extra["carried"]:
            return u.module(x)
        carry_in = None
        if k > 0:
            carry_in = state.get(u.name)
            if carry_in is None:
                raise eng.StripeError("[Monoload] internal error: {} has no carry for chunk {}".format(u.name, k))
        carry_out = []
        y = u.module(x, carry_in, carry_out)
        state[u.name] = carry_out
        return y

    def unit_cache(self, u, r, w, e):
        return _carry_bytes(u.extra["carried"], r, w, e)

    # cost model (the engine passes width x planes as the width)

    def _p(self, u):
        return max(1, self._planes[u.extra["i"]])

    def unit_peak(self, u, r, w, e, ws, s):
        p = self._p(u)
        W = w // p
        if u.kind == RES:
            return res_peak(u.cin, u.cout, u.shortcut, r, W, p, e, ws, s)
        if u.kind == UPS:
            return ups_peak(u.cin, u.cout, u.extra["fct"], r, W, p, e, ws, s)
        if u.kind == CONV:
            return conv_peak(u.cin, u.cout, r, W, p, e, ws, s)
        if isinstance(u.module, hyr.RMS_norm):
            return s + 2 * r * w * u.cin * e + grouped_norm_bytes(u.cin, r * w, e)
        return s + r * w * u.cin * e

    def unit_largest(self, u, r, w, e, ws):
        p = self._p(u)
        W = w // p
        a, b = r * w * u.cin * e, r * w * u.cout * e
        ap = _padded(u.cin, r, W, p, e)
        if u.kind == RES:
            return max(ap, _padded(u.cout, r, W, p, e), min(ws, 27 * max(u.cin, u.cout) * r * w * e))
        if u.kind == UPS:
            return max(ap, b * u.extra["fct"], min(ws, 27 * u.cin * r * w * e))
        if u.kind == CONV:
            return max(ap, b, min(ws, 27 * u.cin * r * w * e))
        return a

    def _prefix_geom(self, w, s):
        """The engine gives the prefix width x the checkpoint planes of one chunk and the input storage at that size;
        this prefix runs on every latent frame at once: (width, frames, input storage)."""
        p0 = max(1, self._planes[0])
        p = max(1, self._frames)
        return w // p0, p, s // p0 * p

    def prefix_peak(self, m, h, w, e, ws, s):
        W, p, s = self._prefix_geom(w, s)
        if isinstance(m, ldm.AttnBlock):
            return attn_peak(m.in_channels, h, W, p, e, ws, s, attn_split(m))
        if isinstance(m, ldm.ResnetBlock):
            return res_peak(m.in_channels, m.out_channels, m.in_channels != m.out_channels, h, W, p, e, ws, s)
        # _ConvIn: the padded latent, the conv output + extra, then the repeated latent and the sum
        zc, c = conv_io(m.dec.conv_in.conv)
        big = h * W * p * c * e
        return s + max(_padded(zc, h, W, p, e) + big + conv3_extra(zc, c, h, (W + 2) * (p + 2), W * p, e, ws), 3 * big)

    def prefix_largest(self, m, h, w, e, ws):
        W, p, _ = self._prefix_geom(w, 0)
        n = h * W * p
        if isinstance(m, ldm.AttnBlock):
            return n * m.in_channels * e
        if isinstance(m, ldm.ResnetBlock):
            c = max(m.in_channels, m.out_channels)
            return max(_padded(c, h, W, p, e), min(ws, 27 * c * n * e))
        zc, c = conv_io(m.dec.conv_in.conv)
        return max(n * c * e, min(ws, 27 * zc * n * e))

    @staticmethod
    def prefix_out_channels(m):
        if isinstance(m, ldm.AttnBlock):
            return m.in_channels
        if isinstance(m, ldm.ResnetBlock):
            return m.out_channels
        return conv_io(m.dec.conv_in.conv)[1]

    @staticmethod
    def prefix_macs(m):
        if isinstance(m, ldm.AttnBlock):
            return 4 * m.in_channels * m.in_channels
        if isinstance(m, ldm.ResnetBlock):
            return _res_macs(m)
        zc, c = conv_io(m.dec.conv_in.conv)
        return 27 * zc * c

    def decode_retained(self, h, w, e):
        """The reference (native) decode keeps, besides its live activations, the carries of every carried conv of the
        stripe part (2 padded frames at its resolution, full width) and the list of decoded chunks (the output again)."""
        n = sum(_carry_bytes((rb.in_channels, rb.out_channels), h, w, e) for rb in self.fsm.decoder.up[0].block)
        hh, ww = h, w
        for u in self.units:
            n += _carry_bytes(u.extra["carried"], hh, ww, e)
            hh, ww = hh * u.scale, ww * u.scale
        frames = 4 * self._frames - 3
        return n + self.out_channels * hh * ww * frames * e

    def output_shape(self, samples):
        t = int(samples.shape[2])
        return (samples.shape[0], self.out_channels, 4 * t - 3,
                int(samples.shape[-2]) * self.scale, int(samples.shape[-1]) * self.scale)

    # self-test

    def selftest_params(self):
        return sum(p.numel() for p in self.fsm.decoder.parameters())

    def selftest_latent(self, n):
        return (1, int(self.fsm.decoder.z_channels), 1 if self.single else SELFTEST_FRAMES, n, n)

    def fp32_copy(self, device):
        """The decoder rebuilt from the model's configuration in fp32 with the model's weights (the encoder is not
        copied); freed after the self-test. Full 3x3x3 kernels (a multi-frame decode reads every time slice)."""
        dec = self.fsm.decoder
        n_up = sum(1 for s in dec.up if hasattr(s, "upsample"))
        n_t = sum(1 for s in dec.up if hasattr(s, "upsample") and s.upsample.tus)
        holder = torch.nn.Module()
        with eng.skeleton():
            holder.decoder = hyr.Decoder(dec.z_channels, conv_io(dec.conv_out.conv)[1], list(dec.block_out_channels)[::-1], dec.num_res_blocks,
                                         2 ** n_up, 2 ** n_t, upsample_match_channel=True, refiner_vae=True)
        holder.to_empty(device=device)
        holder.to(dtype=torch.float32)
        holder.decoder.load_state_dict(dec.state_dict(), strict=True)
        holder.eval()
        shim = _Shim(holder)
        reason, _ = hv15_structure(shim)
        if reason:
            raise eng.StripeError("fp32 copy does not match: " + reason)
        return HV15Stripe(shim, self.single, module=holder)

    def reference_decode(self, z):
        return self.module.decoder(z)


def decoder_terms(vae, samples):
    """None: the layer-2 estimate of this decoder is the generic one for a single frame (an upper bound, DESIGN §9.31)
    and the measured probe for several (vae_video)."""
    return None


def match(vae, samples, vae_options):
    """(bound adapter, None) or (None, reason)."""
    fsm = vae.first_stage_model
    reason, _ = hv15_structure(fsm)
    if reason:
        return None, reason
    if vae_options:
        return None, "vae_options {} given".format(sorted(vae_options))
    if samples.ndim != 5:
        return None, "latent has {} dims".format(samples.ndim)
    if samples.shape[1] != fsm.decoder.z_channels:
        return None, "latent has {} channels, the decoder takes {}".format(samples.shape[1], fsm.decoder.z_channels)
    if int(samples.shape[2]) > 1 and not MULTIFRAME_ENABLED:
        return None, "video (HunyuanVideo 1.5): layer 1 off (MULTIFRAME_ENABLED, DESIGN §9.41)"
    return HV15Stripe(fsm, int(samples.shape[2]) == 1), None


class _Shim:
    """Duck-types the attributes hv15_structure / HV15Stripe read from the first-stage model."""

    def __init__(self, holder):
        self.decoder = holder.decoder
        self._holder = holder

    def named_modules(self):
        return self._holder.named_modules()
