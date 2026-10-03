"""Layer-1 adapter: the LDM decoder (comfy.ldm.modules.diffusionmodules.model.Decoder)
-- the VAEs of SD1.5 / SDXL (AutoencoderKL, with post_quant_conv), SD3 and
Flux `ae` (AutoencodingEngine) -- decoding an image (docs/DESIGN.md §9.14).

In 62b3c94 a 4D decode of a 2D Decoder (conv3d off, so not carried) is a plain
sequence of module calls ([post_quant_conv] then Decoder.forward):

    conv_in -> mid.block_1 -> mid.attn_1 -> mid.block_2
    -> up[L-1].block[*] -> up[L-1].upsample -> up[L-2].block[*] -> ... -> up[0].block[*]
    -> norm_out -> nonlinearity (F.silu) -> conv_out

The prefix runs up to the last block of the lowest resolution (H/8, the mid
attention included); its output is the checkpoint. The rest is local in H
except for its GroupNorms (32 groups over the whole image): every
ResnetBlock's norm1 / norm2 and norm_out need statistics of their whole input.
The engine gets them in statistics passes over stripes (each from the latest
save), freezes them, and the final pass decodes the output stripes with the
norms using the frozen statistics (vae_engine.GlobalNorms). Which inputs are
kept whole (saves) is the GroupNorm scheme:

    A  none: every pass starts from the H/8 checkpoint
    D  the output of the first stripe level (H/4)
    B  the output of every level below full resolution (H/4, H/2)
    C  B and the output of every block at full resolution

Recognized only when every piece is the one checked here (exact classes,
kernels, paddings, no attention in the up levels, no time dimension / carried
convs, no tanh_out / give_pre_end, no hooks); anything else stays on layer 2
with the reason logged. Tensors are 4D [B, C, H, W].
"""

import torch

from comfy.ldm.models import autoencoder as ae
from comfy.ldm.modules.diffusionmodules import model as ldm

from . import vae_engine as eng
from .vae_engine import CONV, POINT, RES, UP, NormRef, Unit, conv_extra, conv_io, norm_temp
from .vae_wan import attention_peak, attn_split, no_dropout

PART = "part"   # the start of a ResnetBlock up to norm2's input (norm1 -> swish -> conv1): a statistics pass's last step

SCHEMES = ("A", "B", "C", "D")
DEFAULT_SCHEME = "A"
_SCHEME = [DEFAULT_SCHEME]


def scheme():
    return _SCHEME[0]


def set_scheme(name):
    """GroupNorm scheme for the next decodes (vae.py: MONOLOAD_VAE_GN_SCHEME, set_gn_scheme)."""
    name = (name or DEFAULT_SCHEME).upper()
    if name not in SCHEMES:
        raise ValueError(name)
    _SCHEME[0] = name


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def _conv2d(m, k, name):
    if not isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d):
        return "{} is {}, not a Conv2d".format(name, type(m).__name__)
    p = k // 2
    if (tuple(m.kernel_size) != (k, k) or tuple(m.stride) != (1, 1) or m.padding != (p, p) or tuple(m.dilation) != (1, 1)
            or m.groups != 1 or m.padding_mode != "zeros"):
        return "{}: kernel {} stride {} padding {} dilation {} groups {} {} (expected {}x{}, stride 1, padding {}, zeros)".format(
            name, tuple(m.kernel_size), tuple(m.stride), m.padding, tuple(m.dilation), m.groups, m.padding_mode, k, k, p)
    return None


def _group_norm(m, c, name):
    if not isinstance(m, torch.nn.GroupNorm):
        return "{} is {}, not a GroupNorm".format(name, type(m).__name__)
    if not m.affine or m.num_channels != c or c % m.num_groups:
        return "{}: GroupNorm({}, {}, affine={}) on {} channels".format(name, m.num_groups, m.num_channels, m.affine, c)
    return None


def _check_resnet(rb, name):
    if type(rb) is not ldm.ResnetBlock:
        return "{} is {}, not a ResnetBlock".format(name, type(rb).__name__)
    ci, co = rb.in_channels, rb.out_channels
    e = (_group_norm(rb.norm1, ci, name + ".norm1") or _group_norm(rb.norm2, co, name + ".norm2")
         or _conv2d(rb.conv1, 3, name + ".conv1") or _conv2d(rb.conv2, 3, name + ".conv2"))
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
            return "{}: 3x3 conv shortcut".format(name)
        e = _conv2d(rb.nin_shortcut, 1, name + ".nin_shortcut")
        if e:
            return e
    return None


def _check_upsample(up, c, name):
    if type(up) is not ldm.Upsample:
        return "{} is {}, not an Upsample".format(name, type(up).__name__)
    sf = up.scale_factor
    if not isinstance(sf, (int, float)) or float(sf) != 2.0:
        return "{}: scale factor {}".format(name, sf)
    if not up.with_conv:
        return "{}: without conv".format(name)
    e = _conv2d(up.conv, 3, name + ".conv")
    if e:
        return e
    if conv_io(up.conv) != (c, c):
        return "{}.conv channels {}".format(name, conv_io(up.conv))
    return None


def _is_ldm(fsm):
    return type(fsm) in (ae.AutoencoderKL, ae.AutoencodingEngineLegacy, ae.AutoencodingEngine) or isinstance(fsm, _Shim)


def ldm_structure(fsm):
    """(None, info) if fsm is an LDM image decoder the stripe decoder handles, else (reason, None)."""
    if not _is_ldm(fsm):
        return "first-stage model is {}, not comfy.ldm.models.autoencoder.AutoencoderKL / AutoencodingEngine".format(type(fsm).__name__), None
    dec = getattr(fsm, "decoder", None)
    if type(dec) is not ldm.Decoder:
        return "decoder is {}, not comfy.ldm.modules.diffusionmodules.model.Decoder".format(type(dec).__name__), None
    pqc = None
    if isinstance(fsm, _Shim):
        pqc = fsm.post_quant_conv
    elif type(fsm) is not ae.AutoencodingEngine:
        if getattr(fsm, "bn", None) is not None:
            return "batch_norm_latent (bn) before the decoder", None
        pqc = fsm.post_quant_conv
    if pqc is not None:
        e = _conv2d(pqc, 1, "post_quant_conv")
        if e:
            return e, None
    if getattr(dec, "carried", False):
        return "decoder with carried 3D convs (time dimension)", None
    if getattr(dec, "tanh_out", False):
        return "decoder with tanh_out", None
    if getattr(dec, "give_pre_end", False):
        return "decoder with give_pre_end", None
    e = _conv2d(dec.conv_in, 3, "decoder.conv_in")
    if e:
        return e, None
    c = conv_io(dec.conv_in)[1]
    if pqc is not None and conv_io(pqc)[1] != conv_io(dec.conv_in)[0]:
        return "post_quant_conv gives {} channels, conv_in takes {}".format(conv_io(pqc)[1], conv_io(dec.conv_in)[0]), None
    for nm in ("block_1", "block_2"):
        e = _check_resnet(getattr(dec.mid, nm), "decoder.mid." + nm)
        if e:
            return e, None
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
    levels = len(dec.up)
    if levels < 2 or levels != dec.num_resolutions:
        return "decoder with {} up levels".format(levels), None
    for i in reversed(range(levels)):
        up = dec.up[i]
        if len(up.attn) > 0:
            return "decoder.up[{}] has attention".format(i), None
        if len(up.block) != dec.num_res_blocks + 1:
            return "decoder.up[{}] has {} blocks".format(i, len(up.block)), None
        for j, rb in enumerate(up.block):
            name = "decoder.up[{}].block[{}]".format(i, j)
            e = _check_resnet(rb, name)
            if e:
                return e, None
            if rb.in_channels != c:
                return "{} takes {} channels, gets {}".format(name, rb.in_channels, c), None
            c = rb.out_channels
        if i != 0:
            e = _check_upsample(getattr(up, "upsample", None), c, "decoder.up[{}].upsample".format(i))
            if e:
                return e, None
        elif hasattr(up, "upsample"):
            return "decoder.up[0] has an upsample", None
    e = _group_norm(dec.norm_out, c, "decoder.norm_out") or _conv2d(dec.conv_out, 3, "decoder.conv_out")
    if e:
        return e, None
    h = eng.hooked_module(fsm)
    if h:
        return "module {} has a forward hook or an instance-level forward".format(h), None
    return None, {"post_quant_conv": pqc}


def _resnet_macs(rb):
    ci, co = rb.in_channels, rb.out_channels
    return 9 * (ci * co + co * co) + (ci * co if ci != co else 0)


def _partial(rb):
    """norm1 -> swish -> conv1 of a ResnetBlock: the input of its norm2 (as ResnetBlock.forward computes it)."""
    def run(x):
        return rb.conv1(rb.swish(rb.norm1(x)))
    return run


def build_units(fsm, pqc):
    """(prefix modules, stripe units)."""
    dec = fsm.decoder
    levels = len(dec.up)
    prefix = [("post_quant_conv", pqc)] if pqc is not None else []
    prefix += [("decoder.conv_in", dec.conv_in), ("decoder.mid.block_1", dec.mid.block_1),
               ("decoder.mid.attn_1", dec.mid.attn_1), ("decoder.mid.block_2", dec.mid.block_2)]
    top = dec.up[levels - 1]
    prefix += [("decoder.up.{}.block.{}".format(levels - 1, j), rb) for j, rb in enumerate(top.block)]
    units = []

    def upsample(i):
        u = dec.up[i].upsample
        c = conv_io(u.conv)[0]
        units.append(Unit(UP, u, 1, 2, c, c, "decoder.up.{}.upsample".format(i), 9 * c * c))
    upsample(levels - 1)
    for i in reversed(range(levels - 1)):
        for j, rb in enumerate(dec.up[i].block):
            name = "decoder.up.{}.block.{}".format(i, j)
            ci, co = rb.in_channels, rb.out_channels
            part = Unit(PART, _partial(rb), 1, 1, ci, co, name + "[:conv1]", 9 * ci * co)
            # with a nin_shortcut the row slice is copied before the call: Slow2d would copy it for the 1x1 conv at the
            # end of the block, when the block's own temporaries have fragmented the arena (DESIGN §9.14)
            units.append(Unit(RES, rb, 2, 1, ci, co, name, _resnet_macs(rb), ci != co,
                              norms=(NormRef(rb.norm1, name + ".norm1"), NormRef(rb.norm2, name + ".norm2", part)), contiguous=ci != co))
        if i != 0:
            upsample(i)
    c = dec.norm_out.num_channels
    units.append(Unit(POINT, dec.norm_out, 0, 1, c, c, "decoder.norm_out", norms=(NormRef(dec.norm_out, "decoder.norm_out"),)))
    units.append(Unit(POINT, ldm.nonlinearity, 0, 1, c, c, "decoder.nonlinearity"))
    cin, cout = conv_io(dec.conv_out)
    units.append(Unit(CONV, dec.conv_out, 1, 1, cin, cout, "decoder.conv_out", 9 * cin * cout))
    return prefix, units


def scheme_positions(units, name):
    """Save positions (unit indices whose input is kept whole) of a GroupNorm scheme."""
    ups = [i for i, u in enumerate(units) if u.kind == UP]
    level_ends = ups[1:]                                     # input of every later upsample = the output of a level
    full = [i + 1 for i, u in enumerate(units) if u.kind == RES and i > ups[-1]]
    return {"A": [], "D": level_ends[:1], "B": level_ends, "C": level_ends + full}[name]


# ---------------------------------------------------------------------------
# memory model (bytes)
# ---------------------------------------------------------------------------
# Live tensors, counted from the 62b3c94 forward code. S is the storage of a
# module's input (held by the caller), A / B one plane of input / output
# channels on the rows the module runs on, e = element size, G the fp32 chunk
# of a GroupNorm with frozen statistics (vae_engine.group_norm_frozen; 0 for the
# native GroupNorm of the prefix, which only adds its output).
#   ResnetBlock   norm1 out A + G | swish in place, conv1: A + B + extra | norm2:
#                 conv1 out B + norm2 out B + G | conv2: 2B + extra | x + h 2B (with
#                 nin_shortcut: h B + its output B, never row-blocked; then 3B)  S + max(...)
#                 A block with a nin_shortcut gets its row slice copied first
#                 (Unit.contiguous): S + A while copying, then A + max(...); else
#                 Slow2d would copy the slice for the 1x1 conv at the end of the
#                 block, into an arena the block has fragmented
#   part          the first two phases of the ResnetBlock                        S + max(...)
#   Upsample      contiguous copy A + interpolated 4A | 4A + conv out 4B + extra S + max(...)
#   norm_out      out A + G                                                       S + A + G
#   nonlinearity  F.silu (not in place)                                           S + A
#   conv_out      output + extra (input is a row slice: copied)                   S + B + extra
#   AttnBlock     (prefix) as Wan's AttentionBlock (vae_wan.attention_peak)

def res_peak(cin, cout, shortcut, r, w, e, ws, s, frozen=True, part=False, contiguous=False):
    """contiguous: the caller copies the row slice first (s + A while copying, then A is the input's storage)."""
    a, b = r * w * cin * e, r * w * cout * e
    g1 = norm_temp(cin, r, w, ws) if frozen else 0
    g2 = norm_temp(cout, r, w, ws) if frozen else 0
    phases = [a + g1, a + b + conv_extra(cin, cout, 3, r, r, w, w, e, ws)]
    if not part:
        phases += [2 * b + g2, 2 * b + conv_extra(cout, cout, 3, r, r, w, w, e, ws), 2 * b]
        if shortcut:
            # nin_shortcut(x): a 1x1 conv is never row-blocked (layer 2 leaves pointwise convs alone); Slow2d copies a
            # non-contiguous row slice whole (none to copy when the caller made it contiguous) and needs no columns
            phases += [2 * b + (0 if contiguous else a) + cin * cout * e, 3 * b]
    if contiguous:
        return max(s + a, a + max(phases))
    return s + max(phases)


def up_peak(cin, cout, r, w, e, ws, s):
    a = r * w * cin * e
    return s + max(5 * a, 4 * a + 4 * r * w * cout * e + conv_extra(cin, cout, 3, 2 * r, 2 * r, 2 * w, 2 * w, e, ws))


def unit_peak(u, r, w, e, ws, s):
    """Peak bytes while unit u runs on r input rows of width w, input storage s."""
    if u.kind in (RES, PART):
        return res_peak(u.cin, u.cout, u.shortcut, r, w, e, ws, s, part=u.kind == PART, contiguous=u.contiguous)
    if u.kind == UP:
        return up_peak(u.cin, u.cout, r, w, e, ws, s)
    if u.kind == CONV:
        return s + r * w * u.cout * e + conv_extra(u.cin, u.cout, 3, r, r, w, w, e, ws, contiguous=False)
    a = r * w * u.cin * e
    return s + a + (norm_temp(u.cin, r, w, ws) if u.norms else 0)


def unit_largest(u, r, w, e, ws):
    """Largest single allocation while unit u runs on r input rows of width w."""
    a, b = r * w * u.cin * e, r * w * u.cout * e
    if u.kind in (RES, PART):
        return max(a, b, min(ws, 9 * max(u.cin, u.cout) * r * w * e), norm_temp(max(u.cin, u.cout), r, w, ws))
    if u.kind == UP:
        return max(4 * a, 4 * b, min(ws, 9 * u.cin * 4 * r * w * e))
    if u.kind == CONV:
        return max(a, b, min(ws, 9 * u.cin * r * w * e))
    return max(a, norm_temp(u.cin, r, w, ws) if u.norms else 0)


def prefix_peak(m, h, w, e, ws, s):
    """Peak bytes while prefix module m runs on the whole h x w image (native GroupNorm), input storage s."""
    if isinstance(m, ldm.AttnBlock):
        return attention_peak(m.in_channels, h, w, e, ws, s, attn_split(m))
    if isinstance(m, ldm.ResnetBlock):
        return res_peak(m.in_channels, m.out_channels, m.in_channels != m.out_channels, h, w, e, ws, s, frozen=False)
    ci, co = conv_io(m)
    k = int(m.kernel_size[-1])
    return s + h * w * co * e + conv_extra(ci, co, k, h, h, w, w, e, ws)


def prefix_largest(m, h, w, e, ws):
    if isinstance(m, ldm.AttnBlock):
        return 3 * h * w * m.in_channels * e      # as vae_wan (q / k / v; score blocks are <= ws / 2 each)
    if isinstance(m, ldm.ResnetBlock):
        c = max(m.in_channels, m.out_channels)
        return max(h * w * e * c, min(ws, 9 * c * h * w * e))
    ci, co = conv_io(m)
    k = int(m.kernel_size[-1])
    return max(h * w * co * e, min(ws, k * k * ci * h * w * e))


def prefix_out_channels(m):
    if isinstance(m, ldm.AttnBlock):
        return m.in_channels
    if isinstance(m, ldm.ResnetBlock):
        return m.out_channels
    return conv_io(m)[1]


def prefix_macs(m):
    if isinstance(m, ldm.AttnBlock):
        return 4 * m.in_channels * m.in_channels
    if isinstance(m, ldm.ResnetBlock):
        return _resnet_macs(m)
    ci, co = conv_io(m)
    k = int(m.kernel_size[-1])
    return k * k * ci * co


# ---------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------

class LDMStripe(eng.StripeAdapter):
    """Layer-1 adapter bound to one LDM first-stage model."""

    name = "LDM stripes"
    hdim = 2                 # rows in [B, C, H, W]

    def __init__(self, fsm, pqc, gn_scheme=None, module=None):
        self.fsm = fsm
        self.module = fsm if module is None else module
        self.pqc = pqc
        self.prefix, self.units = build_units(fsm, pqc)
        self.scale = 1
        for u in self.units:
            self.scale *= u.scale
        self.ckpt_channels = prefix_out_channels(self.prefix[-1][1])
        self.out_channels = self.units[-1].cout
        self.gn_scheme = gn_scheme or scheme()
        self.saves = scheme_positions(self.units, self.gn_scheme)
        self.name = "LDM stripes, GroupNorm scheme {}".format(self.gn_scheme)
        dec = fsm.decoder
        self.key = ("LDM", type(fsm).__name__ if not isinstance(fsm, _Shim) else fsm.kind, pqc is not None, conv_io(dec.conv_in)[0],
                    int(dec.ch), int(dec.num_res_blocks), tuple((u.kind, u.cin, u.cout) for u in self.units), self.gn_scheme)

    unit_peak = staticmethod(unit_peak)
    unit_largest = staticmethod(unit_largest)
    prefix_peak = staticmethod(prefix_peak)
    prefix_largest = staticmethod(prefix_largest)
    prefix_out_channels = staticmethod(prefix_out_channels)
    prefix_macs = staticmethod(prefix_macs)

    def save_positions(self):
        return self.saves

    def output_shape(self, samples):
        return (samples.shape[0], self.out_channels, int(samples.shape[-2]) * self.scale, int(samples.shape[-1]) * self.scale)

    # self-test

    def selftest_params(self):
        return sum(p.numel() for p in self.fsm.decoder.parameters()) + (self.pqc.weight.numel() if self.pqc is not None else 0)

    def selftest_latent(self, n):
        return (1, conv_io(self.pqc if self.pqc is not None else self.fsm.decoder.conv_in)[0], n, n)

    def fp32_copy(self, device):
        """[post_quant_conv +] decoder rebuilt from the model's configuration in
        fp32 with the model's weights (the encoder is not copied); freed after
        the self-test."""
        fsm = self.fsm
        dec = fsm.decoder
        ch = int(dec.ch)
        mult = []
        for i in range(len(dec.up)):
            co = dec.up[i].block[0].out_channels
            if co % ch:
                raise eng.StripeError("decoder.up[{}] has {} channels, not a multiple of ch {}".format(i, co, ch))
            mult.append(co // ch)
        holder = torch.nn.Module()
        with torch.device(device):
            holder.decoder = ldm.Decoder(ch=ch, out_ch=conv_io(dec.conv_out)[1], ch_mult=mult, num_res_blocks=int(dec.num_res_blocks),
                                         attn_resolutions=[], dropout=0.0, resamp_with_conv=True, in_channels=dec.in_channels,
                                         resolution=dec.resolution, z_channels=conv_io(dec.conv_in)[0])
            pqc = None
            if self.pqc is not None:
                ci, co = conv_io(self.pqc)
                holder.post_quant_conv = pqc = type(self.pqc)(ci, co, 1)
        holder.to(device=device, dtype=torch.float32)
        holder.decoder.load_state_dict(dec.state_dict(), strict=True)
        if pqc is not None:
            pqc.load_state_dict(self.pqc.state_dict(), strict=True)
        holder.bn = None
        holder.max_batch_size = None
        holder.eval()
        shim = _Shim(holder, pqc, type(fsm).__name__)
        reason, info = ldm_structure(shim)
        if reason:
            raise eng.StripeError("fp32 copy does not match: " + reason)
        return _CopyStripe(shim, pqc, self.gn_scheme, holder, type(fsm))


class _CopyStripe(LDMStripe):
    """The adapter on the fp32 copy; its reference is the model class' own decode."""

    def __init__(self, shim, pqc, gn_scheme, holder, cls):
        super().__init__(shim, pqc, gn_scheme, module=holder)
        self.cls = cls

    def reference_decode(self, z):
        return self.cls.decode(self.module, z)


def match(vae, samples, vae_options):
    """(bound adapter, None) or (None, reason)."""
    fsm = vae.first_stage_model
    reason, info = ldm_structure(fsm)
    if reason:
        return None, reason
    if vae_options:
        return None, "vae_options {} given".format(sorted(vae_options))
    if samples.ndim != 4:
        return None, "latent has {} dims".format(samples.ndim)
    pqc = info["post_quant_conv"]
    zc = conv_io(pqc if pqc is not None else fsm.decoder.conv_in)[0]
    if samples.shape[1] != zc:
        return None, "latent has {} channels, the decoder takes {}".format(samples.shape[1], zc)
    return LDMStripe(fsm, pqc), None


class _Shim:
    """Duck-types the attributes ldm_structure / LDMStripe read from the first-stage model."""

    def __init__(self, holder, pqc, kind):
        self.decoder = holder.decoder
        self.post_quant_conv = pqc
        self.kind = kind
        self._holder = holder

    def named_modules(self):
        return self._holder.named_modules()
