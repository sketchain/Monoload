"""Layer-1 adapter: the Wan 2.2 VAE decoding a single frame (docs/DESIGN.md §9.23).

The Wan 2.2 VAE (comfy.ldm.wan.vae2_2.WanVAE with its own Decoder3d; z 48,
16x: the decoder works at 8x with 12 output channels, then unpatchify(2)) is a
different class from the Wan 2.1 one (monoload/vae_wan.py), with the same
CausalConv3d / RMS_norm / AttentionBlock. In 62b3c94 WanVAE.decode always passes
a feat_cache list, but for a single frame (the first chunk) every cached call
takes the uncached path: CausalConv3d gets cache_x None and T=1 (the
autopad="causal_zero" fast path), Resample("upsample3d") only marks its cache
("Rep") and skips time_conv. So a T=1 decode is the module calls

    conv2 -> decoder.conv1 -> decoder.middle[*] -> decoder.upsamples[*](first_chunk=True)
    -> decoder.head[*] -> unpatchify(patch_size=2)

and calling the modules without a cache (feat_cache None) computes the same.
Each Up_ResidualBlock is x_main = ResidualBlocks then Resample (nearest x2 +
3x3 conv) on x, plus avg_shortcut = DupUp3D(x, first_chunk) (channels repeated,
depth-to-space 2x2, for a temporal block the last of the two time slices), the
sum of the two; the last one has no Resample and no shortcut.

The prefix is conv2, decoder.conv1 and middle (with the global attention); its
output is the checkpoint (latent resolution H/16, 1024 channels: ~66 MiB at 4K,
bf16). The rest is local in H and runs in stripes (monoload/vae_engine.py):

  * an Up_ResidualBlock with a shortcut is one unit (DUPUP: the shortcut needs
    the block's input, so the block cannot be cut into a chain), called as the
    decode calls it (first_chunk=True). Scale 2; with k ResidualBlocks its
    halo at the output level is 4k + 1: k blocks of two 3x3 convs need 2k input
    rows on each side, the conv after the upsampling one output row, the
    shortcut is per row. need_in with that halo is exactly the input rows the
    block reads;
  * the ResidualBlocks of an Up_ResidualBlock without a shortcut are units of
    their own (RES), as are the head's RMS_norm, SiLU and conv;
  * unpatchify is the last unit (UNPATCH: depth-to-space 2x2 per row, halo 0).

Tensors are 5D [B, C, T, H, W]. No norm needs whole-image statistics (RMS_norm
is per position), so there are no statistics passes and no schemes.
"""

import torch

import comfy.ldm.wan.vae as wan
import comfy.ldm.wan.vae2_2 as wan22

from . import vae_engine as eng
from .vae_engine import CONV, POINT, RES, Unit, conv_extra, conv_io
from .vae_wan import _causal_conv, _rms, attention_peak, attn_split, no_dropout, res_peak

DUPUP = "dupup"       # an Up_ResidualBlock with its DupUp3D shortcut (scale 2)
UNPATCH = "unpatch"   # unpatchify(patch_size=2) of the decoder output (scale 2)
PATCH = 2             # WanVAE.decode's unpatchify patch size (62b3c94)


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def _check_residual(rb, name):
    if type(rb) is not wan22.ResidualBlock:
        return "{} is {}, not a vae2_2.ResidualBlock".format(name, type(rb).__name__)
    r = list(rb.residual)
    kinds = [type(x) for x in r]
    want = [wan.RMS_norm, torch.nn.SiLU, wan.CausalConv3d, wan.RMS_norm, torch.nn.SiLU, torch.nn.Dropout, wan.CausalConv3d]
    if kinds != want:
        return "{}.residual is [{}]".format(name, ", ".join(k.__name__ for k in kinds))
    for i, chk in ((0, _rms), (3, _rms), (5, no_dropout)):
        e = chk(r[i])
        if e:
            return "{}.residual[{}]: {}".format(name, i, e)
    for i in (2, 6):
        e = _causal_conv(r[i], 3)
        if e:
            return "{}.residual[{}]: {}".format(name, i, e)
    if conv_io(r[2]) != (rb.in_dim, rb.out_dim) or conv_io(r[6]) != (rb.out_dim, rb.out_dim):
        return "{}: conv channels {} / {} for {} -> {}".format(name, conv_io(r[2]), conv_io(r[6]), rb.in_dim, rb.out_dim)
    if isinstance(rb.shortcut, torch.nn.Identity):
        if rb.in_dim != rb.out_dim:
            return "{}: identity shortcut with {} -> {} channels".format(name, rb.in_dim, rb.out_dim)
    else:
        e = _causal_conv(rb.shortcut, 1)
        if e:
            return "{}.shortcut: {}".format(name, e)
    return None


def _check_resample(rs, c, name):
    if type(rs) is not wan22.Resample:
        return "{} is {}, not a vae2_2.Resample".format(name, type(rs).__name__)
    if rs.mode not in ("upsample2d", "upsample3d"):
        return "{} mode {}".format(name, rs.mode)
    seq = list(rs.resample)
    if len(seq) != 2 or type(seq[0]) is not torch.nn.Upsample or not isinstance(seq[1], torch.nn.Conv2d) or isinstance(seq[1], torch.nn.Conv3d):
        return "{}.resample is [{}]".format(name, ", ".join(type(x).__name__ for x in seq))
    up, conv = seq
    sf = up.scale_factor if isinstance(up.scale_factor, tuple) else (up.scale_factor, up.scale_factor)
    if up.mode not in ("nearest-exact", "nearest") or tuple(float(s) for s in sf) != (2.0, 2.0) or up.size is not None:
        return "{} upsample mode {} scale {}".format(name, up.mode, up.scale_factor)
    if (tuple(conv.kernel_size) != (3, 3) or tuple(conv.stride) != (1, 1) or tuple(conv.padding) != (1, 1)
            or tuple(conv.dilation) != (1, 1) or conv.groups != 1 or conv.padding_mode != "zeros"):
        return "{} conv kernel {} stride {} padding {}".format(name, conv.kernel_size, conv.stride, conv.padding)
    if conv_io(conv) != (c, c):
        return "{} conv channels {} on {} channels".format(name, conv_io(conv), c)
    return None


def _check_dupup(d, ci, co, name):
    if type(d) is not wan22.DupUp3D:
        return "{} is {}, not a DupUp3D".format(name, type(d).__name__)
    if d.in_channels != ci or d.out_channels != co or d.factor_s != 2 or d.factor_t not in (1, 2):
        return "{}: DupUp3D({} -> {}, factor_t {}, factor_s {}) on a {} -> {} block".format(
            name, d.in_channels, d.out_channels, d.factor_t, d.factor_s, ci, co)
    if d.factor != d.factor_t * 4 or d.repeats * ci != co * d.factor:
        return "{}: DupUp3D repeats {} factor {}".format(name, d.repeats, d.factor)
    return None


def _check_up_block(blk, c, name):
    """(error, output channels) of decoder.upsamples[i]."""
    if type(blk) is not wan22.Up_ResidualBlock:
        return "{} is {}, not an Up_ResidualBlock".format(name, type(blk).__name__), None
    mods = list(blk.upsamples)
    rs = mods[-1] if mods and isinstance(mods[-1], wan22.Resample) else None
    res = mods[:-1] if rs is not None else mods
    if not res:
        return "{} has no ResidualBlock".format(name), None
    ci = c
    for j, rb in enumerate(res):
        e = _check_residual(rb, "{}.upsamples[{}]".format(name, j))
        if e:
            return e, None
        if rb.in_dim != c:
            return "{}.upsamples[{}] takes {} channels, gets {}".format(name, j, rb.in_dim, c), None
        c = rb.out_dim
    if (rs is None) != (blk.avg_shortcut is None):
        return "{}: Resample {} with avg_shortcut {}".format(name, rs is not None, type(blk.avg_shortcut).__name__), None
    if rs is not None:
        e = _check_resample(rs, c, "{}.upsamples[{}]".format(name, len(res))) or _check_dupup(blk.avg_shortcut, ci, c, name + ".avg_shortcut")
        if e:
            return e, None
    return None, c


def wan22_structure(fsm):
    """(None, info) if fsm is a Wan 2.2 VAE the stripe decoder handles, else (reason, None)."""
    if not _is_wan22(fsm):
        return "first-stage model is {}.{}, not comfy.ldm.wan.vae2_2.WanVAE".format(type(fsm).__module__, type(fsm).__name__), None
    dec = getattr(fsm, "decoder", None)
    if type(dec) is not wan22.Decoder3d:
        return "decoder is {}, not vae2_2.Decoder3d".format(type(dec).__name__), None
    e = _causal_conv(fsm.conv2, 1) or _causal_conv(dec.conv1, 3)
    if e:
        return "conv2 / decoder.conv1: " + e, None
    if conv_io(fsm.conv2)[1] != conv_io(dec.conv1)[0]:
        return "conv2 gives {} channels, decoder.conv1 takes {}".format(conv_io(fsm.conv2)[1], conv_io(dec.conv1)[0]), None
    c = conv_io(dec.conv1)[1]
    mid = list(dec.middle)
    for i, m in enumerate(mid):
        if isinstance(m, wan.AttentionBlock):
            if type(m) is not wan.AttentionBlock or int(m.norm.gamma.shape[0]) != c:
                return "decoder.middle[{}]: attention on {} channels".format(i, c), None
            continue
        e = _check_residual(m, "decoder.middle[{}]".format(i))
        if e:
            return e, None
        if m.in_dim != c or m.out_dim != c:
            return "decoder.middle[{}]: {} -> {} channels on {}".format(i, m.in_dim, m.out_dim, c), None
    ups = list(dec.upsamples)
    if not ups or not any(b.avg_shortcut is not None for b in ups if type(b) is wan22.Up_ResidualBlock):
        return "decoder.upsamples has no upsampling block", None
    for i, blk in enumerate(ups):
        e, c = _check_up_block(blk, c, "decoder.upsamples[{}]".format(i))
        if e:
            return e, None
    head = list(dec.head)
    if [type(x) for x in head] != [wan.RMS_norm, torch.nn.SiLU, wan.CausalConv3d]:
        return "decoder.head is [{}]".format(", ".join(type(x).__name__ for x in head)), None
    e = _rms(head[0]) or _causal_conv(head[2], 3)
    if e:
        return "decoder.head: " + e, None
    if int(head[0].gamma.shape[0]) != c or conv_io(head[2])[0] != c:
        return "decoder.head on {} channels gets {}".format(conv_io(head[2])[0], c), None
    if conv_io(head[2])[1] % (PATCH * PATCH):
        return "decoder.head gives {} channels (unpatchify {}x{} needs a multiple of {})".format(conv_io(head[2])[1], PATCH, PATCH, PATCH * PATCH), None
    h = eng.hooked_module(fsm)
    if h:
        return "module {} has a forward hook or an instance-level forward".format(h), None
    return None, {}


class _UpBlock:
    """An Up_ResidualBlock called as WanVAE.decode calls it on the first chunk (first_chunk=True; no cache: the same)."""
    __slots__ = ("blk",)

    def __init__(self, blk):
        self.blk = blk

    def __call__(self, x):
        return self.blk(x, None, [0], True)


def _unpatch(x):
    return wan22.unpatchify(x, patch_size=PATCH)


def _res_macs(rb):
    ci, co = rb.in_dim, rb.out_dim
    return 9 * (ci * co + co * co) + (ci * co if ci != co else 0)


def build_units(fsm):
    """(prefix modules, stripe units)."""
    dec = fsm.decoder
    prefix = [("conv2", fsm.conv2), ("decoder.conv1", dec.conv1)]
    prefix += [("decoder.middle.{}".format(i), m) for i, m in enumerate(dec.middle)]
    units = []
    for i, blk in enumerate(dec.upsamples):
        name = "decoder.upsamples.{}".format(i)
        mods = list(blk.upsamples)
        if blk.avg_shortcut is None:
            for j, rb in enumerate(mods):
                units.append(Unit(RES, rb, 2, 1, rb.in_dim, rb.out_dim, "{}.upsamples.{}".format(name, j), _res_macs(rb),
                                  not isinstance(rb.shortcut, torch.nn.Identity)))
            continue
        res = mods[:-1]
        ci, co = res[0].in_dim, res[-1].out_dim
        blocks = tuple((rb.in_dim, rb.out_dim, not isinstance(rb.shortcut, torch.nn.Identity)) for rb in res)
        # MACs per output pixel: the blocks at the input resolution (a quarter of the output pixels), the upsampling conv
        macs = sum(_res_macs(rb) for rb in res) // 4 + 9 * co * co
        units.append(Unit(DUPUP, _UpBlock(blk), 4 * len(res) + 1, 2, ci, co, name, macs,
                          extra={"blocks": blocks, "ft": int(blk.avg_shortcut.factor_t)}))
    rms, silu, conv = list(dec.head)
    c = int(rms.gamma.shape[0])
    cin, cout = conv_io(conv)
    units.append(Unit(POINT, rms, 0, 1, c, c, "decoder.head.0"))
    units.append(Unit(POINT, silu, 0, 1, c, c, "decoder.head.1"))
    units.append(Unit(CONV, conv, 1, 1, cin, cout, "decoder.head.2", 9 * cin * cout))
    units.append(Unit(UNPATCH, _unpatch, 0, PATCH, cout, cout // (PATCH * PATCH), "unpatchify"))
    return prefix, units


def signature(fsm, units):
    """Structure key (self-test cache)."""
    dec = fsm.decoder
    return ("WanVAE2.2", int(fsm.conv2.weight.shape[0]), tuple(dec.dim_mult), int(dec.dim), int(dec.num_res_blocks),
            tuple((u.kind, u.cin, u.cout, (u.extra or {}).get("ft")) for u in units))


# ---------------------------------------------------------------------------
# memory model (bytes)
# ---------------------------------------------------------------------------
# Live tensors, counted from the 62b3c94 forward code (as vae_wan; S the storage
# of the unit's input, held by the caller, e the element size, A / B one plane
# of input / output channels at the unit's input rows).
#   ResidualBlock     as vae_wan.res_peak (the same forward)
#   DUPUP (k blocks)  the block's input x stays alive to the end (the shortcut reads it):
#                     block 1 on x: res_peak(S) | block j: S + res_peak(input B)
#                     | Resample: S + its input B + upsampled 4B + conv out 4B + extra
#                     | DupUp3D: S + x_main 4B + repeat_interleave 4ft B + its permuted copy 4ft B
#                     | the sum: S + 4B + shortcut 4ft B + out 4B
#   head              RMS / SiLU / conv as vae_wan (the conv's input is a row slice: copied)
#   UNPATCH           einops rearrange of a row slice: one copy, S + A
#   prefix            conv2, conv1, middle (ResidualBlock / AttentionBlock) as vae_wan

def dupup_peak(u, r, w, e, ws, s):
    blocks, ft = u.extra["blocks"], u.extra["ft"]
    b = r * w * u.cout * e
    phases = []
    for j, (ci, co, sc) in enumerate(blocks):
        phases.append(res_peak(ci, co, sc, r, w, e, ws, s) if j == 0 else s + res_peak(ci, co, sc, r, w, e, ws, b))
    co = u.cout
    phases.append(s + 9 * b + conv_extra(co, co, 3, 2 * r, 2 * r, 2 * w, 2 * w, e, ws))
    phases.append(s + 4 * b + 8 * ft * b)
    phases.append(s + 8 * b + 4 * ft * b)
    return max(phases)


def unit_peak(u, r, w, e, ws, s):
    """Peak bytes while unit u runs on r input rows of width w, input storage s."""
    if u.kind == DUPUP:
        return dupup_peak(u, r, w, e, ws, s)
    if u.kind == RES:
        return res_peak(u.cin, u.cout, u.shortcut, r, w, e, ws, s)
    if u.kind == CONV:
        return s + r * w * u.cout * e + conv_extra(u.cin, u.cout, 3, r, r, w, w, e, ws, contiguous=False)
    if u.kind == UNPATCH:
        return s + r * w * u.cin * e
    return s + 2 * r * w * u.cin * e


def unit_largest(u, r, w, e, ws):
    """Largest single allocation while unit u runs on r input rows of width w."""
    a, b = r * w * u.cin * e, r * w * u.cout * e
    if u.kind == DUPUP:
        cmax = max(max(ci, co) for ci, co, _ in u.extra["blocks"])
        return max(a, 4 * u.extra["ft"] * b, 4 * b, min(ws, 9 * u.cout * 4 * r * w * e), min(ws, 9 * cmax * r * w * e))
    if u.kind == RES:
        return max(a, b, min(ws, 9 * max(u.cin, u.cout) * r * w * e))
    if u.kind == CONV:
        return max(a, b, min(ws, 9 * u.cin * r * w * e))
    return a


def prefix_peak(m, h, w, e, ws, s):
    """Peak bytes while prefix module m runs on the whole h x w latent, input storage s."""
    if isinstance(m, wan.AttentionBlock):
        return attention_peak(int(m.norm.gamma.shape[0]), h, w, e, ws, s, attn_split(m))
    if isinstance(m, wan22.ResidualBlock):
        return res_peak(m.in_dim, m.out_dim, not isinstance(m.shortcut, torch.nn.Identity), h, w, e, ws, s)
    ci, co = conv_io(m)
    k = int(m.kernel_size[-1])
    return s + h * w * co * e + conv_extra(ci, co, k, h, h, w, w, e, ws)


def prefix_largest(m, h, w, e, ws):
    if isinstance(m, wan.AttentionBlock):
        return 3 * h * w * int(m.norm.gamma.shape[0]) * e
    if isinstance(m, wan22.ResidualBlock):
        c = max(m.in_dim, m.out_dim)
        return max(h * w * e * c, min(ws, 9 * c * h * w * e))
    ci, co = conv_io(m)
    k = int(m.kernel_size[-1])
    return max(h * w * co * e, min(ws, k * k * ci * h * w * e))


def prefix_out_channels(m):
    if isinstance(m, wan.AttentionBlock):
        return int(m.norm.gamma.shape[0])
    if isinstance(m, wan22.ResidualBlock):
        return m.out_dim
    return conv_io(m)[1]


def prefix_macs(m):
    if isinstance(m, wan.AttentionBlock):
        c = int(m.norm.gamma.shape[0])
        return 4 * c * c
    if isinstance(m, wan22.ResidualBlock):
        return _res_macs(m)
    ci, co = conv_io(m)
    k = int(m.kernel_size[-1])
    return k * k * ci * co


# ---------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------

class Wan22Stripe(eng.StripeAdapter):
    """Layer-1 adapter bound to one Wan 2.2 WanVAE."""

    name = "Wan 2.2 stripes"
    hdim = 3                 # rows in [B, C, T, H, W]

    def __init__(self, fsm, module=None):
        self.fsm = fsm
        self.module = fsm if module is None else module
        self.prefix, self.units = build_units(fsm)
        self.scale = 1
        for u in self.units:
            self.scale *= u.scale
        self.ckpt_channels = prefix_out_channels(self.prefix[-1][1])
        self.out_channels = self.units[-1].cout
        self.key = signature(fsm, self.units)

    unit_peak = staticmethod(unit_peak)
    unit_largest = staticmethod(unit_largest)
    prefix_peak = staticmethod(prefix_peak)
    prefix_largest = staticmethod(prefix_largest)
    prefix_out_channels = staticmethod(prefix_out_channels)
    prefix_macs = staticmethod(prefix_macs)

    def decode_retained(self, h, w, e):
        """WanVAE.decode passes a feat_cache even for one frame: every CausalConv3d it calls with the cache (decoder.conv1,
        the two convs of each ResidualBlock, the head's conv; not conv2, the 1x1 shortcuts, nor time_conv, which a first
        chunk skips) first clones its input ("cache_x = x[:, :, -CACHE_T:].clone()": the whole input for T=1) into
        feat_cache, which lives until decode returns. The sum of those inputs, at their resolution."""
        dec = self.fsm.decoder
        px = h * w
        n = conv_io(dec.conv1)[0] * px
        for m in dec.middle:
            if isinstance(m, wan22.ResidualBlock):
                n += (m.in_dim + m.out_dim) * px
        for blk in dec.upsamples:
            for m in blk.upsamples:
                if isinstance(m, wan22.ResidualBlock):
                    n += (m.in_dim + m.out_dim) * px
            if blk.avg_shortcut is not None:
                px *= 4
        n += conv_io(dec.head[2])[0] * px
        return n * e

    def output_shape(self, samples):
        return (samples.shape[0], self.out_channels, samples.shape[2], int(samples.shape[-2]) * self.scale, int(samples.shape[-1]) * self.scale)

    # self-test

    def selftest_params(self):
        """What fp32_copy copies: conv2 and the decoder, each CausalConv3d's weight as its last time slice."""
        n = 0
        for m in [self.fsm.conv2] + list(self.fsm.decoder.modules()):
            for name, p in m.named_parameters(recurse=False):
                n += p.numel() // (int(p.shape[2]) if isinstance(m, wan.CausalConv3d) and name == "weight" else 1)
        return n

    def selftest_latent(self, n):
        return (1, int(self.fsm.conv2.weight.shape[1]), 1, n, n)

    def fp32_copy(self, device):
        """conv2 + decoder rebuilt from the model's configuration in fp32 with the
        model's weights (the encoder is not copied); freed after the self-test.

        Each CausalConv3d keeps only the last time slice of its kernel: a single
        frame never reads the others (62b3c94: every call of a T=1 decode takes
        the autopad="causal_zero" path, which uses weight[:, :, -T:]; time_conv
        is skipped). The full kernels in fp32 would be ~2.2 GiB for the real
        decoder, the slices ~0.75 GiB (tests/test_vae_wan22.py checks that the
        copy's own WanVAE.decode equals the model's)."""
        fsm = self.fsm
        dec = fsm.decoder
        holder = torch.nn.Module()
        z = int(fsm.conv2.weight.shape[0])
        with eng.skeleton():
            holder.conv2 = wan.CausalConv3d(z, z, 1)
            holder.decoder = wan22.Decoder3d(dec.dim, dec.z_dim, list(dec.dim_mult), dec.num_res_blocks, list(dec.attn_scales),
                                             list(dec.temperal_upsample), 0.0)
            for m in holder.modules():
                if isinstance(m, wan.CausalConv3d) and m.weight.shape[2] > 1:
                    m.weight = torch.nn.Parameter(torch.empty(m.weight.shape[:2] + (1,) + m.weight.shape[3:]), requires_grad=False)
                    m.kernel_size = (1,) + tuple(m.kernel_size[1:])
                    m._padding = 0
        src = {"conv2": fsm.conv2, "decoder": dec}
        sd = {}
        for prefix, mod in src.items():
            names = dict(mod.named_modules())
            for k, v in mod.state_dict().items():
                owner = names.get(k.rpartition(".")[0]) if "." in k else mod
                if isinstance(owner, wan.CausalConv3d) and k.endswith("weight") and v.ndim == 5 and v.shape[2] > 1:
                    v = v[:, :, -1:]
                sd[prefix + "." + k] = v
        holder.to_empty(device=device)
        holder.to(dtype=torch.float32)
        holder.load_state_dict(sd, strict=True)
        holder.eval()
        shim = _Shim(holder)
        reason, _ = wan22_structure(shim)
        if reason:
            raise eng.StripeError("fp32 copy does not match: " + reason)
        return Wan22Stripe(shim, module=holder)

    def reference_decode(self, z):
        return wan22.WanVAE.decode(self.module, z)


def match(vae, samples, vae_options):
    """(bound adapter, None) or (None, reason)."""
    fsm = vae.first_stage_model
    reason, _ = wan22_structure(fsm)
    if reason:
        return None, reason
    if vae_options:
        return None, "vae_options {} given".format(sorted(vae_options))
    if samples.ndim != 5 or samples.shape[2] != 1:
        return None, "latent is not a single frame ({})".format(list(samples.shape))
    if samples.shape[1] != fsm.conv2.weight.shape[1]:
        return None, "latent has {} channels, conv2 expects {}".format(samples.shape[1], fsm.conv2.weight.shape[1])
    return Wan22Stripe(fsm), None


class _Shim:
    """Duck-types the attributes wan22_structure / Wan22Stripe read from a WanVAE."""

    def __init__(self, holder):
        self.conv2 = holder.conv2
        self.decoder = holder.decoder
        self._holder = holder

    def named_modules(self):
        return self._holder.named_modules()


def _is_wan22(fsm):
    return type(fsm) is wan22.WanVAE or isinstance(fsm, _Shim)
