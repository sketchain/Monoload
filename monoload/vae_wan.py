"""Layer-1 adapter: the Wan 2.1 VAE decoding a single frame (docs/DESIGN.md §9.13).

The Wan 2.1 VAE (comfy.ldm.wan.vae.WanVAE with Decoder3d; qwen_image_vae)
decoding a single frame. In 62b3c94 a T=1 decode is a plain sequence of module
calls (feat_map is None, CausalConv3d takes its autopad="causal_zero" fast
path, run_up never splits frames):

    conv2 -> decoder.conv1 -> decoder.middle[*] -> decoder.upsamples[*] -> decoder.head[*]

The modules up to the last one before the first Resample (conv1, middle with
the global attention, the ResidualBlocks of the lowest resolution) are the
prefix; their output is the checkpoint (H/8, 384 channels: ~0.1 GiB at 4K).
Everything after it is local in H -- ResidualBlocks (two 3x3 convs, RMS norm
per position), Resample (nearest x2 + 3x3 conv), RMS/SiLU/3x3 conv of the head
-- and runs in stripes (monoload/vae_engine.py). Tensors are 5D [B, C, T, H, W].
"""

import torch

import comfy.ldm.wan.vae as wan
from comfy.ldm.modules.diffusionmodules import model as ldm_model

from . import vae_engine as eng
from .vae_engine import CONV, POINT, RES, UP, Unit, conv_extra, conv_io


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

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


def no_dropout(m):
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
    for i, chk in ((0, _rms), (3, _rms), (5, no_dropout)):
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
    h = eng.hooked_module(fsm)
    if h:
        return "module {} has a forward hook or an instance-level forward".format(h), None
    return None, {"first_resample": first}


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
            cin, cout = conv_io(m.resample[1])
            units.append(Unit(UP, m, 1, 2, cin, cout, name, 9 * cin * cout))
        else:
            c1i, c1o = conv_io(m.residual[2])
            c2i, c2o = conv_io(m.residual[6])
            sc = not isinstance(m.shortcut, torch.nn.Identity)
            macs = 9 * (c1i * c1o + c2i * c2o) + (c1i * c2o if sc else 0)
            units.append(Unit(RES, m, 2, 1, c1i, c2o, name, macs, sc))
    rms, silu, conv = list(dec.head)
    c = int(rms.gamma.shape[0])
    cin, cout = conv_io(conv)
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
# the weight copy (vae_engine.conv_extra).

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


def unit_largest(u, r, w, e, ws):
    """Largest single allocation while unit u runs on r input rows of width w."""
    a, b = r * w * u.cin * e, r * w * u.cout * e
    if u.kind == RES:
        return max(a, b, min(ws, 9 * max(u.cin, u.cout) * r * w * e))
    if u.kind == UP:
        return max(4 * a, 4 * b, min(ws, 9 * u.cin * 4 * r * w * e))
    if u.kind == CONV:
        return max(a, b, min(ws, 9 * u.cin * r * w * e))
    return a


def prefix_largest(m, h, w, e, ws):
    if isinstance(m, wan.AttentionBlock):
        return 3 * h * w * int(m.norm.gamma.shape[0]) * e      # qkv (score blocks are <= ws / 2 each)
    if isinstance(m, wan.ResidualBlock):
        ci, co = conv_io(m.residual[2])[0], conv_io(m.residual[6])[1]
        return max(h * w * e * max(ci, co), min(ws, 9 * max(ci, co) * h * w * e))
    ci, co = conv_io(m)
    k = int(m.kernel_size[-1])
    return max(h * w * co * e, min(ws, k * k * ci * h * w * e))


def attn_split(m):
    return m.__dict__.get("optimized_attention") is ldm_model.normal_attention


def attention_peak(c, h, w, e, ws, s, split):
    """norm out P + qkv 3P + attention out P + one score block and its softmax
    (split attention; SDPA / xformers copy q/k/v/out: +3P)."""
    p, n = h * w * c * e, h * w
    rows = min(n, max(1, ws // (2 * n * e)))
    scores = 2 * rows * n * e + 2 * rows * c * e
    return s + (5 if split else 8) * p + scores


def prefix_peak(m, h, w, e, ws, s):
    """Peak bytes while prefix module m runs on the whole h x w image, input storage s."""
    if isinstance(m, wan.AttentionBlock):
        return attention_peak(int(m.norm.gamma.shape[0]), h, w, e, ws, s, attn_split(m))
    if isinstance(m, wan.ResidualBlock):
        ci, co = conv_io(m.residual[2])[0], conv_io(m.residual[6])[1]
        return res_peak(ci, co, not isinstance(m.shortcut, torch.nn.Identity), h, w, e, ws, s)
    ci, co = conv_io(m)
    k = int(m.kernel_size[-1])
    return s + h * w * co * e + conv_extra(ci, co, k, h, h, w, w, e, ws)


def prefix_out_channels(m):
    if isinstance(m, wan.AttentionBlock):
        return int(m.norm.gamma.shape[0])
    if isinstance(m, wan.ResidualBlock):
        return conv_io(m.residual[6])[1]
    return conv_io(m)[1]


def prefix_macs(m):
    if isinstance(m, wan.AttentionBlock):
        c = int(m.norm.gamma.shape[0])
        return 4 * c * c          # qkv + proj (the attention itself is the same in both)
    if isinstance(m, wan.ResidualBlock):
        a, b = conv_io(m.residual[2]), conv_io(m.residual[6])
        return 9 * (a[0] * a[1] + b[0] * b[1]) + (a[0] * b[1] if not isinstance(m.shortcut, torch.nn.Identity) else 0)
    ci, co = conv_io(m)
    k = int(m.kernel_size[-1])
    return k * k * ci * co


# ---------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------

class WanStripe(eng.StripeAdapter):
    """Layer-1 adapter bound to one WanVAE."""

    name = "Wan 2.1 stripes"
    hdim = 3                 # rows in [B, C, T, H, W]
    scale = 8

    def __init__(self, fsm, first_resample, module=None):
        self.fsm = fsm
        self.module = fsm if module is None else module
        self.prefix, self.units = build_units(fsm, first_resample)
        last = self.prefix[-1][1]
        self.ckpt_channels = prefix_out_channels(last)
        self.out_channels = self.units[-1].cout
        self.key = signature(fsm, self.units)

    unit_peak = staticmethod(unit_peak)
    unit_largest = staticmethod(unit_largest)
    prefix_peak = staticmethod(prefix_peak)
    prefix_largest = staticmethod(prefix_largest)
    prefix_out_channels = staticmethod(prefix_out_channels)
    prefix_macs = staticmethod(prefix_macs)

    def output_shape(self, samples):
        return (samples.shape[0], self.out_channels, samples.shape[2], int(samples.shape[-2]) * self.scale, int(samples.shape[-1]) * self.scale)

    # self-test

    def selftest_params(self):
        return sum(p.numel() for p in self.fsm.decoder.parameters()) + self.fsm.conv2.weight.numel()

    def selftest_latent(self, n):
        return (1, int(self.fsm.conv2.weight.shape[1]), 1, n, n)

    def fp32_copy(self, device):
        """conv2 + decoder rebuilt from the model's configuration in fp32 with the
        model's weights (the encoder is not copied); freed after the self-test."""
        fsm = self.fsm
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
        shim = _Shim(holder)
        reason, info = wan_structure(shim)
        if reason:
            raise eng.StripeError("fp32 copy does not match: " + reason)
        return WanStripe(shim, info["first_resample"], module=holder)

    def reference_decode(self, z):
        return wan.WanVAE.decode(self.module, z)


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
