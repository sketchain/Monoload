"""Op-level chunking for VAE decoding (layer 2 of docs/DESIGN.md §9).

Inside `OpChunking(model, budget)` -- and only there, only on the modules of
`model` -- two kinds of operators are replaced per instance; the decoder's own
forward runs unchanged, so the result is the whole-image computation up to
floating-point differences (different GEMM shapes), with ~1x the arithmetic:

  conv      every torch.nn.Conv2d / Conv3d (comfy.ops variants included) gets
            an instance attribute `_conv_forward`. Both ways a comfy.ops conv
            runs end there with the final weight:
              cast path      forward_comfy_cast_weights -> CastBiasWeightContext
                             (cast_bias_weight: dtype/device cast, weight_function
                             / bias_function such as Monoload's runtime LoRA)
                             -> self._conv_forward(input, weight, bias[, autopad])
              non-cast path  torch.nn.ConvNd.forward
                             -> self._conv_forward(input, self.weight, self.bias)
            so weight processing is never bypassed (the weight functions run
            once per conv call, not per chunk). When the estimated im2col
            workspace of the whole conv exceeds the budget, the output is
            computed in blocks of output rows (H). Each block reads exactly the
            input rows it needs, real neighbouring rows across block
            boundaries; zero padding is added only at the real top/bottom edge
            of the image. Each block goes through the original _conv_forward
            (so comfy's Conv3d autopad="causal_zero" weight truncation and any
            backend workaround still apply) with the H padding of the module
            temporarily set to 0. Output is preallocated once and filled block
            by block. A single-frame Conv3d call (effective kT=1) on the
            SlowDilated3d backend runs as the equivalent F.conv2d instead
            (DESIGN §9.13.9).
  attention the `optimized_attention` instance attribute of attention blocks
            (comfy.ldm.modules.diffusionmodules.model.AttnBlock, Wan's
            AttentionBlock) when it is one of ComfyUI's VAE attention functions
            (split / pytorch / xformers): queries are processed in blocks, K/V
            stay whole, so every query still takes a softmax over the whole
            image. The arithmetic per block is that of the original function.

Everything else (GroupNorm, RMS norm, upsampling, ...) stays native.
"""

import torch
import torch.nn.functional as F

import comfy.ops
from comfy.ldm.modules.diffusionmodules import model as ldm_model

GIB = 1024 ** 3
MIB = 1024 ** 2


class OpStats:
    """What the op-level chunking did during one managed decode."""

    def __init__(self):
        self.conv_modules = 0         # convs that got the wrapper
        self.attn_modules = 0         # attention blocks that got the wrapper
        self.attn_unmanaged = []      # attention blocks with an unknown implementation (left native)
        self.conv_calls = 0
        self.conv_chunked = 0         # conv calls that were split into row blocks
        self.conv_blocks = 0          # row blocks run in total
        self.conv_ws_max_full = 0     # largest whole-conv workspace estimate seen (bytes)
        self.conv_ws_max_block = 0    # largest per-block workspace estimate actually run (bytes)
        self.conv_rows_min = None     # smallest row-block height used
        self.conv_boundaries = set()  # (output rows of that conv, first row of a block > 0): where blocks meet
        self.attn_calls = 0
        self.attn_tokens_max = 0      # largest N (keys = queries)
        self.attn_rows_min = None     # smallest query-block size used
        self.attn_score_max = 0       # largest per-block score bytes (x2: scores + softmax)
        self.cache_releases = 0       # allocator cache emptied between the layer-1 prefix and its stripes
        self.conv3d_as_2d = 0         # single-frame Conv3d calls run as conv2d (no per-channel bias fill)

    def as_dict(self):
        d = dict(self.__dict__)
        d["conv_boundaries"] = sorted(self.conv_boundaries)
        return d


# ---------------------------------------------------------------------------
# convolution
# ---------------------------------------------------------------------------

def _tuple(v, n):
    if isinstance(v, (tuple, list)):
        return tuple(int(x) for x in v)
    return (int(v),) * n


def _explicit_zero_padding(mod, ksize, dilation):
    """(lo, hi) per spatial dim for padding_mode == 'zeros', or None when the
    module's padding cannot be expressed that way."""
    nd = len(ksize)
    p = mod.padding
    if isinstance(p, str):
        if p == "valid":
            return [(0, 0)] * nd
        if p == "same":
            out = []
            for k, d in zip(ksize, dilation):
                total = d * (k - 1)
                out.append((total // 2, total - total // 2))  # torch: extra on the high side
            return out
        return None
    p = _tuple(p, nd)
    if len(p) != nd:
        return None
    return [(x, x) for x in p]


def conv_out_size(size, lo, hi, k, s, d):
    return (size + lo + hi - d * (k - 1) - 1) // s + 1


def conv_rows_input_range(o0, o1, stride, dilation, kernel, pad_lo):
    """Input rows needed for output rows [o0, o1) along one dim, in input
    coordinates (may extend past the real edges into the padding):
    output row o reads padded rows s*o .. s*o + d*(k-1), i.e. input rows
    s*o - p_lo .. s*o - p_lo + d*(k-1)."""
    lo = stride * o0 - pad_lo
    hi = stride * (o1 - 1) + dilation * (kernel - 1) - pad_lo + 1
    return lo, hi


def conv_workspace_per_row(weight_shape, ksize_eff, out_other, batch, elem):
    """im2col / vol2col columns per output row: (Cin/groups) x prod(k) x
    (output elements of the other spatial dims) x batch x dtype bytes."""
    n = weight_shape[1]
    for k in ksize_eff:
        n *= k
    for o in out_other:
        n *= o
    return n * batch * elem


def slow_dilated3d(x):
    """True where torch runs a 5D conv on x with the SlowDilated3d backend
    (CUDA / HIP without cuDNN / MIOpen, e.g. ComfyUI's AMD default): it fills
    the output with the bias one channel at a time (one fill_ per output
    channel and call) before the GEMM."""
    return x.is_cuda and not torch.backends.cudnn.enabled and not getattr(comfy.ops, "NVIDIA_MEMORY_CONV_BUG_WORKAROUND", False)


# Conv3d classes whose _conv_forward is known: torch's, and comfy.ops' (causal_zero autopad = weight[:, :, -T:])
_CONV3D_FORWARDS = (torch.nn.Conv3d._conv_forward, comfy.ops.disable_weight_init.Conv3d._conv_forward)


class _ConvChunker:
    __slots__ = ("mod", "orig", "budget", "stats", "as2d")

    def __init__(self, mod, orig, budget, stats, as2d=False):
        self.mod = mod
        self.orig = orig
        self.budget = budget
        self.stats = stats
        self.as2d = as2d   # Conv3d with a known _conv_forward: single-frame calls may run as conv2d

    def _conv(self, x, weight, bias, args, kwargs):
        """self.orig(x, weight, bias, ...), except a single-frame Conv3d call on
        the SlowDilated3d backend: that runs as F.conv2d on frame 0 with the
        (effective) kT=1 weight -- the Slow2d backend sets the bias with one
        copy instead of a fill_ per output channel; im2col columns and the
        GEMM (operands, shapes, beta=1 onto the bias) are those of the 3D path."""
        mod = self.mod
        if self.as2d and x.shape[2] == 1 and slow_dilated3d(x) and mod.padding_mode == "zeros" and isinstance(mod.padding, tuple):
            autopad = kwargs.get("autopad", args[0] if args else None)
            pt, ph, pw = (int(p) for p in mod.padding)
            if pt == 0 and (autopad == "causal_zero" or weight.shape[2] == 1):
                self.stats.conv3d_as_2d += 1
                y = F.conv2d(x[:, :, 0], weight[:, :, -1], bias, _tuple(mod.stride, 3)[1:], (ph, pw), _tuple(mod.dilation, 3)[1:], mod.groups)
                return y.unsqueeze(2)
        return self.orig(x, weight, bias, *args, **kwargs)

    def __call__(self, input, weight, bias, *args, **kwargs):
        st = self.stats
        st.conv_calls += 1
        mod = self.mod
        nd = weight.ndim - 2
        if nd not in (2, 3) or input.ndim != weight.ndim or input.shape[0] == 0:
            return self.orig(input, weight, bias, *args, **kwargs)
        ksize = list(weight.shape[2:])
        if nd == 3 and kwargs.get("autopad", args[0] if args else None) == "causal_zero":
            ksize[0] = min(ksize[0], input.shape[2])  # comfy Conv3d truncates the time kernel to the input frames
        stride = _tuple(mod.stride, nd)
        dilation = _tuple(mod.dilation, nd)

        x = input
        if mod.padding_mode != "zeros":
            # Same as torch: pad the whole input in that mode, then convolve
            # without padding. Row blocks of the padded input are then exact.
            pads = None
        else:
            pads = _explicit_zero_padding(mod, ksize, dilation)
            if pads is None:
                return self.orig(input, weight, bias, *args, **kwargs)

        hd = nd - 1                # H among the spatial dims (T, H, W) / (H, W)
        hdim = 2 + hd              # H in the tensor
        if pads is None:
            rp = list(mod._reversed_padding_repeated_twice)  # (W_lo, W_hi, H_lo, H_hi[, T_lo, T_hi])
            pad_pairs = [(rp[2 * (nd - 1 - i)], rp[2 * (nd - 1 - i) + 1]) for i in range(nd)]
        else:
            pad_pairs = pads
        sizes = list(input.shape[2:])
        outs = [conv_out_size(sizes[i], pad_pairs[i][0], pad_pairs[i][1], ksize[i], stride[i], dilation[i]) for i in range(nd)]
        if min(outs) <= 0:
            return self.orig(input, weight, bias, *args, **kwargs)
        elem = input.element_size()
        pointwise = nd == 2 and ksize == [1, 1] and stride == (1, 1) and all(p == (0, 0) for p in pad_pairs)
        if pointwise:
            # Slow2d multiplies a 1x1 / stride 1 / unpadded conv directly (no
            # im2col): no workspace to bound.
            return self.orig(input, weight, bias, *args, **kwargs)
        per_row = conv_workspace_per_row(weight.shape, ksize, [o for i, o in enumerate(outs) if i != hd], input.shape[0], elem)
        h_out = outs[hd]
        full = per_row * h_out
        st.conv_ws_max_full = max(st.conv_ws_max_full, full)
        if full <= self.budget:
            return self._conv(input, weight, bias, args, kwargs)

        rows = max(1, int(self.budget // per_row))
        st.conv_chunked += 1
        st.conv_ws_max_block = max(st.conv_ws_max_block, per_row * min(rows, h_out))
        st.conv_rows_min = rows if st.conv_rows_min is None else min(st.conv_rows_min, rows)

        if pads is None:
            x = F.pad(input, mod._reversed_padding_repeated_twice, mode=mod.padding_mode)
            sizes = list(x.shape[2:])
            pad_pairs = [(0, 0)] * nd
        H = sizes[hd]
        kH, sH, dH = ksize[hd], stride[hd], dilation[hd]
        ph_lo = pad_pairs[hd][0]
        # Padding of the other dims: symmetric goes to the module (as native),
        # asymmetric is added explicitly to every block; H is handled per block.
        mod_pad = []
        other_explicit = {}
        for i in range(nd):
            lo, hi = pad_pairs[i]
            if i == hd:
                mod_pad.append(0)
            elif lo == hi:
                mod_pad.append(lo)
            else:
                mod_pad.append(0)
                other_explicit[i] = (lo, hi)

        saved_padding = mod.padding
        saved_mode = mod.padding_mode
        out = None
        try:
            mod.padding = tuple(mod_pad)
            mod.padding_mode = "zeros"
            for o0 in range(0, h_out, rows):
                o1 = min(h_out, o0 + rows)
                lo, hi = conv_rows_input_range(o0, o1, sH, dH, kH, ph_lo)
                a, b = max(lo, 0), min(hi, H)
                top, bottom = a - lo, hi - b
                xc = x.narrow(hdim, a, b - a)
                fpad = []
                for i in reversed(range(nd)):
                    if i == hd:
                        fpad += [top, bottom]
                    elif i in other_explicit:
                        fpad += list(other_explicit[i])
                    else:
                        fpad += [0, 0]
                if any(fpad):
                    xc = F.pad(xc, fpad)
                yc = self._conv(xc, weight, bias, args, kwargs)
                del xc
                if yc.shape[hdim] != o1 - o0:
                    raise RuntimeError("[Monoload] internal error: conv row block produced {} rows, expected {}".format(yc.shape[hdim], o1 - o0))
                if out is None:
                    shape = list(yc.shape)
                    shape[hdim] = h_out
                    out = torch.empty(shape, dtype=yc.dtype, device=yc.device)
                out.narrow(hdim, o0, o1 - o0).copy_(yc)
                del yc
                st.conv_blocks += 1
                if o0 > 0:
                    st.conv_boundaries.add((h_out, o0))
        finally:
            mod.padding = saved_padding
            mod.padding_mode = saved_mode
        return out


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------

def attention_rows(budget, batch, tokens, elem):
    """Query rows per block: the score block and its softmax (both
    rows x tokens) together stay within the budget."""
    return max(1, int(budget // (2 * max(1, batch) * max(1, tokens) * elem)))


def split_attention_chunked(q, k, v, budget, stats=None):
    """normal_attention + slice_attention of comfy/ldm/modules/diffusionmodules/model.py
    with a fixed query block size from the budget instead of the free-memory
    driven `steps` (same ops per block: bmm, * scale, softmax in the input
    dtype, bmm with V)."""
    orig_shape = q.shape
    b = orig_shape[0]
    c = orig_shape[1]
    q = q.reshape(b, c, -1)
    q = q.permute(0, 2, 1)   # b,hw,c
    k = k.reshape(b, c, -1)  # b,c,hw
    v = v.reshape(b, c, -1)
    n = q.shape[1]
    rows = attention_rows(budget, b, k.shape[2], q.element_size())
    _note_attn(stats, n, rows, b, k.shape[2], q.element_size())
    r1 = torch.zeros_like(k, device=q.device)
    scale = (int(q.shape[-1]) ** (-0.5))
    for i in range(0, n, rows):
        end = min(n, i + rows)
        s1 = torch.bmm(q[:, i:end], k) * scale
        s2 = torch.nn.functional.softmax(s1, dim=2).permute(0, 2, 1)
        del s1
        r1[:, :, i:end] = torch.bmm(v, s2)
        del s2
    h_ = r1.reshape(orig_shape)
    del r1
    return h_


def pytorch_attention_chunked(q, k, v, budget, stats=None):
    """pytorch_attention (SDPA, one head) with query blocks."""
    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )
    n = q.shape[2]
    rows = attention_rows(budget, B, k.shape[2], q.element_size())
    _note_attn(stats, n, rows, B, k.shape[2], q.element_size())
    if rows >= n:
        out = comfy.ops.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    else:
        out = torch.empty_like(q)
        for i in range(0, n, rows):
            end = min(n, i + rows)
            out[:, :, i:end] = comfy.ops.scaled_dot_product_attention(q[:, :, i:end], k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    return out.transpose(2, 3).reshape(orig_shape)


def xformers_attention_chunked(q, k, v, budget, stats=None):
    """xformers_attention with query blocks (falls back to the chunked split
    attention where xformers raises NotImplementedError, like native falls
    back to slice_attention)."""
    import xformers.ops
    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    qq, kk, vv = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )
    n = qq.shape[1]
    rows = attention_rows(budget, B, kk.shape[1], qq.element_size())
    try:
        out = torch.empty_like(qq)
        for i in range(0, n, rows):
            end = min(n, i + rows)
            out[:, i:end] = xformers.ops.memory_efficient_attention(qq[:, i:end], kk, vv, attn_bias=None)
        _note_attn(stats, n, rows, B, kk.shape[1], qq.element_size())
        return out.transpose(1, 2).reshape(orig_shape)
    except NotImplementedError:
        del qq, kk, vv
        return split_attention_chunked(q, k, v, budget, stats)


def _note_attn(stats, n, rows, batch, tokens, elem):
    if stats is None:
        return
    stats.attn_calls += 1
    stats.attn_tokens_max = max(stats.attn_tokens_max, tokens)
    r = min(rows, n)
    stats.attn_rows_min = r if stats.attn_rows_min is None else min(stats.attn_rows_min, r)
    stats.attn_score_max = max(stats.attn_score_max, 2 * batch * r * tokens * elem)


def known_attention():
    """ComfyUI's VAE attention functions -> chunked replacement."""
    return {
        ldm_model.normal_attention: split_attention_chunked,
        ldm_model.pytorch_attention: pytorch_attention_chunked,
        ldm_model.xformers_attention: xformers_attention_chunked,
    }


class _AttnChunker:
    __slots__ = ("impl", "budget", "stats")

    def __init__(self, impl, budget, stats):
        self.impl = impl
        self.budget = budget
        self.stats = stats

    def __call__(self, q, k, v):
        return self.impl(q, k, v, self.budget, self.stats)


# ---------------------------------------------------------------------------
# the context
# ---------------------------------------------------------------------------

_MISSING = object()


class OpChunking:
    """Install the per-instance conv / attention replacements on `model` for
    the duration of the `with` block; everything is restored in __exit__,
    also when the block raises (OOM included)."""

    def __init__(self, model, budget, stats=None):
        self.model = model
        self.budget = int(budget)
        self.stats = stats if stats is not None else OpStats()
        self._saved = []

    def __enter__(self):
        known = known_attention()
        try:
            for name, m in self.model.named_modules():
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)):
                    prev = m.__dict__.get("_conv_forward", _MISSING)
                    orig = m._conv_forward  # bound method (or a previous instance override)
                    self._saved.append((m, "_conv_forward", prev))
                    as2d = isinstance(m, torch.nn.Conv3d) and prev is _MISSING and type(m)._conv_forward in _CONV3D_FORWARDS
                    m.__dict__["_conv_forward"] = _ConvChunker(m, orig, self.budget, self.stats, as2d)
                    self.stats.conv_modules += 1
                fn = m.__dict__.get("optimized_attention", None)
                if fn is not None:
                    impl = known.get(fn)
                    if impl is None:
                        self.stats.attn_unmanaged.append("{} ({})".format(name, getattr(fn, "__name__", type(fn).__name__)))
                        continue
                    self._saved.append((m, "optimized_attention", fn))
                    m.__dict__["optimized_attention"] = _AttnChunker(impl, self.budget, self.stats)
                    self.stats.attn_modules += 1
        except BaseException:
            self._restore()
            raise
        return self.stats

    def _restore(self):
        while self._saved:
            m, attr, prev = self._saved.pop()
            if prev is _MISSING:
                m.__dict__.pop(attr, None)
            else:
                m.__dict__[attr] = prev

    def __exit__(self, *exc):
        self._restore()
        return False


def fmt_bytes(n):
    if n is None:
        return "n/a"
    if n >= GIB:
        return "{:.2f} GiB".format(n / GIB)
    return "{:.0f} MiB".format(n / MIB) if n >= MIB else "{:.0f} KiB".format(n / 1024)
