"""Op-level exact chunking for the LTX-2.5 diffusion video VAE decoder
(comfy.ldm.lightricks.vae.na_diffusion_decoder, ComfyUI v0.46.3; sd.py calls it
"LTX 2.4", the commit that added it "LTX 2.5").

The decoder (NADiffusionDecoder) is a transformer on channels-last tokens
[B, T, H, W, C]: stages 1-4 are NA blocks (neighbourhood attention + SwiGLU)
followed by linear pixel-shuffle upsamples, then stage 5 runs eight
DiffusionNABlocks (kernel 11^3, 256 channels, at pixel/4 resolution) on the
patchified pixel noise x_t guided by the stage-4 output ("context"), one x0
step. Native peak of stage 5 is about six times S, the stage-5 activation
(S = T x H/4 x W/4 x 256 x elem): whole q, k, v and the attention output next
to x and the context. It grows linearly with frames and resolution.

What this module does, without changing results (up to GEMM row grouping):

* NeighborhoodAttention3D.forward (per instance) runs in T slabs. A block of
  query frames [a, b) uses the slab [a - (k_t - 1), b + (k_t - 1)) clipped to
  [0, T): with NATTEN's window (exactly k_t frames, centred, shifted inward at
  the grid borders) every query's window in the slab equals its window in the
  whole volume when the halo is k_t - 1 (the window can sit up to k_t - 1
  frames away from its query at a border; the slab then reaches that border
  too). H and W are not split, so RoPE uses the global tables sliced in T.
  The residual add writes back into x itself (add_to is x), so the result of a
  block is kept pending until no later slab reads those frames of the old x.
* SwiGLU.forward runs over token chunks bounded by the workspace (native
  chunks whole frames: 65536 tokens or one frame, whichever is more).
* NADiffusionDecoder.forward (per instance): the same steps as native, but
  the last upsample computes only the frames native keeps (native drops the
  leading duplicate frame and the trailing padding with views that keep the
  whole tensor alive), x = conv_in_x_t(patchify(x_t)) and
  conv_out(norm_out(x)) run in T chunks writing straight into their outputs,
  and the context is released as soon as the last step's blocks are done.
  x_t is drawn whole from the same generator, so the noise is the same.

Peak of stage 5 then is x + context + one slab's q/k/v/output + the pending
results + the pixel tensors, about 2.2 x S plus the workspace.

The conv VAE (causal_video_autoencoder.VideoVAE) is not handled here.

Entry points: kind(fsm); chunker(model, workspace, stats), registered in
vae_ops.EXTRA_CHUNKERS on import (OpChunking enters it: a context manager for an
LTX-2.5 diffusion VAE, None for any other model); chunking(model, workspace,
stats), the context manager itself (usable on its own, as tests/bench_ltx.py
does); plan(fsm, latent_shape, elem, workspace) (block sizes and the peak
estimate). Every replacement is an instance attribute marked
monoload_replacement = True, restored by the context manager.
"""

import math

import torch
from einops import rearrange

from . import vae_ops

try:
    import comfy_kitchen
    from comfy.ldm.lightricks.vae import na_diffusion_decoder as nd
except Exception:  # pragma: no cover - ComfyUI without the LTX-2.5 decoder
    comfy_kitchen = None
    nd = None

_MISSING = object()

# Grid limits of comfy_kitchen's HIP na3d kernel (T * H and B * heads); beyond them it falls back to another backend.
HIP_GRID_MAX = 65535
QKV_CHUNK_ELEMS = 2 ** 25   # native NeighborhoodAttention3D's qkv/proj chunk (elements of [tokens, dim])
SWIGLU_TERMS = 3            # [tokens, hidden] temporaries alive at once in SwiGLU: gate, silu(gate) * up, up
SLACK = 64 * 1024 * 1024    # small per-op temporaries not modelled (RoPE tables, norm weights, rounding)


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def kind(fsm):
    """"diffusion" for the LTX-2.5 diffusion VAE this module chunks, else None (by class and the attributes used)."""
    if nd is None or comfy_kitchen is None:
        return None
    if type(fsm) is not nd.CausalDiffusionVAE:
        return None
    dec = getattr(fsm, "decoder", None)
    if type(dec) is not nd.NADiffusionDecoder:
        return None
    need = ("conv_in", "det_stages", "upsamples", "t_embedder", "conv_in_x_t", "shared_adaln", "diff_blocks",
            "norm_out", "conv_out", "patch_size", "out_channels", "trailing_pad_latent_frames", "temporal_upscale",
            "default_inference_timesteps", "timestep_scale_multiplier", "model_output_type")
    if any(not hasattr(dec, a) for a in need):
        return None
    if len(dec.upsamples) != len(dec.det_stages) or dec.model_output_type not in ("x0", "v"):
        return None
    for m in dec.modules():
        if type(m) is nd.NeighborhoodAttention3D and len(m.kernel_size) != 3:
            return None
    return "diffusion"


def na_modules(dec):
    return [m for m in dec.modules() if type(m) is nd.NeighborhoodAttention3D]


# ---------------------------------------------------------------------------
# neighbourhood attention in T slabs
# ---------------------------------------------------------------------------

def t_blocks(t, kt, block):
    """[(a, b, s, e)]: query frames [a, b) and the slab [s, e) they need; one block (0, t, 0, t) if no split."""
    halo = kt - 1
    if block >= t or kt >= t:
        return [(0, t, 0, t)]
    out = []
    for a in range(0, t, block):
        b = min(a + block, t)
        out.append((a, b, max(0, a - halo), min(t, b + halo)))
    return out


def na_frame_bytes(h, w, dim, elem):
    return h * w * dim * elem


def na_slab_bytes(block, kt, t, h, w, dim, elem):
    """Workspace of one NA call at query block `block`: q, k, v and the attention output over the slab, plus up to
    two pending block results (only when the residual writes back into x; counted always)."""
    n = min(t, block + 2 * (kt - 1)) if (block < t and kt < t) else t
    f = na_frame_bytes(h, w, dim, elem)
    return 4 * n * f + 2 * min(block, t) * f + qkv_temp_bytes(h, w, dim, elem)


def qkv_temp_bytes(h, w, dim, elem):
    """Temporaries of one qkv chunk (native chunk size): pre(x) slice, the [tokens, 3 dim] qkv output, the fp32
    RoPE matrices (tokens x head_dim/2 x 4 fp32: 8 bytes per channel)."""
    hw = h * w
    frames = max(1, QKV_CHUNK_ELEMS // max(hw * dim, 1))
    tokens = frames * hw
    return tokens * dim * elem * 4 + tokens * dim * 8


def na_block_frames(t, kt, h, w, dim, elem, workspace):
    """Largest query block (frames) whose slab workspace fits `workspace` and whose slab stays inside the HIP
    kernel's grid; 1 if none fits (the estimate then says what it costs). t when no split is needed."""
    if kt >= t:
        return t
    if na_slab_bytes(t, kt, t, h, w, dim, elem) <= workspace and t * h <= HIP_GRID_MAX:
        return t
    best = 1
    lo, hi = 1, t - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        n = min(t, mid + 2 * (kt - 1))
        if na_slab_bytes(mid, kt, t, h, w, dim, elem) <= workspace and n * h <= HIP_GRID_MAX:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


class _NASlabs:
    """Replacement for one NeighborhoodAttention3D.forward: the same computation in T slabs (module docstring)."""

    monoload_replacement = True

    def __init__(self, attn, orig, workspace, stats):
        self.attn, self.orig, self.workspace, self.stats = attn, orig, workspace, stats

    def __call__(self, x, pre=None, add_to=None):
        attn = self.attn
        batch, t, h, w, dim = x.shape
        kt = attn.kernel_size[0]
        block = na_block_frames(t, kt, h, w, dim, x.element_size(), self.workspace)
        _note(self.stats, "ltx_na_calls")
        if block >= t:
            return self.orig(x, pre=pre, add_to=add_to)
        _note(self.stats, "ltx_na_split")
        blocks = t_blocks(t, kt, block)
        _note(self.stats, "ltx_na_slabs", len(blocks))
        _note_min(self.stats, "ltx_na_block_min", block)

        inv_freqs = tuple(nd.rope_inv_freqs(d, attn.rope_base, device=x.device) for d in attn.rope_split)
        tables = nd._rope_tables((t, h, w), inv_freqs, x.device)
        q_weight = (attn.q_norm.weight.detach() * attn.scale).to(x.dtype)
        k_weight = attn.k_norm.weight.detach().to(x.dtype)
        sub = max(1, QKV_CHUNK_ELEMS // max(h * w * dim, 1))
        nh, hd = attn.num_heads, attn.head_dim
        res = add_to if add_to is not None else torch.empty_like(x)
        # the residual writes into x itself: a block's result may only land once no later slab reads those frames
        hazard = add_to is not None and add_to.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        pending = []

        for i, (a, b, s, e) in enumerate(blocks):
            n = e - s
            shape = (batch, n, h, w, nh, hd)
            q = torch.empty(shape, dtype=x.dtype, device=x.device)
            k = torch.empty(shape, dtype=x.dtype, device=x.device)
            v = torch.empty(shape, dtype=x.dtype, device=x.device)
            for t0 in range(s, e, sub):
                t1 = min(t0 + sub, e)
                sl = x[:, t0:t1] if pre is None else pre(x[:, t0:t1])
                qc, kc, vc = attn.qkv(sl).chunk(3, dim=-1)
                del sl
                cshape = (batch, t1 - t0, h, w, nh, hd)
                q[:, t0 - s:t1 - s] = qc.reshape(cshape)
                k[:, t0 - s:t1 - s] = kc.reshape(cshape)
                v[:, t0 - s:t1 - s] = vc.reshape(cshape)
                del qc, kc, vc
                freqs = nd._rope_matrices_slice(tables, t0, t1, h, w)
                nt = (t1 - t0) * h * w
                for bi in range(batch):
                    comfy_kitchen.rms_rope_(
                        q[bi, t0 - s:t1 - s].view(1, nt, nh, hd),
                        k[bi, t0 - s:t1 - s].view(1, nt, nh, hd),
                        freqs, q_weight, k_weight)
                del freqs
            # this slab has read x[s:e]: earlier results no later slab reads can land now
            nxt = blocks[i + 1][2] if i + 1 < len(blocks) else t
            pending = _flush(res, pending, nxt, add_to is not None)
            out = comfy_kitchen.na3d(q, k, v, list(attn.kernel_size), None, 1.0)
            del q, k, v
            out = out.reshape(batch, n, h, w, dim)
            if hazard:
                y = torch.empty((batch, b - a, h, w, dim), dtype=x.dtype, device=x.device)
                for t0 in range(a, b, sub):
                    t1 = min(t0 + sub, b)
                    y[:, t0 - a:t1 - a] = attn.proj(out[:, t0 - s:t1 - s])
                del out
                pending.append((a, b, y))
                pending = _flush(res, pending, nxt, True)
            else:
                for t0 in range(a, b, sub):
                    t1 = min(t0 + sub, b)
                    if add_to is not None:
                        res[:, t0:t1] += attn.proj(out[:, t0 - s:t1 - s])
                    else:
                        res[:, t0:t1] = attn.proj(out[:, t0 - s:t1 - s])
                del out
        _flush(res, pending, t, add_to is not None)
        return res


def _flush(res, pending, first_read, add):
    """Write the pending results whose frames end at or before `first_read` (no later slab reads them)."""
    keep = []
    for (a, b, y) in pending:
        if b <= first_read:
            if add:
                res[:, a:b] += y
            else:
                res[:, a:b] = y
        else:
            keep.append((a, b, y))
    return keep


# ---------------------------------------------------------------------------
# SwiGLU in bounded token chunks
# ---------------------------------------------------------------------------

def swiglu_token_bytes(dim, hidden, elem):
    """Temporaries per token of one SwiGLU chunk: pre(x), SWIGLU_TERMS [hidden] tensors, the [dim] output."""
    return (2 * dim + SWIGLU_TERMS * hidden) * elem


def swiglu_chunk_tokens(t, h, w, dim, hidden, elem, workspace):
    native = max(1, nd.MLP_TOKEN_CHUNK // max(h * w, 1)) * h * w
    fit = max(1, int(workspace // swiglu_token_bytes(dim, hidden, elem)))
    return min(native, fit, t * h * w)


class _SwiGLUChunks:
    monoload_replacement = True

    def __init__(self, mlp, orig, workspace, stats):
        self.mlp, self.orig, self.workspace, self.stats = mlp, orig, workspace, stats

    def __call__(self, x, pre=None, add_to=None):
        mlp = self.mlp
        batch, t, h, w, dim = x.shape
        hidden = mlp.w_up.weight.shape[0]
        native = max(1, nd.MLP_TOKEN_CHUNK // max(h * w, 1)) * h * w
        chunk = swiglu_chunk_tokens(t, h, w, dim, hidden, x.element_size(), self.workspace)
        if chunk >= native or not x.is_contiguous() or (add_to is not None and not add_to.is_contiguous()):
            return self.orig(x, pre=pre, add_to=add_to)
        _note(self.stats, "ltx_mlp_split")
        out = add_to if add_to is not None else torch.empty_like(x)
        # tokens flattened into the W axis, so `pre` (norm + modulate with [1, 1, 1, 1, C] tables) broadcasts as native
        n = t * h * w
        xf = x.view(batch, 1, 1, n, dim)
        of = out.view(batch, 1, 1, n, dim)
        for r0 in range(0, n, chunk):
            r1 = min(r0 + chunk, n)
            sl = xf[:, :, :, r0:r1] if pre is None else pre(xf[:, :, :, r0:r1])
            y = mlp.w_down(torch.nn.functional.silu(mlp.w_gate(sl)) * mlp.w_up(sl))
            del sl
            if add_to is not None:
                of[:, :, :, r0:r1] += y
            else:
                of[:, :, :, r0:r1] = y
            del y
        return out


# ---------------------------------------------------------------------------
# the decoder forward
# ---------------------------------------------------------------------------

def _upsample_kept(up, x, drop_leading, crop_tail):
    """up(x) with native's frame drops applied (leading duplicate frame, then crop_tail trailing frames), computing
    and storing only the kept frames; same chunks as LinearPixelShuffleUpsample.forward."""
    batch, t, h, w, _ = x.shape
    p1, p2, p3 = up.stride
    total = t * p1
    lo = 1 if (p1 == 2 and drop_leading) else 0
    hi = total - crop_tail
    out = None
    chunk = max(1, nd.MLP_TOKEN_CHUNK // max(h * w, 1))
    for t0 in range(0, t, chunk):
        t1 = min(t0 + chunk, t)
        g0, g1 = max(t0 * p1, lo), min(t1 * p1, hi)
        if g0 >= g1:
            continue
        y = rearrange(up.proj(x[:, t0:t1]), "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c", p1=p1, p2=p2, p3=p3)
        if out is None:
            out = torch.empty((batch, hi - lo, h * p2, w * p3, up.out_channels), dtype=y.dtype, device=y.device)
        out[:, g0 - lo:g1 - lo] = y[:, g0 - t0 * p1:g1 - t0 * p1]
        del y
    return out


def _pre_diffusion(dec, z, drop_leading_frame, pad_trailing):
    n = dec.trailing_pad_latent_frames if pad_trailing else 0
    if n > 0:
        z = torch.cat([z, z[:, :, -1:].expand(-1, -1, n, -1, -1)], dim=2)
    x = z.permute(0, 2, 3, 4, 1)
    x = dec.conv_in(x)
    last = len(dec.det_stages) - 1
    for stage_i, blocks in enumerate(dec.det_stages):
        for block in blocks:
            x = block(x)
        if stage_i < last:
            x = dec.upsamples[stage_i](x, drop_leading_frame=drop_leading_frame)
        else:
            x = _upsample_kept(dec.upsamples[stage_i], x, drop_leading_frame, n * dec.temporal_upscale)
    return x


def _diff_step(dec, holder, x_t, t, release):
    """forward_diff_step with T-chunked input/output projections; holder[0] is the context, dropped after the blocks
    when `release`."""
    ps = dec.patch_size
    batch, _, t5, hh, ww = x_t.shape
    h5, w5 = hh // ps, ww // ps
    chunk = max(1, nd.MLP_TOKEN_CHUNK // max(h5 * w5, 1))
    x = None
    for t0 in range(0, t5, chunk):
        t1 = min(t0 + chunk, t5)
        xc = dec.conv_in_x_t(nd.patchify(x_t[:, :, t0:t1], patch_size_hw=ps, patch_size_t=1).permute(0, 2, 3, 4, 1))
        if x is None:
            x = torch.empty((batch, t5) + tuple(xc.shape[2:]), dtype=xc.dtype, device=xc.device)
        x[:, t0:t1] = xc
        del xc
    t_emb = dec.t_embedder(dec.timestep_scale_multiplier * t, dtype=x.dtype)
    modulation = dec.shared_adaln(t_emb)
    for block in dec.diff_blocks:
        x = block(x, holder[0], modulation)
    if release:
        holder[0] = None
    out = None
    for t0 in range(0, t5, chunk):
        t1 = min(t0 + chunk, t5)
        y = dec.conv_out(dec.norm_out(x[:, t0:t1])).permute(0, 4, 1, 2, 3)
        y = nd.unpatchify(y, patch_size_hw=ps, patch_size_t=1)
        if out is None:
            out = torch.empty((batch, y.shape[1], t5) + tuple(y.shape[3:]), dtype=y.dtype, device=y.device)
        out[:, :, t0:t1] = y
        del y
    return out


class _DecoderForward:
    """Replacement for NADiffusionDecoder.forward: native's steps, leaner (module docstring)."""

    monoload_replacement = True

    def __init__(self, dec, stats):
        self.dec, self.stats = dec, stats

    def __call__(self, z, generator=None, drop_leading_frame=True, pad_trailing=True):
        dec = self.dec
        _note(self.stats, "ltx_decodes")
        holder = [_pre_diffusion(dec, z, drop_leading_frame, pad_trailing)]
        batch, t5, h5, w5, _ = holder[0].shape
        pixel_shape = (batch, dec.out_channels, t5, h5 * dec.patch_size, w5 * dec.patch_size)
        x_t = torch.randn(pixel_shape, dtype=z.dtype, device=z.device, generator=generator)
        timesteps = dec.default_inference_timesteps.to(z.device)
        num_steps = timesteps.shape[0]
        for i in range(num_steps):
            t_now = timesteps[i].expand(batch)
            last = i == num_steps - 1
            model_out = _diff_step(dec, holder, x_t, t_now, last)
            if dec.model_output_type == "x0":
                x0 = model_out
                if last:
                    return x0
                velocity = (x_t.float() - x0.float()) / timesteps[i]
            else:
                velocity = model_out.float()
                if last:
                    return (x_t.float() - timesteps[i] * velocity).to(z.dtype)
            del model_out
            t_next = timesteps[i + 1] if i + 1 < num_steps else torch.zeros_like(timesteps[i])
            x_t = (x_t.float() - (timesteps[i] - t_next) * velocity).to(z.dtype)
        return x_t


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def _note(stats, key, n=1):
    if stats is not None:
        setattr(stats, key, getattr(stats, key, 0) + n)


def _note_min(stats, key, v):
    if stats is not None:
        cur = getattr(stats, key, None)
        setattr(stats, key, v if cur is None else min(cur, v))


class chunking:
    """Install the replacements on an LTX-2.5 diffusion VAE (any module whose kind() is "diffusion", searched from
    `model`) for the `with` block; restored in __exit__, also on errors. A no-op for other models. Signature of
    OpChunking's extension point: (model, workspace, stats)."""

    def __init__(self, model, workspace, stats=None):
        self.model, self.workspace, self.stats = model, int(workspace), stats
        self._saved = []

    def __enter__(self):
        try:
            for fsm in self.model.modules():
                if kind(fsm) != "diffusion":
                    continue
                dec = fsm.decoder
                self._set(dec, "forward", _DecoderForward(dec, self.stats))
                for m in dec.modules():
                    if type(m) is nd.NeighborhoodAttention3D:
                        self._set(m, "forward", _NASlabs(m, m.forward, self.workspace, self.stats))
                        _note(self.stats, "ltx_na_modules")
                    elif type(m) is nd.SwiGLU:
                        self._set(m, "forward", _SwiGLUChunks(m, m.forward, self.workspace, self.stats))
        except BaseException:
            self._restore()
            raise
        return self.stats

    def _set(self, m, attr, fn):
        self._saved.append((m, attr, m.__dict__.get(attr, _MISSING)))
        m.__dict__[attr] = fn

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


# ---------------------------------------------------------------------------
# shapes and the peak estimate
# ---------------------------------------------------------------------------

def plan(fsm, latent_shape, elem, workspace):
    """Shapes, block sizes and an upper bound (bytes) of what one chunked decode of a latent of `latent_shape`
    ([B, 128, T, h, w]) adds on the device, weights excluded. elem: bytes per element of the VAE dtype.

    Phases (live tensors at each phase's worst moment):
      stage i blocks:  x_i + max(NA slab workspace, SwiGLU chunk)
      upsample i:      x_i + up_i output + one proj chunk (+ the kept-frame output only, for the last)
      stage 5 input:   context + x_t + x + one patchify/proj chunk
      stage 5 blocks:  context + x_t + x + max(context_proj chunk, NA slab workspace, SwiGLU chunk)
      stage 5 output:  x_t + x + pixel output + one norm/proj chunk
    plus the latent and SLACK per phase."""
    dec = fsm.decoder
    b, _, lt, lh, lw = latent_shape
    pad = dec.trailing_pad_latent_frames
    t, h, w = lt + pad, lh, lw
    t_store = t   # frames of the storage behind x (an upsample's leading-frame drop is a view)
    lat = math.prod(latent_shape) * elem
    phases = []
    stages = []
    for i, blocks in enumerate(dec.det_stages):
        blk = blocks[0]
        dim = blk.attn.dim
        kt = blk.attn.kernel_size[0]
        hidden = blk.mlp.w_up.weight.shape[0]
        x_bytes = b * t_store * h * w * dim * elem
        bf = na_block_frames(t, kt, h, w, dim, elem, workspace)
        na_ws = b * na_slab_bytes(bf, kt, t, h, w, dim, elem)
        mlp_ws = b * swiglu_chunk_tokens(t, h, w, dim, hidden, elem, workspace) * swiglu_token_bytes(dim, hidden, elem)
        phases.append(("stage{} blocks".format(i + 1), x_bytes + max(na_ws, mlp_ws)))
        stages.append({"stage": i + 1, "t": t, "h": h, "w": w, "dim": dim, "kernel_t": kt, "na_block": bf})
        up = dec.upsamples[i]
        p1, p2, p3 = up.stride
        t2 = t * p1
        lo = 1 if p1 == 2 else 0
        last = i == len(dec.det_stages) - 1
        hi = t2 - (pad * dec.temporal_upscale if last else 0)
        kept = (hi - lo) if last else t2
        up_out = b * kept * h * p2 * w * p3 * up.out_channels * elem
        ch = max(1, nd.MLP_TOKEN_CHUNK // max(h * w, 1))
        proj_chunk = b * min(ch, t) * h * w * up.proj.weight.shape[0] * elem * 2   # proj output + rearranged copy
        phases.append(("upsample{}".format(i + 1), x_bytes + up_out + proj_chunk))
        t = (t2 - lo) if not last else kept
        t_store = t2 if not last else kept
        h, w = h * p2, w * p3
    c5 = dec.context_channels
    t5, h5, w5 = t, h, w
    ps = dec.patch_size
    ctx = b * t5 * h5 * w5 * c5 * elem
    x5 = b * t5 * h5 * w5 * dec.conv_in_x_t.weight.shape[0] * elem
    pix = b * dec.out_channels * t5 * h5 * ps * w5 * ps * elem
    ch5 = max(1, nd.MLP_TOKEN_CHUNK // max(h5 * w5, 1))
    tok_chunk = b * min(ch5, t5) * h5 * w5
    in_chunk = tok_chunk * (dec.conv_in_x_t.weight.shape[1] * 2 + c5) * elem
    blk = dec.diff_blocks[0]
    kt5 = blk.attn.kernel_size[0]
    hidden5 = blk.mlp.w_up.weight.shape[0]
    bf5 = na_block_frames(t5, kt5, h5, w5, c5, elem, workspace)
    na5 = b * na_slab_bytes(bf5, kt5, t5, h5, w5, c5, elem)
    mlp5 = b * swiglu_chunk_tokens(t5, h5, w5, c5, hidden5, elem, workspace) * swiglu_token_bytes(c5, hidden5, elem)
    ctxp = tok_chunk * c5 * elem
    out_chunk = tok_chunk * (c5 + dec.conv_out.weight.shape[0] * 2) * elem
    phases.append(("stage5 input", ctx + pix + x5 + in_chunk))
    phases.append(("stage5 blocks", ctx + pix + x5 + max(ctxp, na5, mlp5)))
    phases.append(("stage5 output", pix + x5 + pix + out_chunk))
    stages.append({"stage": len(dec.det_stages) + 1, "t": t5, "h": h5, "w": w5, "dim": c5, "kernel_t": kt5, "na_block": bf5})
    worst = max(phases, key=lambda p: p[1])
    return {"total": int(worst[1] + lat + SLACK), "phase": worst[0], "phases": phases, "stages": stages,
            "context": int(ctx), "x5": int(x5), "pixels": int(pix), "frames": t5}


def native_estimate(fsm, latent_shape, elem):
    """Rough native stage-5 peak for comparison: context (with the 16 padding frames native keeps) + x_t + x +
    q, k, v and the attention output + the norm/proj temporaries (about six S)."""
    p = plan(fsm, latent_shape, elem, 1 << 62)
    dec = fsm.decoder
    pad_frames = dec.trailing_pad_latent_frames * dec.temporal_upscale
    s = p["x5"]
    ctx = p["context"] * (p["frames"] + pad_frames + 1) / max(p["frames"], 1)
    return int(ctx + p["pixels"] + s * 5)


def chunker(model, workspace, stats):
    """vae_ops.EXTRA_CHUNKERS entry: chunking() for a model holding an LTX-2.5 diffusion VAE, else None."""
    try:
        mods = model.modules()
    except AttributeError:
        return None
    if not any(kind(m) == "diffusion" for m in mods):
        return None
    return chunking(model, workspace, stats)


if nd is not None and chunker not in vae_ops.EXTRA_CHUNKERS:
    vae_ops.EXTRA_CHUNKERS.append(chunker)
