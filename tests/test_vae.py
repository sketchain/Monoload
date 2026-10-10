"""Managed VAE decode (monoload/vae.py) and op-level chunking (monoload/vae_ops.py).

No model files needed: the decoders are ComfyUI's own classes with small
channel counts and random weights.

  1. chunked conv == unchunked conv over a matrix of Conv2d / Conv3d
     configurations (kernel, stride, dilation, groups, padding incl. 'same'
     and reflect/replicate/circular, comfy.ops cast path with weight_function,
     Conv3d autopad="causal_zero", Wan CausalConv3d), at a workspace small
     enough for one-row blocks and at a medium one;
  2. chunked attention == ComfyUI's split / pytorch VAE attention;
  3. comfy.sd.VAE built from synthetic state dicts -- SDXL-like (AutoencoderKL,
     post_quant_conv, 4 latent channels), Flux-like (AutoencodingEngine, 16
     channels), Wan 2.1 / qwen_image_vae-like (WanVAE, 5D T=1) -- managed decode
     vs native VAE.decode: raw decoder output and process_output pixels,
     shape / dtype / device / NHWC, batch of 2, 5D T=1, own memory estimate
     passed to load_models_gpu, RNG untouched;
  4. what stays native (multi-frame video, ...), OOM retry with smaller
     blocks, MonoloadVAEOOMError at the floor, decode_tiled_ never called,
     every per-instance override removed afterwards;
  5. coverage fixes: SVD's VideoDecoder (frames mixed across the batch), audio
     VAEs with a 2D latent and the pixel-space VAE stay native; the result is
     native's and load_models_gpu gets ComfyUI's own estimate.

    python tests/test_vae.py           # COMFY_ARGS default: --cpu --fp16-unet
"""

import math

import torch
import torch.nn.functional as F

from common import check, expect_raises, finish
import comfy.model_management
import comfy.ops
import comfy.sd
from comfy.ldm.modules.diffusionmodules import model as ldm_model
from comfy.ldm.modules.diffusionmodules.model import Decoder
import comfy.ldm.wan.vae as wan_vae
from monoload import vae as mvae
from monoload import vae_ops
from monoload.errors import MonoloadVAEOOMError

torch.manual_seed(0)
OPS = comfy.ops.disable_weight_init
MC = comfy.ops.manual_cast


def rel_err(a, b):
    a = a.double()
    b = b.double()
    return float((a - b).abs().max() / max(1e-12, float(b.abs().max())))


def no_overrides(model):
    return not any(("_conv_forward" in m.__dict__) or isinstance(m.__dict__.get("optimized_attention"), vae_ops._AttnChunker)
                   for m in model.modules())


# ---------------------------------------------------------------------------
# 1. conv
# ---------------------------------------------------------------------------

def init_conv(m, dtype=torch.float32):
    with torch.no_grad():
        w = torch.randn(m.weight.shape) / math.sqrt(m.weight[0].numel())
        m.weight = torch.nn.Parameter(w.to(dtype), requires_grad=False)
        if m.bias is not None:
            m.bias = torch.nn.Parameter((torch.randn(m.bias.shape) * 0.1).to(dtype), requires_grad=False)
    return m


def run_conv_case(name, mod, x, budgets=(1, 64 * 1024), call=None, expect_chunked=True, tol=1e-5, **kw):
    call = call or (lambda: mod(x, **kw))
    ref = call()
    for b in budgets:
        st = vae_ops.OpStats()
        pad_before = mod.padding
        with vae_ops.OpChunking(mod, b, st):
            y = call()
        e = rel_err(y, ref)
        if not expect_chunked:
            chunk_ok = st.conv_chunked == 0
        elif b == 1:
            chunk_ok = st.conv_chunked > 0 and st.conv_blocks > 1  # one-row blocks everywhere
        else:
            chunk_ok = True  # medium workspace: chunked only if the whole conv does not fit
        check("conv {} [workspace {}]: chunked == native (rel max|Δ| {:.2g}, {} block(s), rows {})".format(
            name, vae_ops.fmt_bytes(b) if b > 1 else "1 B", e, st.conv_blocks, st.conv_rows_min),
            y.shape == ref.shape and y.dtype == ref.dtype and e <= tol and chunk_ok and no_overrides(mod) and mod.padding == pad_before,
            "shape {} vs {}, chunked {}".format(tuple(y.shape), tuple(ref.shape), st.conv_chunked))


def conv_matrix():
    x2 = torch.randn(2, 8, 23, 17)
    cases = [
        ("3x3 p1", dict(k=3, p=1)),
        ("3x3 p0", dict(k=3, p=0)),
        ("5x5 p2 dil2", dict(k=5, p=2, d=2)),
        ("3x3 s2 p1", dict(k=3, p=1, s=2)),
        ("4x4 s3 p2", dict(k=4, p=2, s=3)),
        ("3x3 groups2", dict(k=3, p=1, g=2)),
        ("3x3 depthwise", dict(k=3, p=1, g=8, cout=8)),
        ("3x1 p(1,0)", dict(k=(3, 1), p=(1, 0))),
        ("1x3 p(0,1)", dict(k=(1, 3), p=(0, 1))),
        ("4x4 'same'", dict(k=4, p="same")),
        ("3x3 'valid'", dict(k=3, p="valid")),
        ("3x3 p1 reflect", dict(k=3, p=1, mode="reflect")),
        ("3x3 p2 replicate", dict(k=3, p=2, mode="replicate")),
        ("3x3 p1 circular", dict(k=3, p=1, mode="circular")),
        ("1x1 s2", dict(k=1, p=0, s=2)),
        ("2x5 s(2,1) dil(1,2) p(1,3)", dict(k=(2, 5), p=(1, 3), s=(2, 1), d=(1, 2))),
    ]
    for name, c in cases:
        m = init_conv(OPS.Conv2d(8, c.get("cout", 12), c["k"], stride=c.get("s", 1), padding=c["p"], dilation=c.get("d", 1),
                                 groups=c.get("g", 1), padding_mode=c.get("mode", "zeros")))
        run_conv_case(name, m, x2)
    # pointwise (no im2col in Slow2d): left alone
    m = init_conv(OPS.Conv2d(8, 12, 1))
    run_conv_case("1x1 s1 (pointwise, not chunked)", m, x2, expect_chunked=False, tol=0.0)

    # comfy.ops cast path: fp16 weight, fp32 input (manual_cast), plus a weight
    # function (as Monoload's runtime LoRA / native LowVramPatch): it must run
    # once per conv call, not per block
    m = init_conv(MC.Conv2d(8, 12, 3, padding=1), torch.float16)
    delta = torch.randn(12, 8, 3, 3) * 0.01
    calls = [0]

    def wf(w):
        calls[0] += 1
        return w + delta.to(w)
    m.weight_function = [wf]
    ref = m(x2)
    calls[0] = 0
    st = vae_ops.OpStats()
    with vae_ops.OpChunking(m, 1, st):
        y = m(x2)
    check("conv cast path + weight_function: chunked == native (rel {:.2g}), weight_function ran once for {} blocks".format(rel_err(y, ref), st.conv_blocks),
          rel_err(y, ref) <= 1e-5 and calls[0] == 1 and st.conv_blocks > 1 and no_overrides(m))

    # bf16 (looser: different GEMM shapes round differently)
    mb = init_conv(OPS.Conv2d(8, 12, 3, padding=1), torch.bfloat16)
    run_conv_case("3x3 p1 bf16", mb, x2.bfloat16(), tol=2e-2)

    # Conv3d
    x3 = torch.randn(1, 8, 3, 19, 13)
    for name, c in [("3x3x3 p1", dict(k=3, p=1)), ("3x1x1 p(1,0,0)", dict(k=(3, 1, 1), p=(1, 0, 0))),
                    ("3x3x3 s(1,2,2) p1", dict(k=3, p=1, s=(1, 2, 2))), ("3x3x3 p1 replicate", dict(k=3, p=1, mode="replicate"))]:
        m = init_conv(OPS.Conv3d(8, 12, c["k"], stride=c.get("s", 1), padding=c["p"], padding_mode=c.get("mode", "zeros")))
        run_conv_case("3D " + name, m, x3)
    # comfy Conv3d autopad="causal_zero" (Wan's T=1 fast path truncates the time kernel)
    m = init_conv(OPS.Conv3d(8, 12, 3, padding=(0, 1, 1)))
    for t in (1, 2):
        xt = torch.randn(1, 8, t, 19, 13)
        run_conv_case("3D causal_zero T={}".format(t), m, xt, call=lambda xt=xt: m(xt, autopad="causal_zero"))
    # Wan CausalConv3d: T=1 (autopad fast path) and T=3 (zeros concatenated in time)
    cc = init_conv(wan_vae.CausalConv3d(8, 12, 3, padding=1))
    for t in (1, 3):
        xt = torch.randn(1, 8, t, 19, 13)
        run_conv_case("Wan CausalConv3d T={}".format(t), cc, xt, call=lambda xt=xt: cc(xt))


# ---------------------------------------------------------------------------
# 2. attention
# ---------------------------------------------------------------------------

def attention_tests():
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        q, k, v = (torch.randn(1, 32, 7, 9).to(dtype) for _ in range(3))
        n = 63
        elem = q.element_size()
        for rows in (1, 5, n, 1000):
            budget = rows * 2 * n * elem
            for native, chunked, label in ((ldm_model.normal_attention, vae_ops.split_attention_chunked, "split"),
                                           (ldm_model.pytorch_attention, vae_ops.pytorch_attention_chunked, "pytorch")):
                ref = native(q, k, v)
                st = vae_ops.OpStats()
                y = chunked(q, k, v, budget, st)
                e = rel_err(y, ref)
                check("attention {} {} query block {}: == native (rel max|Δ| {:.2g})".format(label, str(dtype)[6:], st.attn_rows_min, e),
                      y.shape == ref.shape and y.dtype == ref.dtype and e <= tol and st.attn_rows_min == min(rows, n))
    # through the module (instance attribute) and restored afterwards
    blk = ldm_model.AttnBlock(32)
    for c in (blk.q, blk.k, blk.v, blk.proj_out):
        init_conv(c)
    blk.norm.weight = torch.nn.Parameter(torch.ones(32), requires_grad=False)
    blk.norm.bias = torch.nn.Parameter(torch.zeros(32), requires_grad=False)
    x = torch.randn(1, 32, 7, 9)
    orig_fn = blk.optimized_attention
    ref = blk(x)
    st = vae_ops.OpStats()
    with vae_ops.OpChunking(blk, 2 * 3 * 63 * 4, st):
        y = blk(x)
    check("AttnBlock under OpChunking: == native (rel {:.2g}), query block {}, attention restored".format(rel_err(y, ref), st.attn_rows_min),
          rel_err(y, ref) <= 1e-5 and st.attn_rows_min == 3 and st.attn_modules == 1 and blk.optimized_attention is orig_fn and no_overrides(blk))
    # unknown implementation: left alone, reported
    blk.optimized_attention = lambda q, k, v: ldm_model.normal_attention(q, k, v)
    st = vae_ops.OpStats()
    with vae_ops.OpChunking(blk, 1, st):
        y = blk(x)
    check("unknown attention implementation: left native and reported", st.attn_modules == 0 and len(st.attn_unmanaged) == 1 and rel_err(y, ref) <= 1e-5)
    # SeedVR2's VAE attention keeps its function in optimized_vae_attention (comfy/ldm/seedvr/vae.py; heads == 1 in the
    # VAE): chunked the same way (review 2026-10 item 12)
    from comfy.ldm.seedvr import vae as seedvr_vae
    torch.manual_seed(4)
    sa = seedvr_vae.Attention(32, heads=1, dim_head=32, norm_num_groups=8, residual_connection=True)
    with torch.no_grad():
        for p_ in sa.parameters():
            p_.copy_(torch.randn(p_.shape) * 0.2)
    orig_fn = sa.optimized_vae_attention
    ref = sa(x)
    st = vae_ops.OpStats()
    with vae_ops.OpChunking(sa, 2 * 3 * 63 * 4, st):
        y = sa(x)
    check("SeedVR2 Attention (optimized_vae_attention) under OpChunking: == native (rel {:.2g}), query block {}, attention restored".format(
          rel_err(y, ref), st.attn_rows_min),
          rel_err(y, ref) <= 1e-5 and st.attn_rows_min == 3 and st.attn_modules == 1 and not st.attn_unmanaged
          and sa.__dict__.get("optimized_vae_attention") is orig_fn)


# ---------------------------------------------------------------------------
# 3. whole decoders through comfy.sd.VAE
# ---------------------------------------------------------------------------

def init_random(model):
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2 and not name.endswith("gamma"):
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p[0].numel()))
            elif name.endswith("gamma") or (p.ndim == 1 and name.endswith("weight")):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
    return model


def ldm_vae(z, post_quant_conv, dtype=None):
    dec = init_random(Decoder(ch=32, out_ch=3, ch_mult=[1, 2, 4, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0,
                              in_channels=3, resolution=256, z_channels=z))
    sd = {"decoder." + k: v for k, v in dec.state_dict().items()}
    if post_quant_conv:
        pq = init_conv(OPS.Conv2d(z, z, 1))
        sd["post_quant_conv.weight"] = pq.weight.data
        sd["post_quant_conv.bias"] = pq.bias.data
    return comfy.sd.VAE(sd=sd, dtype=dtype)


def wan_vae_model(dtype=None):
    m = init_random(wan_vae.WanVAE(dim=16, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                                   temperal_downsample=[False, True, True], image_channels=3, conv_out_channels=3, dropout=0.0))
    return comfy.sd.VAE(sd=m.state_dict(), dtype=dtype)


class Spy:
    """Records load_models_gpu(memory_required=...) and decode_tiled_ calls."""

    def __enter__(self):
        self.loads = []
        self.tiled = 0
        self._lm = comfy.model_management.load_models_gpu
        self._dt = comfy.sd.VAE.decode_tiled_
        spy = self

        def lm(models, memory_required=0, **kw):
            spy.loads.append(memory_required)
            return spy._lm(models, memory_required=memory_required, **kw)

        def dt(*a, **kw):
            spy.tiled += 1
            return spy._dt(*a, **kw)
        comfy.model_management.load_models_gpu = lm
        comfy.sd.VAE.decode_tiled_ = dt
        return self

    def __exit__(self, *exc):
        comfy.model_management.load_models_gpu = self._lm
        comfy.sd.VAE.decode_tiled_ = self._dt
        return False


def native_decode(v, latent, raw=False):
    po = v.process_output
    if raw:
        v.process_output = lambda image: image
    try:
        return mvae._ORIG["decode"](v, latent)
    finally:
        v.process_output = po


def managed_decode(v, latent, raw=False):
    po = v.process_output
    if raw:
        v.process_output = lambda image: image
    try:
        return comfy.sd.VAE.decode(v, latent)
    finally:
        v.process_output = po


def decoder_case(label, v, latent, budget, tol):
    mvae.set_workspace(budget)
    rng = torch.get_rng_state()
    with Spy() as spy:
        out = managed_decode(v, latent)
    last = mvae.last_decode()
    ref = native_decode(v, latent)
    raw = managed_decode(v, latent, raw=True)
    raw_ref = native_decode(v, latent, raw=True)
    st = last.get("stats", {})
    e_raw = (raw.float() - raw_ref.float()).abs().max().item()
    e_px = (out.float() - ref.float()).abs().max().item()
    check("{} [workspace {}]: managed == native, raw max|Δ| {:.2g}, pixels max|Δ| {:.2g}; {} of {} conv calls chunked ({} blocks), "
          "attention query block {} of {}".format(label, vae_ops.fmt_bytes(budget), e_raw, e_px, st.get("conv_chunked"), st.get("conv_calls"),
                                                 st.get("conv_blocks"), st.get("attn_rows_min"), st.get("attn_tokens_max")),
          e_raw <= tol and e_px <= tol and last.get("strategy") == "layer2" and st.get("conv_chunked", 0) > 0)
    check("{}: output shape {} / dtype {} / device {} as native, in [0, 1]".format(label, tuple(out.shape), out.dtype, out.device),
          out.shape == ref.shape and out.dtype == ref.dtype and out.device == ref.device and float(out.min()) >= 0 and float(out.max()) <= 1)
    check("{}: load_models_gpu got Monoload's estimate {} (native estimate {}), decode_tiled_ not called, RNG untouched".format(
        label, vae_ops.fmt_bytes(last["estimate"]["total"]), vae_ops.fmt_bytes(last.get("native_estimate"))),
        spy.loads and spy.loads[-1] == last["estimate"]["total"] and spy.tiled == 0 and torch.equal(rng, torch.get_rng_state()))
    check("{}: no per-instance override left on the model".format(label), no_overrides(v.first_stage_model))
    return out, ref


def decoder_tests():
    g = torch.Generator().manual_seed(7)
    sdxl = ldm_vae(4, True)
    flux = ldm_vae(16, False)
    wan = wan_vae_model()
    check("synthetic VAEs detected as AutoencoderKL / AutoencodingEngine / WanVAE",
          type(sdxl.first_stage_model).__name__ == "AutoencoderKL" and type(flux.first_stage_model).__name__ == "AutoencodingEngine"
          and type(wan.first_stage_model).__name__ == "WanVAE", "{} {} {}".format(type(sdxl.first_stage_model).__name__,
                                                                                type(flux.first_stage_model).__name__, type(wan.first_stage_model).__name__))
    l4 = torch.randn(1, 4, 12, 10, generator=g)
    l16 = torch.randn(1, 16, 12, 10, generator=g)
    l5 = torch.randn(1, 16, 1, 12, 10, generator=g)
    for budget in (16 * 1024, 256 * 1024):
        decoder_case("SDXL-like (KL, z4)", sdxl, l4, budget, 1e-4)
        decoder_case("Flux-like (z16)", flux, l16, budget, 1e-4)
        decoder_case("Wan/qwen-like (5D T=1)", wan, l5, budget, 1e-4)
    # batch of two: sequential, same result as native's batch
    decoder_case("SDXL-like batch 2", sdxl, torch.randn(2, 4, 12, 10, generator=g), 16 * 1024, 1e-4)
    decoder_case("Wan/qwen-like batch 2", wan, torch.randn(2, 16, 1, 12, 10, generator=g), 16 * 1024, 1e-4)
    # 5D latent into a 2D VAE: native takes frame 0, so does the managed path
    decoder_case("SDXL-like 5D latent (frame 0)", sdxl, torch.randn(1, 4, 3, 12, 10, generator=g), 16 * 1024, 1e-4)
    # odd sizes
    decoder_case("Flux-like 7x13 latent", flux, torch.randn(1, 16, 7, 13, generator=g), 16 * 1024, 1e-4)
    # bf16 VAE (as --bf16-vae): reported, loose bound
    sdxl_bf = ldm_vae(4, True, dtype=torch.bfloat16)
    decoder_case("SDXL-like bf16", sdxl_bf, l4, 16 * 1024, 0.05)
    wan_bf = wan_vae_model(dtype=torch.bfloat16)
    decoder_case("Wan/qwen-like bf16", wan_bf, l5, 16 * 1024, 0.05)
    # workspace large enough for everything: nothing chunked, identical to native
    mvae.set_workspace(1 << 40)
    out = managed_decode(flux, l16)
    ref = native_decode(flux, l16)
    st = mvae.last_decode()["stats"]
    check("huge workspace: nothing chunked, bit-identical to native", st["conv_chunked"] == 0 and torch.equal(out, ref))
    return sdxl, wan


# ---------------------------------------------------------------------------
# 4. native fallbacks, OOM
# ---------------------------------------------------------------------------

def fallback_and_oom_tests(sdxl, wan):
    g = torch.Generator().manual_seed(3)
    mvae.set_workspace(16 * 1024)
    # multi-frame video: native
    lat = torch.randn(1, 16, 3, 6, 5, generator=g)
    with Spy() as spy:
        out = comfy.sd.VAE.decode(wan, lat)
    ref = native_decode(wan, lat)
    last = mvae.last_decode()
    check("multi-frame video latent (T=3): left native (logged), result == native", last.get("strategy") == "native"
          and "multi-frame" in last.get("reason", "") and torch.equal(out, ref) and spy.tiled == 0)
    check("_native_reason: comfy_has_chunked_io / 1D latent / 4D latent for a 3D VAE stay native",
          mvae._native_reason(type("V", (), {"first_stage_model": type("M", (), {"comfy_has_chunked_io": True})(), "latent_dim": 2})(), l := torch.zeros(1, 4, 8, 8)) is not None
          and mvae._native_reason(type("V", (), {"first_stage_model": object(), "latent_dim": 1})(), torch.zeros(1, 4, 8)) is not None
          and mvae._native_reason(wan, l) is not None and mvae._native_reason(sdxl, l) is None)

    # OOM inside a conv while the workspace is above a threshold: retried with smaller blocks
    lat = torch.randn(1, 4, 12, 10, generator=g)
    ref = native_decode(sdxl, lat)
    orig_call = vae_ops._ConvChunker.__call__
    seen = []

    def make_oom(threshold):
        def call(self, *a, **kw):
            seen.append(self.budget)
            if self.budget > threshold:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return orig_call(self, *a, **kw)
        return call

    mvae.set_workspace(512 * 1024 * 1024)
    vae_ops._ConvChunker.__call__ = make_oom(128 * 1024 * 1024)
    try:
        with Spy() as spy:
            out = comfy.sd.VAE.decode(sdxl, lat)
    finally:
        vae_ops._ConvChunker.__call__ = orig_call
    last = mvae.last_decode()
    check("simulated OOM above 128 MiB: retried at {} after {} retries, result == native (max|Δ| {:.2g}), decode_tiled_ not called".format(
        vae_ops.fmt_bytes(last.get("workspace")), last.get("retries"), (out - ref).abs().max().item()),
        last.get("retries") == 2 and last.get("workspace") == 128 * 1024 * 1024 and (out - ref).abs().max().item() <= 1e-4
        and spy.tiled == 0 and no_overrides(sdxl.first_stage_model))

    vae_ops._ConvChunker.__call__ = make_oom(0)
    try:
        with Spy() as spy:
            expect_raises("OOM even at the smallest workspace: MonoloadVAEOOMError (no tiled fallback)", MonoloadVAEOOMError,
                          lambda: comfy.sd.VAE.decode(sdxl, lat), "never falls back to the approximate tiled", "64 MiB")
    finally:
        vae_ops._ConvChunker.__call__ = orig_call
    check("after the failed decode: decode_tiled_ not called, no override left", spy.tiled == 0 and no_overrides(sdxl.first_stage_model))

    # a non-OOM error propagates unchanged, overrides removed
    def boom(self, *a, **kw):
        raise ValueError("boom")
    vae_ops._ConvChunker.__call__ = boom
    try:
        expect_raises("non-OOM error inside the decode propagates as is", ValueError, lambda: comfy.sd.VAE.decode(sdxl, lat), "boom")
    finally:
        vae_ops._ConvChunker.__call__ = orig_call
    check("after the non-OOM error: no override left", no_overrides(sdxl.first_stage_model))


def misc_tests():
    ok = (mvae.parse_size("1G") == 1 << 30 and mvae.parse_size("512M") == 512 << 20 and mvae.parse_size("768") == 768 << 20
          and mvae.parse_size("1.5GiB") == int(1.5 * (1 << 30)) and mvae.parse_size("4096K") == 4 << 20 and mvae.parse_size("2048b") == 2048)
    check("MONOLOAD_VAE_WORKSPACE parsing (1G, 512M, 768 = MiB, 1.5GiB, 4096K, 2048b)", ok)


def timing_sync_test(v):
    """The decode time in the log is measured between two device syncs (kernels
    run asynchronously; an unsynchronized host clock stops too early)."""
    calls = []
    orig = comfy.model_management.synchronize
    comfy.model_management.synchronize = lambda: calls.append(1)
    try:
        mvae.set_workspace(16 * 1024)
        comfy.sd.VAE.decode(v, torch.randn(1, 4, 12, 10, generator=torch.Generator().manual_seed(5)))
    finally:
        comfy.model_management.synchronize = orig
    check("managed decode synchronizes the device before starting and before stopping the clock ({} calls)".format(len(calls)), len(calls) == 2)


# ---------------------------------------------------------------------------
# 5. coverage fixes (phase 4b-0): what must stay native
# ---------------------------------------------------------------------------

def svd_vae():
    """comfy.sd.VAE of SVD's structure (sd.py builds it full size from the mix_factor key), random weights."""
    from comfy.ldm.models.autoencoder import AutoencodingEngine
    enc = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4],
           'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
    m = AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                           encoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Encoder", 'params': enc},
                           decoder_config={'target': "comfy.ldm.modules.temporal_ae.VideoDecoder",
                                           'params': dict(enc, video_kernel_size=[3, 1, 1], alpha=0.0)})
    init_random(m)
    sd = m.state_dict()
    del m
    return comfy.sd.VAE(sd=sd, dtype=torch.float32)


def coverage_tests(sdxl, wan):
    g = torch.Generator().manual_seed(11)
    mvae.set_workspace(16 * 1024)
    # SVD: 3 frames as a batch; a frame-by-frame (managed) decode would differ
    v = svd_vae()
    lat = torch.randn(3, 4, 6, 8, generator=g)
    with Spy() as spy:
        out = comfy.sd.VAE.decode(v, lat)
    last = mvae.last_decode()
    ref = native_decode(v, lat)
    alone = torch.cat([native_decode(v, lat[i:i + 1]) for i in range(lat.shape[0])])
    check("SVD VideoDecoder (frames as the batch): left native, result == native ({}), not the frame-by-frame decode (max|Δ| {:.2g})".format(
        last.get("reason", "")[:90], (alone - ref).abs().max().item()),
        last.get("strategy") == "native" and "batch" in last.get("reason", "") and torch.equal(out, ref)
        and (alone - ref).abs().max().item() > 1e-3 and spy.tiled == 0)
    del v

    # audio VAEs with a 2D latent: marked by extra_1d_channel (ACE-Step, LTX audio) or by an audio upscale ratio (MiniMax audio)
    def fake(**kw):
        return type("V", (), dict({"first_stage_model": torch.nn.Conv2d(1, 1, 3), "latent_dim": 2, "upscale_ratio": 8, "extra_1d_channel": None}, **kw))()
    ace = mvae._native_reason(fake(extra_1d_channel=16, upscale_ratio=4096), torch.zeros(1, 8, 16, 32))
    mmx = mvae._native_reason(fake(upscale_ratio=800), torch.zeros(1, 32, 2, 40))
    img = [mvae._native_reason(fake(upscale_ratio=r), torch.zeros(1, 4, 8, 8)) for r in (4, 8, 16, 32)]
    check("audio VAEs with a 2D latent stay native (ACE-like: {}; MiniMax-like: {}); image ratios 4 / 8 / 16 / 32 do not".format(ace, mmx),
          ace is not None and "audio" in ace and mmx is not None and "audio" in mmx and img == [None] * 4)

    # pixel space: an identity, nothing to chunk
    px = comfy.sd.VAE(sd={"pixel_space_vae": torch.tensor(1.0)}, dtype=torch.float32)
    lat = torch.rand(1, 3, 16, 24, generator=g) * 2 - 1
    with Spy() as spy:
        out = comfy.sd.VAE.decode(px, lat)
    last = mvae.last_decode()
    ref = native_decode(px, lat)
    native_est = px.memory_used_decode(lat.shape, px.vae_dtype)
    check("pixel-space VAE: left native ({}), result == native, load_models_gpu got ComfyUI's estimate {} (not Monoload's 2 GiB)".format(
        last.get("reason", "")[:80], spy.loads), last.get("strategy") == "native" and "nothing to manage" in last.get("reason", "")
        and torch.equal(out, ref) and spy.loads == [native_est])

    # image VAEs are still managed
    check("image VAEs still managed: SDXL-like 4D, Wan-like 5D T=1",
          mvae._native_reason(sdxl, torch.zeros(1, 4, 8, 8)) is None and mvae._native_reason(wan, torch.zeros(1, 16, 1, 8, 8)) is None)


def main():
    if not mvae.is_installed():
        check("vae.install() on this ComfyUI", mvae.install())
    mvae.set_stripe(False)  # this file tests layer 2; layer 1 (Wan stripes): tests/test_vae_stripe.py
    conv_matrix()
    attention_tests()
    sdxl, wan = decoder_tests()
    fallback_and_oom_tests(sdxl, wan)
    coverage_tests(sdxl, wan)
    timing_sync_test(sdxl)
    misc_tests()
    finish()


if __name__ == "__main__":
    main()
