"""Scheme G (the grouped norm, DESIGN §9.29) on the two other big video norms (DESIGN §9.37, PLAN-video H4):
comfy.ldm.hunyuan_video.vae_refiner.RMS_norm (HunyuanVideo 1.5 / the HunyuanImage Refiner file) and LTX's
comfy.ldm.lightricks.vae.pixel_norm.PixelNorm.

Random weights, no model files. On a CUDA / HIP device when there is one (the real machine), else the CPU.

  1. refiner RMS_norm: 1024 channels replaced (4 x 256) only inside OpChunking, 384 left alone; == the module's forward
     (fp32 <= 1e-6 relative, bf16 <= 1 ulp), 384 bit-identical; not a hook for recognition; restored;
  2. PixelNorm: every one gets the replacement, which runs the class's forward bit-identically at <= 510 channels and
     the grouped mean of squares above (fp32 <= 1e-6 relative; bf16 within 2 ulp of the native result and no farther
     from fp64 than native + 1 ulp); restored;
  3. whole decoders under OpChunking == native: LTX 2 VideoVAE(version=2) decoder at full width (1024 / 512 / 256 / 128
     channel PixelNorms), the refiner_vae=True vae_refiner.Decoder with the threshold lowered to 16 channels so that
     its small norms are grouped too; the counts in OpStats.norm_grouped.

    python tests/test_vae_norm_g.py
"""

import torch

from common import check, finish
import comfy.ldm.hunyuan_video.vae_refiner as hyr
import comfy.ldm.lightricks.vae.causal_video_autoencoder as ltx
import comfy.ldm.lightricks.vae.pixel_norm as pn
from monoload import vae_engine as eng
from monoload import vae_ops
from test_vae import init_random

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def overrides(model):
    return [n for n, m in model.named_modules() if "forward" in m.__dict__ or "_conv_forward" in m.__dict__]


def err(a, b):
    return float((a.float() - b.float()).abs().max())


def ulp_ok(got, ref, n):
    ulp = (ref.float().abs() * 2 ** -7).clamp_min(2 ** -133)
    return bool(((got.float() - ref.float()).abs() <= n * ulp).all())


def refiner_norm_tests():
    torch.manual_seed(11)
    holder = torch.nn.Module()
    holder.big = hyr.RMS_norm(1024)
    holder.small = hyr.RMS_norm(384)
    with torch.no_grad():
        holder.big.gamma.uniform_(0.5, 1.5)
        holder.small.gamma.uniform_(0.5, 1.5)
    holder.to(DEV)
    for dt, tol in ((torch.float32, 1e-6), (torch.bfloat16, None)):
        hm = holder.to(dt)
        x = torch.randn(1, 1024, 3, 13, 24, device=DEV).to(dt).narrow(3, 2, 9)   # a row slice, as in the stripes
        y = torch.randn(1, 384, 3, 9, 24, device=DEV).to(dt)
        ref_big, ref_small = hm.big(x), hm.small(y)
        with vae_ops.OpChunking(hm, 1 << 20) as st:
            installed = vae_ops.is_replacement(hm.big.__dict__.get("forward")) and "forward" not in hm.small.__dict__
            hooked = eng.hooked_module(hm)
            got_big, got_small = hm.big(x), hm.small(y)
        ok_big = ulp_ok(got_big, ref_big, 1) if tol is None else err(got_big, ref_big) <= tol * float(ref_big.abs().max())
        check("{} {}: refiner RMS_norm 1024 ch grouped only inside OpChunking ({} replaced), 384 ch left alone; == the module's "
              "forward (max|Δ| {:.2g}); 384 ch bit-identical; not a hook; restored".format(
                  DEV.type, str(dt).replace("torch.", ""), st.norm_grouped, err(got_big, ref_big)),
              installed and st.norm_grouped == 1 and ok_big and torch.equal(got_small, ref_small) and hooked is None
              and not overrides(hm))


def pixel_norm_tests():
    torch.manual_seed(12)
    holder = torch.nn.Module()
    holder.n = pn.PixelNorm()
    holder.a = pn.PixelNorm(dim=1, eps=1e-6)    # LTX 2 audio's
    for dt in (torch.float32, torch.bfloat16):
        x = (torch.randn(1, 1024, 3, 13, 24, device=DEV) * 3).to(dt).narrow(3, 2, 9)
        x512 = torch.randn(2, 512, 2, 7, 11, device=DEV).to(dt)
        y = torch.randn(1, 384, 3, 9, 24, device=DEV).to(dt)
        z = torch.randn(1, 128, 40, 16, device=DEV).to(dt)
        refs = [holder.n(x), holder.n(x512), holder.n(y), holder.a(z)]
        with vae_ops.OpChunking(holder, 1 << 20) as st:
            installed = all(vae_ops.is_replacement(m.__dict__.get("forward")) for m in (holder.n, holder.a))
            hooked = eng.hooked_module(holder)
            gots = [holder.n(x), holder.n(x512), holder.n(y), holder.a(z)]
        if dt == torch.float32:
            ok = all(err(g, r) <= 1e-6 * float(r.abs().max()) for g, r in zip(gots[:2], refs[:2]))
            detail = "max|Δ| {:.2g}".format(max(err(g, r) for g, r in zip(gots[:2], refs[:2])))
        else:
            ok = True
            parts = []
            for g, r, inp in zip(gots[:2], refs[:2], (x, x512)):
                f64 = inp.double() / torch.sqrt(torch.mean(inp.double() ** 2, dim=1, keepdim=True) + holder.n.eps)
                e_g, e_r = err(g, f64), err(r, f64)
                one = float((f64.abs() * 2 ** -7).max())
                ok = ok and ulp_ok(g, r, 2) and e_g <= e_r + one
                parts.append("vs fp64 {:.2g} (native {:.2g})".format(e_g, e_r))
            detail = "; ".join(parts)
        check("{} {}: PixelNorm 1024 / 512 ch grouped inside OpChunking ({} replaced: every PixelNorm), == native ({}); "
              "384 ch and the 128 ch audio one bit-identical; not a hook; restored".format(
                  DEV.type, str(dt).replace("torch.", ""), st.norm_grouped, detail),
              installed and st.norm_grouped == 2 and ok and torch.equal(gots[2], refs[2]) and torch.equal(gots[3], refs[3])
              and hooked is None and not overrides(holder))
    # a PixelNorm over another dim than the channels: the class's own forward
    holder.n.dim = -1
    x = torch.randn(1, 16, 2, 3, 1024, device=DEV)
    ref = holder.n(x)
    with vae_ops.OpChunking(holder, 1 << 20):
        got = holder.n(x)
    holder.n.dim = 1
    check("PixelNorm over dim -1 (1024 wide): the class's forward, bit-identical", torch.equal(got, ref))


def ltx_decoder_tests():
    torch.manual_seed(13)
    vae = init_random(ltx.VideoVAE(version=2)).to(DEV).float().eval()
    dec = vae.decoder
    norms = sum(1 for m in dec.modules() if type(m) is pn.PixelNorm)
    z = torch.randn(1, 128, 2, 4, 6, device=DEV)
    with torch.no_grad():
        ref = dec(z, timestep=0.05)
        with vae_ops.OpChunking(dec, 256 << 20) as st:
            got = dec(z, timestep=0.05)
    rel = err(got, ref) / float(ref.abs().max())
    check("{}: LTX 2 VideoVAE decoder (random weights, {} PixelNorms, latent 2x4x6) under OpChunking == native "
          "(rel max|Δ| {:.2g}), {} norms replaced, nothing left installed".format(DEV.type, norms, rel, st.norm_grouped),
          rel <= 1e-5 and st.norm_grouped == norms > 0 and not overrides(dec))
    del vae, dec


def refiner_decoder_tests():
    torch.manual_seed(14)
    boc = [32, 32, 64, 64, 64]
    dd = {"block_out_channels": boc, "in_channels": 3, "out_channels": 3, "num_res_blocks": 2, "ffactor_spatial": 16,
          "ffactor_temporal": 4, "downsample_match_channel": True, "upsample_match_channel": True, "refiner_vae": True,
          "z_channels": 32}
    dec = init_random(hyr.Decoder(**dd)).to(DEV).float().eval()
    norms = sum(1 for m in dec.modules() if type(m) is hyr.RMS_norm)
    z = torch.randn(1, 32, 2, 4, 6, device=DEV)
    orig = vae_ops.GROUP_NORM_MAX
    vae_ops.GROUP_NORM_MAX = 16
    try:
        with torch.no_grad():
            ref = dec(z)
            with vae_ops.OpChunking(dec, 256 << 20) as st:
                got = dec(z)
    finally:
        vae_ops.GROUP_NORM_MAX = orig
    rel = err(got, ref) / float(ref.abs().max())
    check("{}: vae_refiner.Decoder(refiner_vae=True), threshold 16 ch: {} of {} RMS norms grouped under OpChunking, "
          "== native (rel max|Δ| {:.2g}), nothing left installed".format(DEV.type, st.norm_grouped, norms, rel),
          rel <= 1e-5 and st.norm_grouped == norms > 0 and not overrides(dec))


def main():
    refiner_norm_tests()
    pixel_norm_tests()
    ltx_decoder_tests()
    refiner_decoder_tests()
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
