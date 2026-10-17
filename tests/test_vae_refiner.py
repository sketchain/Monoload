"""The HunyuanImage 2.1 Refiner VAE (comfy.ldm.hunyuan_video.vae_refiner.Decoder, refiner_vae=False;
DESIGN §9.25 / §9.30).

No model files needed: ComfyUI's own AutoencodingEngine with vae_refiner.Decoder / Encoder with small channels and
random weights, through comfy.sd.VAE (it detects the kind from the 5D decoder.conv_in with 32 input channels; its
hard-coded block_out_channels are swapped for small ones while it builds), latent 32 channels at 1/16, 5D T=1, fp32
on the CPU unless noted.

  1. layer-2 estimate: 6 activation copies for this decoder, by its class (a forward hook does not change it);
     HunyuanVideo 1.5 (the same class with refiner_vae=True) keeps the generic 4; the managed decode == native;
  2. caching allocator (tests/alloc_sim.py / vae_inventory.py, full-size Refiner on the meta device): layer 2's
     simulated reserved peak <= the layer-2 estimate (1344x768, 1536x1536, 2048x2048; the 4 generic copies are
     not a bound at 2048x2048).

    python tests/test_vae_refiner.py
"""

import torch

from common import check, finish
import comfy.sd
import comfy.ldm.hunyuan_video.vae_refiner as hyr
from comfy.ldm.models.autoencoder import AutoencodingEngine
from monoload import vae as mvae
from monoload import vae_hunyuan as vh
from test_vae import init_random, managed_decode, native_decode

torch.manual_seed(0)
Z = 32
BOC = [32, 32, 64, 64, 64]     # reversed in the decoder: 64 (H/16, 2 x z: repeat 2), 64, 64, 32, 32 channels
DD = {"block_out_channels": BOC, "in_channels": 3, "out_channels": 3, "num_res_blocks": 2, "ffactor_spatial": 16,
      "ffactor_temporal": 4, "downsample_match_channel": True, "upsample_match_channel": True, "refiner_vae": False, "z_channels": Z}
_DEC, _ENC = hyr.Decoder, hyr.Encoder
G = float(1 << 30)


def refiner_model(**kw):
    dd = dict(DD, **kw)
    return init_random(AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                          encoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Encoder", 'params': dd},
                                          decoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Decoder", 'params': dd}))


def refiner_vae(dtype=None, model=None):
    """comfy.sd.VAE on a small Refiner state dict (sd.py's block_out_channels swapped for BOC while it builds)."""
    m = model if model is not None else refiner_model()
    hyr.Decoder = lambda **kw: _DEC(**dict(kw, block_out_channels=BOC))
    hyr.Encoder = lambda **kw: _ENC(**dict(kw, block_out_channels=BOC))
    try:
        return comfy.sd.VAE(sd=m.state_dict(), dtype=dtype)
    finally:
        hyr.Decoder, hyr.Encoder = _DEC, _ENC


def lat_(h, w, b=1, seed=4):
    return torch.randn(b, Z, 1, h, w, generator=torch.Generator().manual_seed(seed))


def estimate_tests():
    v = refiner_vae()
    dec = v.first_stage_model.decoder
    check("comfy.sd.VAE builds the Refiner: vae_refiner.Decoder, refiner_vae False, latent_dim 3",
          type(dec) is _DEC and not dec.refiner_vae and v.latent_dim == 3)
    lat = lat_(5, 6)
    est, probe = mvae._layer2_estimate(v, lat, {}, mvae.workspace())
    four = mvae.estimate(v, lat, mvae.workspace(), probe)
    check("layer-2 estimate with {} activation copies: {:.0f} KiB (the generic 4: {:.0f} KiB), nothing retained".format(
        vh.REFINER_ACTIVATION_COPIES, est["total"] / 1024, four["total"] / 1024),
        est["total"] == four["total"] + 2 * four["activation"] and est["retained"] == 0)
    hk = dec.conv_out.register_forward_hook(lambda *a: None)
    try:
        terms = mvae._decoder_terms(v, lat)
    finally:
        hk.remove()
    check("forward hook: still {} activation copies (by the decoder's class)".format(terms["copies"]),
          terms == {"retained": 0, "copies": vh.REFINER_ACTIVATION_COPIES})
    out = managed_decode(v, lat, raw=True)
    ref = native_decode(v, lat, raw=True)
    rec = mvae.last_decode()
    check("managed decode: layer 2, == native (max|Δ| {:.2g})".format(float((out - ref).abs().max())),
          rec.get("strategy") == "layer2" and out.shape == ref.shape and float((out - ref).abs().max()) <= 1e-5)
    # HunyuanVideo 1.5 builds the same class with refiner_vae=True (CarriedConv3d, RMS norm, the first frame apart):
    # not this term
    vid = refiner_model(refiner_vae=True)
    shim = type("V", (), {"first_stage_model": vid})()
    check("HunyuanVideo 1.5's decoder (refiner_vae=True): the generic terms", mvae._decoder_terms(shim, lat) == {"retained": 0, "copies": mvae.ACTIVATION_COPIES})


def allocator_tests():
    import alloc_sim
    import vae_inventory as vi
    kind = next(k for k in vi.KINDS if k[0] == "hy_refiner")
    v = vi.build(kind)
    for w, h in ((1344, 768), (1536, 1536), (2048, 2048)):
        shape = vi.latent_shape(v, w, h, 1)
        t = vi.trace(v, shape, 2)
        lat = torch.empty(shape, device="meta", dtype=v.vae_dtype)
        with alloc_sim._patched(mvae.mm, "get_free_memory", lambda dev=None, torch_free_too=False: (48 << 30, 48 << 30) if torch_free_too else 48 << 30):
            mvae._PROBES.clear()
            p = mvae._probe(v, lat, {})
        est = mvae.estimate(v, lat, mvae.DEFAULT_WORKSPACE, p, **mvae._decoder_terms(v, lat))
        four = mvae.estimate(v, lat, mvae.DEFAULT_WORKSPACE, p)
        check("Refiner {}x{} layer 2: simulated reserved {:.2f} GiB <= estimate {:.2f} GiB (6 activation copies; 4 would give {:.2f})".format(
            w, h, t["reserved"] / G, est["total"] / G, four["total"] / G), t["reserved"] <= est["total"])
        if (w, h) == (2048, 2048):
            check("... at 2048x2048 the generic 4 copies are not a bound ({:.2f} > {:.2f} GiB)".format(t["reserved"] / G, four["total"] / G),
                  t["reserved"] > four["total"])


def main():
    if not mvae.is_installed():
        check("vae.install() on this ComfyUI", mvae.install())
    estimate_tests()
    allocator_tests()
    finish()


if __name__ == "__main__":
    main()
