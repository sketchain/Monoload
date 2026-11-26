"""Multi-frame (video) decodes, layer 2: which decoders are managed with more than one latent frame (H1), how their
output reaches the pixel buffer (H3), and the shape probe that bounds their memory (docs/PLAN-video.md §1,
docs/DESIGN.md §9.35).

  MULTIFRAME     first-stage model classes whose T > 1 decode Monoload manages (layer 2), plus (multiframe()) the
                 generic classes AutoencoderKL / AutoencodingEngine when their decoder is HunyuanVideo 1.0's / 1.5's
                 (_BY_DECODER, by structure). The others stay native (vae._native_reason). A class is added once the real machine (bench_vae --frames) and
                 tests/vae_inventory.py (the simulator) show the layer-2 estimate is an upper bound for it.
  WHOLE          classes decoded whole by their own decode (no pieces here), whose layer-2 estimate is always the
                 measured probe (vae._probe_mf), a single frame included: LTX's convolutional VideoVAE (0.9.x /
                 LTX-2 / 2.5; it has chunked output, comfy_has_chunked_io, and writes into the output buffer itself)
                 and LTX's diffusion decoder (CausalDiffusionVAE / NADiffusionDecoder; sd.py: decoder.conv_in_x_t.weight).
                 Group F (docs/PLAN-video.md) adds their own op handling through vae_ops.EXTRA_CHUNKERS.
  chunks()       the decode of one sample as a generator of output pieces in the final layout, [1, C, f, H, W],
                 consecutive along the frames: for the decoders below the pieces the decoder already produces
                 (ComfyUI keeps them in a list and torch.cat's them at the end: two copies of the output in the VAE
                 dtype on the device), otherwise the whole decode as one piece. Numerically the same as decode():
                 the same calls on the same tensors, torch.cat is a copy.
  output_frames()  frames of the decoded output, from sd.py's own upscale_ratio.
"""

import torch
import torch.nn.functional as F

import comfy.ldm.hunyuan_video.vae_refiner as hyr
import comfy.ldm.lightricks.vae.causal_video_autoencoder as ltx
import comfy.ldm.lightricks.vae.na_diffusion_decoder as ltx_diffusion
import comfy.ldm.wan.vae as wan
import comfy.ldm.wan.vae2_2 as wan22
from comfy.ldm.models import autoencoder as ae
from comfy.ldm.modules.diffusionmodules import model as ldm


def _wan21_chunks(fsm, z):
    """comfy.ldm.wan.vae.WanVAE.decode (v0.46.3), one piece per decoder output chunk instead of a list + torch.cat."""
    iter_ = 1 + z.shape[2] // 2
    feat_map = [None] * wan.count_cache_layers(fsm.decoder) if iter_ > 1 else None
    x = fsm.conv2(z)
    for i in range(iter_):
        xi = x[:, :, :1] if i == 0 else x[:, :, 1 + 2 * (i - 1):1 + 2 * i]
        pieces = fsm.decoder(xi, feat_cache=feat_map, feat_idx=[0])
        del xi
        while pieces:
            p = pieces.pop(0)
            yield p
            del p


def _wan22_chunks(fsm, z):
    """comfy.ldm.wan.vae2_2.WanVAE.decode (v0.46.3, 5D latent), one piece per latent frame instead of an O(T^2)
    torch.cat chain; unpatchify (a rearrange of channels into pixels, frames untouched) per piece."""
    feat_map = [None] * wan22.count_conv3d(fsm.decoder)
    x = fsm.conv2(z)
    for i in range(z.shape[2]):
        if i == 0:
            out = fsm.decoder(x[:, :, i:i + 1], feat_cache=feat_map, feat_idx=[0], first_chunk=True)
        else:
            out = fsm.decoder(x[:, :, i:i + 1], feat_cache=feat_map, feat_idx=[0])
        p = wan22.unpatchify(out, patch_size=fsm.patch_size)
        del out
        yield p
        del p


def _hv10(fsm):
    """HunyuanVideo 1.0's VAE (also Kandinsky 5's): sd.py builds a generic AutoencoderKL (ddconfig conv3d, time_compress 4;
    key decoder.conv_in.conv.weight) whose LDM Decoder uses CarriedConv3d (carried) and splits the frames after mid. The
    class alone is the image VAEs' too, so the decoder's structure decides."""
    if type(fsm) is not ae.AutoencoderKL or getattr(fsm, "bn", None) is not None or getattr(fsm, "max_batch_size", None) is not None:
        return False
    dec = getattr(fsm, "decoder", None)
    return (type(dec) is ldm.Decoder and dec.carried is True and not dec.tanh_out and not dec.fused_norm
            and type(dec.conv_in) is ldm.CarriedConv3d)


def _hv15(fsm):
    """HunyuanVideo 1.5's VAE (and the HunyuanImage 2.1 Refiner file, decoding one frame): a generic AutoencodingEngine
    (key decoder.conv_in.conv.weight, 32 channels) with vae_refiner.Decoder, refiner_vae=True (CarriedConv3d, RMS norm,
    frames split after mid). refiner_vae=False (sd.py's 5D decoder.conv_in.weight) decodes whole: not this."""
    if type(fsm) is not ae.AutoencodingEngine:
        return False
    dec = getattr(fsm, "decoder", None)
    return type(dec) is hyr.Decoder and dec.refiner_vae is True and type(dec.conv_in) is ldm.CarriedConv3d


def _hv10_chunks(fsm, z):
    """comfy.ldm.models.autoencoder.AutoencodingEngineLegacy.decode + ldm Decoder.forward (v0.46.3, carried, no
    fused norm, no tanh), one piece per 2-latent-frame chunk instead of a list + torch.cat."""
    dec = fsm.decoder
    h = fsm.post_quant_conv(z)
    h = ldm.conv_carry_causal_3d([h], dec.conv_in)
    h = dec.mid.block_1(h, None)
    h = dec.mid.attn_1(h)
    h = dec.mid.block_2(h, None)
    hs = list(torch.split(h, 2, dim=2))
    del h
    carry_in = None
    n = len(hs)
    for i in range(n):
        h1 = hs[i]
        hs[i] = None
        carry_out = [] if i < n - 1 else None
        for i_level in reversed(range(dec.num_resolutions)):
            for i_block in range(dec.num_res_blocks + 1):
                h1 = dec.up[i_level].block[i_block](h1, None, carry_in, carry_out)
            if i_level != 0:
                h1 = dec.up[i_level].upsample(h1, carry_in, carry_out)
        h1 = dec.norm_out(h1)
        h1 = [ldm.nonlinearity(h1)]
        h1 = ldm.conv_carry_causal_3d(h1, dec.conv_out, carry_in, carry_out)
        carry_in = carry_out
        yield h1
        del h1


def _hv15_chunks(fsm, z):
    """comfy.ldm.hunyuan_video.vae_refiner.Decoder.forward (v0.46.3, refiner_vae=True; AutoencodingEngine.decode calls
    it directly), one piece per 2-latent-frame chunk instead of a list + torch.cat."""
    dec = fsm.decoder
    x = ldm.conv_carry_causal_3d([z], dec.conv_in) + z.repeat_interleave(dec.block_out_channels[0] // dec.z_channels, 1)
    x = dec.mid.block_2(dec.mid.attn_1(dec.mid.block_1(x)))
    xs = list(torch.split(x, 2, dim=2))
    del x
    carry_in = None
    n = len(xs)
    for i in range(n):
        x1 = xs[i]
        xs[i] = None
        carry_out = [] if i < n - 1 else None
        for stage in dec.up:
            for blk in stage.block:
                x1 = blk(x1, None, carry_in, carry_out)
            if hasattr(stage, "upsample"):
                x1 = stage.upsample(x1, carry_in, carry_out)
        x1 = [F.silu(dec.norm_out(x1))]
        x1 = ldm.conv_carry_causal_3d(x1, dec.conv_out, carry_in, carry_out)
        carry_in = carry_out
        yield x1
        del x1


# first-stage model class -> chunk generator; a class here is also managed with T > 1 (MULTIFRAME)
_CHUNKS = {
    wan.WanVAE: _wan21_chunks,
    wan22.WanVAE: _wan22_chunks,
}

# generic first-stage model classes (shared with image VAEs): (structure test on the model, chunk generator, name)
_BY_DECODER = (
    (_hv10, _hv10_chunks, "HunyuanVideo 1.0 (AutoencoderKL + carried Decoder)"),
    (_hv15, _hv15_chunks, "HunyuanVideo 1.5 (AutoencodingEngine + vae_refiner.Decoder)"),
)

WHOLE = frozenset((ltx.VideoVAE, ltx_diffusion.CausalDiffusionVAE))

MULTIFRAME = frozenset(_CHUNKS) | WHOLE

MULTIFRAME_NAMES = tuple(name for _, _, name in _BY_DECODER)   # the generic-class ones, for vae.nr_multiframe


def _decoder_chunks(fsm):
    for test, f, _ in _BY_DECODER:
        if test(fsm):
            return f
    return None


def multiframe(fsm):
    """Whether this first-stage model's T > 1 decode is managed: its exact class (a subclass may decode differently), or
    for a generic class, its decoder's structure (_BY_DECODER)."""
    return type(fsm) in MULTIFRAME or _decoder_chunks(fsm) is not None


def measured(fsm):
    """Whether layer 2 always bounds this model's decode with the measured probe (WHOLE), whatever the frames."""
    return type(fsm) in WHOLE


def chunked_io_managed(fsm):
    """Whether a model with chunked output (comfy_has_chunked_io: it writes into the output buffer itself) is managed:
    LTX's VideoVAE. The others (MiniMax H3, SeedVR2, RelayFlash: they tile themselves) stay native."""
    return type(fsm) is ltx.VideoVAE


def chunks(fsm, z, vae_options):
    """Pieces of the decode of one sample z ([1, C, T, h, w] on the device), see the module docstring."""
    f = _CHUNKS.get(type(fsm)) or _decoder_chunks(fsm)
    if f is not None and not vae_options and z.ndim == 5:
        return f(fsm, z)
    return iter((fsm.decode(z, **vae_options),))


def output_frames(vae, t):
    """Frames out of t latent frames (sd.py's upscale_ratio[0]), None if the VAE has no such rule."""
    r = getattr(vae, "upscale_ratio", None)
    if isinstance(r, tuple) and r and callable(r[0]):
        return int(r[0](t))
    return None
