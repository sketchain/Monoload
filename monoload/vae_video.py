"""Multi-frame (video) decodes, layer 2: which decoders are managed with more than one latent frame (H1), how their
output reaches the pixel buffer (H3), and the shape probe that bounds their memory (docs/PLAN-video.md §1,
docs/DESIGN.md §9.35).

  MULTIFRAME     first-stage model classes whose T > 1 decode Monoload manages (layer 2). The others stay native
                 (vae._native_reason). A class is added once the real machine (bench_vae --frames) and
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

import comfy.ldm.cogvideo.vae as cogvideo
import comfy.ldm.cosmos.vae as cosmos
import comfy.ldm.genmo.vae.model as genmo
import comfy.ldm.lightricks.vae.causal_video_autoencoder as ltx
import comfy.ldm.lightricks.vae.na_diffusion_decoder as ltx_diffusion
import comfy.ldm.wan.vae as wan
import comfy.ldm.wan.vae2_2 as wan22

try:   # group C's module (CogVideoX); until it is merged CogVideoX stays native at T > 1
    from . import vae_cogvideo
except ImportError:
    vae_cogvideo = None
try:   # group D (Mochi, Cosmos 1.0); until merged their T > 1 stays native
    from . import vae_mochi
except ImportError:
    vae_mochi = None
try:
    from . import vae_cosmos
except ImportError:
    vae_cosmos = None


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


# first-stage model class -> chunk generator; a class here is also managed with T > 1 (MULTIFRAME)
_CHUNKS = {
    wan.WanVAE: _wan21_chunks,
    wan22.WanVAE: _wan22_chunks,
}
if vae_cogvideo is not None:
    _CHUNKS[cogvideo.AutoencoderKLCogVideoX] = vae_cogvideo.chunks   # group C: up_blocks[1]'s time upsampling per chunk
if vae_mochi is not None:
    _CHUNKS[genmo.VideoVAE] = vae_mochi.chunks          # group D: one frame at a time, causal-conv caches
if vae_cosmos is not None:
    _CHUNKS[cosmos.CausalContinuousVideoTokenizer] = vae_cosmos.chunks

WHOLE = frozenset((ltx.VideoVAE, ltx_diffusion.CausalDiffusionVAE))

MULTIFRAME = frozenset(_CHUNKS) | WHOLE


def multiframe(fsm):
    """Whether this first-stage model's T > 1 decode is managed (its exact class: a subclass may decode differently)."""
    return type(fsm) in MULTIFRAME


def measured(fsm):
    """Whether layer 2 always bounds this model's decode with the measured probe (WHOLE), whatever the frames."""
    return type(fsm) in WHOLE


def chunked_io_managed(fsm):
    """Whether a model with chunked output (comfy_has_chunked_io: it writes into the output buffer itself) is managed:
    LTX's VideoVAE. The others (MiniMax H3, SeedVR2, RelayFlash: they tile themselves) stay native."""
    return type(fsm) is ltx.VideoVAE


def chunks(fsm, z, vae_options):
    """Pieces of the decode of one sample z ([1, C, T, h, w] on the device), see the module docstring."""
    f = _CHUNKS.get(type(fsm))
    if f is not None and not vae_options and z.ndim == 5:
        return f(fsm, z)
    return iter((fsm.decode(z, **vae_options),))


def output_frames(vae, t):
    """Frames out of t latent frames (sd.py's upscale_ratio[0]), None if the VAE has no such rule. CogVideoX: its
    decoder's own rule (sd.py's 4t - 3 holds for odd t only; an even t decodes to 4t frames)."""
    if vae_cogvideo is not None:
        n = vae_cogvideo.output_frames(getattr(vae, "first_stage_model", None), t)
        if n is not None:
            return n
    r = getattr(vae, "upscale_ratio", None)
    if isinstance(r, tuple) and r and callable(r[0]):
        return int(r[0](t))
    return None
