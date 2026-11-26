"""Phase 4a: inventory of every VAE comfy.sd.VAE (ComfyUI v0.46.3) can build.

For each kind the first-stage model is built full size on the meta device with
the configuration comfy/sd.py uses, and its state dict (meta tensors) is handed
to comfy.sd.VAE itself (dtype bf16, is_amd() true as on CT 700), so the branch
sd.py picks, latent_dim, the ratios and memory_used_decode are ComfyUI's own.
No model files, no memory (meta tensors). Then, for typical latents:

  * structure   module counts: GroupNorm (whole-tensor statistics), RMS /
                pixel / layer / batch norms, attention blocks (with an
                optimized_attention ComfyUI function or their own), Conv2d /
                Conv3d / ConvTranspose, Linear;
  * path        what Monoload does now: native (and why), layer 1 (which
                adapter) or layer 2 (and why not layer 1);
  * peaks       tests/alloc_sim.py traces on the CT 700 backends (bf16, 4D conv
                = Slow2d im2col, 5D conv = SlowDilated3d vol2col, upsampling
                copies, the caching allocator): native decode and layer 2
                (1 GiB workspace), reserved increase as bench_vae measures it
                (weights loaded, cache emptied; the output buffer, fp32, on the
                device as with --gpu-only). A multi-frame layer 2 (H1 of
                docs/PLAN-video.md, DESIGN §9.35) decodes into the output buffer
                piece by piece as Monoload does (vae._decode_sample), and its
                estimate comes from vae._probe_mf run on the simulator (its peak
                meter is the simulated allocator's). Layer 1 for the kinds with
                an adapter (alloc_sim.decode_trace). ComfyUI's own estimate
                (memory_used_decode, AMD) for comparison.

    python tests/vae_inventory.py                  # all kinds, all cases
    python tests/vae_inventory.py --only wan21,ldm_flux2 --no-trace
    python tests/vae_inventory.py --json out.json

A kind that cannot be built or traced on the meta device is reported with the
error; its numbers then have to be estimated by hand (docs/DESIGN.md §9.16).
"""

import argparse
import contextlib
import itertools
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import common  # noqa: E402,F401  (ComfyUI environment)
import torch  # noqa: E402

import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402

import alloc_sim  # noqa: E402
from monoload import vae as mvae, vae_ops  # noqa: E402

G = float(1 << 30)
GIB = 1 << 30
FREE_FOR_NATIVE = 48 * GIB   # what get_free_memory reports to native slice_attention / batching in the trace
TOTAL = int(62.5 * GIB)      # CT 700 GTT
CAP = 60 * GIB               # the most a decode can add on CT 700 (62.5 GiB GTT, minus the weights and the rest of the machine)


class CappedSim(alloc_sim.AllocatorSim):
    """The caching allocator with a device limit: a request that needs a new
    segment beyond CAP first releases every cached free segment and retries
    (as the CUDA / HIP allocator does on OOM); still beyond CAP is a real OOM
    (counted; native decode would then fall back to tiled). The trace goes on."""

    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        self.base = 0
        self.cache_flushes = 0
        self.ooms = 0

    def malloc(self, nbytes):
        import bisect
        size = self.round_size(max(int(nbytes), 1))
        small = size <= alloc_sim.K_SMALL_SIZE
        pool = self.pools[small]
        if bisect.bisect_left(pool, (size, -1)) >= len(pool):
            seg = self.allocation_size(size)
            if self.reserved - self.base + seg > self.cap:
                self.empty_cache()
                self.cache_flushes += 1
                if self.reserved - self.base + seg > self.cap:
                    self.ooms += 1
        return super().malloc(nbytes)


# ---------------------------------------------------------------------------
# the kinds: how comfy/sd.py builds them (configs copied from 0.31.0)
# ---------------------------------------------------------------------------

def _meta_sd(build, rename=None, drop=()):
    try:
        with torch.device("meta"):
            m = build()
        sd = m.state_dict()
    except RuntimeError as e:
        if "meta" not in str(e):
            raise
        m = build()   # an __init__ that reads tensor values (e.g. Cosmos' wavelet buffers): built on the CPU, then shapes only
        sd = {k: torch.empty(v.shape, dtype=v.dtype, device="meta") for k, v in m.state_dict().items()}
        del m
    if rename:
        out = {}
        for k, v in sd.items():
            for a, b in rename:
                if k.startswith(a):
                    k = b + k[len(a):]
                    break
            out[k] = v
        sd = out
    return {k: v for k, v in sd.items() if not any(k.startswith(d) for d in drop)}


LDM_DD = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4],
          'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}


def _ldm_kl(z=4, ch_mult=(1, 2, 4, 4), bn=False):
    from comfy.ldm.models.autoencoder import AutoencoderKL
    dd = dict(LDM_DD, z_channels=z, ch_mult=list(ch_mult))
    if bn:
        dd["batch_norm_latent"] = True
    return lambda: AutoencoderKL(ddconfig=dd, embed_dim=z)


def _ldm_engine(z=16):
    from comfy.ldm.models.autoencoder import AutoencodingEngine
    dd = dict(LDM_DD, z_channels=z)
    return lambda: AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                      encoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Encoder", 'params': dd},
                                      decoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Decoder", 'params': dd})


def _svd():
    from comfy.ldm.models.autoencoder import AutoencodingEngine
    enc = dict(LDM_DD)
    dec = dict(enc, video_kernel_size=[3, 1, 1], alpha=0.0)
    return lambda: AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                      encoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Encoder", 'params': enc},
                                      decoder_config={'target': "comfy.ldm.modules.temporal_ae.VideoDecoder", 'params': dec})


def _hy_image21():
    from comfy.ldm.models.autoencoder import AutoencodingEngine
    dd = {"block_out_channels": [128, 256, 512, 512, 1024, 1024], "in_channels": 3, "out_channels": 3, "num_res_blocks": 2, "ffactor_spatial": 32,
          "downsample_match_channel": True, "upsample_match_channel": True, "z_channels": 64}
    return lambda: AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                      encoder_config={'target': "comfy.ldm.hunyuan_video.vae.Encoder", 'params': dd},
                                      decoder_config={'target': "comfy.ldm.hunyuan_video.vae.Decoder", 'params': dd})


def _hy_refiner(video):
    from comfy.ldm.models.autoencoder import AutoencodingEngine
    dd = {"block_out_channels": [128, 256, 512, 1024, 1024], "in_channels": 3, "out_channels": 3, "num_res_blocks": 2, "ffactor_spatial": 16,
          "ffactor_temporal": 4, "downsample_match_channel": True, "upsample_match_channel": True, "z_channels": 32}
    if not video:
        dd["refiner_vae"] = False
    reg = "EmptyRegularizer" if video else "DiagonalGaussianRegularizer"
    return lambda: AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder." + reg},
                                      encoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Encoder", 'params': dd},
                                      decoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Decoder", 'params': dd})


def _hunyuan_video():
    from comfy.ldm.models.autoencoder import AutoencoderKL
    dd = dict(LDM_DD, z_channels=16, conv3d=True, time_compress=4)
    return lambda: AutoencoderKL(ddconfig=dd, embed_dim=16)


def _wan21(dim=96):
    import comfy.ldm.wan.vae as wan
    return lambda: wan.WanVAE(dim=dim, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True],
                              image_channels=3, conv_out_channels=3, dropout=0.0)


def _wan22():
    import comfy.ldm.wan.vae2_2 as wan22
    return lambda: wan22.WanVAE(dim=160, z_dim=48, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True],
                                dropout=0.0)


def _cosmos():
    import comfy.ldm.cosmos.vae as cv
    dd = {'z_channels': 16, 'latent_channels': 16, 'z_factor': 1, 'resolution': 1024, 'in_channels': 3, 'out_channels': 3, 'channels': 128,
          'channels_mult': [2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [32], 'dropout': 0.0, 'patch_size': 4, 'num_groups': 1,
          'temporal_compression': 8, 'spacial_compression': 8}
    return lambda: cv.CausalContinuousVideoTokenizer(**dd)


def _taesd(ch):
    import comfy.taesd.taesd as t
    return lambda: t.TAESD(latent_channels=ch)


def _taehv(ch):
    import comfy.taesd.taehv as t
    return lambda: t.TAEHV(latent_channels=ch, latent_format=None)


def _mochi():
    import comfy.ldm.genmo.vae.model as m
    return lambda: m.VideoVAE()


def _ltxv(version):
    import comfy.ldm.lightricks.vae.causal_video_autoencoder as l
    return lambda: l.VideoVAE(version=version, config=None)


def _cogvideo():
    import comfy.ldm.cogvideo.vae as c
    return lambda: c.AutoencoderKLCogVideoX(latent_channels=16)


def _seedvr2():
    import comfy.ldm.seedvr.vae as s
    return lambda: s.VideoAutoencoderKLWrapper()


def _mage():
    import comfy.ldm.mage_flow.vae as m
    return lambda: m.MageVAE()


def _stage_a():
    from comfy.ldm.cascade.stage_a import StageA
    return StageA


def _previewer():
    from comfy.ldm.cascade.stage_c_coder import StageC_coder
    return lambda: StageC_coder().previewer


def _minimax_video():
    import comfy.ldm.minimax.vae as m
    return lambda: m.MiniMaxH3VideoVAE()


def _minimax_image(cls):
    import comfy.ldm.minimax.vae as m
    return lambda: getattr(m, cls)()


def _ltx_diffusion():
    import comfy.ldm.lightricks.vae.na_diffusion_decoder as n
    return lambda: n.CausalDiffusionVAE(config=None)


def _relayflash(factor):
    import comfy.ldm.wan.model_vsr as r
    return lambda: r.RelayFlash(factor)


def _pixel():
    import comfy.pixel_space_convert as p
    return p.PixelspaceConversionVAE


IMG = [(1344, 768, 1), (3840, 2160, 1)]          # (width, height, frames): the bench resolutions
IMG1 = [(1344, 768, 1)]

# name -> (used by, builder, sd rename, sd drop, cases, metadata); a case (w, h, frames, [batch]); frames 1 = image
KINDS = [
    ("ldm_sd", "SD1.x / SD2.x（AutoencoderKL，z 4）", _ldm_kl(4), None, (), IMG, None),
    ("ldm_sdxl", "SDXL / Pony / Illustrious（同一结构，z 4；用户：waiIllustriousSDXL 内置）", _ldm_kl(4), None, (), IMG, None),
    ("ldm_flux", "Flux.1 / Z-Image / Lumina 2 / Chroma / HiDream / SD3（ae，z 16；用户：ae.safetensors）", _ldm_engine(16), None, (), IMG, None),
    ("ldm_flux2", "Flux 2 / Ideogram 4 / Lens / Ernie-Image（batch_norm_latent，z 32 → latent 128）", _ldm_kl(32, bn=True), None, (), IMG, None),
    ("ldm_x4", "SD x4 upscaler（ch_mult [1,2,4]，4x）", _ldm_kl(4, (1, 2, 4)), None, (), IMG1, None),
    ("svd", "SVD img2vid（VideoDecoder：batch 当时间轴）", _svd(), None, (), [(1024, 576, 1, 14)], None),
    ("hunyuan_video", "HunyuanVideo 1.0 / Kandinsky 5 视频（LDM conv3d，CarriedConv3d）", _hunyuan_video(), None, (), [(1344, 768, 1), (848, 480, 73)], None),
    ("hy_image21", "HunyuanImage 2.1（32x）", _hy_image21(), None, (), IMG, None),
    ("hy_refiner", "vae_refiner refiner_vae=False（5D decoder.conv_in.weight，ComfyUI #10150；不是官方 Refiner 文件）", _hy_refiner(False), None, (), IMG1, None),
    ("hy_video15", "HunyuanVideo 1.5 / HunyuanImage 2.1 Refiner 官方文件（vae_refiner，refiner_vae=True，16x，RMS norm）", _hy_refiner(True), None, (), [(1344, 768, 1), (1280, 720, 121)], None),
    ("wan21", "Wan 2.1 / Qwen-Image / Krea 2 / Anima / Cosmos Predict 2 / JoyImage（用户：qwen_image_vae）", _wan21(), None, (), IMG + [(832, 480, 81)], None),
    ("wan22", "Wan 2.2 5B（vae2_2，48 ch，16x）", _wan22(), None, (), IMG + [(1280, 704, 121)], None),
    ("mochi", "Mochi", _mochi(), None, (), [(848, 480, 85)], None),
    ("ltxv", "LTX-Video 0.9.0（comfy_has_chunked_io）", _ltxv(0), None, (), [(768, 512, 97)], None),
    ("ltxv2", "LTX-Video 0.9.5+ / LTX 2（version 2）", _ltxv(2), None, (), [(768, 512, 97)], None),
    ("cogvideox", "CogVideoX", _cogvideo(), None, (), [(720, 480, 49)], None),
    ("cosmos", "Cosmos 1.0（CV8x8x8）", _cosmos(), None, (), [(1280, 704, 121)], None),
    ("seedvr2", "SeedVR2（handles_tiling）", _seedvr2(), None, (), [(1920, 1080, 1)], None),
    ("mage", "Mage-VAE（Mage Flow）", _mage(), [("dconv_encoder.", "student.dconv_encoder."), ("decoder_model.", "pipeline.")], (), IMG1, None),
    ("taesd", "TAESD（SD / SDXL 预览）", _taesd(4), None, (), IMG, None),
    ("taef1", "TAEF1 / TAESD3（16 ch）", _taesd(16), None, (), IMG1, None),
    ("taef2", "TAEF2（Flux 2，128 ch）", _taesd(128), None, (), IMG1, {"tae_latent_channels": 128}),
    ("taehv", "TAEHV / lighttaew2.1（Wan 2.1 / HunyuanVideo 预览）", _taehv(16), None, (), [(832, 480, 81)], None),
    ("taew22", "TAEW2.2（48 ch）", _taehv(48), None, (), [(1280, 704, 121)], None),
    ("stage_a", "Stable Cascade Stage A（VQGAN，4x）", _stage_a(), None, (), IMG1, None),
    ("stage_c_prev", "Stable Cascade Stage C previewer", _previewer(), None, (), [(1024, 1024, 1)], None),
    ("minimax_video", "MiniMax H3 视频（comfy_has_chunked_io，内部分块）", _minimax_video(), None, (), [(832, 480, 73)], None),
    ("minimax_image", "MiniMax-H3-Image（单帧 2D ViT decoder，16 头）", _minimax_image("MiniMaxH3ImageVAE"), None, (), IMG, None),
    ("ph3_image", "p-h3-image（单帧 2D ViT decoder，32 头）", _minimax_image("PH3ImageVAE"), None, (), IMG, None),
    ("ltx_diffusion", "LTX 扩散 decoder（na_diffusion_decoder.CausalDiffusionVAE；sd.py 注释写「LTX 2.4」，F 组核实）", _ltx_diffusion(), None, (),
     [(768, 512, 97)], None),
    ("relayflash", "RelayVSR Flash x2（wan.model_vsr.RelayFlash，comfy_has_chunked_io，Wan 2.2 latent）", _relayflash(2), None, (),
     [(1280, 704, 1)], {"config": '{"upscale_factor": 2}'}),
    ("pixel", "像素空间（Chroma Radiance / Z-Image pixel / PixelDiT / HiDream O1）", _pixel(), None, (), IMG1, None),
]

# built by comfy.sd.VAE but not traced here (audio, 3D shapes, splats): the path only, from sd.py's attributes
UNTRACED = [
    ("oobleck", "Stable Audio 1（AudioOobleckVAE）", dict(latent_dim=1)),
    ("sa3", "Stable Audio 3", dict(latent_dim=1)),
    ("mmaudio", "MMAudio", dict(latent_dim=1)),
    ("ace", "ACE-Step 音频（MusicDCAE，[B, 8, 16, T]）", dict(latent_dim=2, extra_1d_channel=16, upscale_ratio=4096)),
    ("ltx_audio", "LTX 2 音频（AudioVAE）", dict(latent_dim=2, extra_1d_channel=16, upscale_ratio=4096)),
    ("minimax_audio", "MiniMax H3 音频（[B, 32, 2, T]）", dict(latent_dim=2, extra_1d_channel=None, upscale_ratio=800)),
    ("hunyuan3d", "Hunyuan3D 2.0 / 2.1（ShapeVAE）", dict(latent_dim=1)),
    ("triposplat", "TripoSplat（OctreeGaussianDecoder，VAE.decode 直接报错）", dict(latent_dim=1)),
    ("minimax_music3", "MiniMax Music3 DAV（音频，[B, 128, T]）", dict(latent_dim=1, upscale_ratio=512)),
    ("trellis2_shape", "Trellis2 形状 VAE（稀疏 3D）", dict(latent_dim=1)),
    ("trellis2_texture", "Trellis2 纹理 VAE（稀疏 3D）", dict(latent_dim=1)),
]


@contextlib.contextmanager
def _patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


# keys real files carry that the module does not save (non-persistent buffers sd.py detects the kind by)
EXTRA_KEYS = {"cosmos": {"decoder.unpatcher3d.wavelets": (8,)}}


class _Gfx1151:
    gcnArchName = "gfx1151"   # CT 700 (Strix Halo); v0.46.3 pytorch_attention_enabled_vae reads it when is_amd()


@contextlib.contextmanager
def _as_ct700():
    """is_amd() true and the GPU a gfx1151, as on CT 700 (no GPU needed)."""
    with _patched(mm, "is_amd", lambda: True), _patched(torch.cuda, "get_device_properties", lambda dev=None: _Gfx1151()):
        yield


def build(kind):
    name, used, builder, rename, drop, cases, meta = kind
    sd = _meta_sd(builder, rename, drop)
    for k, shape in EXTRA_KEYS.get(name, {}).items():
        sd[k] = torch.empty(shape, device="meta")
    try:
        with _as_ct700(), torch.device("meta"):
            v = comfy.sd.VAE(sd=sd, dtype=torch.bfloat16, metadata=meta)
    except RuntimeError as e:
        if "meta" not in str(e):
            raise
        # an __init__ reading tensor values: build it on the CPU (zeros), then move the model to the meta device
        sd = {k: torch.zeros(t.shape, dtype=t.dtype) for k, t in sd.items()}
        with _as_ct700():
            v = comfy.sd.VAE(sd=sd, dtype=torch.bfloat16, metadata=meta)
        del sd
        v.first_stage_model.to("meta")
    v.device = v.output_device = torch.device("meta")
    return v


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def structure(fsm):
    c = {}

    def add(k, n=1):
        c[k] = c.get(k, 0) + n

    for _, m in fsm.named_modules():
        t = type(m).__name__
        if isinstance(m, torch.nn.GroupNorm):
            add("GroupNorm")
        elif isinstance(m, torch.nn.LayerNorm):
            add("LayerNorm")
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            add("BatchNorm")
        elif "RMS" in t:
            add("RMSNorm")
        elif "PixelNorm" in t:
            add("PixelNorm")
        elif isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.ConvTranspose3d):
            add("ConvT")
        elif isinstance(m, torch.nn.Conv3d):
            add("Conv3d")
        elif isinstance(m, torch.nn.Conv2d):
            add("Conv2d")
        elif isinstance(m, torch.nn.Conv1d):
            add("Conv1d")
        elif isinstance(m, torch.nn.Linear):
            add("Linear")
        if "optimized_attention" in m.__dict__:
            add("attn(known)" if m.__dict__["optimized_attention"] in vae_ops.known_attention() else "attn(other)")
        elif "attn" in t.lower() or "attention" in t.lower():
            add("attn-like:" + t)
    return c


class _SimMeter:
    """vae._PEAK_METER's meter on the simulator, as vae._CudaPeak: the cache emptied at start(), the allocated peak
    above what was allocated then, and frag(): how much more was reserved at the peak."""

    def __init__(self, sim, tracer):
        self.sim, self.tracer, self.base, self.base_res = sim, tracer, 0, 0

    def start(self):
        self.tracer.poll()
        self.sim.empty_cache()
        self.sim.reset_peak()
        self.base, self.base_res = self.sim.allocated, self.sim.reserved

    def peak(self):
        self.tracer.poll()
        return self.sim.peak_allocated - self.base

    def frag(self):
        return max(0, self.sim.peak_reserved - self.base_res - self.peak())


def path(v, shape):
    lat = torch.empty(shape, device="meta", dtype=v.vae_dtype)
    with _patched(mvae, "_PEAK_METER", [lambda dev: object()]):   # CT 700 has allocator statistics; the meta device none
        reason = mvae._native_reason(v, lat)
    if reason is not None:
        return "native", reason
    if v.latent_dim == 2 and lat.ndim == 5:
        lat = lat[:, :, 0]
    why = []
    for a in mvae.STRIPE_ADAPTERS:
        bound, w = a.match(v, lat, {})
        if bound is not None:
            return "layer1", bound.name
        why.append(w)
    return "layer2", "; ".join(why)


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------

def latent_shape(v, w, h, frames, batch=1):
    r = v.spacial_compression_decode()
    lh, lw = round(h / r), round(w / r)
    if v.latent_dim == 3:
        dr = v.downscale_ratio
        t = dr[0](frames) if isinstance(dr, tuple) and callable(dr[0]) else 1
        return (batch, v.latent_channels, max(1, t), lh, lw)
    return (batch, v.latent_channels, lh, lw)


def trace(v, shape, layer, ws=GIB):
    """(alloc, reserved) increase of one decode of a meta latent `shape` (bytes), largest single tensor, output shape."""
    fsm = v.first_stage_model
    sim = CappedSim(CAP)
    sim.tag = "weights"
    for p in itertools.chain(fsm.parameters(), fsm.buffers()):
        sim.malloc(p.numel() * p.element_size())
    sim.empty_cache()
    sim.reset_peak()
    base_alloc, base_res = sim.allocated, sim.reserved
    sim.base = base_res
    lat = torch.empty(shape, device="meta", dtype=v.vae_dtype)
    tracer = alloc_sim.make_tracer(sim)
    tracer.static.update(t.untyped_storage()._cdata for t in itertools.chain(fsm.parameters(), fsm.buffers(), [lat]))
    biggest = [0]

    def soft_empty_cache(force=False):
        tracer.poll()
        sim.empty_cache()

    stats = vae_ops.OpStats()
    out_shape = None
    mf_est = mf_parts = None
    with contextlib.ExitStack() as es:
        es.enter_context(_patched(vae_ops, "slow_dilated3d", lambda x: True))
        es.enter_context(_patched(mm, "soft_empty_cache", soft_empty_cache))
        es.enter_context(_patched(mm, "get_free_memory", lambda dev=None, torch_free_too=False: (FREE_FOR_NATIVE, FREE_FOR_NATIVE) if torch_free_too
                                  else FREE_FOR_NATIVE))
        es.enter_context(_patched(mm, "get_total_memory", lambda dev=None, torch_total_too=False: (TOTAL, TOTAL) if torch_total_too else TOTAL))
        es.enter_context(_patched(mm, "intermediate_device", lambda: torch.device("meta")))   # --gpu-only: intermediates stay on the device
        orig_cpu = torch.Tensor.cpu

        def host_copy(t, *args, **kwargs):
            # a copy to host memory (CogVideoX keeps decoded chunks there): not device memory, so not tracked
            if not t.is_meta:
                return orig_cpu(t, *args, **kwargs)
            from torch.utils._python_dispatch import _disable_current_modes
            with _disable_current_modes():
                h = torch.empty(t.shape, dtype=t.dtype, device="meta")
            tracer.static.add(h.untyped_storage()._cdata)
            return h
        es.enter_context(_patched(torch.Tensor, "cpu", host_copy))
        es.enter_context(torch.inference_mode())
        es.enter_context(tracer)
        if layer == 2 and mvae._measured(v, lat):
            # H1: Monoload's own estimate (multi-frame, or a decoder vae_video always measures: LTX), its probe measured on this simulator (before the decode's OpChunking)
            mvae._PROBES.pop(fsm, None)
            with _patched(mvae, "_PEAK_METER", [lambda dev: _SimMeter(sim, tracer)]):
                probe = mvae._probe_mf(v, lat, {}, ws)
            mf_parts = mvae.estimate(v, lat, ws, probe)
            mf_est = mf_parts["total"]
            mf_parts = dict(mf_parts, slope=probe.mf_slope, intercept=probe.mf_intercept, frag=probe.mf_frag)
            soft_empty_cache()
            sim.reset_peak()
        if layer == 2:
            es.enter_context(vae_ops.OpChunking(fsm, ws, stats))
        n = shape[0] if layer == 0 else 1      # native decodes the batch at once (memory permitting), layer 2 one sample at a time
        for i in range(0, shape[0], n):
            z = torch.empty((n,) + tuple(shape[1:]), device="meta", dtype=v.vae_dtype)   # the latent copied to the device
            if getattr(fsm, "comfy_has_chunked_io", False):
                pix = torch.empty(fsm.decode_output_shape(z.shape), device="meta", dtype=torch.float32)
                fsm.decode(z, output_buffer=pix)
                out_shape = tuple(pix.shape)
            elif layer == 2 and mvae._output_shape(v, z) is not None:   # H3: pieces straight into the buffer
                pix = torch.empty(mvae._output_shape(v, z), device="meta", dtype=torch.float32)
                mvae._decode_sample(v, z, {}, pix)
                out_shape = tuple(pix.shape)
            else:
                out = fsm.decode(z)
                biggest[0] = max(biggest[0], out.numel() * out.element_size())
                pix = torch.empty(out.shape, device="meta", dtype=torch.float32)
                pix.copy_(out)
                out_shape = tuple(out.shape)
                del out
            del z, pix
        tracer.poll()
    big = max([b for b, _ in sim.peak_blocks if _ != "weights"] or [0])
    return {"alloc": sim.peak_allocated - base_alloc, "reserved": sim.peak_reserved - base_res, "largest_block": big,
            "out_shape": out_shape, "cache_flushes": sim.cache_flushes, "oom": sim.ooms, "conv_chunked": stats.conv_chunked, "attn_calls": stats.attn_calls,
            "attn_unmanaged": list(stats.attn_unmanaged), "mf_estimate": mf_est,
            "mf_parts": mf_parts}


L1_SIM = {"ldm_sdxl": "sdxl", "ldm_sd": "sdxl", "ldm_flux": "flux", "ldm_flux2": "flux2", "wan21": "qwen", "wan22": "wan22", "hy_image21": "hy21"}


def layer1_trace(name, w, h):
    model = L1_SIM.get(name)
    if model is None:
        return None
    i = alloc_sim.decode_trace(w, h, "bf16", None, layer=1, model=model)
    return {"reserved": i["reserved"], "alloc": i["alloc"], "estimate": i["estimate"], "rows": i["rows"], "stripes": i["stripes"]}


def native_estimate(v, shape):
    try:
        return float(v.memory_used_decode(shape, v.vae_dtype))
    except Exception as e:
        return "error: {}".format(e)


# ---------------------------------------------------------------------------

def gib(x):
    return "{:.2f}".format(x / G) if isinstance(x, (int, float)) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated kind names")
    ap.add_argument("--no-trace", action="store_true", help="structure and path only")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    report = []
    for kind in KINDS:
        name, used = kind[0], kind[1]
        if only and name not in only:
            continue
        entry = {"name": name, "used_by": used, "cases": []}
        report.append(entry)
        print("=" * 100)
        print("{}  —  {}".format(name, used))
        try:
            v = build(kind)
        except Exception as e:
            entry["error"] = "build: {}: {}".format(type(e).__name__, e)
            print("   BUILD FAILED:", entry["error"])
            continue
        fsm = v.first_stage_model
        if fsm is None:
            entry["error"] = "comfy.sd.VAE did not recognize the state dict"
            print("   ", entry["error"])
            continue
        dec = getattr(fsm, "decoder", None)
        entry.update(model=type(fsm).__name__, decoder=type(dec).__name__ if isinstance(dec, torch.nn.Module) else "-",
                     latent_dim=v.latent_dim, latent_channels=v.latent_channels, spatial=v.spacial_compression_decode(),
                     chunked_io=bool(getattr(fsm, "comfy_has_chunked_io", False)), handles_tiling=bool(v.handles_tiling),
                     structure=structure(fsm), params=sum(p.numel() for p in fsm.parameters()))
        print("   model {} / decoder {} | latent_dim {} ch {} spatial x{} | chunked_io {} handles_tiling {} | {:.0f} M params".format(
            entry["model"], entry["decoder"], v.latent_dim, v.latent_channels, entry["spatial"], entry["chunked_io"], entry["handles_tiling"],
            entry["params"] / 1e6))
        print("   structure:", ", ".join("{} {}".format(k, n) for k, n in sorted(entry["structure"].items())))
        for case in kind[5]:
            w, h, frames = case[:3]
            batch = case[3] if len(case) > 3 else 1
            shape = latent_shape(v, w, h, frames, batch)
            c = {"case": "{}x{}{}{}".format(w, h, "x{}f".format(frames) if frames > 1 else "", " batch {}".format(batch) if batch > 1 else ""),
                 "latent": list(shape)}
            entry["cases"].append(c)
            c["path"], c["why"] = path(v, shape)
            c["native_estimate"] = native_estimate(v, shape)
            print("   -- {} latent {}: path {} ({})".format(c["case"], list(shape), c["path"], c["why"][:160]))
            if a.no_trace:
                continue
            for layer, key in ((0, "native"), (2, "layer2")):
                t0 = time.time()
                try:
                    c[key] = trace(v, shape, layer)
                    t = c[key]
                    print("      {:7s} sim reserved {:>7s} alloc {:>7s} GiB, largest block {:>6s} GiB{}, out {}{}  [{:.0f} s]".format(
                        key, gib(t["reserved"]), gib(t["alloc"]), gib(t["largest_block"]),
                        " (OOM {}x beyond {:.0f} GiB: native goes tiled)".format(t["oom"], CAP / G) if t["oom"] else
                        " (cache released {}x at the {:.0f} GiB limit)".format(t["cache_flushes"], CAP / G) if t["cache_flushes"] else "", t["out_shape"],
                        ", {} convs chunked, {} attn calls{}".format(t["conv_chunked"], t["attn_calls"],
                                                                     ", unmanaged attn " + ", ".join(t["attn_unmanaged"][:3]) if t["attn_unmanaged"] else "")
                        if layer == 2 else "", time.time() - t0))
                    if t.get("mf_estimate") is not None:
                        mp = t.get("mf_parts") or {}
                        print("      layer2  multi-frame estimate {} GiB (probe on the simulator: slope {:.0f} B/px, intercept {} GiB, frag {} GiB; "
                              "activation {}, workspace {}, output {} GiB){}".format(
                                  gib(t["mf_estimate"]), mp.get("slope") or 0, gib(mp.get("intercept") or 0), gib(mp.get("frag") or 0), gib(mp.get("activation", 0)),
                                  gib(mp.get("workspace", 0)), gib(mp.get("output", 0)),
                                  "" if t["mf_estimate"] >= t["reserved"] else "  ** NOT AN UPPER BOUND **"))
                except Exception as e:
                    c[key] = {"error": "{}: {}".format(type(e).__name__, str(e)[:300])}
                    print("      {:7s} TRACE FAILED: {}".format(key, c[key]["error"]))
                    if os.environ.get("INVENTORY_TB"):
                        traceback.print_exc()
            if frames == 1 and batch == 1:
                try:
                    l1 = layer1_trace(name, w, h)
                except Exception as e:
                    l1 = {"error": str(e)}
                if l1:
                    c["layer1"] = l1
                    print("      layer1  sim reserved {:>7s} GiB (estimate {} GiB, {} x {} rows)".format(gib(l1.get("reserved")), gib(l1.get("estimate")),
                                                                                                     l1.get("stripes"), l1.get("rows")))
            print("      ComfyUI memory_used_decode (AMD): {} GiB".format(gib(c["native_estimate"])))
    for name, used, attrs in UNTRACED:
        if only and name not in only:
            continue

        class _V:
            pass
        v = _V()
        v.first_stage_model = torch.nn.Identity()
        v.latent_dim = attrs["latent_dim"]
        v.extra_1d_channel = attrs.get("extra_1d_channel")
        v.upscale_ratio = attrs.get("upscale_ratio", 8)
        shape = (1, 8, 16, 64) if v.latent_dim == 2 else (1, 64, 256)
        reason = mvae._native_reason(v, torch.empty(shape, device="meta"))
        p = "native" if reason else "managed (layer 2)"
        report.append({"name": name, "used_by": used, "untraced": True, "latent_dim": v.latent_dim, "extra_1d_channel": v.extra_1d_channel,
                       "cases": [{"case": "latent {}".format(list(shape)), "path": p, "why": reason or ""}]})
        print("=" * 100)
        print("{}  —  {}: latent_dim {}, extra_1d_channel {} -> {} {}".format(name, used, v.latent_dim, v.extra_1d_channel, p, reason or ""))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # skip interpreter teardown (the meta-built patchers' destructors print noise there)
