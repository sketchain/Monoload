"""Phase 4a probe: decodes the current code manages that it should not, or
manages differently from native. Real (small) decodes on the CPU with random
weights, through comfy.sd.VAE; reports, does not assert (it is an inventory
probe, not a test).

  1. SVD (VideoDecoder): its time mixing runs over the batch (timesteps =
     batch size); layer 2 decodes one sample at a time, so each frame is
     decoded alone. Native vs managed on a 4-frame batch.
  2. Audio VAEs with a 2D latent (latent_dim 2): ACE-Step (MusicDCAE,
     [B, 8, 16, T]) and MiniMax H3 audio ([B, 32, 2, T]). The design says
     audio stays native; _native_reason only looks at latent_dim. What the
     managed decode does with them (path, probe, result vs native).
  3. The pixel-space "VAE" (identity): managed, with a shape probe for nothing.
  4. Feasibility of layer 2 on multi-frame video latents (phase 4b): the
     decoder's own decode (with its temporal causal caches) under OpChunking
     with a tiny workspace (most convs in row blocks) vs without, on small
     random-weight Wan 2.1, Wan 2.2, HunyuanVideo 1.0, HunyuanVideo 1.5 and
     CogVideoX decoders, fp32.

    tests/docker_run.sh python tests/probe_vae_gaps.py
"""

import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import common  # noqa: E402,F401
import torch  # noqa: E402

import comfy.sd  # noqa: E402
from monoload import vae as mvae  # noqa: E402

mvae.install()


def init_random(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p[0].numel()))
            elif name.endswith("gamma") or name.endswith("weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
    return model


def native(v, latent):
    return mvae._ORIG["decode"](v, latent)


def managed(v, latent):
    out = comfy.sd.VAE.decode(v, latent)
    return out, mvae.last_decode()


def compare(label, v, latent):
    print("--", label, "latent", list(latent.shape))
    lat = torch.empty(latent.shape, device="meta")
    print("   _native_reason:", mvae._native_reason(v, lat))
    t0 = time.time()
    try:
        ref = native(v, latent)
    except Exception as e:
        print("   native decode failed: {}: {}".format(type(e).__name__, str(e)[:300]))
        return
    t1 = time.time()
    try:
        out, last = managed(v, latent)
    except Exception as e:
        print("   managed decode failed: {}: {}".format(type(e).__name__, str(e)[:300]))
        return
    t2 = time.time()
    print("   managed strategy: {}{}".format(last.get("strategy"), " ({})".format(last.get("layer1") or last.get("reason") or "")))
    print("   probe used: {}, estimate {}".format(last.get("probe"), (last.get("estimate") or {}).get("total")))
    if out.shape != ref.shape:
        print("   SHAPES DIFFER: managed {} native {}".format(tuple(out.shape), tuple(ref.shape)))
        return
    nan_m, nan_n = int(torch.isnan(out).sum()), int(torch.isnan(ref).sum())
    if nan_m or nan_n:
        print("   NaN: managed {} / native {} elements (random weights); comparing the rest".format(nan_m, nan_n))
        ok = ~(torch.isnan(out) | torch.isnan(ref))
        out, ref = out[ok], ref[ok]
    d = (out.float() - ref.float()).abs()
    scale = max(1e-6, ref.float().abs().max().item())
    print("   output {}: max|managed - native| {:.3g} (rel {:.3g}), mean {:.3g}; native {:.1f} s, managed {:.1f} s".format(
        tuple(out.shape), d.max().item(), d.max().item() / scale, d.mean().item(), t1 - t0, t2 - t1))
    return d.max().item() / scale


def svd():
    from comfy.ldm.models.autoencoder import AutoencodingEngine
    enc = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4],
           'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
    dec = dict(enc, video_kernel_size=[3, 1, 1], alpha=0.0)
    m = AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                           encoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Encoder", 'params': enc},
                           decoder_config={'target': "comfy.ldm.modules.temporal_ae.VideoDecoder", 'params': dec})
    init_random(m)
    v = comfy.sd.VAE(sd=m.state_dict(), dtype=torch.float32)
    del m
    lat = torch.randn(4, 4, 12, 16, generator=torch.Generator().manual_seed(0))
    rel = compare("SVD VideoDecoder, 4 frames as a batch", v, lat)
    # what one frame alone gives natively: the managed decode equals this when it decodes frame by frame
    ref1 = torch.cat([native(v, lat[i:i + 1]) for i in range(lat.shape[0])])
    out, _ = managed(v, lat)
    print("   managed vs native frame-by-frame: max|Δ| {:.3g} (so the managed decode = each frame decoded alone)".format(
        (out - ref1).abs().max().item()))
    return rel


def ace():
    import comfy.ldm.ace.vae.music_dcae_pipeline as ace_m
    m = init_random(ace_m.MusicDCAE(source_sample_rate=44100))
    v = comfy.sd.VAE(sd=m.state_dict(), dtype=torch.float32)
    del m
    print("   built: {} latent_dim {} extra_1d_channel {}".format(type(v.first_stage_model).__name__, v.latent_dim, v.extra_1d_channel))
    compare("ACE-Step MusicDCAE", v, 0.1 * torch.randn(1, 8, 16, 32, generator=torch.Generator().manual_seed(0)))


def minimax_audio():
    import comfy.ldm.minimax.audio_vae as mm_a
    m = init_random(mm_a.MiniMaxH3AudioVAE())
    v = comfy.sd.VAE(sd=m.state_dict(), dtype=torch.float32)
    del m
    print("   built: {} latent_dim {} extra_1d_channel {}".format(type(v.first_stage_model).__name__, v.latent_dim, v.extra_1d_channel))
    compare("MiniMax H3 audio", v, 0.1 * torch.randn(1, 32, 2, 40, generator=torch.Generator().manual_seed(0)))


def pixel():
    v = comfy.sd.VAE(sd={"pixel_space_vae": torch.tensor(1.0)}, dtype=torch.float32)
    compare("pixel space", v, torch.rand(1, 3, 64, 96) * 2 - 1)


def video_layer2():
    from comfy.ldm.models.autoencoder import AutoencoderKL, AutoencodingEngine
    import comfy.ldm.wan.vae as wan
    import comfy.ldm.wan.vae2_2 as wan22
    import comfy.ldm.cogvideo.vae as cog
    from monoload import vae_ops
    g = torch.Generator().manual_seed(0)
    hv15 = {"block_out_channels": [16, 32, 64, 64, 64], "in_channels": 3, "out_channels": 3, "num_res_blocks": 1, "ffactor_spatial": 16,
            "ffactor_temporal": 4, "downsample_match_channel": True, "upsample_match_channel": True, "z_channels": 32}
    cases = [
        ("Wan 2.1 (dim 16), 5 latent frames", lambda: wan.WanVAE(dim=16, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                                                                 temperal_downsample=[False, True, True], dropout=0.0), (1, 16, 5, 12, 16)),
        ("Wan 2.2 (dim 16 / dec 32), 4 latent frames", lambda: wan22.WanVAE(dim=16, dec_dim=32, z_dim=48, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                                                                            attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0),
         (1, 48, 4, 6, 8)),
        ("HunyuanVideo 1.0 (LDM conv3d, ch 32), 5 latent frames",
         lambda: AutoencoderKL(ddconfig={'double_z': True, 'z_channels': 16, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 32,
                                         'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0, 'conv3d': True,
                                         'time_compress': 4}, embed_dim=16), (1, 16, 5, 12, 16)),
        ("HunyuanVideo 1.5 (vae_refiner, small), 5 latent frames",
         lambda: AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.EmptyRegularizer"},
                                    encoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Encoder", 'params': hv15},
                                    decoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Decoder", 'params': hv15}), (1, 32, 5, 6, 8)),
        ("CogVideoX (full size), 5 latent frames", lambda: cog.AutoencoderKLCogVideoX(latent_channels=16), (1, 16, 5, 8, 10)),
    ]
    for label, build, shape in cases:
        print("--", label, list(shape))
        try:
            m = init_random(build()).eval()
            z = torch.randn(shape, generator=g)
            with torch.inference_mode():
                ref = m.decode(z)
                stats = vae_ops.OpStats()
                with vae_ops.OpChunking(m, 16 * 1024, stats):
                    out = m.decode(z)
            d = (out - ref).abs().max().item() / max(1e-6, ref.abs().max().item())
            print("   out {}: layer 2 vs native rel max|Δ| {:.2g}; {} of {} conv calls in row blocks ({} blocks), {} attention calls chunked{}".format(
                tuple(out.shape), d, stats.conv_chunked, stats.conv_calls, stats.conv_blocks, stats.attn_calls,
                ", unmanaged attention: " + ", ".join(stats.attn_unmanaged[:3]) if stats.attn_unmanaged else ""))
            del m
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("   FAILED: {}: {}".format(type(e).__name__, str(e)[:300]))


def main():
    for label, fn in (("1. SVD", svd), ("2a. ACE-Step audio", ace), ("2b. MiniMax H3 audio", minimax_audio), ("3. pixel space", pixel),
                      ("4. layer 2 on multi-frame video decoders", video_layer2)):
        print("=" * 90)
        print(label)
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("   PROBE FAILED: {}: {}".format(type(e).__name__, str(e)[:300]))
        common.free_all()
    sys.stdout.flush()
    os._exit(0)   # skip interpreter teardown (ComfyUI's patcher destructors print noise there)


if __name__ == "__main__":
    main()
