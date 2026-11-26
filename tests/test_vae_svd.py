"""SVD's VideoDecoder in layer 2, the whole batch at once (DESIGN §9.38, PLAN-video group E).

SVD's decoder takes the batch as its time axis (temporal_ae.VideoDecoder: timesteps = the batch size) and mixes the
frames. Native ComfyUI splits the batch by free memory and only mixes the frames inside each part; Monoload decodes the
whole batch in one go under OpChunking, its memory bounded by the measured probe (H1's _probe_mf, over the whole
batch). Random weights at SVD's full width (sd.py's structure), no model files. Needs a CUDA / HIP device (the real
machine): COMFY_ARGS defaults to "" here, not --cpu.

  1. 14 frames at a 40 x 48 latent: layer 2, result == native's whole-batch decode (fp32, <= 1e-4 relative), and not
     the frame-by-frame one; the measured peak of the call stays under the recorded estimate; no tiled fallback;
  2. a second decode of the same shape reuses the probe (no new probe time), a different batch size gets its own;
  3. without allocator statistics (the meter returns None) SVD stays native, with the batch reason.

    python tests/test_vae_svd.py
"""

import os

os.environ.setdefault("COMFY_ARGS", "")

import torch  # noqa: E402

from common import check, finish  # noqa: E402
import comfy.sd  # noqa: E402
from monoload import vae as mvae  # noqa: E402
from test_vae import Spy, native_decode, rel_err, svd_vae  # noqa: E402


def peak_of(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    return out, torch.cuda.max_memory_allocated() - base


def main():
    if not torch.cuda.is_available():
        check("a CUDA / HIP device (this test runs on the real machine)", False)
        return finish()
    if not mvae.is_installed():
        check("vae.install() on this ComfyUI", mvae.install())
    mvae.set_workspace(64 << 20)
    v = svd_vae()
    g = torch.Generator().manual_seed(7)
    lat = torch.randn(14, 4, 40, 48, generator=g)

    with Spy() as spy:
        out, peak = peak_of(lambda: comfy.sd.VAE.decode(v, lat))
    last = mvae.last_decode()
    est = last.get("estimate", {}).get("total", 0)
    ref = native_decode(v, lat)
    alone = torch.cat([native_decode(v, lat[i:i + 1]) for i in range(lat.shape[0])])
    e = rel_err(out, ref)
    e_alone = rel_err(alone, ref)
    check("SVD 14 frames: layer 2 ({}), == native whole batch (rel {:.2g}), not frame by frame (rel {:.2g}); peak {:.0f} MiB <= estimate "
          "{:.0f} MiB; no tiled".format(last.get("strategy"), e, e_alone, peak / 2 ** 20, est / 2 ** 20),
          last.get("strategy") == "layer2" and e <= 1e-4 and e_alone > 1e-3 and 0 < peak <= est and spy.tiled == 0)
    probe_s = last.get("probe")

    comfy.sd.VAE.decode(v, lat)
    key = mvae._probe_key(v, lat)
    same = mvae._PROBES.get(v.first_stage_model, {}).get(key)
    lat7 = lat[:7]
    out7 = comfy.sd.VAE.decode(v, lat7)
    key7 = mvae._probe_key(v, lat7)
    check("probe cached per batch size ({} / {}), 7 frames decoded too (rel {:.2g} vs native)".format(key, key7, rel_err(out7, native_decode(v, lat7))),
          probe_s and same is not None and key != key7 and key7 in mvae._PROBES.get(v.first_stage_model, {})
          and rel_err(out7, native_decode(v, lat7)) <= 1e-4)

    meter = mvae._PEAK_METER[0]
    mvae._PEAK_METER[0] = lambda dev: None
    try:
        out = comfy.sd.VAE.decode(v, lat[:3])
    finally:
        mvae._PEAK_METER[0] = meter
    last = mvae.last_decode()
    check("no allocator statistics: SVD left native ({}), result == native".format(last.get("reason", "")[:90]),
          last.get("strategy") == "native" and "batch" in last.get("reason", "") and torch.equal(out, native_decode(v, lat[:3])))
    finish()


if __name__ == "__main__":
    main()
