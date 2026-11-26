"""CogVideoX VAE (group C) on the real machine, small sizes:

  1. chunks() == the decoder's own decode, bit for bit (layer 2 streams the same calls);
  2. frames out of an even latent T: 4t (sd.py's upscale_ratio says 4t - 3), from the decoder itself;
  3. layer 1 (decode_rows) against chunks() in the VAE dtype (informative; the fp32 check is 4);
  4. the layer-1 self-test (fp32 copy, rel err <= SELFTEST_TOL).

  python tests/check_cogvideo.py --vae cogvideox_vae_bf16.safetensors [--res 480x320] [--latent-frames 13,12]
"""

import argparse

import common  # noqa: F401  (ComfyUI set up)
from common import check, finish, torch
import nodes
import comfy.model_management as mm

from monoload import vae_cogvideo as vc
from monoload import vae_engine as eng
from monoload.vae_ops import MIB, OpChunking, OpStats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", required=True)
    p.add_argument("--res", default="480x320", help="output WxH")
    p.add_argument("--latent-frames", default="13,12", help="latent T values (odd and even)")
    p.add_argument("--rows", type=int, default=64, help="layer-1 slice rows at the output resolution")
    p.add_argument("--workspace-mib", type=int, default=256)
    a = p.parse_args()
    vae = nodes.VAELoader().load_vae(a.vae)[0]
    fsm = vae.first_stage_model
    reason = vc.layer1_structure(fsm)
    check("CogVideoX layer-1 structure", not reason, reason or "split {}".format(vc.structure(fsm)[1]))
    if reason:
        finish()
    mm.load_models_gpu([vae.patcher])
    dev, dt = vae.device, vae.vae_dtype
    w, h = (int(v) for v in a.res.split("x"))
    ws = a.workspace_mib * MIB
    out = {}
    with torch.inference_mode():
        for t in (int(v) for v in a.latent_frames.split(",")):
            g = torch.Generator().manual_seed(t)
            z = torch.randn((1, fsm.decoder.conv_in.conv.in_channels, t, h // 8, w // 8), generator=g).to(dev, dt)
            nat = fsm.decode(z)
            mm.soft_empty_cache()
            ch = torch.cat(list(vc.chunks(fsm, z)), dim=2)
            check("t={}: chunks() == decode() bit for bit".format(t), torch.equal(nat, ch),
                  "max|diff| {:.3g}".format(float((nat.float() - ch.float()).abs().max())))
            want = 4 * t if t % 2 == 0 else 4 * t - 3
            check("t={}: frames {} (expected {}), output_frames {}".format(t, nat.shape[2], want, vc.output_frames(fsm, t)),
                  nat.shape[2] == want == vc.output_frames(fsm, t))
            del nat
            mm.soft_empty_cache()
            dest = torch.empty_like(ch)
            with OpChunking(fsm.decoder, ws, OpStats()):
                vc.decode_rows(fsm, z, dest, a.rows, ws)
            d = (dest.float() - ch.float()).abs()
            out[t] = (float(d.max()), float(d.mean()))
            check("t={}: layer 1 vs chunks() in {} finite".format(t, dt), bool(torch.isfinite(dest).all()),
                  "max|diff| {:.3g}, mean {:.3g}".format(*out[t]))
            del ch, dest, z, d
            mm.soft_empty_cache()
    ok, detail = vc.CogStripe(fsm).self_test_run(vae)
    check("layer-1 self-test (fp32, tol {:g})".format(eng.SELFTEST_TOL), ok, detail)
    finish({"layer1_vs_chunks": out})


if __name__ == "__main__":
    main()
