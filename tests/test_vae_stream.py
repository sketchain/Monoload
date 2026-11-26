"""Group D: the streamed multi-frame decodes of Mochi (monoload/vae_mochi.py) and Cosmos 1.0 (monoload/vae_cosmos.py)
against the whole-video native decode (DESIGN §9.40).

Random weights at the real sizes (sd.py's configs), no model files. On a CUDA / HIP device when there is one (the
real machine), else the CPU.

  1. Mochi VideoVAE, latent [1, 12, 4, 6, 8] (19 frames, 48 x 64): the pieces (one frame each, 19) concatenated ==
     fsm.decode in fp32 (relative max error <= 1e-5), also under OpChunking with a 1 MiB workspace (every conv in
     row blocks, the real layer-2 path); no instance forward left afterwards;
  2. Cosmos CausalContinuousVideoTokenizer (CV8x8x8), latent [1, 16, 3, 6, 8] (17 frames, 48 x 64): the same (5
     pieces: 1 + 4 x 4 frames);
  3. bf16: streamed and native both against the fp32 decode: the streamed error at most 2x native's (+1e-3);
  4. what is not streamed decodes whole (one piece == decode): a user's instance forward on a decoder module, Cosmos
     with one latent frame, a model of another class.

    python tests/test_vae_stream.py
"""

import torch

from common import check, finish
import comfy.ldm.cosmos.vae as cvae
import comfy.ldm.genmo.vae.model as genmo
from monoload import vae_cosmos, vae_mochi, vae_ops
from test_vae import init_random

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

COSMOS_CFG = {'z_channels': 16, 'latent_channels': 16, 'z_factor': 1, 'resolution': 1024, 'in_channels': 3, 'out_channels': 3,
              'channels': 128, 'channels_mult': [2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [32], 'dropout': 0.0,
              'patch_size': 4, 'num_groups': 1, 'temporal_compression': 8, 'spacial_compression': 8}


def overrides(model):
    return [n for n, m in model.named_modules() if "forward" in m.__dict__ or "_conv_forward" in m.__dict__]


def rel(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12))


def streamed(mod, fsm, z, ws=None):
    pieces = []
    if ws is None:
        for p in mod.chunks(fsm, z):
            pieces.append(p.clone())
    else:
        with vae_ops.OpChunking(fsm, ws, vae_ops.OpStats()):
            for p in mod.chunks(fsm, z):
                pieces.append(p.clone())
    return pieces


def run(name, mod, fsm, z, frames, n_pieces):
    fsm.to(DEV, torch.float32)
    z = z.to(DEV, torch.float32)
    with torch.inference_mode():
        check("{}: recognised".format(name), mod.structure(fsm)[0] is None, str(mod.structure(fsm)[0]))
        ref = fsm.decode(z)
        check("{}: native frames {}".format(name, frames), ref.shape[2] == frames, str(tuple(ref.shape)))
        for ws, label in ((None, "plain"), (1 << 20, "OpChunking 1 MiB")):
            ps = streamed(mod, fsm, z, ws)
            out = torch.cat(ps, dim=2)
            check("{} ({}): {} pieces, {} frames".format(name, label, len(ps), out.shape[2]),
                  len(ps) == n_pieces and out.shape == ref.shape, "{} pieces, {}".format(len(ps), tuple(out.shape)))
            if out.shape == ref.shape:
                e = rel(out, ref)
                check("{} ({}) == native fp32 (rel {:.2e})".format(name, label, e), e <= 1e-5)
            check("{} ({}): nothing left on the modules".format(name, label), not overrides(fsm), ", ".join(overrides(fsm)[:3]))
        # bf16 against the fp32 truth
        fb = fsm.to(torch.bfloat16)
        zb = z.to(torch.bfloat16)
        nat = fb.decode(zb)
        st = torch.cat(streamed(mod, fb, zb), dim=2)
        en, es = rel(nat, ref), rel(st, ref)
        check("{} bf16: streamed vs fp32 {:.2e}, native vs fp32 {:.2e}".format(name, es, en), es <= 2 * en + 1e-3)
        fsm.to(torch.float32)   # bf16-rounded weights from here on: a new reference
        ref = fsm.decode(z)
        # a user's hook on a decoder module: whole decode
        target = next(m for m in fsm.decoder.modules() if isinstance(m, torch.nn.Conv3d))
        target.forward = target.forward
        try:
            ps = streamed(mod, fsm, z)
            check("{}: a user's instance forward -> one whole piece".format(name), len(ps) == 1 and rel(ps[0], ref) == 0.0)
        finally:
            del target.forward
    fsm.to("cpu")


def main():
    torch.manual_seed(0)
    mochi = init_random(genmo.VideoVAE())
    z = torch.randn(1, 12, 4, 6, 8)
    run("Mochi", vae_mochi, mochi, z, 19, 19)
    del mochi

    cosmos = init_random(cvae.CausalContinuousVideoTokenizer(**COSMOS_CFG))
    with torch.no_grad():   # latent std as in a real file (init_random makes it ~0.05)
        cosmos.latent_std.uniform_(0.5, 1.5)
    z = torch.randn(1, 16, 3, 6, 8)
    run("Cosmos", vae_cosmos, cosmos, z, 17, 5)
    with torch.inference_mode():
        cosmos.to(DEV, torch.float32)
        z1 = torch.randn(1, 16, 1, 6, 8, device=DEV)
        ps = list(vae_cosmos.chunks(cosmos, z1))
        check("Cosmos one latent frame -> one whole piece", len(ps) == 1 and rel(ps[0], cosmos.decode(z1)) == 0.0)
        other = torch.nn.Conv3d(1, 1, 1)
        check("another class is not recognised", vae_mochi.structure(other)[0] is not None and vae_cosmos.structure(other)[0] is not None)
    finish()


if __name__ == "__main__":
    main()
