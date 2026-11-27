"""Layer 1 for HunyuanVideo 1.5's VAE and the HunyuanImage 2.1 Refiner file (monoload/vae_hunyuan15.py, DESIGN §9.38):
vae_refiner.Decoder with refiner_vae=True, multi-frame (time chunks, carries per stripe) and single frame.

No model files needed: ComfyUI's own AutoencodingEngine with vae_refiner.Decoder / Encoder (refiner_vae=True) with
small channels and random weights, through comfy.sd.VAE (it detects HunyuanVideo 1.5 from decoder.conv_in.conv.weight
with 32 input channels; its hard-coded block_out_channels are swapped for small ones while it builds), latent 32
channels at 1/16, 5D, fp32 on the CPU unless noted. The adapter is put in vae.STRIPE_ADAPTERS for the test (the
registration in vae.py is the adapter thread's).

  1. recognition: matched by structure (single frame for T = 1, time chunks otherwise); refiner_vae=False, a forward
     hook, vae_options, a 4D latent, a wrong channel count -> not matched, with the reason;
  2. single frame through the managed decode (the Refiner): layer 1 == native at stripe heights 7 / 40 / default, odd
     latent, 1x1, batch 2, a 16 KiB workspace (row blocks inside the stripes);
  3. multi-frame (bound.plan / bound.run: on the CPU vae._native_reason keeps T > 1 native, no peak meter): T = 2, 3,
     5, 6, 9 at stripe heights 7 / 40 / whole, odd latent, 16 KiB workspace == native decode (the carries per stripe
     and unit, the first chunk's first frame spatial only); the output frames are 4T - 3;
  4. the self-test passes (single and multi-frame cached apart); injected bugs (halos too small, the carries
     dropped between chunks) -> self-test fails, layer 2, == native;
  5. caching allocator (tests/alloc_sim.py, full size on the meta device): layer-1 reserved <= estimate for the
     Refiner (1344x768, 2048x2048, 3072x3072, bf16 / fp16 / fp32), and the estimate far below layer 2's; the
     HunyuanVideo 1.5 video plans (832x480x81, 1280x720x121) and the multi-frame self-test bound are printed, not
     checked: they are under the simulated reserved (DESIGN §9.41; a video takes layer 1 only past layer 2's room).

    python tests/test_vae_hunyuan15.py
"""

import torch

from common import check, finish
import comfy.sd
import comfy.ldm.hunyuan_video.vae_refiner as hyr
from comfy.ldm.models.autoencoder import AutoencodingEngine
from monoload import vae as mvae
from monoload import vae_engine as eng
from monoload import vae_ops
from monoload import vae_hunyuan15 as h15
from test_vae import Spy, init_random, managed_decode, native_decode

torch.manual_seed(0)
Z = 32
BOC = [32, 32, 64, 64, 64]     # reversed in the decoder: 64 (H/16, 2 x z: repeat 2), 64, 64, 32, 32 channels
DD = {"block_out_channels": BOC, "in_channels": 3, "out_channels": 3, "num_res_blocks": 2, "ffactor_spatial": 16,
      "ffactor_temporal": 4, "downsample_match_channel": True, "upsample_match_channel": True, "refiner_vae": True, "z_channels": Z}
_DEC, _ENC = hyr.Decoder, hyr.Encoder
G = float(1 << 30)


def hv15_model(**kw):
    dd = dict(DD, **kw)
    return init_random(AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.EmptyRegularizer"},
                                          encoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Encoder", 'params': dd},
                                          decoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Decoder", 'params': dd}))


def hv15_vae(dtype=None, model=None):
    """comfy.sd.VAE on a small HunyuanVideo 1.5 state dict (sd.py's block_out_channels swapped for BOC while it builds)."""
    m = model if model is not None else hv15_model()
    hyr.Decoder = lambda *a, **kw: _DEC(*a, **dict(kw, block_out_channels=BOC))
    hyr.Encoder = lambda *a, **kw: _ENC(*a, **dict(kw, block_out_channels=BOC))
    try:
        return comfy.sd.VAE(sd=m.state_dict(), dtype=dtype)
    finally:
        hyr.Decoder, hyr.Encoder = _DEC, _ENC


def lat_(h, w, t=1, b=1, seed=4):
    return torch.randn(b, Z, t, h, w, generator=torch.Generator().manual_seed(seed))


def err(a, b):
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().max())


# fp32 test model; on the GPU the summation order inside a conv differs with its row extent, so the stripes match the
# whole decode to 1e-4 (the self-test's tolerance), on the CPU to 1e-5
TOL = 1e-4 if torch.cuda.is_available() else 1e-5


def _on_model(fsm, lat):
    """lat on the device (and in the dtype) of the decoder's weights: fsm.decode called directly does not move it."""
    p = next(fsm.decoder.parameters())
    return lat.to(p.device, p.dtype)


class registered:
    """vae_hunyuan15 in vae.STRIPE_ADAPTERS for the duration."""

    def __enter__(self):
        self.had = h15 in mvae.STRIPE_ADAPTERS
        if not self.had:
            mvae.STRIPE_ADAPTERS.append(h15)
        return self

    def __exit__(self, *exc):
        if not self.had:
            mvae.STRIPE_ADAPTERS.remove(h15)


# ---------------------------------------------------------------------------
# 1. recognition
# ---------------------------------------------------------------------------

def recognition_tests(v):
    fsm = v.first_stage_model
    check("comfy.sd.VAE builds HunyuanVideo 1.5's decoder: vae_refiner.Decoder, refiner_vae=True, latent_dim 3",
          type(fsm.decoder) is _DEC and fsm.decoder.refiner_vae is True and v.latent_dim == 3)
    b1, why1 = h15.match(v, lat_(4, 4), {})
    b5, why5 = h15.match(v, lat_(4, 4, t=5), {})
    check("matched: T = 1 single frame ({}), T = 5 time chunks ({}); keys differ".format(b1 and b1.name, b5 and b5.name),
          b1 is not None and b1.single and b5 is not None and not b5.single and b1.key != b5.key, "{} / {}".format(why1, why5))
    chunks = b5.time_chunks((1, Z, 5, 4, 4))
    check("T = 5: chunks {} (2, 2, 1 latent frames; 5 + 8 + 4 output frames)".format(chunks),
          [(c.ct0, c.ct1, c.ot0, c.ot1) for c in chunks] == [(0, 2, 0, 5), (2, 4, 5, 13), (4, 5, 13, 17)])
    other = hv15_model(refiner_vae=False)
    shim = type("V", (), {"first_stage_model": other})()
    b, why = h15.match(shim, lat_(4, 4), {})
    check("refiner_vae=False: not matched ({})".format(why), b is None and "refiner_vae=False" in why)
    hk = fsm.decoder.up[1].block[0].conv1.register_forward_hook(lambda *a: None)
    try:
        b, why = h15.match(v, lat_(4, 4), {})
    finally:
        hk.remove()
    check("forward hook: not matched ({})".format(why), b is None and "hook" in why)
    for label, lat, opts, want in (("vae_options", lat_(4, 4), {"x": 1}, "vae_options"),
                                   ("4D latent", torch.randn(1, Z, 4, 4), {}, "dims"),
                                   ("16 channels", torch.randn(1, 16, 1, 4, 4), {}, "channels")):
        b, why = h15.match(v, lat, opts)
        check("{}: not matched ({})".format(label, why), b is None and want in why)


# ---------------------------------------------------------------------------
# 2. single frame (the Refiner), managed decode
# ---------------------------------------------------------------------------

def compare(label, v, latent, rows=None, tol=TOL):
    mvae.set_stripe_rows(rows)
    try:
        with Spy() as spy:
            raw = managed_decode(v, latent, raw=True)
    finally:
        mvae.set_stripe_rows(None)
    last = mvae.last_decode()
    ref = native_decode(v, latent, raw=True)
    e = err(raw, ref)
    check("{}: {} stripes of {} rows; max|Δ| {:.2g}".format(label, last.get("stripes"), last.get("rows"), e),
          last.get("strategy") == "layer1" and str(last.get("adapter", "")).startswith("HunyuanVideo 1.5") and e <= tol
          and raw.shape == ref.shape and spy.tiled == 0, "strategy {} note {}".format(last.get("strategy"), last.get("layer1")))
    return last


def single_tests(v):
    lat = lat_(6, 5)
    for rows in (7, 40, None):
        compare("single frame 6x5 latent (output 96x80), rows {}".format(rows or "default"), v, lat, rows=rows)
    compare("odd 7x3 latent, rows 9", v, lat_(7, 3, seed=5), rows=9)
    compare("1x1 latent, rows 5", v, lat_(1, 1, seed=6), rows=5)
    compare("batch 2, rows 16", v, lat_(6, 5, b=2, seed=7), rows=16)
    mvae.set_workspace(16 * 1024)
    try:
        last = compare("workspace 16 KiB, rows 32", v, lat, rows=32)
    finally:
        mvae.set_workspace(mvae.DEFAULT_WORKSPACE)
    check("  ... conv calls inside the stripes were split into row blocks ({})".format(last["stats"]["conv_chunked"]), last["stats"]["conv_chunked"] > 0)
    out = managed_decode(v, lat)
    ref = native_decode(v, lat)
    check("with process_output: shape {} (native {}), max|Δ| {:.2g}".format(tuple(out.shape), tuple(ref.shape), err(out, ref)),
          out.shape == ref.shape and err(out, ref) <= TOL)


# ---------------------------------------------------------------------------
# 3. multi-frame, the adapter directly
# ---------------------------------------------------------------------------

def direct(v, lat, rows, ws=None):
    bound, why = h15.match(v, lat, {})
    assert bound is not None, why
    ws = ws or mvae.layer1_workspace()
    plan = bound.plan(v, lat, 0, ws, rows=rows)
    stats = vae_ops.OpStats()
    with torch.inference_mode():
        out = bound.run(v, lat, plan, ws, stats)
    return out, plan, stats


def multi_tests(v):
    fsm = v.first_stage_model
    for t, h, w, rows in ((2, 6, 5, 40), (3, 6, 5, 7), (5, 6, 5, 7), (5, 6, 5, 40), (5, 6, 5, 96), (6, 4, 4, 16), (9, 3, 4, 9), (5, 7, 3, 11)):
        lat = lat_(h, w, t=t, seed=10 + t)
        po = v.process_output
        v.process_output = lambda image: image
        try:
            out, plan, _ = direct(v, lat, rows)
        finally:
            v.process_output = po
        with torch.inference_mode():
            ref = fsm.decode(_on_model(fsm, lat)).float()
        e = err(out, ref)
        check("T = {} latent {}x{}, rows {}: {} stripes x {} time chunks, {} frames (4T - 3 = {}); max|Δ| {:.2g} vs native".format(
            t, h, w, rows, len(plan.stripes), len(plan.chunks), out.shape[2], 4 * t - 3, e),
            out.shape == ref.shape and out.shape[2] == 4 * t - 3 and e <= TOL)
    lat = lat_(6, 5, t=5, seed=21)
    v.process_output = lambda image: image
    try:
        out, plan, stats = direct(v, lat, 32, ws=16 * 1024)
    finally:
        v.process_output = po
    with torch.inference_mode():
        ref = fsm.decode(_on_model(fsm, lat)).float()
    check("T = 5, workspace 16 KiB: {} conv calls in row blocks, max|Δ| {:.2g}".format(stats.conv_chunked, err(out, ref)),
          stats.conv_chunked > 0 and err(out, ref) <= TOL)
    bound, _ = h15.match(v, lat, {})
    p = bound.plan(v, lat, 0, mvae.layer1_workspace(), rows=16)
    check("plan planes per level {} (the largest chunk: 2 latent frames -> 4 -> 8)".format(p.planes),
          p.planes[0] == 2 and p.planes[-1] == 8 and p.ckpt_bytes == bound.ckpt_channels * 6 * 5 * 5 * 4)


# ---------------------------------------------------------------------------
# 4. self-test
# ---------------------------------------------------------------------------

def selftest_tests(v):
    eng._SELFTEST.clear()
    for t in (1, 5):
        bound, _ = h15.match(v, lat_(6, 5, t=t), {})
        ok, detail = eng.self_test(bound, v)
        check("self-test ({}) passes: {}".format("single frame" if t == 1 else "multi-frame", detail), ok)
    check("single and multi-frame cached apart ({} entries)".format(len(eng._SELFTEST)), len(eng._SELFTEST) == 2)

    orig_build, orig_call = h15.build_units, h15.HV15Stripe.call_unit

    def small_halo(fsm, single):
        # every halo 0 (an UpSmpl's halo alone at 0 gives only ~1.6e-5 at the boundary rows of this small fp32 model on
        # the GPU, below the self-test's tolerance)
        prefix, units = orig_build(fsm, single)
        for u in units:
            u.halo = 0
        return prefix, units

    def no_carry(self, u, x, state, k):
        return u.module(x) if not u.extra["carried"] else u.module(x, None, [])

    lat1 = lat_(6, 5)
    for label, name, fn, t in (("halos 0", "build_units", small_halo, 1), ("carries dropped between chunks", "call_unit", no_carry, 5)):
        eng._SELFTEST.clear()
        if name == "build_units":
            h15.build_units = fn
        else:
            h15.HV15Stripe.call_unit = fn
        try:
            bound, _ = h15.match(v, lat_(6, 5, t=t), {})
            ok, detail = eng.self_test(bound, v)
            if t == 1:
                with Spy() as spy:
                    out = managed_decode(v, lat1, raw=True)
                last = mvae.last_decode()
        finally:
            h15.build_units, h15.HV15Stripe.call_unit = orig_build, orig_call
        # a wrong value fails the comparison; dropped carries already break the frame count (StripeError), also a failure
        check("injected bug: {} -> self-test fails ({})".format(label, detail[:120]), not ok)
        if t == 1:
            ref = native_decode(v, lat1, raw=True)
            note = last.get("layer1") or ""
            check("  ... the decode goes to layer 2, == native (max|Δ| {:.2g})".format(err(out, ref)),
                  last.get("strategy") == "layer2" and "self-test failed" in note and err(out, ref) <= TOL and spy.tiled == 0)
    eng._SELFTEST.clear()


# ---------------------------------------------------------------------------
# 5. caching allocator
# ---------------------------------------------------------------------------

def allocator_tests():
    import alloc_sim
    for w, h, frames, dt in ((1344, 768, 1, "bf16"), (2048, 2048, 1, "fp16"), (3072, 3072, 1, "bf16"), (2048, 2048, 1, "fp32"),
                             (832, 480, 81, "fp16"), (1280, 720, 121, "bf16")):
        i = alloc_sim.decode_trace(w, h, dt, None, model="hv15", frames=frames)
        l2 = alloc_sim.decode_trace(w, h, dt, None, layer=2, model="hv15", frames=frames) if frames == 1 else None
        msg = "{} {}x{}x{} layer 1: {}, {} x {} rows, simulated reserved {:.2f} GiB <= estimate {:.2f} GiB{}".format(
            "Refiner" if frames == 1 else "HunyuanVideo 1.5", w, h, frames, dt, i["stripes"], i["rows"], i["reserved"] / G, i["estimate"] / G,
            "" if l2 is None else " (layer 2 reserves {:.2f})".format(l2["reserved"] / G))
        if frames == 1:
            check(msg, i["reserved"] <= i["estimate"] and (l2 is None or i["estimate"] < l2["reserved"]))
        else:
            print("[INFO] {} ({})".format(msg, "holds" if i["reserved"] <= i["estimate"] else "does not hold: the arena fragmentation of the carries"))
    for dt, t in (("bf16", 1), ("fp32", 1), ("bf16", 5)):
        pk, left, bound = alloc_sim.selftest_trace("hv15", dt, latent_frames=t)
        msg = "self-test ({}, {}): simulated reserved peak {:.0f} MiB <= selftest_memory {:.0f} MiB, nothing left".format(
            dt, "single frame" if t == 1 else "multi-frame", pk / 2 ** 20, bound / 2 ** 20)
        if t == 1:
            check(msg, pk <= bound and left == 0)
        else:
            print("[INFO] {} ({})".format(msg, "holds" if pk <= bound and left == 0 else "does not hold"))


def main():
    if not mvae.is_installed():
        check("vae.install() on this ComfyUI", mvae.install())
    with registered():
        v = hv15_vae(dtype=torch.float32)
        b5, _ = h15.match(v, lat_(4, 4, t=5), {})
        b1, _ = h15.match(v, lat_(4, 4), {})
        check("video: layer 1 as the fallback (prefer_layer2), one frame: layer 1 as usual",
              h15.MULTIFRAME_ENABLED and b5.prefer_layer2 is True and b1.prefer_layer2 is False)
        h15.MULTIFRAME_ENABLED = False
        b, why = h15.match(v, lat_(4, 4, t=5), {})
        h15.MULTIFRAME_ENABLED = True
        check("MULTIFRAME_ENABLED False: a video not matched ({})".format(why), b is None and "layer 1 off" in why)
        recognition_tests(v)
        single_tests(v)
        multi_tests(v)
        selftest_tests(v)
        allocator_tests()
    finish()


if __name__ == "__main__":
    main()
