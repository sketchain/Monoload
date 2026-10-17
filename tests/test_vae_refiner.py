"""The HunyuanImage 2.1 Refiner VAE (monoload/vae_refiner.py; vae_refiner.Decoder, refiner_vae=False; DESIGN §9.25,
§9.30, §9.34).

No model files needed: ComfyUI's own AutoencodingEngine with vae_refiner.Decoder / Encoder with small channels and
random weights, through comfy.sd.VAE (it detects the kind from the 5D decoder.conv_in with 32 input channels; its
hard-coded block_out_channels are swapped for small ones while it builds), latent 32 channels at 1/16, 5D T=1, fp32
on the CPU unless noted.

  1. recognition: recognized by its structure (the conv_in + repeat step and the T=1 stages in the prefix, UpSmpl
     units with halo 2 and planes x2 / x1, the last-frame step with planes x1/4, 25 norms with whole-image
     statistics over all frames); neither the HunyuanImage 2.1 adapter takes it nor it HunyuanImage 2.1; what it
     does not handle (a reflect-padded conv, a forward hook, vae_options, a wrong channel count, several latent
     frames, HunyuanVideo 1.5's refiner_vae=True) -> layer 2 with the reason (== native) or no match; the prefix
     step is bit-identical to the decoder's own first line;
  2. UpSmpl's halo (2) is the exact boundary of the rows a cut slice changes, with and without time upsampling;
  3. whole decoder vs native VAE.decode: schemes A / D / B / C x stripe heights, odd latent, 1x1 latent, batch 2,
     16 KiB workspace; bf16 against an fp32 truth at the level of native bf16; the engine's planes per level;
  4. self-test: passes (12 x 12 decoder input), cached apart from HunyuanImage 2.1; the fp32 copy keeps the middle
     time slice of the convs that see one frame, its own decode == the model's, selftest_params counts it;
     injected bugs (the conv_in step without the repeat, stripe-local statistics, the first frame instead of the
     last) -> self-test fails, layer 2, == native; the first decode's estimate includes the self-test;
  5. OOM -> smaller stripes, at the floor MonoloadVAEOOMError, never tiled or layer 2; the layer-2 estimate: 6
     activation copies for this decoder, by its class (a forward hook does not change it; HunyuanVideo 1.5 keeps
     the generic 4); MONOLOAD_VAE_BUDGET: layer 2 when its estimate fits, else layer 1, else an error naming every
     candidate;
  6. the Monoload VAE Settings node (forced scheme / rows on a copy, layer 2 only) and the Info node (the
     Refiner's scheme hint);
  7. caching allocator (tests/alloc_sim.py / vae_inventory.py, full-size Refiner on the meta device): layer 1
     reserved <= estimate (1344x768 default B, 2048x2048 A / D / B / C, 2560x1536 B), the self-test's peak <= its
     bound, layer 2's simulated reserved <= the layer-2 estimate (1344x768, 1536x1536, 2048x2048; the 4 generic
     copies are not a bound at 2048x2048).

    python tests/test_vae_refiner.py
"""

import torch

from common import check, expect_raises, finish
import comfy.sd
import comfy.ldm.hunyuan_video.vae_refiner as hyr
from comfy.ldm.models.autoencoder import AutoencodingEngine
from monoload import vae as mvae
from monoload import vae_engine as eng
from monoload import vae_ops
from monoload import vae_hunyuan as vh, vae_refiner as vr
from monoload.errors import MonoloadError, MonoloadVAEOOMError
from test_vae import Spy, init_random, managed_decode, native_decode
from test_vae_ldm import no_overrides

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


def lat_(h, w, b=1, seed=4, t=1):
    return torch.randn(b, Z, t, h, w, generator=torch.Generator().manual_seed(seed))


def err(a, b):
    return float((a.float() - b.float()).abs().max())


# ---------------------------------------------------------------------------
# 1. recognition, 2. halo
# ---------------------------------------------------------------------------

def recognition_tests():
    v = refiner_vae()
    fsm = v.first_stage_model
    dec = fsm.decoder
    check("comfy.sd.VAE builds the Refiner: vae_refiner.Decoder, refiner_vae False, latent_dim 3",
          type(dec) is _DEC and not dec.refiner_vae and v.latent_dim == 3)
    lat = lat_(5, 4)
    bound, why = vr.match(v, lat, {})
    ups = [(u.kind, u.halo, u.scale, u.tscale) for u in bound.units if u.scale > 1] if bound else []
    last = bound.units[-1] if bound else None
    check("recognized as '{}': prefix {}, UpSmpl units {}, last step {} (planes x{}), {} norms with whole-image statistics, output {}".format(
              bound.name if bound else why, [n for n, _ in bound.prefix] if bound else "-", ups, last.kind if last else "-",
              last.tscale if last else "-", len(bound.norm_refs()) if bound else 0, bound.output_shape(lat) if bound else "-"),
          bound is not None and [n for n, _ in bound.prefix] == ["decoder.conv_in+repeat", "decoder.mid.block_1", "decoder.mid.attn_1", "decoder.mid.block_2",
                                                                "decoder.up.0.block.0", "decoder.up.0.block.1", "decoder.up.0.block.2"]
          and ups == [(vr.UPS, 2, 2, 2), (vr.UPS, 2, 2, 2), (vr.UPS, 2, 2, 1), (vr.UPS, 2, 2, 1)] and last.kind == vr.LAST and last.tscale * 4 == 1
          and len(bound.norm_refs()) == 25 and bound.scale == 16 and bound.output_shape(lat) == (1, 3, 1, 80, 64) and bound.ckpt_channels == 64
          and bound.hdim == 3 and bound.tdim == 2, str(why))
    check("the T=1 convs (refiner_structure's t1): conv_in, mid, up[0]'s blocks and its upsample conv ({})".format(len(bound.info["t1"])),
          bound.info["t1"][0] == "conv_in" and bound.info["t1"][-1] == "up.0.upsample.conv" and len(bound.info["t1"]) == 12 and bound.info["planes"] == 4)
    sel, _ = mvae._select_layer1(v, lat, {})
    check("vae._select_layer1 picks the Refiner adapter", sel is not None and sel.name.startswith("HunyuanImage 2.1 Refiner stripes"))
    from test_vae_hunyuan import hy_vae, lat_ as hy_lat
    hv = hy_vae()
    _, why_h = vh.match(v, lat, {})
    _, why_r = vr.match(hv, hy_lat(5, 4), {})
    check("the HunyuanImage 2.1 adapter does not take it ({}), nor this one HunyuanImage 2.1 ({})".format(why_h, why_r),
          "vae_refiner.Decoder" in why_h and "hunyuan_video.vae.Decoder" in why_r)

    seen = []
    h = dec.mid.block_1.register_forward_pre_hook(lambda m, args: seen.append(args[0].clone()))
    try:
        native_decode(v, lat)
    finally:
        h.remove()
    mine = vr.ConvIn(dec)(lat)
    check("ConvIn is bit-identical to the decoder's first line (conv_in(z) + z.repeat_interleave({}, 1))".format(dec.block_out_channels[0] // Z),
          len(seen) == 1 and torch.equal(seen[0], mine))

    up_conv = dec.up[2].upsample.conv
    old = up_conv.padding_mode
    up_conv.padding_mode = "reflect"
    try:
        _, why = vr.match(v, lat, {})
        out = managed_decode(v, lat, raw=True)
        rec = mvae.last_decode()
        ref = native_decode(v, lat, raw=True)
    finally:
        up_conv.padding_mode = old
    check("reflect padding in an UpSmpl conv -> not recognized, layer 2, == native (max|Δ| {:.2g}): {}".format(err(out, ref), why),
          why is not None and "up[2].upsample.conv" in why and rec.get("strategy") == "layer2" and err(out, ref) <= 1e-5)
    hk = dec.up[3].block[1].norm2.register_forward_hook(lambda m, a, o: None)
    try:
        _, why = vr.match(v, lat, {})
        out = managed_decode(v, lat, raw=True)
        rec = mvae.last_decode()
        ref = native_decode(v, lat, raw=True)
    finally:
        hk.remove()
    check("a forward hook in the decoder -> layer 2, == native (max|Δ| {:.2g}): {}".format(err(out, ref), why),
          why is not None and "hook" in why and rec.get("strategy") == "layer2" and err(out, ref) <= 1e-5)
    _, why = vr.match(v, lat, {"x": 1})
    check("vae_options given -> no match: {}".format(why), why is not None and "vae_options" in why)
    _, why = vr.match(v, torch.randn(1, 16, 1, 5, 4), {})
    check("latent with 16 channels -> no match: {}".format(why), why is not None and "channels" in why)
    _, why = vr.match(v, lat_(5, 4, t=2), {})
    check("latent with 2 frames -> no match: {}".format(why), why is not None and "frames" in why)
    vid = refiner_model(refiner_vae=True)
    why = vr.refiner_structure(vid)[0]
    check("HunyuanVideo 1.5's decoder (refiner_vae=True) -> no match: {}".format(why), why is not None and "refiner_vae" in why)
    check("... no instance override left on the model", no_overrides(fsm))
    return v


def halo_tests(v):
    dec = v.first_stage_model.decoder
    for i, t in ((0, 1), (1, 2), (3, 4)):
        up = dec.up[i].upsample
        c = up.conv.weight.shape[1]
        x = torch.randn(1, c, t, 20, 9, generator=torch.Generator().manual_seed(11 + i))
        with torch.inference_mode():
            whole = up(x)
            part = up(x[:, :, :, 6:])
        d = (part - whole[:, :, :, 12:]).abs().amax(dim=(0, 1, 2, 4))
        bad = [r for r in range(d.shape[0] - 4) if float(d[r]) > 1e-5]
        check("decoder.up[{}].upsample (UpSmpl, {} frame(s) -> {}) on rows from 6: output rows {} after the cut differ (by {:.2g} / {:.2g}), every "
              "later row by <= {:.1g}: halo 2".format(i, t, whole.shape[2], bad, float(d[0]), float(d[1]), float(d[2:].max())),
              bad == [0, 1] and float(d[:2].min()) > 0.1 and float(d[2:].max()) <= 1e-5)


# ---------------------------------------------------------------------------
# 3. whole decoder
# ---------------------------------------------------------------------------

def compare(label, v, latent, rows=None, scheme="B", tol=1e-5):
    mvae.set_gn_scheme(scheme)
    mvae.set_stripe_rows(rows)
    try:
        with Spy() as spy:
            raw = managed_decode(v, latent, raw=True)
    finally:
        mvae.set_stripe_rows(None)
    last = mvae.last_decode()
    ref = native_decode(v, latent, raw=True)
    e = err(raw, ref)
    passes = 0 if last.get("stripes") == 1 else 25
    check("{}: scheme {}, {} stripes of {} rows, {} statistics passes, saves {}; max|Δ| {:.2g}".format(
        label, scheme, last.get("stripes"), last.get("rows"), last.get("passes"),
        "+".join(vae_ops.fmt_bytes(b) for b in last.get("saves") or []) or "none", e),
        last.get("strategy") == "layer1" and (last.get("adapter") or "").startswith("HunyuanImage 2.1 Refiner stripes") and e <= tol
        and raw.shape == ref.shape and last.get("passes") == passes and spy.tiled == 0 and no_overrides(v.first_stage_model),
        "strategy {} note {}".format(last.get("strategy"), last.get("layer1")))
    return last


def decoder_tests(v):
    lat = lat_(5, 4)
    bound, _ = vr.match(v, lat, {})
    p = bound.plan(v, lat, 0, 1 << 20, rows=24)
    check("the plan's planes per level: {} (1 in the prefix, x2 at the two time upsamplings, 1 after the last-frame step)".format(p.planes),
          p.planes[0] == 1 and p.planes[1] == 2 and max(p.planes) == 4 and p.planes[-1] == 1 and p.planes[-2] == 4)
    for scheme in "ADBC":
        for rows in (7, 40, None):
            compare("5x4 latent (output 80x64), rows {}".format(rows or "default"), v, lat, rows=rows, scheme=scheme)
    compare("odd 3x5 latent, rows 9", v, lat_(3, 5, seed=5), rows=9, scheme="C")
    compare("1x1 latent, rows 5", v, lat_(1, 1, seed=6), rows=5, scheme="A")
    compare("batch 2, rows 16", v, lat_(5, 4, b=2, seed=7), rows=16, scheme="B")
    mvae.set_workspace(16 * 1024)
    try:
        last = compare("workspace 16 KiB, rows 32", v, lat, rows=32, scheme="D")
    finally:
        mvae.set_workspace(mvae.DEFAULT_WORKSPACE)
    check("  ... conv calls inside the stripes were split into row blocks ({})".format(last["stats"]["conv_chunked"]), last["stats"]["conv_chunked"] > 0)
    m = refiner_model()
    truth = native_decode(refiner_vae(model=m), lat, raw=True).float()
    vb = refiner_vae(dtype=torch.bfloat16, model=m)
    nb = native_decode(vb, lat, raw=True).float()
    for scheme in ("A", "B"):
        mvae.set_gn_scheme(scheme)
        mvae.set_stripe_rows(24)
        try:
            mb = managed_decode(vb, lat, raw=True).float()
        finally:
            mvae.set_stripe_rows(None)
        last = mvae.last_decode()
        rn = float((nb - truth).pow(2).mean().sqrt())
        rm = float((mb - truth).pow(2).mean().sqrt())
        check("bf16 VAE, scheme {}: RMSE vs fp32 truth: layer 1 {:.3g}, native {:.3g}".format(scheme, rm, rn),
              last.get("strategy") == "layer1" and rm <= 1.5 * rn + 1e-4)
    mvae.set_gn_scheme(None)
    return lat


# ---------------------------------------------------------------------------
# 4. self-test
# ---------------------------------------------------------------------------

def selftest_tests(v, lat):
    eng._SELFTEST.clear()
    bound, _ = vr.match(v, lat, {})
    ok, detail = eng.self_test(bound, v)
    check("self-test of the Refiner structure (scheme {}, decoder input {} x {}) passes: {}".format(
        bound.gn_scheme, bound.selftest_size, bound.selftest_size, detail), ok and bound.selftest_size == 12)
    from test_vae_hunyuan import hy_vae, lat_ as hy_lat
    managed_decode(hy_vae(), hy_lat(5, 4))
    check("cached apart from the HunyuanImage 2.1 structure ({} entries)".format(len(eng._SELFTEST)), len(eng._SELFTEST) == 2)

    tb = bound.fp32_copy(torch.device("cpu"))
    mods = dict(tb.module.decoder.named_modules())
    sliced = [n for n in bound.info["t1"] if tuple(mods[n].kernel_size) == (1, 3, 3) and tuple(mods[n].padding) == (0, 1, 1)]
    others = [n for n, m in mods.items() if isinstance(m, torch.nn.Conv3d) and n not in bound.info["t1"] and m.kernel_size[0] == 1 and m.kernel_size[1] == 3]
    n_copy = sum(p.numel() for p in tb.module.parameters())
    z = torch.randn(1, Z, 1, 6, 5, generator=torch.Generator().manual_seed(9))
    with torch.inference_mode():
        a = tb.reference_decode(z)
        b = AutoencodingEngine.decode(v.first_stage_model, z)
    rel = err(a, b) / max(1.0, float(b.abs().max()))
    check("fp32 copy: the {} convs that see one frame keep their middle time slice (no other conv sliced), {} parameters = selftest_params "
          "(the model's decoder {}); its own decode == the model's (relative {:.2g})".format(
              len(sliced), n_copy, sum(p.numel() for p in v.first_stage_model.decoder.parameters()), rel),
          len(sliced) == len(bound.info["t1"]) and not others and n_copy == bound.selftest_params() and rel <= 1e-5 and a.shape == b.shape)
    del tb

    orig_call, orig_add, orig_last = vr.ConvIn.__call__, eng.Moments.add, vr.LastFrame.__call__

    def no_repeat(self, z):              # the first line without its repeated-latent shortcut
        return self.dec.conv_in(z)

    def local_add(self, x, hdim, ws):    # statistics of each stripe on its own
        self.n, self.mean, self.m2 = 0, None, None
        return orig_add(self, x, hdim, ws)

    def first_frame(self, x):            # the first of the 4 frames instead of the last
        return x[:, :, :1]

    for label, target, attr, fn in (("the conv_in step without the repeat", vr.ConvIn, "__call__", no_repeat),
                                    ("stripe-local GroupNorm statistics", eng.Moments, "add", local_add),
                                    ("the first frame instead of the last", vr.LastFrame, "__call__", first_frame)):
        eng._SELFTEST.clear()
        setattr(target, attr, fn)
        try:
            with Spy() as spy:
                out = managed_decode(v, lat, raw=True)
            last = mvae.last_decode()
        finally:
            vr.ConvIn.__call__, eng.Moments.add, vr.LastFrame.__call__ = orig_call, orig_add, orig_last
        ref = native_decode(v, lat, raw=True)
        note = last.get("layer1") or ""
        check("injected bug: {} -> self-test fails, layer 2, == native (max|Δ| {:.2g}): {}".format(label, err(out, ref), note[:130]),
              last.get("strategy") == "layer2" and "self-test failed" in note and err(out, ref) <= 1e-5 and spy.tiled == 0)
    eng._SELFTEST.clear()
    managed_decode(v, lat)
    e1 = mvae.last_decode()["estimate"]
    managed_decode(v, lat)
    e2 = mvae.last_decode()["estimate"]
    st = bound.selftest_memory()
    check("first decode records the self-test ({}; estimate {} = max(plan {}, self-test)), the next one does not ({})".format(
        vae_ops.fmt_bytes(e1.get("selftest")), vae_ops.fmt_bytes(e1["total"]), vae_ops.fmt_bytes(e1["plan"]), vae_ops.fmt_bytes(e2["total"])),
        e1.get("selftest") == st and e1["total"] == max(e1["plan"], st) and e2.get("selftest") == 0 and e2["total"] == e2["plan"])


# ---------------------------------------------------------------------------
# 5. OOM, layer-2 estimate, budget
# ---------------------------------------------------------------------------

def oom_budget_tests(v, lat):
    mvae.set_gn_scheme("B")
    managed_decode(v, lat)      # self-test outside the injected OOM
    orig_run, orig_stripes, orig_l2 = eng.run_passes, eng.run_stripes, mvae._run
    calls_l2 = [0]

    def l2(*a, **kw):
        calls_l2[0] += 1
        return orig_l2(*a, **kw)

    def oom_above(limit):
        def run(plan, *a, **kw):
            if max(b - a_ for a_, b in plan.stripes) > limit:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return orig_run(plan, *a, **kw)

        def stripes(units, plan, *a, **kw):
            if max(b - a_ for a_, b in plan.stripes) > limit:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return orig_stripes(units, plan, *a, **kw)
        eng.run_stripes = stripes
        return run
    mvae._run = l2
    try:
        eng.run_passes = oom_above(24)
        mvae.set_stripe_rows(80)
        with Spy() as spy:
            out = managed_decode(v, lat, raw=True)
        last = mvae.last_decode()
        ref = native_decode(v, lat, raw=True)
        check("simulated OOM above 24-row stripes: retried with {}-row stripes after {} retries, == native (max|Δ| {:.2g})".format(
            last.get("rows"), last.get("retries"), err(out, ref)),
            last.get("strategy") == "layer1" and last.get("rows") <= 24 and last.get("retries") >= 1 and err(out, ref) <= 1e-5
            and spy.tiled == 0 and calls_l2[0] == 0)
        eng.run_passes = oom_above(0)
        with Spy() as spy:
            expect_raises("OOM even with the smallest stripes -> MonoloadVAEOOMError", MonoloadVAEOOMError,
                          lambda: managed_decode(v, lat), "never falls back to the approximate tiled", "nor to layer 2")
        check("... neither tiled nor layer 2 was called, no override left", spy.tiled == 0 and calls_l2[0] == 0 and no_overrides(v.first_stage_model))
    finally:
        eng.run_passes, eng.run_stripes, mvae._run = orig_run, orig_stripes, orig_l2
        mvae.set_stripe_rows(None)
        mvae.set_gn_scheme(None)

    est, probe = mvae._layer2_estimate(v, lat, {}, mvae.workspace())
    four = mvae.estimate(v, lat, mvae.workspace(), probe)
    check("layer-2 estimate with {} activation copies for this decoder: {} (the generic 4 copies: {}), nothing retained".format(
        vr.RefinerStripe.activation_copies, vae_ops.fmt_bytes(est["total"]), vae_ops.fmt_bytes(four["total"])),
        est["total"] == four["total"] + 2 * four["activation"] and est["retained"] == 0 and vr.RefinerStripe.activation_copies == 6)
    # a forward hook keeps layer 1 off this VAE, not the copies of its own decode: the layer-2 terms go by the decoder's class
    hk = v.first_stage_model.decoder.conv_out.register_forward_hook(lambda *a: None)
    try:
        hooked, why = vr.match(v, lat, {})
        terms = mvae._decoder_terms(v, lat)
    finally:
        hk.remove()
    check("forward hook: layer 1 refuses the VAE ({}), the layer-2 estimate still takes {} activation copies".format((why or "")[:60], terms["copies"]),
          hooked is None and terms == {"retained": 0, "copies": vr.RefinerStripe.activation_copies})
    vid = refiner_model(refiner_vae=True)
    shim = type("V", (), {"first_stage_model": vid})()
    check("HunyuanVideo 1.5's decoder (refiner_vae=True): the generic terms", mvae._decoder_terms(shim, lat) == {"retained": 0, "copies": mvae.ACTIVATION_COPIES})
    for bud, want in ((est["total"], "layer2"), (est["total"] - 1, "layer1")):
        mvae.set_budget(bud)
        try:
            out = managed_decode(v, lat, raw=True)
            last = mvae.last_decode()
        finally:
            mvae.set_budget(None)
        ref = native_decode(v, lat, raw=True)
        check("budget {} ({} the layer-2 estimate): {}{}, == native (max|Δ| {:.2g})".format(
            vae_ops.fmt_bytes(bud), "=" if bud == est["total"] else "just below", last.get("strategy"),
            " ({})".format(last.get("adapter")) if last.get("adapter") else "", err(out, ref)),
            last.get("strategy") == want and err(out, ref) <= 1e-5)
    mvae.set_budget(1024)
    try:
        expect_raises("budget 1 KiB: nothing fits -> MonoloadError naming each candidate's need", MonoloadError,
                      lambda: managed_decode(v, lat), "layer 2", "layer 1", "scheme A", "scheme C")
    finally:
        mvae.set_budget(None)


# ---------------------------------------------------------------------------
# 6. nodes
# ---------------------------------------------------------------------------

def node_tests(v, lat):
    from monoload.nodes import NODE_CLASS_MAPPINGS
    from test_vae_node import node_apply
    from test_vae_hunyuan import hy_vae
    settings = NODE_CLASS_MAPPINGS["MonoloadVAESettings"]
    info = NODE_CLASS_MAPPINGS["MonoloadInfo"]
    copy_ = node_apply(settings, v, gn_scheme="C", stripe_rows=16)
    out = managed_decode(copy_, lat, raw=True)
    last = mvae.last_decode()
    ref = native_decode(v, lat, raw=True)
    check("VAE Settings node (scheme C, 16 rows forced on a copy): {} stripes of {} rows, scheme {} [{}], == native (max|Δ| {:.2g})".format(
        last.get("stripes"), last.get("rows"), last.get("gn_scheme"), (last.get("settings_source") or {}).get("gn_scheme"), err(out, ref)),
        last.get("strategy") == "layer1" and last.get("gn_scheme") == "C" and last.get("rows") == 16
        and (last.get("settings_source") or {}).get("gn_scheme") == "node" and err(out, ref) <= 1e-5)
    t = getattr(info(), info.FUNCTION)(vae=copy_)["result"][0]
    hint = next((l for l in t.splitlines() if "GroupNorm scheme C:" in l), "")
    check("Info node shows the Refiner layer-1 decode and its own scheme hint: {}".format(hint.strip()[:150]),
          "layer 1 (HunyuanImage 2.1 Refiner stripes, GroupNorm scheme C" in t and "Refiner 2048x2048" in hint)
    t2 = getattr(info(), info.FUNCTION)(vae=node_apply(settings, hy_vae(), gn_scheme="C"))["result"][0]
    check("... HunyuanImage 2.1 keeps its own hint", "HunyuanImage 2.1 4K" in t2 and "Refiner" not in t2)
    l2 = node_apply(settings, v, mode="layer 2 only")
    managed_decode(l2, lat)
    check("VAE Settings node mode 'layer 2 only': layer 2", mvae.last_decode().get("strategy") == "layer2")


# ---------------------------------------------------------------------------
# 7. caching allocator
# ---------------------------------------------------------------------------

def allocator_tests():
    import alloc_sim
    for w, h, scheme in ((1344, 768, None), (2048, 2048, "A"), (2048, 2048, "D"), (2048, 2048, None), (2048, 2048, "C"), (2560, 1536, None)):
        i = alloc_sim.decode_trace(w, h, "bf16", None, model="hyr", scheme=scheme)
        check("Refiner {}x{} scheme {}: {} x {} rows, recompute {:.2f}x, simulated reserved {:.3f} GiB <= estimate {:.3f} GiB, "
              "{:.3f} stays (output)".format(w, h, scheme or "default (B)", i["stripes"], i["rows"], i["plan"].recompute, i["reserved"] / G,
                                             i["estimate"] / G, i["stays"] / G),
              i["reserved"] <= i["estimate"] and i["stays"] <= i["plan"].out_segment)
    for dt in ("bf16", "fp32"):
        info = {}
        pk, left, bound = alloc_sim.selftest_trace("hyr", dt, probe=True, info=info)
        check("Refiner self-test ({} VAE, after the budget's shape probe): simulated reserved peak {:.0f} MiB <= selftest_memory "
              "{:.0f} MiB, nothing left".format(dt, pk / 2 ** 20, bound / 2 ** 20), pk <= bound and left == 0 and info["after_probe"] == 0)
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
    mvae.set_stripe(True)
    v = recognition_tests()
    halo_tests(v)
    lat = decoder_tests(v)
    selftest_tests(v, lat)
    oom_budget_tests(v, lat)
    node_tests(v, lat)
    allocator_tests()
    finish()


if __name__ == "__main__":
    main()
