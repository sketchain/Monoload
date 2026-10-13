"""Layer 1 for the HunyuanImage 2.1 VAE (monoload/vae_hunyuan.py, DESIGN §9.24).

No model files needed: ComfyUI's own AutoencodingEngine with
hunyuan_video.vae.Decoder / Encoder with small channels and random weights,
through comfy.sd.VAE (it detects the kind from decoder.conv_in's 64 input
channels; its hard-coded block_out_channels are swapped for small ones while
it builds), latent 64 channels at 1/32, fp32 on the CPU unless noted.

  1. recognition: recognized by its structure (the conv_in + repeat step first
     in the prefix, the up stages from the lowest resolution, PixelUnshuffle2D
     units with halo 2, 31 norms with whole-image statistics); its key differs
     from the LDM ones; neither the LDM adapter takes it nor it Flux ae's;
     what it does not handle (a reflect-padded conv, a forward hook,
     vae_options, a wrong channel count) -> layer 2 with the reason, == native;
     the prefix step is bit-identical to the decoder's own first line;
  2. PixelUnshuffle2D's halo (2) is the exact boundary of the rows a cut slice changes (beyond it: rounding only);
  3. whole decoder vs native VAE.decode: schemes A / D / B / C x stripe heights,
     odd latent, 1x1 latent, batch 2, 16 KiB workspace; bf16 against an fp32
     truth at the level of native bf16;
  4. self-test: passes (12 x 12 decoder input), cached per scheme apart from the
     LDM ones; injected bugs (the conv_in step without the repeat, stripe-local
     statistics) -> self-test fails, layer 2, == native; the first decode's
     estimate includes the self-test;
  5. OOM -> smaller stripes, at the floor MonoloadVAEOOMError, never tiled or
     layer 2; the layer-2 estimate uses 6 activation copies for it;
     MONOLOAD_VAE_BUDGET: layer 2 when its estimate fits, else layer 1, else an
     error naming every candidate;
  6. the Monoload VAE Settings node (forced scheme / rows on a copy, layer 2
     only) and the Info node (the HunyuanImage scheme hint);
  7. caching allocator (tests/alloc_sim.py, full-size HunyuanImage 2.1 on the
     meta device): layer 1 reserved <= estimate (1344x768, 2688x1536, 4K =
     3840x2176 default B; 4K A / D / C), the self-test's peak <= its bound,
     layer 2's simulated peak <= the layer-2 estimate (1344x768, 4K).

    python tests/test_vae_hunyuan.py
"""

import torch

from common import check, expect_raises, finish
import comfy.sd
import comfy.ldm.hunyuan_video.vae as hy
from comfy.ldm.models.autoencoder import AutoencodingEngine
from monoload import vae as mvae
from monoload import vae_engine as eng
from monoload import vae_ops
from monoload import vae_hunyuan as vh, vae_ldm as vl
from monoload.errors import MonoloadError, MonoloadVAEOOMError
from test_vae import Spy, init_random, managed_decode, native_decode
from test_vae_ldm import ldm_vae, no_overrides

torch.manual_seed(0)
Z = 64
BOC = [32, 32, 64, 64, 128, 128]     # the decoder's levels: 128 (H/32, 2 x z: repeat 2), 128, 64, 64, 32, 32 channels
DD = {"block_out_channels": BOC, "in_channels": 3, "out_channels": 3, "num_res_blocks": 2, "ffactor_spatial": 32,
      "downsample_match_channel": True, "upsample_match_channel": True, "z_channels": Z}
_DEC, _ENC = hy.Decoder, hy.Encoder


def hy_model():
    return init_random(AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                          encoder_config={'target': "comfy.ldm.hunyuan_video.vae.Encoder", 'params': DD},
                                          decoder_config={'target': "comfy.ldm.hunyuan_video.vae.Decoder", 'params': DD}))


def hy_vae(dtype=None, model=None):
    """comfy.sd.VAE on a small HunyuanImage 2.1 state dict (sd.py's block_out_channels swapped for BOC while it builds)."""
    m = model if model is not None else hy_model()
    hy.Decoder = lambda **kw: _DEC(**dict(kw, block_out_channels=BOC))
    hy.Encoder = lambda **kw: _ENC(**dict(kw, block_out_channels=BOC))
    try:
        return comfy.sd.VAE(sd=m.state_dict(), dtype=dtype)
    finally:
        hy.Decoder, hy.Encoder = _DEC, _ENC


def lat_(h, w, b=1, seed=4):
    return torch.randn(b, Z, h, w, generator=torch.Generator().manual_seed(seed))


def err(a, b):
    return float((a.float() - b.float()).abs().max())


# ---------------------------------------------------------------------------
# 1. recognition, 2. halo
# ---------------------------------------------------------------------------

def recognition_tests():
    v = hy_vae()
    fsm = v.first_stage_model
    lat = lat_(5, 4)
    bound, why = vh.match(v, lat, {})
    kinds = [(u.kind, u.halo, u.scale) for u in bound.units if u.scale > 1] if bound else []
    check("HunyuanImage 2.1 (AutoencodingEngine + hunyuan_video.vae.Decoder, latent {} ch at 1/{}) recognized as '{}': prefix {}, "
          "{} PixelUnshuffle2D units {}, {} norms with whole-image statistics, output {}".format(
              v.latent_channels, v.spacial_compression_decode(), bound.name if bound else why, [n for n, _ in bound.prefix] if bound else "-",
              len(kinds), kinds[:1], len(bound.norm_refs()) if bound else 0, bound.output_shape(lat) if bound else "-"),
          bound is not None and [n for n, _ in bound.prefix] == ["decoder.conv_in+repeat", "decoder.mid.block_1", "decoder.mid.attn_1", "decoder.mid.block_2",
                                                                "decoder.up.0.block.0", "decoder.up.0.block.1", "decoder.up.0.block.2"]
        and kinds == [(vh.PU, 2, 2)] * 5 and len(bound.norm_refs()) == 31 and bound.scale == 32 and bound.output_shape(lat) == (1, 3, 160, 128)
        and bound.ckpt_channels == 128, str(why))
    sdxl = ldm_vae(4, True)
    other, _ = vl.match(sdxl, torch.randn(1, 4, 12, 10), {})
    _, why_l = vl.match(v, lat, {})
    flux = ldm_vae(16, False)
    _, why_f = vh.match(flux, torch.randn(1, 16, 12, 10), {})
    check("its key differs from the LDM ones; the LDM adapter does not take it ({}), nor this one Flux ae ({})".format(why_l, why_f),
          other is not None and other.key != bound.key and "hunyuan_video.vae.Decoder" in why_l and "diffusionmodules.model.Decoder" in why_f)
    sel, _ = mvae._select_layer1(v, lat, {})
    check("vae._select_layer1 picks the HunyuanImage adapter", sel is not None and sel.name.startswith("HunyuanImage 2.1 stripes"))

    seen = []
    h = fsm.decoder.mid.block_1.register_forward_pre_hook(lambda m, args: seen.append(args[0].clone()))
    try:
        native_decode(v, lat)
    finally:
        h.remove()
    mine = vh.ConvIn(fsm.decoder)(lat)
    check("ConvIn is bit-identical to the decoder's first line (conv_in(z) + z.repeat_interleave({}, 1))".format(fsm.decoder.block_out_channels[0] // Z),
          len(seen) == 1 and torch.equal(seen[0], mine))

    pu_conv = fsm.decoder.up[2].upsample.conv
    old = pu_conv.padding_mode
    pu_conv.padding_mode = "reflect"
    try:
        _, why = vh.match(v, lat, {})
        out = managed_decode(v, lat, raw=True)
        last = mvae.last_decode()
        ref = native_decode(v, lat, raw=True)
    finally:
        pu_conv.padding_mode = old
    check("reflect padding in a PixelUnshuffle2D conv -> not recognized, layer 2, == native (max|Δ| {:.2g}): {}".format(err(out, ref), why),
          why is not None and "up[2].upsample.conv" in why and last.get("strategy") == "layer2" and err(out, ref) <= 1e-5)
    hk = fsm.decoder.up[4].block[1].norm2.register_forward_hook(lambda m, a, o: None)
    try:
        _, why = vh.match(v, lat, {})
        out = managed_decode(v, lat, raw=True)
        last = mvae.last_decode()
        ref = native_decode(v, lat, raw=True)
    finally:
        hk.remove()
    check("a forward hook in the decoder -> layer 2, == native (max|Δ| {:.2g}): {}".format(err(out, ref), why),
          why is not None and "hook" in why and last.get("strategy") == "layer2" and err(out, ref) <= 1e-5)
    _, why = vh.match(v, lat, {"x": 1})
    check("vae_options given -> no match: {}".format(why), why is not None and "vae_options" in why)
    _, why = vh.match(v, torch.randn(1, 32, 5, 4), {})
    check("latent with 32 channels -> no match: {}".format(why), why is not None and "channels" in why)
    check("... no instance override left on the model", no_overrides(fsm))
    return v


def halo_tests(v):
    dec = v.first_stage_model.decoder
    for i in (0, 4):
        pu = dec.up[i].upsample
        c = pu.conv.weight.shape[1]
        x = torch.randn(1, c, 20, 9, generator=torch.Generator().manual_seed(11 + i))
        with torch.inference_mode():
            whole = pu(x)
            part = pu(x[:, :, 6:])
        d = (part - whole[:, :, 12:]).abs().amax(dim=(0, 1, 3))
        # beyond the halo only rounding differs (the CPU conv may pick another algorithm for another input height)
        bad = [r for r in range(d.shape[0] - 4) if float(d[r]) > 1e-5]
        check("decoder.up[{}].upsample (PixelUnshuffle2D) on rows from 6: output rows {} after the cut differ (by {:.2g} / {:.2g}), every later "
              "row by <= {:.1g}: halo 2".format(i, bad, float(d[0]), float(d[1]), float(d[2:].max())),
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
    passes = 0 if last.get("stripes") == 1 else 31
    check("{}: scheme {}, {} stripes of {} rows, {} statistics passes, saves {}; max|Δ| {:.2g}".format(
        label, scheme, last.get("stripes"), last.get("rows"), last.get("passes"),
        "+".join(vae_ops.fmt_bytes(b) for b in last.get("saves") or []) or "none", e),
        last.get("strategy") == "layer1" and (last.get("adapter") or "").startswith("HunyuanImage 2.1 stripes") and e <= tol and raw.shape == ref.shape
        and last.get("passes") == passes and spy.tiled == 0 and no_overrides(v.first_stage_model),
        "strategy {} note {}".format(last.get("strategy"), last.get("layer1")))
    return last


def decoder_tests(v):
    lat = lat_(5, 4)
    for scheme in "ADBC":
        for rows in (7, 40, None):
            compare("5x4 latent (output 160x128), rows {}".format(rows or "default"), v, lat, rows=rows, scheme=scheme)
    compare("odd 3x5 latent, rows 9", v, lat_(3, 5, seed=5), rows=9, scheme="C")
    compare("1x1 latent, rows 5", v, lat_(1, 1, seed=6), rows=5, scheme="A")
    compare("batch 2, rows 16", v, lat_(5, 4, b=2, seed=7), rows=16, scheme="B")
    mvae.set_workspace(16 * 1024)
    try:
        last = compare("workspace 16 KiB, rows 32", v, lat, rows=32, scheme="D")
    finally:
        mvae.set_workspace(mvae.DEFAULT_WORKSPACE)
    check("  ... conv calls inside the stripes were split into row blocks ({})".format(last["stats"]["conv_chunked"]), last["stats"]["conv_chunked"] > 0)
    m = hy_model()
    truth = native_decode(hy_vae(model=m), lat, raw=True).float()
    vb = hy_vae(dtype=torch.bfloat16, model=m)
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
    bound, _ = vh.match(v, lat, {})
    ok, detail = eng.self_test(bound, v)
    check("self-test of the HunyuanImage structure (scheme {}, decoder input {} x {}) passes: {}".format(
        bound.gn_scheme, bound.selftest_size, bound.selftest_size, detail), ok and bound.selftest_size == 12)
    managed_decode(ldm_vae(4, True), torch.randn(1, 4, 12, 10))
    check("cached apart from the SDXL-like structure ({} entries)".format(len(eng._SELFTEST)), len(eng._SELFTEST) == 2)
    orig_call, orig_add = vh.ConvIn.__call__, eng.Moments.add

    def no_repeat(self, z):              # the first line without its repeated-latent shortcut
        return self.dec.conv_in(z)

    def local_add(self, x, hdim, ws):    # statistics of each stripe on its own
        self.n, self.mean, self.m2 = 0, None, None
        return orig_add(self, x, hdim, ws)

    for label, target, attr, fn in (("the conv_in step without the repeat", vh.ConvIn, "__call__", no_repeat),
                                    ("stripe-local GroupNorm statistics", eng.Moments, "add", local_add)):
        eng._SELFTEST.clear()
        setattr(target, attr, fn)
        try:
            with Spy() as spy:
                out = managed_decode(v, lat, raw=True)
            last = mvae.last_decode()
        finally:
            vh.ConvIn.__call__, eng.Moments.add = orig_call, orig_add
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
        mvae.set_stripe_rows(96)
        with Spy() as spy:
            out = managed_decode(v, lat, raw=True)
        last = mvae.last_decode()
        ref = native_decode(v, lat, raw=True)
        check("simulated OOM above 24-row stripes: retried with {}-row stripes after {} retries, == native (max|Δ| {:.2g})".format(
            last.get("rows"), last.get("retries"), err(out, ref)),
            last.get("strategy") == "layer1" and last.get("rows") <= 24 and last.get("retries") == 2 and err(out, ref) <= 1e-5
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
        vh.HunyuanStripe.activation_copies, vae_ops.fmt_bytes(est["total"]), vae_ops.fmt_bytes(four["total"])),
        est["total"] == four["total"] + 2 * four["activation"] and est["retained"] == 0)
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
    check("Info node shows the HunyuanImage layer-1 decode and its own scheme hint: {}".format(hint.strip()[:150]),
          "layer 1 (HunyuanImage 2.1 stripes, GroupNorm scheme C" in t and "HunyuanImage 2.1 4K" in hint)
    t2 = getattr(info(), info.FUNCTION)(vae=node_apply(settings, ldm_vae(4, True), gn_scheme="C"))["result"][0]
    check("... an LDM VAE keeps the SDXL hint", "SDXL 4K" in t2 and "HunyuanImage" not in t2)
    l2 = node_apply(settings, v, mode="layer 2 only")
    managed_decode(l2, lat)
    check("VAE Settings node mode 'layer 2 only': layer 2", mvae.last_decode().get("strategy") == "layer2")


# ---------------------------------------------------------------------------
# 7. caching allocator
# ---------------------------------------------------------------------------

def allocator_tests():
    import alloc_sim
    G = float(1 << 30)
    for w, h, scheme in ((1344, 768, None), (2688, 1536, None), (3840, 2176, None), (3840, 2176, "A"), (3840, 2176, "D"), (3840, 2176, "C")):
        i = alloc_sim.decode_trace(w, h, "bf16", None, model="hy21", scheme=scheme)
        check("HunyuanImage 2.1 {}x{} scheme {}: {} x {} rows, recompute {:.2f}x, simulated reserved {:.3f} GiB <= estimate {:.3f} GiB, "
              "{:.3f} stays (output)".format(w, h, scheme or "default (B)", i["stripes"], i["rows"], i["plan"].recompute, i["reserved"] / G,
                                             i["estimate"] / G, i["stays"] / G),
              i["reserved"] <= i["estimate"] and i["stays"] <= i["plan"].out_segment)
    for dt in ("bf16", "fp32"):
        info = {}
        pk, left, bound = alloc_sim.selftest_trace("hy21", dt, probe=True, info=info)
        check("HunyuanImage 2.1 self-test ({} VAE, after the budget's shape probe): simulated reserved peak {:.0f} MiB <= selftest_memory "
              "{:.0f} MiB, nothing left".format(dt, pk / 2 ** 20, bound / 2 ** 20), pk <= bound and left == 0 and info["after_probe"] == 0)
    import vae_inventory as vi
    kind = next(k for k in vi.KINDS if k[0] == "hy_image21")
    v = vi.build(kind)
    for w, h in ((1344, 768), (3840, 2176)):
        shape = vi.latent_shape(v, w, h, 1)
        t = vi.trace(v, shape, 2)
        lat = torch.empty(shape, device="meta", dtype=v.vae_dtype)
        with alloc_sim._patched(mvae.mm, "get_free_memory", lambda dev=None, torch_free_too=False: (48 << 30, 48 << 30) if torch_free_too else 48 << 30):
            p = mvae._probe(v, lat, {})
        est = mvae.estimate(v, lat, mvae.DEFAULT_WORKSPACE, p, **mvae._decoder_terms(v, lat, {}))
        four = mvae.estimate(v, lat, mvae.DEFAULT_WORKSPACE, p)
        check("HunyuanImage 2.1 {}x{} layer 2: simulated reserved {:.2f} GiB <= estimate {:.2f} GiB (6 activation copies; 4 would give {:.2f})".format(
            w, h, t["reserved"] / G, est["total"] / G, four["total"] / G), t["reserved"] <= est["total"])


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
