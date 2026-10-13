"""Layer 1 for the Wan 2.2 VAE single frame (monoload/vae_wan22.py, DESIGN §9.23).

No model files needed: ComfyUI's own vae2_2.WanVAE with small channels (dim 16,
decoder dim 32) and random weights, through comfy.sd.VAE (it detects the Wan 2.2
VAE from the state dict; its hard-coded dim 160 / decoder dim 256 is swapped for
the small one while it builds), latent 48 channels at 1/16, 5D T=1, fp32 on the
CPU unless noted.

  1. recognition: Wan 2.2 recognized by its structure (prefix conv2 / conv1 /
     middle, compound up blocks with halo 13, the last level's ResidualBlocks,
     head, unpatchify); its key differs from Wan 2.1's, and neither adapter takes
     the other's model; what it does not handle (a non-zero padding mode, a
     bilinear upsampling, a forward hook, vae_options, a wrong channel count) ->
     layer 2 with the reason, == native; a multi-frame latent stays native;
  2. whole decoder vs native VAE.decode: stripe heights 7 / 40 / default, odd
     latent, 1x1 latent, batch 2, 16 KiB workspace (row blocks inside the
     stripes); bf16 against an fp32 truth at the level of native bf16;
  3. the compound unit's halo (13) is the exact boundary of its output rows that
     a cut slice changes; the self-test's fp32 copy keeps each CausalConv3d's last time slice only:
     its own WanVAE.decode equals the model's, selftest_params counts it;
  4. self-test: passes, cached apart from Wan 2.1; injected bugs (a halo far too
     small, a wrong unpatchify) -> self-test fails, layer 2, == native; the
     first decode's estimate includes the self-test;
  5. OOM -> smaller stripes, at the floor MonoloadVAEOOMError, never tiled or
     layer 2; the layer-2 estimate counts the feat_cache clones WanVAE.decode
     keeps (decode_retained); MONOLOAD_VAE_BUDGET: layer 2 when its estimate
     fits, else layer 1, else an error naming both;
  6. the Monoload VAE Settings node (forced rows, layer 2 only) and the Info node;
  7. caching allocator (tests/alloc_sim.py, full-size Wan 2.2 on the meta
     device): layer 1 reserved <= estimate (1344x768, 2688x1536, 4K default;
     4K 32 / 512 rows; fp32 4K), the self-test's peak <= its bound, layer 2's
     simulated peak <= the layer-2 estimate (1344x768, 4K).

    python tests/test_vae_wan22.py
"""

import torch
from einops import rearrange

from common import check, expect_raises, finish
import comfy.sd
import comfy.ldm.wan.vae as wan
import comfy.ldm.wan.vae2_2 as wan22
from monoload import vae as mvae
from monoload import vae_engine as eng
from monoload import vae_ops
from monoload import vae_wan, vae_wan22 as w22
from monoload.errors import MonoloadError, MonoloadVAEOOMError
from test_vae import Spy, init_random, managed_decode, native_decode, wan_vae_model
from test_vae_ldm import no_overrides

torch.manual_seed(0)
Z = 48
CFG = dict(dim=160, z_dim=Z, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)
_REAL = wan22.WanVAE


def _small(**kw):
    return _REAL(**dict(kw, dim=16, dec_dim=32))


def wan22_model():
    return init_random(_small(**CFG))


def wan22_vae(dtype=None, model=None):
    """comfy.sd.VAE on a small Wan 2.2 state dict (sd.py builds vae2_2.WanVAE(dim=160, ...): the small one instead)."""
    m = model if model is not None else wan22_model()
    wan22.WanVAE = _small
    try:
        return comfy.sd.VAE(sd=m.state_dict(), dtype=dtype)
    finally:
        wan22.WanVAE = _REAL


def lat_(h, w, b=1, seed=4):
    return torch.randn(b, Z, 1, h, w, generator=torch.Generator().manual_seed(seed))


def err(a, b):
    return float((a.float() - b.float()).abs().max())


# ---------------------------------------------------------------------------
# 1. recognition
# ---------------------------------------------------------------------------

def recognition_tests():
    v = wan22_vae()
    fsm = v.first_stage_model
    lat = lat_(6, 5)
    bound, why = w22.match(v, lat, {})
    kinds = [(u.kind, u.halo, u.scale) for u in bound.units] if bound else []
    check("Wan 2.2 (vae2_2.WanVAE, latent {} ch at 1/{}) recognized as '{}': prefix {}, units {}".format(
        v.latent_channels, v.spacial_compression_decode(), bound.name if bound else why, [n for n, _ in bound.prefix] if bound else "-", kinds),
        bound is not None and type(fsm) is _REAL and [n for n, _ in bound.prefix] == ["conv2", "decoder.conv1", "decoder.middle.0", "decoder.middle.1",
                                                                                  "decoder.middle.2"]
        and kinds == [("dupup", 13, 2)] * 3 + [("res", 2, 1)] * 3 + [("point", 0, 1)] * 2 + [("conv", 1, 1), ("unpatch", 0, 2)]
        and bound.scale == 16 and bound.output_shape(lat) == (1, 3, 1, 96, 80) and bound.ckpt_channels == 128, str(why))
    w21 = wan_vae_model()
    b21, _ = vae_wan.match(w21, torch.randn(1, 16, 1, 6, 5), {})
    _, why_a = vae_wan.match(v, lat, {})
    _, why_b = w22.match(w21, torch.randn(1, 16, 1, 6, 5), {})
    check("keys differ from Wan 2.1 (separate self-tests); the Wan 2.1 adapter does not take it ({}), nor this one Wan 2.1 ({})".format(why_a, why_b),
          b21 is not None and b21.key != bound.key and "vae2_2" in why_a and "comfy.ldm.wan.vae.WanVAE" in why_b)
    sel, _ = mvae._select_layer1(v, lat, {})
    check("vae._select_layer1 picks the Wan 2.2 adapter", sel is not None and sel.name == "Wan 2.2 stripes")

    up_conv = fsm.decoder.upsamples[1].upsamples[3].resample[1]
    up = fsm.decoder.upsamples[0].upsamples[3].resample[0]
    rb_conv = fsm.decoder.upsamples[3].upsamples[1].residual[2]
    cases = [("reflect padding in an upsampling conv", up_conv, "padding_mode", "reflect", "upsamples[1].upsamples[3]"),
             ("bilinear upsampling", up, "mode", "bilinear", "upsample mode bilinear")]
    for label, mod, attr, val, want in cases:
        old = getattr(mod, attr)
        setattr(mod, attr, val)
        try:
            _, why = w22.match(v, lat, {})
            out = managed_decode(v, lat, raw=True)
            last = mvae.last_decode()
            ref = native_decode(v, lat, raw=True)
        finally:
            setattr(mod, attr, old)
        check("{} -> not recognized, layer 2, == native (max|Δ| {:.2g}): {}".format(label, err(out, ref), why),
              why is not None and want in why and last.get("strategy") == "layer2" and err(out, ref) <= 1e-5)
    h = rb_conv.register_forward_hook(lambda m, a, o: None)
    try:
        _, why = w22.match(v, lat, {})
        out = managed_decode(v, lat, raw=True)
        last = mvae.last_decode()
        ref = native_decode(v, lat, raw=True)
    finally:
        h.remove()
    check("a forward hook in the decoder -> layer 2, == native (max|Δ| {:.2g}): {}".format(err(out, ref), why),
          why is not None and "hook" in why and last.get("strategy") == "layer2" and err(out, ref) <= 1e-5)
    _, why = w22.match(v, lat, {"x": 1})
    check("vae_options given -> no match: {}".format(why), why is not None and "vae_options" in why)
    _, why = w22.match(v, torch.randn(1, 16, 1, 6, 5), {})
    check("latent with 16 channels -> no match: {}".format(why), why is not None and "channels" in why)
    _, why = w22.match(v, lat_(6, 5).expand(1, Z, 3, 6, 5), {})
    reason = mvae._native_reason(v, torch.randn(1, Z, 3, 6, 5))
    check("3 latent frames: no layer-1 match ({}), and the decode stays native ({})".format(why, reason),
          why is not None and "single frame" in why and reason is not None and "multi-frame" in reason)
    check("... no instance override left on the model", no_overrides(fsm))
    return v


# ---------------------------------------------------------------------------
# 2. whole decoder
# ---------------------------------------------------------------------------

def compare(label, v, latent, rows=None, tol=1e-5):
    mvae.set_stripe_rows(rows)
    try:
        with Spy() as spy:
            raw = managed_decode(v, latent, raw=True)
    finally:
        mvae.set_stripe_rows(None)
    last = mvae.last_decode()
    ref = native_decode(v, latent, raw=True)
    e = err(raw, ref)
    check("{}: {} stripes of {} rows, recompute {:.2f}x; max|Δ| {:.2g}".format(label, last.get("stripes"), last.get("rows"), last.get("recompute") or 0, e),
          last.get("strategy") == "layer1" and last.get("adapter") == "Wan 2.2 stripes" and e <= tol and raw.shape == ref.shape
          and spy.tiled == 0 and no_overrides(v.first_stage_model), "strategy {} note {}".format(last.get("strategy"), last.get("layer1")))
    return last


def decoder_tests(v):
    lat = lat_(6, 5)
    for rows in (7, 40, None):
        compare("6x5 latent (output 96x80), rows {}".format(rows or "default"), v, lat, rows=rows)
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
          out.shape == ref.shape and err(out, ref) <= 1e-5)
    m = wan22_model()
    truth = native_decode(wan22_vae(model=m), lat, raw=True).float()
    vb = wan22_vae(dtype=torch.bfloat16, model=m)
    nb = native_decode(vb, lat, raw=True).float()
    mvae.set_stripe_rows(24)
    try:
        mb = managed_decode(vb, lat, raw=True).float()
    finally:
        mvae.set_stripe_rows(None)
    last = mvae.last_decode()
    rn = float((nb - truth).pow(2).mean().sqrt())
    rm = float((mb - truth).pow(2).mean().sqrt())
    check("bf16 VAE: RMSE vs fp32 truth: layer 1 {:.3g}, native {:.3g}".format(rm, rn), last.get("strategy") == "layer1" and rm <= 1.5 * rn + 1e-4)
    return lat


def halo_tests(v):
    """The compound unit's halo is the exact boundary: a slice that starts at a row that is not the image's top
    gets rows 0 .. halo-1 of its output wrong (zero padding) and every later row bit-identical."""
    fsm = v.first_stage_model
    for i in range(3):
        blk = fsm.decoder.upsamples[i]
        c = blk.upsamples[0].in_dim
        x = torch.randn(1, c, 1, 40, 12, generator=torch.Generator().manual_seed(9 + i))
        xa = 10
        with torch.inference_mode():
            whole = blk(x, None, [0], True)
            part = blk(x[:, :, :, xa:], None, [0], True)
        d = (part - whole[:, :, :, 2 * xa:]).abs().amax(dim=(0, 1, 2, 4))
        bad = [r for r in range(d.shape[0] - 20) if float(d[r]) > 0]
        check("decoder.upsamples[{}] on rows from {}: output rows {} .. {} after the cut differ (row 12 by {:.1g}), the rest are bit-identical: "
              "halo 13 = 4 x 3 blocks + 1".format(i, xa, bad[0] if bad else "-", bad[-1] if bad else "-", float(d[12])),
              bad == list(range(13)) and float(d[13:].abs().max()) == 0)


# ---------------------------------------------------------------------------
# 3. the fp32 copy, 4. self-test
# ---------------------------------------------------------------------------

def copy_tests(v, lat):
    bound, _ = w22.match(v, lat, {})
    tb = bound.fp32_copy(torch.device("cpu"))
    n_copy = sum(p.numel() for p in tb.module.parameters())
    n_full = sum(p.numel() for p in v.first_stage_model.decoder.parameters()) + sum(p.numel() for p in v.first_stage_model.conv2.parameters())
    z = lat_(5, 4, seed=8)
    with torch.inference_mode():
        a = tb.reference_decode(z)
        b = v.first_stage_model.decode(z)
    kts = {tuple(m.kernel_size) for m in tb.module.modules() if isinstance(m, wan.CausalConv3d)}
    check("fp32 copy: each CausalConv3d keeps its last time slice (kernels {}), {} of {} parameters = selftest_params(); its WanVAE.decode "
          "== the model's (max|Δ| {:.2g})".format(sorted(kts), n_copy, n_full, err(a, b)),
          n_copy == bound.selftest_params() and n_copy < n_full and all(k[0] == 1 for k in kts) and tb.key == bound.key and err(a, b) <= 1e-6)


def selftest_tests(v, lat):
    eng._SELFTEST.clear()
    bound, _ = w22.match(v, lat, {})
    ok, detail = eng.self_test(bound, v)
    check("self-test of the Wan 2.2 structure passes: {}".format(detail), ok)
    managed_decode(wan_vae_model(), torch.randn(1, 16, 1, 6, 5))
    check("cached apart from Wan 2.1 ({} entries)".format(len(eng._SELFTEST)), len(eng._SELFTEST) == 2)

    orig_build, orig_unpatch = w22.build_units, w22._unpatch

    def small_halo(fsm):
        prefix, units = orig_build(fsm)
        units[2].halo = 1            # needs 13 rows at its output level (a shortfall of a few rows is below the
                                     # tolerance: the zero padding's effect decays, see halo_tests)
        return prefix, units

    def wrong_unpatch(x):            # q / r swapped: a transposed 2x2 patch
        return rearrange(x, "b (c q r) f h w -> b c f (h q) (w r)", q=2, r=2)

    for label, name, fn, want in (("a compound unit's halo far too small", "build_units", small_halo, "max|stripes - whole|"),
                                  ("a wrong unpatchify", "_unpatch", wrong_unpatch, "max|stripes - whole|")):
        eng._SELFTEST.clear()
        setattr(w22, name, fn)
        try:
            with Spy() as spy:
                out = managed_decode(v, lat, raw=True)
            last = mvae.last_decode()
        finally:
            w22.build_units, w22._unpatch = orig_build, orig_unpatch
        ref = native_decode(v, lat, raw=True)
        note = last.get("layer1") or ""
        check("injected bug: {} -> self-test fails, layer 2, == native (max|Δ| {:.2g}): {}".format(label, err(out, ref), note[:140]),
              last.get("strategy") == "layer2" and "self-test failed" in note and want in note and err(out, ref) <= 1e-5 and spy.tiled == 0)
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
    managed_decode(v, lat)      # self-test outside the injected OOM
    orig_stripes, orig_l2 = eng.run_stripes, mvae._run
    calls_l2 = [0]

    def l2(*a, **kw):
        calls_l2[0] += 1
        return orig_l2(*a, **kw)

    def oom_above(limit):
        def stripes(units, plan, *a, **kw):
            if max(b - a_ for a_, b in plan.stripes) > limit:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return orig_stripes(units, plan, *a, **kw)
        return stripes
    mvae._run = l2
    try:
        eng.run_stripes = oom_above(24)
        mvae.set_stripe_rows(96)
        with Spy() as spy:
            out = managed_decode(v, lat, raw=True)
        last = mvae.last_decode()
        ref = native_decode(v, lat, raw=True)
        check("simulated OOM above 24-row stripes: retried with {}-row stripes after {} retries, == native (max|Δ| {:.2g})".format(
            last.get("rows"), last.get("retries"), err(out, ref)),
            last.get("strategy") == "layer1" and last.get("rows") <= 24 and last.get("retries") == 2 and err(out, ref) <= 1e-5
            and spy.tiled == 0 and calls_l2[0] == 0)
        eng.run_stripes = oom_above(0)
        with Spy() as spy:
            expect_raises("OOM even with the smallest stripes -> MonoloadVAEOOMError", MonoloadVAEOOMError,
                          lambda: managed_decode(v, lat), "never falls back to the approximate tiled", "nor to layer 2")
        check("... neither tiled nor layer 2 was called, no override left", spy.tiled == 0 and calls_l2[0] == 0 and no_overrides(v.first_stage_model))
    finally:
        eng.run_stripes, mvae._run = orig_stripes, orig_l2
        mvae.set_stripe_rows(None)

    bound, _ = w22.match(v, lat, {})
    est, _ = mvae._layer2_estimate(v, lat, {}, mvae.workspace())
    want = bound.decode_retained(*bound.decoder_hw(lat), 4)
    # the clones WanVAE.decode keeps: count them in a native decode
    kept = []
    orig_fwd = wan.CausalConv3d.forward

    def fwd(self, x, cache_x=None, cache_list=None, cache_idx=None):
        if cache_list is not None or cache_x is not None or self is v.first_stage_model.decoder.conv1 or self is v.first_stage_model.decoder.head[2]:
            kept.append(x.numel() * x.element_size())
        return orig_fwd(self, x, cache_x, cache_list, cache_idx)
    wan.CausalConv3d.forward = fwd
    try:
        native_decode(v, lat)
    finally:
        wan.CausalConv3d.forward = orig_fwd
    check("layer-2 estimate counts the feat_cache clones WanVAE.decode keeps: retained {} = the inputs of the {} cached CausalConv3d calls "
          "of a native decode ({})".format(vae_ops.fmt_bytes(est["retained"]), len(kept), vae_ops.fmt_bytes(sum(kept))),
          est["retained"] == want == sum(kept) > 0)
    for bud, want_s in ((est["total"], "layer2"), (est["total"] - 1, "layer1")):
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
            last.get("strategy") == want_s and err(out, ref) <= 1e-5)
    mvae.set_budget(1024)
    try:
        expect_raises("budget 1 KiB: nothing fits -> MonoloadError naming each candidate's need", MonoloadError,
                      lambda: managed_decode(v, lat), "layer 2", "layer 1")
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
    copy_ = node_apply(settings, v, stripe_rows=16)
    out = managed_decode(copy_, lat, raw=True)
    last = mvae.last_decode()
    ref = native_decode(v, lat, raw=True)
    check("VAE Settings node on Wan 2.2 (16 rows forced on a copy): {} stripes of {} rows [{}], == native (max|Δ| {:.2g})".format(
        last.get("stripes"), last.get("rows"), (last.get("settings_source") or {}).get("stripe_rows"), err(out, ref)),
        last.get("strategy") == "layer1" and last.get("rows") == 16 and (last.get("settings_source") or {}).get("stripe_rows") == "node"
        and err(out, ref) <= 1e-5)
    t = getattr(info(), info.FUNCTION)(vae=copy_)["result"][0]
    line = next((l for l in t.splitlines() if "last decode" in l), "")
    check("Info node shows the Wan 2.2 layer-1 decode: {}".format(line.strip()[:150]), "layer 1 (Wan 2.2 stripes" in t)
    l2 = node_apply(settings, v, mode="layer 2 only")
    managed_decode(l2, lat)
    check("VAE Settings node mode 'layer 2 only' on Wan 2.2: layer 2", mvae.last_decode().get("strategy") == "layer2")


# ---------------------------------------------------------------------------
# 7. caching allocator
# ---------------------------------------------------------------------------

def allocator_tests():
    import alloc_sim
    G = float(1 << 30)
    for w, h, rows, dt in ((1344, 768, None, "bf16"), (2688, 1536, None, "bf16"), (3840, 2160, None, "bf16"), (3840, 2160, 32, "bf16"),
                           (3840, 2160, 512, "bf16"), (3840, 2160, None, "fp32")):
        i = alloc_sim.decode_trace(w, h, dt, rows, model="wan22")
        check("Wan 2.2 {}x{} {} rows {}: {} x {} rows, simulated reserved {:.3f} GiB <= estimate {:.3f} GiB, {:.3f} stays (output)".format(
            w, h, dt, rows or "default", i["stripes"], i["rows"], i["reserved"] / G, i["estimate"] / G, i["stays"] / G),
            i["reserved"] <= i["estimate"] and i["stays"] <= i["plan"].out_segment)
    for dt in ("bf16", "fp32"):
        info = {}
        pk, left, bound = alloc_sim.selftest_trace("wan22", dt, probe=True, info=info)
        check("Wan 2.2 self-test ({} VAE, after the budget's shape probe): simulated reserved peak {:.0f} MiB <= selftest_memory {:.0f} MiB, "
              "nothing left".format(dt, pk / 2 ** 20, bound / 2 ** 20), pk <= bound and left == 0 and info["after_probe"] == 0)
    import vae_inventory as vi
    kind = next(k for k in vi.KINDS if k[0] == "wan22")
    v = vi.build(kind)
    for w, h in ((1344, 768), (3840, 2160)):
        shape = vi.latent_shape(v, w, h, 1)
        t = vi.trace(v, shape, 2)
        lat = torch.empty(shape, device="meta", dtype=v.vae_dtype)
        with alloc_sim._patched(mvae.mm, "get_free_memory", lambda dev=None, torch_free_too=False: (48 << 30, 48 << 30) if torch_free_too else 48 << 30):
            p = mvae._probe(v, lat, {})
        est = mvae.estimate(v, lat, mvae.DEFAULT_WORKSPACE, p, **mvae._decoder_terms(v, lat, {}))
        check("Wan 2.2 {}x{} layer 2: simulated reserved {:.2f} GiB <= estimate {:.2f} GiB (of which feat_cache clones {:.2f}; without them "
              "{:.2f})".format(w, h, t["reserved"] / G, est["total"] / G, est["retained"] / G, (est["total"] - est["retained"]) / G),
              t["reserved"] <= est["total"])


def main():
    if not mvae.is_installed():
        check("vae.install() on this ComfyUI", mvae.install())
    mvae.set_stripe(True)
    v = recognition_tests()
    lat = decoder_tests(v)
    halo_tests(v)
    copy_tests(v, lat)
    selftest_tests(v, lat)
    oom_budget_tests(v, lat)
    node_tests(v, lat)
    allocator_tests()
    finish()


if __name__ == "__main__":
    main()
