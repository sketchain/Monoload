"""Layer 1 for the Flux 2 VAE (monoload/vae_ldm.py with a batch-norm latent, DESIGN §9.18).

No model files needed: ComfyUI's own LDM Decoder with small channels and random
weights, post_quant_conv and the latent BatchNorm statistics, through
comfy.sd.VAE (it builds an AutoencoderKL with batch_norm_latent from the
'bn.running_mean' key: latent 128 channels at H/16), fp32 on the CPU unless noted.

  1. recognition: Flux 2-like recognized (its own structure key, the latent step
     first in the prefix); patch size, an affine BatchNorm, a feature count not 4x
     post_quant_conv's input, wrong latent channels -> layer 2 with the reason,
     == native;
  2. the latent step (LatentUnpatch) is bit-identical to what native decode feeds
     post_quant_conv;
  3. whole decoder vs native VAE.decode: schemes A / D / B / C x stripe heights,
     odd latent, batch 2, tiny workspace; bf16 against an fp32 truth at the level
     of native bf16;
  4. self-test: passes, cached separately from an SDXL-like decoder; an injected
     bug in the latent step / stripe-local statistics -> self-test fails,
     layer 2, == native; its reference decode runs in layer-2 row blocks of its
     own size; the first decode's estimate includes it, a budget below a pending
     self-test keeps layer 1 out of the first decode only (DESIGN §9.20);
  5. OOM -> smaller stripes, at the floor MonoloadVAEOOMError, never tiled or
     layer 2; MONOLOAD_VAE_BUDGET picks among layer 2 and the layer-1 schemes;
  6. the Monoload VAE Settings node (forced scheme / rows on a copy) and the
     Monoload Info node apply to it unchanged;
  7. caching allocator (tests/alloc_sim.py, full-size Flux 2 on the meta
     device): reserved <= estimate (1344x768, 2688x1536, 4K; schemes A / B / C /
     D at 4K), and the same as Flux ae's.

    python tests/test_vae_flux2.py
"""

import math

import torch

from common import check, expect_raises, finish
import comfy.ops
import comfy.sd
from comfy.ldm.modules.diffusionmodules import model as ldm
from monoload import vae as mvae
from monoload import vae_engine as eng
from monoload import vae_ldm as vl
from monoload import vae_ops
from monoload.errors import MonoloadError, MonoloadVAEOOMError
from test_vae import Spy, init_random, managed_decode, native_decode
from test_vae_ldm import ldm_vae, no_overrides

torch.manual_seed(0)
OPS = comfy.ops.disable_weight_init
Z = 32          # Flux 2: z 32, latent 4 x 32 = 128 channels


def flux2_vae(ch=32, dtype=None):
    g = torch.Generator().manual_seed(21)
    dec = init_random(ldm.Decoder(ch=ch, out_ch=3, ch_mult=[1, 2, 4, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0,
                                  in_channels=3, resolution=256, z_channels=Z))
    sd = {"decoder." + k: v for k, v in dec.state_dict().items()}
    sd["post_quant_conv.weight"] = torch.randn(Z, Z, 1, 1, generator=g) / math.sqrt(Z)
    sd["post_quant_conv.bias"] = torch.randn(Z, generator=g) * 0.1
    sd["bn.running_mean"] = torch.randn(4 * Z, generator=g) * 0.3
    sd["bn.running_var"] = torch.rand(4 * Z, generator=g) * 2 + 0.2
    sd["bn.num_batches_tracked"] = torch.tensor(1000)
    return comfy.sd.VAE(sd=sd, dtype=dtype)


# ---------------------------------------------------------------------------
# 1, 2. recognition, the latent step
# ---------------------------------------------------------------------------

def recognition_tests():
    v = flux2_vae()
    g = torch.Generator().manual_seed(3)
    lat = torch.randn(1, 4 * Z, 6, 5, generator=g)
    fsm = v.first_stage_model
    bound, why = vl.match(v, lat, {})
    sdxl = ldm_vae(4, True)
    other, _ = vl.match(sdxl, torch.randn(1, 4, 12, 10), {})
    check("Flux 2-like (AutoencoderKL, batch-norm latent, latent {} ch at 1/16): recognized as '{}', latent step first in the prefix ({}), "
          "{} norms with whole-image statistics, decoder at {} (latent {})".format(
              v.latent_channels, bound.name if bound else why, [n for n, _ in bound.prefix[:2]] if bound else "-",
              len(bound.norm_refs()) if bound else 0, bound.decoder_hw(lat) if bound else "-", tuple(lat.shape[-2:])),
          bound is not None and "batch-norm latent" in bound.name and isinstance(bound.prefix[0][1], vl.LatentUnpatch)
          and bound.prefix[1][0] == "post_quant_conv" and len(bound.norm_refs()) == 19 and bound.decoder_hw(lat) == (12, 10)
          and bound.output_shape(lat) == (1, 3, 96, 80), str(why))
    check("its structure key differs from the SDXL-like decoder's (separate self-test)", other is not None and bound.key != other.key)

    # the latent step == the tensor native decode hands post_quant_conv
    seen = []
    h = fsm.post_quant_conv.register_forward_pre_hook(lambda m, args: seen.append(args[0].clone()))
    try:
        native_decode(v, lat)
    finally:
        h.remove()
    mine = vl.LatentUnpatch(fsm)(lat)
    check("LatentUnpatch is bit-identical to native's latent step (BatchNorm undone, 2x2 un-patchified): {} -> {}".format(
        tuple(lat.shape), tuple(mine.shape)), len(seen) == 1 and torch.equal(seen[0], mine))

    cases = []
    bn = fsm.bn

    def case(label, apply, undo, want, decode=True):
        apply()
        try:
            _, why = vl.match(v, lat, {})
            out = ref = last = None
            if decode:
                out = managed_decode(v, lat, raw=True)
                last = mvae.last_decode()
                ref = native_decode(v, lat, raw=True)
        finally:
            undo()
        cases.append((label, why, last, out, ref, want))

    old_ps = list(fsm.ps)
    # (native cannot decode with it either: 128 channels would reach a 32-channel post_quant_conv)
    case("patch size [1, 1]", lambda: setattr(fsm, "ps", [1, 1]), lambda: setattr(fsm, "ps", old_ps), "patch size", decode=False)
    aff = torch.nn.BatchNorm2d(4 * Z, affine=True).eval()
    aff.load_state_dict(dict(bn.state_dict(), weight=torch.ones(4 * Z), bias=torch.zeros(4 * Z)))
    case("affine BatchNorm", lambda: fsm.__dict__["_modules"].__setitem__("bn", aff), lambda: fsm.__dict__["_modules"].__setitem__("bn", bn), "affine")
    for label, why, last, out, ref, want in cases:
        if out is None:
            check("{} -> not recognized: {}".format(label, why), why is not None and want in why)
            continue
        e = float((out - ref).abs().max())
        check("{} -> not recognized, layer 2, == native (max|Δ| {:.2g}): {}".format(label, e, why),
              why is not None and want in why and last.get("strategy") == "layer2" and e <= 1e-4)
    _, why = vl.match(v, torch.randn(1, Z, 12, 10), {})
    check("latent with {} channels (the decoder takes {}) -> no match: {}".format(Z, 4 * Z, why), why is not None and "channels" in why)
    check("... no instance override left on the model", no_overrides(fsm))
    return v


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
    e = float((raw.float() - ref.float()).abs().max())
    passes = 0 if last.get("stripes") == 1 else 19
    check("{}: scheme {}, {} stripes of {} rows, {} statistics passes, saves {}; max|Δ| {:.2g}".format(
        label, scheme, last.get("stripes"), last.get("rows"), last.get("passes"),
        "+".join(vae_ops.fmt_bytes(b) for b in last.get("saves") or []) or "none", e),
        last.get("strategy") == "layer1" and "batch-norm latent" in last.get("adapter", "") and e <= tol and raw.shape == ref.shape
        and last.get("passes") == passes and spy.tiled == 0 and no_overrides(v.first_stage_model),
        "strategy {} note {}".format(last.get("strategy"), last.get("layer1")))
    return last


def decoder_tests(v):
    g = torch.Generator().manual_seed(4)
    lat = torch.randn(1, 4 * Z, 6, 5, generator=g)
    for scheme in "ADBC":
        for rows in (7, 40, None):
            compare("6x5 latent (decoder 12x10), rows {}".format(rows or "default"), v, lat, rows=rows, scheme=scheme)
    compare("odd 7x3 latent, rows 9", v, torch.randn(1, 4 * Z, 7, 3, generator=g), rows=9, scheme="C")
    compare("1x1 latent, rows 5", v, torch.randn(1, 4 * Z, 1, 1, generator=g), rows=5, scheme="A")
    compare("batch 2, rows 16", v, torch.randn(2, 4 * Z, 6, 5, generator=g), rows=16, scheme="B")
    mvae.set_workspace(16 * 1024)
    last = compare("workspace 16 KiB, rows 32", v, lat, rows=32, scheme="B")
    check("  ... conv calls inside the stripes were split into row blocks ({})".format(last["stats"]["conv_chunked"]), last["stats"]["conv_chunked"] > 0)
    mvae.set_workspace(mvae.DEFAULT_WORKSPACE)
    truth = native_decode(v, lat, raw=True).float()
    vb = flux2_vae(dtype=torch.bfloat16)
    nb = native_decode(vb, lat, raw=True).float()
    for scheme in ("A", "B"):
        mvae.set_gn_scheme(scheme)
        mvae.set_stripe_rows(24)
        mb = managed_decode(vb, lat, raw=True).float()
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
    bound, _ = vl.match(v, lat, {})
    ok, detail = eng.self_test(bound, v)
    check("self-test of the Flux 2 structure passes: {}".format(detail), ok)
    sd = ldm_vae(4, True)
    managed_decode(sd, torch.randn(1, 4, 12, 10))
    check("cached separately from the SDXL-like structure ({} entries)".format(len(eng._SELFTEST)), len(eng._SELFTEST) == 2)
    orig_call, orig_add = vl.LatentUnpatch.__call__, eng.Moments.add

    def wrong_step(self, z):           # the BatchNorm undone with eps missing
        o = self.owner
        z = z * torch.sqrt(o.bn.running_var.view(1, -1, 1, 1).to(z)) + o.bn.running_mean.view(1, -1, 1, 1).to(z) + 0.05
        from einops import rearrange
        return rearrange(z, "... (c pi pj) i j -> ... c (i pi) (j pj)", pi=2, pj=2)

    def local_add(self, x, hdim, ws):  # statistics of each stripe on its own
        self.n, self.mean, self.m2 = 0, None, None
        return orig_add(self, x, hdim, ws)

    for label, target, attr, fn in (("a wrong latent step", vl.LatentUnpatch, "__call__", wrong_step),
                                    ("stripe-local GroupNorm statistics", eng.Moments, "add", local_add)):
        eng._SELFTEST.clear()
        setattr(target, attr, fn)
        try:
            out = managed_decode(v, lat, raw=True)
            last = mvae.last_decode()
        finally:
            vl.LatentUnpatch.__call__, eng.Moments.add = orig_call, orig_add
        ref = native_decode(v, lat, raw=True)
        note = last.get("layer1") or ""
        check("injected bug: {} -> self-test fails, layer 2, == native (max|Δ| {:.2g}): {}".format(label, float((out - ref).abs().max()), note[:120]),
              last.get("strategy") == "layer2" and "self-test failed" in note and float((out - ref).abs().max()) <= 1e-4)
    eng._SELFTEST.clear()


def selftest_memory_tests(v, lat):
    """DESIGN §9.20: the self-test's reference decode runs in layer-2 row blocks of its own size; a pending self-test
    is part of the first decode's estimate and of the budget comparison."""
    bound, _ = vl.match(v, lat, {})
    seen = set()
    orig = vae_ops._ConvChunker.__call__

    def spy(self, *a, **kw):
        seen.add(self.budget)
        return orig(self, *a, **kw)
    eng._SELFTEST.clear()
    vae_ops._ConvChunker.__call__ = spy
    try:
        ok, _ = eng.self_test(bound, v)
    finally:
        vae_ops._ConvChunker.__call__ = orig
    check("self-test: the reference decode runs under layer-2 chunking with its own workspace ({}) besides the stripes' ({}); passes".format(
        vae_ops.fmt_bytes(eng.SELFTEST_REF_WORKSPACE), vae_ops.fmt_bytes(eng.SELFTEST_WORKSPACE)),
        ok and seen == {eng.SELFTEST_REF_WORKSPACE, eng.SELFTEST_WORKSPACE})

    # the first decode's record: estimate = max(plan, self-test); a later decode: the plan's
    eng._SELFTEST.clear()
    managed_decode(v, lat)
    e1 = mvae.last_decode()["estimate"]
    managed_decode(v, lat)
    e2 = mvae.last_decode()["estimate"]
    st = bound.selftest_memory()
    check("first decode records the self-test ({} -> estimate {} = max(plan {}, self-test)); the next one does not (estimate {})".format(
        vae_ops.fmt_bytes(e1.get("selftest")), vae_ops.fmt_bytes(e1["total"]), vae_ops.fmt_bytes(e1["plan"]), vae_ops.fmt_bytes(e2["total"])),
        e1.get("selftest") == st and e1["total"] == max(e1["plan"], st) and e2.get("selftest") == 0 and e2["total"] == e2["plan"])

    # budget: a pending self-test larger than the budget keeps layer 1 out of the first decode, not of later ones
    est2 = mvae._layer2_estimate(v, lat, {}, mvae.workspace())[0]["total"]
    bud = est2 - 1                                   # layer 2 does not fit
    orig_sm = vl.LDMStripe.selftest_memory
    vl.LDMStripe.selftest_memory = lambda self: bud + 1
    eng._SELFTEST.clear()
    mvae.set_budget(bud)
    try:
        expect_raises("budget below a pending self-test (and layer 2): MonoloadError naming the self-test", MonoloadError,
                      lambda: managed_decode(v, lat), "self-test")
        for b in bound.variants():
            eng._SELFTEST[b.key] = (True, "stub")
        out = managed_decode(v, lat, raw=True)
        last = mvae.last_decode()
    finally:
        vl.LDMStripe.selftest_memory = orig_sm
        mvae.set_budget(None)
        eng._SELFTEST.clear()
    ref = native_decode(v, lat, raw=True)
    check("... once the self-tests have run, the same budget gives layer 1 ({}), == native (max|Δ| {:.2g})".format(
        last.get("adapter"), float((out - ref).abs().max())), last.get("strategy") == "layer1" and float((out - ref).abs().max()) <= 1e-5)


# ---------------------------------------------------------------------------
# 5. OOM, budget
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
            last.get("rows"), last.get("retries"), float((out - ref).abs().max())),
            last.get("strategy") == "layer1" and last.get("rows") <= 24 and last.get("retries") == 2
            and float((out - ref).abs().max()) <= 1e-5 and spy.tiled == 0 and calls_l2[0] == 0)
        eng.run_passes = oom_above(0)
        with Spy() as spy:
            expect_raises("OOM even with the smallest stripes -> MonoloadVAEOOMError", MonoloadVAEOOMError,
                          lambda: managed_decode(v, lat), "never falls back to the approximate tiled", "nor to layer 2")
        check("... neither tiled nor layer 2 was called, no override left", spy.tiled == 0 and calls_l2[0] == 0 and no_overrides(v.first_stage_model))
    finally:
        eng.run_passes, eng.run_stripes, mvae._run = orig_run, orig_stripes, orig_l2
        mvae.set_stripe_rows(None)
        mvae.set_gn_scheme(None)

    # budget: layer 2 when it fits, else the fastest layer-1 scheme within it, nothing fits -> an error naming the needs
    est = mvae._layer2_estimate(v, lat, {}, mvae.workspace())[0]["total"]
    for bud, want in ((est, "layer2"), (est - 1, "layer1")):
        mvae.set_budget(bud)
        try:
            out = managed_decode(v, lat, raw=True)
            last = mvae.last_decode()
        finally:
            mvae.set_budget(None)
        ref = native_decode(v, lat, raw=True)
        check("budget {} ({} the layer-2 estimate): {}{}, == native (max|Δ| {:.2g})".format(
            vae_ops.fmt_bytes(bud), "=" if bud == est else "just below", last.get("strategy"),
            " ({})".format(last.get("adapter")) if last.get("adapter") else "", float((out - ref).abs().max())),
            last.get("strategy") == want and float((out - ref).abs().max()) <= 1e-5)
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
    copy_ = node_apply(settings, v, gn_scheme="C", stripe_rows=16)
    out = managed_decode(copy_, lat, raw=True)
    last = mvae.last_decode()
    ref = native_decode(v, lat, raw=True)
    check("VAE Settings node on Flux 2 (scheme C, 16 rows forced on a copy): {} stripes of {} rows, scheme {} [{}], == native (max|Δ| {:.2g})".format(
        last.get("stripes"), last.get("rows"), last.get("gn_scheme"), (last.get("settings_source") or {}).get("gn_scheme"),
        float((out - ref).abs().max())),
        last.get("strategy") == "layer1" and last.get("gn_scheme") == "C" and last.get("rows") == 16
        and (last.get("settings_source") or {}).get("gn_scheme") == "node" and float((out - ref).abs().max()) <= 1e-5)
    t = getattr(info(), info.FUNCTION)(vae=copy_)["result"][0]
    line = next((l for l in t.splitlines() if "last decode" in l), "")
    check("Info node shows the Flux 2 layer-1 decode: {}".format(line.strip()[:150]),
          "layer 1 (LDM stripes (batch-norm latent), GroupNorm scheme C" in t and "GroupNorm scheme C:" in t)
    l2 = node_apply(settings, v, mode="layer 2 only")
    managed_decode(l2, lat)
    check("VAE Settings node mode 'layer 2 only' on Flux 2: layer 2", mvae.last_decode().get("strategy") == "layer2")


# ---------------------------------------------------------------------------
# 7. caching allocator
# ---------------------------------------------------------------------------

def allocator_tests():
    import alloc_sim
    G = float(1 << 30)
    for model in ("flux2", "sdxl"):
        tail = {}
        pk, left, bound = alloc_sim.selftest_trace(model, info=tail)
        pk0 = alloc_sim.selftest_trace(model, tail=False)[0]
        check("{} self-test (full size, meta): simulated reserved peak {:.0f} MiB <= selftest_memory {:.0f} MiB, nothing left ({:.0f} MiB); "
              "was ~740 MiB with the reference unchunked; the final comparison allocates {:.2f} MiB, adds {:.2f} MiB to reserved "
              "({:.0f} MiB before it), peak {:.0f} MiB without it".format(
                  model, pk / 2 ** 20, bound / 2 ** 20, left / 2 ** 20, tail["tail_alloc"] / 2 ** 20, tail["tail_reserved"] / 2 ** 20,
                  tail["before_tail"] / 2 ** 20, pk0 / 2 ** 20),
              pk <= bound and left == 0 and pk <= 0.45 * G and tail["tail_alloc"] > 0 and tail["before_tail"] + tail["tail_reserved"] <= bound)
    # review 2026-10 item 21: with a budget the first decode runs the layer-2 shape probe right before the first-use
    # self-test; the cache is emptied in between, so the self-test's peak is what it is alone (before: Wan fp32 426 of
    # its 430 MiB bound, the probe's cached blocks in the way)
    rows = []
    ok = True
    for model in ("qwen", "flux2"):
        info = {}
        pk_p, left_p, bound = alloc_sim.selftest_trace(model, "fp32", info=info, probe=True)
        pk = alloc_sim.selftest_trace(model, "fp32")[0]
        ok = ok and info["after_probe"] == 0 and pk_p == pk <= bound and left_p == 0
        rows.append("{} fp32 {:.0f} MiB after the probe ({:.0f} alone, bound {:.0f}, reserved when the self-test starts {:.0f})".format(
            model, pk_p / 2 ** 20, pk / 2 ** 20, bound / 2 ** 20, info["after_probe"] / 2 ** 20))
    check("self-test right after the budget's shape probe: the same peak as alone, nothing of the probe left: " + "; ".join(rows), ok)
    for w, h, scheme in ((1344, 768, None), (2688, 1536, None), (3840, 2160, None), (3840, 2160, "A"), (3840, 2160, "C"), (3840, 2160, "D")):
        i = alloc_sim.decode_trace(w, h, "bf16", None, model="flux2", scheme=scheme)
        f = alloc_sim.decode_trace(w, h, "bf16", None, model="flux", scheme=scheme)
        check("Flux 2 {}x{} scheme {}: {} x {} rows, simulated reserved {:.3f} GiB <= estimate {:.3f} GiB (Flux ae: {:.3f} / {:.3f})".format(
            w, h, scheme or "default (B)", i["stripes"], i["rows"], i["reserved"] / G, i["estimate"] / G, f["reserved"] / G, f["estimate"] / G),
            i["reserved"] <= i["estimate"] and abs(i["reserved"] - f["reserved"]) <= 0.02 * G)
    # CT 700 command AA (DESIGN §9.21, §9.22): 4K, the budget-3G plan (B, 12 x 180 rows, workspace 128 MiB). The first
    # decode with a budget runs the layer-2 shape probe first; its blocks, left cached, took the decode's first requests
    # (2.57 GiB measured vs 2.44); and the output in the arena kept all of it reserved after the decode (2.44 measured).
    # Now the cache is emptied before the output and the arena are allocated, and again after the decode.
    kw = dict(w=3840, h=2160, dtype="bf16", rows=180, model="flux2", scheme="B", ws=128 << 20)
    o1 = alloc_sim.decode_trace(probe=True, **dict(kw, **alloc_sim._version("w128")))
    o2 = alloc_sim.decode_trace(**dict(kw, **alloc_sim._version("w128")))
    n1 = alloc_sim.decode_trace(probe=True, **kw)
    n2 = alloc_sim.decode_trace(**kw)
    check("Flux 2 4K B 12 x 180 (budget 3G): decode 1 (after the shape probe) {:.3f} / decode 2 {:.3f} GiB, {:.3f} stays (output {:.3f}), "
          "estimate {:.3f}; up to 6324592 {:.3f} / {:.3f}, {:.3f} stayed (measured 2.57 / 2.44, 2.44; bench Z 4K -b3 GTT 2.44)".format(
              n1["reserved"] / G, n2["reserved"] / G, n2["stays"] / G, n2["plan"].out_segment / G, n2["estimate"] / G,
              o1["reserved"] / G, o2["reserved"] / G, o2["stays"] / G),
          abs(n1["reserved"] - n2["reserved"]) <= 0.005 * G and n1["stays"] == n2["stays"] <= n2["plan"].out_segment
          and n2["reserved"] <= n2["estimate"] and n2["reserved"] <= o2["reserved"] + 0.01 * G
          and o1["reserved"] > o2["reserved"] + 0.05 * G and o2["stays"] >= o2["arena"] - 0.01 * G
          and abs(o1["reserved"] / G - 2.57) <= 0.03 and abs(o2["reserved"] / G - 2.44) <= 0.03)
    # scheme A (no saves), 4K, 17 x 128 rows, workspace 128 MiB (budget 1.5G): the prefix leaves a hole of the checkpoint's
    # size (127 MiB) in front of it, which the output used to fill; without it a 128 MiB column block was stranded outside
    # the arena. move_low puts the checkpoint into the hole.
    from monoload import vae_engine as eng
    kw = dict(w=3840, h=2160, dtype="bf16", rows=128, model="flux2", scheme="A", ws=128 << 20)
    o = alloc_sim.decode_trace(**dict(kw, **alloc_sim._version("w128")))
    eng.CKPT_LOW = False
    try:
        off = alloc_sim.decode_trace(**kw)
    finally:
        eng.CKPT_LOW = True
    i = alloc_sim.decode_trace(**kw)
    check("Flux 2 4K A 17 x 128 (budget 1.5G): reserved {:.3f} GiB = output {:.3f} + arena {:.3f} (estimate {:.3f}); up to 6324592 {:.3f}; "
          "without moving the checkpoint {:.3f} (bench Z 4K -b1.5 GTT 1.11)".format(i["reserved"] / G, i["plan"].out_segment / G, i["arena"] / G, i["estimate"] / G,
                                                        o["reserved"] / G, off["reserved"] / G),
          i["reserved"] <= o["reserved"] + 0.005 * G and i["reserved"] <= i["plan"].out_segment + i["arena"] + 0.005 * G
          and off["reserved"] > i["reserved"] + 0.1 * G and abs(o["reserved"] / G - 1.11) <= 0.03)


def main():
    if not mvae.is_installed():
        check("vae.install() on this ComfyUI", mvae.install())
    mvae.set_stripe(True)
    v = recognition_tests()
    lat = decoder_tests(v)
    selftest_tests(v, lat)
    selftest_memory_tests(v, lat)
    oom_budget_tests(v, lat)
    node_tests(v, lat)
    allocator_tests()
    finish()


if __name__ == "__main__":
    main()
