"""Layer 1 for the LDM decoder (monoload/vae_ldm.py) and the GroupNorm statistics
across stripes (monoload/vae_engine.py, DESIGN §9.14).

No model files needed: ComfyUI's own Decoder with small channels and random
weights, through comfy.sd.VAE (SDXL-like AutoencoderKL with post_quant_conv,
Flux-like AutoencodingEngine), fp32 on the CPU unless noted.

  1. statistics: Moments (fp32 var_mean per chunk, Chan merge) against fp64
     over random chunkings, also with a large mean (where E[x^2] - E[x]^2 in
     fp32 breaks down); group_norm_frozen == F.group_norm with the whole
     tensor's statistics (fp32 and bf16, whole tensor and row slices);
     GlobalNorms installs / restores the instance forward, uses the comfy.ops
     weight path (weight_function), refuses a norm without statistics;
  2. recognition: SDXL-like / Flux-like recognized; attention in an up level,
     tanh_out, give_pre_end, carried 3D convs, a non-2x upsample, an upsample
     without conv, batch_norm_latent, a 3x3 conv shortcut, Dropout in training,
     a forward hook, an instance forward, vae_options, wrong latent channels
     -> layer 2 with the reason, result == native;
  3. whole decoder vs native VAE.decode: schemes A / D / B / C x stripe heights
     1, 7, 40, default; odd / tiny latents, batch 2, wide groups (ch 64),
     tiny workspace (conv row blocks and 1-row norm chunks); errors near stripe
     boundaries no larger than elsewhere; bf16 against an fp32 truth at the
     level of native bf16;
  4. self-test: stripe-local statistics, a lost stripe and a halo one row short
     are caught (layer 2, == native); one self-test per structure and scheme;
  5. plans: 19 statistics passes, saves per scheme, live = persistent +
     max(prefix, final, passes), each pass within the peak floor, recompute
     and arena ordered C < B < D < A / A < D < B < C on the full-size SDXL
     decoder; OOM -> smaller stripes, at the floor MonoloadVAEOOMError, never
     tiled or layer 2; MONOLOAD_VAE_GN_SCHEME;
  6. caching allocator (tests/alloc_sim.py, full-size SDXL / Flux on the meta
     device): SDXL / Flux layer-2 CT 700 readings reproduced; layer-1 reserved
     <= estimate.

    python tests/test_vae_ldm.py
"""

import math

import torch
import torch.nn.functional as F

from common import check, expect_raises, finish
import comfy.model_management
import comfy.ops
from comfy.ldm.modules.diffusionmodules import model as ldm
from monoload import vae as mvae
from monoload import vae_engine as eng
from monoload import vae_ldm as vl
from monoload import vae_ops
from monoload.errors import MonoloadVAEOOMError
from test_vae import Spy, init_random, managed_decode, native_decode
import comfy.sd

torch.manual_seed(0)
OPS = comfy.ops.disable_weight_init


def no_overrides(model):
    return not any(("_conv_forward" in m.__dict__) or ("forward" in m.__dict__) or isinstance(m.__dict__.get("optimized_attention"), vae_ops._AttnChunker)
                   for m in model.modules())


def ldm_vae(z, post_quant_conv, ch=32, dtype=None):
    dec = init_random(ldm.Decoder(ch=ch, out_ch=3, ch_mult=[1, 2, 4, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0,
                                  in_channels=3, resolution=256, z_channels=z))
    sd = {"decoder." + k: v for k, v in dec.state_dict().items()}
    if post_quant_conv:
        pq = OPS.Conv2d(z, z, 1)
        with torch.no_grad():
            pq.weight = torch.nn.Parameter(torch.randn(z, z, 1, 1) / math.sqrt(z), requires_grad=False)
            pq.bias = torch.nn.Parameter(torch.randn(z) * 0.1, requires_grad=False)
        sd["post_quant_conv.weight"] = pq.weight.data
        sd["post_quant_conv.bias"] = pq.bias.data
    return comfy.sd.VAE(sd=sd, dtype=dtype)


# ---------------------------------------------------------------------------
# 1. statistics
# ---------------------------------------------------------------------------

def statistics_tests():
    g = torch.Generator().manual_seed(2)
    for label, x in (("N(0, 1)", torch.randn(1, 32, 37, 11, generator=g)),
                     ("mean 1000, std 0.01 (cancellation for E[x^2] - E[x]^2)", 1000 + 0.01 * torch.randn(1, 32, 37, 11, generator=g))):
        xd = x.double().view(8, -1)
        var_t, mean_t = torch.var_mean(xd, dim=1, correction=0)
        worst_m = worst_v = 0.0
        for ws in (4, 44 * 4 * 32, 3 * 11 * 32 * 4, 1 << 30):
            m = eng.Moments(8)
            for a, b in ((0, 5), (5, 6), (6, 30), (30, 37)):
                m.add(x[:, :, a:b], 2, ws)
            var = m.m2.double() / m.n
            mean = (m.shift + m.mean).double()
            worst_m = max(worst_m, float((mean - mean_t).abs().max() / mean_t.abs().max().clamp(min=1)))
            worst_v = max(worst_v, float(((var - var_t).abs() / var_t).max()))
        naive = (x.view(8, -1).pow(2).mean(1) - x.view(8, -1).mean(1).pow(2)).double()
        e_naive = float(((naive - var_t).abs() / var_t).max())
        vt32, _ = torch.var_mean(x.view(8, -1), dim=1, correction=0)   # what native fp32 GroupNorm's own reduction achieves
        e_torch = float(((vt32.double() - var_t).abs() / var_t).max())
        check("Moments ({}): chunks of 1 / 4 / 3 / all rows, stripes of 5 / 1 / 24 / 7 rows: mean rel {:.2g}, var rel {:.2g} vs fp64 "
              "(torch's own fp32 var_mean over the whole tensor: {:.2g}; naive fp32 E[x^2]-E[x]^2: {:.2g})".format(
                  label, worst_m, worst_v, e_torch, e_naive), worst_m <= 1e-6 and worst_v <= 1e-5)
    for dtype, tol in ((torch.float32, 2e-6), (torch.bfloat16, 0.0)):
        gn = OPS.GroupNorm(8, 32, eps=1e-6, affine=True)
        gn.weight = torch.nn.Parameter((1 + 0.1 * torch.randn(32, generator=g)).to(dtype), requires_grad=False)
        gn.bias = torch.nn.Parameter((0.1 * torch.randn(32, generator=g)).to(dtype), requires_grad=False)
        x = (torch.randn(1, 32, 23, 9, generator=g) * 2 + 0.5).to(dtype)
        ref = F.group_norm(x, 8, gn.weight, gn.bias, gn.eps)
        m = eng.Moments(8)
        m.add(x, 2, 1 << 30)
        mean, rstd = m.freeze(gn.eps)
        y = eng.group_norm_frozen(x, gn.weight, gn.bias, mean, rstd, 2, 3 * 9 * 32 * 4)
        ys = eng.group_norm_frozen(x[:, :, 5:17], gn.weight, gn.bias, mean, rstd, 2, 4)
        e = float((y.float() - ref.float()).abs().max())
        es = float((ys.float() - ref[:, :, 5:17].float()).abs().max())
        ulp = 2 ** -7 * float(ref.float().abs().max())
        check("group_norm_frozen ({}): whole tensor max|Δ| {:.2g}, rows 5..17 with the whole-tensor statistics {:.2g} vs F.group_norm".format(
            str(dtype)[6:], e, es), y.dtype == dtype and (e <= tol if dtype == torch.float32 else e <= 2 * ulp) and (es <= tol if dtype == torch.float32 else es <= 2 * ulp))
    # GlobalNorms: instance forward, comfy.ops weight path, restore, missing statistics
    gn = OPS.GroupNorm(8, 32, eps=1e-6, affine=True)
    gn.weight = torch.nn.Parameter(1 + 0.1 * torch.randn(32, generator=g), requires_grad=False)
    gn.bias = torch.nn.Parameter(0.1 * torch.randn(32, generator=g), requires_grad=False)
    x = torch.randn(1, 32, 12, 7, generator=g)
    gn.weight_function = [lambda w: w * 2]
    ref = F.group_norm(x, 8, gn.weight * 2, gn.bias, gn.eps)
    nr = eng.NormRef(gn, "gn")
    with eng.GlobalNorms([nr], 2, 1 << 20) as norms:
        m = eng.Moments(8)
        m.add(x, 2, 1 << 20)
        norms.store[gn] = m.freeze(gn.eps)
        y = gn(x[:, :, 3:9])
        inside = "forward" in gn.__dict__
    del gn.weight_function
    e = float((y - ref[:, :, 3:9]).abs().max())
    check("GlobalNorms: rows 3..9 normalized with the whole-image statistics through the weight_function path (max|Δ| {:.2g}), "
          "forward restored".format(e), e <= 2e-6 and inside and "forward" not in gn.__dict__)
    with eng.GlobalNorms([nr], 2, 1 << 20):
        expect_raises("a norm without frozen statistics is an internal error", eng.StripeError, lambda: gn(x), "statistics")
    check("... restored after the error", "forward" not in gn.__dict__)


# ---------------------------------------------------------------------------
# 2. recognition
# ---------------------------------------------------------------------------

def recognition_tests():
    g = torch.Generator().manual_seed(3)
    sd = ldm_vae(4, True)
    fx = ldm_vae(16, False)
    lat4 = torch.randn(1, 4, 12, 10, generator=g)
    lat16 = torch.randn(1, 16, 12, 10, generator=g)
    for label, v, lat in (("SDXL-like (AutoencoderKL, post_quant_conv, z 4)", sd, lat4), ("Flux-like (AutoencodingEngine, z 16)", fx, lat16)):
        bound, why = vl.match(v, lat, {})
        check("{}: recognized ({} prefix modules, {} units, {} norms with whole-image statistics)".format(
            label, len(bound.prefix) if bound else 0, len(bound.units) if bound else 0, len(bound.norm_refs()) if bound else 0),
            bound is not None and len(bound.norm_refs()) == 19, str(why))
    dec = sd.first_stage_model.decoder
    rb = dec.up[1].block[0]
    cases = []

    def case(label, apply, undo, want, decode=True):
        """Mutate, match, and (where the native forward still runs and is deterministic) decode managed and native."""
        apply()
        try:
            _, why = vl.match(sd, lat4, {})
            out = ref = last = None
            if decode:
                out = managed_decode(sd, lat4, raw=True)
                last = mvae.last_decode()
                ref = native_decode(sd, lat4, raw=True)
        finally:
            undo()
        cases.append((label, why, last, out, ref, want))

    def setattr_case(label, obj, name, value, want, decode=True):
        old = obj.__dict__.get(name, "<missing>")
        case(label, lambda: setattr(obj, name, value), lambda: (setattr(obj, name, old) if old != "<missing>" else obj.__dict__.pop(name, None)),
             want, decode)

    setattr_case("tanh_out", dec, "tanh_out", True, "tanh_out")
    setattr_case("give_pre_end", dec, "give_pre_end", True, "give_pre_end")
    setattr_case("carried 3D convs", dec, "carried", True, "carried", decode=False)
    up = dec.up[2].upsample
    setattr_case("upsample scale (1, 2, 2)", up, "scale_factor", (1.0, 2.0, 2.0), "scale factor", decode=False)
    setattr_case("upsample without conv", up, "with_conv", False, "without conv")
    setattr_case("3x3 conv shortcut", rb, "use_conv_shortcut", True, "conv shortcut", decode=False)
    fsm = sd.first_stage_model
    bn = torch.nn.BatchNorm2d(16).eval()
    case("batch_norm_latent", lambda: fsm.__dict__.__setitem__("bn", bn), lambda: fsm.__dict__.__setitem__("bn", None), "bn", decode=False)
    c1 = dec.up[1].block[-1].out_channels
    atts = [init_random(ldm.AttnBlock(c1)) for _ in dec.up[1].block]
    case("attention in an up level", lambda: dec.up[1].attn.extend(atts), lambda: [dec.up[1].attn.__delitem__(0) for _ in atts], "has attention")
    drop = dec.up[0].block[2].dropout
    case("Dropout p=0.5 in training mode", lambda: (setattr(drop, "p", 0.5), drop.train()), lambda: (setattr(drop, "p", 0.0), drop.eval()), "Dropout",
         decode=False)
    hooks = []
    case("forward hook", lambda: hooks.append(dec.up[0].block[1].conv2.register_forward_hook(lambda *a: None)), lambda: hooks.pop().remove(), "hook")
    nm = dec.mid.block_1.norm2
    case("instance-level forward", lambda: setattr(nm, "forward", nm.forward), lambda: nm.__dict__.pop("forward"), "instance-level forward")
    for label, why, last, out, ref, want in cases:
        if out is None:
            check("{} -> not recognized: {}".format(label, why), why is not None and want in why)
            continue
        e = float((out - ref).abs().max())
        check("{} -> not recognized, layer 2, == native (max|Δ| {:.2g}): {}".format(label, e, why),
              why is not None and want in why and last.get("strategy") == "layer2" and e <= 1e-4)
    check("... no instance override left on the model", no_overrides(sd.first_stage_model))
    _, why = vl.match(sd, lat4, {"x": 1})
    _, why2 = vl.match(sd, torch.randn(1, 16, 12, 10), {})
    check("vae_options / wrong latent channels -> no match ({}; {})".format(why, why2), why is not None and why2 is not None)
    return sd, fx, lat4, lat16


# ---------------------------------------------------------------------------
# 3. whole decoder
# ---------------------------------------------------------------------------

def compare(label, v, latent, rows=None, scheme="A", tol=1e-5, width=4, expect_passes=19):
    mvae.set_gn_scheme(scheme)
    mvae.set_stripe_rows(rows)
    try:
        with Spy() as spy:
            raw = managed_decode(v, latent, raw=True)
    finally:
        mvae.set_stripe_rows(None)
    last = mvae.last_decode()
    ref = native_decode(v, latent, raw=True)
    d = (raw.float() - ref.float()).abs()
    e = float(d.max())
    h = ref.shape[2]
    rows_b = set()
    for _, o0 in last.get("boundaries", []):
        rows_b.update(r for r in range(o0 - width, o0 + width) if 0 <= r < h)
    mask = torch.zeros(h, dtype=torch.bool)
    if rows_b:
        mask[torch.tensor(sorted(rows_b))] = True
    eb = float(d[:, :, mask].max()) if mask.any() else 0.0
    ei = float(d[:, :, ~mask].max()) if (~mask).any() else 0.0
    ok = (last.get("strategy") == "layer1" and e <= tol and raw.shape == ref.shape and raw.dtype == ref.dtype and eb <= max(tol, 3 * ei)
          and last.get("passes") == expect_passes and spy.tiled == 0 and no_overrides(v.first_stage_model))
    check("{}: scheme {}, {} stripes of {} rows, {} statistics passes (rows {}), saves {}, recompute {:.2f}x; max|Δ| {:.2g} (near boundaries {:.2g}, "
          "elsewhere {:.2g})".format(label, scheme, last.get("stripes"), last.get("rows"), last.get("passes"), last.get("pass_rows"),
                                     "+".join(vae_ops.fmt_bytes(b) for b in last.get("saves") or []) or "none", last.get("recompute") or 0, e, eb, ei),
          ok, "strategy {} note {}".format(last.get("strategy"), last.get("layer1")))
    return last


def decoder_tests(sd, fx, lat4, lat16):
    mvae.set_budget(None)
    for label, v, lat, schemes, heights in (("SDXL-like 12x10", sd, lat4, "ADBC", (7, 40, None)), ("Flux-like 12x10", fx, lat16, "AC", (7, None))):
        for scheme in schemes:
            for rows in heights:
                compare("{}, rows {}".format(label, rows or "default"), v, lat, rows=rows, scheme=scheme)
    compare("SDXL-like 12x10, rows 1 (96 stripes)", sd, lat4, rows=1, scheme="A")
    g = torch.Generator().manual_seed(4)
    for scheme in ("A", "C"):
        compare("odd 13x9 latent, rows 9", sd, torch.randn(1, 4, 13, 9, generator=g), rows=9, scheme=scheme)
        compare("tiny 1x1 latent, rows 3", sd, torch.randn(1, 4, 1, 1, generator=g), rows=3, scheme=scheme)
        compare("3x5 latent, rows 5", sd, torch.randn(1, 4, 3, 5, generator=g), rows=5, scheme=scheme)
    compare("batch 2, rows 16", sd, torch.randn(2, 4, 12, 10, generator=g), rows=16, scheme="B")
    wide = ldm_vae(4, True, ch=64)
    compare("ch 64 (2+ channels per group everywhere), rows 24", wide, lat4, rows=24, scheme="D")
    mvae.set_workspace(16 * 1024)
    last = compare("workspace 16 KiB (conv row blocks inside the stripes, 1-row norm chunks), rows 32", sd, lat4, rows=32, scheme="B")
    check("  ... conv calls inside the stripes were split into row blocks ({})".format(last["stats"]["conv_chunked"]), last["stats"]["conv_chunked"] > 0)
    mvae.set_workspace(mvae.DEFAULT_WORKSPACE)
    # bf16: against an fp32 truth, at the level of native bf16
    truth = native_decode(sd, lat4, raw=True).float()
    sb = ldm_vae(4, True, dtype=torch.bfloat16)
    nb = native_decode(sb, lat4, raw=True).float()
    for scheme in ("A", "C"):
        mvae.set_gn_scheme(scheme)
        mvae.set_stripe_rows(24)
        mb = managed_decode(sb, lat4, raw=True).float()
        mvae.set_stripe_rows(None)
        last = mvae.last_decode()
        rn = float((nb - truth).pow(2).mean().sqrt())
        rm = float((mb - truth).pow(2).mean().sqrt())
        check("bf16 VAE, scheme {}: RMSE vs fp32 truth: layer 1 {:.3g}, native {:.3g}; max|Δ| {:.3g} / {:.3g}".format(
            scheme, rm, rn, float((mb - truth).abs().max()), float((nb - truth).abs().max())),
            last.get("strategy") == "layer1" and rm <= 1.5 * rn + 1e-4)
    mvae.set_gn_scheme("A")
    return wide


# ---------------------------------------------------------------------------
# 4. self-test
# ---------------------------------------------------------------------------

def selftest_tests(sd, lat4):
    mvae.set_gn_scheme("A")
    eng._SELFTEST.clear()
    bound, _ = vl.match(sd, lat4, {})
    ok, detail = eng.self_test(bound, sd)
    check("self-test of the real structure passes: {}".format(detail), ok)
    orig_add, orig_need = eng.Moments.add, eng.need_in
    seen = {}

    def local_add(self, x, hdim, ws):          # statistics of each stripe on its own (what tiled decoding does)
        self.n, self.mean, self.m2 = 0, None, None
        return orig_add(self, x, hdim, ws)

    def lossy_add(self, x, hdim, ws):          # the first stripe of every pass is lost
        k = id(self)
        if k not in seen:
            seen[k] = True
            return
        return orig_add(self, x, hdim, ws)

    def short_need(unit, a, b, h_in, h_out):
        if unit.kind == eng.RES:
            return max(0, a - 1), min(h_in, b + 1)
        return orig_need(unit, a, b, h_in, h_out)

    for label, target, attr, fn in (("stripe-local GroupNorm statistics", eng.Moments, "add", local_add),
                                    ("one stripe missing from the statistics", eng.Moments, "add", lossy_add),
                                    ("halo one row short", eng, "need_in", short_need)):
        eng._SELFTEST.clear()
        setattr(target, attr, fn)
        try:
            out = managed_decode(sd, lat4, raw=True)
            last = mvae.last_decode()
        finally:
            eng.Moments.add, eng.need_in = orig_add, orig_need
        ref = native_decode(sd, lat4, raw=True)
        note = last.get("layer1") or ""
        check("injected bug: {} -> self-test fails, layer 2, == native (max|Δ| {:.2g}): {}".format(label, float((out - ref).abs().max()), note[:150]),
              last.get("strategy") == "layer2" and "self-test failed" in note and float((out - ref).abs().max()) <= 1e-4)
    eng._SELFTEST.clear()
    for scheme in "AD":
        mvae.set_gn_scheme(scheme)
        managed_decode(sd, lat4)
    check("one self-test per structure and GroupNorm scheme ({} cached)".format(len(eng._SELFTEST)), len(eng._SELFTEST) == 2)
    mvae.set_gn_scheme("A")


# ---------------------------------------------------------------------------
# 5. plans, OOM, switches
# ---------------------------------------------------------------------------

def plan_tests(sd, lat4):
    import alloc_sim
    G = float(1 << 30)
    v = alloc_sim.meta_vae(torch.bfloat16, "sdxl")
    lat = torch.empty(1, 4, 270, 480, device="meta")
    outb = 3 * 2160 * 3840 * 4
    res = {}
    for scheme in "ADBC":
        mvae.set_gn_scheme(scheme)
        bound, why = mvae._select_layer1(v, lat, {})
        plan, _, _, _ = mvae.choose_plan(v, lat, bound, outb)
        res[scheme] = plan
        floor_ok = scheme == "C" or plan.pass_bytes <= max(plan.prefix_bytes, plan.stripe_bytes)   # C: two full-size saves at once
        parts = plan.live_peak == plan.persistent + max(plan.prefix_bytes, plan.stripe_bytes, plan.pass_bytes)
        check("SDXL 4K scheme {}: {} x {} rows, {} statistics passes, saves {}, live {:.2f} GiB (prefix {:.2f}, final {:.2f}, passes {:.2f}), arena {:.2f}, "
              "estimate {:.2f}, recompute {:.1f}x".format(scheme, len(plan.stripes), plan.rows, len(plan.passes),
                                                         "+".join(vae_ops.fmt_bytes(plan.save_bytes[p]) for p in plan.saves) or "none",
                                                         plan.live_peak / G, plan.prefix_bytes / G, plan.stripe_bytes / G, plan.pass_bytes / G,
                                                         plan.arena / G, plan.estimate / G, plan.recompute),
              len(plan.passes) == 19 and len(plan.saves) == {"A": 0, "D": 1, "B": 2, "C": 5}[scheme] and floor_ok and parts
              and plan.estimate == plan.arena + plan.largest + eng.ESTIMATE_PAD)
    check("schemes: arena A < D < B < C; recompute A > D > B, A > D > C (C's saves share one pool, B's are separate: {} / {})".format(
          res["C"].save_layout, res["B"].save_layout),
          res["A"].recompute > res["D"].recompute > res["B"].recompute and res["D"].recompute > res["C"].recompute
          and res["A"].arena < res["D"].arena < res["B"].arena < res["C"].arena)
    mvae.set_gn_scheme("A")
    # OOM
    orig_run = eng.run_passes
    calls_l2 = [0]
    orig_l2 = mvae._run

    def l2(*a, **kw):
        calls_l2[0] += 1
        return orig_l2(*a, **kw)

    def oom_above(limit):
        def run(plan, *a, **kw):
            if max(b - a_ for a_, b in plan.stripes) > limit:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return orig_run(plan, *a, **kw)
        return run
    mvae._run = l2
    try:
        eng.run_passes = oom_above(24)
        mvae.set_stripe_rows(96)
        with Spy() as spy:
            out = managed_decode(sd, lat4, raw=True)
        last = mvae.last_decode()
        ref = native_decode(sd, lat4, raw=True)
        check("simulated OOM above 24-row stripes: retried with {}-row stripes after {} retries, == native (max|Δ| {:.2g})".format(
            last.get("rows"), last.get("retries"), float((out - ref).abs().max())),
            last.get("strategy") == "layer1" and last.get("rows") <= 24 and last.get("retries") == 2
            and float((out - ref).abs().max()) <= 1e-5 and spy.tiled == 0 and calls_l2[0] == 0)
        eng.run_passes = oom_above(0)
        with Spy() as spy:
            expect_raises("OOM even with the smallest stripes -> MonoloadVAEOOMError", MonoloadVAEOOMError,
                          lambda: managed_decode(sd, lat4), "不会退回到 tiled", "第二层")
        check("... neither tiled nor layer 2 was called, no override left", spy.tiled == 0 and calls_l2[0] == 0 and no_overrides(sd.first_stage_model))
    finally:
        eng.run_passes = orig_run
        mvae._run = orig_l2
        mvae.set_stripe_rows(None)
    expect_raises("set_gn_scheme('E') -> ValueError", ValueError, lambda: mvae.set_gn_scheme("E"))
    import os
    old = os.environ.get("MONOLOAD_VAE_GN_SCHEME")
    try:
        os.environ["MONOLOAD_VAE_GN_SCHEME"] = "d"
        a = mvae._gn_scheme_from_env()
        os.environ["MONOLOAD_VAE_GN_SCHEME"] = "x"
        b = mvae._gn_scheme_from_env()
    finally:
        if old is None:
            os.environ.pop("MONOLOAD_VAE_GN_SCHEME", None)
        else:
            os.environ["MONOLOAD_VAE_GN_SCHEME"] = old
    check("MONOLOAD_VAE_GN_SCHEME: 'd' -> D, unknown -> the default {} (with a warning)".format(vl.DEFAULT_SCHEME), a == "D" and b == vl.DEFAULT_SCHEME)


# ---------------------------------------------------------------------------
# 6. caching allocator
# ---------------------------------------------------------------------------

def allocator_tests():
    import alloc_sim
    G = float(1 << 30)
    MIB = 1 << 20
    for label, kw, mres in (
            ("SDXL layer 2, 4K, workspace 1 GiB (af9abc6)", dict(w=3840, h=2160, layer=2, ws=1 << 30, model="sdxl", **alloc_sim._version("v1")), 14.86),
            ("SDXL layer 2, 4K, workspace 1 GiB (current)", dict(w=3840, h=2160, layer=2, ws=1 << 30, model="sdxl"), 14.99),
            ("SDXL layer 2, 4K, workspace 512 MiB (current)", dict(w=3840, h=2160, layer=2, ws=512 * MIB, model="sdxl"), 15.86),
            ("SDXL layer 2, 2688, workspace 128 MiB (current)", dict(w=2688, h=1536, layer=2, ws=128 * MIB, model="sdxl"), 7.67),
            ("Flux layer 2, 2688, workspace 1 GiB (af9abc6)", dict(w=2688, h=1536, layer=2, ws=1 << 30, model="flux", **alloc_sim._version("v1")), 9.24)):
        i = alloc_sim.decode_trace(**kw)
        check("allocator simulator reproduces CT 700: {}: reserved {:.2f} GiB (measured {:.2f})".format(label, i["reserved"] / G, mres),
              abs(i["reserved"] / G - mres) <= 0.05)
    for model, w, h, dt, scheme, rows in (("sdxl", 1344, 768, "bf16", "A", None), ("sdxl", 3840, 2160, "bf16", "A", None),
                                          ("sdxl", 3840, 2160, "bf16", "D", 32), ("flux", 2688, 1536, "bf16", "B", None),
                                          ("sdxl", 1920, 1088, "fp32", "C", None)):
        i = alloc_sim.decode_trace(w, h, dt, rows, model=model, scheme=scheme)
        check("{} {}x{} {} scheme {} {}: {} x {} rows, simulated reserved {:.3f} GiB <= estimate {:.3f} GiB (arena {:.3f}, live {:.3f})".format(
            model, w, h, dt, scheme, "default" if rows is None else "{} rows".format(rows), i["stripes"], i["rows"], i["reserved"] / G,
            i["estimate"] / G, i["arena"] / G, i["live_peak"] / G), i["reserved"] <= i["estimate"])


def main():
    if not mvae.is_installed():
        check("vae.install() on this ComfyUI", mvae.install())
    mvae.set_stripe(True)
    statistics_tests()
    sd, fx, lat4, lat16 = recognition_tests()
    decoder_tests(sd, fx, lat4, lat16)
    selftest_tests(sd, lat4)
    plan_tests(sd, lat4)
    allocator_tests()
    finish()


if __name__ == "__main__":
    main()
