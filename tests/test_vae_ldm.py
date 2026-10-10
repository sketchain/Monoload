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
     without conv, a batch-norm latent unlike Flux 2's (tests/test_vae_flux2.py), a 3x3 conv shortcut, Dropout in training,
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
  5b. MONOLOAD_VAE_BUDGET: layer 2 when it fits; else the layer-1 scheme with
     the lowest predicted time (one stripe: no passes, the default scheme);
     a scheme failing its self-test -> the next; forced scheme / rows /
     layer 2 outrank the budget; nothing fits -> MonoloadError naming the
     needs; the README examples on the full-size SDXL decoder;
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
    bn = torch.nn.BatchNorm2d(16, affine=False).eval()
    # a BatchNorm latent that is not Flux 2's (no 2x2 patch size, 16 features for a z 4 decoder): Flux 2 itself, tests/test_vae_flux2.py
    case("batch_norm_latent without Flux 2's patch size", lambda: fsm.__dict__.__setitem__("bn", bn), lambda: fsm.__dict__.__setitem__("bn", None),
         "bn: patch size", decode=False)
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
    if last.get("stripes") == 1:
        expect_passes = 0   # one stripe: every norm sees the whole image, no statistics passes
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
    mvae.set_gn_scheme(None)
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
    # review 2026-10 item 06: an OOM inside the first-use self-test -> MonoloadVAEOOMError with the message of the table,
    # recorded as an OOM, not cached (the next decode runs the self-test again), no layer-2 fallback; budget path too
    orig_run = eng._self_test_run

    def oom_run(bound, vae):
        raise comfy.model_management.OOM_EXCEPTION("injected")
    rows = []
    for label, bud in (("default policy", None), ("budget 4 GiB", 4 << 30)):
        eng._SELFTEST.clear()
        mvae.set_budget(bud)
        eng._self_test_run = oom_run
        try:
            try:
                managed_decode(sd, lat4)
                err = None
            except Exception as e:
                err = e
            last = mvae.last_decode()
        finally:
            eng._self_test_run = orig_run
            mvae.set_budget(None)
        good = (isinstance(err, MonoloadVAEOOMError) and "self-test" in str(err) and bound.name in str(err)
                and last.get("strategy") == "error" and last.get("kind") == "oom" and not eng._SELFTEST)
        rows.append("{}: {} / record {} {}".format(label, type(err).__name__, last.get("strategy"), last.get("kind")))
        check("OOM in the first-use self-test, {}: MonoloadVAEOOMError ({}), recorded as an OOM ({} / {}), not cached ({} entries)".format(
              label, str(err).splitlines()[0][:110] if err else None, last.get("strategy"), last.get("kind"), len(eng._SELFTEST)), good)
    out = managed_decode(sd, lat4)
    check("... the next decode runs the self-test again and decodes on layer 1 ({}; cached {})".format(
          mvae.last_decode().get("strategy"), len(eng._SELFTEST)), mvae.last_decode().get("strategy") == "layer1" and len(eng._SELFTEST) == 1
          and out is not None)
    eng._SELFTEST.clear()
    for scheme in "AD":
        mvae.set_gn_scheme(scheme)
        managed_decode(sd, lat4)
    check("one self-test per structure and GroupNorm scheme ({} cached)".format(len(eng._SELFTEST)), len(eng._SELFTEST) == 2)
    mvae.set_gn_scheme(None)


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
              and plan.out_segment == eng.out_segment(outb) and plan.estimate == plan.out_segment + plan.arena + plan.largest + eng.ESTIMATE_PAD
              and plan.arena % (2 * eng.MIB) == 0)   # what the allocator reserves for it (review 2026-10 item 07: B's layout slack)
    check("schemes: arena A < D < B < C; recompute A > D > B, A > D > C (C's saves share one pool, B's are separate: {} / {})".format(
          res["C"].save_layout, res["B"].save_layout),
          res["A"].recompute > res["D"].recompute > res["B"].recompute and res["D"].recompute > res["C"].recompute
          and res["A"].arena < res["D"].arena < res["B"].arena < res["C"].arena)
    mvae.set_gn_scheme("A")   # self-tested above: the injected OOM below must not hit a first self-test
    # OOM
    orig_run = eng.run_passes
    calls_l2 = [0]
    orig_l2 = mvae._run

    def l2(*a, **kw):
        calls_l2[0] += 1
        return orig_l2(*a, **kw)

    orig_stripes = eng.run_stripes

    def oom_above(limit):
        # in the statistics passes, and in the output stripes (one stripe: no passes)
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
                          lambda: managed_decode(sd, lat4), "never falls back to the approximate tiled", "nor to layer 2")
        check("... neither tiled nor layer 2 was called, no override left", spy.tiled == 0 and calls_l2[0] == 0 and no_overrides(sd.first_stage_model))
    finally:
        eng.run_passes = orig_run
        eng.run_stripes = orig_stripes
        mvae._run = orig_l2
        mvae.set_stripe_rows(None)
        mvae.set_gn_scheme(None)
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
    check("MONOLOAD_VAE_GN_SCHEME: 'd' -> D forced, unknown -> the default {} not forced (with a warning)".format(vl.DEFAULT_SCHEME),
          a == ("D", True) and b == (vl.DEFAULT_SCHEME, False))
    mvae.set_gn_scheme("C")
    c = (mvae.gn_scheme(), mvae.gn_scheme_forced())
    mvae.set_gn_scheme(None)
    check("set_gn_scheme('C') forces C; set_gn_scheme(None) -> the default {} (scheme B since the CT 700 measurements), not forced".format(
        mvae.gn_scheme()), c == ("C", True) and mvae.gn_scheme() == "B" == vl.DEFAULT_SCHEME and not mvae.gn_scheme_forced())


# ---------------------------------------------------------------------------
# 5b. MONOLOAD_VAE_BUDGET: the fastest candidate within the budget
# ---------------------------------------------------------------------------

def _stub_selftest(bound):
    return True, "stub"


def budget_tests(sd, lat4):
    from monoload.errors import MonoloadError
    mvae.set_gn_scheme(None)
    mvae.set_stripe_rows(None)
    ref = native_decode(sd, lat4, raw=True)

    def run(bud):
        mvae.set_budget(bud)
        try:
            with Spy() as spy:
                out = managed_decode(sd, lat4, raw=True)
        finally:
            mvae.set_budget(None)
        last = mvae.last_decode()
        return last, float((out - ref).abs().max()), spy.tiled

    l2 = mvae._layer2_estimate(sd, lat4, {}, mvae.workspace())[0]["total"]
    last, e, tiled = run(l2)
    check("budget = the layer-2 estimate ({}): layer 2 (it fits and is the fastest), == native (max|Δ| {:.2g}); policy: {}".format(
        vae_ops.fmt_bytes(l2), e, last.get("policy")),
        last.get("strategy") == "layer2" and [c["layer"] for c in last["candidates"]] == [2] and e <= 1e-4 and tiled == 0)
    last, e, _ = run(l2 - 1)
    check("budget just below it: layer 1, one stripe (no statistics passes), the default scheme {} wins the tie: {} x {} rows, scheme {}, "
          "{} passes, == native (max|Δ| {:.2g})".format(vl.DEFAULT_SCHEME, last.get("stripes"), last.get("rows"), last.get("gn_scheme"), last.get("passes"), e),
          last.get("strategy") == "layer1" and last.get("stripes") == 1 and last.get("passes") == 0 and last.get("gn_scheme") == vl.DEFAULT_SCHEME
          and e <= 1e-5 and len(last["candidates"]) == 5 and not last["candidates"][0]["fits"])

    # a small workspace, so that the activations decide; this decoder is so small that layer 2 always fits before
    # any layer-1 plan (whose arena has a fixed pad), so layer 2's estimate is made huge for the ranking tests
    mvae.set_workspace(16 * 1024)
    orig_l2 = mvae._layer2_estimate

    def big_l2(*a, **kw):
        est, probe = orig_l2(*a, **kw)
        return dict(est, total=1 << 40), probe
    try:
        l2 = orig_l2(sd, lat4, {}, mvae.workspace())[0]["total"]
        mvae._layer2_estimate = big_l2
        floor = min(b.smallest_plan(sd, lat4, 16 * 1024, b.output_bytes(sd, lat4)).estimate for b in vl.match(sd, lat4, {})[0].variants())
        pick = None
        bud = floor
        while bud < 8 * floor and pick is None:
            mvae.set_budget(bud)
            try:
                d = mvae.choose_budget(sd, lat4, {}, bud, selftest=_stub_selftest)
            except MonoloadError:      # nothing fits yet (a scheme not self-tested yet counts its self-test, DESIGN §9.20)
                bud = int(bud * 1.03) + 1
                continue
            finally:
                mvae.set_budget(None)
            fit1 = [c for c in d["candidates"] if c["layer"] == 1 and c["fits"]]
            if d["layer"] == 1 and len(d["plan"].stripes) > 1 and len(fit1) >= 3 and len({c["gn_scheme"] for c in fit1}) >= 3:
                pick = bud
            bud = int(bud * 1.03) + 1
        check("a budget above the smallest layer-1 need ({}) where 3+ schemes fit with several stripes: {}".format(
            vae_ops.fmt_bytes(floor), vae_ops.fmt_bytes(pick) if pick else None), pick is not None)
        if pick is None:
            return
        last, e, tiled = run(pick)
        cands = last.get("candidates") or []
        fit1 = [c for c in cands if c["layer"] == 1 and c["fits"]]
        chosen = [c for c in fit1 if c["gn_scheme"] == last.get("gn_scheme")]
        check("budget {}: scheme {} ({} x {} rows, estimate {}, predicted {:.3g} s) = the fastest predicted of {} fitting ({}); layer 2 ({}, stubbed) does not fit; "
              "== native (max|Δ| {:.2g})".format(
                  vae_ops.fmt_bytes(pick), last.get("gn_scheme"), last.get("stripes"), last.get("rows"), vae_ops.fmt_bytes(last["estimate"]["total"]),
                  last.get("predicted_seconds") or 0, len(fit1), ", ".join("{} {:.3g} s".format(c["gn_scheme"], c["seconds"]) for c in fit1),
                  vae_ops.fmt_bytes(cands[0]["estimate"]) if cands else "?", e),
              last.get("strategy") == "layer1" and cands and cands[0]["layer"] == 2 and not cands[0]["fits"] and len(chosen) == 1
              and chosen[0]["seconds"] == min(c["seconds"] for c in fit1) and last["estimate"]["total"] <= pick and e <= 1e-5 and tiled == 0
              and "fastest" in last.get("policy", ""))
        # the chosen scheme fails its self-test: the next fastest
        bound = vl.LDMStripe(sd.first_stage_model, sd.first_stage_model.post_quant_conv, gn_scheme=last["gn_scheme"])
        first = last["gn_scheme"]
        eng._SELFTEST[bound.key] = (False, "injected")
        try:
            last2, e2, _ = run(pick)
        finally:
            eng._SELFTEST.pop(bound.key, None)
        rest = sorted((c for c in fit1 if c["gn_scheme"] != first), key=lambda c: c["seconds"])
        check("... scheme {} marked as failing its self-test: the next fastest, {} (== native, max|Δ| {:.2g})".format(first, last2.get("gn_scheme"), e2),
              last2.get("strategy") == "layer1" and rest and last2.get("gn_scheme") == rest[0]["gn_scheme"] and e2 <= 1e-5)
        # forced settings outrank the budget
        mvae.set_stripe_rows(16)
        last, e, _ = run(pick)
        fit16 = [c for c in last.get("candidates") or [] if c["fits"]]
        check("MONOLOAD_VAE_STRIPE_ROWS=16 with that budget: 16-row stripes, scheme {} = the fastest fitting at 16 rows (of {}), no layer 2 "
              "(== native, max|Δ| {:.2g})".format(last.get("gn_scheme"), len(fit16), e),
              last.get("strategy") == "layer1" and last.get("rows") == 16 and all(c["layer"] == 1 and c["rows"] == 16 for c in last["candidates"])
              and fit16 and last.get("gn_scheme") == min(fit16, key=lambda c: c["seconds"])["gn_scheme"] and e <= 1e-5)
        last, e, _ = run(1 << 20)
        check("... with a 1 MiB budget: runs anyway at 16 rows (forced outranks the budget, logged: {})".format(last.get("policy")),
              last.get("strategy") == "layer1" and last.get("rows") == 16 and "above the budget" in last.get("policy", "") and e <= 1e-5)
        mvae.set_stripe_rows(None)
        mvae._layer2_estimate = orig_l2
        mvae.set_gn_scheme("A")
        last, e, _ = run(1 << 30)
        mvae.set_gn_scheme(None)
        check("MONOLOAD_VAE_GN_SCHEME=A with a 1 GiB budget (layer 2 would fit): layer 1, scheme A, tallest stripes within it, layer 2 not considered "
              "(== native, max|Δ| {:.2g})".format(e),
              last.get("strategy") == "layer1" and last.get("gn_scheme") == "A" and [c["gn_scheme"] for c in last["candidates"]] == ["A"] and e <= 1e-5)
        mvae.set_gn_scheme("C")
        mvae.set_budget(1 << 20)
        try:
            expect_raises("MONOLOAD_VAE_GN_SCHEME=C with a 1 MiB budget -> MonoloadError naming scheme C's need (no layer 2)", MonoloadError,
                          lambda: managed_decode(sd, lat4), "scheme C")
        finally:
            mvae.set_budget(None)
            mvae.set_gn_scheme(None)
        mvae.set_stripe(False)
        last, e, _ = run(1 << 20)
        mvae.set_stripe(True)
        check("MONOLOAD_DISABLE_VAE_STRIPE=1 with a 1 MiB budget: layer 2 anyway ({})".format(last.get("policy")),
              last.get("strategy") == "layer2" and "forced" in last.get("policy", "") and e <= 1e-4)
        mvae.set_budget(1 << 20)
        try:
            expect_raises("1 MiB budget, nothing forced -> MonoloadError naming what layer 2 and each scheme need", MonoloadError,
                          lambda: managed_decode(sd, lat4), "MONOLOAD_VAE_BUDGET", "layer 2 needs about", "scheme A", "scheme B", "scheme C", "scheme D")
            dec = sd.first_stage_model.decoder
            dec.tanh_out = True
            try:
                expect_raises("... an unrecognized decoder (tanh_out): MonoloadError naming layer 2's need and why layer 1 is out", MonoloadError,
                              lambda: managed_decode(sd, lat4), "layer 2 needs about", "layer 1 not available", "tanh_out")
                mvae.set_budget(l2)
                managed_decode(sd, lat4)
                check("... with a budget layer 2 fits: layer 2", mvae.last_decode().get("strategy") == "layer2")
            finally:
                dec.tanh_out = False
        finally:
            mvae.set_budget(None)
    finally:
        mvae._layer2_estimate = orig_l2
        mvae.set_workspace(mvae.DEFAULT_WORKSPACE)
        mvae.set_gn_scheme(None)
        mvae.set_stripe_rows(None)
        mvae.set_stripe(True)
        mvae.set_budget(None)


def probe_request_tests(sd, lat4):
    """Review 2026-10 item 23: what the 8 x 8 shape probe asks load_models_gpu for has no workspace in it."""
    import comfy.model_management as mm
    asked = []
    orig = mm.load_models_gpu

    def spy(models, memory_required=0, **kw):
        asked.append(memory_required)
        return orig(models, memory_required=memory_required, **kw)
    mvae._PROBES.clear()
    mm.load_models_gpu = spy
    try:
        est, probe = mvae._layer2_estimate(sd, lat4, {}, 1 << 30)
    finally:
        mm.load_models_gpu = orig
    small = mvae.estimate(sd, lat4[0:1, ..., :mvae.PROBE_SIZE, :mvae.PROBE_SIZE], 0)["total"]
    check("the shape probe asks load_models_gpu for {} (its own small estimate), not {} (with 2 x the 1 GiB workspace)".format(
          vae_ops.fmt_bytes(asked[0]) if asked else None,
          vae_ops.fmt_bytes(mvae.estimate(sd, lat4[0:1, ..., :mvae.PROBE_SIZE, :mvae.PROBE_SIZE], 1 << 30)["total"])),
          probe is not None and asked == [small] and small < (64 << 20))


def budget_plan_tests():
    """The README examples: full-size SDXL (meta device), budgets 20 / 3 / 1.5 GiB."""
    import alloc_sim
    G = float(1 << 30)
    v = alloc_sim.meta_vae(torch.bfloat16, "sdxl")
    want = {(1344, 768): ("L2", "L1 1 stripe", "C"), (2688, 1536): ("L2", "C", "B"), (3840, 2160): ("L2", "B", "A")}
    for (w, h), exp in want.items():
        lat = torch.empty(1, 4, h // 8, w // 8, device="meta")
        with vae_ops.OpChunking(v.first_stage_model, 1 << 30, vae_ops.OpStats()):
            mvae._probe(v, lat, {})
        got = []
        ok = True
        for b, e in zip((20, 3, 1.5), exp):
            bud = int(b * G)
            mvae.set_budget(bud)
            d = mvae.choose_budget(v, lat, {}, bud, selftest=_stub_selftest)
            mvae.set_budget(None)
            if d["layer"] == 2:
                got.append("L2 {:.2f}".format(d["estimate"]["total"] / G))
                ok = ok and e == "L2" and d["estimate"]["total"] <= bud
            else:
                p = d["plan"]
                got.append("{} {}r {:.2f} ~{:.0f}s".format(d["bound"].gn_scheme, p.rows, p.estimate / G, d["bound"].predict_seconds(p)))
                ok = ok and p.estimate <= bud and (len(p.stripes) == 1 if e == "L1 1 stripe" else d["bound"].gn_scheme == e)
                fit = [c for c in d["candidates"] if c["layer"] == 1 and c["fits"]]
                ok = ok and d["bound"].predict_seconds(p) == min(c["seconds"] for c in fit)
        check("SDXL {}x{} budget 20 / 3 / 1.5 GiB -> {} (expected {})".format(w, h, " | ".join(got), " / ".join(exp)), ok)
    i = alloc_sim.decode_trace(3840, 2160, "bf16", 180, model="sdxl", scheme="B", ws=128 << 20)
    check("SDXL 4K budget 3 GiB plan (B, 180 rows, workspace 128 MiB): simulated reserved {:.2f} GiB <= estimate {:.2f} GiB".format(
        i["reserved"] / G, i["estimate"] / G), i["reserved"] <= i["estimate"] <= 3 * G)
    # a very tall B plan: the dead saves' holes are useless to its temporaries (Plan.front_arena), few stripes / a large
    # workspace fragment more (arena_bytes); 4K B with 540 / 768 rows stranded several requests before (DESIGN §9.14.11)
    for rows, ws in ((540, 128), (768, 64)):
        i = alloc_sim.decode_trace(3840, 2160, "bf16", rows, model="sdxl", scheme="B", ws=ws << 20)
        check("SDXL 4K B {} rows, workspace {} MiB: simulated reserved {:.2f} GiB <= estimate {:.2f} GiB (arena {:.2f})".format(
            rows, ws, i["reserved"] / G, i["estimate"] / G, i["arena"] / G), i["reserved"] <= i["estimate"])
    # the time model against CT 700 (01377c4): 4K D, 309-row stripes, workspace 64 MiB measured 63.2 s (the levels-only model said 53.6)
    lat = torch.empty(1, 4, 270, 480, device="meta")
    bd = vl.LDMStripe(v.first_stage_model, v.first_stage_model.post_quant_conv, gn_scheme="D")
    t64 = bd.predict_seconds(bd.plan(v, lat, 0, 64 << 20, rows=309, out_bytes=bd.output_bytes(v, lat)))
    t128 = bd.predict_seconds(bd.plan(v, lat, 0, 128 << 20, rows=309, out_bytes=bd.output_bytes(v, lat)))
    check("time model: 4K D 309 rows, workspace 64 MiB {:.1f} s (CT 700 63.2 s), 128 MiB {:.1f} s: a smaller workspace is slower".format(t64, t128),
          abs(t64 / 63.2 - 1) <= 0.08 and t64 > t128)


def estimate_tests():
    """Saves whose place in the arena is guaranteed (vae_engine.saves_fit) are not in the estimate's largest."""
    import alloc_sim
    G = float(1 << 30)
    M = 1 << 20
    check("saves_fit: fits / does not fit / a freed block is reused / freed neighbours merge",
          eng.saves_fit(100 * M, [("alloc", 0, 10 * M), ("alloc", 1, 80 * M)])
          and not eng.saves_fit(100 * M, [("alloc", 0, 30 * M), ("alloc", 1, 80 * M)])
          and eng.saves_fit(100 * M, [("alloc", 0, 30 * M), ("free", 0), ("alloc", 1, 80 * M)])
          and eng.saves_fit(100 * M, [("alloc", 0, 30 * M), ("alloc", 1, 30 * M), ("alloc", 2, 30 * M), ("free", 0), ("free", 1),
                                      ("alloc", 3, 55 * M)])
          and not eng.saves_fit(100 * M, [("alloc", 0, 30 * M), ("alloc", 1, 30 * M), ("alloc", 2, 30 * M), ("free", 0), ("free", 2),
                                          ("alloc", 3, 55 * M)]))
    v = alloc_sim.meta_vae(torch.bfloat16, "sdxl")
    lat = torch.empty(1, 4, 270, 480, device="meta")
    res = []
    ok = True
    for scheme in "ADBC":
        b = vl.LDMStripe(v.first_stage_model, v.first_stage_model.post_quant_conv, gn_scheme=scheme)
        p, _, _, _ = mvae.choose_plan(v, lat, b, b.output_bytes(v, lat))
        biggest = max([0] + [p.save_bytes[q] for q in p.saves])
        res.append("{} arena {:.2f} largest {:.2f} estimate {:.2f}{}".format(scheme, p.arena / G, p.largest / G, p.estimate / G,
                                                                           " (saves guaranteed)" if p.saves_guaranteed else ""))
        # D's largest is a statistics-pass temporary exactly as large as its H/4 save; B's and C's saves are larger than any temporary
        ok = ok and p.ckpt_front == bool(p.saves) and (not p.saves or p.saves_guaranteed)
        if scheme in "BC":
            ok = ok and p.largest < max(biggest, p.pool_bytes if p.save_layout == "pool" else 0)
        if scheme == "B":
            ok = ok and p.estimate < 2.75 * G
    check("SDXL 4K default plans: the checkpoint is allocated first, the saves' places are guaranteed, largest is not a save: " + "; ".join(res), ok)


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
    budget_tests(sd, lat4)
    probe_request_tests(sd, lat4)
    budget_plan_tests()
    estimate_tests()
    allocator_tests()
    finish()


if __name__ == "__main__":
    main()
