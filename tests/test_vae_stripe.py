"""Layer 1: stripe decoding of the Wan 2.1 VAE (monoload/vae_engine.py, monoload/vae_wan.py).

No model files needed: ComfyUI's own WanVAE class with small channels and
random weights, fp32 on the CPU.

  1. intervals: need_in / stripe_needs against a brute-force dependency walk
     over random unit chains; valid_out against the real modules
     (ResidualBlock with and without a 1x1 shortcut, Resample, the head's 3x3
     CausalConv3d): exactly the rows valid_out names agree with the whole-image
     result, the next row at an inner edge does not;
  2. whole decoder through comfy.sd.VAE: layer 1 vs native VAE.decode for
     stripe heights 1, 7 (not a divisor), 40, the whole image and the
     budget-chosen height, odd and tiny latents, batch 2, layer-2 conv blocks
     inside the stripes, bf16; errors near stripe boundaries no larger than
     elsewhere;
  3. recognition: a Dropout in training mode / a forward hook / layer 1
     switched off -> layer 2; SDXL / Flux structures are not the Wan adapter's
     (they go to the LDM adapter, tests/test_vae_ldm.py);
  4. self-test: a halo one row short (caught by the per-unit validity check) and
     a halo one row short with a matching wrong validity rule (caught
     numerically) both fail the self-test and the decode falls back to layer 2;
  5. budget (explicit too small -> error naming the need), OOM -> smaller
     stripes, at the floor MonoloadVAEOOMError, never tiled, never layer 2;
  6. default policy (the peak of 128-row stripes, tallest stripes within it),
     memory model (estimate = persistent + max(prefix, stripes), monotone in
     the stripe height, the largest stripe runs first), the allocator cache is
     emptied after the self-test (not during a decode), the arena;
  7. single-frame Conv3d as conv2d (forced on the CPU): same result as the
     module, only where the 3D call is single-frame with an effective kT=1 and
     a known _conv_forward;
  8. caching allocator (tests/alloc_sim.py, the full-size Wan decoder on the
     meta device): the simulator reproduces CT 700 readings of the earlier
     versions, and the estimate is >= the simulated reserved peak of the
     current code (arena) for 1344 .. 8K, bf16 / fp32, several stripe heights.

    python tests/test_vae_stripe.py
"""

import random

import torch

from common import check, expect_raises, finish
import comfy.model_management
import comfy.sd
import comfy.ldm.wan.vae as wan
from monoload import vae as mvae
from monoload import vae_engine as eng
from monoload import vae_wan as vw
from monoload import vae_ops
from monoload.errors import MonoloadError, MonoloadVAEOOMError
from test_vae import Spy, init_conv, init_random, ldm_vae, managed_decode, native_decode, no_overrides, wan_vae_model

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. intervals
# ---------------------------------------------------------------------------

def brute_needs(units, heights, o0, o1):
    rows = set(range(o0, o1))
    out = [None] * (len(units) + 1)
    out[-1] = (o0, o1)
    for i in reversed(range(len(units))):
        u = units[i]
        h_in, h_out = heights[i], heights[i + 1]
        new = set()
        for y in rows:
            for c in range(y - u.halo, y + u.halo + 1):
                if u.kind == eng.UP:
                    if 0 <= c < h_out:
                        new.add(c // 2)
                elif 0 <= c < h_in:
                    new.add(c)
        rows = new
        out[i] = (min(rows), max(rows) + 1)
        if len(rows) != out[i][1] - out[i][0]:
            return None  # not contiguous: would be a bug in the rules
    return out


def interval_tests():
    rng = random.Random(1)
    kinds = [(eng.POINT, 0, 1), (eng.CONV, 1, 1), (eng.RES, 2, 1), (eng.UP, 1, 2)]
    bad = 0
    n = 0
    for _ in range(300):
        chain = [eng.Unit(*rng.choice(kinds)[:1], None, 0, 1, 1, 1, "u") for _ in range(rng.randint(1, 9))]
        for u in chain:
            k = next(k for k in kinds if k[0] == u.kind)
            u.halo, u.scale = k[1], k[2]
        h0 = rng.randint(1, 12)
        heights = [h0]
        for u in chain:
            heights.append(heights[-1] * u.scale)
        for o0, o1 in eng.split_rows(heights[-1], rng.randint(1, heights[-1])):
            n += 1
            if eng.stripe_needs(chain, heights, o0, o1) != brute_needs(chain, heights, o0, o1):
                bad += 1
    check("need_in / stripe_needs == brute-force dependency walk ({} stripes over 300 random unit chains)".format(n), bad == 0, "{} mismatches".format(bad))
    ok = all(sum(b - a for a, b in eng.split_rows(h, r)) == h and max(b - a for a, b in eng.split_rows(h, r)) <= r
             and max(b - a for a, b in eng.split_rows(h, r)) - min(b - a for a, b in eng.split_rows(h, r)) <= 1
             for h in range(1, 60) for r in range(1, 70))
    check("split_rows: balanced stripes covering the image, none above the requested height", ok)

    # valid_out against the real modules
    def rms(c):
        m = wan.RMS_norm(c, images=False)
        m.gamma.data = 1 + 0.1 * torch.randn(m.gamma.shape)
        return m

    def rb(ci, co):
        m = init_random(wan.ResidualBlock(ci, co))
        return m.eval()
    mods = [("ResidualBlock 8->8", eng.Unit(eng.RES, rb(8, 8), 2, 1, 8, 8, "rb")),
            ("ResidualBlock 6->8 (1x1 shortcut)", eng.Unit(eng.RES, rb(6, 8), 2, 1, 6, 8, "rb2")),
            ("Resample upsample2d 8->4", eng.Unit(eng.UP, init_random(wan.Resample(8, "upsample2d")), 1, 2, 8, 4, "up")),
            ("Resample upsample3d 8->4", eng.Unit(eng.UP, init_random(wan.Resample(8, "upsample3d")), 1, 2, 8, 4, "up3")),
            ("head CausalConv3d 3x3", eng.Unit(eng.CONV, init_conv(wan.CausalConv3d(8, 3, 3, padding=1)), 1, 1, 8, 3, "conv"))]
    for label, u in mods:
        cin = u.cin
        H, W = 17, 9
        x = torch.randn(1, cin, 1, H, W)
        full = u.module(x)
        h_out = H * u.scale
        worst, tight, cases = 0.0, True, 0
        for xa in range(0, H):
            for xb in range(xa + 1, H + 1):
                y = u.module(x[:, :, :, xa:xb])
                va, vb = eng.valid_out(u, xa, xb, h_out)
                if vb <= va:
                    continue
                cases += 1
                oa = xa * u.scale
                worst = max(worst, float((y[:, :, :, va - oa:vb - oa] - full[:, :, :, va:vb]).abs().max()))
                if va > oa and va - 1 - oa >= 0:  # the row just outside at an inner top edge must be affected by the padding
                    if torch.allclose(y[:, :, :, va - 1 - oa], full[:, :, :, va - 1], atol=1e-6):
                        tight = False
        check("valid_out == exact rows of {} on every slice of {} rows ({} slices, max|Δ| {:.2g}; the next row at an inner edge differs: {})".format(
            label, H, cases, worst, tight), worst <= 1e-5 and tight)


# ---------------------------------------------------------------------------
# 2. whole decoder
# ---------------------------------------------------------------------------

def compare_layer1(label, v, latent, rows=None, tol=1e-5, expect_stripes=None, width=4):
    mvae.set_stripe_rows(rows)
    try:
        raw = managed_decode(v, latent, raw=True)
    finally:
        mvae.set_stripe_rows(None)
    last = mvae.last_decode()
    ref = native_decode(v, latent, raw=True)
    d = (raw.float() - ref.float()).abs()
    e = float(d.max())
    h = ref.shape[-3] if ref.ndim == 5 else ref.shape[1]
    rows_b = set()
    for _, o0 in last.get("boundaries", []):
        rows_b.update(r for r in range(o0 - width, o0 + width) if 0 <= r < h)
    dd = d.reshape(-1, d.shape[-3], d.shape[-2], d.shape[-1])
    mask = torch.zeros(h, dtype=torch.bool)
    if rows_b:
        mask[torch.tensor(sorted(rows_b))] = True
    eb = float(dd[:, mask].max()) if mask.any() else 0.0
    ei = float(dd[:, ~mask].max()) if (~mask).any() else 0.0
    ok = (last.get("strategy") == "layer1" and e <= tol and raw.shape == ref.shape and raw.dtype == ref.dtype
          and eb <= max(tol, 3 * ei) and no_overrides(v.first_stage_model))
    if expect_stripes is not None:
        ok = ok and last.get("stripes") == expect_stripes
    check("{}: layer 1 == native ({} stripes of {} rows, recompute {:.2f}x; max|Δ| {:.2g}, near boundaries {:.2g}, elsewhere {:.2g})".format(
        label, last.get("stripes"), last.get("rows"), last.get("recompute") or 0, e, eb, ei), ok,
        "strategy {} note {}".format(last.get("strategy"), last.get("layer1")))
    return last


def decoder_tests():
    v = wan_vae_model()
    g = torch.Generator().manual_seed(11)
    lat = torch.randn(1, 16, 1, 12, 10, generator=g)
    mvae.set_budget(None)
    for rows, n in ((1, 96), (7, 14), (40, 3), (96, 1)):
        compare_layer1("12x10 latent, stripe height {}".format(rows), v, lat, rows=rows, expect_stripes=n)
    # budget-chosen height with several stripes (tiny workspace, so the activations decide). This decoder is so
    # small that layer 2 fits any budget a layer-1 plan fits (the arena has a fixed pad): with a budget, layer 2
    # is chosen when it fits (the fastest, DESIGN §9.14.10), so its estimate is made huge for the stripe checks
    mvae.set_workspace(16 * 1024)
    bound, _ = vw.match(v, lat, {})
    p24 = bound.plan(v, lat, 0, 16 * 1024, rows=24, out_bytes=bound.output_bytes(v, lat))
    mvae.set_budget(p24.estimate)
    l2 = mvae._layer2_estimate(v, lat, {}, mvae.workspace())[0]["total"]
    ref = native_decode(v, lat, raw=True)
    out = managed_decode(v, lat, raw=True)
    last = mvae.last_decode()
    check("budget {} >= layer 2's estimate {}: layer 2 (it fits and is the fastest), == native (max|Δ| {:.2g}); {}".format(
        vae_ops.fmt_bytes(p24.estimate), vae_ops.fmt_bytes(l2), float((out - ref).abs().max()), last.get("policy")),
        l2 <= p24.estimate and last.get("strategy") == "layer2" and [c["layer"] for c in last.get("candidates") or []] == [2]
        and float((out - ref).abs().max()) <= 1e-4)
    orig_l2 = mvae._layer2_estimate

    def big_l2(*a, **kw):
        est, probe = orig_l2(*a, **kw)
        return dict(est, total=1 << 40), probe
    mvae._layer2_estimate = big_l2
    try:
        last = compare_layer1("12x10 latent, height from a budget that fits 24-row stripes (layer 2 stubbed out)", v, lat)
    finally:
        mvae._layer2_estimate = orig_l2
    cands = last.get("candidates") or []
    check("budget-chosen plan: estimate {} <= budget {}, {} stripes of {} rows (24 rows fit, the whole image does not); candidates: {}".format(
        vae_ops.fmt_bytes(last["estimate"]["total"]), vae_ops.fmt_bytes(last["budget"]), last.get("stripes"), last.get("rows"),
        ", ".join("layer {} {}".format(c["layer"], vae_ops.fmt_bytes(c["estimate"])) for c in cands)),
        last.get("strategy") == "layer1" and last["estimate"]["total"] <= last["budget"] and 1 < last["stripes"] <= 4 and last["rows"] >= 24
        and [c["layer"] for c in cands] == [2, 1] and cands[1].get("seconds") is None)
    mvae.set_workspace(mvae.DEFAULT_WORKSPACE)
    mvae.set_budget(None)
    compare_layer1("12x10 latent, default policy (96 rows, shorter than {} rows: 1 stripe)".format(mvae.DEFAULT_POLICY_ROWS), v, lat, expect_stripes=1)
    compare_layer1("odd 13x9 latent, stripe height 9", v, torch.randn(1, 16, 1, 13, 9, generator=g), rows=9)
    compare_layer1("tiny 1x1 latent, stripe height 3", v, torch.randn(1, 16, 1, 1, 1, generator=g), rows=3)
    compare_layer1("3x5 latent, stripe height 5", v, torch.randn(1, 16, 1, 3, 5, generator=g), rows=5)
    compare_layer1("batch 2, stripe height 16", v, torch.randn(2, 16, 1, 12, 10, generator=g), rows=16)
    mvae.set_workspace(16 * 1024)
    last = compare_layer1("layer-2 conv blocks inside the stripes (workspace 16 KiB), stripe height 32", v, lat, rows=32)
    check("  ... conv calls inside the stripes were split into row blocks ({})".format(last["stats"]["conv_chunked"]), last["stats"]["conv_chunked"] > 0)
    mvae.set_workspace(mvae.DEFAULT_WORKSPACE)
    vb = wan_vae_model(dtype=torch.bfloat16)
    compare_layer1("bf16 VAE, stripe height 24", vb, lat, rows=24, tol=0.05)
    # output in pixels, layout, process_output
    mvae.set_stripe_rows(24)
    out = managed_decode(v, lat)
    mvae.set_stripe_rows(None)
    ref = native_decode(v, lat)
    check("pixels: shape {} / dtype / device as native, in [0, 1], max|Δ| {:.2g}".format(tuple(out.shape), float((out - ref).abs().max())),
          out.shape == ref.shape and out.dtype == ref.dtype and out.device == ref.device and float(out.min()) >= 0 and float(out.max()) <= 1
          and float((out - ref).abs().max()) <= 1e-5)
    return v, lat


# ---------------------------------------------------------------------------
# 3. recognition
# ---------------------------------------------------------------------------

def recognition_tests(v, lat):
    eng._SELFTEST.clear()
    mvae.set_stripe(False)
    managed_decode(v, lat)
    last = mvae.last_decode()
    check("MONOLOAD_DISABLE_VAE_STRIPE (set_stripe(False)): Wan decode uses layer 2", last.get("strategy") == "layer2" and "layer 2 only" in (last.get("layer1") or ""))
    mvae.set_stripe(True)
    d = v.first_stage_model.decoder.upsamples[-1].residual[5]
    d.p, d.training = 0.5, True
    managed_decode(v, lat)
    last = mvae.last_decode()
    d.p, d.training = 0.0, False
    check("Dropout in training mode with p>0 -> not recognized, layer 2 ({})".format(last.get("layer1")),
          last.get("strategy") == "layer2" and "Dropout" in (last.get("layer1") or ""))
    h = v.first_stage_model.decoder.head[2].register_forward_hook(lambda *a: None)
    managed_decode(v, lat)
    last = mvae.last_decode()
    h.remove()
    check("forward hook on a module -> not recognized, layer 2 ({})".format(last.get("layer1")),
          last.get("strategy") == "layer2" and "hook" in (last.get("layer1") or ""))
    check("4D latent / multi-frame / vae_options -> no match",
          vw.match(v, lat[:, :, 0], {})[0] is None and vw.match(v, torch.cat([lat, lat], 2), {})[0] is None and vw.match(v, lat, {"x": 1})[0] is None)
    g = torch.Generator().manual_seed(3)
    for label, vv, l4 in (("SDXL-like", ldm_vae(4, True), torch.randn(1, 4, 12, 10, generator=g)),
                          ("Flux-like", ldm_vae(16, False), torch.randn(1, 16, 12, 10, generator=g))):
        mvae.set_workspace(16 * 1024)
        out = managed_decode(vv, l4, raw=True)
        last = mvae.last_decode()
        ref = native_decode(vv, l4, raw=True)
        mvae.set_workspace(mvae.DEFAULT_WORKSPACE)
        why = vw.match(vv, l4, {})[1]
        check("{} (LDM Decoder) is not the Wan adapter's ({}); decoded by {}: {} chunked conv calls, max|Δ| vs native {:.2g}".format(
            label, why, last.get("adapter") or last.get("strategy"), last["stats"]["conv_chunked"], float((out - ref).abs().max())),
            "not comfy.ldm.wan.vae.WanVAE" in (why or "") and last.get("strategy") == "layer1" and "LDM" in (last.get("adapter") or "")
            and last["stats"]["conv_chunked"] > 0 and float((out - ref).abs().max()) <= 1e-4)


# ---------------------------------------------------------------------------
# 4. self-test
# ---------------------------------------------------------------------------

def selftest_tests(v, lat):
    eng._SELFTEST.clear()
    bound, _ = vw.match(v, lat, {})
    ok, detail = eng.self_test(bound, v)
    check("self-test of the real structure passes: {}".format(detail), ok)
    orig_need, orig_valid = eng.need_in, eng.valid_out

    def short_need(unit, a, b, h_in, h_out):
        if unit.kind == eng.RES:
            return max(0, a - 1), min(h_in, b + 1)
        return orig_need(unit, a, b, h_in, h_out)

    def short_valid(unit, xa, xb, h_out):
        if unit.kind == eng.RES:
            return xa + (1 if xa > 0 else 0), xb - (1 if xb < h_out else 0)
        return orig_valid(unit, xa, xb, h_out)

    for label, patches in (("halo one row short (validity check)", {"need_in": short_need}),
                           ("halo one row short + matching wrong validity rule (numeric comparison)", {"need_in": short_need, "valid_out": short_valid})):
        eng._SELFTEST.clear()
        for k, f in patches.items():
            setattr(eng, k, f)
        try:
            out = managed_decode(v, lat, raw=True)
            last = mvae.last_decode()
        finally:
            eng.need_in, eng.valid_out = orig_need, orig_valid
        ref = native_decode(v, lat, raw=True)
        note = last.get("layer1") or ""
        check("injected bug: {} -> self-test fails, decode falls back to layer 2, result == native (max|Δ| {:.2g}): {}".format(
            label, float((out - ref).abs().max()), note[:160]),
            last.get("strategy") == "layer2" and "self-test failed" in note and float((out - ref).abs().max()) <= 1e-4)
    eng._SELFTEST.clear()
    rng = torch.get_rng_state()
    eng.self_test(bound, v)
    check("self-test leaves the RNG state unchanged and is cached per structure",
          torch.equal(rng, torch.get_rng_state()) and bound.key in eng._SELFTEST)


# ---------------------------------------------------------------------------
# 5. budget, OOM
# ---------------------------------------------------------------------------

def budget_oom_tests(v, lat):
    mvae.set_budget(1 << 20)
    expect_raises("budget too small (MONOLOAD_VAE_BUDGET) -> MonoloadError naming what is needed", MonoloadError,
                  lambda: managed_decode(v, lat), "MONOLOAD_VAE_BUDGET", "needs")
    mvae.set_budget(None)

    orig_run = eng.run_stripes
    calls_l2 = [0]
    orig_l2 = mvae._run

    def l2(*a, **kw):
        calls_l2[0] += 1
        return orig_l2(*a, **kw)

    def oom_above(limit):
        def run(units, plan, ckpt, write):
            if max(b - a for a, b in plan.stripes) > limit:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return orig_run(units, plan, ckpt, write)
        return run

    mvae._run = l2
    try:
        eng.run_stripes = oom_above(24)
        mvae.set_stripe_rows(96)
        with Spy() as spy:
            out = managed_decode(v, lat, raw=True)
        last = mvae.last_decode()
        ref = native_decode(v, lat, raw=True)
        check("simulated OOM above 24-row stripes: retried with {}-row stripes after {} retries, result == native (max|Δ| {:.2g})".format(
            last.get("rows"), last.get("retries"), float((out - ref).abs().max())),
            last.get("strategy") == "layer1" and last.get("rows") <= 24 and last.get("retries") == 2
            and float((out - ref).abs().max()) <= 1e-5 and spy.tiled == 0 and calls_l2[0] == 0)
        eng.run_stripes = oom_above(0)
        with Spy() as spy:
            expect_raises("OOM even with the smallest stripes -> MonoloadVAEOOMError", MonoloadVAEOOMError,
                          lambda: managed_decode(v, lat), "never falls back to the approximate tiled", "nor to layer 2")
        check("... neither tiled nor layer 2 was called, no override left", spy.tiled == 0 and calls_l2[0] == 0 and no_overrides(v.first_stage_model))
    finally:
        eng.run_stripes = orig_run
        mvae._run = orig_l2
        mvae.set_stripe_rows(None)


# ---------------------------------------------------------------------------
# 6. default policy, memory model, allocator cache
# ---------------------------------------------------------------------------

def policy_memory_tests(v):
    g = torch.Generator().manual_seed(5)
    lat = torch.randn(1, 16, 1, 40, 6, generator=g)   # 320 output rows
    bound, _ = vw.match(v, lat, {})
    outb = bound.output_bytes(v, lat)
    ws = mvae.layer1_workspace()
    mvae.set_budget(None)
    ref = bound.plan(v, lat, 0, ws, rows=mvae.DEFAULT_POLICY_ROWS, out_bytes=outb)
    last = compare_layer1("320-row image, default policy", v, lat, expect_stripes=3)
    check("default policy: target = arena (peak) of {}-row stripes ({}), tallest stripes within it: {} x {} rows, arena {} ({})".format(
        mvae.DEFAULT_POLICY_ROWS, vae_ops.fmt_bytes(ref.arena), last["stripes"], last["rows"], vae_ops.fmt_bytes(last["estimate"]["arena"]), last["policy"]),
        last["policy"].startswith("default") and last["budget"] == ref.arena and last["estimate"]["arena"] <= ref.arena
        and bound.plan(v, lat, 0, ws, rows=160, out_bytes=outb).arena > ref.arena)
    mvae.set_stripe_rows(64)
    p, bud, _, policy = mvae.choose_plan(v, lat, bound, outb)
    mvae.set_budget(ref.estimate)
    pb, budb, _, policyb = mvae.choose_plan(v, lat, bound, outb)
    mvae.set_stripe_rows(None)
    pc, budc, _, policyc = mvae.choose_plan(v, lat, bound, outb)
    mvae.set_budget(None)
    check("MONOLOAD_VAE_STRIPE_ROWS overrides the policy and the budget ({} x {} rows: {}); MONOLOAD_VAE_BUDGET alone: tallest within it ({} x {} rows: {})".format(
        len(pb.stripes), max(b - a for a, b in pb.stripes), policyb, len(pc.stripes), max(b - a for a, b in pc.stripes), policyc),
        len(p.stripes) == 5 and len(pb.stripes) == 5 and "forced" in policy and "forced" in policyb
        and policyc.startswith("budget ") and "(from environment variable MONOLOAD_VAE_BUDGET)" in policyc and budc == ref.estimate and pc.estimate <= budc and len(pc.stripes) <= 3)
    for w in (16 * 1024, 1 << 20, 384 << 20):
        plans = [bound.plan(v, lat, 0, w, rows=r, out_bytes=outb) for r in (1, 8, 16, 40, 64, 107, 160, 320)]
        mono = all(a.estimate <= b.estimate for a, b in zip(plans, plans[1:]))
        parts = all(q.live_peak == q.persistent + max(q.prefix_bytes, q.stripe_bytes) and q.prefix_bytes == q.prefix_live
                    and q.stripe_bytes == q.ckpt_bytes + q.stripe_live and q.arena == eng.arena_bytes(q.live_peak, len(q.stripes))
                    and q.estimate == q.arena + q.largest + eng.ESTIMATE_PAD for q in plans)
        order = True
        for q in plans:
            size = [sum(n[1] - n[0] for n in needs) for needs in q.needs]
            order = order and sorted(q.order) == list(range(len(q.stripes))) and size[q.order[0]] == max(size)
        check("workspace {}: estimate monotone in the stripe height ({} .. {}), live peak = persistent + max(prefix, stripes), "
              "estimate = arena + largest allocation + small-pool pad, largest stripe first".format(vae_ops.fmt_bytes(w), vae_ops.fmt_bytes(plans[0].estimate), vae_ops.fmt_bytes(plans[-1].estimate)),
              mono and parts and order)
    # allocator: cache emptied after the self-test, not during the decode; the arena is reserved where supported
    calls, arenas = [], []
    orig = comfy.model_management.soft_empty_cache
    orig_sup, orig_res = eng.arena_supported, eng.reserve_arena
    comfy.model_management.soft_empty_cache = lambda force=False: calls.append(force)
    try:
        eng._SELFTEST.clear()
        ok, _ = eng.self_test(bound, v)
        n_selftest = len(calls)
        mvae.set_stripe_rows(40)
        lat2 = torch.randn(2, 16, 1, 12, 10, generator=g)
        managed_decode(v, lat2)
        last_cpu = mvae.last_decode()
        eng.arena_supported = lambda device: True
        eng.reserve_arena = lambda device, n: (arenas.append(n), orig_res(device, n))
        out = managed_decode(v, lat2, raw=True)
        last = mvae.last_decode()
        mvae.set_stripe_rows(None)
    finally:
        comfy.model_management.soft_empty_cache = orig
        eng.arena_supported, eng.reserve_arena = orig_sup, orig_res
    ref = native_decode(v, lat2, raw=True)
    check("allocator: cache emptied once after the self-test ({} call), never during a decode; arena {} reserved once for a batch of 2 where "
          "supported (CPU: none), result unchanged (max|Δ| {:.2g})".format(n_selftest, vae_ops.fmt_bytes(last["stats"]["arena"]), float((out - ref).abs().max())),
          ok and n_selftest == 1 and len(calls) == 1 and all(calls) and last_cpu["stats"]["arena"] == 0
          and arenas == [last["stats"]["arena"]] and last["stats"]["arena"] == last["estimate"]["arena"]
          and float((out - ref).abs().max()) <= 1e-5)
    import os
    saved = {k: os.environ.get(k) for k in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")}
    try:
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:512"
        no_split = eng.arena_supported(torch.device("cuda")) if torch.cuda.is_available() else False
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
        no_exp = eng.arena_supported(torch.device("cuda")) if torch.cuda.is_available() else False
    finally:
        for k, val in saved.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
    check("arena only with the default allocator config: not on the CPU, not with max_split_size_mb / expandable_segments",
          not eng.arena_supported(torch.device("cpu")) and not no_split and not no_exp)


# ---------------------------------------------------------------------------
# 7. single-frame Conv3d as conv2d
# ---------------------------------------------------------------------------

def conv2d_route_tests(v, lat):
    orig_gate = vae_ops.slow_dilated3d
    vae_ops.slow_dilated3d = lambda x: True   # as on CUDA / HIP without cuDNN
    try:
        g = torch.Generator().manual_seed(9)
        cases = []
        for label, mod, x in (
                ("3x3 CausalConv3d, T=1 (causal_zero)", init_conv(wan.CausalConv3d(8, 12, 3, padding=1)), torch.randn(1, 8, 1, 21, 13, generator=g)),
                ("1x1 CausalConv3d shortcut, T=1, sliced input", init_conv(wan.CausalConv3d(8, 12, 1)), torch.randn(1, 8, 1, 30, 13, generator=g)[:, :, :, 3:24]),
                ("3x3 CausalConv3d, T=3 (not single-frame)", init_conv(wan.CausalConv3d(8, 12, 3, padding=1)), torch.randn(1, 8, 3, 21, 13, generator=g))):
            ref = mod(x)
            for budget in (1 << 30, 4 * 1024):
                st = vae_ops.OpStats()
                with vae_ops.OpChunking(mod, budget, st):
                    y = mod(x)
                cases.append((label, budget, float((y - ref).abs().max()), st.conv3d_as_2d, st.conv_chunked, y.shape == ref.shape))
        for label, budget, e, n2d, nch, shape_ok in cases:
            single = "T=3" not in label
            check("{} (workspace {}): max|Δ| vs the module {:.2g}, {} call(s) as conv2d, {} in row blocks".format(
                label, vae_ops.fmt_bytes(budget), e, n2d, nch), shape_ok and e <= 1e-5 and ((n2d > 0) == single))
        m = init_conv(wan.CausalConv3d(8, 12, 3, padding=1))
        m._conv_forward = m._conv_forward   # an instance-level override from someone else: left alone
        st = vae_ops.OpStats()
        with vae_ops.OpChunking(m, 1 << 30, st):
            m(torch.randn(1, 8, 1, 9, 9, generator=g))
        del m.__dict__["_conv_forward"]
        check("Conv3d with an instance-level _conv_forward override: not routed", st.conv3d_as_2d == 0)
        last = compare_layer1("layer 1 with single-frame Conv3d as conv2d, stripe height 24", v, lat, rows=24)
        check("  ... {} Conv3d calls ran as conv2d".format(last["stats"]["conv3d_as_2d"]), last["stats"]["conv3d_as_2d"] > 0)
    finally:
        vae_ops.slow_dilated3d = orig_gate
    st = vae_ops.OpStats()
    m = init_conv(wan.CausalConv3d(8, 12, 3, padding=1))
    with vae_ops.OpChunking(m, 1 << 30, st):
        m(torch.randn(1, 8, 1, 9, 9))
    check("CPU tensor: SlowDilated3d gate off ({}), Conv3d left as is".format(vae_ops.slow_dilated3d(torch.zeros(1))), st.conv3d_as_2d == 0)


# ---------------------------------------------------------------------------
# 8. caching allocator: simulator fidelity, estimate >= reserved
# ---------------------------------------------------------------------------

def allocator_tests():
    import alloc_sim
    G = float(1 << 30)
    pk, left, bound = alloc_sim.selftest_trace("qwen")
    check("Wan self-test (qwen_image_vae size, meta): simulated reserved peak {:.0f} MiB <= selftest_memory {:.0f} MiB, nothing left".format(
        pk / 2 ** 20, bound / 2 ** 20), pk <= bound and left == 0)
    for label, kw, mres in (
            ("4e54d20, 4K, 128-row stripes", dict(w=3840, h=2160, rows=128, **alloc_sim._version("v1")), 1.07),
            ("4e54d20, 4K, 5 x 432 rows", dict(w=3840, h=2160, rows=512, **alloc_sim._version("v1")), 2.66),
            ("725a010, 4K, 14 x 155 rows", dict(w=3840, h=2160, rows=155, **alloc_sim._version("v2")), 1.44),
            ("725a010, 4K, 5 x 432 rows", dict(w=3840, h=2160, rows=512, **alloc_sim._version("v2")), 2.09),
            ("725a010, fp32 2688, 12 x 128 rows", dict(w=2688, h=1536, dtype="fp32", rows=128, **alloc_sim._version("v2")), 1.41),
            ("85a5c6f (arena, workspace 384 MiB), 4K default", dict(w=3840, h=2160, **alloc_sim._version("v3")), 1.13),
            ("85a5c6f, 4K, 5 x 432 rows", dict(w=3840, h=2160, rows=512, **alloc_sim._version("v3")), 1.82),
            ("current (workspace 128 MiB), 4K default", dict(w=3840, h=2160), 0.87),
            ("current (workspace 128 MiB), 1344 default", dict(w=1344, h=768), 0.36)):
        i = alloc_sim.decode_trace(**kw)
        check("allocator simulator reproduces CT 700: {}: reserved {:.2f} GiB (measured {:.2f})".format(label, i["reserved"] / G, mres),
              abs(i["reserved"] / G - mres) <= 0.03)
    for w, h, dt, rows in ((1344, 768, "bf16", None), (1344, 768, "bf16", 768), (1920, 1088, "bf16", None), (2688, 1536, "bf16", None),
                           (3840, 2160, "bf16", None), (3840, 2160, "bf16", 32), (3840, 2160, "bf16", 512), (3840, 2160, "fp32", None),
                           (7680, 4320, "bf16", None)):
        i = alloc_sim.decode_trace(w, h, dt, rows)
        check("{}x{} {} {}: {} x {} rows, simulated reserved {:.3f} GiB <= estimate {:.3f} GiB (arena {:.3f}, live peak {:.3f})".format(
            w, h, dt, "default" if rows is None else "{} rows".format(rows), i["stripes"], i["rows"], i["reserved"] / G, i["estimate"] / G,
            i["arena"] / G, i["live_peak"] / G), i["reserved"] <= i["estimate"])


def main():
    if not mvae.is_installed():
        check("vae.install() on this ComfyUI", mvae.install())
    mvae.set_stripe(True)
    interval_tests()
    v, lat = decoder_tests()
    recognition_tests(v, lat)
    selftest_tests(v, lat)
    budget_oom_tests(v, lat)
    policy_memory_tests(v)
    conv2d_route_tests(v, lat)
    allocator_tests()
    finish()


if __name__ == "__main__":
    main()
