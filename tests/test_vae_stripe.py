"""Layer 1: stripe decoding of the Wan 2.1 VAE (monoload/vae_stripe.py).

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
     switched off -> layer 2; SDXL / Flux structures keep layer 2 with the same
     result as before;
  4. self-test: a halo one row short (caught by the per-unit validity check) and
     a halo one row short with a matching wrong validity rule (caught
     numerically) both fail the self-test and the decode falls back to layer 2;
  5. budget (explicit too small -> error naming the need), OOM -> smaller
     stripes, at the floor MonoloadVAEOOMError, never tiled, never layer 2;
  6. default policy (the peak of 128-row stripes, tallest stripes within it),
     memory model (estimate = persistent + max(prefix, stripes), monotone in
     the stripe height, the largest stripe runs first), the allocator cache is
     emptied after the self-test and between prefix and stripes;
  7. single-frame Conv3d as conv2d (forced on the CPU): same result as the
     module, only where the 3D call is single-frame with an effective kT=1 and
     a known _conv_forward.

    python tests/test_vae_stripe.py
"""

import random

import torch

from common import check, expect_raises, finish
import comfy.model_management
import comfy.sd
import comfy.ldm.wan.vae as wan
from monoload import vae as mvae
from monoload import vae_stripe as vs
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
                if u.kind == vs.UP:
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
    kinds = [(vs.POINT, 0, 1), (vs.CONV, 1, 1), (vs.RES, 2, 1), (vs.UP, 1, 2)]
    bad = 0
    n = 0
    for _ in range(300):
        chain = [vs.Unit(*rng.choice(kinds)[:1], None, 0, 1, 1, 1, "u") for _ in range(rng.randint(1, 9))]
        for u in chain:
            k = next(k for k in kinds if k[0] == u.kind)
            u.halo, u.scale = k[1], k[2]
        h0 = rng.randint(1, 12)
        heights = [h0]
        for u in chain:
            heights.append(heights[-1] * u.scale)
        for o0, o1 in vs.split_rows(heights[-1], rng.randint(1, heights[-1])):
            n += 1
            if vs.stripe_needs(chain, heights, o0, o1) != brute_needs(chain, heights, o0, o1):
                bad += 1
    check("need_in / stripe_needs == brute-force dependency walk ({} stripes over 300 random unit chains)".format(n), bad == 0, "{} mismatches".format(bad))
    ok = all(sum(b - a for a, b in vs.split_rows(h, r)) == h and max(b - a for a, b in vs.split_rows(h, r)) <= r
             and max(b - a for a, b in vs.split_rows(h, r)) - min(b - a for a, b in vs.split_rows(h, r)) <= 1
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
    mods = [("ResidualBlock 8->8", vs.Unit(vs.RES, rb(8, 8), 2, 1, 8, 8, "rb")),
            ("ResidualBlock 6->8 (1x1 shortcut)", vs.Unit(vs.RES, rb(6, 8), 2, 1, 6, 8, "rb2")),
            ("Resample upsample2d 8->4", vs.Unit(vs.UP, init_random(wan.Resample(8, "upsample2d")), 1, 2, 8, 4, "up")),
            ("Resample upsample3d 8->4", vs.Unit(vs.UP, init_random(wan.Resample(8, "upsample3d")), 1, 2, 8, 4, "up3")),
            ("head CausalConv3d 3x3", vs.Unit(vs.CONV, init_conv(wan.CausalConv3d(8, 3, 3, padding=1)), 1, 1, 8, 3, "conv"))]
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
                va, vb = vs.valid_out(u, xa, xb, h_out)
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
    # budget-chosen height with several stripes (tiny workspace, so the activations decide)
    mvae.set_workspace(16 * 1024)
    bound, _ = vs.match(v, lat, {})
    p24 = bound.plan(v, lat, 0, 16 * 1024, rows=24, out_bytes=mvae._out_bytes(v, lat, 96, 80, 3))
    mvae.set_budget(p24.estimate)
    last = compare_layer1("12x10 latent, height from a budget that fits 24-row stripes", v, lat)
    check("budget-chosen plan: estimate {} <= budget {}, {} stripes of {} rows (24 rows fit, the whole image does not)".format(
        vae_ops.fmt_bytes(last["estimate"]["total"]), vae_ops.fmt_bytes(last["budget"]), last["stripes"], last["rows"]),
        last["estimate"]["total"] <= last["budget"] and 1 < last["stripes"] <= 4 and last["rows"] >= 24)
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
    vs._SELFTEST.clear()
    mvae.set_stripe(False)
    managed_decode(v, lat)
    last = mvae.last_decode()
    check("MONOLOAD_DISABLE_VAE_STRIPE (set_stripe(False)): Wan decode uses layer 2", last.get("strategy") == "layer2" and "disabled" in (last.get("layer1") or ""))
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
          vs.match(v, lat[:, :, 0], {})[0] is None and vs.match(v, torch.cat([lat, lat], 2), {})[0] is None and vs.match(v, lat, {"x": 1})[0] is None)
    g = torch.Generator().manual_seed(3)
    for label, vv, l4 in (("SDXL-like", ldm_vae(4, True), torch.randn(1, 4, 12, 10, generator=g)),
                          ("Flux-like", ldm_vae(16, False), torch.randn(1, 16, 12, 10, generator=g))):
        mvae.set_workspace(16 * 1024)
        out = managed_decode(vv, l4, raw=True)
        last = mvae.last_decode()
        ref = native_decode(vv, l4, raw=True)
        mvae.set_workspace(mvae.DEFAULT_WORKSPACE)
        check("{} (LDM Decoder) keeps layer 2: {} chunked conv calls, max|Δ| vs native {:.2g} ({})".format(
            label, last["stats"]["conv_chunked"], float((out - ref).abs().max()), last.get("layer1")),
            last.get("strategy") == "layer2" and "not comfy.ldm.wan.vae.WanVAE" in (last.get("layer1") or "")
            and last["stats"]["conv_chunked"] > 0 and float((out - ref).abs().max()) <= 1e-4)


# ---------------------------------------------------------------------------
# 4. self-test
# ---------------------------------------------------------------------------

def selftest_tests(v, lat):
    vs._SELFTEST.clear()
    bound, _ = vs.match(v, lat, {})
    ok, detail = vs.self_test(bound, v)
    check("self-test of the real structure passes: {}".format(detail), ok)
    orig_need, orig_valid = vs.need_in, vs.valid_out

    def short_need(unit, a, b, h_in, h_out):
        if unit.kind == vs.RES:
            return max(0, a - 1), min(h_in, b + 1)
        return orig_need(unit, a, b, h_in, h_out)

    def short_valid(unit, xa, xb, h_out):
        if unit.kind == vs.RES:
            return xa + (1 if xa > 0 else 0), xb - (1 if xb < h_out else 0)
        return orig_valid(unit, xa, xb, h_out)

    for label, patches in (("halo one row short (validity check)", {"need_in": short_need}),
                           ("halo one row short + matching wrong validity rule (numeric comparison)", {"need_in": short_need, "valid_out": short_valid})):
        vs._SELFTEST.clear()
        for k, f in patches.items():
            setattr(vs, k, f)
        try:
            out = managed_decode(v, lat, raw=True)
            last = mvae.last_decode()
        finally:
            vs.need_in, vs.valid_out = orig_need, orig_valid
        ref = native_decode(v, lat, raw=True)
        note = last.get("layer1") or ""
        check("injected bug: {} -> self-test fails, decode falls back to layer 2, result == native (max|Δ| {:.2g}): {}".format(
            label, float((out - ref).abs().max()), note[:160]),
            last.get("strategy") == "layer2" and "self-test failed" in note and float((out - ref).abs().max()) <= 1e-4)
    vs._SELFTEST.clear()
    rng = torch.get_rng_state()
    vs.self_test(bound, v)
    check("self-test leaves the RNG state unchanged and is cached per structure",
          torch.equal(rng, torch.get_rng_state()) and bound.key in vs._SELFTEST)


# ---------------------------------------------------------------------------
# 5. budget, OOM
# ---------------------------------------------------------------------------

def budget_oom_tests(v, lat):
    mvae.set_budget(1 << 20)
    expect_raises("budget too small (MONOLOAD_VAE_BUDGET) -> MonoloadError naming what is needed", MonoloadError,
                  lambda: managed_decode(v, lat), "MONOLOAD_VAE_BUDGET", "需要")
    mvae.set_budget(None)

    orig_run = vs.run_stripes
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
        vs.run_stripes = oom_above(24)
        mvae.set_stripe_rows(96)
        with Spy() as spy:
            out = managed_decode(v, lat, raw=True)
        last = mvae.last_decode()
        ref = native_decode(v, lat, raw=True)
        check("simulated OOM above 24-row stripes: retried with {}-row stripes after {} retries, result == native (max|Δ| {:.2g})".format(
            last.get("rows"), last.get("retries"), float((out - ref).abs().max())),
            last.get("strategy") == "layer1" and last.get("rows") <= 24 and last.get("retries") == 2
            and float((out - ref).abs().max()) <= 1e-5 and spy.tiled == 0 and calls_l2[0] == 0)
        vs.run_stripes = oom_above(0)
        with Spy() as spy:
            expect_raises("OOM even with the smallest stripes -> MonoloadVAEOOMError", MonoloadVAEOOMError,
                          lambda: managed_decode(v, lat), "不会退回到 tiled", "第二层")
        check("... neither tiled nor layer 2 was called, no override left", spy.tiled == 0 and calls_l2[0] == 0 and no_overrides(v.first_stage_model))
    finally:
        vs.run_stripes = orig_run
        mvae._run = orig_l2
        mvae.set_stripe_rows(None)


# ---------------------------------------------------------------------------
# 6. default policy, memory model, allocator cache
# ---------------------------------------------------------------------------

def policy_memory_tests(v):
    g = torch.Generator().manual_seed(5)
    lat = torch.randn(1, 16, 1, 40, 6, generator=g)   # 320 output rows
    bound, _ = vs.match(v, lat, {})
    outb = mvae._out_bytes(v, lat, 320, 48, 3)
    ws = mvae.layer1_workspace()
    mvae.set_budget(None)
    ref = bound.plan(v, lat, 0, ws, rows=mvae.DEFAULT_POLICY_ROWS, out_bytes=outb)
    last = compare_layer1("320-row image, default policy", v, lat, expect_stripes=3)
    check("default policy: target = estimate of {}-row stripes ({}), tallest stripes within it: {} x {} rows, estimate {} ({})".format(
        mvae.DEFAULT_POLICY_ROWS, vae_ops.fmt_bytes(ref.estimate), last["stripes"], last["rows"], vae_ops.fmt_bytes(last["estimate"]["total"]), last["policy"]),
        last["policy"].startswith("default") and last["budget"] == ref.estimate and last["estimate"]["total"] <= ref.estimate
        and bound.plan(v, lat, 0, ws, rows=160, out_bytes=outb).estimate > ref.estimate)
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
        and policyc == "MONOLOAD_VAE_BUDGET" and budc == ref.estimate and pc.estimate <= budc and len(pc.stripes) <= 3)
    for w in (16 * 1024, 1 << 20, 384 << 20):
        plans = [bound.plan(v, lat, 0, w, rows=r, out_bytes=outb) for r in (1, 8, 16, 40, 64, 107, 160, 320)]
        mono = all(a.estimate <= b.estimate for a, b in zip(plans, plans[1:]))
        parts = all(q.estimate == q.persistent + max(q.prefix_bytes, q.stripe_bytes)
                    and q.prefix_bytes == vs.with_slack(q.prefix_live) and q.stripe_bytes == q.ckpt_bytes + vs.with_slack(q.stripe_live) for q in plans)
        order = True
        for q in plans:
            size = [sum(n[1] - n[0] for n in needs) for needs in q.needs]
            order = order and sorted(q.order) == list(range(len(q.stripes))) and size[q.order[0]] == max(size)
        check("workspace {}: estimate monotone in the stripe height ({} .. {}), = persistent + max(prefix, stripes) with the allocator slack, "
              "largest stripe first".format(vae_ops.fmt_bytes(w), vae_ops.fmt_bytes(plans[0].estimate), vae_ops.fmt_bytes(plans[-1].estimate)),
              mono and parts and order)
    # allocator cache: emptied after the self-test and between prefix and stripes
    calls = []
    orig = comfy.model_management.soft_empty_cache
    comfy.model_management.soft_empty_cache = lambda force=False: calls.append(force)
    try:
        vs._SELFTEST.clear()
        ok, _ = vs.self_test(bound, v)
        n_selftest = len(calls)
        mvae.set_stripe_rows(40)
        managed_decode(v, torch.randn(2, 16, 1, 12, 10, generator=g))
        mvae.set_stripe_rows(None)
        last = mvae.last_decode()
    finally:
        comfy.model_management.soft_empty_cache = orig
    check("allocator cache emptied after the self-test ({} call) and once per sample between prefix and stripes ({} for batch 2)".format(
        n_selftest, last["stats"]["cache_releases"]), ok and n_selftest == 1 and last["stats"]["cache_releases"] == 2 and all(calls))


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
    finish()


if __name__ == "__main__":
    main()
