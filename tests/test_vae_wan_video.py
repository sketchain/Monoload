"""Layer 1 for the Wan 2.1 / 2.2 VAEs decoding video (monoload/vae_wan_video.py, docs/PLAN-video.md group A).

The adapter is called directly (vae.py sends a multi-frame latent to native until hook H1 opens it), so this
tests the adapter itself: recognition, exactness against WanVAE.decode, the self-test, the memory bound.

Small models (default; ComfyUI's own WanVAE / vae2_2.WanVAE with small channels and random weights, fp32):
  1. recognition: Wan 2.1 / 2.2 with T > 1 -> the video adapter (key apart from the single-frame one); T = 1,
     vae_options, a wrong channel count -> not taken, with the reason; the image adapters still refuse T > 1;
  2. whole decode vs native WanVAE.decode, fp32: several frame counts (odd / even: Wan 2.1's last chunk of 1),
     stripe heights (7 / 16 / 40 / whole), batch 2, a small workspace (layer-2 row blocks inside the stripes);
     rel err <= 1e-4;
  3. the self-test (vae_engine.self_test with the video prefix / stripes): passes; an injected bug (the stripe
     caches shared across stripes) -> fails;
  4. with Monoload's VAE switch off (MONOLOAD=0) or the adapter not registered, comfy.sd.VAE.decode of a
     multi-frame latent is bit-identical to native.

Real models (--wan21 / --wan22 PATH to the VAE safetensors; GPU):
  5. fp32 at a small geometry: L1 == native fp32 (rel err <= 1e-4);
  6. bf16 sweep over sizes x frame counts: L1 output vs native bf16 (max abs diff), reserved growth during the
     L1 decode <= the plan's estimate, native's reserved growth for comparison (skipped with --no-native),
     seconds; a table at the end (frame count effect on the peak).

    python tests/test_vae_wan_video.py
    python tests/test_vae_wan_video.py --wan21 /opt/ComfyUI/models/vae/wan_2.1_vae.safetensors --wan22 /opt/ComfyUI/models/vae/wan2.2_vae.safetensors
"""

import argparse
import gc
import os
import sys
import time

_ap = argparse.ArgumentParser()
_ap.add_argument("--wan21", help="Wan 2.1 VAE file (real-model section)")
_ap.add_argument("--wan22", help="Wan 2.2 VAE file (real-model section)")
_ap.add_argument("--no-fp32", action="store_true", help="skip the real-model fp32 self-test / exactness section")
_ap.add_argument("--route", help="WxHxF: also decode this size through comfy.sd.VAE.decode with vae_wan_video put into "
                 "vae.STRIPE_ADAPTERS for the run (needs the multi-frame entry, hook H1): layer 1 chosen, reserved <= its estimate")
_ap.add_argument("--route-l2", default="", help="comma list of models (2.1, 2.2) whose --route size is also decoded with "
                 "layer 1 switched off (layer 2, H1), for comparison")
_ap.add_argument("--fill", action="store_true", help="sweep: the tallest stripes that fit the free memory (default: the "
                 "plan a decode with no budget picks, vae.choose_plan)")
_ap.add_argument("--sizes", default="832x480,1280x720", help="WxH list of the bf16 sweep")
_ap.add_argument("--frames", default="5,9,17,33,49,81", help="pixel frame counts of the bf16 sweep (1 + 4 k)")
_ap.add_argument("--no-native", action="store_true", help="skip native decodes in the sweep (no diff, no native peak)")
_ap.add_argument("--native-dir", help="shared directory of native results, one file per model / size / frames: an existing file is "
                 "read instead of running native again, a new native run is written there")
_ap.add_argument("--no-small", action="store_true", help="skip the small-model sections")
_ap.add_argument("--workspace", default="64M", help="layer-2 workspace inside the stripes on the GPU")
_ap.add_argument("--rows", type=int, default=None, help="force a stripe height in the sweep (default: the plan for the free memory)")
_ap.add_argument("--mem-history", action="store_true", help="sweep: record the allocator's history during each layer-1 "
                 "decode and print what is live at the allocated peak and the segments at the reserved peak (slow)")
OPTS, _rest = _ap.parse_known_args()
sys.argv = [sys.argv[0]] + _rest

import torch  # noqa: E402

from common import check, finish  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.ldm.wan.vae as wan  # noqa: E402
import comfy.ldm.wan.vae2_2 as wan22  # noqa: E402
from monoload import vae as mvae  # noqa: E402
from monoload import vae_engine as eng  # noqa: E402
from monoload import vae_wan, vae_wan22, vae_wan_video as wv  # noqa: E402
from monoload.vae_ops import MIB, OpStats, fmt_bytes  # noqa: E402

torch.manual_seed(0)
GIB = 1024 ** 3


def err(a, b):
    return float((a.float().cpu() - b.float().cpu()).abs().max())


def rel(a, b):
    return err(a, b) / max(1.0, float(b.float().abs().max()))


def raw_native(v, lat):
    """The model's own WanVAE.decode on the VAE's device and dtype (what comfy's native VAE.decode calls), moved
    to the CPU: [B, C, T, H, W]."""
    mm.load_models_gpu([v.patcher])
    with torch.inference_mode():
        out = v.first_stage_model.decode(lat.to(device=v.device, dtype=v.vae_dtype))
        return out.to("cpu")


def l1_decode(v, lat, rows=None, ws=MIB, budget=None):
    """The video adapter's run() with a plan (forced rows, or the largest under budget); process_output off.
    Returns (pixels [B, C, T, H, W] like WanVAE.decode, plan)."""
    bound, why = wv.match(v, lat, {})
    if bound is None:
        raise AssertionError("not taken: " + why)
    mm.load_models_gpu([v.patcher])
    out_b = bound.output_bytes(v, lat)
    if rows is None and budget is None:
        rows = 10 ** 6
    plan = bound.plan(v, lat, budget if budget is not None else 0, ws, rows=rows, out_bytes=out_b)
    po = v.process_output
    v.process_output = lambda image: image
    try:
        with torch.inference_mode():
            out = bound.run(v, lat, plan, ws, OpStats())
    finally:
        v.process_output = po
    return out, plan


# ---------------------------------------------------------------------------
# small models
# ---------------------------------------------------------------------------

def wan21_small(dtype=None):
    from test_vae import wan_vae_model
    return wan_vae_model(dtype)


def wan22_small(dtype=None):
    from test_vae_wan22 import wan22_vae
    return wan22_vae(dtype)


def lat21(t, h, w, b=1, seed=3):
    return torch.randn(b, 16, t, h, w, generator=torch.Generator().manual_seed(seed))


def lat22(t, h, w, b=1, seed=3):
    return torch.randn(b, 48, t, h, w, generator=torch.Generator().manual_seed(seed))


def recognition_tests(v21, v22):
    for label, v, lat_fn, cls, img in (("Wan 2.1", v21, lat21, wv.Wan21Video, vae_wan), ("Wan 2.2", v22, lat22, wv.Wan22Video, vae_wan22)):
        b, why = wv.match(v, lat_fn(5, 6, 5), {})
        check("{} T=5 -> '{}' ({} units, kt {})".format(label, b.name if b else why, len(b.units) if b else 0, b.kt if b else "-"),
              b is not None and type(b) is cls and b.key[0] == "video")
        b1, why1 = wv.match(v, lat_fn(1, 6, 5), {})
        check("{} T=1 not taken by the video adapter ({})".format(label, why1), b1 is None)
        bi, whyi = img.match(v, lat_fn(5, 6, 5), {})
        check("{} T=5 not taken by the single-frame adapter ({})".format(label, whyi), bi is None)
        bo, whyo = wv.match(v, lat_fn(5, 6, 5), {"tile": 1})
        check("{} vae_options -> not taken ({})".format(label, whyo), bo is None)
        wrong = torch.zeros(1, 7, 5, 6, 5)
        bw, whyw = wv.match(v, wrong, {})
        check("{} wrong channel count -> not taken ({})".format(label, whyw), bw is None)
        if b is not None:
            img_b, _ = img.match(v, lat_fn(1, 6, 5), {})
            check("{} video key differs from the single-frame key".format(label), img_b is not None and img_b.key != b.key)
            t = 5
            shape = b.output_shape(lat_fn(t, 6, 5))
            check("{} output shape {} (frames 1 + 4 (T - 1) = {})".format(label, list(shape), 1 + 4 * (t - 1)), shape[2] == 1 + 4 * (t - 1))


def exactness_tests(v21, v22):
    cases = [
        ("Wan 2.1", v21, lat21, [(2, 6, 5), (3, 6, 5), (4, 7, 6), (5, 6, 5), (9, 5, 4)]),
        ("Wan 2.2", v22, lat22, [(2, 6, 5), (3, 5, 6), (5, 4, 4)]),
    ]
    for label, v, lat_fn, geos in cases:
        for t, h, w in geos:
            lat = lat_fn(t, h, w)
            ref = raw_native(v, lat)
            for rows in (7, 16, 40, None):
                out, plan = l1_decode(v, lat, rows=rows)
                e = rel(out, ref)
                check("{} T={} {}x{} rows {}: {} stripes, rel err {:.2g}, shape {}".format(
                    label, t, h, w, rows or "whole", len(plan.stripes), e, list(out.shape)),
                    out.shape == ref.shape and e <= 1e-4)
        lat = lat_fn(3, 6, 5, b=2, seed=9)
        ref = raw_native(v, lat)
        out, plan = l1_decode(v, lat, rows=16)
        check("{} batch 2: rel err {:.2g}".format(label, rel(out, ref)), out.shape == ref.shape and rel(out, ref) <= 1e-4)
        lat = lat_fn(4, 6, 6, seed=5)
        ref = raw_native(v, lat)
        out, plan = l1_decode(v, lat, rows=12, ws=16 * 1024)
        check("{} 16 KiB workspace (row blocks inside the stripes): rel err {:.2g}".format(label, rel(out, ref)), rel(out, ref) <= 1e-4)


def selftest_tests(v21, v22):
    for label, v, lat_fn in (("Wan 2.1", v21, lat21), ("Wan 2.2", v22, lat22)):
        b, _ = wv.match(v, lat_fn(5, 6, 5), {})
        eng._SELFTEST.pop(b.key, None)
        ok, detail = eng.self_test(b, v)
        check("{} self-test: {} ({}), bound {}".format(label, "pass" if ok else "FAIL", detail, fmt_bytes(b.selftest_memory())), ok)
        # injected bug: one cache list for every stripe (the caches of the previous stripe's rows leak in)
        orig_state, orig_call = wv.WanVideoStripe.stripe_state, wv.WanVideoStripe.call_unit
        shared = {}

        def bad_state(self):
            return shared.setdefault(id(self), orig_state(self))

        def bad_call(self, u, x, state, k):
            try:
                return orig_call(self, u, x, state, k)
            except RuntimeError:
                raise eng.StripeError("shape mismatch from a shared cache")
        wv.WanVideoStripe.stripe_state, wv.WanVideoStripe.call_unit = bad_state, bad_call
        try:
            eng._SELFTEST.pop(b.key, None)
            ok2, detail2 = eng.self_test(b, v)
        finally:
            wv.WanVideoStripe.stripe_state, wv.WanVideoStripe.call_unit = orig_state, orig_call
            eng._SELFTEST.pop(b.key, None)
        check("{} self-test with the caches shared across stripes -> fails ({})".format(label, detail2), not ok2)


def native_identity_tests(v21, v22):
    for label, v, lat_fn in (("Wan 2.1", v21, lat21), ("Wan 2.2", v22, lat22)):
        lat = lat_fn(3, 6, 5, seed=11)
        a = comfy.sd.VAE.decode(v, lat)
        bnat = mvae._ORIG["decode"](v, lat)
        check("{} comfy.sd.VAE.decode of T=3 bit-identical to native (adapter not registered: {})".format(
            label, "vae_wan_video" not in [m.__name__.rsplit(".", 1)[-1] for m in mvae.STRIPE_ADAPTERS]), torch.equal(a, bnat))


# ---------------------------------------------------------------------------
# real models
# ---------------------------------------------------------------------------

def load_real(path, dtype):
    sd = comfy.utils.load_torch_file(path)
    return comfy.sd.VAE(sd=sd, dtype=dtype)


def lat_real(v, t, h, w, seed=1):
    c = v.latent_channels
    return torch.randn(1, c, t, h, w, generator=torch.Generator().manual_seed(seed))


def _frame(ev):
    """'file:line name' of the innermost python frame of an allocation inside monoload or comfy."""
    for fr in ev.get("frames") or ():
        f = fr.get("filename", "")
        if ("monoload" in f or "/comfy" in f) and "test_vae_wan_video" not in f:
            return "{}:{} {}".format(os.path.basename(f), fr.get("line"), fr.get("name"))
    fr = (ev.get("frames") or [{}])[0]
    return "{}:{} {}".format(os.path.basename(fr.get("filename", "?")), fr.get("line"), fr.get("name"))


def mem_history_report(top=14):
    """Replay the recorded allocator trace: the live blocks at the allocated peak (grouped by allocating frame)
    and the segments at the reserved peak (blocks live before the recording are not seen)."""
    tr = torch.cuda.memory._snapshot()["device_traces"][torch.cuda.current_device()]
    live, cur, peak, at_peak = {}, 0, 0, []
    segs, seg, segpeak, seg_at = {}, 0, 0, []
    for ev in tr:
        a = ev["action"]
        if a == "alloc":
            live[ev["addr"]] = ev
            cur += ev["size"]
            if cur > peak:
                peak, at_peak = cur, list(live.values())
        elif a == "free_completed":
            e = live.pop(ev["addr"], None)
            if e is not None:
                cur -= e["size"]
        elif a in ("segment_alloc", "segment_map"):
            segs[ev["addr"]] = ev["size"]
            seg += ev["size"]
            if seg > segpeak:
                segpeak, seg_at = seg, sorted(segs.values(), reverse=True)
        elif a in ("segment_free", "segment_unmap"):
            seg -= segs.pop(ev["addr"], 0)
    groups = {}
    for e in at_peak:
        k = _frame(e)
        n, b = groups.get(k, (0, 0))
        groups[k] = (n + 1, b + e["size"])
    print("  mem history: allocated peak +{} ({} live blocks), reserved peak +{} in {} new segments".format(
        fmt_bytes(peak), len(at_peak), fmt_bytes(segpeak), len(seg_at)))
    for k, (n, b) in sorted(groups.items(), key=lambda kv: -kv[1][1])[:top]:
        print("    live at peak: {:>10}  x{:<4} {}".format(fmt_bytes(b), n, k))
    print("    segments at reserved peak: " + ", ".join(fmt_bytes(x) for x in seg_at[:top]))


def measure(fn, history=False):
    """(result or exception, seconds, reserved growth, allocated growth)."""
    gc.collect()
    mm.soft_empty_cache(True)
    torch.cuda.synchronize()
    base_r, base_a = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    if history:
        torch.cuda.memory._record_memory_history(max_entries=2000000, stacks="python")
    t0 = time.perf_counter()
    try:
        res = fn()
        torch.cuda.synchronize()
    except Exception as e:
        if not mm.is_oom(e):
            raise
        res = e
    dt = time.perf_counter() - t0
    if history:
        try:
            mem_history_report()
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
    return res, dt, torch.cuda.max_memory_reserved() - base_r, torch.cuda.max_memory_allocated() - base_a


def real_fp32(label, path):
    v = load_real(path, torch.float32)
    sc = v.spacial_compression_decode()
    lat = lat_real(v, 9, 192 // sc, 320 // sc)
    mm.load_models_gpu([v.patcher])
    b, why = wv.match(v, lat, {})
    check("{} (real, fp32) recognized: {}".format(label, b.name if b else why), b is not None)
    if b is None:
        return
    eng._SELFTEST.pop(b.key, None)
    ok, detail = eng.self_test(b, v)
    check("{} (real) self-test: {} ({}), bound {}".format(label, "pass" if ok else "FAIL", detail, fmt_bytes(b.selftest_memory())), ok)
    with torch.inference_mode():
        ref = raw_native(v, lat)
    out, plan = l1_decode(v, lat, rows=48, ws=64 * MIB)
    e = rel(out.cpu(), ref.cpu())
    check("{} (real, fp32) 320x192 x {} frames, {} stripes: rel err vs native {:.2g}".format(label, out.shape[2], len(plan.stripes), e), e <= 1e-4)
    del v, out, ref
    gc.collect()
    mm.soft_empty_cache(True)


def real_sweep(label, path, table):
    v = load_real(path, torch.bfloat16)
    mm.load_models_gpu([v.patcher])
    sc = v.spacial_compression_decode()
    ws = mvae.parse_size(OPTS.workspace)
    for size in OPTS.sizes.split(","):
        W, H = [int(x) for x in size.split("x")]
        for nf in [int(x) for x in OPTS.frames.split(",")]:
            t = 1 + (nf - 1) // 4
            lat = lat_real(v, t, H // sc, W // sc)
            b, why = wv.match(v, lat, {})
            if b is None:
                check("{} {}x{}x{}: not taken ({})".format(label, W, H, nf, why), False)
                continue
            free = mm.get_free_memory(v.device)
            if OPTS.rows:
                plan = b.plan(v, lat, 0, ws, rows=OPTS.rows, out_bytes=b.output_bytes(v, lat))
            elif OPTS.fill:
                plan = b.plan(v, lat, free, ws, out_bytes=b.output_bytes(v, lat)) or b.smallest_plan(v, lat, ws, b.output_bytes(v, lat))
            else:
                # what a decode picks with no budget (vae.choose_plan: the tallest stripes whose arena stays at that of
                # 128-row stripes), at its workspace
                plan, _, ws, _ = mvae.choose_plan(v, lat, b, b.output_bytes(v, lat))
            po = v.process_output
            v.process_output = lambda image: image

            def run_l1():
                with torch.inference_mode():
                    return b.run(v, lat, plan, ws, OpStats())
            try:
                out, dt, dres, dalloc = measure(run_l1, OPTS.mem_history)
            finally:
                v.process_output = po
            row = {"model": label, "size": size, "frames": nf, "stripes": len(plan.stripes), "rows": plan.rows,
                   "estimate": plan.estimate, "l1_res": dres, "l1_alloc": dalloc, "l1_s": dt, "ckpt": plan.ckpt_bytes}
            if isinstance(out, Exception):
                check("{} {}x{} x{}: L1 OOM: {}".format(label, W, H, nf, str(out).splitlines()[0][:160]), False)
                table.append(row)
                continue
            check("{} {}x{} x{} frames: L1 {} stripes x {} rows, reserved +{} (allocated +{}) <= estimate {} ({:.1f}s)".format(
                label, W, H, nf, len(plan.stripes), plan.rows, fmt_bytes(dres), fmt_bytes(dalloc), fmt_bytes(plan.estimate), dt),
                dres <= plan.estimate and bool(torch.isfinite(out).all()))
            known = native_file(path, size, nf)
            if known is not None and os.path.exists(known):
                row.update(read_native(known))
                print("  native from {}: reserved +{}".format(known, fmt_bytes(row.get("native_res") or 0)))
            elif not OPTS.no_native:
                def run_nat():
                    with torch.inference_mode():
                        return raw_native(v, lat)
                ref, ndt, nres, nalloc = measure(run_nat)
                row.update(native_res=nres, native_s=ndt)
                if isinstance(ref, Exception):
                    row["diff"] = None
                    print("  native OOM at {}x{} x{}".format(W, H, nf))
                else:
                    d = err(out, ref)
                    row["diff"] = d
                    check("{} {}x{} x{}: L1 vs native bf16 max abs diff {:.3g} (native reserved +{}, {:.1f}s)".format(
                        label, W, H, nf, d, fmt_bytes(nres), ndt), out.shape == ref.shape and d <= 0.25)
                    del ref
                if known is not None:
                    write_native(known, path, lat, None if row.get("diff") is None else nres, ndt, nalloc, size, nf)
            table.append(row)
            del out
            gc.collect()
    del v
    gc.collect()
    mm.soft_empty_cache(True)


def route_test(label, path):
    """The whole managed path (hook H1's entry + this adapter): comfy.sd.VAE.decode picks layer 1, self-test first."""
    if not hasattr(mvae, "vae_video"):
        print("  route: this vae.py has no multi-frame entry (H1), skipped")
        return
    if wv not in mvae.STRIPE_ADAPTERS:
        mvae.STRIPE_ADAPTERS.insert(2, wv)
    W, H, nf = [int(x) for x in OPTS.route.split("x")]
    v = load_real(path, torch.bfloat16)
    sc = v.spacial_compression_decode()
    lat = lat_real(v, 1 + (nf - 1) // 4, H // sc, W // sc)
    eng._SELFTEST.clear()
    mm.load_models_gpu([v.patcher])   # weights in place first: the reserved growth is the decode's

    def run():
        with torch.inference_mode():
            return comfy.sd.VAE.decode(v, lat)
    out, dt, dres, dalloc = measure(run)
    rec = mvae.last_decode()
    est = (rec.get("estimate") or {}).get("total") if isinstance(rec.get("estimate"), dict) else rec.get("estimate")
    ok = not isinstance(out, Exception) and rec.get("strategy") == "layer1" and est is not None and dres <= est
    check("{} route {}x{}x{}: comfy.sd.VAE.decode -> {} ({}), reserved +{} <= estimate incl. self-test {} ({:.1f}s)".format(
        label, W, H, nf, rec.get("strategy"), rec.get("adapter") or rec.get("reason"), fmt_bytes(dres),
        fmt_bytes(est or 0), dt), ok)
    if not isinstance(out, Exception):
        check("{} route: output {} frames, finite".format(label, out.shape[1]),
              out.shape[1] == nf and bool(torch.isfinite(out).all()))
    del out
    if label.split()[-1] in [x.strip() for x in OPTS.route_l2.split(",") if x.strip()]:
        old = mvae.stripe_enabled()
        mvae.set_stripe(False)
        try:
            out2, dt2, dres2, _ = measure(run)
        finally:
            mvae.set_stripe(old)
        rec2 = mvae.last_decode()
        e2 = rec2.get("estimate")
        e2 = e2.get("total") if isinstance(e2, dict) else e2
        print("  route layer 2 for comparison: {} reserved +{} (estimate {}, {:.1f}s)".format(
            rec2.get("strategy"), fmt_bytes(dres2), fmt_bytes(e2 or 0), dt2))
        del out2
    del v
    gc.collect()
    mm.soft_empty_cache(True)


def native_file(path, size, nf):
    # shared layout (README.md in that directory): <vae file stem>__<W>x<H>x<frames>f.md
    if not OPTS.native_dir:
        return None
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(OPTS.native_dir, "{}__{}x{}f.md".format(stem, size, nf))


def read_native(path):
    import re
    text = open(path, encoding="utf-8").read()
    d = {}
    m = re.search(r"reserved\s*Δ\s*([0-9.]+)\s*GiB", text)
    if m:
        d["native_res"] = int(float(m.group(1)) * GIB)
    m = re.search(r"warm[^0-9]*([0-9.]+)\s*s", text) or re.search(r"耗时[^0-9]*([0-9.]+)\s*s", text)
    if m:
        d["native_s"] = float(m.group(1))
    return d


def write_native(path, vae_path, lat, res, secs, alloc, size, nf):
    import datetime
    os.makedirs(os.path.dirname(path), exist_ok=True)
    name = os.path.basename(vae_path)
    W, H = size.split("x")
    peak = "OOM" if res is None else "reserved Δ {:.2f} GiB，alloc Δ {:.2f} GiB".format(res / GIB, alloc / GIB)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# {}，{}×{}×{} 帧，原生\n\n".format(name, W, H, nf))
        f.write("- vae：{}，bf16（first_stage_model.decode 本身，不经 comfy VAE.decode 包装）\n".format(name))
        f.write("- latent：{}，随机 randn（generator seed 1）\n".format(list(lat.shape)))
        f.write("- 原生峰值：{}（权重已加载、empty_cache 之后的 max_memory_reserved 增量，含 GPU 上的 latent 和输出）\n".format(peak))
        f.write("- 耗时：{:.1f} s（单次，cold）\n".format(secs))
        f.write("- 环境：ComfyUI v0.46.3（cab5f4e），comfyui-latest（CT 103），{}，wan-video-l1 线程"
                "（tests/test_vae_wan_video.py）\n".format(datetime.date.today().isoformat()))
        f.write("- 原始日志：/srv/monoload-wt/logs/wanv-*.log\n")


def print_table(table):
    if not table:
        return
    print("\nmodel    size       frames  stripes x rows   ckpt      estimate   L1 reserved  L1 s    native reserved  native s  max|diff|")
    for r in table:
        print("{:<8} {:<10} {:>6}  {:>4} x {:<6}  {:>9} {:>10} {:>12} {:>6} {:>16} {:>9} {:>10}".format(
            r["model"], r["size"], r["frames"], r["stripes"], r["rows"], fmt_bytes(r["ckpt"]), fmt_bytes(r["estimate"]),
            fmt_bytes(r["l1_res"]), "{:.1f}".format(r["l1_s"]),
            fmt_bytes(r["native_res"]) if r.get("native_res") is not None else "-",
            "{:.1f}".format(r["native_s"]) if r.get("native_s") is not None else "-",
            "{:.3g}".format(r["diff"]) if r.get("diff") is not None else "-"))


if __name__ == "__main__":
    if not mvae.is_installed():
        check("vae.install() on this ComfyUI", mvae.install())
    if not OPTS.no_small:
        v21, v22 = wan21_small(), wan22_small()
        recognition_tests(v21, v22)
        exactness_tests(v21, v22)
        selftest_tests(v21, v22)
        native_identity_tests(v21, v22)
    if OPTS.wan21 or OPTS.wan22:
        if not torch.cuda.is_available():
            check("real-model section needs a GPU", False)
        else:
            table = []
            for label, path in (("Wan 2.1", OPTS.wan21), ("Wan 2.2", OPTS.wan22)):
                if path:
                    if not OPTS.no_fp32:
                        real_fp32(label, path)
                    real_sweep(label, path, table)
                    if OPTS.route:
                        route_test(label, path)
            print_table(table)
    finish()
