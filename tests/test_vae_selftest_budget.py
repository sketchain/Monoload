"""A forced layer-1 configuration whose self-test fails, under a budget
(monoload/vae.py choose_budget, review 02): the same decision whether the
failure happens in this decode or was cached by an earlier one -- layer 2 if
it fits the budget, else MonoloadError naming the forced setting, the failed
variant and layer 2's need. Layer 2 only and forced stripe rows keep their
own semantics. The layer-2 record counts a self-test that failed in that
decode (review 2026-10 item 09). Self-test failures and layer 2's estimate are injected; no
model files: the small SDXL-like VAE of tests/test_vae_ldm.py, fp32 on the CPU.

    MODELS=/tmp/nomodels tests/docker_run.sh python tests/test_vae_selftest_budget.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import check, finish  # noqa: E402
import torch  # noqa: E402

from monoload import vae as mvae  # noqa: E402
from monoload import vae_engine as eng  # noqa: E402
from monoload import vae_ldm as vl  # noqa: E402
from monoload.errors import MonoloadError  # noqa: E402
from test_vae import managed_decode, native_decode  # noqa: E402
from test_vae_ldm import ldm_vae  # noqa: E402

GIB = 1 << 30


def main():
    if not mvae.is_installed():
        check("vae.install() on this ComfyUI", mvae.install())
    mvae.set_stripe(True)
    v = ldm_vae(4, True)
    lat = torch.randn(1, 4, 64, 6, generator=torch.Generator().manual_seed(9))   # 512 output rows
    ref = native_decode(v, lat, raw=True)
    variants = vl.match(v, lat, {})[0].variants()
    key = {b.gn_scheme: b.key for b in variants}
    l2_total = [1 << 40]
    orig_l2, orig_run = mvae._layer2_estimate, eng._self_test_run
    fail = set()

    def l2(*a, **kw):
        est, probe = orig_l2(*a, **kw)
        return dict(est, total=l2_total[0]), probe

    def selftest_run(b, vae):
        return (False, "injected") if b.gn_scheme in fail else (True, "stub")
    mvae._layer2_estimate, eng._self_test_run = l2, selftest_run

    def state(failing, cached):
        """Self-test results: `failing` fail; cached=True: recorded already (an earlier decode), else run now."""
        fail.clear()
        fail.update(failing)
        eng._SELFTEST.clear()
        for s, k in key.items():
            if cached or s not in failing:
                eng._SELFTEST[k] = (s not in failing, "injected" if s in failing else "stub")

    def decode():
        try:
            out = managed_decode(v, lat, raw=True)
        except MonoloadError as e:
            return "error", str(e)
        m = mvae.last_decode()
        assert float((out - ref).abs().max()) <= 1e-4
        return "layer{}".format(1 if m.get("strategy") == "layer1" else 2) + (" " + m["gn_scheme"] if m.get("gn_scheme") else ""), ""

    def both(label, failing, expect, words=()):
        got = {}
        for cached in (False, True):
            state(failing, cached)
            got[cached] = decode()
        ok = got[False][0] == got[True][0] == expect and all(w in got[False][1] and w in got[True][1] for w in words)
        check("{}: fresh failure -> {}, cached failure -> {} (expected {}){}".format(
            label, got[False][0], got[True][0], expect, (": " + got[True][1][:200]) if got[True][1] else ""), ok)

    mvae.set_budget(GIB)
    try:
        mvae.set_gn_scheme("C")
        l2_total[0] = 1 << 40
        both("forced scheme C fails its self-test, layer 2 above the budget", {"C"}, "error",
             ("scheme C", "self-test failed", "LDM stripes", "GroupNorm scheme C", "layer 2 needs about 1024.00 GiB"))
        l2_total[0] = 1 << 20
        both("forced scheme C fails its self-test, layer 2 within the budget", {"C"}, "layer2")
        # review 2026-10 item 09: the layer-2 record of such a decode counts the self-test that ran before it (fresh
        # failure), as a layer-1 record does; none when the failure was cached; also on the default path (no budget)
        need = next(b for b in variants if b.gn_scheme == "C").selftest_memory()
        recs = {}
        for label, bud, cached in (("budget, fresh", GIB, False), ("budget, cached", GIB, True), ("no budget, fresh", None, False)):
            mvae.set_budget(bud)
            state({"C"}, cached)
            decode()
            recs[label] = dict(mvae.last_decode().get("estimate") or {}, strategy=mvae.last_decode().get("strategy"))
        mvae.set_budget(GIB)
        check("layer-2 record after a failed self-test: estimate total = max(layer 2 {}, self-test {}), selftest field ({})".format(
              recs["budget, fresh"].get("decode"), need, "; ".join("{}: total {} selftest {}".format(k, r.get("total"), r.get("selftest"))
                                                                    for k, r in recs.items())),
              all(r["strategy"] == "layer2" for r in recs.values())
              and recs["budget, fresh"]["selftest"] == need and recs["budget, fresh"]["total"] == max(need, recs["budget, fresh"]["decode"])
              and recs["budget, cached"]["selftest"] == 0 and recs["budget, cached"]["total"] == recs["budget, cached"]["decode"]
              and recs["no budget, fresh"]["selftest"] == need and recs["no budget, fresh"]["total"] >= need)
        mvae.set_gn_scheme(None)
        mvae.set_stripe_rows(64)
        l2_total[0] = 1 << 40
        both("forced 64 rows, every scheme fails its self-test, layer 2 above the budget", {"A", "B", "C", "D"}, "error",
             ("64 rows", "self-test failed"))
        l2_total[0] = 1 << 20
        both("forced 64 rows, every scheme fails, layer 2 within the budget", {"A", "B", "C", "D"}, "layer2")
        state(set(), True)
        first = decode()[0].split()[-1]        # the scheme chosen when nothing fails
        got = {}
        for cached in (False, True):
            state({first}, cached)
            got[cached] = decode()[0]
        check("forced 64 rows, the chosen scheme {} fails: the next fastest either way (fresh {}, cached {})".format(first, got[False], got[True]),
              got[False] == got[True] and got[True].startswith("layer1") and not got[True].endswith(first))
        mvae.set_budget(1 << 20)
        state(set(), True)
        got = decode()
        check("forced 64 rows above a 1 MiB budget, no failure: runs anyway (unchanged): {}".format(got[0]), got[0].startswith("layer1"))
        mvae.set_stripe_rows(None)
        mvae.set_stripe(False)
        l2_total[0] = 1 << 40
        state({"A", "B", "C", "D"}, True)
        got = decode()
        check("layer 2 only with a 1 MiB budget: layer 2 anyway (unchanged): {}".format(got[0]), got[0] == "layer2")
        mvae.set_stripe(True)
        mvae.set_budget(GIB)
        both("nothing forced, every scheme fails, layer 2 above the budget", {"A", "B", "C", "D"}, "error", ("self-test failed",))
    finally:
        mvae._layer2_estimate, eng._self_test_run = orig_l2, orig_run
        eng._SELFTEST.clear()
        mvae.set_budget(None)
        mvae.set_gn_scheme(None)
        mvae.set_stripe_rows(None)
        mvae.set_stripe(True)
    finish()


if __name__ == "__main__":
    main()
