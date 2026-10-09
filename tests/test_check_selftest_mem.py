"""tests/check_selftest_mem.py (CT 700 diagnostic) chooses the layer-1 variant
the way the decodes will: with a budget that picks another GroupNorm scheme
than the default, the measured self-test is that scheme's, and the decodes run
no further self-test. No model files: the small SDXL-like VAE of
tests/test_vae_ldm.py, fp32 on the CPU.

    MODELS=/tmp/nomodels tests/docker_run.sh python tests/test_check_selftest_mem.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from common import check, finish  # noqa: E402
from monoload import vae as mvae  # noqa: E402
from monoload import vae_engine as eng  # noqa: E402
from monoload import vae_ldm as vl  # noqa: E402
from monoload.errors import MonoloadError  # noqa: E402
from test_vae import managed_decode  # noqa: E402
from test_vae_ldm import ldm_vae  # noqa: E402
import check_selftest_mem as C  # noqa: E402


def main():
    if not mvae.is_installed():
        check("vae.install() on this ComfyUI", mvae.install())
    mvae.set_stripe(True)
    sd = ldm_vae(4, True)
    lat = torch.randn(1, 4, 64, 6, generator=torch.Generator().manual_seed(3))   # 512 output rows
    ws = 16 * 1024
    mvae.set_workspace(ws)
    orig_l2 = mvae._layer2_estimate

    def big_l2(*a, **kw):      # this decoder is so small that layer 2 would always fit first
        est, probe = orig_l2(*a, **kw)
        return dict(est, total=1 << 40), probe
    mvae._layer2_estimate = big_l2
    # this decoder is so small that its self-test bound (~0.1 GiB) is above every plan: made small here, so that the
    # budget's choice depends on the plans (what this test is about is the script's order, not the bound)
    orig_stm = eng.StripeAdapter.selftest_memory
    eng.StripeAdapter.selftest_memory = lambda self: 1 << 20
    try:
        # a budget at which the budget picks the same non-default scheme (several stripes) whether or not the
        # self-tests have run (a pending self-test counts in the estimate, DESIGN §9.20)
        variants = vl.match(sd, lat, {})[0].variants()
        floor = min(b.smallest_plan(sd, lat, ws, b.output_bytes(sd, lat)).estimate for b in variants)

        def choice(bud, tested):
            eng._SELFTEST.clear()
            if tested:
                eng._SELFTEST.update({b.key: (True, "stub") for b in variants})
            mvae.set_budget(bud)
            try:
                d = mvae.choose_budget(sd, lat, {}, bud, selftest=lambda b: (True, "stub"))
            except MonoloadError:
                return None
            finally:
                mvae.set_budget(None)
                eng._SELFTEST.clear()
            return d["bound"].gn_scheme if d["layer"] == 1 and len(d["plan"].stripes) > 1 else None
        bud, pick = floor, None
        while bud < 16 * floor and pick is None:
            c = choice(bud, False)
            if c is not None and c != vl.DEFAULT_SCHEME and choice(bud, True) == c:
                pick = bud
            bud = int(bud * 1.03) + 1
        check("a budget at which the budget picks another scheme than the default {}: {}".format(vl.DEFAULT_SCHEME, pick), pick is not None)
        if pick is None:
            finish()   # exits with the failure above (a bare return would skip finish() and exit 0)
        mvae.set_budget(pick)
        eng._SELFTEST.clear()
        mvae._PROBES.clear()
        default_bound = mvae._select_layer1(sd, lat, {})[0]
        bound, why = C.selected_bound(sd, lat, pick)
        check("selected_bound: scheme {} (the default bound is {}), no self-test run and no shape probe kept while choosing".format(
            getattr(bound, "gn_scheme", None), default_bound.gn_scheme),
            bound is not None and bound.gn_scheme != default_bound.gn_scheme and not eng._SELFTEST
            and sd.first_stage_model not in mvae._PROBES)
        ok, _ = eng.self_test(bound, sd)
        managed_decode(sd, lat)
        m = mvae.last_decode()
        check("after the self-test of the selected bound, decode 1 runs scheme {} with no self-test in it (selftest {})".format(
            m.get("gn_scheme"), C.selftest_in_decode(m)),
            ok and m.get("gn_scheme") == bound.gn_scheme and C.selftest_in_decode(m) == 0)
        # the script's earlier order: the default bound self-tested, then the budget picks another scheme
        eng._SELFTEST.clear()
        eng.self_test(default_bound, sd)
        managed_decode(sd, lat)
        m = mvae.last_decode()
        check("(the earlier order: default {} self-tested, decode 1 with scheme {} ran its own self-test too: selftest {} > 0, now reported)".format(
            default_bound.gn_scheme, m.get("gn_scheme"), C.selftest_in_decode(m)), C.selftest_in_decode(m) > 0)
    finally:
        eng.StripeAdapter.selftest_memory = orig_stm
        mvae._layer2_estimate = orig_l2
        mvae.set_budget(None)
        mvae.set_workspace(mvae.DEFAULT_WORKSPACE)
    finish()


if __name__ == "__main__":
    main()
