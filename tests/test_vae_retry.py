"""Layer-1 OOM retries (monoload/vae.py _decode_layer1, review 06): stripes and
workspace are halved after an OOM; a step whose plan would need more than the
first plan's estimate is skipped (halving again); at the smallest stripes and
workspace the OOM is reported (MonoloadVAEOOMError), naming the plan that ran
out of memory last and, apart, the steps skipped after it (review 2026-10
item 08). OOMs and estimates are
injected; no model files: the small SDXL-like VAE of tests/test_vae_ldm.py,
fp32 on the CPU.

    MODELS=/tmp/nomodels tests/docker_run.sh python tests/test_vae_retry.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import check, finish  # noqa: E402
import torch  # noqa: E402

import comfy.model_management as mm  # noqa: E402
from monoload import vae as mvae  # noqa: E402
from monoload import vae_engine as eng  # noqa: E402
from monoload import vae_ops  # noqa: E402
from monoload.errors import MonoloadVAEOOMError  # noqa: E402
from test_vae import managed_decode, native_decode  # noqa: E402
from test_vae_ldm import ldm_vae  # noqa: E402


RUNS = []   # (rows, workspace, estimate) of every run of the last case


class Logs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def run_case(v, lat, ooms, inflate):
    """Decode with the first `ooms` runs raising an OOM and the estimate of every retry plan whose core height is
    in `inflate` set above the first plan's. -> (rows of each run, exception or None, output, log lines)."""
    runs = []
    RUNS.clear()
    orig_run, orig_plan = eng.StripeAdapter.run, eng.StripeAdapter.plan
    first = {}

    def run(self, vae, samples, plan, ws, stats):
        runs.append(max(b - a for a, b in plan.stripes))
        RUNS.append((runs[-1], ws, plan.estimate))
        if len(runs) <= ooms:
            raise mm.OOM_EXCEPTION("injected")
        return orig_run(self, vae, samples, plan, ws, stats)

    def plan(self, vae, samples, budget, ws, rows=None, **kw):
        p = orig_plan(self, vae, samples, budget, ws, rows=rows, **kw)
        first.setdefault("est", p.estimate)
        if rows in inflate:
            p.estimate = first["est"] + (1 << 20)
        return p
    logs = Logs()
    logging.getLogger().addHandler(logs)
    eng.StripeAdapter.run, eng.StripeAdapter.plan = run, plan
    err = out = None
    try:
        out = managed_decode(v, lat, raw=True)
    except Exception as e:
        err = e
    finally:
        eng.StripeAdapter.run, eng.StripeAdapter.plan = orig_run, orig_plan
        logging.getLogger().removeHandler(logs)
    return runs, err, out, logs.lines


def main():
    if not mvae.is_installed():
        check("vae.install() on this ComfyUI", mvae.install())
    mvae.set_stripe(True)
    v = ldm_vae(4, True)
    lat = torch.randn(1, 4, 64, 6, generator=torch.Generator().manual_seed(5))   # 512 output rows
    ref = native_decode(v, lat, raw=True)
    mvae.set_stripe_rows(128)
    try:
        managed_decode(v, lat)        # the self-test, outside the injected runs
        runs, err, out, _ = run_case(v, lat, 1, set())
        last = mvae.last_decode()
        check("one OOM: retried with 64-row stripes (runs {}), {} retry, == native (max|Δ| {:.2g})".format(
            runs, last.get("retries"), float((out - ref).abs().max()) if out is not None else -1),
            err is None and runs == [128, 64] and last.get("retries") == 1 and float((out - ref).abs().max()) <= 1e-5)
        runs, err, out, lines = run_case(v, lat, 1, {64})
        skip = [l for l in lines if "skipped" in l]
        last = mvae.last_decode()
        check("one OOM, the 64-row plan's estimate above the first plan's: skipped, retried with 32 rows (runs {}), logged: {}".format(
            runs, skip[0][:110] if skip else None),
            err is None and runs == [128, 32] and len(skip) == 1 and "64-row" in skip[0] and last.get("retries") == 1
            and float((out - ref).abs().max()) <= 1e-5)
        runs, err, out, lines = run_case(v, lat, 1, {64, 32, 16, 8})
        check("every smaller plan above the first plan's estimate: no retry runs (runs {}), {} steps skipped, then the OOM error: {}".format(
            runs, sum("skipped" in l for l in lines), type(err).__name__),
            isinstance(err, MonoloadVAEOOMError) and runs == [128] and sum("skipped" in l for l in lines) == 4)
        runs, err, out, lines = run_case(v, lat, 10, {16, 8})
        r_rows, r_ws, r_est = RUNS[-1]
        text = str(err)
        want = ("32-row stripes and workspace {}".format(vae_ops.fmt_bytes(r_ws)), "estimate {}".format(vae_ops.fmt_bytes(r_est)),
                "Not tried", "16-row stripes, workspace", "8-row stripes, workspace", "(2 retries)")
        check("OOM down to 32 rows, the 16- and 8-row plans above the first plan's estimate: the error names the plan that ran last "
              "(32 rows, its workspace and estimate) and lists the skipped ones apart (runs {}): {}".format(runs, text[text.find("still"):][:60]),
              isinstance(err, MonoloadVAEOOMError) and runs == [128, 64, 32] and all(x in text for x in want), "missing {}".format(
                  [x for x in want if x not in text]))
        runs, err, out, lines = run_case(v, lat, 10, set())
        check("OOM at every height: runs {}, then the OOM error (nothing skipped)".format(runs),
              isinstance(err, MonoloadVAEOOMError) and runs == [128, 64, 32, 16, 8] and not any("skipped" in l for l in lines))
    finally:
        mvae.set_stripe_rows(None)
    finish()


if __name__ == "__main__":
    main()
