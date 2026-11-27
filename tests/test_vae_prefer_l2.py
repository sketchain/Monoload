"""Layer 2 first for an adapter whose layer 1 is the fallback (prefer_layer2, CogVideoX), and the multi-frame
estimate's fragmentation ratio (DESIGN §9.35). No model files, no GPU: the decode paths are replaced by recorders.
    python tests/test_vae_prefer_l2.py
"""

import types

from common import check, finish
import torch

from monoload import vae as mvae

GIB = 1024 ** 3


def fake_vae():
    v = types.SimpleNamespace(first_stage_model=torch.nn.Module(), latent_dim=3, device=torch.device("cpu"),
                              output_device=torch.device("cpu"), vae_dtype=torch.bfloat16,
                              throw_exception_if_invalid=lambda: None, vae_output_dtype=lambda: torch.float32)
    v.patcher = types.SimpleNamespace(model_size=lambda: 1 * GIB)
    return v


def route(prefer, est_total, total, cap=None, unbounded=False):
    """Which path _managed_decode takes; returns (path, self-test ran)."""
    calls = {"path": None, "selftest": False}
    bound = types.SimpleNamespace(name="Fake", prefer_layer2=prefer)
    saved = {k: getattr(mvae, k) for k in ("_select_layer1", "_layer2_estimate", "_decode_layer2", "_decode_layer1",
                                           "_layer1_self_test", "budget")}
    saved_total, saved_res = mvae.mm.get_total_memory, mvae.mm.extra_reserved_memory

    def est(vae, s, o, ws):
        if unbounded:
            raise mvae._UnboundedDecode("probe failed")
        return {"total": est_total}, None

    def l2(self, s, o, t0, note, est=None, probe=None, **kw):
        calls["path"] = "layer2"
        return "l2"

    def l1(self, s, b, t0, detail, **kw):
        calls["path"] = "layer1"
        return "l1"

    def st(vae, b):
        calls["selftest"] = True
        return True, "ok"

    try:
        mvae._select_layer1 = lambda vae, s, o: (bound, None)
        mvae._layer2_estimate = est
        mvae._decode_layer2 = l2
        mvae._decode_layer1 = l1
        mvae._layer1_self_test = st
        mvae.budget = lambda: None
        mvae.mm.get_total_memory = lambda dev=None, torch_total_too=False: total
        mvae.mm.extra_reserved_memory = lambda: 0
        import os
        old = os.environ.pop("MONOLOAD_VAE_FIT_CAP", None)
        if cap:
            os.environ["MONOLOAD_VAE_FIT_CAP"] = cap
        try:
            mvae._managed_decode(fake_vae(), torch.zeros(1, 16, 5, 8, 8), {})
        finally:
            os.environ.pop("MONOLOAD_VAE_FIT_CAP", None)
            if old is not None:
                os.environ["MONOLOAD_VAE_FIT_CAP"] = old
    finally:
        for k, f in saved.items():
            setattr(mvae, k, f)
        mvae.mm.get_total_memory, mvae.mm.extra_reserved_memory = saved_total, saved_res
    return calls["path"], calls["selftest"]


def main():
    p, st = route(True, 26 * GIB, 62 * GIB)
    check("prefer_layer2, estimate 26 GiB on a 62 GiB device (1 GiB weights): layer 2, no layer-1 self-test", p == "layer2" and not st, (p, st))
    p, st = route(True, 58 * GIB, 62 * GIB, cap="55G")
    check("prefer_layer2, 58 GiB with MONOLOAD_VAE_FIT_CAP=55G: layer 1, its self-test runs", p == "layer1" and st, (p, st))
    p, st = route(True, 58 * GIB, 62 * GIB)
    check("prefer_layer2, 58 GiB on 62 GiB without a cap (61 GiB room): layer 2", p == "layer2", (p, st))
    p, st = route(True, 1, 62 * GIB, unbounded=True)
    check("prefer_layer2, layer-2 probe failed: layer 1 (not native)", p == "layer1" and st, (p, st))
    p, st = route(False, 1, 62 * GIB)
    check("no prefer_layer2: layer 1 as before", p == "layer1" and st, (p, st))

    # the fragmentation ratio scales the allocated bound; the absolute fragmentation is a floor
    v = fake_vae()
    lat = torch.zeros(1, 16, 5, 100, 100)
    pr = mvae._Probe(1000.0, 0, 0.0, mf_slope=1000.0, mf_intercept=0, mf_frag=0, mf_ratio=1.4)
    e = mvae.estimate(v, lat, 0, pr)
    alloc = mvae.MF_MARGIN * 1000.0 * 100 * 100
    check("ratio 1.4: activation = 1.4 x the allocated bound", abs(e["activation"] - int(alloc * 1.4)) <= 1, (e["activation"], alloc))
    pr = mvae._Probe(1000.0, 0, 0.0, mf_slope=1000.0, mf_intercept=0, mf_frag=10 ** 9, mf_ratio=1.01)
    e = mvae.estimate(v, lat, 0, pr)
    check("small ratio, large absolute fragmentation: allocated + fragmentation", e["activation"] == int(alloc + 10 ** 9), e["activation"])
    finish()


if __name__ == "__main__":
    main()
