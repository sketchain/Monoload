"""The Monoload VAE Settings node (monoload/nodes/vae_settings.py,
monoload/vae_overrides.py, vae.resolve_settings).

No model files needed: small SDXL-like / Flux-like VAEs with random weights
(tests/test_vae_ldm.py), fp32 on the CPU.

  1. interface: the package registry (monoload/nodes) maps the node,
     category Monoload, display name, INPUT_TYPES / RETURN_TYPES / FUNCTION;
     the node is called the way ComfyUI calls it (an instance, FUNCTION,
     keyword inputs). Registration by ComfyUI's own loader is checked in
     tests/test_entry.py (every switch combination, MONOLOAD_DISABLE too);
  2. priority, item by item: node > environment (the global settings) >
     default, for budget, GroupNorm scheme, stripe height and mode, with the
     source of each in last_decode() and the log;
  3. copy: weights / patcher shared, the input VAE unchanged and still
     decoding with the global settings; one loaded model in ComfyUI's model
     management for both; chained nodes; encode;
  4. semantics as the environment variables: budget too small ->
     MonoloadError naming the needs; forced scheme / height; layer 2 only;
     native; the global settings restored after every decode (also after an
     error);
  5. several copies with different settings, interleaved;
  6. the wrapper not installed: the copy decodes natively (the global
     defaults MONOLOAD=0 / MONOLOAD_DISABLE_VAE / MONOLOAD_EXACT /
     MONOLOAD_DISABLE: tests/test_master_switch.py).

    python tests/test_vae_node.py
"""

import logging

import torch

from common import check, expect_raises, finish
import comfy.sd
from monoload import vae as mvae
from monoload import vae_overrides as vo
from monoload.errors import MonoloadError
from test_vae import init_random, managed_decode, native_decode
from test_vae_ldm import ldm_vae

NODE = "MonoloadVAESettings"


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def __enter__(self):
        logging.getLogger().addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger().removeHandler(self)


def registration():
    from monoload.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    cls = NODE_CLASS_MAPPINGS.get(NODE)
    check("node in the registry: {} ({})".format(NODE, NODE_DISPLAY_NAME_MAPPINGS.get(NODE)),
          cls is not None and NODE_DISPLAY_NAME_MAPPINGS.get(NODE) == "Monoload VAE Settings")
    it = cls.INPUT_TYPES()["required"]
    check("interface: category {}, inputs {}, returns {}, function {}".format(cls.CATEGORY, list(it), cls.RETURN_TYPES, cls.FUNCTION),
          cls.CATEGORY == "Monoload" and list(it) == ["vae", "budget", "budget_gib", "gn_scheme", "stripe_rows", "mode"] and cls.RETURN_TYPES == ("VAE",)
          and it["gn_scheme"][0] == ["default", "A", "B", "C", "D"] and it["mode"][0] == ["default", "auto", "layer 2 only", "native"]
          and it["budget"][0] == ["default", "unlimited", "custom"] and it["budget"][1]["default"] == "default"
          and it["budget_gib"][1]["default"] == 0.0 and it["budget_gib"][1]["step"] == 0.01 and it["budget_gib"][1]["round"] == 0.01
          and it["stripe_rows"][1]["default"] == 0)
    return cls


def node_apply(cls, vae, **kw):
    """Call the node the way ComfyUI does: an instance, its FUNCTION, keyword inputs (defaults for the rest)."""
    args = {k: v[1]["default"] for k, v in cls.INPUT_TYPES()["required"].items() if k != "vae"}
    args.update(kw)
    return getattr(cls(), cls.FUNCTION)(vae=vae, **args)[0]


def reset_globals():
    mvae.set_budget(None)
    mvae.set_gn_scheme(None)
    mvae.set_stripe_rows(None)
    mvae.set_stripe(True)


def snapshot():
    return dict(mvae._SETTINGS), mvae.vae_ldm.scheme()


def priority_tests(cls, sd):
    plain = node_apply(cls, sd)
    eff, src = mvae.resolve_settings(plain)
    check("node left at its defaults, no environment: every item from the default ({})".format(mvae.settings_note(eff, src)),
          src == {"budget": "default", "gn_scheme": "default", "stripe_rows": "default", "mode": "default"}
          and eff["budget"] is None and eff["gn_scheme"] == "B" and not eff["gn_forced"] and eff["stripe_rows"] is None and eff["mode"] == "auto")
    # the environment (global settings) for every item
    mvae.set_budget(8 << 30)
    mvae.set_gn_scheme("D")
    mvae.set_stripe_rows(32)
    mvae.set_stripe(False)
    try:
        eff, src = mvae.resolve_settings(plain)
        check("environment set, node at its defaults: every item from the environment ({})".format(mvae.settings_note(eff, src)),
              set(src.values()) == {"env"} and eff["budget"] == 8 << 30 and eff["gn_scheme"] == "D" and eff["gn_forced"]
              and eff["stripe_rows"] == 32 and eff["mode"] == "layer2")
        rows = []
        ok = True
        for item, kw, want in (("budget", {"budget": "custom", "budget_gib": 3.0}, ("budget", 3 << 30)), ("gn_scheme", {"gn_scheme": "A"}, ("gn_scheme", "A")),
                               ("stripe_rows", {"stripe_rows": 64}, ("stripe_rows", 64)), ("mode", {"mode": "auto"}, ("mode", "auto"))):
            eff, src = mvae.resolve_settings(node_apply(cls, sd, **kw))
            others = [k for k in src if k != item]
            good = src[item] == "node" and eff[want[0]] == want[1] and all(src[k] == "env" for k in others)
            ok = ok and good
            rows.append("{} -> {} (others env)".format(item, eff[want[0]]))
        check("item by item: the node's value for that item, the environment for the others: " + "; ".join(rows), ok)
    finally:
        reset_globals()


def copy_tests(cls, sd, lat):
    big = node_apply(cls, sd, budget="custom", budget_gib=1024.0)
    check("the copy shares the first-stage model and the patcher; the input VAE has no settings",
          big is not sd and big.first_stage_model is sd.first_stage_model and big.patcher is sd.patcher and vo.overrides(sd) == {}
          and vo.overrides(big) == {"budget": 1024 << 30})
    ref = native_decode(sd, lat, raw=True)
    out = managed_decode(big, lat, raw=True)
    last_big = mvae.last_decode()
    out2 = managed_decode(sd, lat, raw=True)
    last_sd = mvae.last_decode()
    check("copy with a 1024 GiB budget -> layer 2 (budget from the node); the input VAE right after -> layer 1, default scheme B "
          "(settings from the default); both == native (max|Δ| {:.2g} / {:.2g})".format(float((out - ref).abs().max()), float((out2 - ref).abs().max())),
          last_big.get("strategy") == "layer2" and last_big["settings_source"]["budget"] == "node"
          and last_sd.get("strategy") == "layer1" and last_sd.get("gn_scheme") == "B" and last_sd["settings_source"]["budget"] == "default"
          and float((out - ref).abs().max()) <= 1e-4 and float((out2 - ref).abs().max()) <= 1e-5)
    import comfy.model_management as mm
    loaded = [lm for lm in mm.current_loaded_models if lm.model is sd.patcher]
    check("ComfyUI's model management: decodes of the copy and of the input loaded one model ({} entry for the shared patcher of {})".format(
          len(loaded), len(mm.current_loaded_models)), len(loaded) == 1)
    chained = node_apply(cls, big, gn_scheme="C", stripe_rows=16)
    check("chained nodes: the downstream node's items override, the ones it leaves unset keep the upstream node's ({})".format(vo.overrides(chained)),
          vo.overrides(chained) == {"budget": 1024 << 30, "gn_scheme": "C", "stripe_rows": 16} and vo.overrides(big) == {"budget": 1024 << 30})
    fsm = sd.first_stage_model
    with torch.no_grad():   # the test VAE was built from decoder weights only
        init_random(fsm.encoder)
        if getattr(fsm, "quant_conv", None) is not None:
            init_random(fsm.quant_conv)
    px = torch.rand(1, 64, 48, 3)
    torch.manual_seed(1)
    e1 = big.encode(px)
    torch.manual_seed(1)
    e2 = sd.encode(px)
    check("encode through the copy == through the input VAE (same seed: the encoder samples the posterior)",
          torch.equal(e1, e2) and bool(torch.isfinite(e1).all()))


def semantics_tests(cls, sd, lat):
    ref = native_decode(sd, lat, raw=True)
    before = snapshot()
    tiny = node_apply(cls, sd, budget="custom", budget_gib=0.001)
    expect_raises("node budget too small -> MonoloadError naming what each candidate needs, the budget's source (the node) and what to "
                  "change on the node", MonoloadError, lambda: managed_decode(tiny, lat),
                  "(from the Monoload VAE Settings node", "layer 2 needs about", "scheme B", "Raise the budget on the Monoload VAE Settings node")
    try:
        managed_decode(tiny, lat)
    except MonoloadError as e:
        check("... and does not point at MONOLOAD_VAE_BUDGET", "MONOLOAD_VAE_BUDGET" not in str(e))
    mvae.set_budget(1 << 20)
    try:
        expect_raises("environment budget too small -> the source is the environment variable, the advice too", MonoloadError,
                      lambda: managed_decode(sd, lat), "(from environment variable MONOLOAD_VAE_BUDGET", "Raise MONOLOAD_VAE_BUDGET")
    finally:
        mvae.set_budget(None)
    # review 2026-10 item 10: an estimate just over the budget (2.1704 GiB against 2.17) must not read like the budget
    near = node_apply(cls, sd, budget="custom", budget_gib=2.17)
    orig_l2, orig_sel = mvae._layer2_estimate, mvae._select_layer1

    def l2(*a, **kw):
        est, probe = orig_l2(*a, **kw)
        return dict(est, total=int(2.1704 * (1 << 30))), probe
    mvae._layer2_estimate, mvae._select_layer1 = l2, lambda *a, **kw: (None, "not recognized (test)")
    try:
        try:
            managed_decode(near, lat)
            text = ""
        except MonoloadError as e:
            text = str(e)
    finally:
        mvae._layer2_estimate, mvae._select_layer1 = orig_l2, orig_sel
    check("budget 2.17 GiB, layer 2 needs 2.1704 GiB: the error shows them apart ({})".format(text[:150]),
          "budget 2.1700 GiB" in text and "layer 2 needs about 2.1704 GiB" in text)
    with LogCapture() as cap:
        managed_decode(node_apply(cls, sd, budget="custom", budget_gib=1024.0, stripe_rows=16), lat)
    line = next((x for x in cap.lines if "budget 1024.00 GiB (from" in x), "")
    check("the decode's budget line names the node, a forced height too: {}".format(line[:150]),
          "budget 1024.00 GiB (from the Monoload VAE Settings node)" in line and "16 rows (from the Monoload VAE Settings node)" in line)
    check("... the global settings are restored after the error", snapshot() == before and mvae.last_decode()["settings_source"]["budget"] == "node")
    forced = node_apply(cls, sd, gn_scheme="D", stripe_rows=24)
    out = managed_decode(forced, lat, raw=True)
    last = mvae.last_decode()
    check("node scheme D + 24-row stripes: layer 1 exactly so (as MONOLOAD_VAE_GN_SCHEME / MONOLOAD_VAE_STRIPE_ROWS), == native (max|Δ| {:.2g})".format(
          float((out - ref).abs().max())), last.get("gn_scheme") == "D" and last.get("rows") == 24 and float((out - ref).abs().max()) <= 1e-5
          and snapshot() == before)
    l2 = node_apply(cls, sd, mode="layer 2 only")
    managed_decode(l2, lat)
    check("node mode layer 2 only -> layer 2", mvae.last_decode().get("strategy") == "layer2" and snapshot() == before)
    nat = node_apply(cls, sd, mode="native")
    with LogCapture() as cap:
        out = managed_decode(nat, lat, raw=True)
    last = mvae.last_decode()
    check("node mode native -> ComfyUI's own decode (logged: {})".format(next((l for l in cap.lines if "left native" in l), "")[:120]),
          last.get("strategy") == "native" and "Monoload VAE Settings" in last.get("reason", "") and torch.equal(out, ref))
    mvae.set_stripe(False)
    try:
        auto = node_apply(cls, sd, mode="auto")
        managed_decode(auto, lat)
        check("MONOLOAD_DISABLE_VAE_STRIPE=1 globally, node mode auto -> layer 1 for this VAE (node > environment)",
              mvae.last_decode().get("strategy") == "layer1")
    finally:
        reset_globals()
    expect_raises("unknown scheme -> ValueError", ValueError, lambda: vo.with_settings(sd, gn_scheme="E"))
    expect_raises("unknown mode -> ValueError", ValueError, lambda: vo.with_settings(sd, mode="fast"))
    with LogCapture() as cap:
        managed_decode(node_apply(cls, sd, budget="custom", budget_gib=1024.0), lat)
    line = next((l for l in cap.lines if "-> layer 2" in l and "settings:" in l), "")
    check("the decode's log line names where each setting came from: ...{}".format(line[line.find("settings:"):][:140]),
          "budget 1024.00 GiB (node)" in line and "mode auto (default)" in line)


def several_tests(cls, sd, lat):
    a = node_apply(cls, sd, gn_scheme="A", stripe_rows=16)
    b = node_apply(cls, sd, mode="layer 2 only")
    c = node_apply(cls, sd, gn_scheme="C")
    got = []
    for v in (a, b, c, a, sd, b):
        managed_decode(v, lat)
        m = mvae.last_decode()
        got.append("{}{}".format(m.get("strategy"), "/" + str(m.get("gn_scheme")) if m.get("strategy") == "layer1" else ""))
    check("several copies decoded interleaved keep their own settings: " + ", ".join(got),
          got == ["layer1/A", "layer2", "layer1/C", "layer1/A", "layer1/B", "layer2"])


def global_switch_tests(cls, sd, lat):
    """The wrapper not installed (e.g. a ComfyUI whose API differs): copies
    decode natively, their settings unused, no budget error. The global
    defaults (MONOLOAD=0, MONOLOAD_DISABLE_VAE, MONOLOAD_EXACT, MONOLOAD_DISABLE)
    are covered by tests/test_master_switch.py."""
    ref = native_decode(sd, lat)
    copy_ = node_apply(cls, sd, budget="custom", budget_gib=0.001)
    mvae.uninstall()
    try:
        again = node_apply(cls, sd, budget="custom", budget_gib=0.001)
        out = comfy.sd.VAE.decode(copy_, lat)
        out2 = comfy.sd.VAE.decode(again, lat)
        check("wrapper not installed: copies decode natively, no budget error", torch.equal(out, ref) and torch.equal(out, out2))
    finally:
        mvae.install()


def main():
    if not mvae.is_installed():
        check("vae.install() on this ComfyUI", mvae.install())
    reset_globals()
    cls = registration()
    g = torch.Generator().manual_seed(7)
    sd = ldm_vae(4, True)
    lat = torch.randn(1, 4, 12, 10, generator=g)
    priority_tests(cls, sd)
    copy_tests(cls, sd, lat)
    semantics_tests(cls, sd, lat)
    several_tests(cls, sd, lat)
    global_switch_tests(cls, sd, lat)
    finish()


if __name__ == "__main__":
    main()
