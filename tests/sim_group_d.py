"""Group D (Mochi, Cosmos 1.0): simulated peaks of the streamed multi-frame layer 2 (monoload/vae_mochi.py,
monoload/vae_cosmos.py) against native and the whole-video layer 2, with the H1 estimate measured on the simulator
(tests/vae_inventory.py's tracer, CT 700 backends, bf16, 60 GiB cap; DESIGN §9.40).

Until the registration is in monoload/vae_video.py (an optional import, the adaptation thread's file), this script
registers the two chunk generators itself (--whole: leaves them out, the whole-video layer 2 as in §9.39).

    python tests/sim_group_d.py                         # both, the default sizes
    python tests/sim_group_d.py --only mochi --sizes 1280x720x85,848x480x163
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import vae_inventory as inv  # noqa: E402  (ComfyUI environment)

import group_d_reg  # noqa: E402
from monoload import vae_cosmos, vae_mochi  # noqa: E402

DEFAULT = {"mochi": ["848x480x85", "1280x720x85", "848x480x163"], "cosmos": ["1280x704x121", "1920x1088x121", "1280x704x57"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="mochi,cosmos")
    ap.add_argument("--sizes", default=None, help="WxHxF,... (default: per kind)")
    ap.add_argument("--whole", action="store_true", help="whole-video layer 2 (no streaming)")
    ap.add_argument("--no-native", action="store_true")
    a = ap.parse_args()
    group_d_reg.register(not a.whole)
    kinds = {k[0]: k for k in inv.KINDS}
    for name in a.only.split(","):
        v = inv.build(kinds[name])
        print("=" * 100)
        print(name, "streamed" if not a.whole else "whole-video layer 2", "| reason:", (vae_mochi if name == "mochi" else vae_cosmos).structure(v.first_stage_model)[0])
        sizes = a.sizes.split(",") if a.sizes else DEFAULT[name]
        for s in sizes:
            w, h, f = (int(x) for x in s.split("x"))
            shape = inv.latent_shape(v, w, h, f)
            print("-- {} latent {} path {}".format(s, list(shape), inv.path(v, shape)[0]))
            for layer, key in (((0, "native"),) if not a.no_native else ()) + ((2, "layer2"),):
                t0 = time.time()
                t = inv.trace(v, shape, layer)
                line = "   {:7s} reserved {:>7s} alloc {:>7s} GiB, largest block {:>6s}{}".format(
                    key, inv.gib(t["reserved"]), inv.gib(t["alloc"]), inv.gib(t["largest_block"]),
                    " OOM x{}".format(t["oom"]) if t["oom"] else "")
                if t.get("mf_estimate") is not None:
                    mp = t["mf_parts"]
                    line += " | estimate {} (slope {:.0f} B/latent px, intercept {}, frag {}, output {}){}".format(
                        inv.gib(t["mf_estimate"]), mp["slope"], inv.gib(mp["intercept"]), inv.gib(mp["frag"]), inv.gib(mp.get("output", 0)),
                        "" if t["mf_estimate"] >= t["reserved"] else "  ** NOT AN UPPER BOUND **")
                print(line + "  [{:.0f} s]".format(time.time() - t0))
                sys.stdout.flush()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
