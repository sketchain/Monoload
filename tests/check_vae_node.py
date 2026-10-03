"""CT 700 check of the Monoload VAE Settings node with a real VAE.

Loads the VAE (from a checkpoint or models/vae), makes a copy with the node
(peak budget, default 3 GiB), decodes a random latent with the copy and then
with the original VAE, and prints for each: what Monoload chose (layer,
scheme, stripes, workspace), where each setting came from, the estimate, the
GTT / reserved increase and the time. The copy should follow the node's
budget; the original VAE should keep the default (scheme B for SDXL).

    docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/check_vae_node.py \\
        --checkpoint waiIllustriousSDXL_v170.safetensors --res 3840x2160 --budget 3

Same launch arguments as tests/bench_vae.py ($COMFY_ARGS, else those of the
container's main process).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_vae as B  # noqa: E402  (ComfyUI environment, measuring helpers)
from bench_vae import gib, mvae, torch  # noqa: E402
from monoload.nodes import NODE_CLASS_MAPPINGS  # noqa: E402


def describe(m):
    if m.get("strategy") == "layer1":
        what = "layer 1, {}, {} stripes of {} rows, workspace {}".format(m.get("adapter"), m.get("stripes"), m.get("rows"),
                                                                       B.vae_ops.fmt_bytes(m.get("workspace")))
    elif m.get("strategy") == "layer2":
        what = "layer 2 (op-level chunking), workspace {}".format(B.vae_ops.fmt_bytes(m.get("workspace")))
    else:
        what = "{} ({})".format(m.get("strategy"), m.get("reason", ""))
    src = m.get("settings_source") or {}
    eff = m.get("settings") or {}
    shown = dict(eff, budget=B.vae_ops.fmt_bytes(eff["budget"]) if eff.get("budget") else None)
    sett = "settings: " + ", ".join("{} {} [{}]".format(k, shown.get(k), src.get(k)) for k in ("budget", "gn_scheme", "stripe_rows", "mode"))
    est = (m.get("estimate") or {}).get("total")
    return what, sett, est


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="checkpoint with a built-in VAE (models/checkpoints)")
    src.add_argument("--vae", help="VAE file (models/vae)")
    p.add_argument("--res", default="3840x2160", help="output resolution WxH (random latent)")
    p.add_argument("--budget", type=float, default=3.0, help="the node's peak budget in GiB")
    p.add_argument("--gn-scheme", default="default", help="the node's GroupNorm scheme (default / A / B / C / D)")
    p.add_argument("--stripe-rows", type=int, default=0, help="the node's stripe height (0 = auto)")
    p.add_argument("--mode", default="default", help="the node's mode (default / auto / layer 2 only / native)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    mvae.install()
    B.configure("monoload")          # the plugin's settings from the environment
    vae, what = B.load_vae(a)
    w, h = (int(x) for x in a.res.split("x"))
    lat = B.random_latent(vae, w, h, a.seed, 1.0)
    node = NODE_CLASS_MAPPINGS["MonoloadVAESettings"]
    copy = getattr(node(), node.FUNCTION)(vae=vae, budget_gib=a.budget, gn_scheme=a.gn_scheme, stripe_rows=a.stripe_rows, mode=a.mode)[0]
    print("VAE: {} ({}), latent {}; node: budget {} GiB, scheme {}, stripe rows {}, mode {}".format(
        what, type(vae.first_stage_model).__name__, list(lat.shape), a.budget, a.gn_scheme, a.stripe_rows, a.mode), flush=True)
    print("copy shares the weights: {}; the original VAE carries no settings: {}".format(
        copy.first_stage_model is vae.first_stage_model and copy.patcher is vae.patcher, not getattr(vae, "_monoload_vae_settings", None)))
    rows = []
    for label, v in (("node copy", copy), ("original", vae), ("node copy again", copy)):
        torch.cuda.empty_cache() if B.is_cuda() else None
        row, _, _ = B.decode_once(v, lat, False, 0.02)
        m = mvae.last_decode()
        what_, sett, est = describe(m)
        rows.append((label, row, what_, sett, est))
        print("{:16s} {:10s} {:7.2f} s | GTT +{} reserved +{} estimate {} GiB | {} | {}".format(
            label, row["status"], row["seconds"], gib(row.get("gtt_delta")).strip(), gib(row.get("res_delta")).strip(), gib(est).strip(), what_, sett),
            flush=True)
    print("\nexpected: the copy follows the node (budget from the node, the fastest configuration whose estimate fits it), "
          "GTT <= estimate <= budget; the original keeps the default (SDXL / Flux: layer 1, GroupNorm scheme B, settings from the default).")


if __name__ == "__main__":
    main()
