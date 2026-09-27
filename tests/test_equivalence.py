"""Correctness: native UNETLoader vs Monoload.

    python tests/test_equivalence.py --family sd15 --source X.safetensors --converted Y.safetensors
"""

import argparse

from common import *  # noqa: F401,F403
from common import check, compare_state, config_snapshot_eq, diff_stats, family_inputs, finish, free_all, load_monoload, load_native, sample  # noqa: E501
from monoload import rebuild
from monoload.loader import load_monoload_diffusion_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--converted", required=True)
    p.add_argument("--steps", type=int, default=4)
    a = p.parse_args()

    native = load_native(a.source)
    mono = load_monoload(a.converted)

    n, problems = compare_state(native.model, mono.model)
    check("all {} params+buffers byte-identical".format(n), not problems, "; ".join(problems[:5]))

    snap_n = rebuild.config_snapshot(native.model.model_config)
    snap_m = rebuild.config_snapshot(mono.model.model_config)
    ok, diffs = config_snapshot_eq(snap_n, snap_m)
    check("model_config key attributes equal", ok, "; ".join(diffs) if diffs else "class={} dtype={} manual_cast={} sampling={}".format(
        snap_n["class"], snap_n["unet_config"].get("dtype"), snap_n["manual_cast_dtype"], snap_n["sampling_settings"]))
    fp_n = rebuild.model_fingerprint(native.model)
    fp_m = rebuild.model_fingerprint(mono.model)
    check("model fingerprint equal", fp_n == fp_m, "native={} mono={}".format(fp_n, fp_m) if fp_n != fp_m else "model_type={}".format(fp_n["model_type"].name))
    check("load/offload devices equal", (native.load_device, native.offload_device, native.model.device) == (mono.load_device, mono.offload_device, mono.model.device),
          "{} / {}".format((native.load_device, native.offload_device, native.model.device), (mono.load_device, mono.offload_device, mono.model.device)))
    check("cached_patcher_init -> monoload loader", mono.cached_patcher_init[0] is load_monoload_diffusion_model)

    pos, neg, latent = family_inputs(a.family)
    out_n = sample(native, pos, neg, latent, steps=a.steps)
    free_all()
    out_m = sample(mono, pos, neg, latent, steps=a.steps)
    free_all()
    d = diff_stats(out_n, out_m)
    check("sampling output identical (seed 1234, {} steps)".format(a.steps), d["bit_exact"], str(d))

    re = mono.cached_patcher_init[0](*mono.cached_patcher_init[1])
    n2, problems = compare_state(mono.model, re.model)
    check("rebuild via cached_patcher_init identical", not problems, "; ".join(problems[:5]))
    finish()


if __name__ == "__main__":
    main()
