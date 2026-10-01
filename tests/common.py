"""Shared helpers for Monoload tests. Run inside the locked ComfyUI image
(tests/docker_run.sh); every test script is a plain `python tests/xxx.py`.

ComfyUI launch args come from $COMFY_ARGS (default: --cpu --fp16-unet).
"""

import hashlib
import json
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from monoload import comfy_env  # noqa: E402

COMFY_ARGS = shlex.split(os.environ.get("COMFY_ARGS", "--cpu --fp16-unet"))
_TEST_ARGV = list(sys.argv)
ROOT, ARGS = comfy_env.setup(COMFY_ARGS)  # ComfyUI parses sys.argv here
sys.argv = _TEST_ARGV

import torch  # noqa: E402
import folder_paths  # noqa: E402
import nodes  # noqa: E402
import comfy.model_management  # noqa: E402

from monoload import hotpatch  # noqa: E402
from monoload.errors import MonoloadError, MonoloadUnsupportedError  # noqa: E402,F401

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("[{}] {}{}".format("PASS" if ok else "FAIL", name, (" — " + detail) if detail else ""), flush=True)
    return ok


def finish(extra=None):
    failed = [r for r in RESULTS if not r[1]]
    print("\n== {} checks, {} failed".format(len(RESULTS), len(failed)))
    if extra is not None:
        print("RESULT_JSON " + json.dumps(extra, default=str))
    sys.exit(1 if failed else 0)


def expect_raises(name, exc_type, fn, *substrings):
    try:
        fn()
    except exc_type as e:
        msg = str(e)
        missing = [s for s in substrings if s not in msg]
        first = msg.strip().splitlines()[0] if msg.strip() else ""
        return check(name, not missing, "missing {} in: {}".format(missing, msg[:500]) if missing else first[:220])
    except Exception as e:
        return check(name, False, "wrong exception {}: {}".format(type(e).__name__, str(e)[:500]))
    return check(name, False, "no exception raised")


# ---------------------------------------------------------------------------
# mode switching / loading / comparing
# ---------------------------------------------------------------------------

def free_all():
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    import gc
    gc.collect()


def set_runtime(enabled):
    """Switch between native ComfyUI and Monoload (all models unloaded first)."""
    free_all()
    if enabled:
        hotpatch.install()
    else:
        hotpatch.uninstall()
    assert hotpatch.is_installed() == enabled


def load_checkpoint(name):
    model, clip, _vae = nodes.CheckpointLoaderSimple().load_checkpoint(name)
    return model, clip


def load_unet(name):
    return nodes.UNETLoader().load_unet(name, "default")[0]


def load_clip(name, type_="stable_diffusion"):
    return nodes.CLIPLoader().load_clip(name, type_)[0]


def apply_loras(model, clip, loras):
    """loras: list of (name, strength_model, strength_clip)."""
    for name, sm, sc in loras:
        model, clip = nodes.LoraLoader().load_lora(model, clip, name, sm, sc)
    return model, clip


def encode(clip, text):
    return nodes.CLIPTextEncode().encode(clip, text)[0]


def sample(model, positive, negative, latent, seed=1234, steps=2, cfg=5.0, sampler="euler", scheduler="normal"):
    out = nodes.common_ksampler(model, seed, steps, cfg, sampler, scheduler, positive, negative, {"samples": latent.clone()}, denoise=1.0)
    return out[0]["samples"]


def byte_view(t):
    return t.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)


def weight_digests(module):
    """Per-tensor blake2b of every param/buffer (bytes), without keeping a copy."""
    out = {}
    for k, v in module.state_dict().items():
        out[k] = (v.dtype, tuple(v.shape), hashlib.blake2b(byte_view(v).numpy().tobytes(), digest_size=16).hexdigest())
    return out


def digests_diff(a, b):
    return [k for k in set(a) | set(b) if a.get(k) != b.get(k)]


def cond_equal(c1, c2):
    """Conditioning lists from CLIPTextEncode: tensors and pooled outputs bit-equal."""
    if len(c1) != len(c2):
        return False
    for (t1, d1), (t2, d2) in zip(c1, c2):
        if not torch.equal(byte_view(t1), byte_view(t2)):
            return False
        p1, p2 = d1.get("pooled_output"), d2.get("pooled_output")
        if (p1 is None) != (p2 is None) or (p1 is not None and not torch.equal(byte_view(p1), byte_view(p2))):
            return False
    return True


def diff_stats(a, b):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    return {"bit_exact": bool(torch.equal(a, b)), "max_abs": float(d.max()), "mean_abs": float(d.mean())}


# ---------------------------------------------------------------------------
# merge paths: MONOLOAD_EXACT=1 (bit-exact) vs default (fused / relaxed)
# ---------------------------------------------------------------------------

EXACT = hotpatch.is_exact()  # from MONOLOAD_EXACT, read when hotpatch was imported
MODE_TAG = "exact" if EXACT else "default"

# Default-path tolerance against native ComfyUI's merge (docs/DESIGN.md §5.5).
# Metric (as in the bench layer probe):  rel = ||Δw|| / ||ΔW_lora||
#   Δw       = w_default - w_native
#   ΔW_lora  = w_native - w_base       (what the LoRA changes)
# Bound:  ||Δw|| <= u * (ROUND_TERM * ||W|| + DELTA_TERM * ||ΔW_lora||), i.e.
#   rel <= u * (ROUND_TERM * ||W|| / ||ΔW_lora|| + DELTA_TERM)
# u = unit roundoff of the dtype the two are compared in (fp16 2^-11, bf16
# 2^-8, fp32 2^-24), W = w_native. The first term is the final rounding of
# W + ΔW (a merge that rounds a slightly different sum lands one ulp away; at
# most ~1/4 of the elements may), the second the error of computing ΔW in the
# compute dtype. A fixed bound on rel alone cannot work: its floor is the
# rounding of W, so it grows as ΔW gets small relative to W.
ROUND_TERM = 1.0
DELTA_TERM = 20.0
UNIT_ROUNDOFF = {torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8, torch.float32: 2.0 ** -24, torch.float64: 2.0 ** -53}


def compare_dtype(*dtypes):
    """The coarsest of the given dtypes (largest unit roundoff). Native merges
    in lora_compute_dtype and rounds the result to the parameter dtype; when
    those are coarser than the compute dtype (e.g. fp16 weights, fp32 compute
    on CPU; or lora_compute_dtype fp16 with fp32 weights) the default path is
    more precise than native there, which is not an error. So both are
    compared at the coarsest precision involved: param, compute and (for
    base patches) lora_compute_dtype."""
    return max(dtypes, key=lambda d: UNIT_ROUNDOFF[d])


def error_bound(dtype, w_norm, change_norm):
    """Bound on ||Δw|| (see above)."""
    return UNIT_ROUNDOFF[dtype] * (ROUND_TERM * w_norm + DELTA_TERM * change_norm)


class ComputeDtypes:
    """Records, per key, the dtype of the temporary cast_bias_weight hands to
    Monoload's weight function (= the compute dtype the merge runs in)."""

    def __enter__(self):
        self.dtypes = {}
        cls = hotpatch.MonoloadRuntimePatch
        self._orig = cls.__call__
        rec = self.dtypes

        def call(f, weight, _orig=self._orig):
            rec[f.key] = weight.dtype
            return _orig(f, weight)
        cls.__call__ = call
        return self.dtypes

    def __exit__(self, *exc):
        hotpatch.MonoloadRuntimePatch.__call__ = self._orig
        return False


def merge_error(patcher, dtypes):
    """Weight-level comparison of the default path with native ComfyUI's merge
    (ModelPatcher.patch_weight_to_device(return_weight=True), i.e. what native
    bakes), over every key of `patcher` that has a Monoload weight function
    and a recorded compute dtype. Also recomputes the MONOLOAD_EXACT path and
    checks it is bit-identical to native.
    Returns dict(keys, rel, tol, max_abs, exact_equal)."""
    import comfy.utils
    from comfy.quant_ops import QuantizedTensor
    comfy.model_management.load_models_gpu([patcher])
    native = hotpatch._ORIG["patch_weight_to_device"]
    prev = hotpatch.is_exact()
    num = den = bound_sq = ulp_num = 0.0
    max_abs = 0.0
    n = 0
    exact_equal = True
    try:
        for key in patcher.patches:
            if key not in dtypes:
                continue
            mod_name, attr = key.rsplit(".", 1)
            mod = comfy.utils.get_attr(patcher.model, mod_name)
            f = next((x for x in mod.__dict__.get(attr + "_function", []) if getattr(x, "is_monoload_patch", False) and x.key == key), None)
            param = getattr(mod, attr)
            if f is None or isinstance(param, QuantizedTensor):
                continue
            cdt = dtypes[key]
            pdt = param.dtype
            base = param.detach().to(cdt, copy=True)          # what cast_bias_weight hands over
            ref = native(patcher, key, device_to=param.device, return_weight=True)
            hotpatch.set_exact(True)
            ex = f(base.clone())
            hotpatch.set_exact(False)
            fast = f(base.clone())
            exact_equal = exact_equal and torch.equal(ex, ref.to(cdt))
            cmp = compare_dtype(pdt, cdt, comfy.model_management.lora_compute_dtype(param.device))
            r = ref.to(cmp).double()
            d = fast.to(cmp).double() - r
            d_sq = float((d * d).sum())
            ch_sq = float(((r - base.to(cmp).double()) ** 2).sum())
            w_norm = float(r.norm())
            num += d_sq
            den += ch_sq
            bound_sq += error_bound(cmp, w_norm, ch_sq ** 0.5) ** 2
            ulp_num += (UNIT_ROUNDOFF[cmp] * w_norm) ** 2
            max_abs = max(max_abs, float(d.abs().max()))
            n += 1
    finally:
        hotpatch.set_exact(prev)
    rel = (num / den) ** 0.5 if den > 0 else float("nan")
    tol = (bound_sq / den) ** 0.5 if den > 0 else 0.0
    ulps = (num / ulp_num) ** 0.5 if ulp_num > 0 else float("nan")
    return {"keys": n, "rel": rel, "tol": tol, "ulps": ulps, "max_abs": max_abs, "exact_equal": exact_equal}


def check_merge_error(name, patcher, dtypes, expect_keys):
    e = merge_error(patcher, dtypes)
    check("{}: weights vs native merge ||Δw||/||ΔW_lora|| = {:.3g} <= {:.3g} ({} of {} keys, max|Δw| {:.3g}, ||Δw|| = {:.2g} u·||W||)".format(
        name, e["rel"], e["tol"], e["keys"], expect_keys, e["max_abs"], e["ulps"]),
        e["keys"] == expect_keys and e["keys"] > 0 and e["rel"] <= e["tol"])
    check("{}: MONOLOAD_EXACT path recomputed on the same keys == native merge".format(name), e["exact_equal"])
    return e
