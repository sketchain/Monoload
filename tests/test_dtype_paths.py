"""MonoloadRuntimePatch numerics over a dtype matrix, for the merge path
selected by MONOLOAD_EXACT.

MONOLOAD_EXACT=1: cast_bias_weight hands the weight function a private copy
of the parameter converted to the compute dtype; MonoloadRuntimePatch reuses
it when that is exact and otherwise re-reads the parameter. Every branch is
checked bit for bit against the native reference (patch_weight_to_device /
patch_hook_weight_to_device arithmetic), including lora_compute_dtype = fp16
as on gfx1151.

default: per patch, plain LoRA/LoCon is one fused addmm_ in the compute
dtype (C), anything else calculate_weight in the compute dtype (A, = native
lowvram LowVramPatch). Checked bit for bit against an independent
implementation of exactly that (addmm_ / LowVramPatch), and against native
merge within the tolerance of docs/DESIGN.md §5.5.

Both: quantized (fp8) parameters == the same path on the parameter
dequantized beforehand.

    python tests/test_dtype_paths.py                 # default path
    MONOLOAD_EXACT=1 python tests/test_dtype_paths.py
"""

import itertools

import torch

from common import EXACT, MODE_TAG, UNIT_ROUNDOFF, check, compare_dtype, error_bound, finish
import comfy.float
import comfy.lora
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
import comfy.weight_adapter
from monoload import hotpatch

DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
KEY = "lin.weight"


class Net(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.lin = comfy.ops.manual_cast.Linear(96, 64, dtype=dtype)


def make_patches(g):
    lora = {
        "lin.lora_up.weight": torch.randn(64, 8, generator=g).half(),
        "lin.lora_down.weight": torch.randn(8, 96, generator=g).half(),
        "lin.alpha": torch.tensor(4.0),
    }
    loaded = comfy.lora.load_lora(lora, {"lin": KEY}, log_missing=False)
    return loaded


def native_base(param, patches, ldt):
    temp = comfy.model_management.cast_to_device(param, param.device, ldt, copy=True)
    out = comfy.lora.calculate_weight(patches, temp, KEY)
    return comfy.float.stochastic_rounding(out, param.dtype, seed=comfy.utils.string_to_seed(KEY))


def native_hooks(current, param, base, hooks):
    temp = comfy.model_management.cast_to_device(current, current.device, torch.float32, copy=True)
    original = {KEY: [(param, lambda a, **k: a)] + list(base or [])}
    out = comfy.lora.calculate_weight(hooks, temp, KEY, original_weights=original)
    return comfy.float.stochastic_rounding(out, param.dtype, seed=comfy.utils.string_to_seed(KEY))


def main():
    g = torch.Generator().manual_seed(0)
    loaded = make_patches(g)
    orig_ldt = comfy.model_management.lora_compute_dtype
    print("merge path: {}".format(MODE_TAG))
    n = exact_cases(g, loaded, orig_ldt) if EXACT else default_cases(orig_ldt)
    n += quantized_cases(loaded, orig_ldt)
    finish({"mode": MODE_TAG, "cases": n})


def exact_cases(g, loaded, orig_ldt):
    n = 0
    for (pn, pdt), (cn, cdt), (ln, ldt), mode in itertools.product(DT.items(), DT.items(), [("fp32", torch.float32), ("fp16", torch.float16)], ("base", "hooks", "base+hooks")):
        net = Net(pdt)
        with torch.no_grad():
            net.lin.weight.copy_(torch.randn(64, 96, generator=g) * 0.05)
        param = net.lin.weight
        patcher = comfy.model_patcher.ModelPatcher(net, torch.device("cpu"), torch.device("cpu"))
        base = hook = None
        if "base" in mode:
            patcher.add_patches(loaded, 0.8, 1.0)
            base = patcher.patches[KEY]
        st = hotpatch._state(patcher)
        if "hooks" in mode:
            hook = [(0.6, loaded[KEY], 1.0, None, None)]
            st.hook_patches[KEY] = hook
        comfy.model_management.lora_compute_dtype = lambda device, _d=ldt: _d
        try:
            f = hotpatch.MonoloadRuntimePatch(KEY, patcher.patches, net.lin, "weight", st)
            weight_in = param.detach().clone().to(cdt)       # what cast_bias_weight passes in
            got = f(weight_in)
            ref = param.detach()
            if base:
                ref = native_base(ref, base, ldt)
            if hook:
                ref = native_hooks(ref, param.detach(), base, hook)
            ref = ref.to(cdt)
        finally:
            comfy.model_management.lora_compute_dtype = orig_ldt
        ok = got.dtype == ref.dtype and torch.equal(got.view(torch.uint8) if got.element_size() == 1 else got, ref)
        exact = cdt == pdt or (pdt, cdt) in hotpatch._EXACT_WIDENING
        check("param {} / compute {} / lora {} / {:10s} ({})".format(pn, cn, ln, mode, "reuse copy" if exact else "re-read param"),
              ok, "" if ok else "max_abs {}".format(float((got.float() - ref.float()).abs().max())))
        n += 1
    return n


# ---------------------------------------------------------------------------
# default path
# ---------------------------------------------------------------------------

def realistic_patches(g):
    """A plain LoRA whose change is a few % of the weight (like real LoRAs),
    a LoHa (not fusable) on the same key, and a full diff."""
    lora = comfy.lora.load_lora({
        "lin.lora_up.weight": (torch.randn(64, 8, generator=g) * 0.02).half(),
        "lin.lora_down.weight": (torch.randn(8, 96, generator=g) * 0.1).half(),
        "lin.alpha": torch.tensor(4.0),
    }, {"lin": KEY}, log_missing=False)
    loha = comfy.lora.load_lora({
        "lin.hada_w1_a": (torch.randn(64, 4, generator=g) * 0.05).half(),
        "lin.hada_w1_b": (torch.randn(4, 96, generator=g) * 0.05).half(),
        "lin.hada_w2_a": (torch.randn(64, 4, generator=g) * 0.05).half(),
        "lin.hada_w2_b": (torch.randn(4, 96, generator=g) * 0.05).half(),
        "lin.alpha": torch.tensor(4.0),
    }, {"lin": KEY}, log_missing=False)
    diff = {KEY: ("diff", ((torch.randn(64, 96, generator=g) * 0.002).half(),))}
    return lora, loha, diff


# variants: (name, base patch builder, hook patches builder, fused?)
def _variants(lora, loha, diff):
    def add(p, d, s, sm=1.0):
        p.add_patches(d, s, sm)
    hook = lambda: [(0.6, lora[KEY], 1.0, None, None)]
    return [
        ("lora", lambda p: add(p, lora, 0.8), None),
        ("lora+lora", lambda p: (add(p, lora, 0.8), add(p, lora, -0.3)), None),
        ("lora sm=0.9", lambda p: add(p, lora, 0.8, 0.9), None),            # strength_model: not fusable -> A
        ("loha", lambda p: add(p, loha, 0.8), None),
        ("lora+loha", lambda p: (add(p, lora, 0.8), add(p, loha, 0.7)), None),
        ("diff", lambda p: add(p, diff, 0.5), None),
        ("hooks", None, hook),
        ("lora+hooks", lambda p: add(p, lora, 0.8), hook),
    ]


def reference_default(patches, hooks, param, weight_in, cdt):
    """Independent implementation of the default path: per patch, plain
    LoRA -> addmm_ in the compute dtype, else native LowVramPatch arithmetic."""
    w = weight_in.clone()

    def apply(plist, original=None):
        nonlocal w
        for p in plist:
            strength, v, sm, offset, fn = p
            plain = (isinstance(v, comfy.weight_adapter.LoRAAdapter) and sm == 1.0 and offset is None and fn is None
                     and all(x is None for x in v.weights[3:6]))
            if plain:
                up, down, alpha = v.weights[0], v.weights[1], v.weights[2]
                w.view(w.shape[0], -1).addmm_(up.flatten(1).to(cdt), down.flatten(1).to(cdt), alpha=float(strength * alpha / down.shape[0]))
            else:
                w = comfy.model_patcher.LowVramPatch(KEY, {KEY: [p]})(w) if original is None else \
                    comfy.lora.calculate_weight([p], w, KEY, intermediate_dtype=w.dtype, original_weights=original)
    if patches:
        apply(patches)
    if hooks:
        apply(hooks, {KEY: [(param, lambda a, **k: a)] + list(patches or [])})
    return w


def default_cases(orig_ldt):
    import comfy.weight_adapter  # noqa: F401
    g = torch.Generator().manual_seed(2)
    lora, loha, diff = realistic_patches(g)
    n = 0
    worst = {}
    for (pn, pdt), (cn, cdt), (ln, ldt) in itertools.product(DT.items(), DT.items(), [("fp32", torch.float32), ("fp16", torch.float16)]):
        for vname, build, hook_fn in _variants(lora, loha, diff):
            net = Net(pdt)
            with torch.no_grad():
                net.lin.weight.copy_(torch.randn(64, 96, generator=g) * 0.05)
            param = net.lin.weight
            patcher = comfy.model_patcher.ModelPatcher(net, torch.device("cpu"), torch.device("cpu"))
            if build:
                build(patcher)
            base = patcher.patches.get(KEY)
            hook = hook_fn() if hook_fn else None
            st = hotpatch._state(patcher)
            if hook:
                st.hook_patches[KEY] = hook
            comfy.model_management.lora_compute_dtype = lambda device, _d=ldt: _d
            try:
                f = hotpatch.MonoloadRuntimePatch(KEY, patcher.patches, net.lin, "weight", st)
                weight_in = param.detach().clone().to(cdt)
                got = f(weight_in.clone())
                ref = reference_default(base, hook, param.detach(), weight_in, cdt)
                nat = param.detach()
                if base:
                    nat = native_base(nat, base, ldt)
                if hook:
                    nat = native_hooks(nat, param.detach(), base, hook)
            finally:
                comfy.model_management.lora_compute_dtype = orig_ldt
            ok = got.dtype == cdt and torch.equal(got, ref)
            cmp = compare_dtype(pdt, cdt, ldt) if base else compare_dtype(pdt, cdt)
            r = nat.to(cmp).double()
            change = float((r - param.detach().to(cmp).double()).norm())
            dw = float((got.to(cmp).double() - r).norm())
            rel = dw / change
            tol = error_bound(cmp, float(r.norm()), change) / change
            check("param {} / compute {} / lora {} / {:11s}: == fused/relaxed reference; vs native {:.2g} <= {:.2g} ({:.2g} u·||W||)".format(
                pn, cn, ln, vname, rel, tol, dw / (UNIT_ROUNDOFF[cmp] * float(r.norm()))), ok and rel <= tol,
                "" if ok else "max_abs vs reference {}".format(float((got.float() - ref.float()).abs().max())))
            k = "{}/{}".format(pn, cn)
            worst[k] = max(worst.get(k, 0.0), rel / tol)
            n += 1
    print("[INFO] worst (||Δw||/||ΔW_lora||) / bound per param/compute dtype: " + ", ".join("{} {:.2g}".format(k, v) for k, v in worst.items()))
    return n


class Holder(torch.nn.Module):
    pass


def quantized_cases(loaded, orig_ldt):
    """Quantized (fp8) parameter: relaxed merge on the dequantized temporary.
    Reference = the bit-exact path applied to the same parameter dequantized
    beforehand (a plain tensor in the dequantized dtype)."""
    from comfy.quant_ops import QuantizedTensor
    g = torch.Generator().manual_seed(1)
    n = 0
    for (dn, ddt), (cn, cdt), (ln, ldt), mode in itertools.product(DT.items(), DT.items(), [("fp32", torch.float32), ("fp16", torch.float16)], ("base", "hooks", "base+hooks")):
        w = (torch.randn(64, 96, generator=g) * 0.05).to(ddt)
        qt = QuantizedTensor.from_float(w, "TensorCoreFP8E4M3Layout")
        deq = qt.dequantize()
        results = []
        for kind in ("quantized", "dequantized"):
            holder = Holder()
            holder.__dict__["weight"] = qt if kind == "quantized" else deq
            patcher = comfy.model_patcher.ModelPatcher(Net(ddt), torch.device("cpu"), torch.device("cpu"))
            base = None
            if "base" in mode:
                patcher.add_patches(loaded, 0.8, 1.0)
                base = patcher.patches[KEY]
            st = hotpatch._state(patcher)
            if "hooks" in mode:
                st.hook_patches[KEY] = [(0.6, loaded[KEY], 1.0, None, None)]
            comfy.model_management.lora_compute_dtype = lambda device, _d=ldt: _d
            try:
                f = hotpatch.MonoloadRuntimePatch(KEY, patcher.patches, holder, "weight", st)
                if kind == "quantized":
                    weight_in = qt.clone().to(dtype=cdt).dequantize()   # what cast_bias_weight passes in
                else:
                    weight_in = deq.clone().to(cdt)
                results.append(f(weight_in))
            finally:
                comfy.model_management.lora_compute_dtype = orig_ldt
        got, ref = results
        ok = got.dtype == ref.dtype and torch.equal(got, ref)
        check("fp8 from {} / compute {} / lora {} / {:10s} == dequantize-first".format(dn, cn, ln, mode), ok,
              "" if ok else "max_abs {}".format(float((got.float() - ref.float()).abs().max())))
        n += 1
    return n


if __name__ == "__main__":
    main()
