"""MonoloadRuntimePatch vs native merge numerics over a dtype matrix.

cast_bias_weight hands the weight function a private copy of the parameter
converted to the compute dtype; MonoloadRuntimePatch reuses it when that is
exact and otherwise re-reads the parameter. This checks every branch against
the native reference (patch_weight_to_device / patch_hook_weight_to_device
arithmetic), including lora_compute_dtype = fp16 as on gfx1151.
"""

import itertools

import torch

from common import check, finish
import comfy.float
import comfy.lora
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
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
    n += quantized_cases(loaded, orig_ldt)
    finish({"cases": n})


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
