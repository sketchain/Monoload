"""Quantized safetensors formats + LoRA, end to end through ComfyUI's own
loader path, CPU, no model files.

For each way a quantized checkpoint marks its layers --
  * comfy_quant tensors in the file (e.g. an int8_tensorwise model),
  * header metadata `_quantization_metadata` only (e.g. flux2_dev_fp8mixed),
  * the legacy `scaled_fp8` marker key + `.scale_weight` (e.g. umt5 / t5 fp8 scaled),
  * both metadata and marker (e.g. mistral_3_small_flux2_fp8)
-- and each QuantizedTensor format (fp8 e4m3 / e5m2, int8_tensorwise, nvfp4, mxfp8,
convrot_w4a4, asym_w4a8_int8, w6a8_int8), a two-layer mixed_precision_ops model is built from the state dict
after comfy.utils.convert_old_quants (as comfy.sd does), one layer quantized
and one plain (a mixed model). With a LoRA on both layers:

  * the quantized layer really loaded as a QuantizedTensor of that layout;
  * Monoload output == the same model whose quantized weight was dequantized
    beforehand + the same LoRA through Monoload (bit for bit, both merge paths);
  * no weight backups, the quantized bytes (qdata + scales) never change, the
    layer is still a QuantizedTensor afterwards, nothing left in the weight
    functions after unpatch;
  * the difference to native ComfyUI (merge + re-quantize + backup) is
    reported, not judged.

    python tests/test_quant_formats.py
"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import check, finish  # noqa: E402
import torch  # noqa: E402

import comfy.lora  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor  # noqa: E402
from monoload import hotpatch  # noqa: E402

CPU = torch.device("cpu")
IN, OUT, RANK = 256, 64, 8
COMPUTE = torch.bfloat16


def comfy_quant(conf):
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def base_weights():
    g = torch.Generator().manual_seed(0)
    return {
        "a.weight": (torch.randn(OUT, IN, generator=g) * 0.05).to(COMPUTE),
        "a.bias": (torch.randn(OUT, generator=g) * 0.01).to(COMPUTE),
        "b.weight": (torch.randn(OUT, IN, generator=g) * 0.05).to(COMPUTE),
        "b.bias": (torch.randn(OUT, generator=g) * 0.01).to(COMPUTE),
    }


def file_state_dict(source, algo):
    """(state_dict, metadata) as a file of this kind would load: layer `a`
    quantized with `algo`, layer `b` plain."""
    w = base_weights()
    sd = {k: v for k, v in w.items() if not k.startswith("a.weight")}
    extra = {"bits": 6} if algo == "w6a8_int8" else {}    # shares AsymW4A8Int8Layout, 6-bit codes
    qt = QuantizedTensor.from_float(w["a.weight"], QUANT_ALGOS[algo]["comfy_tensor_layout"], **extra)
    qsd = qt.state_dict("a.weight")           # a.weight (+ a.weight_scale, a.weight_scale_2)
    metadata = {}
    if source == "comfy_quant":
        sd.update(qsd)
        sd["a.comfy_quant"] = comfy_quant({"format": algo})
    elif source == "qmeta":
        sd.update(qsd)
        metadata["_quantization_metadata"] = json.dumps({"format_version": "1.0", "layers": {"a": {"format": algo}}})
    elif source == "marker":
        assert algo == "float8_e4m3fn"
        sd["a.weight"] = qsd["a.weight"]
        sd["a.scale_weight"] = qsd["a.weight_scale"]
        sd["scaled_fp8"] = torch.empty(0, dtype=torch.float8_e4m3fn)
    elif source == "both":
        sd.update(qsd)
        sd["scaled_fp8"] = torch.empty(0, dtype=torch.float8_e4m3fn)
        metadata["_quantization_metadata"] = json.dumps({"format_version": "1.0", "layers": {"a": {"format": algo}}})
    else:
        raise ValueError(source)
    return sd, metadata


class Net(torch.nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.a = ops.Linear(IN, OUT)
        self.b = ops.Linear(IN, OUT)

    def forward(self, x):
        return self.a(x) + self.b(x)


def build(source, algo, dequantize_first=False):
    sd, metadata = file_state_dict(source, algo)
    sd, _ = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
    sd.pop("scaled_fp8", None)
    net = Net(comfy.ops.mixed_precision_ops({}, compute_dtype=COMPUTE))
    net.load_state_dict(sd, strict=False)
    if dequantize_first and isinstance(net.a.weight, QuantizedTensor):
        net.a.weight = torch.nn.Parameter(net.a.weight.dequantize(), requires_grad=False)
        net.a.layout_type = None
    return net


def make_lora():
    g = torch.Generator().manual_seed(1)
    lora = {}
    for n in ("a", "b"):
        lora[n + ".lora_up.weight"] = (torch.randn(OUT, RANK, generator=g) * 0.02).half()
        lora[n + ".lora_down.weight"] = (torch.randn(RANK, IN, generator=g) * 0.1).half()
        lora[n + ".alpha"] = torch.tensor(4.0)
    return comfy.lora.load_lora(lora, {"a": "a.weight", "b": "b.weight"}, log_missing=False)


def layout_name(t):
    lc = t._layout_cls
    return lc if isinstance(lc, str) else lc.__name__


def quant_digest(t):
    h = hashlib.blake2b(digest_size=16)
    for v in t.state_dict("w").values():
        h.update(v.detach().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def run(net, loaded, x):
    p = comfy.model_patcher.ModelPatcher(net, CPU, CPU)
    p.add_patches(loaded, 0.8, 1.0)
    p.load(CPU, full_load=True)
    y = net(x)
    info = {"backup": len(p.backup), "quantized": isinstance(net.a.weight, QuantizedTensor)}
    p.unpatch_model(CPU)
    info["leftover"] = sum(len(getattr(m, "weight_function", [])) + len(getattr(m, "bias_function", []))
                           for m in (net.a, net.b))
    return y, info


CASES = [("comfy_quant", a) for a in ("float8_e4m3fn", "float8_e5m2", "int8_tensorwise", "nvfp4", "mxfp8",
                                         "convrot_w4a4", "asym_w4a8_int8", "w6a8_int8")] + \
        [("qmeta", "float8_e4m3fn"), ("marker", "float8_e4m3fn"), ("both", "float8_e4m3fn")]


def main():
    loaded = make_lora()
    x = (torch.randn(4, IN, generator=torch.Generator().manual_seed(2)) * 0.5).to(COMPUTE)
    n = 0

    native = {}
    hotpatch.uninstall()
    for source, algo in CASES:
        try:
            native[(source, algo)], _ = run(build(source, algo), loaded, x)
        except Exception as e:   # native runs the re-quantized layer through the format's kernel; not every one has a CPU kernel
            native[(source, algo)] = "{}: {}".format(type(e).__name__, str(e)[:60])

    hotpatch.install()
    try:
        for exact in (False, True):
            hotpatch.set_exact(exact)
            tag = "exact" if exact else "fused"
            for source, algo in CASES:
                name = "{} / {} / {}".format(source, algo, tag)
                net = build(source, algo)
                layout = QUANT_ALGOS[algo]["comfy_tensor_layout"]
                loaded_ok = isinstance(net.a.weight, QuantizedTensor) and layout_name(net.a.weight) == layout \
                    and not isinstance(net.b.weight, QuantizedTensor)
                check(name + ": loads as " + layout + " (layer b plain)", loaded_ok)
                before = quant_digest(net.a.weight) if loaded_ok else None
                y, info = run(net, loaded, x)
                ref, _ = run(build(source, algo, dequantize_first=True), loaded, x)
                same = torch.equal(y, ref)
                check(name + ": == dequantize-first + LoRA", same,
                      "" if same else "max_abs {}".format(float((y.float() - ref.float()).abs().max())))
                untouched = info["backup"] == 0 and info["quantized"] and info["leftover"] == 0 \
                    and loaded_ok and quant_digest(net.a.weight) == before
                check(name + ": no backup, quantized bytes unchanged, clean unpatch", untouched, str(info))
                nat = native[(source, algo)]
                if isinstance(nat, str):
                    print("    vs native: n/a on CPU ({})".format(nat))
                else:
                    print("    vs native (merge + re-quantize): max_abs {:.3g}".format(float((y.float() - nat.float()).abs().max())))
                n += 1
    finally:
        hotpatch.set_exact(False)
    finish({"cases": n})


if __name__ == "__main__":
    main()
