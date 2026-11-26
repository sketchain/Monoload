"""ComfyUI-GGUF model + LoRA through Monoload's GGUF adapter (monoload/gguf.py),
CPU, no model files: a small GGMLOps model built from GGMLTensors the way
ComfyUI-GGUF's loader makes them (Q8_0, Q4_0, Q5_1 quantized weights, an F16
weight, F32 biases), wrapped in the pack's own GGUFModelPatcher.

Checked, for compute dtype fp32 and bf16, full load and lowvram load:
  * the adapter found the pack (by structure) and drives its patcher;
  * exact merge: output bit for bit == native ComfyUI-GGUF (quantized layers:
    its per-forward merge on the dequantized temporary; the F16 layer and the
    F32 bias with a diff: its baked merge, fp32 + seed-0 rounding);
  * default (fused) merge: close to native (reported, bounded);
  * native GGUF backs up the F16 / F32 keys; Monoload never does: no backup,
    the GGML bytes never change, nothing left on GGMLTensor.patches;
  * an F32 `modulation` on the model itself (non-leaf, like Wan's): native
    load drops its LoRA; Monoload's parameter swap == native GGUF arithmetic
    (fp32, seed 0) bit for bit, no backup;
  * after unpatch no runtime patch is left and the output is the base model's;
  * saving a GGUF model raises a clear error (native cannot save one either).

The pack is looked up under ComfyUI's custom_nodes (any folder whose nodes.py
defines GGUFModelPatcher), or $GGUF_NODE_PATH.

    python tests/test_gguf_lora.py
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, check, expect_raises, finish  # noqa: E402
import torch  # noqa: E402

import comfy.lora  # noqa: E402
import comfy.model_management  # noqa: E402
from monoload import gguf as mgguf  # noqa: E402
from monoload import hotpatch  # noqa: E402
from monoload.errors import MonoloadUnsupportedError  # noqa: E402

CPU = torch.device("cpu")
IN, OUT, RANK = 256, 64, 8


def find_pack():
    cands = [os.environ["GGUF_NODE_PATH"]] if os.environ.get("GGUF_NODE_PATH") else []
    cn = os.path.join(ROOT, "custom_nodes")
    if os.path.isdir(cn):
        cands += [os.path.join(cn, d) for d in sorted(os.listdir(cn))]
    for d in cands:
        f = os.path.join(d, "nodes.py")
        if os.path.isfile(f) and "class GGUFModelPatcher" in open(f, encoding="utf-8", errors="replace").read():
            return d
    return None


def import_pack(path):
    name = os.path.basename(os.path.normpath(path))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_state_dict(gguf, GGMLTensor):
    g = torch.Generator().manual_seed(0)
    Q = gguf.GGMLQuantizationType

    def ggml(w, qtype):
        if qtype in (Q.F32, Q.F16):
            t = w.to(torch.float32 if qtype == Q.F32 else torch.float16)
            return GGMLTensor(t, tensor_type=qtype, tensor_shape=t.shape)
        raw = torch.from_numpy(gguf.quants.quantize(w.float().numpy(), qtype))
        return GGMLTensor(raw, tensor_type=qtype, tensor_shape=torch.Size(w.shape))

    sd = {}
    for name, qtype in (("q8", Q.Q8_0), ("q4", Q.Q4_0), ("q5", Q.Q5_1), ("f16", Q.F16)):
        sd[name + ".weight"] = ggml(torch.randn(OUT, IN, generator=g) * 0.05, qtype)
        sd[name + ".bias"] = ggml(torch.randn(OUT, generator=g) * 0.01, Q.F32)
    sd["modulation"] = ggml(torch.randn(2, OUT, generator=g) * 0.1, Q.F32)   # like Wan's: F32, on a non-leaf module
    return sd


def make_lora():
    g = torch.Generator().manual_seed(1)
    lora = {}
    for n in ("q8", "q4", "q5", "f16"):
        lora[n + ".lora_up.weight"] = (torch.randn(OUT, RANK, generator=g) * 0.02).half()
        lora[n + ".lora_down.weight"] = (torch.randn(RANK, IN, generator=g) * 0.1).half()
        lora[n + ".alpha"] = torch.tensor(4.0)
    lora["q8.diff_b"] = torch.randn(OUT, generator=g) * 0.01      # a diff on the F32 bias of a quantized layer
    lora["modulation.diff"] = torch.randn(2, OUT, generator=g) * 0.05
    keys = {n: n + ".weight" for n in ("q8", "q4", "q5", "f16")}
    keys["modulation"] = "modulation"
    return comfy.lora.load_lora(lora, keys, log_missing=False)


class Net(torch.nn.Module):
    def __init__(self, ops, sd):
        super().__init__()
        for n in ("q8", "q4", "q5", "f16"):
            setattr(self, n, ops.Linear(IN, OUT))
        self.modulation = torch.nn.Parameter(sd["modulation"], requires_grad=False)
        self.load_state_dict(sd, strict=False)

    def forward(self, x):
        m = comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device)
        return (self.q8(x) + self.q4(x) + self.q5(x) + self.f16(x)) * (1 + m[0]) + m[1]


def raw_digest(net):
    return [(n, p.data.as_subclass(torch.Tensor).clone()) for n, p in net.named_parameters()]


def same_raw(a, b):
    return all(n1 == n2 and torch.equal(t1.view(torch.uint8) if t1.dtype != torch.uint8 else t1,
                                        t2.view(torch.uint8) if t2.dtype != torch.uint8 else t2)
               for (n1, t1), (n2, t2) in zip(a, b))


def run(pack, sd, loaded, x, lowvram, with_lora=True, native_fill=False):
    net = Net(pack.ops.GGMLOps, sd)
    p = pack.nodes.GGUFModelPatcher(net, load_device=CPU, offload_device=CPU)
    if with_lora:
        p.add_patches(loaded, 0.8, 1.0)
    before = raw_digest(net)
    if lowvram:
        p.load(CPU, lowvram_model_memory=1)
    else:
        p.load(CPU, full_load=True)
    if native_fill:   # native load never patches a parameter of a non-leaf module: native's arithmetic on it
        p.patch_weight_to_device("modulation", device_to=CPU)
    y = net(x)
    info = {"backup": sorted(p.backup)}
    p.unpatch_model(CPU)
    info["leftover"] = sum(len(m.__dict__.get("weight_function", [])) + len(m.__dict__.get("bias_function", []))
                           + len(m.__dict__.get("_monoload_param_swaps") or {}) for m in net.modules())
    info["ggml_patches"] = sum(len(getattr(q, "patches", []) or []) for q in net.parameters())
    info["raw_same"] = same_raw(before, raw_digest(net))
    info["after"] = net(x)
    return y, info, p


def main():
    path = find_pack()
    if not check("ComfyUI-GGUF found under custom_nodes (or $GGUF_NODE_PATH)", path is not None, str(path)):
        return finish()
    pack = import_pack(path)
    import gguf
    sd = make_state_dict(gguf, pack.ops.GGMLTensor)
    loaded = make_lora()
    n = 0

    hotpatch.install()
    check("adapter installed on {}".format(pack.nodes.GGUFModelPatcher.__qualname__),
          pack.nodes.GGUFModelPatcher in mgguf._INSTALLED and pack.ops.GGMLLayer in mgguf._LAYERS)

    for cdt in (torch.float32, torch.bfloat16):
        x = (torch.randn(3, IN, generator=torch.Generator().manual_seed(2)) * 0.5).to(cdt)
        for lowvram in (False, True):
            tag = "{} / {}".format(str(cdt).split(".")[1], "lowvram" if lowvram else "full load")
            hotpatch.set_exact(False)
            base, _, _ = run(pack, sd, loaded, x, lowvram, with_lora=False)

            hotpatch.uninstall()
            mgguf.uninstall()
            y_drop, _, _ = run(pack, sd, {k: v for k, v in loaded.items() if k == "modulation"}, x, lowvram)
            check(tag + ": native load drops the LoRA on the non-leaf modulation", torch.equal(y_drop, base))
            y_nat, info_nat, _ = run(pack, sd, loaded, x, lowvram, native_fill=True)
            hotpatch.install()
            check(tag + ": native GGUF backs up the F16 / F32 keys",
                  {"f16.weight", "q8.bias", "modulation"} <= set(info_nat["backup"]), str(info_nat["backup"]))

            for exact in (True, False):
                hotpatch.set_exact(exact)
                name = "{} / {}".format(tag, "exact" if exact else "fused")
                y, info, _ = run(pack, sd, loaded, x, lowvram)
                d = float((y.float() - y_nat.float()).abs().max())
                if exact:
                    check(name + ": == native ComfyUI-GGUF bit for bit", torch.equal(y, y_nat), "max_abs {:.3g}".format(d))
                else:
                    scale = float(y_nat.float().abs().max())
                    # fused does not round the F16 layer's merged weight back to fp16 (native bakes it in), so the
                    # bound is the coarser of the compute dtype and fp16
                    tol = 4 * (2 ** -8 if cdt == torch.bfloat16 else 2 ** -11) * scale
                    check(name + ": close to native (max_abs {:.3g}, bound {:.3g})".format(d, tol), d <= tol)
                ok = not info["backup"] and info["raw_same"] and info["ggml_patches"] == 0 and info["leftover"] == 0
                check(name + ": no backup, GGML bytes unchanged, no GGMLTensor.patches, clean unpatch", ok,
                      str({k: v for k, v in info.items() if k != "after"}))
                check(name + ": base model after unpatch", torch.equal(info["after"], base))
                n += 1
    hotpatch.set_exact(False)

    net = Net(pack.ops.GGMLOps, sd)
    p = pack.nodes.GGUFModelPatcher(net, load_device=CPU, offload_device=CPU)
    p.add_patches(loaded, 0.8, 1.0)
    expect_raises("saving a GGUF model: clear error", MonoloadUnsupportedError, lambda: p.model_state_dict_for_saving(net),
                  "gguf_save")
    finish({"cases": n})


if __name__ == "__main__":
    main()
