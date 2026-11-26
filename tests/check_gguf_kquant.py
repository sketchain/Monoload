"""Real machine, per layer: Monoload's runtime LoRA on every GGML quantization type of a real .gguf file
(K-quants included, which the synthetic CPU test cannot make), against native ComfyUI-GGUF.

For each quantization type in the file (Q2_K .. Q6_K, Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0, IQ4_NL / IQ4_XS, ...), the
first --per-type 2D tensors become the weight of a one-layer GGMLOps model on the compute device, with a random LoRA
(--rank) on it. Only those tensors are read from the file (memory-mapped), so a 20 GB file costs a few hundred MB.
Per tensor and per merge mode it reports:
  max|Δ| vs native   native = ComfyUI-GGUF with Monoload uninstalled (its own merge on the dequantized temporary);
                     exact must be 0, fused is reported with the bound used by tests/test_gguf_lora.py
  ms / forward       median of --reps forwards (synchronized)
  peak extra         torch max_memory_allocated during the forwards minus before (the dequantized temporary + merge)
  resident extra     memory_allocated after loading with the LoRA minus without (must be ~0 for Monoload: no backup)
  backups            must be 0 for Monoload

    python tests/check_gguf_kquant.py --gguf wan2.1-t2v-14b-Q4_K_M.gguf [--per-type 2] [--rank 16] [--dtype bf16]

--gguf is a path or a file name under ComfyUI's diffusion_models / unet / text_encoders / clip folders.
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if "COMFY_ARGS" not in os.environ:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from monoload import comfy_env as _ce
    import shlex
    _auto = _ce.pid1_comfy_args()
    os.environ["COMFY_ARGS"] = shlex.join(_auto) if _auto is not None else "--cpu"

from common import check, finish  # noqa: E402
import torch  # noqa: E402

import comfy.lora  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import folder_paths  # noqa: E402
from monoload import gguf as mgguf  # noqa: E402
from monoload import hotpatch  # noqa: E402
from test_gguf_lora import find_pack, import_pack  # noqa: E402


def resolve(name):
    if os.path.isfile(name):
        return name
    for folder in ("diffusion_models", "unet", "text_encoders", "clip"):
        try:
            p = folder_paths.get_full_path(folder, name)
        except Exception:
            p = None
        if p:
            return p
    raise SystemExit("not found: {}".format(name))


def pick_tensors(path, pack, per_type):
    import gguf
    reader = gguf.GGUFReader(path)
    torch_types = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16}
    picked = {}
    for t in reader.tensors:
        if t.tensor_type in torch_types:
            continue
        shape = pack.loader.get_orig_shape(reader, t.name) or torch.Size(reversed([int(v) for v in t.shape]))
        if len(shape) != 2 or shape[0] * shape[1] > 64 * 2 ** 20:
            continue
        lst = picked.setdefault(t.tensor_type, [])
        if len(lst) < per_type:
            lst.append((t.name, t, torch.Size(shape)))
    return picked


class One(torch.nn.Module):
    def __init__(self, ops, weight, shape):
        super().__init__()
        self.l = ops.Linear(shape[1], shape[0], bias=False)
        self.l.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return self.l(x)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def alloc(dev):
    return torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0


def run(pack, ggml_cls, tensor, shape, dev, cdt, lora, reps, with_lora=True):
    raw = torch.from_numpy(tensor.data.copy()).to(dev)   # only this tensor is read from the memory-mapped file
    w = ggml_cls(raw, tensor_type=tensor.tensor_type, tensor_shape=shape)
    net = One(pack.ops.GGMLOps, w, shape)
    p = pack.nodes.GGUFModelPatcher(net, load_device=dev, offload_device=dev)
    if with_lora:
        p.add_patches(lora, 1.0, 1.0)
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(64, shape[1], generator=g, device=dev, dtype=torch.float32).to(cdt)
    sync(dev)
    a0 = alloc(dev)
    p.load(dev, full_load=True)
    sync(dev)
    resident = alloc(dev) - a0
    with torch.no_grad():
        y = net(x)                                  # warm-up / result
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        base = alloc(dev)
        times = []
        for _ in range(reps):
            sync(dev)
            t0 = time.perf_counter()
            net(x)
            sync(dev)
            times.append(time.perf_counter() - t0)
        peak = (torch.cuda.max_memory_allocated(dev) - base) if dev.type == "cuda" else 0
    info = {"ms": statistics.median(times) * 1e3, "peak": peak, "resident": resident, "backups": len(p.backup)}
    p.unpatch_model(dev)
    y = y.float()
    del net, p, w, raw
    return y, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--per-type", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    a = ap.parse_args()
    cdt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    dev = mm.get_torch_device()
    path = resolve(a.gguf)
    pack_path = find_pack()
    if not check("ComfyUI-GGUF found", pack_path is not None):
        return finish()
    pack = import_pack(pack_path)
    picked = pick_tensors(path, pack, a.per_type)
    print("file {}  device {}  compute {}  types: {}".format(os.path.basename(path), dev, a.dtype,
                                                            ", ".join("{} x{}".format(q.name, len(v)) for q, v in picked.items())))
    hotpatch.install()
    rows = []
    for qtype, items in picked.items():
        for name, tensor, shape in items:
            g = torch.Generator().manual_seed(1)
            lora = comfy.lora.load_lora({
                "l.lora_up.weight": (torch.randn(shape[0], a.rank, generator=g) * 0.02).half(),
                "l.lora_down.weight": (torch.randn(a.rank, shape[1], generator=g) * 0.1).half(),
                "l.alpha": torch.tensor(float(a.rank))}, {"l": "l.weight"}, log_missing=False)
            tag = "{} {} {}".format(qtype.name, name, list(shape))
            hotpatch.uninstall()
            mgguf.uninstall()
            y_nat, i_nat = run(pack, pack.ops.GGMLTensor, tensor, shape, dev, cdt, lora, a.reps)
            _, i_base = run(pack, pack.ops.GGMLTensor, tensor, shape, dev, cdt, lora, a.reps, with_lora=False)
            hotpatch.install()
            res = {}
            for exact in (True, False):
                hotpatch.set_exact(exact)
                y, info = run(pack, pack.ops.GGMLTensor, tensor, shape, dev, cdt, lora, a.reps)
                d = float((y - y_nat).abs().max())
                res["exact" if exact else "fused"] = (d, info)
            hotpatch.set_exact(False)
            scale = float(y_nat.abs().max())
            tol = 4 * (2 ** -8 if cdt == torch.bfloat16 else 2 ** -11) * scale
            de, ie = res["exact"]
            df, if_ = res["fused"]
            check(tag + ": exact == native bit for bit", de == 0.0, "max|Δ| {:.3g}".format(de))
            check(tag + ": fused close to native (max|Δ| {:.3g}, bound {:.3g})".format(df, tol), df <= tol)
            check(tag + ": no backup, no resident extra over the no-LoRA load",
                  ie["backups"] == 0 and if_["backups"] == 0 and ie["resident"] <= i_base["resident"] + 2 ** 20,
                  "resident Monoload {:.1f} MiB / no LoRA {:.1f} / native {:.1f}; native backups {}".format(
                      ie["resident"] / 2 ** 20, i_base["resident"] / 2 ** 20, i_nat["resident"] / 2 ** 20, i_nat["backups"]))
            rows.append((tag, i_base, i_nat, ie, if_))
    print("\n{:60s} | {:>22s} | {:>22s} | {:>22s} | {:>22s}".format("tensor", "no LoRA ms / peak MiB", "native", "exact", "fused"))
    for tag, b, n, e, f in rows:
        cell = lambda i: "{:8.3f} / {:9.1f}".format(i["ms"], i["peak"] / 2 ** 20)
        print("{:60s} | {:>22s} | {:>22s} | {:>22s} | {:>22s}".format(tag[:60], cell(b), cell(n), cell(e), cell(f)))
    finish({"tensors": len(rows)})


if __name__ == "__main__":
    main()
