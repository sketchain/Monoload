"""LoRA on parameters that are not a comfy.ops layer's weight / bias (video
LoRAs: Wan's blocks.N.modulation / head.modulation, LTX's scale_shift_table,
any .diff / .diff_b on them; also a plain torch layer's weight), CPU, no model
files. Monoload merges them with a parameter swap around the owning module's
forward (MonoloadParamSwap): a merged temporary in place of the parameter
while that module runs, the stored parameter back afterwards, no backup.

The model mirrors Wan / LTX: a block with a comfy.ops Linear child and its own
`modulation` parameter read through comfy.model_management.cast_to in its
forward; a top-level `scale_shift_table` read in the root's forward (both are
non-leaf modules, which native ModelPatcher.load never patches); a plain
torch.nn.LayerNorm (a leaf, natively baked with a backup).

Checked, parameter dtype fp32 and bf16, full load and lowvram load:
  * native ComfyUI v0.46.3 drops the patches on the non-leaf parameters
    (recorded: the output equals the base model's for those keys); under
    lowvram it also leaves the plain torch layer unloaded and unpatched;
  * Monoload == native's own arithmetic bit for bit (native load followed by
    native patch_weight_to_device on the dropped keys, i.e. what native
    would compute if it patched them), exact and fused merge alike;
  * no backup, stored parameters unchanged, after the forward the stored
    parameter objects are back in place;
  * after unpatch no swap is left and the output is the base model's;
  * a hook LoRA on modulation == native hooks bit for bit, no hook backup.

    python tests/test_lora_param_swap.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import check, finish  # noqa: E402
import torch  # noqa: E402

import comfy.hooks  # noqa: E402
import comfy.lora  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402
from monoload import hotpatch  # noqa: E402

CPU = torch.device("cpu")
D = 64
NONLEAF = ("blk.modulation", "scale_shift_table")
ops = comfy.ops.disable_weight_init


class Blk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = ops.Linear(D, D)
        self.modulation = torch.nn.Parameter(torch.empty(1, 2, D))

    def forward(self, x):
        e = mm.cast_to(self.modulation, dtype=x.dtype, device=x.device).chunk(2, dim=1)
        return self.lin(x) * (1 + e[0][0]) + e[1][0]


class Net(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.blk = Blk()
        self.norm = torch.nn.LayerNorm(D)
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, D))
        with torch.no_grad():
            self.blk.lin.weight.copy_(torch.randn(D, D, generator=g) * 0.1)
            self.blk.lin.bias.copy_(torch.randn(D, generator=g) * 0.01)
            self.blk.modulation.copy_(torch.randn(1, 2, D, generator=g) * 0.1)
            self.norm.weight.copy_(1 + torch.randn(D, generator=g) * 0.1)
            self.norm.bias.copy_(torch.randn(D, generator=g) * 0.01)
            self.scale_shift_table.copy_(torch.randn(2, D, generator=g) * 0.1)
        self.to(dtype)

    def forward(self, x):
        t = self.scale_shift_table.to(dtype=x.dtype)
        return self.norm(self.blk(x)) * (1 + t[0]) + t[1]


def lora_sd(seed, with_lin):
    g = torch.Generator().manual_seed(seed)
    sd = {
        "blk.modulation.diff": torch.randn(1, 2, D, generator=g) * 0.05,
        "scale_shift_table.diff": torch.randn(2, D, generator=g) * 0.05,
        "norm.diff": torch.randn(D, generator=g) * 0.05,
        "norm.diff_b": torch.randn(D, generator=g) * 0.01,
    }
    if with_lin:
        sd["blk.lin.lora_up.weight"] = torch.randn(D, 4, generator=g) * 0.02
        sd["blk.lin.lora_down.weight"] = torch.randn(4, D, generator=g) * 0.1
    key_map = {"blk.modulation": "blk.modulation", "scale_shift_table": "scale_shift_table", "norm": "norm.weight",
               "blk.lin": "blk.lin.weight"}
    return comfy.lora.load_lora(sd, key_map, log_missing=False)


def stored(net):
    return [(n, p.detach().clone()) for n, p in net.named_parameters()]


def same(a, b):
    return len(a) == len(b) and all(n1 == n2 and torch.equal(t1, t2) for (n1, t1), (n2, t2) in zip(a, b))


def leftover(net):
    return sum(len(m.__dict__.get("_monoload_param_swaps") or {}) + len(m.__dict__.get("weight_function") or [])
               + len(m.__dict__.get("bias_function") or []) for m in net.modules())


def run(dtype, x, lowvram, patches=None, native_fill=False):
    net = Net(dtype)
    params_before = {n: p for n, p in net.named_parameters()}
    w0 = stored(net)
    p = comfy.model_patcher.ModelPatcher(net, CPU, CPU)
    if patches is not None:
        p.add_patches(patches, 0.8, 1.0)
    if lowvram:
        p.load(CPU, lowvram_model_memory=1)
    else:
        p.load(CPU, full_load=True)
    if native_fill:          # native's arithmetic on the keys native load drops (non-leaf; under lowvram also the
        for k in NONLEAF + ("norm.weight", "norm.bias"):   # plain torch layer, which native leaves unloaded)
            if k not in p.backup:
                p.patch_weight_to_device(k, device_to=CPU)
    y = net(x)
    info = {"backup": sorted(p.backup), "stored same": same(stored(net), w0),
            "objects back": all(params_before[n] is q for n, q in net.named_parameters())}
    swaps = {k: hotpatch._has_runtime_patch(p, k) for k in NONLEAF} if hotpatch.is_installed() else {}
    info["swaps"] = swaps
    merged = {}
    for k in NONLEAF:
        m, attr = k.rsplit(".", 1) if "." in k else ("", k)
        mod = net.get_submodule(m) if m else net
        sw = (mod.__dict__.get("_monoload_param_swaps") or {}).get(attr)
        merged[k] = sw.merged(getattr(mod, attr)) if sw is not None else getattr(mod, attr).detach().clone()
    p.unpatch_model(CPU)
    info["leftover"] = leftover(net)
    info["restored"] = same(stored(net), w0)
    info["after"] = net(x)
    return y, info, merged


def run_hooks(dtype, x):
    net = Net(dtype)
    w0 = stored(net)
    p = comfy.model_patcher.ModelPatcher(net, CPU, CPU)
    group = comfy.hooks.create_hook_lora(None, 0.6, 0.0)
    p.add_hook_patches(group.hooks[0], {k: v for k, v in lora_sd(7, False).items() if k == "blk.modulation"}, 0.6)
    p.hook_mode = comfy.hooks.EnumHookMode.MinVram
    p.load(CPU, full_load=True)
    p.patch_hooks(group)
    y = net(x)
    info = {"backups": len(p.backup) + len(p.hook_backup), "stored same": same(stored(net), w0)}
    p.unpatch_hooks()
    info["off"] = net(x)
    p.unpatch_model(CPU)
    info["restored"] = same(stored(net), w0)
    return y, info


def main():
    hotpatch.install()
    n = 0
    for dtype in (torch.float32, torch.bfloat16):
        x = (torch.randn(3, D, generator=torch.Generator().manual_seed(2))).to(dtype)
        dn = str(dtype).split(".")[1]
        for lowvram in (False, True):
            tag = "{} / {}".format(dn, "lowvram" if lowvram else "full load")
            hotpatch.uninstall()
            base, _, _ = run(dtype, x, lowvram)
            only_nonleaf = {k: v for k, v in lora_sd(1, False).items() if k in NONLEAF}
            y_drop, _, _ = run(dtype, x, lowvram, only_nonleaf)
            check(tag + ": native load drops the LoRA on non-leaf parameters (output == base)", torch.equal(y_drop, base))
            y_nat, info_nat, m_nat = run(dtype, x, lowvram, lora_sd(1, True), native_fill=True)
            check(tag + ": native arithmetic bakes them with a backup", {"norm.weight", "norm.bias"} | set(NONLEAF)
                  <= set(info_nat["backup"]), str(info_nat["backup"]))
            hotpatch.install()
            for exact in (True, False):
                hotpatch.set_exact(exact)
                name = "{} / {}".format(tag, "exact" if exact else "fused")
                y, info, merged = run(dtype, x, lowvram, lora_sd(1, True))
                mm_ok = all(torch.equal(merged[k], m_nat[k]) for k in NONLEAF)
                check(name + ": merged modulation / scale_shift_table == native arithmetic bit for bit", mm_ok)
                d = float((y.float() - y_nat.float()).abs().max())
                if exact:
                    check(name + ": output == native (+ the dropped keys) bit for bit", torch.equal(y, y_nat),
                          "max_abs {:.3g}".format(d))
                else:
                    tol = 4 * (2 ** -8 if dtype == torch.bfloat16 else 2 ** -20) * float(y_nat.float().abs().max())
                    check(name + ": output close to native (max_abs {:.3g}, bound {:.3g}; only blk.lin is fused)".format(d, tol),
                          d <= tol)
                ok = (not info["backup"] and info["stored same"] and info["objects back"] and all(info["swaps"].values())
                      and info["leftover"] == 0 and info["restored"])
                check(name + ": swaps installed, no backup, stored parameters unchanged and back in place, clean unpatch",
                      ok, str({k: v for k, v in info.items() if k != "after"}))
                check(name + ": base model after unpatch", torch.equal(info["after"], base))
                n += 1
        hotpatch.set_exact(False)

        hotpatch.uninstall()
        yh_nat, ih_nat = run_hooks(dtype, x)
        hotpatch.install()
        yh, ih = run_hooks(dtype, x)
        check(dn + ": hook LoRA on modulation == native bit for bit (native backups {})".format(ih_nat["backups"]),
              torch.equal(yh, yh_nat), "max_abs {:.3g}".format(float((yh.float() - yh_nat.float()).abs().max())))
        check(dn + ": hook LoRA: no backup, stored unchanged, hooks off == native hooks off, restored",
              ih["backups"] == 0 and ih["stored same"] and torch.equal(ih["off"], ih_nat["off"]) and ih["restored"], str(
                  {k: v for k, v in ih.items() if k != "off"}))
    finish({"cases": n})


if __name__ == "__main__":
    main()
