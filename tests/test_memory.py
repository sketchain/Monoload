"""Memory: peak RSS while loading (native vs Monoload), mmap check, and
LoRA sampling peak. Run each mode in a fresh process.

    python tests/test_memory.py --mode native   --name SRC.safetensors
    python tests/test_memory.py --mode monoload --name CONV.safetensors [--lora L.safetensors --family sd15]
"""

import argparse
import time

from common import *  # noqa: F401,F403
from common import MemWatch, check, family_inputs, finish, gib, load_monoload, load_native, sample, free_all
from monoload import rebuild
import nodes
import folder_paths


def model_bytes(model):
    return sum(e.tensor.numel() * e.tensor.element_size() for e in rebuild.state_entries(model))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["native", "monoload"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--lora")
    p.add_argument("--family", default="sd15")
    p.add_argument("--steps", type=int, default=2)
    a = p.parse_args()

    result = {"mode": a.mode, "name": a.name, "staging": os.environ.get("MONOLOAD_STAGING", "auto"),
              "buffer_mb": os.environ.get("MONOLOAD_BUFFER_MB") or "default"}
    t0 = time.perf_counter()
    with MemWatch() as w:
        patcher = load_native(a.name) if a.mode == "native" else load_monoload(a.name)
    result["load_seconds"] = time.perf_counter() - t0
    mb = model_bytes(patcher.model)
    rep = w.report()
    result.update({"model_bytes": mb, "load": rep})
    ratio = rep["peak_delta"] / mb
    print("load: model {}  RSS before {}  peak delta {} ({:.2f}x model)  after {}  mapped {}".format(
        gib(mb), gib(rep["rss_before"]), gib(rep["peak_delta"]), ratio, gib(rep["rss_after"]), rep["mapped_model_files"]))

    if a.mode == "monoload":
        check("no model file mapped during Monoload load", not rep["mapped_model_files"], str(rep["mapped_model_files"]))
        staged = result["staging"] == "always"
        from monoload import transfer
        allowance = (2 * transfer.buffer_bytes_from() if staged else 0) + 256 * 2 ** 20
        check("peak delta <= model + {} buffers + 256 MiB".format("2x" if staged else "0x"), rep["peak_delta"] <= mb + allowance,
              "peak delta {} vs model {} + allowance {}".format(gib(rep["peak_delta"]), gib(mb), gib(allowance)))

    if a.lora:
        pos, neg, latent = family_inputs(a.family)
        sample(patcher, pos, neg, latent, steps=1)  # warm-up (allocator, attention buffers)
        with MemWatch() as w0:
            sample(patcher, pos, neg, latent, steps=a.steps)
        free_all()
        lora_sd_patcher = nodes.LoraLoaderModelOnly().load_lora_model_only(patcher, a.lora, 0.8)[0]
        sample(lora_sd_patcher, pos, neg, latent, steps=1)
        with MemWatch() as w1:
            sample(lora_sd_patcher, pos, neg, latent, steps=a.steps)
        r0, r1 = w0.report(), w1.report()
        result["sample_plain"] = r0
        result["sample_lora"] = r1
        n_patched = len(lora_sd_patcher.patches)
        print("sampling peak delta: plain {}  with LoRA ({} patched keys) {}  | RSS after: plain {} lora {}".format(
            gib(r0["peak_delta"]), n_patched, gib(r1["peak_delta"]), gib(r0["rss_after"]), gib(r1["rss_after"])))
        result["lora_backup_len"] = len(lora_sd_patcher.backup)
        if a.mode == "monoload":
            check("LoRA: ModelPatcher.backup empty", len(lora_sd_patcher.backup) == 0, "backup={}".format(len(lora_sd_patcher.backup)))
            grow = (r1["peak_rss_sampled"] - r0["peak_rss_sampled"])
            lora_bytes = os.path.getsize(folder_paths.get_full_path_or_raise("loras", a.lora))
            largest = max(e.tensor.numel() for e in rebuild.state_entries(patcher.model)) * 4
            bound = lora_bytes + 8 * largest
            result["lora_growth"] = {"grow": grow, "lora_file": lora_bytes, "largest_layer_fp32": largest, "bound": bound}
            check("LoRA sampling peak growth <= LoRA file + 8 x largest layer (fp32)", grow <= bound,
                  "peak(lora) - peak(plain) = {} ; LoRA file {} ; largest layer fp32 {} ; bound {} ; model {}".format(
                      gib(grow), gib(lora_bytes), gib(largest), gib(bound), gib(mb)))
        else:
            print("native LoRA backup entries: {}, bytes {}".format(
                len(lora_sd_patcher.backup),
                gib(sum(b.weight.numel() * b.weight.element_size() for b in lora_sd_patcher.backup.values()))))
    finish(result)


if __name__ == "__main__":
    main()
