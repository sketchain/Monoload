"""fp8 (scaled) diffusion model + LoRA: relaxed runtime merge.

Checked:
  * fp8 model + LoRA (Monoload) is bit-identical to the same model whose fp8
    weights were dequantized beforehand + LoRA (Monoload, same merge path:
    bit-exact with MONOLOAD_EXACT=1, otherwise the default fused/relaxed one).
    "Dequantized beforehand" covers the layers the LoRA patches: fp8 layers
    without a weight function never dequantize in native ComfyUI either (the
    QuantizedTensor goes straight into F.linear via comfy_kitchen, a different
    op order), so a fully dequantized model differs from the fp8 model even
    without any LoRA. That full-dequantization difference is reported, not judged.
  * no backups; the fp8 weights (qdata + scales) never change
  * error vs native ComfyUI (merge + re-quantize to fp8 + backup) is reported, not judged

    python tests/test_quant.py --unet sd15_unet_fp8_scaled.safetensors --clip clip_l.safetensors
"""

import argparse
import hashlib

import torch

from common import (MODE_TAG, apply_loras, byte_view, check, diff_stats, encode, finish, free_all, load_clip, load_unet, sample,
                    set_runtime)
from comfy.quant_ops import QuantizedTensor

COMBOS = {
    "duck": [("rubber_duck.safetensors", 0.8, 0.8)],
    "locon": [("lycoris_annalise.safetensors", 0.7, 0.9)],
}
POS = "a photo of a yellow rubber duck on a wooden table"
NEG = "blurry"


def quantized_modules(model):
    return [m for m in model.modules() if isinstance(m.__dict__.get("_parameters", {}).get("weight"), QuantizedTensor)]


def dequantize_in_place(patcher, only_keys=None):
    """Reference model: QuantizedTensor weights (all, or those whose key is in
    only_keys) replaced by their plain dequantized tensor, the layer told it is
    no longer quantized."""
    n = 0
    names = {id(m): n_ for n_, m in patcher.model.named_modules()}
    for m in quantized_modules(patcher.model):
        if only_keys is not None and names[id(m)] + ".weight" not in only_keys:
            continue
        w = m.weight
        m.weight = torch.nn.Parameter(w.dequantize(), requires_grad=False)
        m.layout_type = None
        n += 1
    return n


def quant_digest(model):
    h = hashlib.blake2b(digest_size=16)
    for name, p in model.named_parameters():
        if isinstance(p, QuantizedTensor):
            h.update(byte_view(p._qdata).numpy().tobytes())
            h.update(byte_view(p._params.scale).numpy().tobytes())
        else:
            h.update(byte_view(p).numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--steps", type=int, default=2)
    a = p.parse_args()
    latent = torch.zeros(1, 4, 32, 32)
    summary = {"mode": MODE_TAG}
    print("merge path: {}".format(MODE_TAG))

    set_runtime(True)
    fp8 = load_unet(a.unet)
    nq = len(quantized_modules(fp8.model))
    check("fp8 model loaded with quantized layers", nq > 0, "{} QuantizedTensor weights, e.g. dtype {}".format(
        nq, quantized_modules(fp8.model)[0].weight.dtype if nq else None))
    clip = load_clip(a.clip)
    d0 = quant_digest(fp8.model)

    pos, neg = encode(clip, POS), encode(clip, NEG)
    plain_q = sample(fp8, pos, neg, latent, steps=a.steps)
    free_all()
    full = load_unet(a.unet)
    dequantize_in_place(full)
    plain_full = sample(full, pos, neg, latent, steps=a.steps)
    del full
    free_all()
    d = diff_stats(plain_q, plain_full)
    print("[INFO] no LoRA: fp8 model vs fully dequantized model: max_abs {:.4g} mean_abs {:.4g} "
          "(unpatched fp8 layers run F.linear on the QuantizedTensor, not dequantize-then-linear)".format(d["max_abs"], d["mean_abs"]))
    summary["no_lora_full_dequant"] = d

    outs = {}
    for name, loras in COMBOS.items():
        mq, cq = apply_loras(fp8, clip, loras)
        pos, neg = encode(cq, POS), encode(cq, NEG)
        out_q = sample(mq, pos, neg, latent, steps=a.steps)
        params = dict(mq.model.named_parameters())
        fp8_keys = {k for k in mq.patches if isinstance(params.get(k), QuantizedTensor)}
        free_all()
        ref = load_unet(a.unet)
        n_deq = dequantize_in_place(ref, only_keys=fp8_keys)
        mr, _ = apply_loras(ref, clip, loras)
        out_r = sample(mr, pos, neg, latent, steps=a.steps)
        free_all()
        d = diff_stats(out_q, out_r)
        check("{}: fp8 + LoRA == dequantize-first + LoRA ({} keys patched, {} fp8 layers dequantized in the reference)".format(
            name, len(mq.patches), n_deq), d["bit_exact"] and n_deq == len(fp8_keys), str(d))
        check("{}: LoRA changes the output".format(name), not torch.equal(out_q, plain_q))
        check("{}: no backups".format(name), len(mq.backup) + len(mq.hook_backup) + len(mr.backup) == 0)
        outs[name] = out_q
        summary[name] = {"vs_dequantized": d, "fp8_keys": len(fp8_keys)}
        del ref, mr
    check("fp8 weights (qdata + scales) unchanged after LoRA", quant_digest(fp8.model) == d0)

    # native ComfyUI for comparison (bakes, re-quantizes to fp8, backs up): report only
    set_runtime(False)
    for name, loras in COMBOS.items():
        mq, cq = apply_loras(fp8, clip, loras)
        pos, neg = encode(cq, POS), encode(cq, NEG)
        out_n = sample(mq, pos, neg, latent, steps=a.steps)
        d = diff_stats(out_n, outs[name])
        effect = diff_stats(plain_q, outs[name])["max_abs"]
        print("[INFO] {}: Monoload (relaxed) vs native fp8 LoRA: max_abs {:.4g} mean_abs {:.4g} (LoRA effect max_abs {:.4g}; native backups {})".format(
            name, d["max_abs"], d["mean_abs"], effect, len(mq.backup)))
        summary[name]["vs_native"] = d
        free_all()
    set_runtime(True)
    finish(summary)


if __name__ == "__main__":
    main()
