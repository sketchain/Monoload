"""Synthesize small SD1.5 LoKr and LoHa files (no public SD1.5 LoKr was found).

Layer names/shapes are taken from a real kohya SD1.5 LoRA, so ComfyUI's key
mapping treats them exactly like downloaded LyCORIS files.

    python tests/make_synthetic_loras.py REFERENCE_LORA OUT_DIR
"""

import os
import sys

import torch
from safetensors.torch import save_file  # writing a small test file, not loading a model


def main(ref, out_dir):
    import json
    import struct
    with open(ref, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    g = torch.Generator().manual_seed(7)
    lokr, loha = {}, {}
    for k, info in header.items():
        if not k.endswith(".lora_down.weight") or len(info["shape"]) != 2:
            continue
        base = k[: -len(".lora_down.weight")]
        up = header[base + ".lora_up.weight"]["shape"]
        out_f, in_f = up[0], info["shape"][1]
        a = next(d for d in (8, 4, 2, 1) if out_f % d == 0)
        b = next(d for d in (8, 4, 2, 1) if in_f % d == 0)
        lokr[base + ".lokr_w1"] = (torch.randn(a, b, generator=g) * 0.5).half()
        lokr[base + ".lokr_w2"] = (torch.randn(out_f // a, in_f // b, generator=g) * 0.01).half()
        r = 4
        for name, shape in (("hada_w1_a", (out_f, r)), ("hada_w1_b", (r, in_f)), ("hada_w2_a", (out_f, r)), ("hada_w2_b", (r, in_f))):
            loha[base + "." + name] = (torch.randn(*shape, generator=g) * 0.1).half()
        loha[base + ".alpha"] = torch.tensor(float(r)).half()
    os.makedirs(out_dir, exist_ok=True)
    save_file(lokr, os.path.join(out_dir, "synthetic_lokr_sd15.safetensors"), metadata={"ss_network_module": "lycoris.kohya", "ss_network_args": '{"algo": "lokr"}'})
    save_file(loha, os.path.join(out_dir, "synthetic_loha_sd15.safetensors"), metadata={"ss_network_module": "lycoris.kohya", "ss_network_args": '{"algo": "loha"}'})
    print("lokr keys", len(lokr), "loha keys", len(loha))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
