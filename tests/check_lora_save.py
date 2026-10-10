"""CT 700 check of review item 01: saving a LoRA'd model with a real checkpoint
and LoRA. Loads the checkpoint (default waiIllustriousSDXL_v170.safetensors),
applies the LoRA through ComfyUI's LoraLoader and saves it with ComfyUI's
CheckpointSave (--all: also ModelSave and CLIPSave) several ways in one
process, then compares the files tensor by tensor (dtype, shape, bytes):

  base          the checkpoint without the LoRA (to count the tensors the LoRA changes)
  monoload      the global default (Monoload's runtime merge) [== native bit for bit]
  node native   Monoload LoRA Settings mode native before saving [== native]
  native        Monoload's hooks uninstalled: ComfyUI as shipped [the reference]

and after saving under Monoload: no weight backups, the weights unchanged,
the runtime patches still there. The files go to a temporary directory
(--out, default /tmp) and are deleted after hashing (an SDXL checkpoint is
about 6.5 GiB each; at most one exists at a time).

    docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/check_lora_save.py --lora Smooth_Booster_v5.safetensors

Across processes (a whole process with MONOLOAD=0, nothing Monoload in it):
    ... check_lora_save.py --lora X --digests /tmp/save_monoload.json            # writes the per-tensor digests
    docker exec ... -e MONOLOAD=0 comfyui python tests/check_lora_save.py --lora X --only-default --compare /tmp/save_monoload.json

Without --lora it lists models/loras and models/checkpoints. Same launch
arguments as tests/bench_lora.py ($COMFY_ARGS, else those of the container's
main process).
"""

import argparse
import glob
import hashlib
import json
import os
import shlex
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if "COMFY_ARGS" not in os.environ:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from monoload import comfy_env as _ce
    _auto = _ce.pid1_comfy_args()
    os.environ["COMFY_ARGS"] = shlex.join(_auto) if _auto is not None else "--cpu"

from common import COMFY_ARGS, free_all, set_runtime, weight_digests  # noqa: E402
import folder_paths  # noqa: E402
import nodes  # noqa: E402
from comfy_extras import nodes_model_merging as nmm  # noqa: E402
from check_lora_node import listing, runtime_patches  # noqa: E402
from monoload import settings  # noqa: E402
from monoload.nodes import NODE_CLASS_MAPPINGS  # noqa: E402


def file_digests(path):
    """{key: [dtype, shape, blake2b of the bytes]} straight from the safetensors file."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, info in header.items():
            if k == "__metadata__":
                continue
            a, b = info["data_offsets"]
            f.seek(base + a)
            out[k] = [info["dtype"], info["shape"], hashlib.blake2b(f.read(b - a), digest_size=16).hexdigest()]
    return out


def save(kind, model, clip, vae, out_dir, tag):
    d = tempfile.mkdtemp(prefix=tag + "_", dir=out_dir)
    rel = os.path.relpath(d, folder_paths.get_output_directory())
    if kind == "CheckpointSave":
        nmm.CheckpointSave().save(model, clip, vae, rel + "/ckpt")
    elif kind == "ModelSave":
        nmm.ModelSave().save(model, rel + "/model")
    else:
        nmm.CLIPSave().save(clip, rel + "/clip")
    res = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        res[os.path.basename(f).split("_0000")[0]] = file_digests(f)
    shutil.rmtree(d, ignore_errors=True)
    return res


def compare(a, b):
    """(tensors compared, tensors that differ, missing / extra keys)."""
    n, diff, keys = 0, [], []
    for f in sorted(set(a) | set(b)):
        da, db = a.get(f, {}), b.get(f, {})
        keys += ["{}: {}".format(f, k) for k in sorted(set(da) ^ set(db))]
        for k in sorted(set(da) & set(db)):
            n += 1
            if da[k] != db[k]:
                diff.append("{}: {}".format(f, k))
    return n, diff, keys


def report(label, ref_label, got, ref, base=None):
    n, diff, keys = compare(got, ref)
    ok = not diff and not keys and n > 0
    print("  {} vs {}: {} tensors, {} differ, {} keys missing / extra -> {}".format(
        label, ref_label, n, len(diff), len(keys), "IDENTICAL" if ok else "DIFFERENT"), flush=True)
    for x in (diff + keys)[:8]:
        print("      " + x)
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="waiIllustriousSDXL_v170.safetensors", help="models/checkpoints file")
    p.add_argument("--lora", help="models/loras file (omit to list the files)")
    p.add_argument("--strength", type=float, default=1.0, help="LoRA strength (model and CLIP)")
    p.add_argument("--all", action="store_true", help="also ModelSave and CLIPSave")
    p.add_argument("--out", default="/tmp", help="where the temporary files go")
    p.add_argument("--only-default", action="store_true", help="only the save under this process's settings (for --compare)")
    p.add_argument("--digests", help="write the digests of the default save to this JSON file")
    p.add_argument("--compare", help="compare the default save with the digests in this JSON file")
    a = p.parse_args()
    if not a.lora or a.lora not in folder_paths.get_filename_list("loras") or a.checkpoint not in folder_paths.get_filename_list("checkpoints"):
        print("not found: --lora {} or --checkpoint {}".format(a.lora, a.checkpoint) if a.lora else "pass --lora <file>; available:")
        listing()
        return 1
    print("ComfyUI args {}; master switch MONOLOAD {}; global merge {}".format(
        " ".join(COMFY_ARGS), "on" if settings.master() else "off", "exact" if settings.exact() else "fused"), flush=True)
    folder_paths.set_output_directory(a.out)
    set_runtime(True)   # Monoload's methods, as the plugin installs them (under MONOLOAD=0 they pass every call through)
    kinds = ["CheckpointSave"] + (["ModelSave", "CLIPSave"] if a.all else [])
    model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(a.checkpoint)
    lm, lc = nodes.LoraLoader().load_lora(model, clip, a.lora, a.strength, a.strength)
    print("checkpoint {}, LoRA {} (strength {}): {} UNet / {} text-encoder keys patched".format(
        a.checkpoint, a.lora, a.strength, len(lm.patches), len(lc.patcher.patches)), flush=True)
    ok = True
    for kind in kinds:
        print(kind, flush=True)
        free_all()
        w_unet, w_clip = weight_digests(lm.model), weight_digests(lc.patcher.model)
        got = save(kind, lm, lc, vae, a.out, "default")
        if settings.master():
            bk = len(lm.backup) + len(lc.patcher.backup)
            rp = runtime_patches(lm) + runtime_patches(lc.patcher)
            same = weight_digests(lm.model) == w_unet and weight_digests(lc.patcher.model) == w_clip
            print("  after the Monoload save: backups {}, runtime patches {}, weights unchanged {}".format(bk, rp, same), flush=True)
            ok = ok and bk == 0 and same and (rp > 0 or kind == "CLIPSave" and not lc.patcher.patches)
        if a.digests and kind == "CheckpointSave":
            with open(a.digests, "w") as f:
                json.dump(got, f)
            print("  digests written to {}".format(a.digests), flush=True)
        if a.compare and kind == "CheckpointSave":
            with open(a.compare) as f:
                ok = report("this process", a.compare, got, json.load(f)) and ok
        if a.only_default:
            continue
        free_all()
        base = save(kind, model, clip, vae, a.out, "base")
        nm, nc = NODE_CLASS_MAPPINGS["MonoloadLoRASettings"]().apply(lm, mode="native", clip=lc)
        free_all()
        node_native = save(kind, nm, nc, vae, a.out, "node_native")
        set_runtime(False)
        try:
            ref = save(kind, lm, lc, vae, a.out, "native")
        finally:
            set_runtime(True)
        _, lora_diff, _ = compare(base, ref)
        print("  the LoRA changes {} tensors of the native save (vs the base)".format(len(lora_diff)), flush=True)
        ok = report("default (Monoload)" if settings.master() else "default (MONOLOAD=0)", "native", got, ref) and ok
        ok = report("node mode native", "native", node_native, ref) and ok
        patched = {"CheckpointSave": lm.patches or lc.patcher.patches, "ModelSave": lm.patches, "CLIPSave": lc.patcher.patches}[kind]
        ok = ok and (len(lora_diff) > 0) == bool(patched)   # e.g. a LoRA without text-encoder keys leaves CLIPSave's file as the base
    print("RESULT: {}".format("OK" if ok else "MISMATCH"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
