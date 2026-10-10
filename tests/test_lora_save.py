"""Saving a LoRA'd model (review 2026-10 item 01): CheckpointSave / ModelSave /
CLIPSave (comfy.sd.save_checkpoint, CLIP.state_dict_for_saving) go through
ModelPatcher.model_state_dict_for_saving, which outputs the stored tensor of
every module flagged comfy_patched_weights. Under Monoload those tensors are
the base, so hotpatch replaces the method: the modules of the patcher's patched
keys go through native's LazyCastingParam (patch_weight_to_device(return_weight=True)),
the exact merge native computes.

  1 no model files: a small comfy.ops model, full load and lowvram load,
    exact and fused global default: model_state_dict_for_saving == native
    (Monoload's methods uninstalled) bit for bit and != the base; afterwards no
    backups, the weights unchanged, the outputs the same as before saving;
  2 the random-weight SD1.5 checkpoint and LoRA of
    tests/make_synthetic_checkpoint.py (under $MODELS): CheckpointLoaderSimple
    -> LoraLoader -> ComfyUI's CheckpointSave / ModelSave / CLIPSave nodes,
    every tensor of every file == the native save bit for bit (and the LoRA'd
    tensors != the base save); the model still runs the same afterwards, no
    backups, the weights unchanged; a node with mode native saves the same.

    python tests/make_synthetic_checkpoint.py $MODELS   # once
    MODELS=... tests/docker_run.sh python tests/test_lora_save.py
"""

import glob
import hashlib
import os
import shutil
import tempfile

import torch

from common import byte_view, check, finish, free_all, set_runtime, weight_digests
import comfy.model_management as mm
import comfy.model_patcher
import comfy.utils
import folder_paths
import nodes
from comfy_extras import nodes_model_merging as nmm
from monoload import hotpatch, settings
from test_master_switch import make_lora, make_net, same, weights

CPU = torch.device("cpu")
CKPT = "synthetic_sd15.safetensors"
LORA = "synthetic_sd15_lora.safetensors"


# ---------------------------------------------------------------------------
# 1. small model
# ---------------------------------------------------------------------------

def saved(p, lowvram):
    if lowvram:
        p.load(CPU, lowvram_model_memory=1)
    else:
        mm.load_models_gpu([p])
    return {k: v.to("cpu").clone() for k, v in p.model_state_dict_for_saving().items()}


def small_tests():
    x = torch.randn(4, 96, generator=torch.Generator().manual_seed(3))
    for exact in (True, False):
        for lowvram in (False, True):
            name = "{} / {}".format("exact" if exact else "fused", "lowvram" if lowvram else "full load")
            net = make_net()
            w0 = weights(net)
            p = comfy.model_patcher.ModelPatcher(net, CPU, CPU)
            p.add_patches(make_lora(5), 0.8)
            settings.set_exact(exact)
            try:
                set_runtime(False)
                ref = saved(p, lowvram)
                p.unpatch_model(CPU)
                set_runtime(True)
                got = saved(p, lowvram)
                before = net(x)
                again = saved(p, lowvram)
                after = net(x)
                ok_state = len(p.backup) == 0 and same(weights(net), w0) and torch.equal(before, after)
                p.unpatch_model(CPU)
            finally:
                settings.set_exact(False)
                free_all()
            check("small model, {}: saved state dict == native bit for bit, LoRA'd keys != base ({}), twice; no backups, weights "
                  "unchanged, outputs the same after saving".format(name, sorted(k for k in ref if not torch.equal(ref[k], w0[k]))),
                  same(got, ref) and same(again, ref) and not torch.equal(got["a.weight"], w0["a.weight"])
                  and not torch.equal(got["b.weight"], w0["b.weight"]) and ok_state)


# ---------------------------------------------------------------------------
# 2. ComfyUI's save nodes on the synthetic SD1.5
# ---------------------------------------------------------------------------

def file_digests(path):
    out = {}
    sd = comfy.utils.load_torch_file(path)
    for k, v in sd.items():
        out[k] = (v.dtype, tuple(v.shape), hashlib.blake2b(byte_view(v).numpy().tobytes(), digest_size=16).hexdigest())
    return out


def save_all(model, clip, vae, out_dir, tag):
    """{node: {file name: tensor digests}} of CheckpointSave, ModelSave, CLIPSave."""
    res = {}
    for name, fn in (("CheckpointSave", lambda d: nmm.CheckpointSave().save(model, clip, vae, d + "/ckpt")),
                     ("ModelSave", lambda d: nmm.ModelSave().save(model, d + "/model")),
                     ("CLIPSave", lambda d: nmm.CLIPSave().save(clip, d + "/clip"))):
        d = os.path.join(out_dir, tag, name)
        fn(os.path.relpath(d, folder_paths.get_output_directory()))
        files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
        res[name] = {os.path.basename(f): file_digests(f) for f in files}
        for f in files:
            os.remove(f)
    return res


def compare(a, b):
    """(identical, number of tensors, tensors that differ)."""
    n, diff = 0, []
    for node_name in a:
        if a[node_name].keys() != b.get(node_name, {}).keys():
            return False, n, ["{}: files {} vs {}".format(node_name, list(a[node_name]), list(b.get(node_name, {})))]
        for f, da in a[node_name].items():
            db = b[node_name][f]
            if da.keys() != db.keys():
                return False, n, ["{}/{}: keys differ".format(node_name, f)]
            for k in da:
                n += 1
                if da[k] != db[k]:
                    diff.append("{}/{}/{}".format(node_name, f, k))
    return not diff and n > 0, n, diff


def clip_out(clip):
    return nodes.CLIPTextEncode().encode(clip, "a photo of a duck")[0][0][0].clone()


def checkpoint_tests():
    have = CKPT in folder_paths.get_filename_list("checkpoints") and LORA in folder_paths.get_filename_list("loras")
    if not check("synthetic SD1.5 checkpoint and LoRA present (tests/make_synthetic_checkpoint.py)", have):
        return
    out_dir = tempfile.mkdtemp(prefix="monoload_save_")
    saved_out = folder_paths.get_output_directory()
    folder_paths.set_output_directory(out_dir)
    try:
        set_runtime(True)
        model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(CKPT)
        lm, lc = nodes.LoraLoader().load_lora(model, clip, LORA, 0.8, 0.6)
        base = save_all(model, clip, vae, out_dir, "base")
        free_all()
        w_unet, w_clip = weight_digests(lm.model), weight_digests(lc.patcher.model)
        enc0 = clip_out(lc)
        mono = save_all(lm, lc, vae, out_dir, "monoload")
        enc1 = clip_out(lc)
        bk = len(lm.backup) + len(lc.patcher.backup)
        rp = sum(1 for m in lc.patcher.model.modules() for f in (m.__dict__.get("weight_function") or ())
                 if getattr(f, "is_monoload_patch", False))
        unchanged = weight_digests(lm.model) == w_unet and weight_digests(lc.patcher.model) == w_clip
        from monoload.nodes import NODE_CLASS_MAPPINGS
        cls = NODE_CLASS_MAPPINGS["MonoloadLoRASettings"]
        nm, nc = cls().apply(lm, mode="native", clip=lc)
        free_all()
        native_node = save_all(nm, nc, vae, out_dir, "native_node")
        free_all()
        set_runtime(False)
        ref = save_all(lm, lc, vae, out_dir, "native")
        free_all()
        set_runtime(True)
        ok, n, diff = compare(mono, ref)
        ok_b, _, base_diff = compare(base, ref)
        check("CheckpointSave / ModelSave / CLIPSave after LoraLoader, default settings: every tensor == native bit for bit "
              "({} tensors, files {}; {} of them carry the LoRA, i.e. differ from the base save){}".format(
                  n, {k: list(v) for k, v in mono.items()}, len(base_diff), "" if ok else "; differ: {}".format(diff[:5])),
              ok and not ok_b and len(base_diff) > 0)
        check("after saving: no backups ({}), weights unchanged, runtime patches in place ({}), text-encoder output the same".format(bk, rp),
              bk == 0 and unchanged and rp > 0 and torch.equal(enc0, enc1))
        ok_n, _, diff_n = compare(native_node, ref)
        check("Monoload LoRA Settings mode native, then the save nodes: == native bit for bit{}".format(
              "" if ok_n else "; differ: {}".format(diff_n[:5])), ok_n)
    finally:
        free_all()
        folder_paths.set_output_directory(saved_out)
        shutil.rmtree(out_dir, ignore_errors=True)


def main():
    hotpatch.install()
    small_tests()
    checkpoint_tests()
    finish()


if __name__ == "__main__":
    main()
