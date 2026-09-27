"""Tampered files must be rejected with clear errors.

Tampered copies are sparse: a rewritten header followed by a hole of the
original data size, so they cost no disk space. Validation happens before any
data is read.

    python tests/test_validation.py --converted CONV.safetensors
"""

import argparse
import json
import logging
import os
import struct

from common import *  # noqa: F401,F403
from common import MODELS, MonoloadFormatError, MonoloadUnsupportedError, check, expect_raises, finish
import comfy.memory_management
from monoload import fmt
from monoload.loader import load_monoload_diffusion_model


def read_raw_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = json.loads(f.read(n))
    return raw, 8 + n


def write_tampered(src, dst, mutate, data_delta=0):
    raw, data_start = read_raw_header(src)
    data_len = os.path.getsize(src) - data_start
    mutate(raw)
    hb = json.dumps(raw, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
    os.truncate(dst, 8 + len(hb) + data_len + data_delta)


def edit_model_meta(fn):
    def mutate(raw):
        m = json.loads(raw["__metadata__"][fmt.META_MODEL])
        fn(m)
        raw["__metadata__"][fmt.META_MODEL] = json.dumps(m)
    return mutate


def tensor_names(raw):
    return [k for k in raw if k != "__metadata__"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--converted", required=True)
    a = p.parse_args()
    src = os.path.join(MODELS, "monoload", a.converted)
    dst = os.path.join(MODELS, "monoload", "_tampered_test.safetensors")
    raw0, _ = read_raw_header(src)
    names = tensor_names(raw0)
    two_d = next(k for k in names if len(raw0[k]["shape"]) == 2 and raw0[k]["shape"][0] != raw0[k]["shape"][1])
    f16 = next(k for k in names if raw0[k]["dtype"] in ("F16", "BF16") and k.startswith("diffusion_model."))
    victim = next(k for k in names if k.startswith("diffusion_model.") and k.endswith(".weight"))

    def rename(raw):
        raw[victim + "_renamed"] = raw.pop(victim)

    def reshape(raw):
        raw[two_d]["shape"] = list(reversed(raw[two_d]["shape"]))

    def delete(raw):
        raw.pop(victim)

    def redtype(raw):
        raw[f16]["dtype"] = "BF16" if raw[f16]["dtype"] == "F16" else "F16"

    def extra(raw):
        end = max(v["data_offsets"][1] for k, v in raw.items() if k != "__metadata__")
        raw["diffusion_model.extra_tensor"] = {"dtype": "F32", "shape": [4], "data_offsets": [end, end + 16]}

    def set_meta(key, value):
        def mutate(raw):
            if value is None:
                raw["__metadata__"].pop(key, None)
            else:
                raw["__metadata__"][key] = value
        return mutate

    def bad_class(m):
        m["config_class"] = "comfy.supported_models.NoSuchModelClass"

    def bad_dtype(m):
        cur = m["dtype_selection"]["unet_dtype"]["__dtype__"]
        m["dtype_selection"]["unet_dtype"] = {"__dtype__": "bfloat16" if cur != "bfloat16" else "float16"}

    def bad_model_type(m):
        m["model_type"]["name"] = "V_PREDICTION" if m["model_type"]["name"] != "V_PREDICTION" else "EPS"

    def bad_unet_config(m):
        uc = m["detected_unet_config"]
        key = "model_channels" if "model_channels" in uc else next(k for k, v in uc.items() if isinstance(v, int) and not isinstance(v, bool) and v > 8)
        uc[key] = uc[key] // 2

    cases = [
        ("tensor renamed", rename, 0, MonoloadFormatError, ["文件里有、模型没有", victim + "_renamed", "模型有、文件里没有"]),
        ("tensor shape changed", reshape, 0, MonoloadFormatError, ["形状", two_d]),
        ("tensor deleted", delete, 0, MonoloadFormatError, ["模型有、文件里没有", victim]),
        ("tensor dtype changed", redtype, 0, MonoloadFormatError, ["dtype", f16]),
        ("extra tensor", extra, 16, MonoloadFormatError, ["文件里有、模型没有", "extra_tensor"]),
        ("format version 2", set_meta(fmt.META_FORMAT_VERSION, "2"), 0, MonoloadFormatError, ["格式版本", "'2'"]),
        ("format marker missing", set_meta(fmt.META_FORMAT, None), 0, MonoloadFormatError, ["不是 Monoload"]),
        ("unknown component", set_meta(fmt.META_COMPONENT, "vae"), 0, MonoloadFormatError, ["组件类型", "vae"]),
        ("unknown quant", set_meta(fmt.META_QUANT, "gguf"), 0, MonoloadFormatError, ["量化类型", "gguf"]),
        ("model class not found", edit_model_meta(bad_class), 0, MonoloadFormatError, ["找不到模型配置类", "NoSuchModelClass"]),
        ("recorded dtype differs from current choice", edit_model_meta(bad_dtype), 0, MonoloadFormatError, ["推理 dtype"]),
        ("model_type tampered", edit_model_meta(bad_model_type), 0, MonoloadFormatError, ["model_type"]),
        ("unet_config tampered", edit_model_meta(bad_unet_config), 0, MonoloadFormatError, []),
        ("metadata JSON corrupt", set_meta(fmt.META_MODEL, "{not json"), 0, MonoloadFormatError, ["metadata 解析失败"]),
        ("truncated file", lambda raw: None, -1024, MonoloadFormatError, ["超出文件末尾"]),
    ]
    for name, mutate, delta, exc, subs in cases:
        write_tampered(src, dst, mutate, delta)
        try:
            expect_raises("reject: " + name, exc, lambda: load_monoload_diffusion_model(dst), *subs, "重新转换")
        finally:
            os.unlink(dst)

    # Not a Monoload file at all
    orig = os.path.join(MODELS, "diffusion_models", os.listdir(os.path.join(MODELS, "diffusion_models"))[0])
    expect_raises("reject: plain safetensors (not converted)", MonoloadFormatError, lambda: load_monoload_diffusion_model(orig), "不是 Monoload")

    # ComfyUI version differs -> warning only, load succeeds
    def other_version(raw):
        env = json.loads(raw["__metadata__"][fmt.META_ENV])
        env["comfyui_version"] = "0.0.1-other"
        raw["__metadata__"][fmt.META_ENV] = json.dumps(env)
    write_tampered(src, dst, other_version)
    records = []

    class H(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())
    h = H(level=logging.WARNING)
    logging.getLogger().addHandler(h)
    try:
        patcher = load_monoload_diffusion_model(dst)
        warned = [m for m in records if "0.0.1-other" in m]
        check("ComfyUI version mismatch only warns", patcher is not None and warned, warned[0] if warned else "no warning")
        del patcher
    except Exception as e:
        check("ComfyUI version mismatch only warns", False, "raised {}".format(e))
    finally:
        logging.getLogger().removeHandler(h)
        os.unlink(dst)

    # DynamicVRAM on -> explicit refusal
    comfy.memory_management.aimdo_enabled = True
    try:
        expect_raises("reject: DynamicVRAM enabled", MonoloadUnsupportedError, lambda: load_monoload_diffusion_model(src), "dynamic_vram", "--gpu-only")
    finally:
        comfy.memory_management.aimdo_enabled = False
    finish()


if __name__ == "__main__":
    main()
