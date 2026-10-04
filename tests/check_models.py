"""Which VAE does each model file on this machine decode with, and what does
Monoload do with it (phase 4a inventory, CT 700).

Reads only the safetensors headers (and the tensors under 64 KiB, which some
detections read); everything is built on the meta device: no memory, no GPU,
safe to run next to a working ComfyUI.

    docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/check_models.py

  models/vae                    the kind comfy.sd.VAE builds (first-stage model,
                                decoder, latent channels / dims, spatial ratio)
                                and Monoload's path for a 1344x768 and a 3840x2160
                                image (and an 81-frame video for video VAEs)
  models/checkpoints            the model ComfyUI detects, its latent format, and
                                the built-in VAE (as above)
  models/diffusion_models, unet the model ComfyUI detects and its latent format,
                                i.e. the VAE family it needs, and the files in
                                models/vae that fit it

--dir DIR reads a models directory directly instead of ComfyUI's folder paths.
"""

import argparse
import json
import logging
import os
import struct
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("COMFY_ARGS", "--cpu")

import common  # noqa: E402,F401  (ComfyUI environment)
import torch  # noqa: E402

import comfy.model_detection as md  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

from monoload import vae as mvae  # noqa: E402

SMALL = 64 * 1024
DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
          "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
          "F8_E4M3": getattr(torch, "float8_e4m3fn", torch.uint8), "F8_E5M2": getattr(torch, "float8_e5m2", torch.uint8),
          "F8_E8M0": getattr(torch, "float8_e8m0fnu", torch.uint8), "U16": getattr(torch, "uint16", torch.int16),
          "U32": getattr(torch, "uint32", torch.int32), "U64": getattr(torch, "uint64", torch.int64)}


def read_header(path):
    """(state dict: meta tensors, real ones for the small entries; metadata dict or None)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        meta = header.pop("__metadata__", None)
        sd = {}
        for k, info in header.items():
            dt = DTYPES.get(info["dtype"])
            if dt is None:
                continue
            shape = info["shape"]
            a, b = info["data_offsets"]
            if b - a <= SMALL:
                f.seek(base + a)
                buf = bytearray(f.read(b - a))
                t = torch.frombuffer(buf, dtype=dt) if buf else torch.empty(0, dtype=dt)
                sd[k] = t.reshape(shape)
            else:
                sd[k] = torch.empty(shape, dtype=dt, device="meta")
    return sd, meta


def files(folder, root):
    if root:
        d = os.path.join(root, folder)
        return [(f, os.path.join(d, f)) for f in sorted(os.listdir(d))] if os.path.isdir(d) else []
    try:
        return [(f, folder_paths.get_full_path(folder, f)) for f in folder_paths.get_filename_list(folder)]
    except Exception:
        return []


def build_vae(sd, meta):
    orig = mm.is_amd
    mm.is_amd = lambda: True
    logging.disable(logging.WARNING)   # the meta build's "missing keys" (encoder-only files etc.) and load messages
    try:
        with torch.device("meta"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = comfy.sd.VAE(sd=sd, dtype=torch.bfloat16, metadata=meta)
    finally:
        mm.is_amd = orig
        logging.disable(logging.NOTSET)
    if v.first_stage_model is None:
        return None
    v.device = v.output_device = torch.device("meta")
    return v


def latent(v, w, h, frames=1):
    r = v.spacial_compression_decode()
    if v.latent_dim == 3:
        dr = v.downscale_ratio
        t = dr[0](frames) if isinstance(dr, tuple) and callable(dr[0]) else 1
        return (1, v.latent_channels, max(1, t), round(h / r), round(w / r))
    return (1, v.latent_channels, round(h / r), round(w / r))


def monoload_path(v, shape):
    lat = torch.empty(shape, device="meta", dtype=v.vae_dtype)
    reason = mvae._native_reason(v, lat)
    if reason is not None:
        return "native: " + reason
    if v.latent_dim == 2 and lat.ndim == 5:
        lat = lat[:, :, 0]
    why = []
    for a in mvae.STRIPE_ADAPTERS:
        bound, w = a.match(v, lat, {})
        if bound is not None:
            return "layer 1 ({})".format(bound.name)
        why.append(w)
    return "layer 2 (layer 1: {})".format("; ".join(why))


def describe_vae(v, indent="   "):
    fsm = v.first_stage_model
    dec = getattr(fsm, "decoder", None)
    print("{}VAE: {} / decoder {} | latent {} ch, latent_dim {}, x{} | {:.0f} M params".format(
        indent, type(fsm).__name__, type(dec).__name__ if isinstance(dec, torch.nn.Module) else "-", v.latent_channels, v.latent_dim,
        v.spacial_compression_decode(), sum(p.numel() for p in fsm.parameters()) / 1e6))
    cases = [(1344, 768, 1), (3840, 2160, 1)] + ([(832, 480, 81)] if v.latent_dim == 3 else [])
    for w, h, fr in cases:
        s = latent(v, w, h, fr)
        print("{}   {}x{}{} latent {}: Monoload {}".format(indent, w, h, " x{} frames".format(fr) if fr > 1 else "", list(s), monoload_path(v, s)))


def detect(sd, prefix, meta):
    cfg = md.model_config_from_unet(sd, prefix, metadata=meta)
    if cfg is None:
        return None, None
    lf = getattr(cfg, "latent_format", None)
    if lf is not None and not isinstance(lf, type):
        lf = type(lf)   # the config holds an instance
    return cfg, lf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="a models directory to read instead of ComfyUI's folder paths")
    a = ap.parse_args()
    vaes = {}
    print("== models/vae")
    for name, path in files("vae", a.dir):
        if not name.endswith(".safetensors"):
            print(" ", name, "(not safetensors: skipped)")
            continue
        print(" ", name)
        try:
            sd, meta = read_header(path)
            v = build_vae(sd, meta)
            if v is None:
                print("   not a VAE comfy.sd.VAE recognizes")
                continue
            vaes[name] = (type(v.first_stage_model).__name__, v.latent_channels, v.latent_dim)
            describe_vae(v)
        except Exception as e:
            print("   failed: {}: {}".format(type(e).__name__, str(e)[:300]))
    print("== models/checkpoints")
    for name, path in files("checkpoints", a.dir):
        if not name.endswith(".safetensors"):
            continue
        print(" ", name)
        try:
            sd, meta = read_header(path)
            prefix = md.unet_prefix_from_state_dict(sd)
            cfg, lf = detect(sd, prefix, meta)
            print("   model: {} (latent format {})".format(type(cfg).__name__ if cfg else "not detected", lf.__name__ if lf else "-"))
            vsd = comfy.utils.state_dict_prefix_replace(sd, {"first_stage_model.": ""}, filter_keys=True)
            if not vsd:
                vsd = comfy.utils.state_dict_prefix_replace(sd, {"vae.": ""}, filter_keys=True)
            v = build_vae(vsd, meta) if vsd else None
            if v is None:
                print("   no built-in VAE")
            else:
                describe_vae(v)
        except Exception as e:
            print("   failed: {}: {}".format(type(e).__name__, str(e)[:300]))
    seen = set()
    for folder in ("diffusion_models", "unet"):
        print("== models/" + folder)
        for name, path in files(folder, a.dir):
            if not name.endswith(".safetensors"):
                continue
            if os.path.realpath(path) in seen:
                continue   # folder_paths lists models/unet under diffusion_models too
            seen.add(os.path.realpath(path))
            print(" ", name)
            try:
                sd, meta = read_header(path)
                prefix = md.unet_prefix_from_state_dict(sd)
                cfg, lf = detect(sd, prefix, meta)
                if cfg is None:
                    print("   model not detected (prefix {!r})".format(prefix))
                    continue
                inst = lf() if lf else None
                ch, dims = getattr(inst, "latent_channels", None), getattr(inst, "latent_dimensions", None)
                fits = [n for n, (cls, c, d) in vaes.items() if c == ch and d == dims]
                print("   model: {} | latent format {} ({} ch, {}D) | VAE files that fit: {}".format(
                    type(cfg).__name__, lf.__name__ if lf else "-", ch, dims, ", ".join(fits) or "none in models/vae"))
            except Exception as e:
                print("   failed: {}: {}".format(type(e).__name__, str(e)[:300]))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # skip interpreter teardown (the meta-built patchers' destructors print noise there)
