"""Managed VAE decoding: lower, bounded memory peak, whole-image semantics.

install() wraps comfy.sd.VAE.decode (the class method, like hotpatch does for
ModelPatcher; the original is kept and uninstall() restores it). For an
image latent (4D, or 5D with T=1) the wrapper replaces what native decode does
around first_stage_model.decode:

  * memory: load_models_gpu() gets Monoload's own upper bound of what the
    managed decode needs (estimate(); docs/DESIGN.md §9.4) instead of
    memory_used_decode (on AMD: 2178 x latent area x 64 x dtype x 2.73, e.g.
    ~92 GiB for an SDXL 4K decode), so it does not unload other models for
    memory that is never used;
  * batch: one sample at a time;
  * operators: layer 2 of the design -- op-level chunking (monoload/vae_ops.py):
    convs in blocks of output rows bounded by the workspace budget, attention
    in blocks of queries over the whole K/V;
  * OOM: retried with half the workspace, down to MIN_WORKSPACE; then
    MonoloadVAEOOMError. decode_tiled_ (tile-local GroupNorm, an
    approximation) is never called;
  * output: device, dtype, process_output (to [0, 1], clamped) and the
    channels-last layout exactly as native.

Everything else -- multi-frame video latents (for now), 1D/audio latents, VAEs
with their own chunked output path (comfy_has_chunked_io), and explicit
VAEDecodeTiled / VAE.decode_tiled -- stays native (logged).

Layer 1 (stripe decoding for recognized decoders, phases 2 and 3) plugs in
through STRIPE_ADAPTERS; it is empty in phase 1.

Switches (read at import / by the plugin entry): MONOLOAD_DISABLE_VAE=1 or
MONOLOAD_EXACT=1 -> not installed; MONOLOAD_VAE_WORKSPACE (default 1G).
"""

import inspect
import logging
import os
import time
import weakref

import torch

import comfy.model_management as mm
import comfy.ops
import comfy.sd
from comfy.ldm.modules.diffusionmodules import model as ldm_model

from .errors import MonoloadVAEOOMError
from .vae_ops import GIB, MIB, OpChunking, OpStats, fmt_bytes

_ORIG = {}

DEFAULT_WORKSPACE = 1 * GIB
MIN_WORKSPACE = 64 * MIB
ACTIVATION_COPIES = 4      # live full-size activations bounded by 4x the largest one (DESIGN §9.4)
PROBE_SIZE = 8             # latent rows/cols of the shape probe


def parse_size(s):
    """'1G', '512M', '1.5g', '262144K', '1073741824B' or a plain number (MiB) -> bytes."""
    t = str(s).strip().lower()
    if t.endswith("ib"):
        t = t[:-2]
    elif t.endswith("b") and len(t) > 1 and not t[-2].isdigit():
        t = t[:-1]
    mult = MIB
    for suf, m in (("k", 1024), ("m", MIB), ("g", GIB), ("t", 1024 * GIB), ("b", 1)):
        if t.endswith(suf):
            t, mult = t[:-1], m
            break
    v = float(t)
    if not v > 0:
        raise ValueError(s)
    return int(v * mult)


def _workspace_from_env():
    raw = os.environ.get("MONOLOAD_VAE_WORKSPACE", "").strip()
    if not raw:
        return DEFAULT_WORKSPACE
    try:
        return parse_size(raw)
    except ValueError:
        logging.warning("[Monoload] MONOLOAD_VAE_WORKSPACE={!r} not understood (examples: 1G, 512M, 768); using {}".format(raw, fmt_bytes(DEFAULT_WORKSPACE)))
        return DEFAULT_WORKSPACE


_SETTINGS = {"workspace": _workspace_from_env()}
_LAST = {}


def workspace():
    return _SETTINGS["workspace"]


def set_workspace(n):
    """Workspace budget in bytes (tests / bench); MONOLOAD_VAE_WORKSPACE at import."""
    _SETTINGS["workspace"] = int(n)


def last_decode():
    """What the last VAE.decode call did (strategy, estimate, budget, retries, OpStats dict)."""
    return dict(_LAST)


# ---------------------------------------------------------------------------
# layer 1 (phases 2/3): stripe decoding adapters for recognized decoders
# ---------------------------------------------------------------------------

# Each entry: an object with
#   name                         short label for logs
#   match(first_stage_model)     -> adapter bound to this model, or None (by real structure, never by file name)
#   bound.self_test(vae)         first use per structure: small latent vs native; False -> layer 1 off for this VAE, loud warning
#   bound.estimate(vae, samples, budget) -> bytes
#   bound.decode(vae, sample, budget)    -> raw decoder output for one sample (before process_output)
# Phase 1 ships none: every image decode goes to layer 2.
STRIPE_ADAPTERS = []


def _select_layer1(vae):
    for a in STRIPE_ADAPTERS:
        bound = a.match(vae.first_stage_model)
        if bound is not None:
            return bound
    return None


# ---------------------------------------------------------------------------
# what is managed
# ---------------------------------------------------------------------------

def _native_reason(vae, samples):
    """None if this decode is managed, else why it stays native."""
    fsm = getattr(vae, "first_stage_model", None)
    if fsm is None:
        return "no VAE model"
    if getattr(samples, "is_nested", False):
        return "nested latent"
    if getattr(fsm, "comfy_has_chunked_io", False):
        return "{} decodes into its own preallocated output (comfy_has_chunked_io)".format(type(fsm).__name__)
    ld = getattr(vae, "latent_dim", 2)
    if ld == 2:
        if samples.ndim in (4, 5):
            return None  # native takes frame 0 of a 5D latent for a 2D VAE; so do we
        return "latent with {} dims for a 2D VAE".format(samples.ndim)
    if ld == 3:
        if samples.ndim == 5 and samples.shape[2] == 1:
            return None
        if samples.ndim == 5:
            return "multi-frame video latent (T={}): not managed yet, phase 1 covers images (4D, and 5D with T=1)".format(samples.shape[2])
        return "latent with {} dims for a 3D VAE".format(samples.ndim)
    return "latent_dim {} (audio / 1D) is not managed".format(ld)


# ---------------------------------------------------------------------------
# memory estimate
# ---------------------------------------------------------------------------

class _Probe:
    __slots__ = ("act_bytes_per_px", "out_numel_per_px", "seconds")

    def __init__(self, act, out, seconds):
        self.act_bytes_per_px = act
        self.out_numel_per_px = out
        self.seconds = seconds


_PROBES = weakref.WeakKeyDictionary()   # first_stage_model -> {(latent channels, ndim): _Probe}


def _latent_px(samples):
    return int(samples.shape[-2]) * int(samples.shape[-1])


def _probe(vae, samples, vae_options):
    """Largest single activation per latent pixel, from one decode of a
    PROBE_SIZE x PROBE_SIZE latent with forward hooks recording the largest
    tensor in or out of any module. The decoders handled here are fully
    convolutional with fixed scale factors, so activation sizes scale with the
    latent area. Cached per model and latent layout; the global RNG state is
    preserved."""
    fsm = vae.first_stage_model
    key = (int(samples.shape[1]), samples.ndim)
    per_model = _PROBES.setdefault(fsm, {})
    hit = per_model.get(key)
    if hit is not None:
        return hit
    shape = list(samples.shape)
    shape[0] = 1
    shape[-2] = shape[-1] = PROBE_SIZE
    biggest = [0]

    def scan(v):
        if isinstance(v, torch.Tensor):
            biggest[0] = max(biggest[0], v.numel() * v.element_size())
        elif isinstance(v, (list, tuple)):
            for x in v:
                scan(x)

    def hook(_m, args, out):
        scan(args)
        scan(out)

    handles = [m.register_forward_hook(hook) for m in fsm.modules()]
    t0 = time.perf_counter()
    devs = [vae.device.index or 0] if vae.device.type == "cuda" else []
    try:
        with torch.inference_mode(), torch.random.fork_rng(devices=devs):
            z = torch.zeros(shape, device=vae.device, dtype=vae.vae_dtype)
            out = fsm.decode(z, **vae_options)
            out_numel = out.numel() if isinstance(out, torch.Tensor) else sum(t.numel() for t in out)
            del out, z
    finally:
        for h in handles:
            h.remove()
    px = PROBE_SIZE * PROBE_SIZE
    p = _Probe(biggest[0] / px, out_numel / px, time.perf_counter() - t0)
    per_model[key] = p
    return p


def _static_act_bytes_per_px(vae):
    """Fallback when the probe fails: the widest conv at full output resolution."""
    cmax = 1
    for m in vae.first_stage_model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)):
            cmax = max(cmax, m.in_channels, m.out_channels)
    r = vae.spacial_compression_decode()
    return cmax * r * r * mm.dtype_size(vae.vae_dtype)


def _same_device(a, b):
    a, b = torch.device(a), torch.device(b)
    return a.type == b.type and (a.index or 0) == (b.index or 0)


def estimate(vae, samples, budget, probe=None):
    """Upper bound (bytes) of what one managed decode of `samples` adds on the
    VAE's device (docs/DESIGN.md §9.4):
        ACTIVATION_COPIES x largest activation of one sample
      + 2 x workspace         (conv columns or attention scores, plus the block copies around them)
      + output buffer         (the whole batch, intermediate dtype, if it lives on the same device)
      + one latent sample"""
    px = _latent_px(samples)
    if probe is not None:
        act = probe.act_bytes_per_px * px
        out_numel = probe.out_numel_per_px * px
    else:
        act = _static_act_bytes_per_px(vae) * px
        r = vae.spacial_compression_decode()
        out_numel = getattr(vae, "output_channels", 3) * r * r * px
    out_bytes = 0
    if _same_device(vae.output_device, vae.device):
        out_bytes = out_numel * samples.shape[0] * mm.dtype_size(vae.vae_output_dtype())
    lat = (samples[0:1].numel()) * mm.dtype_size(vae.vae_dtype)
    total = ACTIVATION_COPIES * act + 2 * budget + out_bytes + lat
    return {"total": int(total), "activation": int(act), "workspace": int(budget), "output": int(out_bytes), "latent": int(lat)}


# ---------------------------------------------------------------------------
# the decode
# ---------------------------------------------------------------------------

def _run(vae, samples_in, vae_options, budget, stats):
    """Layer 2: the original decoder forward under OpChunking, one sample at a
    time, written into the output buffer like native decode."""
    fsm = vae.first_stage_model
    pixel_samples = None
    n = samples_in.shape[0]
    with OpChunking(fsm, budget, stats):
        for i in range(n):
            samples = samples_in[i:i + 1].to(device=vae.device, dtype=vae.vae_dtype)
            out = fsm.decode(samples, **vae_options)
            del samples
            if pixel_samples is None:
                pixel_samples = torch.empty((n,) + tuple(out.shape[1:]), device=vae.output_device, dtype=vae.vae_output_dtype())
            pixel_samples[i:i + 1].copy_(out)   # = native's .to(output_device, output dtype, copy=True) + copy_
            del out
            vae.process_output(pixel_samples[i:i + 1])
    return pixel_samples


def _native_estimate(vae, shape):
    try:
        return int(vae.memory_used_decode(shape, vae.vae_dtype))
    except Exception:
        return None


def _decode(self, samples_in, vae_options={}):
    reason = _native_reason(self, samples_in)
    if reason is not None:
        logging.info("[Monoload] VAE decode left native: {}".format(reason))
        _LAST.clear()
        _LAST.update({"strategy": "native", "reason": reason})
        return _ORIG["decode"](self, samples_in, vae_options)
    return _managed_decode(self, samples_in, vae_options)


def _managed_decode(self, samples_in, vae_options):
    self.throw_exception_if_invalid()
    if self.latent_dim == 2 and samples_in.ndim == 5:
        samples_in = samples_in[:, :, 0]
    t0 = time.perf_counter()
    budget = workspace()
    floor = min(MIN_WORKSPACE, budget)
    retries = 0
    stats = None
    with mm.cuda_device_context(self.device):
        probe = None
        cached = _PROBES.get(self.first_stage_model, {}).get((int(samples_in.shape[1]), samples_in.ndim))
        if cached is None:
            # the shape probe needs the weights where they compute
            mm.load_models_gpu([self.patcher], memory_required=estimate(self, samples_in[0:1, ..., :PROBE_SIZE, :PROBE_SIZE], budget)["total"],
                               force_full_load=self.disable_offload)
            try:
                probe = _probe(self, samples_in, vae_options)
            except Exception as e:
                # a real problem with this decoder will surface again in the decode itself
                logging.warning("[Monoload] VAE shape probe failed ({}: {}); memory estimate falls back to the widest conv at full resolution".format(type(e).__name__, e))
        else:
            probe = cached
        est = estimate(self, samples_in, budget, probe)
        mm.load_models_gpu([self.patcher], memory_required=est["total"], force_full_load=self.disable_offload)
        while True:
            stats = OpStats()
            oom = False
            try:
                pixel_samples = _run(self, samples_in, vae_options, budget, stats)
            except Exception as e:
                mm.raise_non_oom(e)
                oom = True
            if not oom:
                break
            # out of the except block: the traceback (and the tensors its frames hold) is gone
            pixel_samples = None
            mm.soft_empty_cache(True)
            if budget <= floor:
                raise MonoloadVAEOOMError(
                    "[Monoload] VAE 解码显存不足：工作区已缩到下限 {}（共重试 {} 次）仍然 OOM。"
                    "Monoload 不会退回到 tiled 近似解码（decode_tiled_）。可以先释放其他模型（/free）、降低分辨率，"
                    "需要原生行为时设 MONOLOAD_DISABLE_VAE=1。latent {}，估算需要 {}。".format(
                        fmt_bytes(budget), retries, list(samples_in.shape), fmt_bytes(est["total"])))
            budget = max(floor, budget // 2)
            retries += 1
            logging.warning("[Monoload] VAE decode ran out of memory; retrying with workspace {} (retry {})".format(fmt_bytes(budget), retries))

    pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
    dt = time.perf_counter() - t0
    native_est = _native_estimate(self, samples_in.shape)
    _LAST.clear()
    _LAST.update({"strategy": "layer2", "estimate": est, "native_estimate": native_est, "workspace": budget,
                  "retries": retries, "seconds": dt, "probe": probe is not None, "stats": stats.as_dict()})
    attn = ""
    if stats.attn_calls:
        attn = ", attention {} call(s) in query blocks of {} / {} tokens".format(stats.attn_calls, stats.attn_rows_min, stats.attn_tokens_max)
    if stats.attn_unmanaged:
        attn += ", attention left native: {}".format(", ".join(stats.attn_unmanaged[:4]))
    logging.info("[Monoload] VAE decode {} -> op-level chunking (workspace {}{}): {} of {} conv call(s) in row blocks{}; "
                 "memory estimate {} (native {}), {:.2f}s".format(
                     "x".join(str(d) for d in samples_in.shape), fmt_bytes(budget), ", {} OOM retries".format(retries) if retries else "",
                     stats.conv_chunked, stats.conv_calls, attn, fmt_bytes(est["total"]), fmt_bytes(native_est), dt))
    return pixel_samples


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def _params(fn):
    return list(inspect.signature(fn).parameters)


def _check_api():
    """None if the ComfyUI pieces this module relies on look as expected,
    else what is different."""
    try:
        if _params(comfy.sd.VAE.decode) != ["self", "samples_in", "vae_options"]:
            return "comfy.sd.VAE.decode{}".format(inspect.signature(comfy.sd.VAE.decode))
        if _params(comfy.ops.disable_weight_init.Conv3d._conv_forward)[:4] != ["self", "input", "weight", "bias"]:
            return "comfy.ops Conv3d._conv_forward{}".format(inspect.signature(comfy.ops.disable_weight_init.Conv3d._conv_forward))
        if _params(torch.nn.Conv2d._conv_forward)[:4] != ["self", "input", "weight", "bias"]:
            return "torch.nn.Conv2d._conv_forward{}".format(inspect.signature(torch.nn.Conv2d._conv_forward))
        if "memory_required" not in _params(mm.load_models_gpu):
            return "load_models_gpu{}".format(inspect.signature(mm.load_models_gpu))
        for name in ("normal_attention", "pytorch_attention", "xformers_attention", "vae_attention"):
            if not callable(getattr(ldm_model, name, None)):
                return "comfy.ldm.modules.diffusionmodules.model.{} missing".format(name)
        for name in ("raise_non_oom", "cuda_device_context", "soft_empty_cache", "dtype_size"):
            if not callable(getattr(mm, name, None)):
                return "comfy.model_management.{} missing".format(name)
    except (TypeError, ValueError) as e:
        return str(e)
    return None


def is_installed():
    return bool(_ORIG)


def install():
    if _ORIG:
        return False
    bad = _check_api()
    if bad is not None:
        logging.warning("[Monoload] VAE decode NOT managed: ComfyUI API differs from what Monoload was written for ({}); VAE stays native".format(bad))
        return False
    _ORIG["decode"] = comfy.sd.VAE.__dict__["decode"]
    _decode.__wrapped__ = _ORIG["decode"]
    comfy.sd.VAE.decode = _decode
    return True


def uninstall():
    if not _ORIG:
        return False
    comfy.sd.VAE.decode = _ORIG.pop("decode")
    return True
