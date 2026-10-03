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

Layer 1 (stripe decoding for recognized decoders) plugs in through
STRIPE_ADAPTERS: the engine (monoload/vae_engine.py) plus one adapter per
decoder structure (monoload/vae_wan.py: the Wan 2.1 VAE single frame;
monoload/vae_ldm.py: the LDM Decoder of SD1.5 / SDXL / SD3 / Flux ae, with
GroupNorm statistics gathered across stripes). A recognized, self-tested
decoder is decoded in
stripes of output rows from a low-resolution checkpoint instead of the
whole-image activations; everything else keeps layer 2. Default stripe height:
the lowest peak that does not cost speed (DEFAULT_POLICY_ROWS, DESIGN
§9.13.4); MONOLOAD_VAE_STRIPE_ROWS overrides it. With MONOLOAD_VAE_BUDGET the
decode is the fastest whose estimate fits the budget: layer 2, or a layer-1
GroupNorm scheme x stripe height (choose_budget, DESIGN §9.14.10).

Switches (read at import; global defaults, a Monoload VAE Settings node
overrides them item by item for its VAE, monoload/settings.py): MONOLOAD=0,
MONOLOAD_DISABLE_VAE=1 or MONOLOAD_EXACT=1 -> native decode (the wrapper
passes the call to ComfyUI); MONOLOAD_VAE_WORKSPACE (default 1G);
MONOLOAD_DISABLE_VAE_STRIPE=1 -> layer 1 off; MONOLOAD_VAE_BUDGET (peak
budget: the fastest decode within it; unset = the default policy);
MONOLOAD_VAE_STRIPE_ROWS (force the stripe core height, for sweeps /
debugging); MONOLOAD_VAE_GN_SCHEME (force A / B / C / D: which GroupNorm
inputs an LDM decode keeps whole, DESIGN §9.14; default B). The forced
settings outrank the budget.
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

from . import settings, vae_engine, vae_ldm, vae_overrides, vae_wan
from .errors import MonoloadError, MonoloadVAEOOMError
from .vae_ops import GIB, MIB, OpChunking, OpStats, fmt_bytes

_ORIG = {}

DEFAULT_WORKSPACE = 1 * GIB
DEFAULT_POLICY_ROWS = 128  # layer-1 default: the peak of 128-row stripes, with the tallest stripes that stay at it (DESIGN §9.13.4)
LAYER1_WORKSPACE = 128 * MIB  # layer-1 workspace without MONOLOAD_VAE_BUDGET (capped by MONOLOAD_VAE_WORKSPACE); 384 MiB up to 5d668b6,
                              # 128 MiB after the CT 700 workspace experiment (DESIGN §9.13.11); OOM retries halve it to MIN_WORKSPACE
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


def _env_flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _budget_from_env():
    raw = os.environ.get("MONOLOAD_VAE_BUDGET", "").strip()
    if not raw:
        return None
    try:
        return parse_size(raw)
    except ValueError:
        logging.warning("[Monoload] MONOLOAD_VAE_BUDGET={!r} not understood (examples: 3G, 2560M); using the default stripe policy".format(raw))
        return None


def _rows_from_env():
    raw = os.environ.get("MONOLOAD_VAE_STRIPE_ROWS", "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
        if v < 1:
            raise ValueError(raw)
        return v
    except ValueError:
        logging.warning("[Monoload] MONOLOAD_VAE_STRIPE_ROWS={!r} is not a positive integer; ignored".format(raw))
        return None


def _gn_scheme_from_env():
    """(scheme, forced): MONOLOAD_VAE_GN_SCHEME, or the default (not forced)."""
    raw = os.environ.get("MONOLOAD_VAE_GN_SCHEME", "").strip().upper()
    if not raw:
        return vae_ldm.DEFAULT_SCHEME, False
    if raw not in vae_ldm.SCHEMES:
        logging.warning("[Monoload] MONOLOAD_VAE_GN_SCHEME={!r} is not one of {}; using {}".format(
            raw, "/".join(vae_ldm.SCHEMES), vae_ldm.DEFAULT_SCHEME))
        return vae_ldm.DEFAULT_SCHEME, False
    return raw, True


def _native_from_env():
    """The variable making the native decode the global default (besides MONOLOAD=0), or None."""
    for name in ("MONOLOAD_DISABLE_VAE", "MONOLOAD_EXACT"):
        if _env_flag(name):
            return name + "=1"
    return None


_GN_ENV = _gn_scheme_from_env()
_SETTINGS = {"native": _native_from_env(), "workspace": _workspace_from_env(), "budget": _budget_from_env(),
             "stripe": not _env_flag("MONOLOAD_DISABLE_VAE_STRIPE"), "stripe_rows": _rows_from_env(),
             "gn_scheme": _GN_ENV[0], "gn_forced": _GN_ENV[1]}
vae_ldm.set_scheme(_SETTINGS["gn_scheme"])
_LAST = {}


def workspace():
    return _SETTINGS["workspace"]


def set_workspace(n):
    """Workspace budget in bytes (tests / bench); MONOLOAD_VAE_WORKSPACE at import."""
    _SETTINGS["workspace"] = int(n)


def budget():
    return _SETTINGS["budget"]


def set_budget(n):
    """Peak budget in bytes, None = the default policy (tests / bench);
    MONOLOAD_VAE_BUDGET at import. With a budget, the fastest decode whose
    estimate fits it is used (choose_budget); none fitting is an error."""
    _SETTINGS["budget"] = int(n) if n else None


def layer1_workspace(bud=None):
    """Workspace of layer 1: a budget's eighth (at least MIN_WORKSPACE), else
    LAYER1_WORKSPACE; never above MONOLOAD_VAE_WORKSPACE."""
    return min(workspace(), max(MIN_WORKSPACE, bud // 8) if bud else LAYER1_WORKSPACE)


def global_mode():
    """The decode mode of a VAE without its own: ("native" / "layer2" / "auto",
    the variable that set it or None for the built-in default)."""
    if not settings.master():
        return "native", "MONOLOAD=0"
    if _SETTINGS["native"]:
        return "native", _SETTINGS["native"]
    if not _SETTINGS["stripe"]:
        return "layer2", "MONOLOAD_DISABLE_VAE_STRIPE=1"
    return "auto", None


def set_native(var):
    """Native decode as the global default, named after the variable that asks
    for it (e.g. "MONOLOAD_DISABLE_VAE=1"); None = off (tests). Read from
    MONOLOAD_DISABLE_VAE / MONOLOAD_EXACT at import."""
    _SETTINGS["native"] = var or None


def stripe_enabled():
    return _SETTINGS["stripe"]


def set_stripe(on):
    """Layer 1 on / off (tests / bench); MONOLOAD_DISABLE_VAE_STRIPE=1 at import."""
    _SETTINGS["stripe"] = bool(on)


def stripe_rows():
    return _SETTINGS["stripe_rows"]


def set_stripe_rows(n):
    """Force the stripe core height (output rows), None = from the policy / budget."""
    _SETTINGS["stripe_rows"] = int(n) if n else None


def gn_scheme():
    return _SETTINGS["gn_scheme"]


def gn_scheme_forced():
    return _SETTINGS["gn_forced"]


def set_gn_scheme(name):
    """Force the GroupNorm scheme of LDM layer-1 decodes (A / B / C / D), None =
    the default, not forced (tests / bench); MONOLOAD_VAE_GN_SCHEME at import.
    A forced scheme is used as is; otherwise MONOLOAD_VAE_BUDGET may pick another."""
    vae_ldm.set_scheme(name)
    _SETTINGS["gn_scheme"] = vae_ldm.scheme()
    _SETTINGS["gn_forced"] = bool(name)


# ---------------------------------------------------------------------------
# per-VAE settings (the Monoload VAE Settings node, monoload/vae_overrides.py)
# ---------------------------------------------------------------------------

def with_settings(vae, budget=0.0, gn_scheme="default", stripe_rows=0, mode="default"):
    """A copy of `vae` with its own decode settings (vae_overrides.with_settings)."""
    return vae_overrides.with_settings(vae, budget=budget, gn_scheme=gn_scheme, stripe_rows=stripe_rows, mode=mode)


def resolve_settings(vae):
    """The settings a decode of `vae` uses, item by item: the VAE's own (set by
    the node on a copy), else the environment (the global settings: the
    MONOLOAD_VAE_* variables, or the set_* functions of tests / bench), else
    Monoload's default. -> {budget, gn_scheme, gn_forced, stripe_rows, mode}
    and {item: "node" / "env" / "default"}."""
    own = vae_overrides.overrides(vae)
    eff, src = {}, {}
    if "budget" in own:
        eff["budget"], src["budget"] = own["budget"], "node"
    else:
        eff["budget"], src["budget"] = _SETTINGS["budget"], "env" if _SETTINGS["budget"] else "default"
    if "gn_scheme" in own:
        eff["gn_scheme"], eff["gn_forced"], src["gn_scheme"] = own["gn_scheme"], True, "node"
    else:
        eff["gn_scheme"], eff["gn_forced"] = _SETTINGS["gn_scheme"], _SETTINGS["gn_forced"]
        src["gn_scheme"] = "env" if _SETTINGS["gn_forced"] else "default"
    if "stripe_rows" in own:
        eff["stripe_rows"], src["stripe_rows"] = own["stripe_rows"], "node"
    else:
        eff["stripe_rows"], src["stripe_rows"] = _SETTINGS["stripe_rows"], "env" if _SETTINGS["stripe_rows"] else "default"
    if "mode" in own:
        eff["mode"], src["mode"] = own["mode"], "node"
    else:
        eff["mode"], var = global_mode()
        src["mode"] = "env" if var else "default"
        if var:
            eff["mode_env"] = var
    return eff, src


def settings_note(eff, src):
    """The decode log's account of its settings and where each came from."""
    scheme = eff["gn_scheme"] if eff["gn_forced"] or not eff["budget"] else "chosen by the budget"
    return "settings: budget {} ({}), GroupNorm scheme {} ({}), stripe rows {} ({}), mode {} ({})".format(
        fmt_bytes(eff["budget"]) if eff["budget"] else "unlimited" if src["budget"] == "node" else "none", src["budget"], scheme, src["gn_scheme"],
        eff["stripe_rows"] or "auto", src["stripe_rows"], {"layer2": "layer 2 only"}.get(eff["mode"], eff["mode"]),
        "env {}".format(eff["mode_env"]) if eff.get("mode_env") else src["mode"])


class _Applied:
    """The resolved settings in place of the global ones for one decode (the
    decode reads them from _SETTINGS / vae_ldm's scheme); restored on exit.
    ComfyUI runs one prompt at a time, so one decode at a time."""

    def __init__(self, eff, note):
        self.eff = eff
        self.note = note

    def __enter__(self):
        self.saved = dict(_SETTINGS), vae_ldm.scheme(), _NOTE[0]
        eff = self.eff
        _SETTINGS.update({"budget": eff["budget"], "stripe": eff["mode"] != "layer2", "stripe_rows": eff["stripe_rows"],
                          "gn_scheme": eff["gn_scheme"], "gn_forced": eff["gn_forced"]})
        vae_ldm.set_scheme(eff["gn_scheme"])
        _NOTE[0] = self.note
        return self

    def __exit__(self, *exc):
        settings, scheme, note = self.saved
        _SETTINGS.clear()
        _SETTINGS.update(settings)
        vae_ldm.set_scheme(scheme)
        _NOTE[0] = note
        return False


_NOTE = [""]   # the settings note of the decode in progress (for its log line and last_decode())


def last_decode():
    """What the last VAE.decode call did (strategy, estimate, budget, retries, OpStats dict)."""
    return dict(_LAST)


# ---------------------------------------------------------------------------
# layer 1 (phases 2/3): stripe decoding adapters for recognized decoders
# ---------------------------------------------------------------------------

# Each entry: a module / object with
#   match(vae, samples, vae_options) -> (bound, None) or (None, reason); by real structure, never by file name
# where bound is a vae_engine.StripeAdapter (the interface is in vae_engine's docstring):
#   bound.name, bound.key            label for logs, structure signature
#   vae_engine.self_test(bound, vae) -> (ok, detail); first use of a structure in this process, cached per
#                                       structure; not ok -> layer 1 off for it (loud warning), layer 2
#   bound.plan(vae, samples, budget, workspace, rows=None, out_bytes=0) -> plan (estimate, stripes, recompute,
#                                       ckpt_bytes, describe()) or None when the budget cannot be met
#   bound.run(vae, samples, plan, workspace, stats) -> pixel_samples (process_output applied), native layout
STRIPE_ADAPTERS = [vae_wan, vae_ldm]

_L1_NOTED = weakref.WeakKeyDictionary()   # first_stage_model -> last logged layer-1 reason


def _select_layer1(vae, samples, vae_options):
    """(bound, None) or (None, why layer 1 is not used)."""
    if not _SETTINGS["stripe"]:
        return None, "layer 1 disabled (MONOLOAD_DISABLE_VAE_STRIPE)"
    reasons = []
    for a in STRIPE_ADAPTERS:
        bound, why = a.match(vae, samples, vae_options)
        if bound is not None:
            return bound, None
        reasons.append(why)
    why = "; ".join(reasons) or "no layer-1 adapter"
    fsm = vae.first_stage_model
    try:
        if _L1_NOTED.get(fsm) != why:
            _L1_NOTED[fsm] = why
            logging.info("[Monoload] VAE layer 1 (stripes) not used for {}: {} -> layer 2".format(type(fsm).__name__, why))
    except TypeError:
        pass
    return None, why


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


def _sync():
    """Wait for the device, so the decode time in the log is the real one
    (kernels run asynchronously; without this the host-side clock stops
    long before the GPU is done)."""
    sync = getattr(mm, "synchronize", None)
    if sync is not None:
        sync()


def _native_estimate(vae, shape):
    try:
        return int(vae.memory_used_decode(shape, vae.vae_dtype))
    except Exception:
        return None


def _decode(self, samples_in, vae_options={}):
    eff, src = resolve_settings(self)
    note = settings_note(eff, src)
    if eff["mode"] == "native":
        # a global native default (MONOLOAD=0 ...) is ComfyUI's own decode, not worth a line per decode
        reason = "mode native (Monoload VAE Settings node)" if src["mode"] == "node" else "mode native ({})".format(eff.get("mode_env"))
        level = logging.INFO if src["mode"] == "node" else logging.DEBUG
    else:
        reason, level = _native_reason(self, samples_in), logging.INFO
    if reason is not None:
        logging.log(level, "[Monoload] VAE decode left native: {}; {}".format(reason, note))
        _LAST.clear()
        _LAST.update({"strategy": "native", "reason": reason, "settings": dict(eff), "settings_source": dict(src)})
        return _ORIG["decode"](self, samples_in, vae_options)
    with _Applied(eff, note):
        try:
            return _managed_decode(self, samples_in, vae_options)
        finally:
            _LAST["settings"], _LAST["settings_source"] = dict(eff), dict(src)


def _managed_decode(self, samples_in, vae_options):
    self.throw_exception_if_invalid()
    if self.latent_dim == 2 and samples_in.ndim == 5:
        samples_in = samples_in[:, :, 0]
    _sync()  # do not count work queued before this decode
    t0 = time.perf_counter()
    with mm.cuda_device_context(self.device):
        bud = budget()
        if bud:
            return _decode_budget(self, samples_in, vae_options, t0, bud)
        bound, why = _select_layer1(self, samples_in, vae_options)
        if bound is not None:
            ok, detail = _layer1_self_test(self, bound)
            if ok:
                return _decode_layer1(self, samples_in, bound, t0, detail)
            why = "layer-1 self-test failed ({})".format(detail)
        return _decode_layer2(self, samples_in, vae_options, t0, why)


def _layer1_self_test(vae, bound):
    hit = vae_engine._SELFTEST.get(bound.key)
    if hit is None:
        # the self-test copies the decoder's weights: they must be loaded
        mm.load_models_gpu([vae.patcher], memory_required=bound.selftest_memory(), force_full_load=vae.disable_offload)
        hit = vae_engine.self_test(bound, vae)
        if hit[0]:
            logging.info("[Monoload] VAE layer 1 ({}) self-test passed: {}".format(bound.name, hit[1]))
        else:
            logging.warning("[Monoload] !!!!!!!! VAE layer 1 ({}) SELF-TEST FAILED: {} !!!!!!!! layer 1 is disabled for this "
                            "decoder structure in this process; decoding with layer 2 (op-level chunking) instead. Please report this.".format(
                                bound.name, hit[1]))
    return hit


def choose_plan(vae, samples_in, bound, out_bytes):
    """(plan, budget, workspace, policy) for a layer-1 decode (DESIGN §9.13.4).

    forced rows   MONOLOAD_VAE_STRIPE_ROWS: exactly that core height.
    budget        MONOLOAD_VAE_BUDGET: the tallest stripes of this bound whose
                  estimate fits it; MonoloadError when no height does. (A
                  managed decode with a budget goes through choose_budget,
                  which also weighs layer 2 and the other variants; this
                  branch is for the simulator / tests.)
    default       the lowest peak that does not cost speed: the peak (the arena,
                  what is reserved) of DEFAULT_POLICY_ROWS-row stripes is the
                  target (shorter stripes were clearly slower on the hardware,
                  128 rows as fast as the tallest), and the tallest stripes whose
                  arena stays at that target are used. Where the whole-image prefix sets the peak
                  (large images) that is taller than DEFAULT_POLICY_ROWS at no
                  extra memory; a small image may become a single stripe.
    """
    bud = budget()
    ws = layer1_workspace(bud)
    forced = _SETTINGS["stripe_rows"]
    if forced:
        plan = bound.plan(vae, samples_in, bud or 0, ws, rows=forced, out_bytes=out_bytes)
        return plan, bud or plan.estimate, ws, "forced {} rows (MONOLOAD_VAE_STRIPE_ROWS){}".format(
            forced, "; estimate above MONOLOAD_VAE_BUDGET" if bud and plan.estimate > bud else "")
    if bud:
        plan = bound.plan(vae, samples_in, bud, ws, out_bytes=out_bytes)
        if plan is None:
            smallest = bound.smallest_plan(vae, samples_in, ws, out_bytes=out_bytes)
            raise MonoloadError(
                "[Monoload] VAE 第一层（条带解码）在峰值预算 MONOLOAD_VAE_BUDGET={} 内放不下：latent {} 最少也需要约 {}（{} 行的条带，"
                "前缀 {}、条带 {}）。请调大 MONOLOAD_VAE_BUDGET 或去掉它（用默认策略），或设 MONOLOAD_DISABLE_VAE_STRIPE=1 改走第二层。".format(
                    fmt_bytes(bud), list(samples_in.shape), fmt_bytes(smallest.estimate), max(b - a for a, b in smallest.stripes),
                    fmt_bytes(smallest.prefix_bytes), fmt_bytes(smallest.stripe_bytes)))
        return plan, bud, ws, "MONOLOAD_VAE_BUDGET"
    ref = bound.plan(vae, samples_in, 0, ws, rows=DEFAULT_POLICY_ROWS, out_bytes=out_bytes)
    target = ref.arena
    plan = bound.plan(vae, samples_in, target, ws, out_bytes=out_bytes, measure="arena") or ref
    return plan, target, ws, "default: peak of {}-row stripes".format(DEFAULT_POLICY_ROWS)


def _selftest_failed(bound):
    hit = vae_engine._SELFTEST.get(bound.key)
    return hit is not None and not hit[0]


def _candidate_label(c):
    if c["layer"] == 2:
        return "layer 2 {}".format(fmt_bytes(c["estimate"]))
    what = "layer 1{} {} rows (workspace {})".format(" scheme " + c["gn_scheme"] if c["gn_scheme"] else "", c["rows"], fmt_bytes(c["workspace"]))
    return "{} {}{}".format(what, fmt_bytes(c["estimate"]), ", ~{:.1f} s".format(c["seconds"]) if c["seconds"] is not None else "")


def choose_budget(vae, samples_in, vae_options, bud, selftest=None):
    """MONOLOAD_VAE_BUDGET: the fastest decode whose estimate (the bound handed to
    load_models_gpu) fits the budget (DESIGN §9.14.10). Returns a dict: layer
    (1 or 2), policy, why (the log line), candidates (what was considered) and
    for layer 1 bound / plan / workspace / selftest, for layer 2 estimate / probe
    / note; raises MonoloadError when nothing fits.

    candidates  layer 2 (workspace MONOLOAD_VAE_WORKSPACE), and for each layer-1
                variant (LDM: one per GroupNorm scheme; Wan: one) the tallest
                stripes whose estimate fits; layer-1 workspace layer1_workspace(budget).
    ranking     layer 2 first: it runs every conv once, layer 1 recomputes (CT 700
                4K: SDXL 9.9 s vs 35.5 s at best, Qwen 6.9 vs 8.4 s); then the
                layer-1 variants by bound.predict_seconds (LDM: the time model,
                vae_ldm.TIME_COEF).
    forced      outrank the budget: MONOLOAD_DISABLE_VAE_STRIPE -> layer 2;
                MONOLOAD_VAE_STRIPE_ROWS -> layer 1 at that height (the scheme is
                still chosen); MONOLOAD_VAE_GN_SCHEME -> layer 1 with that scheme
                for an LDM decoder (the height is still chosen). A forced
                configuration that does not fit runs anyway (logged).
    none fits   MonoloadError naming what each candidate needs.

    selftest(bound) -> (ok, detail): the layer-1 self-test (default: run it,
    cached per structure); a variant that fails it is skipped."""
    if selftest is None:
        selftest = lambda b: _layer1_self_test(vae, b)   # noqa: E731
    rows = _SETTINGS["stripe_rows"]
    scheme = gn_scheme() if gn_scheme_forced() else None
    head = "MONOLOAD_VAE_BUDGET {}".format(fmt_bytes(bud))
    considered = []

    def layer2(policy, why, note):
        est, probe = _layer2_estimate(vae, samples_in, vae_options, workspace())
        c = {"layer": 2, "estimate": est["total"], "fits": est["total"] <= bud, "seconds": None}
        considered.append(c)
        return c, {"layer": 2, "estimate": est, "probe": probe, "note": note, "candidates": considered,
                   "policy": "{}: {}{}".format(head, policy, "" if c["fits"] else "; estimate above the budget"),
                   "why": "{} -> layer 2 (estimate {}): {}".format(head, fmt_bytes(est["total"]), why)}

    bound, l1_note = _select_layer1(vae, samples_in, vae_options)
    if not _SETTINGS["stripe"]:
        return layer2("layer 2 forced (MONOLOAD_DISABLE_VAE_STRIPE)", "forced by MONOLOAD_DISABLE_VAE_STRIPE", l1_note)[1]
    variants = []
    if bound is not None:
        variants = [v for v in bound.variants(scheme if bound.schemes else None) if not _selftest_failed(v)]
        if not variants:
            l1_note = "layer-1 self-test failed"
    forced = []
    if variants and rows:
        forced.append("{} rows (MONOLOAD_VAE_STRIPE_ROWS)".format(rows))
    if variants and scheme and bound.schemes:
        forced.append("scheme {} (MONOLOAD_VAE_GN_SCHEME)".format(scheme))
    if not forced:
        c, d = layer2("layer 2 fits, the fastest candidate",
                      "fits, and layer 2 is the fastest (every conv once, no recompute){}".format(
                          "; layer 1 not available: {}".format(l1_note) if l1_note else ""), l1_note)
        if c["fits"]:
            return d

    # layer-1 workspace: the budget's (layer1_workspace), else LAYER1_WORKSPACE, else MIN_WORKSPACE (a larger
    # workspace raises the estimate: the budget's eighth can keep a variant out that fits with a smaller one)
    ws_opts = sorted({layer1_workspace(bud), min(workspace(), LAYER1_WORKSPACE), min(workspace(), MIN_WORKSPACE)}, reverse=True)
    fits, over = [], []
    for v in variants:
        outb = v.output_bytes(vae, samples_in)
        p = None
        for w in ws_opts:
            q = v.plan(vae, samples_in, bud, w, rows=rows, out_bytes=outb)
            if q is None or q.estimate > bud:
                continue
            if p is None:
                p = q
                if v.predict_seconds(q) is None:
                    break          # nothing to rank by: the largest workspace that fits
            elif v.predict_seconds(q) < v.predict_seconds(p):
                p = q              # a smaller workspace allows taller stripes (ties: the larger workspace)
        if p is None:
            p = (v.plan(vae, samples_in, bud, ws_opts[-1], rows=rows, out_bytes=outb) if rows
                 else v.smallest_plan(vae, samples_in, ws_opts[-1], out_bytes=outb))
        c = {"layer": 1, "adapter": v.name, "gn_scheme": getattr(v, "gn_scheme", None), "rows": max(b - a for a, b in p.stripes),
             "estimate": p.estimate, "seconds": v.predict_seconds(p), "fits": p.estimate <= bud, "workspace": p.workspace,
             "prefix": p.prefix_bytes, "stripes": p.stripe_bytes}
        considered.append(c)
        (fits if c["fits"] else over).append((v, p, c))
    fits.sort(key=lambda x: (x[2]["seconds"] is None, x[2]["seconds"] or 0.0))
    pick, over_budget = fits, False
    if not fits and rows and over:
        pick, over_budget = sorted(over, key=lambda x: x[1].estimate), True   # forced rows outrank the budget
    pre = "forced {}; ".format(", ".join(forced)) if forced else ""
    for v, p, c in pick:
        ok, detail = selftest(v)
        if not ok:
            c["selftest"] = "failed"
            continue
        if over_budget:
            reason = pre + "no variant fits the budget at that height, the smallest estimate is used"
        elif len(fits) > 1 or any(o["layer"] == 2 for o in considered):
            reason = pre + ("the fastest predicted that fits" if c["seconds"] is not None else "the candidate that fits")
        else:
            reason = pre + "the only candidate"
        others = "; ".join(_candidate_label(o) + ("" if o["fits"] else " (over)") for o in considered if o is not c) or "none"
        return {"layer": 1, "bound": v, "plan": p, "workspace": p.workspace, "selftest": detail, "candidates": considered,
                "policy": "{}: {}".format(head, "forced rows, estimate above the budget" if over_budget else "fastest within it"),
                "why": "{} -> {}: {}; others: {}".format(head, _candidate_label(c), reason, others)}
    if forced and pick:
        # every forced layer-1 variant that fits failed its self-test: layer 2, as without a budget
        return layer2("layer 2 (layer-1 self-test failed)", "layer-1 self-test failed", "layer-1 self-test failed")[1]
    needs = []
    for c in considered:
        if c["layer"] == 2:
            needs.append("第二层需要约 {}".format(fmt_bytes(c["estimate"])))
        else:
            needs.append("第一层{}用 {} 行的条带需要约 {}（前缀 {}、条带 {}）{}".format(
                "（GroupNorm 方案 {}）".format(c["gn_scheme"]) if c["gn_scheme"] else "", c["rows"], fmt_bytes(c["estimate"]),
                fmt_bytes(c["prefix"]), fmt_bytes(c["stripes"]), "，但自检未通过" if c.get("selftest") else ""))
    if not variants:
        needs.append("第一层不可用（{}）".format(l1_note))
    raise MonoloadError(
        "[Monoload] VAE 解码在峰值预算 MONOLOAD_VAE_BUDGET={} 内放不下（latent {}）：{}。请调大 MONOLOAD_VAE_BUDGET，"
        "或去掉它（用默认策略）。".format(fmt_bytes(bud), list(samples_in.shape), "；".join(needs)))


def _decode_budget(self, samples_in, vae_options, t0, bud):
    try:
        d = choose_budget(self, samples_in, vae_options, bud)
    except MonoloadError:
        _LAST.clear()
        _LAST.update({"strategy": "error", "budget": bud})
        raise
    logging.info("[Monoload] VAE " + d["why"] + ("; " + _NOTE[0] if _NOTE[0] else ""))
    if d["layer"] == 2:
        return _decode_layer2(self, samples_in, vae_options, t0, d["note"], est=d["estimate"], probe=d["probe"], policy=d["policy"],
                              considered=d["candidates"])
    return _decode_layer1(self, samples_in, d["bound"], t0, d["selftest"], choice=(d["plan"], bud, d["workspace"], d["policy"]),
                          considered=d["candidates"])


def _decode_layer1(self, samples_in, bound, t0, selftest, choice=None, considered=None):
    outb = bound.output_bytes(self, samples_in)
    forced = _SETTINGS["stripe_rows"]
    plan, bud, ws, policy = choice or choose_plan(self, samples_in, bound, outb)
    floor_ws = min(MIN_WORKSPACE, ws)
    min_rows = min(vae_engine.MIN_ROWS, max(b - a for a, b in plan.stripes))
    mm.load_models_gpu([self.patcher], memory_required=plan.estimate, force_full_load=self.disable_offload)
    retries = 0
    first_est = plan.estimate
    while True:
        stats = OpStats()
        oom = False
        try:
            pixel_samples = bound.run(self, samples_in, plan, ws, stats)
        except Exception as e:
            mm.raise_non_oom(e)
            oom = True
        if not oom:
            break
        pixel_samples = None
        mm.soft_empty_cache(True)
        rows = max(b - a for a, b in plan.stripes)
        if rows <= min_rows and ws <= floor_ws:
            raise MonoloadVAEOOMError(
                "[Monoload] VAE 解码显存不足：第一层（条带解码）的条带已缩到 {} 行、工作区 {}（共重试 {} 次）仍然 OOM。"
                "Monoload 不会退回到 tiled 近似解码，也不会退回第二层（第二层峰值更高）。可以先释放其他模型（/free）、降低分辨率，"
                "需要原生行为时设 MONOLOAD_DISABLE_VAE=1。latent {}，估算需要 {}。".format(
                    rows, fmt_bytes(ws), retries, list(samples_in.shape), fmt_bytes(plan.estimate)))
        rows = max(min_rows, rows // 2)
        ws = max(floor_ws, ws // 2)
        retries += 1
        plan = bound.plan(self, samples_in, bud, ws, rows=rows, out_bytes=outb)
        logging.warning("[Monoload] VAE decode ran out of memory; retrying layer 1 with {}-row stripes, workspace {} (retry {})".format(
            rows, fmt_bytes(ws), retries))

    pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
    _sync()
    dt = time.perf_counter() - t0
    native_est = _native_estimate(self, samples_in.shape)
    boundaries = [(plan.h_out, a) for a, _ in plan.stripes[1:]]
    _LAST.clear()
    _LAST.update({"strategy": "layer1", "adapter": bound.name, "estimate": {"total": plan.estimate, "first": first_est,
                  "prefix": plan.prefix_bytes, "stripes": plan.stripe_bytes, "checkpoint": plan.ckpt_bytes, "persistent": plan.persistent,
                  "live": plan.live_peak, "arena": plan.arena},
                  "native_estimate": native_est, "budget": bud, "policy": policy, "workspace": ws, "retries": retries, "seconds": dt,
                  "stripes": len(plan.stripes), "rows": max(b - a for a, b in plan.stripes), "recompute": plan.recompute,
                  "checkpoint_bytes": plan.ckpt_bytes, "boundaries": boundaries, "forced_rows": forced, "selftest": selftest,
                  "passes": len(plan.passes), "saves": [plan.save_bytes[p] for p in plan.saves], "gn_scheme": getattr(bound, "gn_scheme", None),
                  "pass_rows": [ps.rows for ps in plan.passes], "candidates": considered,
                  "predicted_seconds": bound.predict_seconds(plan), "stats": stats.as_dict()})
    logging.info("[Monoload] VAE decode {} -> layer 1 ({}): {}; {}, workspace {}{}; arena {}, memory estimate {} (native {}), {:.2f}s; {}".format(
        "x".join(str(d) for d in samples_in.shape), bound.name, plan.describe(), policy, fmt_bytes(ws),
        ", {} OOM retries".format(retries) if retries else "", fmt_bytes(stats.arena) if stats.arena else "none",
        fmt_bytes(plan.estimate), fmt_bytes(native_est), dt, _NOTE[0]))
    return pixel_samples


def _layer2_estimate(vae, samples_in, vae_options, ws):
    """(estimate dict, probe or None) of a layer-2 decode with workspace ws; the
    shape probe (first decode of this model and latent layout) loads the weights."""
    probe = _PROBES.get(vae.first_stage_model, {}).get((int(samples_in.shape[1]), samples_in.ndim))
    if probe is None:
        # the shape probe needs the weights where they compute
        mm.load_models_gpu([vae.patcher], memory_required=estimate(vae, samples_in[0:1, ..., :PROBE_SIZE, :PROBE_SIZE], ws)["total"],
                           force_full_load=vae.disable_offload)
        try:
            probe = _probe(vae, samples_in, vae_options)
        except Exception as e:
            # a real problem with this decoder will surface again in the decode itself
            logging.warning("[Monoload] VAE shape probe failed ({}: {}); memory estimate falls back to the widest conv at full resolution".format(type(e).__name__, e))
    return estimate(vae, samples_in, ws, probe), probe


def _decode_layer2(self, samples_in, vae_options, t0, l1_note, est=None, probe=None, policy=None, considered=None):
    budget = workspace()
    floor = min(MIN_WORKSPACE, budget)
    retries = 0
    stats = None
    with mm.cuda_device_context(self.device):
        if est is None:
            est, probe = _layer2_estimate(self, samples_in, vae_options, budget)
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
    _sync()
    dt = time.perf_counter() - t0
    native_est = _native_estimate(self, samples_in.shape)
    _LAST.clear()
    _LAST.update({"strategy": "layer2", "estimate": est, "native_estimate": native_est, "workspace": budget,
                  "retries": retries, "seconds": dt, "probe": probe is not None, "stats": stats.as_dict(), "layer1": l1_note,
                  "budget": _SETTINGS["budget"], "policy": policy, "candidates": considered})
    attn = ""
    if stats.attn_calls:
        attn = ", attention {} call(s) in query blocks of {} / {} tokens".format(stats.attn_calls, stats.attn_rows_min, stats.attn_tokens_max)
    if stats.attn_unmanaged:
        attn += ", attention left native: {}".format(", ".join(stats.attn_unmanaged[:4]))
    logging.info("[Monoload] VAE decode {} -> layer 2, op-level chunking (workspace {}{}): {} of {} conv call(s) in row blocks{}; "
                 "{}memory estimate {} (native {}), {:.2f}s; {}".format(
                     "x".join(str(d) for d in samples_in.shape), fmt_bytes(budget), ", {} OOM retries".format(retries) if retries else "",
                     stats.conv_chunked, stats.conv_calls, attn, policy + "; " if policy else "", fmt_bytes(est["total"]), fmt_bytes(native_est), dt,
                     _NOTE[0]))
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
        for name in ("CastBiasWeightContext", "CastWeightBiasOp", "run_every_op"):
            if getattr(comfy.ops, name, None) is None:
                return "comfy.ops.{} missing".format(name)
        if not issubclass(comfy.ops.disable_weight_init.GroupNorm, torch.nn.GroupNorm):
            return "comfy.ops GroupNorm is not a torch.nn.GroupNorm"
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
