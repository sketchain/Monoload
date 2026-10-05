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

Everything else -- multi-frame video latents (for now), audio VAEs (1D latents,
and 2D ones such as ACE-Step / LTX audio / MiniMax audio), VAEs with their own
chunked output path (comfy_has_chunked_io), decoders that mix frames across the
batch (SVD's VideoDecoder: a sample-by-sample decode would change the result),
models with nothing layer 2 chunks (the pixel-space VAE), and explicit
VAEDecodeTiled / VAE.decode_tiled -- stays native (logged).

Layer 1 (stripe decoding for recognized decoders) plugs in through
STRIPE_ADAPTERS: the engine (monoload/vae_engine.py) plus one adapter per
decoder structure (monoload/vae_wan.py: the Wan 2.1 VAE single frame;
monoload/vae_ldm.py: the LDM Decoder of SD1.5 / SDXL / SD3 / Flux ae / Flux 2,
with GroupNorm statistics gathered across stripes). A recognized, self-tested
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

import glob
import inspect
import logging
import os
import re
import threading
import time
import weakref

import torch

import comfy.model_management as mm
import comfy.ops
import comfy.sd
from comfy.ldm.modules.diffusionmodules import model as ldm_model

from . import settings, vae_engine, vae_ldm, vae_ops, vae_overrides, vae_wan
from .errors import MonoloadError, MonoloadVAEOOMError
from .messages import label, msg
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
        logging.warning(msg("vae.env_workspace", raw=raw, default=fmt_bytes(DEFAULT_WORKSPACE)))
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
        logging.warning(msg("vae.env_budget", raw=raw))
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
        logging.warning(msg("vae.env_rows", raw=raw))
        return None


def _gn_scheme_from_env():
    """(scheme, forced): MONOLOAD_VAE_GN_SCHEME, or the default (not forced)."""
    raw = os.environ.get("MONOLOAD_VAE_GN_SCHEME", "").strip().upper()
    if not raw:
        return vae_ldm.DEFAULT_SCHEME, False
    if raw not in vae_ldm.SCHEMES:
        logging.warning(msg("vae.env_scheme", raw=raw, schemes="/".join(vae_ldm.SCHEMES), default=vae_ldm.DEFAULT_SCHEME))
        return vae_ldm.DEFAULT_SCHEME, False
    return raw, True


def _native_from_env():
    """The variable making the native decode the global default (besides MONOLOAD=0), or None."""
    for name in ("MONOLOAD_DISABLE_VAE", "MONOLOAD_EXACT"):
        if _env_flag(name):
            return name + "=1"
    return None


_GN_ENV = _gn_scheme_from_env()
_SETTINGS = {"src": {}, "native": _native_from_env(), "workspace": _workspace_from_env(), "budget": _budget_from_env(),
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
    scheme = eff["gn_scheme"] if eff["gn_forced"] or not eff["budget"] else msg("vae.by_budget")
    return msg("vae.settings_note",
               budget=fmt_bytes(eff["budget"]) if eff["budget"] else msg("vae.unlimited") if src["budget"] == "node" else msg("vae.none"),
               budget_src=label(src["budget"]), scheme=scheme, scheme_src=label(src["gn_scheme"]), rows=eff["stripe_rows"] or msg("vae.auto"),
               rows_src=label(src["stripe_rows"]), mode=msg("vae.layer2_only") if eff["mode"] == "layer2" else label(eff["mode"]),
               mode_src="{} {}".format(label("env"), eff["mode_env"]) if eff.get("mode_env") else label(src["mode"]))


class _Applied:
    """The resolved settings in place of the global ones for one decode (the
    decode reads them from _SETTINGS / vae_ldm's scheme); restored on exit.
    ComfyUI runs one prompt at a time, so one decode at a time."""

    def __init__(self, eff, note, src=None):
        self.eff = eff
        self.note = note
        self.src = src or {}

    def __enter__(self):
        self.saved = dict(_SETTINGS), vae_ldm.scheme(), _NOTE[0]
        eff = self.eff
        _SETTINGS.update({"src": dict(self.src), "budget": eff["budget"], "stripe": eff["mode"] != "layer2", "stripe_rows": eff["stripe_rows"],
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


_RECORDS = weakref.WeakKeyDictionary()   # VAE object -> the record of its last decode (the Monoload Info node)


def decode_record(vae):
    """The record of the last decode of this VAE object (a copy made by the
    Monoload VAE Settings node is its own object), or None: last_decode()'s
    fields plus "mem" (measured: reserved_peak / gtt_peak increases in bytes,
    None where not measurable) and "when" (time.time())."""
    try:
        r = _RECORDS.get(vae)
    except TypeError:
        return None
    return dict(r) if r is not None else None


def _record(vae, mem=None):
    try:
        _RECORDS[vae] = dict(_LAST, mem=mem, when=time.time())
    except TypeError:
        pass


def _gtt_files():
    out = {}
    for d in glob.glob("/sys/class/drm/card*"):
        if re.search(r"/card\d+$", d):
            p = os.path.join(d, "device", "mem_info_gtt_used")
            if os.path.exists(p):
                out.setdefault(os.path.realpath(os.path.join(d, "device")), p)
    return list(out.values())


def _read_gtt(files):
    try:
        return sum(int(open(p).read()) for p in files)
    except (OSError, ValueError):
        return None


class _MemProbe:
    """Measured memory of one managed decode: the increase of torch's peak
    reserved memory on the VAE's CUDA / ROCm device, and the peak increase of
    amdgpu GTT used (sampled every 20 ms by a thread, where the sysfs file
    exists). Nothing is measured on the CPU.

    The allocator's cache is emptied first (soft_empty_cache): blocks left
    cached by earlier work (e.g. the sampler's activations) would otherwise
    count in the starting point and be reused by the decode, so the increase
    understated its footprint (CT 700: +1.25 GiB measured on a first decode
    whose arena alone is 1.97 GiB). selftest: whether the first-use layer-1
    self-test ran inside this decode (its time and memory are included)."""

    def __init__(self, device):
        self.device = device
        self.result = {"reserved_peak": None, "gtt_peak": None, "selftest": False}

    def __enter__(self):
        dev = self.device
        self.cuda = getattr(dev, "type", None) == "cuda" and torch.cuda.is_available()
        self.selftests = len(vae_engine._SELFTEST)
        if self.cuda:
            mm.soft_empty_cache()
            self.base_res = torch.cuda.memory_reserved(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        self.files = _gtt_files() if self.cuda else []
        self.thread = None
        if self.files:
            self.base_gtt = self.peak_gtt = _read_gtt(self.files)
            self.stop = threading.Event()
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.02):
            v = _read_gtt(self.files)
            if v is not None and (self.peak_gtt is None or v > self.peak_gtt):
                self.peak_gtt = v

    def __exit__(self, *exc):
        if self.thread is not None:
            self.stop.set()
            self.thread.join()
            v = _read_gtt(self.files)
            if v is not None and self.peak_gtt is not None:
                self.peak_gtt = max(self.peak_gtt, v)
            if self.base_gtt is not None and self.peak_gtt is not None:
                self.result["gtt_peak"] = self.peak_gtt - self.base_gtt
        if self.cuda:
            self.result["reserved_peak"] = torch.cuda.max_memory_reserved(self.device) - self.base_res
        self.result["selftest"] = len(vae_engine._SELFTEST) > self.selftests
        return False


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


_ENV_VARS = {"budget": "MONOLOAD_VAE_BUDGET", "stripe_rows": "MONOLOAD_VAE_STRIPE_ROWS", "gn_scheme": "MONOLOAD_VAE_GN_SCHEME",
             "mode": "MONOLOAD_DISABLE_VAE_STRIPE=1"}


def _from(item):
    """Where the decode in progress got `item` from, for messages: the node or the environment variable."""
    if _SETTINGS["src"].get(item) == "node":
        return msg("vae.src_node")
    return msg("vae.src_env", var=_ENV_VARS[item])


def _budget_advice(layer2=False):
    node = _SETTINGS["src"].get("budget") == "node"
    a = msg("vae.advice_node") if node else msg("vae.advice_env")
    if layer2:
        a += msg("vae.advice_l2_node") if node else msg("vae.advice_l2_env")
    return a


def _select_layer1(vae, samples, vae_options):
    """(bound, None) or (None, why layer 1 is not used)."""
    if not _SETTINGS["stripe"]:
        return None, msg("vae.l1_disabled", src=_from("mode"))
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
            logging.info(msg("vae.l1_not_used", model=type(fsm).__name__, why=why))
    except TypeError:
        pass
    return None, why


# ---------------------------------------------------------------------------
# what is managed
# ---------------------------------------------------------------------------

AUDIO_MIN_RATIO = 64   # an upscale ratio above this is latent frames -> audio samples (images: 1 / 4 / 8 / 16 / 32)


def _batch_time_modules():
    """Module classes that mix frames across the batch (the batch is their time axis): SVD's VideoDecoder."""
    try:
        from comfy.ldm.modules import temporal_ae
    except Exception:
        return ()
    return tuple(c for c in (getattr(temporal_ae, n, None) for n in ("VideoResBlock", "AE3DConv", "AttnVideoBlock")) if isinstance(c, type))


_TRAITS = weakref.WeakKeyDictionary()   # first_stage_model -> (mixes frames across the batch, has a conv / known attention)


def _model_traits(fsm):
    """(mixes frames across the batch, has something layer 2 chunks), from the model's modules (cached per model)."""
    try:
        hit = _TRAITS.get(fsm)
    except TypeError:
        hit = None
    if hit is not None:
        return hit
    batch_time = _batch_time_modules()
    known = vae_ops.known_attention()
    mixes = ops = False
    modules = fsm.modules() if isinstance(fsm, torch.nn.Module) else ()
    for m in modules:
        if batch_time and isinstance(m, batch_time):
            mixes = True
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)) or m.__dict__.get("optimized_attention") in known:
            ops = True
    hit = (mixes, ops)
    try:
        _TRAITS[fsm] = hit
    except TypeError:
        pass
    return hit


def _audio_ratio(vae):
    """The latent -> samples ratio of an audio VAE with a 2D latent, else None.
    ComfyUI marks most of them with extra_1d_channel (ACE-Step, LTX audio); the
    others (MiniMax H3 audio) have an upscale ratio no image VAE has."""
    r = getattr(vae, "upscale_ratio", None)
    if getattr(vae, "extra_1d_channel", None) is not None:
        return r if isinstance(r, (int, float)) else "?"
    if isinstance(r, (int, float)) and not isinstance(r, bool) and r > AUDIO_MIN_RATIO:
        return r
    return None


def _native_reason(vae, samples):
    """None if this decode is managed, else why it stays native."""
    fsm = getattr(vae, "first_stage_model", None)
    if fsm is None:
        return msg("vae.nr_no_model")
    if getattr(samples, "is_nested", False):
        return msg("vae.nr_nested")
    model = type(fsm).__name__
    if getattr(fsm, "comfy_has_chunked_io", False):
        return msg("vae.nr_chunked_io", model=model)
    ld = getattr(vae, "latent_dim", 2)
    if ld not in (2, 3):
        return msg("vae.nr_1d", ld=ld)
    ratio = _audio_ratio(vae)
    if ratio is not None:
        return msg("vae.nr_audio", model=model, ndim=samples.ndim, ratio=ratio)
    if ld == 2 and samples.ndim not in (4, 5):   # native takes frame 0 of a 5D latent for a 2D VAE; so do we
        return msg("vae.nr_dims", ndim=samples.ndim, ld=ld)
    if ld == 3:
        if samples.ndim != 5:
            return msg("vae.nr_dims", ndim=samples.ndim, ld=ld)
        if samples.shape[2] != 1:
            return msg("vae.nr_multiframe", t=samples.shape[2])
    mixes, ops = _model_traits(fsm)
    if mixes:
        return msg("vae.nr_batch_time", model=model)
    if not ops:
        return msg("vae.nr_no_ops", model=model)
    return None


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
        reason = msg("vae.native_node") if src["mode"] == "node" else msg("vae.native_global", var=eff.get("mode_env"))
        level = logging.INFO if src["mode"] == "node" else logging.DEBUG
    else:
        reason, level = _native_reason(self, samples_in), logging.INFO
    if reason is not None:
        logging.log(level, msg("vae.left_native", reason=reason, note=note))
        _LAST.clear()
        _LAST.update({"strategy": "native", "reason": reason, "settings": dict(eff), "settings_source": dict(src)})
        t0 = time.perf_counter()
        try:
            out = _ORIG["decode"](self, samples_in, vae_options)
        except Exception as e:
            _error_record(e)
            _LAST["settings"], _LAST["settings_source"] = dict(eff), dict(src)
            _record(self)
            raise
        _LAST["seconds"] = time.perf_counter() - t0
        _record(self)
        return out
    probe = _MemProbe(getattr(self, "device", None))
    # this call's record starts empty: a decode that fails before writing its own fields must not leave the fields of
    # the previous decode (of any VAE) in its record
    _LAST.clear()
    with _Applied(eff, note, src):
        try:
            with probe:
                return _managed_decode(self, samples_in, vae_options)
        except Exception as e:
            _error_record(e)
            raise
        finally:
            _LAST["settings"], _LAST["settings_source"] = dict(eff), dict(src)
            _record(self, probe.result)


def _error_record(e):
    """The record of a decode that raised e: strategy "error", kind "budget"
    (nothing fits the budget, written by _decode_budget), "oom" or "other",
    and the error's first line. Fields of a run that got further (strategy,
    plan) are dropped."""
    if _LAST.get("strategy") != "error":
        kind = "oom" if isinstance(e, MonoloadVAEOOMError) or mm.is_oom(e) else "other"
        _LAST.clear()
        _LAST.update({"strategy": "error", "kind": kind, "budget": budget()})
    text = str(e).strip().splitlines()
    _LAST["error"] = "{}: {}".format(type(e).__name__, text[0] if text else "")


_SELFTEST_IN_DECODE = [0]   # selftest_memory() of a first-use self-test run inside the decode in progress (0: none)


def _managed_decode(self, samples_in, vae_options):
    _SELFTEST_IN_DECODE[0] = 0
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
        need = bound.selftest_memory()
        mm.load_models_gpu([vae.patcher], memory_required=need, force_full_load=vae.disable_offload)
        hit = vae_engine.self_test(bound, vae)
        _SELFTEST_IN_DECODE[0] = max(_SELFTEST_IN_DECODE[0], need)
        if hit[0]:
            logging.info(msg("vae.selftest_ok", name=bound.name, detail=hit[1]))
        else:
            logging.warning(msg("vae.selftest_failed", name=bound.name, detail=hit[1]))
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
        return plan, bud or plan.estimate, ws, msg("vae.policy_forced_rows", rows=forced, src=_from("stripe_rows"),
                                                   over=msg("vae.over_budget_note") if bud and plan.estimate > bud else "")
    if bud:
        plan = bound.plan(vae, samples_in, bud, ws, out_bytes=out_bytes)
        if plan is None:
            smallest = bound.smallest_plan(vae, samples_in, ws, out_bytes=out_bytes)
            raise MonoloadError(msg("vae.err_l1_budget", budget=fmt_bytes(bud), src=_from("budget"), advice=_budget_advice(layer2=True),
                                    shape=list(samples_in.shape), need=fmt_bytes(smallest.estimate),
                                    rows=max(b - a for a, b in smallest.stripes), prefix=fmt_bytes(smallest.prefix_bytes),
                                    stripes=fmt_bytes(smallest.stripe_bytes)))
        return plan, bud, ws, msg("vae.budget_head", budget=fmt_bytes(bud), src=_from("budget"))
    ref = bound.plan(vae, samples_in, 0, ws, rows=DEFAULT_POLICY_ROWS, out_bytes=out_bytes)
    target = ref.arena
    plan = bound.plan(vae, samples_in, target, ws, out_bytes=out_bytes, measure="arena") or ref
    return plan, target, ws, msg("vae.policy_default", rows=DEFAULT_POLICY_ROWS)


def _selftest_failed(bound):
    hit = vae_engine._SELFTEST.get(bound.key)
    return hit is not None and not hit[0]


def _selftest_pending(bound):
    """What the first-use self-test of this structure will reserve before the decode, 0 once it has run (DESIGN §9.20)."""
    return 0 if bound.key in vae_engine._SELFTEST else bound.selftest_memory()


def _selftest_note(c):
    return msg("vae.cand_selftest", st=fmt_bytes(c["selftest"]), plan=fmt_bytes(c["plan_estimate"])) if c.get("selftest") else ""


def _candidate_label(c):
    if c["layer"] == 2:
        return msg("vae.cand_layer2", est=fmt_bytes(c["estimate"]))
    return msg("vae.cand_layer1", scheme=msg("vae.cand_scheme", scheme=c["gn_scheme"]) if c["gn_scheme"] else "", rows=c["rows"],
               ws=fmt_bytes(c["workspace"]), est=fmt_bytes(c["estimate"]) + _selftest_note(c),
               secs=msg("vae.cand_secs", secs=c["seconds"]) if c["seconds"] is not None else "")


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
    forced      each has its own relation to the budget (DESIGN §9.14.10):
                  layer 2 only (MONOLOAD_DISABLE_VAE_STRIPE / node mode): layer
                    2, also when its estimate is above the budget (logged);
                  fixed height (MONOLOAD_VAE_STRIPE_ROWS): layer 1 at that
                    height, layer 2 not considered; the scheme is the fastest
                    that fits at that height, and when none fits the one with
                    the lowest estimate runs anyway (logged);
                  fixed scheme (MONOLOAD_VAE_GN_SCHEME, LDM): layer 1 with that
                    scheme, layer 2 not considered, the tallest stripes that
                    fit; when no height fits: MonoloadError (it does not run
                    over the budget, and does not fall back to layer 2);
                  fixed height and scheme: that configuration, run anyway.
                When the forced layer-1 configuration cannot run because its
                self-test failed (in this decode or cached from an earlier one,
                the same decision): layer 2 if it fits the budget, else
                MonoloadError naming the forced setting, the variant(s) that
                failed and what layer 2 needs.
    none fits   MonoloadError naming what each candidate needs.

    selftest(bound) -> (ok, detail): the layer-1 self-test (default: run it,
    cached per structure); a variant that fails it is skipped."""
    if selftest is None:
        selftest = lambda b: _layer1_self_test(vae, b)   # noqa: E731
    rows = _SETTINGS["stripe_rows"]
    scheme = gn_scheme() if gn_scheme_forced() else None
    head = msg("vae.budget_head", budget=fmt_bytes(bud), src=_from("budget"))
    considered = []

    def layer2(policy, why, note):
        est, probe = _layer2_estimate(vae, samples_in, vae_options, workspace())
        c = {"layer": 2, "estimate": est["total"], "fits": est["total"] <= bud, "seconds": None}
        considered.append(c)
        return c, {"layer": 2, "estimate": est, "probe": probe, "note": note, "candidates": considered,
                   "policy": msg("vae.policy_join", head=head, policy=policy, over="" if c["fits"] else msg("vae.est_above")),
                   "why": msg("vae.why_layer2", head=head, est=fmt_bytes(est["total"]), why=why)}

    bound, l1_note = _select_layer1(vae, samples_in, vae_options)
    if not _SETTINGS["stripe"]:
        return layer2(msg("vae.l2_forced_policy", src=_from("mode")), msg("vae.l2_forced_why", src=_from("mode")), l1_note)[1]
    # a forced layer-1 configuration is read from the effective settings, not from the variants left after failed
    # self-tests: the decision must not depend on whether a failure was cached by an earlier decode (review 02)
    forced = []
    if bound is not None and rows:
        forced.append(msg("vae.forced_rows", rows=rows, src=_from("stripe_rows")))
    if bound is not None and scheme and bound.schemes:
        forced.append(msg("vae.forced_scheme", scheme=scheme, src=_from("gn_scheme")))
    variants, failed = [], []
    if bound is not None:
        for v in bound.variants(scheme if bound.schemes else None):
            (failed if _selftest_failed(v) else variants).append(v)
        if not variants:
            l1_note = msg("vae.selftest_failed_short")
    if not forced:
        c, d = layer2(msg("vae.l2_fits_policy"),
                      msg("vae.l2_fits_why", l1=msg("vae.l1_unavailable_note", why=l1_note) if l1_note else ""), l1_note)
        if c["fits"]:
            return d

    # layer-1 workspace: the budget's (layer1_workspace), else LAYER1_WORKSPACE, else MIN_WORKSPACE (a larger
    # workspace raises the estimate: the budget's eighth can keep a variant out that fits with a smaller one)
    ws_opts = sorted({layer1_workspace(bud), min(workspace(), LAYER1_WORKSPACE), min(workspace(), MIN_WORKSPACE)}, reverse=True)
    fits, over = [], []
    for v in variants:
        outb = v.output_bytes(vae, samples_in)
        st = _selftest_pending(v)   # a first use runs the self-test before the decode: it must fit the budget too
        p = None
        for w in ws_opts:
            q = v.plan(vae, samples_in, bud, w, rows=rows, out_bytes=outb)
            if q is None or max(q.estimate, st) > bud:
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
        est = max(p.estimate, st)
        c = {"layer": 1, "adapter": v.name, "gn_scheme": getattr(v, "gn_scheme", None), "rows": max(b - a for a, b in p.stripes),
             "estimate": est, "plan_estimate": p.estimate, "selftest": st if st > p.estimate else 0, "seconds": v.predict_seconds(p),
             "fits": est <= bud, "workspace": p.workspace, "prefix": p.prefix_bytes, "stripes": p.stripe_bytes}
        considered.append(c)
        (fits if c["fits"] else over).append((v, p, c))
    fits.sort(key=lambda x: (x[2]["seconds"] is None, x[2]["seconds"] or 0.0))
    pick, over_budget = fits, False
    if not fits and rows and over:
        pick, over_budget = sorted(over, key=lambda x: x[1].estimate), True   # forced rows outrank the budget
    pre = msg("vae.forced_pre", what=", ".join(forced)) if forced else ""
    for v, p, c in pick:
        ok, detail = selftest(v)
        if not ok:
            c["selftest_result"] = "failed"
            failed.append(v)
            continue
        if over_budget:
            reason = pre + msg("vae.reason_over")
        elif len(fits) > 1 or any(o["layer"] == 2 for o in considered):
            reason = pre + (msg("vae.reason_fastest") if c["seconds"] is not None else msg("vae.reason_fits"))
        else:
            reason = pre + msg("vae.reason_only")
        others = msg("vae.need_sep").join(_candidate_label(o) + ("" if o["fits"] else msg("vae.cand_over"))
                                         for o in considered if o is not c) or msg("vae.others_none")
        return {"layer": 1, "bound": v, "plan": p, "workspace": p.workspace, "selftest": detail, "candidates": considered,
                "policy": msg("vae.policy_join", head=head, policy=msg("vae.policy_forced_over") if over_budget else msg("vae.policy_fastest"), over=""),
                "why": msg("vae.why_layer1", head=head, cand=_candidate_label(c), reason=reason, others=others)}
    if forced and (pick or not variants):
        # the forced layer-1 configuration cannot run: its self-test failed, now or earlier in this process. Layer 2,
        # but only within the budget (DESIGN §9.14.10, review 02); the same decision either way
        note = msg("vae.selftest_failed_short")
        c2, d2 = layer2(msg("vae.l2_after_selftest"), note, note)
        if c2["fits"]:
            return d2
        raise MonoloadError(msg("vae.err_forced_selftest", budget=fmt_bytes(bud), src=_from("budget"), what=", ".join(forced),
                                failed=", ".join(v.name for v in failed), l2=fmt_bytes(c2["estimate"]), shape=list(samples_in.shape),
                                advice=_budget_advice()))
    needs = []
    for c in considered:
        if c["layer"] == 2:
            needs.append(msg("vae.need_layer2", est=fmt_bytes(c["estimate"])))
        else:
            needs.append(msg("vae.need_layer1", scheme=msg("vae.need_scheme", scheme=c["gn_scheme"]) if c["gn_scheme"] else "", rows=c["rows"],
                             est=fmt_bytes(c["estimate"]) + _selftest_note(c), prefix=fmt_bytes(c["prefix"]), stripes=fmt_bytes(c["stripes"]),
                             failed=msg("vae.need_failed") if c.get("selftest_result") else ""))
    if not variants:
        needs.append(msg("vae.need_l1_unavailable", why=l1_note))
    raise MonoloadError(msg("vae.err_budget", budget=fmt_bytes(bud), src=_from("budget"), advice=_budget_advice(),
                            shape=list(samples_in.shape), needs=msg("vae.need_sep").join(needs)))


def _decode_budget(self, samples_in, vae_options, t0, bud):
    try:
        d = choose_budget(self, samples_in, vae_options, bud)
    except MonoloadError:
        _LAST.clear()
        _LAST.update({"strategy": "error", "kind": "budget", "budget": bud})
        raise
    logging.info(msg("vae.budget_log", why=d["why"], note=("; " + _NOTE[0]) if _NOTE[0] else ""))
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
            raise MonoloadVAEOOMError(msg("vae.err_oom_l1", rows=rows, ws=fmt_bytes(ws), retries=retries, shape=list(samples_in.shape),
                                          est=fmt_bytes(plan.estimate)))
        rows = max(min_rows, rows // 2)
        ws = max(floor_ws, ws // 2)
        retries += 1
        plan = bound.plan(self, samples_in, bud, ws, rows=rows, out_bytes=outb)
        logging.warning(msg("vae.retry_l1", rows=rows, ws=fmt_bytes(ws), retries=retries))

    pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
    _sync()
    dt = time.perf_counter() - t0
    native_est = _native_estimate(self, samples_in.shape)
    boundaries = [(plan.h_out, a) for a, _ in plan.stripes[1:]]
    _LAST.clear()
    st = _SELFTEST_IN_DECODE[0]
    total = max(plan.estimate, st)   # the first use ran the self-test before the decode: the call's peak is the larger
    _LAST.update({"strategy": "layer1", "adapter": bound.name, "estimate": {"total": total, "plan": plan.estimate, "selftest": st, "first": first_est,
                  "prefix": plan.prefix_bytes, "stripes": plan.stripe_bytes, "checkpoint": plan.ckpt_bytes, "persistent": plan.persistent,
                  "live": plan.live_peak, "arena": plan.arena, "output": plan.out_segment},
                  "native_estimate": native_est, "budget": bud, "policy": policy, "workspace": ws, "retries": retries, "seconds": dt,
                  "stripes": len(plan.stripes), "rows": max(b - a for a, b in plan.stripes), "recompute": plan.recompute,
                  "checkpoint_bytes": plan.ckpt_bytes, "boundaries": boundaries, "forced_rows": forced, "selftest": selftest,
                  "passes": len(plan.passes), "saves": [plan.save_bytes[p] for p in plan.saves], "gn_scheme": getattr(bound, "gn_scheme", None),
                  "pass_rows": [ps.rows for ps in plan.passes], "candidates": considered,
                  "predicted_seconds": bound.predict_seconds(plan), "stats": stats.as_dict()})
    logging.info(msg("vae.log_layer1", shape="x".join(str(d) for d in samples_in.shape), name=bound.name, plan=plan.describe(), policy=policy,
                     ws=fmt_bytes(ws), retries=msg("vae.retries", n=retries) if retries else "",
                     arena=fmt_bytes(stats.arena) if stats.arena else msg("vae.none"),
                     est=fmt_bytes(plan.estimate) + (msg("vae.est_selftest", st=fmt_bytes(st)) if st else ""),
                     native=fmt_bytes(native_est), secs=dt, note=_NOTE[0]))
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
            logging.warning(msg("vae.probe_failed", err="{}: {}".format(type(e).__name__, e)))
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
                raise MonoloadVAEOOMError(msg("vae.err_oom_l2", ws=fmt_bytes(budget), retries=retries, shape=list(samples_in.shape),
                                              est=fmt_bytes(est["total"])))
            budget = max(floor, budget // 2)
            retries += 1
            logging.warning(msg("vae.retry_l2", ws=fmt_bytes(budget), retries=retries))

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
        attn = msg("vae.attn_blocks", calls=stats.attn_calls, sizes="{} / {}".format(stats.attn_rows_min, stats.attn_tokens_max))
    if stats.attn_unmanaged:
        attn += msg("vae.attn_native", what=", ".join(stats.attn_unmanaged[:4]))
    logging.info(msg("vae.log_layer2", shape="x".join(str(d) for d in samples_in.shape), ws=fmt_bytes(budget),
                     retries=msg("vae.retries", n=retries) if retries else "", chunked=stats.conv_chunked, calls=stats.conv_calls, attn=attn,
                     policy=policy + "; " if policy else "", est=fmt_bytes(est["total"]), native=fmt_bytes(native_est), secs=dt, note=_NOTE[0]))
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
        logging.warning(msg("vae.api_differs", bad=bad))
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
