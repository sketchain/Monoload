"""Runtime LoRA merge for ComfyUI-GGUF models (city96/ComfyUI-GGUF and its
forks, e.g. kyuz0/ComfyUI-GGUF-H3). See docs/GGUF-QUANT.md §3.

Natively GGUFModelPatcher keeps a quantized layer's patches on the
GGMLTensor and merges them into the dequantized temporary on every forward
(GGMLLayer.get_weight, fp32 intermediate), but bakes F16 / F32 layers into
the weight with a backup, and never applies hook LoRAs to quantized layers.
With the adapter installed, a patched key of a GGUF model gets a
MonoloadRuntimePatch like any other model (no backup, nothing written to the
GGMLTensor), and GGMLLayer.cast_bias_weight applies the runtime patches to
the dequantized temporary of a quantized layer (F16 / F32 layers go through
comfy.ops.cast_bias_weight, which calls the weight functions itself).

Found by structure, not by name (the node pack's folder name varies):
a ModelPatcher subclass with patch_on_device / mmap_released whose package
also has ops.GGMLLayer (get_weight / cast_bias_weight / is_ggml_quantized)
and dequant.is_quantized. find_and_install() runs at hotpatch.install() and
at the start of every prompt (custom nodes may load after Monoload). A GGUF
version whose methods do not look like the ones this adapter was written
against stays native (with a warning).
"""

import inspect
import logging
import sys

import torch

from . import hotpatch
from .errors import MonoloadUnsupportedError
from .messages import msg

_INSTALLED = {}       # GGUFModelPatcher class -> (orig patch_weight_to_device, orig load)
_LAYERS = {}          # GGMLLayer class -> orig cast_bias_weight
_REJECTED = set()
_QUANT_CHECKS = []    # the found packs' dequant.is_quantized(tensor)


def is_ggml(t):
    return isinstance(t, torch.Tensor) and hasattr(t, "tensor_type") and hasattr(t, "tensor_shape")


def is_ggml_quantized(t):
    """A GGML tensor of a quantized type (Q4_K, Q8_0, ...); F16 / F32 GGMLTensors are not."""
    if getattr(t, "tensor_type", None) is None:
        return False
    return any(f(t) for f in _QUANT_CHECKS)


def model_has_ggml_quantized(model):
    return bool(_QUANT_CHECKS) and any(is_ggml_quantized(p) for p in model.parameters())


def _looks_like_patcher(cls):
    return (isinstance(cls, type) and issubclass(cls, hotpatch.ModelPatcher) and cls is not hotpatch.ModelPatcher
            and "patch_weight_to_device" in cls.__dict__ and hasattr(cls, "patch_on_device") and hasattr(cls, "mmap_released"))


def _looks_like_layer(cls):
    return isinstance(cls, type) and all(callable(getattr(cls, n, None)) for n in ("get_weight", "cast_bias_weight", "is_ggml_quantized"))


def _compatible(patcher_cls, layer_cls):
    try:
        params = list(inspect.signature(patcher_cls.__dict__["patch_weight_to_device"]).parameters)
        load_params = inspect.signature(patcher_cls.__dict__["load"]).parameters
    except (KeyError, TypeError, ValueError):
        return False
    return params[:4] == ["self", "key", "device_to", "inplace_update"] and "force_patch_weights" in load_params \
        and "cast_bias_weight" in layer_cls.__dict__


def _sibling(cls, name):
    pkg = cls.__module__.rsplit(".", 1)[0] if "." in cls.__module__ else None
    return sys.modules.get(pkg + "." + name) if pkg else None


def find_and_install():
    """Install the adapter on every GGUF node pack loaded so far. Returns the number of newly adapted patcher classes."""
    n = 0
    for mod in list(sys.modules.values()):
        patcher_cls = getattr(mod, "GGUFModelPatcher", None)
        if not _looks_like_patcher(patcher_cls) or patcher_cls in _INSTALLED or patcher_cls in _REJECTED:
            continue
        layer_cls = getattr(_sibling(patcher_cls, "ops"), "GGMLLayer", None)
        quant_check = getattr(_sibling(patcher_cls, "dequant"), "is_quantized", None)
        if not _looks_like_layer(layer_cls) or not callable(quant_check) or not _compatible(patcher_cls, layer_cls):
            _REJECTED.add(patcher_cls)
            logging.warning(msg("lora.gguf_unknown", cls="{}.{}".format(patcher_cls.__module__, patcher_cls.__qualname__)))
            continue
        _install(patcher_cls, layer_cls, quant_check)
        n += 1
    return n


def _install(patcher_cls, layer_cls, quant_check):
    orig_pwtd = patcher_cls.__dict__["patch_weight_to_device"]
    orig_load = patcher_cls.__dict__["load"]

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        if not hotpatch._enabled(self):
            if return_weight:   # native GGUF has no return_weight: same TypeError as without Monoload
                return orig_pwtd(self, key, device_to=device_to, inplace_update=inplace_update, return_weight=return_weight)
            return orig_pwtd(self, key, device_to=device_to, inplace_update=inplace_update)
        if return_weight:
            raise MonoloadUnsupportedError("gguf_save", msg("lora.gguf_save"), key=key)
        if key not in self.patches:
            return None
        weight = hotpatch.comfy.utils.get_attr(self.model, key)
        if getattr(weight, "patches", None):
            weight.patches = []          # left by a native load (switch changed while loaded): never merge twice
        hotpatch._state(self)
        hotpatch._install_runtime_patch(self, key)
        return None

    def load(self, *args, force_patch_weights=False, **kwargs):
        if not hotpatch._enabled(self):
            return orig_load(self, *args, force_patch_weights=force_patch_weights, **kwargs)
        # GGUF always asks ModelPatcher.load for force_patch_weights=True (quantized layers take no LowVramPatch);
        # under Monoload that only means lowvram layers get a runtime patch too
        self.__dict__["_monoload_gguf_forced"] = True
        try:
            return orig_load(self, *args, force_patch_weights=force_patch_weights, **kwargs)
        finally:
            self.__dict__.pop("_monoload_gguf_forced", None)

    patch_weight_to_device.__wrapped__ = orig_pwtd
    load.__wrapped__ = orig_load
    patcher_cls.patch_weight_to_device = patch_weight_to_device
    patcher_cls.load = load
    hotpatch._ADAPTED_PWTD.add(patch_weight_to_device)
    _INSTALLED[patcher_cls] = (orig_pwtd, orig_load)
    if quant_check not in _QUANT_CHECKS:
        _QUANT_CHECKS.append(quant_check)

    if layer_cls not in _LAYERS:
        orig_cbw = layer_cls.__dict__["cast_bias_weight"]

        def cast_bias_weight(s, *args, **kwargs):
            weight, bias = orig_cbw(s, *args, **kwargs)
            wf = [f for f in s.__dict__.get("weight_function", ()) if hotpatch._is_runtime_patch(f)]
            bf = [f for f in s.__dict__.get("bias_function", ()) if hotpatch._is_runtime_patch(f)]
            if wf:
                weight = _private(weight, s.weight)
                for f in wf:
                    weight = f(weight)
            if bf and bias is not None:
                bias = _private(bias, s.bias)
                for f in bf:
                    bias = f(bias)
            return weight, bias

        cast_bias_weight.__wrapped__ = orig_cbw
        layer_cls.cast_bias_weight = cast_bias_weight
        _LAYERS[layer_cls] = orig_cbw


def _private(t, param):
    """The dequantized temporary as a plain tensor the runtime patch may modify in place: for an F16 / F32 tensor
    whose dtype already matches, GGUF hands back (a view of) the parameter itself."""
    if is_ggml(t):
        t = t.as_subclass(torch.Tensor)
    if param is not None and t.untyped_storage().data_ptr() == param.untyped_storage().data_ptr():
        t = t.clone()
    return t


def uninstall():
    for cls, (pwtd, load) in _INSTALLED.items():
        hotpatch._ADAPTED_PWTD.discard(cls.__dict__.get("patch_weight_to_device"))
        cls.patch_weight_to_device = pwtd
        cls.load = load
    _INSTALLED.clear()
    for cls, cbw in _LAYERS.items():
        cls.cast_bias_weight = cbw
    _LAYERS.clear()
    _REJECTED.clear()
    _QUANT_CHECKS.clear()


def summary(patcher):
    """(quantized GGML keys patched, F16 / F32 GGML keys patched) for the Info node; None if not an adapted GGUF model."""
    if type(patcher) not in _INSTALLED:
        return None
    q = plain = 0
    for key in patcher.patches:
        try:
            p = hotpatch.comfy.utils.get_attr(patcher.model, key)
        except AttributeError:
            continue
        if is_ggml_quantized(p):
            q += 1
        elif is_ggml(p):
            plain += 1
    return q, plain
