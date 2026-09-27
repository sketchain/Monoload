"""Runtime LoRA merge for every ComfyUI ModelPatcher.

install() replaces a handful of methods on comfy.model_patcher.ModelPatcher
(the class itself, so every instance and every subclass that does not
override them is covered: UNETLoader, CheckpointLoaderSimple, CLIPLoader,
VAELoader, ...). With it installed, a LoRA/patch is never baked into the
weights and nothing is ever backed up: each patched layer gets a
MonoloadRuntimePatch in its weight_function/bias_function list, and
comfy.ops' cast_bias_weight merges it into a temporary copy of that one layer
when the layer runs. Results are bit-identical to native ComfyUI.

uninstall() restores the original methods (models should be unloaded first).
"""

import logging
import weakref

import torch

import comfy.float
import comfy.lora
import comfy.memory_management
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import comfy.weight_adapter
from comfy.model_patcher import LowVramPatch, ModelPatcher, get_key_weight

from .errors import MonoloadError, MonoloadUnsupportedError

_ORIG = {}
_ORIG_DYNAMIC = {}
_WARNED = set()

# dtype pairs (param dtype, compute dtype) where param -> compute is exact, so
# the compute-dtype copy handed to the weight function can stand in for the
# parameter itself without changing a single bit of the result.
_EXACT_WIDENING = {
    (torch.float16, torch.float32), (torch.bfloat16, torch.float32),
    (torch.float16, torch.float64), (torch.bfloat16, torch.float64),
    (torch.float32, torch.float64),
}


def _identity(a, **kwargs):
    return a


# ---------------------------------------------------------------------------
# per-patcher state
# ---------------------------------------------------------------------------

class _State:
    __slots__ = ("hook_patches", "device_cache")

    def __init__(self):
        self.hook_patches = {}   # key -> hook patch list currently in effect
        self.device_cache = {}   # (id(tensor), device) -> (tensor, tensor on device)


def _state(patcher):
    st = patcher.__dict__.get("_monoload_state")
    if st is None:
        st = _State()
        patcher.__dict__["_monoload_state"] = st
    return st


def _to_device(value, device, cache):
    """Same structure as comfy.lora.prefetch_prepared_value: tensors (also inside
    weight adapters / tuples / lists) moved to `device` once and cached."""
    if isinstance(value, torch.Tensor):
        if value.device == device:
            return value
        k = (id(value), device)
        hit = cache.get(k)
        if hit is None:
            hit = (value, value.to(device))
            cache[k] = hit
        return hit[1]
    if isinstance(value, comfy.weight_adapter.WeightAdapterBase):
        return type(value)(value.loaded_keys, _to_device(value.weights, device, cache))
    if isinstance(value, tuple):
        return tuple(_to_device(v, device, cache) for v in value)
    if isinstance(value, list):
        return [_to_device(v, device, cache) for v in value]
    return value


# ---------------------------------------------------------------------------
# the weight function
# ---------------------------------------------------------------------------

class MonoloadRuntimePatch(LowVramPatch):
    """Weight function computing patched(W) for one key while the layer runs.

    Numerics replicate ModelPatcher.patch_weight_to_device (baked LoRA) and
    patch_hook_weight_to_device (hook LoRA), so outputs are bit-identical:
      base  : W -> lora_compute_dtype -> calculate_weight -> stochastic_rounding(param dtype)
      hooks : -> float32 -> calculate_weight(original_weights) -> stochastic_rounding(param dtype)
    followed by the cast to the compute dtype cast_bias_weight asked for.

    `weight` is the private temporary that cast_bias_weight made
    (copy=True because a weight function is present), so when it carries the
    parameter's exact values it is used directly (and modified in place)
    instead of re-reading the parameter.
    """

    is_monoload_patch = True

    def __init__(self, key, patches, module, attr, state):
        super().__init__(key, patches)
        self._module = weakref.ref(module)
        self.attr = attr
        self.state = state
        self.seed = comfy.utils.string_to_seed(key)
        self._base_sig = None
        self._base_moved = None

    def memory_required(self):
        return 0

    def prepare(self, destination, stream, copy=True, commit=True):
        return None

    def _base_on(self, base, device):
        sig = (device,) + tuple(id(p) for p in base)
        if sig != self._base_sig:
            self._base_moved = _to_device(list(base), device, self.state.device_cache)
            self._base_sig = sig
        return self._base_moved

    def __call__(self, weight):
        key = self.key
        base = self.patches.get(key)
        hooks = self.state.hook_patches.get(key)
        if not base and not hooks:
            return weight
        param = getattr(self._module(), self.attr)
        device = weight.device
        pdt = param.dtype
        cdt = weight.dtype
        src_exact = cdt == pdt or (pdt, cdt) in _EXACT_WIDENING

        w = None
        if base:
            ldt = comfy.model_management.lora_compute_dtype(device)
            if src_exact:
                temp = weight if cdt == ldt else weight.to(ldt)
            else:
                temp = comfy.model_management.cast_to_device(param, device, ldt, copy=True)
            out = comfy.lora.calculate_weight(self._base_on(base, device), temp, key)
            w = comfy.float.stochastic_rounding(out, pdt, seed=self.seed)
            del temp, out

        if hooks:
            if w is not None:
                temp = w.to(torch.float32)
            elif src_exact:
                temp = weight.to(torch.float32)
            else:
                temp = comfy.model_management.cast_to_device(param, device, torch.float32, copy=True)
            original = {key: [(param, _identity)] + list(base or [])}
            out = comfy.lora.calculate_weight(_to_device(hooks, device, self.state.device_cache), temp, key, original_weights=original)
            w = comfy.float.stochastic_rounding(out, pdt, seed=self.seed)
            del temp, out

        return w.to(dtype=cdt)


def _is_runtime_patch(f):
    return getattr(f, "is_monoload_patch", False)


# ---------------------------------------------------------------------------
# helpers on a patcher
# ---------------------------------------------------------------------------

def _active(patcher):
    """Monoload only drives patchers whose class still uses ModelPatcher's own
    patch_weight_to_device. Subclasses with their own weight patching (e.g.
    ComfyUI-GGUF) keep their native behaviour."""
    cls = type(patcher)
    if cls.patch_weight_to_device is _patch_weight_to_device:
        return True
    if cls not in _WARNED:
        _WARNED.add(cls)
        logging.warning("[Monoload] {}.{} overrides patch_weight_to_device; Monoload leaves it native".format(cls.__module__, cls.__qualname__))
    return False


def _has_patches(patcher):
    return len(patcher.patches) > 0 or len(getattr(patcher, "hook_patches", {}) or {}) > 0


def _first_key(patcher):
    if patcher.patches:
        return next(iter(patcher.patches))
    return None


def _check_dynamic(patcher):
    if comfy.memory_management.aimdo_enabled and _has_patches(patcher):
        raise MonoloadUnsupportedError(
            "dynamic_vram",
            "ComfyUI 开启了 DynamicVRAM（comfy-aimdo），Monoload 不支持在这种模式下打 LoRA。"
            "请用 --gpu-only / --highvram / --disable-dynamic-vram 启动，或设 MONOLOAD_DISABLE=1 关闭 Monoload。",
            key=_first_key(patcher))


def _module_for_key(patcher, key):
    parts = key.rsplit(".", 1)
    if len(parts) != 2:
        raise MonoloadUnsupportedError("lora_non_comfy_ops_param", "patch 的目标不是某个层的参数", key=key)
    return comfy.utils.get_attr(patcher.model, parts[0]), parts[1]


def _install_runtime_patch(patcher, key):
    st = _state(patcher)
    module, attr = _module_for_key(patcher, key)
    if not hasattr(module, "comfy_cast_weights") or attr not in ("weight", "bias"):
        raise MonoloadUnsupportedError(
            "lora_non_comfy_ops_param",
            "被 LoRA/patch 修改的参数所在的模块 {} 不是 comfy.ops 层，没有运行时合并路径；"
            "Monoload 不会退回到「改权重+备份」。".format(type(module).__name__),
            key=key)
    weight, set_func, convert_func = get_key_weight(patcher.model, key)
    if set_func is not None or convert_func is not None:
        raise MonoloadUnsupportedError(
            "lora_quantized_param",
            "量化参数（{}）上的 LoRA 暂不支持运行时合并".format(type(weight).__name__),
            key=key)
    patches = list(patcher.patches.get(key, [])) + list(st.hook_patches.get(key, []))
    new_shape = comfy.lora.calculate_shape(patches, weight, key)
    if tuple(new_shape) != tuple(weight.shape):
        raise MonoloadUnsupportedError(
            "lora_shape_change",
            "patch 会把权重形状从 {} 改成 {}，运行时合并无法支持".format(list(weight.shape), list(new_shape)),
            key=key)
    fn_attr = attr + "_function"
    funcs = [f for f in getattr(module, fn_attr, []) if not (_is_runtime_patch(f) and f.key == key)]
    # LoRA first: under a native full load it is baked into the weight, so it
    # comes before any weight_wrapper_patches.
    funcs.insert(0, MonoloadRuntimePatch(key, patcher.patches, module, attr, st))
    setattr(module, fn_attr, funcs)


def _has_runtime_patch(patcher, key):
    module, attr = _module_for_key(patcher, key)
    return any(_is_runtime_patch(f) and f.key == key for f in module.__dict__.get(attr + "_function", []))


def _remove_runtime_patches(patcher, keep=None):
    for m in patcher.model.modules():
        touched = False
        for fn_attr in ("weight_function", "bias_function"):
            funcs = m.__dict__.get(fn_attr, None)
            if not funcs:
                continue
            kept = [f for f in funcs if not _is_runtime_patch(f) or (keep is not None and f.key in keep)]
            if len(kept) != len(funcs):
                setattr(m, fn_attr, kept)
                touched = True
        if touched and hasattr(m, "comfy_patched_weights"):
            del m.comfy_patched_weights


def _drop_shadowed_runtime_patches(patcher):
    """After a native partial unload a module may carry both our patch and a
    native LowVramPatch for the same key (native would have restored the
    backup and switched that layer to LowVramPatch). Keep only the native one
    so the result is exactly what native ComfyUI computes."""
    for m in patcher.model.modules():
        for fn_attr in ("weight_function", "bias_function"):
            funcs = m.__dict__.get(fn_attr, None)
            if not funcs:
                continue
            native_keys = {f.key for f in funcs if type(f) is LowVramPatch}
            if native_keys:
                kept = [f for f in funcs if not (_is_runtime_patch(f) and f.key in native_keys)]
                if len(kept) != len(funcs):
                    setattr(m, fn_attr, kept)


def _assert_no_backup(patcher):
    if len(patcher.backup) > 0 or len(patcher.hook_backup) > 0:
        raise MonoloadError("[Monoload] 内部错误：出现了权重备份 {}".format(list(patcher.backup)[:5] + list(patcher.hook_backup)[:5]))


# ---------------------------------------------------------------------------
# ModelPatcher method replacements
# ---------------------------------------------------------------------------

def _patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
    if key not in self.patches or return_weight:
        # Unpatched keys are a no-op natively; return_weight only computes a
        # temporary merged tensor and never writes or backs up.
        return _ORIG["patch_weight_to_device"](self, key, device_to=device_to, inplace_update=inplace_update,
                                               return_weight=return_weight, force_cast=force_cast)
    _install_runtime_patch(self, key)
    return None


def _load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
    if not _active(self):
        return _ORIG["load"](self, device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load)
    _check_dynamic(self)
    if force_patch_weights and len(self.patches) > 0:
        raise MonoloadUnsupportedError(
            "force_patch_weights",
            "有节点要求把 LoRA/patch 直接烘焙进权重（force_patch_weights，常见于保存/合并模型）。"
            "Monoload 只做运行时临时合并，不改权重、不备份；需要保存合并结果时请设 MONOLOAD_DISABLE=1 后运行。",
            key=_first_key(self))
    # Native load() wipes weight_function on every fully-loaded comfy.ops
    # module but skips modules already flagged comfy_patched_weights (natively
    # their patches are baked in). Ours are not, so make every patched module
    # go through patch_weight_to_device again.
    for key in self.patches:
        parts = key.rsplit(".", 1)
        if len(parts) == 2:
            try:
                m = comfy.utils.get_attr(self.model, parts[0])
            except AttributeError:
                continue
            if hasattr(m, "comfy_patched_weights"):
                del m.comfy_patched_weights
    r = _ORIG["load"](self, device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load)
    _assert_no_backup(self)
    return r


def _partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
    if not _active(self):
        return _ORIG["partially_unload"](self, device_to, memory_to_free=memory_to_free, force_patch_weights=force_patch_weights)
    if force_patch_weights and len(self.patches) > 0:
        raise MonoloadUnsupportedError("force_patch_weights", "Monoload 模型不支持 force_patch_weights", key=_first_key(self))
    freed = _ORIG["partially_unload"](self, device_to, memory_to_free=memory_to_free, force_patch_weights=force_patch_weights)
    _drop_shadowed_runtime_patches(self)
    _assert_no_backup(self)
    return freed


def _unpatch_model(self, device_to=None, unpatch_weights=True):
    r = _ORIG["unpatch_model"](self, device_to=device_to, unpatch_weights=unpatch_weights)
    if unpatch_weights and _active(self):
        _remove_runtime_patches(self)
        _state(self).device_cache.clear()
    return r


def _patch_hooks(self, hooks):
    if not _active(self):
        return _ORIG["patch_hooks"](self, hooks)
    st = _state(self)
    with self.use_ejected():
        st.hook_patches.clear()
        if hooks is not None:
            combined = self.get_combined_hook_patches(hooks=hooks)
            if combined:
                _check_dynamic(self)
                model_keys = set(self.model_state_dict().keys())
                for key, patches in combined.items():
                    if key not in model_keys:
                        logging.warning("Hook could not patch. Key does not exist in model: {}".format(key))
                        continue
                    st.hook_patches[key] = patches
                for key in st.hook_patches:
                    if not _has_runtime_patch(self, key):
                        _install_runtime_patch(self, key)
        _remove_runtime_patches(self, keep=set(self.patches) | set(st.hook_patches))
        self.current_hooks = hooks


def _unpatch_hooks(self, whitelist_keys_set=None):
    if not _active(self):
        return _ORIG["unpatch_hooks"](self, whitelist_keys_set)
    st = _state(self)
    with self.use_ejected():
        if whitelist_keys_set:
            for k in list(st.hook_patches):
                if k in whitelist_keys_set:
                    st.hook_patches.pop(k)
        else:
            st.hook_patches.clear()
            self.current_hooks = None
        if st.hook_patches or self.patches:
            _remove_runtime_patches(self, keep=set(self.patches) | set(st.hook_patches))
        else:
            _remove_runtime_patches(self)


def _patch_hook_weight_to_device(self, hooks, combined_patches, key, original_weights, memory_counter):
    if not _active(self):
        return _ORIG["patch_hook_weight_to_device"](self, hooks, combined_patches, key, original_weights, memory_counter)
    if key not in combined_patches:
        return
    raise MonoloadError("[Monoload] 内部错误：hook 不应走写权重的路径（key={}）".format(key))


def _patch_cached_hook_weights(self, cached_weights, key, memory_counter):
    if not _active(self):
        return _ORIG["patch_cached_hook_weights"](self, cached_weights, key, memory_counter)
    raise MonoloadError("[Monoload] 内部错误：hook 不应走缓存权重的路径（key={}）".format(key))


def _dynamic_load(self, *args, **kwargs):
    _check_dynamic(self)
    return _ORIG_DYNAMIC["load"](self, *args, **kwargs)


_REPLACEMENTS = {
    "patch_weight_to_device": _patch_weight_to_device,
    "load": _load,
    "partially_unload": _partially_unload,
    "unpatch_model": _unpatch_model,
    "patch_hooks": _patch_hooks,
    "unpatch_hooks": _unpatch_hooks,
    "patch_hook_weight_to_device": _patch_hook_weight_to_device,
    "patch_cached_hook_weights": _patch_cached_hook_weights,
}


def is_installed():
    return bool(_ORIG)


def install():
    if _ORIG:
        return False
    for name, fn in _REPLACEMENTS.items():
        _ORIG[name] = ModelPatcher.__dict__[name]
        setattr(ModelPatcher, name, fn)
    dyn = getattr(comfy.model_patcher, "ModelPatcherDynamic", None)
    if dyn is not None and "load" in dyn.__dict__:
        _ORIG_DYNAMIC["load"] = dyn.__dict__["load"]
        dyn.load = _dynamic_load
    return True


def uninstall():
    """Restore native ModelPatcher. Unload all models first."""
    if not _ORIG:
        return False
    for name, fn in _ORIG.items():
        setattr(ModelPatcher, name, fn)
    _ORIG.clear()
    dyn = getattr(comfy.model_patcher, "ModelPatcherDynamic", None)
    if dyn is not None and "load" in _ORIG_DYNAMIC:
        dyn.load = _ORIG_DYNAMIC.pop("load")
    return True
