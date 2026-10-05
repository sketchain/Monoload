"""Runtime LoRA merge for every ComfyUI ModelPatcher.

install() replaces a handful of methods on comfy.model_patcher.ModelPatcher
(the class itself, so every instance and every subclass that does not
override them is covered: UNETLoader, CheckpointLoaderSimple, CLIPLoader,
VAELoader, ...). With it installed, a LoRA/patch is never baked into the
weights and nothing is ever backed up: each patched layer gets a
MonoloadRuntimePatch in its weight_function/bias_function list, and
comfy.ops' cast_bias_weight merges it into a temporary copy of that one layer
when the layer runs.

Two merge paths (global default: MONOLOAD_EXACT, read at import; set_exact()
at runtime; a model's Monoload LoRA Settings node may choose its own):
  default         plain LoRA / LoCon patches are added with one fused
                  addmm_ into the compute-dtype temporary (C); every other
                  patch type goes through comfy.lora.calculate_weight with the
                  compute dtype as intermediate dtype (A, the numerics of
                  native ComfyUI's lowvram LowVramPatch). Not bit-identical to
                  a native (baked) merge; see docs/DESIGN.md for the error.
  MONOLOAD_EXACT=1  bit-identical to native ComfyUI.

Which patchers Monoload drives (_enabled): a patcher whose Monoload LoRA
Settings mode is enable / native (monoload/lora_overrides.py, carried in
model_options), else the master switch (monoload/settings.py). For one it
does not drive (MONOLOAD=0, or mode native) every replaced method passes the
call straight to ComfyUI's original (_active() is False), so it behaves
exactly as native; the installed methods only cost one check per call.

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

from . import lora_overrides, settings
from .errors import MonoloadError, MonoloadUnsupportedError
from .messages import msg

_ORIG = {}
_ORIG_DYNAMIC = {}
_WARNED = set()


def set_exact(on):
    """True: bit-exact merge (= native ComfyUI) as the global default. False:
    fused/relaxed default. Takes effect at the next layer call; nothing has to
    be reloaded. A model whose Monoload LoRA Settings node chose a merge keeps it."""
    settings.set_exact(on)


def is_exact():
    """The global default merge (MONOLOAD_EXACT at import)."""
    return settings.exact()


_MATH_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

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


try:
    from comfy.quant_ops import QuantizedTensor as _QuantizedTensor
except Exception:  # pragma: no cover
    _QuantizedTensor = ()


def _is_quantized(t):
    return bool(_QuantizedTensor) and isinstance(t, _QuantizedTensor)


# ---------------------------------------------------------------------------
# the binding: what the runtime patches of one model compute with
# ---------------------------------------------------------------------------

class _Binding:
    """One per model (patcher.model), shared by every clone: the runtime
    patches on the model's modules read it on every call (DESIGN §7, review
    01). It holds the patches dict and the merge (exact) of the patcher whose
    weights are active (bind() at load / partially_load / install), the hook
    patches in effect (written by whichever clone patches or unpatches hooks
    last: natively the clones share the weights and hook_backup the same way)
    and the device-side copies of the LoRA tensors."""
    __slots__ = ("patches", "exact", "hook_patches", "device_cache", "__weakref__")

    def __init__(self):
        self.patches = {}        # the active patcher's patches (key -> patch list)
        self.exact = None        # its merge (Monoload LoRA Settings), None = the global default
        self.hook_patches = {}   # key -> hook patch list currently in effect
        self.device_cache = {}   # (id(tensor), device) -> (tensor, tensor on device)

    def bind(self, patcher):
        if self.patches is not patcher.patches:
            self.patches = patcher.patches
        self.exact = lora_overrides.merge_exact(patcher)


def _binding(patcher):
    """The model's binding (created on first use, not bound)."""
    b = patcher.model.__dict__.get("_monoload_binding")
    if b is None:
        b = _Binding()
        patcher.model.__dict__["_monoload_binding"] = b
    return b


def _state(patcher):
    """The model's binding, bound to `patcher`."""
    b = _binding(patcher)
    b.bind(patcher)
    return b


def _release_binding(patcher):
    """No runtime patch left on the model: drop what the binding holds (the
    patcher's patches dict, hook patches, device copies), so nothing of a
    clone that is gone stays reachable from the model."""
    b = patcher.model.__dict__.get("_monoload_binding")
    if b is not None:
        b.patches, b.exact = {}, None
        b.hook_patches.clear()
        b.device_cache.clear()


def _to_device(value, device, cache, limit, flags):
    """Same structure as comfy.lora.prefetch_prepared_value: tensors (also inside
    weight adapters / tuples / lists) moved to `device`. Tensors smaller than
    the patched weight (`limit` elements: LoRA factors, alphas, ...) are moved
    once and cached; weight-sized ones (model merge, full diffs) are moved per
    call like native does, so no second copy of a model ever stays resident.
    flags["transient"] is set when anything was moved without caching."""
    if isinstance(value, torch.Tensor):
        if value.device == device:
            return value
        if value.numel() >= limit:
            flags["transient"] = True
            return value.to(device)
        k = (id(value), device)
        hit = cache.get(k)
        if hit is None:
            hit = (value, value.to(device))
            cache[k] = hit
        return hit[1]
    if isinstance(value, comfy.weight_adapter.WeightAdapterBase):
        return type(value)(value.loaded_keys, _to_device(value.weights, device, cache, limit, flags))
    if isinstance(value, tuple):
        return tuple(_to_device(v, device, cache, limit, flags) for v in value)
    if isinstance(value, list):
        return [_to_device(v, device, cache, limit, flags) for v in value]
    return value


# ---------------------------------------------------------------------------
# the weight function
# ---------------------------------------------------------------------------

class MonoloadRuntimePatch(LowVramPatch):
    """Weight function computing patched(W) for one key while the layer runs.

    Default path (_call_fast): merged straight into the compute-dtype
    temporary, plain LoRA/LoCon via one fused addmm_ per patch, everything
    else via calculate_weight(intermediate_dtype=compute dtype); see
    _merge_fast.

    MONOLOAD_EXACT=1 path: numerics replicate ModelPatcher.patch_weight_to_device (baked LoRA) and
    patch_hook_weight_to_device (hook LoRA), so outputs are bit-identical:
      base  : W -> lora_compute_dtype -> calculate_weight -> stochastic_rounding(param dtype)
      hooks : -> float32 -> calculate_weight(original_weights) -> stochastic_rounding(param dtype)
    followed by the cast to the compute dtype cast_bias_weight asked for.

    `weight` is the private temporary that cast_bias_weight made
    (copy=True because a weight function is present), so when it carries the
    parameter's exact values it is used directly (and modified in place)
    instead of re-reading the parameter.

    Quantized parameters (QuantizedTensor, e.g. fp8 scaled) are the one
    relaxed case: the LoRA is merged on the dequantized temporary and the
    result is used as is, not re-quantized. This equals running the bit-exact
    path on a model whose quantized weights were dequantized beforehand; it
    is not bit-identical to native, which re-quantizes after merging.
    """

    is_monoload_patch = True

    def __init__(self, key, patches, module, attr, binding):
        # not LowVramPatch.__init__: `patches` is the binding's (the active patcher's), read at every call
        self.key = key
        self.convert_func = self.set_func = self.prepared_patches = None
        self.binding = binding
        if binding.patches is not patches and not binding.patches:
            binding.patches = patches
        self._module = weakref.ref(module)
        self.attr = attr
        self._fn_attr = attr + "_function"
        self.seed = comfy.utils.string_to_seed(key)
        self._base_sig = None
        self._base_moved = None

    @property
    def patches(self):
        return self.binding.patches

    @property
    def exact(self):
        return self.binding.exact

    def memory_required(self):
        return 0

    def prepare(self, destination, stream, copy=True, commit=True):
        return None

    def _base_on(self, base, device, limit):
        sig = (device,) + tuple(id(p) for p in base)
        if sig == self._base_sig:
            return self._base_moved
        flags = {"transient": False}
        moved = _to_device(list(base), device, self.binding.device_cache, limit, flags)
        if not flags["transient"]:
            self._base_moved = moved
            self._base_sig = sig
        return moved

    def _source(self, weight, param, device):
        """A private tensor holding exactly the parameter's values (in its own
        dtype or an exact widening of it), or None if only the parameter can
        provide that. For quantized parameters "the parameter's values" are
        its dequantized values (dtype = param.dtype = the dequantized dtype)."""
        cdt = weight.dtype
        pdt = param.dtype
        if _is_quantized(param):
            if cdt == pdt:
                return weight  # cast_bias_weight dequantized straight into pdt
            return comfy.model_management.cast_to_device(param, device, None, copy=True).dequantize()
        if cdt == pdt or (pdt, cdt) in _EXACT_WIDENING:
            return weight
        return None

    def _native_base(self):
        """A native LowVramPatch for this key follows on the module (a lowvram
        layer): it applies the key's own patches, so this one must not."""
        module = self._module()
        funcs = getattr(module, self._fn_attr, ()) if module is not None else ()
        return any(type(f) is LowVramPatch and f.key == self.key for f in funcs)

    def __call__(self, weight):
        key = self.key
        b = self.binding
        base_all = b.patches.get(key)
        hooks = b.hook_patches.get(key)
        if not base_all and not hooks:
            return weight
        # on a lowvram layer native's LowVramPatch (after this one) adds the key's patches: only the hooks here, i.e.
        # native's order (the hook merged into the stored weight, the LoRA added when the layer runs)
        base = None if base_all and self._native_base() else base_all
        if not base and not hooks:
            return weight
        param = getattr(self._module(), self.attr)
        if not (settings.exact() if b.exact is None else b.exact):
            return self._call_fast(weight, param, base, hooks, base_all)
        device = weight.device
        pdt = param.dtype  # for a QuantizedTensor: its dequantized dtype
        cdt = weight.dtype
        src = self._source(weight, param, device)

        w = None
        if base:
            ldt = comfy.model_management.lora_compute_dtype(device)
            if src is not None:
                temp = src if src.dtype == ldt else src.to(ldt)
            else:
                temp = comfy.model_management.cast_to_device(param, device, ldt, copy=True)
            out = comfy.lora.calculate_weight(self._base_on(base, device, param.numel()), temp, key)
            w = comfy.float.stochastic_rounding(out, pdt, seed=self.seed)
            del temp, out

        if hooks:
            if w is not None:
                temp = w.to(torch.float32)
            elif src is not None:
                temp = src.to(torch.float32)
            else:
                temp = comfy.model_management.cast_to_device(param, device, torch.float32, copy=True)
            orig_param = param.dequantize() if _is_quantized(param) else param
            original = {key: [(orig_param, _identity)] + list(base_all or [])}
            moved_hooks = _to_device(hooks, device, self.binding.device_cache, param.numel(), {"transient": False})
            out = comfy.lora.calculate_weight(moved_hooks, temp, key, original_weights=original)
            w = comfy.float.stochastic_rounding(out, pdt, seed=self.seed)
            del temp, out

        return w.to(dtype=cdt)

    def _call_fast(self, weight, param, base, hooks, base_all=None):
        """Default path. The temporary is the one cast_bias_weight made, in the
        compute dtype (the tensor native lowvram LowVramPatch merges into). For
        a quantized parameter whose dequantized dtype differs from the compute
        dtype it is re-read as dequantize -> compute dtype, i.e. what a model
        dequantized beforehand would hand over."""
        key = self.key
        device = weight.device
        cdt = weight.dtype
        w = weight
        if _is_quantized(param) and cdt != param.dtype:
            w = comfy.model_management.cast_to_device(param, device, None, copy=True).dequantize().to(cdt)
        if w.dtype not in _MATH_DTYPES:
            w = w.to(torch.float32)
        if base:
            w = _merge_fast(self._base_on(base, device, param.numel()), w, key)
        if hooks:
            orig_param = param.dequantize() if _is_quantized(param) else param
            original = {key: [(orig_param, _identity)] + list((base if base_all is None else base_all) or [])}
            moved_hooks = _to_device(hooks, device, self.binding.device_cache, param.numel(), {"transient": False})
            w = _merge_fast(moved_hooks, w, key, original_weights=original)
        return w.to(dtype=cdt)


def fused_factors(patch):
    """(up, down, scale) when `patch` is a plain LoRA / LoCon (LoRAAdapter
    without Tucker mid, DoRA or reshape; no offset, custom function or
    strength_model), else None. scale = strength * alpha / rank, the factor
    LoRAAdapter.calculate_weight applies to up @ down."""
    strength, v, strength_model, offset, function = patch
    if not isinstance(v, comfy.weight_adapter.LoRAAdapter) or offset is not None or function is not None or strength_model != 1.0:
        return None
    up, down, alpha, mid, dora, reshape = v.weights[:6]
    if mid is not None or dora is not None or reshape is not None:
        return None
    a = (alpha / down.shape[0]) if alpha is not None else 1.0
    return up, down, float(strength * a)


def _merge_fast(patches, w, key, original_weights=None):
    """Apply `patches` in order to the temporary `w` (modified in place):
      C  plain LoRA / LoCon: w.view(out, -1).addmm_(up, down, alpha=scale), in
         w's dtype (one pass over the weight, no weight-sized intermediate)
      A  anything else: calculate_weight with w's dtype as intermediate dtype
    Patches are independent steps of calculate_weight, so mixing the two per
    patch keeps the order of operations of a native merge."""
    for p in patches:
        f = fused_factors(p)
        if f is not None and w.is_contiguous():
            up, down, scale = f
            w2 = w.view(w.shape[0], -1)
            up = up.flatten(1).to(w.dtype)
            down = down.flatten(1).to(w.dtype)
            if up.shape[0] == w2.shape[0] and down.shape[1] == w2.shape[1] and up.shape[1] == down.shape[0]:
                if scale != 0.0:
                    w2.addmm_(up, down, alpha=scale)
                continue
        w = comfy.lora.calculate_weight([p], w, key, intermediate_dtype=w.dtype, original_weights=original_weights)
    return w


def _is_runtime_patch(f):
    return getattr(f, "is_monoload_patch", False)


# ---------------------------------------------------------------------------
# helpers on a patcher
# ---------------------------------------------------------------------------

def _enabled(patcher):
    """Monoload drives this patcher's LoRA: its Monoload LoRA Settings mode
    (enable / native), else the master switch (MONOLOAD)."""
    return lora_overrides.enabled(patcher)


def _active(patcher):
    """Monoload drives this patcher: it is enabled (_enabled) and its class
    still uses ModelPatcher's own patch_weight_to_device. Subclasses with
    their own weight patching (e.g. ComfyUI-GGUF) keep their native
    behaviour. Not active -> every replaced method is the native one."""
    if not _enabled(patcher):
        return False
    cls = type(patcher)
    if cls.patch_weight_to_device is _patch_weight_to_device:
        return True
    if cls not in _WARNED:
        _WARNED.add(cls)
        logging.warning(msg("lora.subclass_native", cls="{}.{}".format(cls.__module__, cls.__qualname__)))
    return False


def _has_patches(patcher):
    return len(patcher.patches) > 0 or len(getattr(patcher, "hook_patches", {}) or {}) > 0


def _first_key(patcher):
    if patcher.patches:
        return next(iter(patcher.patches))
    return None


def _check_dynamic(patcher):
    if comfy.memory_management.aimdo_enabled and _has_patches(patcher):
        raise MonoloadUnsupportedError("dynamic_vram", msg("lora.dynamic_vram"), key=_first_key(patcher))


def _module_for_key(patcher, key):
    parts = key.rsplit(".", 1)
    if len(parts) != 2:
        raise MonoloadUnsupportedError("lora_non_comfy_ops_param", msg("lora.not_layer_param"), key=key)
    return comfy.utils.get_attr(patcher.model, parts[0]), parts[1]


def _install_runtime_patch(patcher, key):
    st = _binding(patcher)
    patcher.model.__dict__["_monoload_runtime"] = True   # runtime patches may live on this model's modules
    module, attr = _module_for_key(patcher, key)
    if not hasattr(module, "comfy_cast_weights") or attr not in ("weight", "bias"):
        raise MonoloadUnsupportedError("lora_non_comfy_ops_param", msg("lora.non_comfy_ops", module=type(module).__name__), key=key)
    # Quantized weights (mixed-precision ops, set_/convert_ functions) are
    # handled in relaxed mode: merged on the dequantized temporary and never
    # re-quantized (see MonoloadRuntimePatch._source).
    weight, _set_func, _convert_func = get_key_weight(patcher.model, key)
    patches = list(patcher.patches.get(key, [])) + list(st.hook_patches.get(key, []))
    new_shape = comfy.lora.calculate_shape(patches, weight, key)
    if tuple(new_shape) != tuple(weight.shape):
        raise MonoloadUnsupportedError("lora_shape_change", msg("lora.shape_change", old=list(weight.shape), new=list(new_shape)), key=key)
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
    if keep is None:
        patcher.model.__dict__.pop("_monoload_runtime", None)
        _release_binding(patcher)


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
        raise MonoloadError(msg("lora.internal_backup", keys=list(patcher.backup)[:5] + list(patcher.hook_backup)[:5]))


# ---------------------------------------------------------------------------
# ModelPatcher method replacements
# ---------------------------------------------------------------------------

def _patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
    if key not in self.patches or return_weight or not _enabled(self):
        # Unpatched keys are a no-op natively; return_weight only computes a
        # temporary merged tensor and never writes or backs up.
        return _ORIG["patch_weight_to_device"](self, key, device_to=device_to, inplace_update=inplace_update,
                                               return_weight=return_weight, force_cast=force_cast)
    _state(self)   # this patcher's weights are being loaded: the runtime patches compute with its patches and merge
    _install_runtime_patch(self, key)
    return None


def _load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
    if not _active(self):
        return _ORIG["load"](self, device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load)
    _check_dynamic(self)
    if force_patch_weights and len(self.patches) > 0:
        raise MonoloadUnsupportedError("force_patch_weights", msg("lora.force_patch"), key=_first_key(self))
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
    _state(self)
    r = _ORIG["load"](self, device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load)
    _assert_no_backup(self)
    return r


def _partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
    if not _active(self):
        return _ORIG["partially_unload"](self, device_to, memory_to_free=memory_to_free, force_patch_weights=force_patch_weights)
    if force_patch_weights and len(self.patches) > 0:
        raise MonoloadUnsupportedError("force_patch_weights", msg("lora.force_patch_unload"), key=_first_key(self))
    freed = _ORIG["partially_unload"](self, device_to, memory_to_free=memory_to_free, force_patch_weights=force_patch_weights)
    _drop_shadowed_runtime_patches(self)
    _assert_no_backup(self)
    return freed


def _unpatch_model(self, device_to=None, unpatch_weights=True):
    r = _ORIG["unpatch_model"](self, device_to=device_to, unpatch_weights=unpatch_weights)
    # runtime patches are removed whoever installed them (the model's flag:
    # e.g. the switch changed while it was loaded); without any, nothing to do
    if unpatch_weights and self.model.__dict__.get("_monoload_runtime", False):
        _remove_runtime_patches(self)
    if unpatch_weights:
        b = self.model.__dict__.get("_monoload_binding")
        if b is not None:
            b.device_cache.clear()
    return r


def _partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
    # The runtime patches already on the model compute with whatever the binding holds. A clone with the same
    # patches_uuid loads without unpatching or calling load() (native returns early when the weights are fully
    # loaded, only re-applying its forced hooks): bind to it first, so its patches, merge and hook state are the ones
    # in effect from here on (also inside the original, whose apply_hooks(forced) runs before it returns).
    if _active(self) and self.model.__dict__.get("_monoload_runtime", False):
        b = self.model.__dict__.get("_monoload_binding")
        if b is not None and self.model.current_weight_patches_uuid == self.patches_uuid:
            b.bind(self)
    return _ORIG["partially_load"](self, device_to, extra_memory=extra_memory, force_patch_weights=force_patch_weights)


def _patch_hooks(self, hooks):
    if not _active(self):
        return _ORIG["patch_hooks"](self, hooks)
    st = _binding(self)
    if self.model.current_weight_patches_uuid in (None, self.patches_uuid):
        st.bind(self)
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
    st = _binding(self)
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
    raise MonoloadError(msg("lora.internal_hook_write", key=key))


def _patch_cached_hook_weights(self, cached_weights, key, memory_counter):
    if not _active(self):
        return _ORIG["patch_cached_hook_weights"](self, cached_weights, key, memory_counter)
    raise MonoloadError(msg("lora.internal_hook_cache", key=key))


def _dynamic_load(self, *args, **kwargs):
    if _enabled(self):
        _check_dynamic(self)
    return _ORIG_DYNAMIC["load"](self, *args, **kwargs)


_REPLACEMENTS = {
    "patch_weight_to_device": _patch_weight_to_device,
    "load": _load,
    "partially_unload": _partially_unload,
    "unpatch_model": _unpatch_model,
    "partially_load": _partially_load,
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
