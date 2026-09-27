"""ModelPatcher for Monoload models: LoRA (and hook LoRA) are merged per layer
at the moment the layer runs, through the `weight_function` / `bias_function`
path that comfy.ops already provides for lowvram. Weights are never modified
in place and nothing is ever backed up.
"""

import logging
import weakref

import torch

import comfy.float
import comfy.lora
import comfy.model_management
import comfy.model_patcher
import comfy.utils
from comfy.model_patcher import LowVramPatch, ModelPatcher, get_key_weight

from .errors import MonoloadError, MonoloadUnsupportedError


def _identity(a, **kwargs):
    return a


class MonoloadRuntimePatch(LowVramPatch):
    """Weight function computing `patched(W)` for one key, on the fly.

    Numerics replicate ModelPatcher.patch_weight_to_device (the path that bakes
    LoRA into the weights under a full load) and patch_hook_weight_to_device,
    so results are bit-identical to native ComfyUI:
      base patches : W -> lora_compute_dtype -> calculate_weight -> stochastic_rounding(param dtype)
      hook patches : -> float32 -> calculate_weight(original_weights) -> stochastic_rounding(param dtype)
    and finally the cast to the compute dtype that cast_bias_weight asked for.
    """

    is_monoload_patch = True

    def __init__(self, key, patches, module, attr, hook_state):
        super().__init__(key, patches)
        self._module = weakref.ref(module)
        self.attr = attr
        self.hook_state = hook_state  # dict shared with the owning patcher: key -> hook patches

    def memory_required(self):
        return 0

    def prepare(self, destination, stream, copy=True, commit=True):
        return None

    def __call__(self, weight):
        base = self.patches.get(self.key)
        hooks = self.hook_state.get(self.key)
        if not base and not hooks:
            return weight
        module = self._module()
        param = getattr(module, self.attr)
        device = weight.device
        param_dtype = param.dtype
        seed = comfy.utils.string_to_seed(self.key)

        if base:
            temp = comfy.model_management.cast_to_device(param, device, comfy.model_management.lora_compute_dtype(device), copy=True)
            out = comfy.lora.calculate_weight(base, temp, self.key)
            w = comfy.float.stochastic_rounding(out, param_dtype, seed=seed)
            del temp, out
        else:
            w = comfy.model_management.cast_to_device(param, device, param_dtype, copy=False)

        if hooks:
            temp = comfy.model_management.cast_to_device(w, device, torch.float32, copy=True)
            original = {self.key: [(param, _identity)] + list(base or [])}
            out = comfy.lora.calculate_weight(hooks, temp, self.key, original_weights=original)
            w = comfy.float.stochastic_rounding(out, param_dtype, seed=seed)
            del temp, out

        return w.to(dtype=weight.dtype)


def _is_runtime_patch(f):
    return getattr(f, "is_monoload_patch", False)


class MonoloadModelPatcher(ModelPatcher):
    """Never backs up and never writes patched weights."""

    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        super().__init__(model, load_device, offload_device, size, weight_inplace_update)
        self.monoload_hook_state = {}

    # ---- installing / removing runtime patches -------------------------

    def _module_for_key(self, key):
        parts = key.rsplit(".", 1)
        if len(parts) != 2:
            raise MonoloadUnsupportedError("lora_non_comfy_ops_param", "patch 的目标不是某个层的参数", key=key)
        return comfy.utils.get_attr(self.model, parts[0]), parts[1]

    def _install_runtime_patch(self, key):
        module, attr = self._module_for_key(key)
        if not hasattr(module, "comfy_cast_weights") or attr not in ("weight", "bias"):
            raise MonoloadUnsupportedError(
                "lora_non_comfy_ops_param",
                "被 LoRA/patch 修改的参数所在的模块 {} 不是 comfy.ops 层，没有运行时合并路径；"
                "Monoload 不会退回到「改权重+备份」。".format(type(module).__name__),
                key=key)
        weight, set_func, convert_func = get_key_weight(self.model, key)
        if set_func is not None or convert_func is not None:
            raise MonoloadUnsupportedError("lora_quantized_param", "量化参数上的 LoRA v1 不支持", key=key)
        patches = list(self.patches.get(key, [])) + list(self.monoload_hook_state.get(key, []))
        new_shape = comfy.lora.calculate_shape(patches, weight, key)
        if tuple(new_shape) != tuple(weight.shape):
            raise MonoloadUnsupportedError(
                "lora_shape_change",
                "patch 会把权重形状从 {} 改成 {}，运行时合并无法支持".format(list(weight.shape), list(new_shape)),
                key=key)
        fn_attr = attr + "_function"
        funcs = [f for f in getattr(module, fn_attr, []) if not (_is_runtime_patch(f) and f.key == key)]
        # LoRA first: under a native full load it is baked into the weight, so
        # it comes before any weight_wrapper_patches.
        funcs.insert(0, MonoloadRuntimePatch(key, self.patches, module, attr, self.monoload_hook_state))
        setattr(module, fn_attr, funcs)

    def _remove_runtime_patches(self, keep=None):
        """Remove runtime patches (all, or those whose key is not in `keep`)."""
        for m in self.model.modules():
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

    def _upgrade_native_lowvram_patches(self):
        """Partial loads attach plain LowVramPatch objects (compute-dtype
        numerics); swap them for runtime patches so results stay bit-exact."""
        for n, m in self.model.named_modules():
            for attr in ("weight", "bias"):
                funcs = m.__dict__.get(attr + "_function", None)
                if funcs and any(type(f) is LowVramPatch for f in funcs):
                    key = "{}.{}".format(n, attr) if n else attr
                    setattr(m, attr + "_function", [f for f in funcs if type(f) is not LowVramPatch])
                    self._install_runtime_patch(key)

    def _assert_no_backup(self):
        if len(self.backup) > 0 or len(self.hook_backup) > 0:
            raise MonoloadError("[Monoload] 内部错误：出现了权重备份 {}".format(list(self.backup)[:5] + list(self.hook_backup)[:5]))

    # ---- ModelPatcher overrides ----------------------------------------

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        if key not in self.patches or return_weight:
            # Unpatched keys are a no-op natively; return_weight only computes
            # a temporary merged tensor and never writes or backs up.
            return super().patch_weight_to_device(key, device_to=device_to, inplace_update=inplace_update,
                                                  return_weight=return_weight, force_cast=force_cast)
        self._install_runtime_patch(key)
        return None

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        if force_patch_weights and len(self.patches) > 0:
            raise MonoloadUnsupportedError(
                "force_patch_weights",
                "有节点要求把 LoRA/patch 直接烘焙进权重（force_patch_weights，常见于保存/合并模型）。"
                "Monoload 模型只做运行时临时合并，不改权重、不备份。",
                key=next(iter(self.patches)))
        # Native load() wipes weight_function on every fully-loaded comfy.ops
        # module but skips modules already flagged comfy_patched_weights (their
        # patches are baked natively). Ours are not baked, so make sure every
        # patched module goes through patch_weight_to_device again.
        for key in self.patches:
            parts = key.rsplit(".", 1)
            if len(parts) == 2:
                m = comfy.utils.get_attr(self.model, parts[0])
                if hasattr(m, "comfy_patched_weights"):
                    del m.comfy_patched_weights
        super().load(device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load)
        self._upgrade_native_lowvram_patches()
        self._assert_no_backup()

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        if force_patch_weights and len(self.patches) > 0:
            raise MonoloadUnsupportedError("force_patch_weights", "Monoload 模型不支持 force_patch_weights", key=next(iter(self.patches)))
        freed = super().partially_unload(device_to, memory_to_free=memory_to_free, force_patch_weights=force_patch_weights)
        self._upgrade_native_lowvram_patches()
        self._assert_no_backup()
        return freed

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)
        if unpatch_weights:
            self._remove_runtime_patches()

    # ---- hooks: switch state, never write weights -----------------------

    def patch_hooks(self, hooks):
        with self.use_ejected():
            self.monoload_hook_state.clear()
            if hooks is not None:
                combined = self.get_combined_hook_patches(hooks=hooks)
                if combined:
                    model_keys = set(self.model_state_dict().keys())
                    for key, patches in combined.items():
                        if key not in model_keys:
                            logging.warning("Hook could not patch. Key does not exist in model: {}".format(key))
                            continue
                        self.monoload_hook_state[key] = patches
                    for key in self.monoload_hook_state:
                        if not self._has_runtime_patch(key):
                            self._install_runtime_patch(key)
            self._remove_runtime_patches(keep=set(self.patches) | set(self.monoload_hook_state))
            self.current_hooks = hooks

    def unpatch_hooks(self, whitelist_keys_set=None):
        with self.use_ejected():
            if whitelist_keys_set:
                for k in list(self.monoload_hook_state):
                    if k in whitelist_keys_set:
                        self.monoload_hook_state.pop(k)
            else:
                self.monoload_hook_state.clear()
                self.current_hooks = None
            if self.monoload_hook_state or self.patches:
                self._remove_runtime_patches(keep=set(self.patches) | set(self.monoload_hook_state))
            else:
                self._remove_runtime_patches()

    def _has_runtime_patch(self, key):
        module, attr = self._module_for_key(key)
        return any(_is_runtime_patch(f) and f.key == key for f in module.__dict__.get(attr + "_function", []))

    def patch_hook_weight_to_device(self, hooks, combined_patches, key, original_weights, memory_counter):
        if key not in combined_patches:
            return
        raise MonoloadError("[Monoload] 内部错误：hook 不应走写权重的路径（key={}）".format(key))

    def patch_cached_hook_weights(self, cached_weights, key, memory_counter):
        raise MonoloadError("[Monoload] 内部错误：hook 不应走缓存权重的路径（key={}）".format(key))
