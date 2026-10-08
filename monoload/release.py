"""Release LoRA state after every prompt; the base models stay resident.

install() wraps execution.PromptExecutor.execute_async. When a prompt has
finished (successfully or not), release_after_prompt():

  1. loaded models: every LoadedModel whose patcher carries weight patches
     (LoRA, hook LoRA, model merges, ...) or a bypass-LoRA injection is
     unpatched in place -- runtime
     patches removed, device-side LoRA copies dropped, nothing moved -- and
     re-pointed at its base patcher (the nearest ancestor without weight
     patches, i.e. the loader node's output), with the model's
     current_weight_patches_uuid set to the base's. For ComfyUI the base is
     then simply "loaded": the next workflow that uses it loads nothing.
  2. output cache: entries whose outputs carry weight patches (a MODEL/CLIP
     patcher with patches or hook patches, a HOOKS group with weight hooks,
     conditioning carrying such hooks) are removed. Loader outputs (plain
     base patchers) and everything else stay cached.
  3. node instance cache: `loaded_lora` (LoraLoader, LoraLoaderModelOnly,
     CreateHookLora, LoraLoaderBypass, ...) is cleared.
  4. gc (LoRA clones die; LoadedModels of patch-free clones such as the
     CLIP clone of a UNet-only LoRA switch to their parent) + re-sync of
     current_weight_patches_uuid for patch-free loaded patchers +
     soft_empty_cache.

What is released is decided per patcher (lora_overrides.wants_release): a
Monoload LoRA Settings node's after_prompt (release / keep), else the global
default -- release for patchers Monoload drives unless MONOLOAD_KEEP_LORA=1,
keep for native ones (MONOLOAD=0, or mode native: native ComfyUI keeps the
LoRA state between prompts). Hook-LoRA groups (no patcher of their own) and
the loaders' file caches follow the global default (enabled()); the file
caches are also cleared whenever anything was released. With nothing to
release possible (global default keep and no node used in this process)
the wrapper returns at once.
"""

import gc
import logging
import time
import uuid
import weakref

import torch

import comfy.hooks
import comfy.model_management
import comfy.model_patcher
from comfy.model_patcher import ModelPatcher

from . import lora_overrides, settings
from .messages import msg

_ORIG = {}


def keep():
    """The global default: LoRA state kept between prompts (MONOLOAD_KEEP_LORA=1)."""
    return settings.keep()


def set_keep(on):
    """Tests; MONOLOAD_KEEP_LORA at import."""
    settings.set_keep(on)


def enabled():
    """The global default after a prompt: release (master switch on and not MONOLOAD_KEEP_LORA)."""
    return settings.master() and not settings.keep()
_MAX_DEPTH = 8


# ---------------------------------------------------------------------------
# what carries LoRA
# ---------------------------------------------------------------------------

BYPASS_INJECTION_KEY = "bypass_lora"  # comfy.sd.load_bypass_lora_for_models / LoraLoaderBypass


def patcher_has_weight_patches(p):
    """Weight patches, hook patches, or bypass-LoRA injections."""
    return (len(getattr(p, "patches", {}) or {}) > 0
            or len(getattr(p, "hook_patches", {}) or {}) > 0
            or BYPASS_INJECTION_KEY in (getattr(p, "injections", {}) or {}))


def _hook_group_has_weights(g):
    for h in getattr(g, "hooks", []) or []:
        if getattr(h, "hook_type", None) == comfy.hooks.EnumHookType.Weight:
            return True
    return False


def _releasable(p):
    return patcher_has_weight_patches(p) and lora_overrides.wants_release(p)


def carries_lora(value, depth=0, glob=None):
    """`value` holds LoRA state that is to be released: a patcher with weight
    patches whose settings say release, or a weight-hook group while the
    global default is release."""
    if glob is None:
        glob = enabled()
    if depth > _MAX_DEPTH or value is None or isinstance(value, (torch.Tensor, str, bytes, int, float, bool)):
        return False
    if isinstance(value, ModelPatcher):
        return _releasable(value)
    if isinstance(value, comfy.hooks.HookGroup):
        return glob and _hook_group_has_weights(value)
    if isinstance(value, comfy.hooks.Hook):
        return glob and getattr(value, "hook_type", None) == comfy.hooks.EnumHookType.Weight
    patcher = getattr(value, "patcher", None)
    if isinstance(patcher, ModelPatcher):  # CLIP, VAE, ...
        if _releasable(patcher):
            return True
        hooks = getattr(value, "apply_hooks_to_conds", None)
        return glob and hooks is not None and _hook_group_has_weights(hooks)
    if isinstance(value, dict):
        return any(carries_lora(v, depth + 1, glob) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(carries_lora(v, depth + 1, glob) for v in value)
    return False


# ---------------------------------------------------------------------------
# loaded models
# ---------------------------------------------------------------------------

def _find_base(p):
    q = p
    while q is not None and patcher_has_weight_patches(q):
        q = q.parent
    if q is not None and q.model is p.model:
        return q
    return None


def _model_has_runtime_patches(model):
    for m in model.modules():
        for attr in ("weight_function", "bias_function"):
            for f in m.__dict__.get(attr, None) or ():
                if getattr(f, "is_monoload_patch", False):
                    return True
    return False


def _release_loaded_models():
    n = 0
    for lm in list(comfy.model_management.current_loaded_models):
        p = lm.model
        if p is None:
            continue
        if not patcher_has_weight_patches(p) and not _model_has_runtime_patches(p.model):
            continue
        if not lora_overrides.wants_release(p):
            continue
        model = p.model
        if getattr(model, "model_lowvram", False):
            # Partially loaded (not the --gpu-only target): an in-place unpatch
            # would also wipe the native lowvram state of offloaded layers, so
            # unload the native way (weights go to the offload device).
            lm.model_unload(unpatch_weights=True)
            comfy.model_management.current_loaded_models[:] = [
                x for x in comfy.model_management.current_loaded_models if x is not lm]
            n += 1
            continue
        base = _find_base(p)
        # unpatch_model() also marks the model as not loaded (loaded weight
        # memory = 0, comfy_patched_weights flags deleted) even when nothing is
        # moved; the weights stay exactly where they are, so keep that state.
        loaded_mem = model.model_loaded_weight_memory
        offload_mem = model.model_offload_buffer_memory
        flagged = [m for m in model.modules() if getattr(m, "comfy_patched_weights", False) is True]
        p.unpatch_hooks()
        p.unpatch_model(device_to=None, unpatch_weights=True)  # in place: nothing is moved
        model.model_loaded_weight_memory = loaded_mem
        model.model_offload_buffer_memory = offload_mem
        for m in flagged:
            m.comfy_patched_weights = True
        if base is not None and base is not p:
            fin = getattr(lm, "_patcher_finalizer", None)
            if fin is not None:
                fin.detach()
                lm._patcher_finalizer = None
            lm._set_model(base)
            lm.device = base.load_device
            model.current_weight_patches_uuid = base.patches_uuid
        else:
            # no clean ancestor: make the next load re-evaluate everything
            model.current_weight_patches_uuid = uuid.uuid4()
        n += 1
    return n


# ---------------------------------------------------------------------------
# execution caches
# ---------------------------------------------------------------------------

def _iter_caches(cache):
    if cache is None or not hasattr(cache, "cache"):
        return
    yield cache
    for sub in list(getattr(cache, "subcaches", {}).values()):
        yield from _iter_caches(sub)


def _drop_key(cache, key):
    del cache.cache[key]
    for side in ("used_generation", "children", "timestamps"):
        d = getattr(cache, side, None)
        if isinstance(d, dict):
            d.pop(key, None)


def _release_outputs(outputs_cache):
    n = 0
    glob = enabled()
    for c in _iter_caches(outputs_cache):
        for key, entry in list(c.cache.items()):
            outs = getattr(entry, "outputs", entry)
            if carries_lora(outs, glob=glob):
                _drop_key(c, key)
                n += 1
    return n


def _release_objects(objects_cache):
    n = 0
    for c in _iter_caches(objects_cache):
        for obj in list(c.cache.values()):
            if getattr(obj, "loaded_lora", None) is not None:
                obj.loaded_lora = None
                n += 1
    return n


def _sync_clean_loaded_models():
    """A LoadedModel whose patcher carries no patches at all may still hold the
    patches_uuid of another clone: e.g. LoraLoader always clones the CLIP and
    add_patches() always rolls a new uuid, even when the LoRA has no
    text-encoder keys; once that clone is garbage, ComfyUI's finalizer points
    the LoadedModel back at the parent but leaves the model's
    current_weight_patches_uuid as it was, so the next prompt would run a full
    load() again. Under Monoload weights are never modified, so a model with
    no runtime patches and no native backup is in exactly the state of any
    patcher without patches: sync the uuid. Native ComfyUI (MONOLOAD=0, a
    Monoload LoRA Settings node with mode native) bakes the LoRA into the
    weights and keeps the originals in the backup, which every clone shares:
    there the weights still carry the dead clone's LoRA, so the uuid stays and
    the next load restores the backup."""
    n = 0
    for lm in comfy.model_management.current_loaded_models:
        p = lm.model
        if p is None or patcher_has_weight_patches(p) or _model_has_runtime_patches(p.model):
            continue
        if len(p.backup) or len(p.hook_backup):
            continue
        if p.model.current_weight_patches_uuid != p.patches_uuid:
            p.model.current_weight_patches_uuid = p.patches_uuid
            n += 1
    return n


def _ancestor_chains():
    """For every loaded model whose patcher is a clone: weak references to its
    ancestors, nearest first (taken before anything is released)."""
    chains = []
    for lm in comfy.model_management.current_loaded_models:
        p = lm.model
        refs = []
        q = getattr(p, "parent", None) if p is not None else None
        while q is not None and len(refs) < 64:
            refs.append(weakref.ref(q))
            q = getattr(q, "parent", None)
        if refs:
            chains.append((lm, refs))
    return chains


def _repoint_orphans(chains):
    """A LoadedModel follows its patcher's parent when the patcher dies
    (ComfyUI's finalizer, one level). When a clone and its parent die in the
    same garbage collection -- e.g. the Monoload LoRA Settings clone of a
    LoraLoader clone whose LoRA has no text-encoder keys, both patch-free,
    kept alive by an error's traceback until the collection -- the parent is
    already gone and the LoadedModel is left without a patcher while its model
    lives on in the base: ComfyUI then reports "memory leak with model ...".
    Point such entries at their nearest living ancestor of the same model."""
    n = 0
    for lm, refs in chains:
        if lm.model is not None or lm.real_model is None or lm.real_model() is None:
            continue
        for r in refs:
            a = r()
            if a is not None and a.model is lm.real_model():
                lm._set_model(a)
                lm.device = a.load_device
                n += 1
                break
    return n


def release_after_prompt(executor):
    glob = enabled()
    if not glob and not lora_overrides.used():
        return {"models": 0, "outputs": 0, "objects": 0, "synced": 0, "repointed": 0}
    t0 = time.perf_counter()
    chains = _ancestor_chains()
    n_models = _release_loaded_models()
    caches = getattr(executor, "caches", None)
    n_out = _release_outputs(getattr(caches, "outputs", None))
    n_obj = _release_objects(getattr(caches, "objects", None)) if glob or n_models or n_out else 0
    n_sync = n_rep = 0
    if n_models or n_out or n_obj:
        gc.collect()  # LoRA clones die here; LoadedModels of clean clones switch to their parents
        n_rep = _repoint_orphans(chains)
        n_sync = _sync_clean_loaded_models()
        comfy.model_management.soft_empty_cache()
        logging.info(msg("release.done", models=n_models, outputs=n_out, objects=n_obj, synced=n_sync, repointed=n_rep,
                         seconds=time.perf_counter() - t0))
    del chains
    return {"models": n_models, "outputs": n_out, "objects": n_obj, "synced": n_sync, "repointed": n_rep}


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def is_installed():
    return bool(_ORIG)


def install():
    import execution

    if _ORIG:
        return False
    orig = execution.PromptExecutor.execute_async
    _ORIG["execute_async"] = orig

    async def execute_async(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        try:
            return await orig(self, prompt, prompt_id, extra_data, execute_outputs)
        finally:
            try:
                release_after_prompt(self)
            except Exception:
                logging.exception(msg("release.failed"))

    execute_async.__wrapped__ = orig
    execution.PromptExecutor.execute_async = execute_async
    return True


def uninstall():
    import execution

    if not _ORIG:
        return False
    execution.PromptExecutor.execute_async = _ORIG.pop("execute_async")
    return True
