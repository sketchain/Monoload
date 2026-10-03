"""Per-model LoRA settings (the Monoload LoRA Settings node).

The node returns a clone of a ModelPatcher (MODEL, or the patcher of a CLIP)
whose model_options carry a dict under KEY with the items it set:
  mode          "enable" (Monoload's runtime merge for this model, also with
                MONOLOAD=0) / "native" (ComfyUI's own LoRA handling)
  merge         "fused" (the default path: fused fp16 addmm / relaxed) /
                "exact" (bit-identical to native ComfyUI)
  after_prompt  "release" / "keep" (the LoRA state after every prompt)
ModelPatcher.clone() deep-copies model_options, so the settings follow every
later clone -- LoraLoader / LoraLoaderModelOnly / CLIP.clone() included --
and the node works before or after the LoRA loaders. A clone whose settings
differ from its parent's gets a new patches_uuid, so ComfyUI reloads the
weights when it switches between the two (they share the model).

resolve() gives every item's effective value and source, item by item: the
node's explicit value > the global default (MONOLOAD, MONOLOAD_EXACT,
MONOLOAD_KEEP_LORA; monoload/settings.py) > Monoload's built-in default.
This module imports neither torch nor ComfyUI.
"""

import uuid

from . import settings

KEY = "monoload_lora"

MODE_CHOICES = ("default", "enable", "native")
MERGE_CHOICES = ("default", "fused", "exact")
AFTER_CHOICES = ("default", "release", "keep")

_USED = [False]   # a node has made settings in this process (the release check looks at patchers only then)


def overrides(patcher):
    """The settings set on this patcher (a copy; {} for a plain one)."""
    mo = getattr(patcher, "model_options", None)
    own = mo.get(KEY) if isinstance(mo, dict) else None
    return dict(own) if own else {}


def used():
    return _USED[0]


def _choice(name, value, choices):
    v = str(value or "default").strip().lower()
    if v not in choices:
        raise ValueError("{} {!r} is not one of {}".format(name, value, "/".join(choices)))
    return v


def _apply(patcher, mode, merge, after_prompt):
    """Set the items on `patcher` (a fresh clone) in place."""
    old = overrides(patcher)
    new = dict(old)
    for item, value, choices in (("mode", mode, MODE_CHOICES), ("merge", merge, MERGE_CHOICES), ("after_prompt", after_prompt, AFTER_CHOICES)):
        v = _choice(item, value, choices)
        if v != "default":
            new[item] = v
    patcher.model_options[KEY] = new
    if new != old:
        patcher.patches_uuid = uuid.uuid4()   # the weights are handled differently: reload when switching
    _USED[0] = True
    return patcher


def with_settings(model, mode="default", merge="default", after_prompt="default"):
    """A clone of the ModelPatcher `model` with these settings ("default" =
    keep what `model` carries, else the global setting). `model` is not changed."""
    return _apply(model.clone(), mode, merge, after_prompt)


def with_settings_clip(clip, mode="default", merge="default", after_prompt="default"):
    """A clone of the CLIP `clip` (CLIP.clone(): its patcher cloned) with these settings."""
    n = clip.clone()
    _apply(n.patcher, mode, merge, after_prompt)
    return n


def resolve(patcher):
    """({mode, merge, after_prompt}, {item: "node" / "env" / "default"}) for this patcher.
    mode "enable" / "native"; merge "fused" / "exact"; after_prompt "release" / "keep"
    (eff["after_from_mode"]: kept because the mode is native, source = the mode's).
    A native patcher keeps its LoRA after a prompt as native ComfyUI does,
    unless its node explicitly chose release."""
    own = overrides(patcher)
    eff, src = {}, {}
    if "mode" in own:
        eff["mode"], src["mode"] = own["mode"], "node"
    else:
        eff["mode"], src["mode"] = ("enable", "default") if settings.master() else ("native", "env")
    if "merge" in own:
        eff["merge"], src["merge"] = own["merge"], "node"
    else:
        eff["merge"], src["merge"] = ("exact", "env") if settings.exact() else ("fused", "default")
    if "after_prompt" in own:
        eff["after_prompt"], src["after_prompt"] = own["after_prompt"], "node"
    elif eff["mode"] == "native":
        eff["after_prompt"], src["after_prompt"] = "keep", src["mode"]
        eff["after_from_mode"] = True
    else:
        eff["after_prompt"], src["after_prompt"] = ("keep", "env") if settings.keep() else ("release", "default")
    return eff, src


def enabled(patcher):
    """Monoload's runtime merge drives this patcher (fast path of resolve()["mode"])."""
    mo = getattr(patcher, "model_options", None)
    own = mo.get(KEY) if isinstance(mo, dict) else None
    if own:
        m = own.get("mode")
        if m is not None:
            return m == "enable"
    return settings.master()


def merge_exact(patcher):
    """True / False when the patcher's node chose the merge, None = the global default at call time."""
    m = overrides(patcher).get("merge")
    return None if m is None else m == "exact"


def wants_release(patcher):
    return resolve(patcher)[0]["after_prompt"] == "release"


ENV_NAMES = {"mode": "MONOLOAD=0", "merge": "MONOLOAD_EXACT=1", "after_prompt": "MONOLOAD_KEEP_LORA=1"}


def note(eff, src):
    """The log's account of the settings and where each came from."""
    def s(item):
        if item == "after_prompt" and eff.get("after_from_mode"):
            return "as mode native, " + s("mode")
        if src[item] == "env":
            return "env {}".format(ENV_NAMES[item])
        return src[item]
    return "LoRA settings: mode {} ({}), merge {} ({}), after prompt {} ({})".format(
        eff["mode"], s("mode"), eff["merge"], s("merge"), eff["after_prompt"], s("after_prompt"))


# ---------------------------------------------------------------------------
# which LoRA files a patcher carries (for the Monoload Info node)
# ---------------------------------------------------------------------------

NAMES_KEY = "monoload_lora_names"   # model_options: [{"name", "strength"}, ...] added by the LoRA loader nodes
_ORIG = {}


def lora_names(patcher):
    """[{"name": file, "strength": s}, ...] in the order the loader nodes applied them."""
    mo = getattr(patcher, "model_options", None)
    return list(mo.get(NAMES_KEY) or ()) if isinstance(mo, dict) else []


def _note_name(patcher, name, strength, before):
    if patcher is None or patcher is before or not strength:
        return
    mo = getattr(patcher, "model_options", None)
    if isinstance(mo, dict):
        mo[NAMES_KEY] = list(mo.get(NAMES_KEY) or ()) + [{"name": name, "strength": float(strength)}]


def install_names(lora_loader_cls):
    """Wrap LoraLoader.load_lora (LoraLoaderModelOnly calls it too) so that the
    clones it returns remember the file name and strength in model_options.
    Only metadata: patches, uuids and numerics are untouched."""
    if _ORIG:
        return False
    orig = lora_loader_cls.__dict__["load_lora"]
    _ORIG["load_lora"] = (lora_loader_cls, orig)

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, *args, **kwargs):
        out = orig(self, model, clip, lora_name, strength_model, strength_clip, *args, **kwargs)
        try:
            m, c = out[0], out[1]
            _note_name(m, lora_name, strength_model, model)
            if c is not None and clip is not None and c is not clip:
                _note_name(getattr(c, "patcher", None), lora_name, strength_clip, getattr(clip, "patcher", None))
        except Exception:   # never break a loader for metadata
            pass
        return out

    load_lora.__wrapped__ = orig
    lora_loader_cls.load_lora = load_lora
    return True


def uninstall_names():
    if not _ORIG:
        return False
    cls, orig = _ORIG.pop("load_lora")
    cls.load_lora = orig
    return True
