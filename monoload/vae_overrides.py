"""Per-VAE settings of the managed decode (the Monoload VAE Settings node).

A VAE copy made by with_settings() carries a dict of the settings set on it
(attribute ATTR); monoload/vae.py resolves every decode's settings item by
item: the copy's own value, else the environment variable, else Monoload's
default (vae.resolve_settings). This module imports neither torch nor
ComfyUI, so the node works (as a no-op) when Monoload is disabled.
"""

import copy

from .messages import msg

ATTR = "_monoload_vae_settings"
GIB = 1 << 30

SCHEME_CHOICES = ("default", "A", "B", "C", "D")
MODE_CHOICES = ("default", "auto", "layer 2 only", "native")
BUDGET_CHOICES = ("default", "unlimited", "custom")   # the node's budget dropdown; "custom" uses budget_gib
MODES = {"auto": "auto", "layer 2 only": "layer2", "layer2": "layer2", "native": "native"}


def overrides(vae):
    """The settings set on this VAE (a copy of the dict; {} for a plain VAE)."""
    return dict(getattr(vae, ATTR, None) or {})


def with_settings(vae, budget=0.0, gn_scheme="default", stripe_rows=0, mode="default"):
    """A copy of `vae` that decodes with these settings. The copy is shallow:
    the first-stage model, the model patcher (what ComfyUI's model management
    loads and unloads) and everything else are shared with `vae`, which is
    not changed. Values meaning "not set" (budget 0, "default", stripe_rows 0)
    keep what `vae` itself carries when it is such a copy (chained nodes),
    else the environment / default applies.

    budget       GiB (MONOLOAD_VAE_BUDGET); "unlimited": no budget for this
                 VAE whatever MONOLOAD_VAE_BUDGET says (the default stripe
                 policy)
    gn_scheme    "A" / "B" / "C" / "D" (MONOLOAD_VAE_GN_SCHEME, forces it)
    stripe_rows  output rows (MONOLOAD_VAE_STRIPE_ROWS, forces it)
    mode         "auto" (layer 1 where recognized), "layer 2 only"
                 (MONOLOAD_DISABLE_VAE_STRIPE=1), "native" (ComfyUI's decode)"""
    new = overrides(vae)
    if isinstance(budget, str) and budget.strip().lower() == "unlimited":
        new["budget"] = None
    elif budget is not None and float(budget) < 0:
        raise ValueError(msg("node.negative", item="budget", value=budget))
    elif budget:
        new["budget"] = int(round(float(budget) * GIB))   # 2.17 -> 2.17 GiB exactly as far as bytes go, not 1 byte under
    s = str(gn_scheme or "default").strip()
    if s.lower() != "default":
        if s.upper() not in SCHEME_CHOICES[1:]:
            raise ValueError(msg("node.bad_choice", item="GroupNorm scheme", value=gn_scheme, choices="/".join(SCHEME_CHOICES)))
        new["gn_scheme"] = s.upper()
    if stripe_rows is not None and int(stripe_rows) < 0:
        raise ValueError(msg("node.negative", item="stripe_rows", value=stripe_rows))
    if stripe_rows:
        new["stripe_rows"] = int(stripe_rows)
    m = str(mode or "default").strip().lower()
    if m != "default":
        if m not in MODES:
            raise ValueError(msg("node.bad_choice", item="mode", value=mode, choices="/".join(MODE_CHOICES)))
        new["mode"] = MODES[m]
    out = copy.copy(vae)
    setattr(out, ATTR, new)
    return out
