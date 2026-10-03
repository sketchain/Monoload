"""Monoload's switches: the master switch and the global defaults.

MONOLOAD (the master switch, read at import; set_master() for tests):
  unset / 1   Monoload on: every model and VAE uses Monoload's default
              strategy (runtime LoRA merge, LoRA released after every prompt,
              managed VAE decode), adjusted by the advanced variables below;
  0           every model and VAE stays native ComfyUI. The hooks are still
              installed but pass every call straight to ComfyUI, so a
              workflow without Monoload nodes runs exactly as native; only
              what a Monoload node explicitly switches on for its own model /
              VAE uses Monoload.
MONOLOAD_DISABLE=1  nothing is installed at all (no hooks); the nodes pass
              their input through unchanged.

The advanced variables (MONOLOAD_EXACT, MONOLOAD_KEEP_LORA,
MONOLOAD_DISABLE_VAE, MONOLOAD_DISABLE_VAE_STRIPE, MONOLOAD_VAE_BUDGET,
MONOLOAD_VAE_GN_SCHEME, MONOLOAD_VAE_STRIPE_ROWS, MONOLOAD_VAE_WORKSPACE) are
global defaults. A value chosen explicitly on a node applies to that one
model / VAE and always beats them; item by item the order is
  node's explicit choice > advanced variable > Monoload's built-in default,
and a node item left at "default" (follow the global setting) inherits the
global one.

This module imports neither torch nor ComfyUI.
"""

import logging
import os

TRUE = ("1", "true", "yes", "on")
FALSE = ("0", "false", "no", "off")


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in TRUE


def _master_from_env():
    raw = os.environ.get("MONOLOAD", "").strip().lower()
    if raw == "" or raw in TRUE:
        return True
    if raw in FALSE:
        return False
    logging.warning("[Monoload] MONOLOAD={!r} not understood (1 = on, 0 = native ComfyUI); Monoload stays on".format(raw))
    return True


_MASTER = [_master_from_env()]


def master():
    """True: Monoload's strategy is the global default. False (MONOLOAD=0):
    everything native unless a node switches Monoload on for its model / VAE."""
    return _MASTER[0]


def set_master(on):
    """Tests / bench; MONOLOAD at import. Unload the models before switching."""
    _MASTER[0] = bool(on)


def disabled():
    """MONOLOAD_DISABLE=1: nothing installed, nodes pass through."""
    return env_flag("MONOLOAD_DISABLE")


def master_note():
    return "on" if master() else "off (MONOLOAD=0): native ComfyUI unless a Monoload node enables Monoload for its model / VAE"
