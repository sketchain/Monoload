"""The text of the Monoload Info node: what Monoload is doing and why.

report(vae=None, model=None) builds it fresh on every call:
  always       Monoload version / commit, the master switch;
  no input     every global default with its source (an environment
               variable, a runtime change by tests / bench, or built-in);
  vae          the VAE's effective decode settings item by item with their
               sources, and the record of the last decode of THIS VAE object
               (a Monoload VAE Settings copy is its own object): layer,
               scheme, stripes and rows, workspace, estimate, measured peak
               reserved / GTT increase, time, OOM retries -- or that it has
               not been decoded yet;
  model        the LoRA files the loader nodes attached (name, strength),
               the patched weights, the mode / merge / after-prompt settings
               with their sources, and the current state in memory.
"""

import os
import time

from . import __version__, settings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_commit(root=ROOT):
    """(short commit, branch or None) of the plugin's checkout, or (None, None)."""
    try:
        gitdir = os.path.join(root, ".git")
        if os.path.isfile(gitdir):
            with open(gitdir) as f:
                line = f.read().strip()
            if line.startswith("gitdir:"):
                gitdir = os.path.normpath(os.path.join(root, line[7:].strip()))
        with open(os.path.join(gitdir, "HEAD")) as f:
            head = f.read().strip()
        if not head.startswith("ref:"):
            return head[:7], None
        ref = head[4:].strip()
        branch = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref
        p = os.path.join(gitdir, ref)
        if os.path.exists(p):
            with open(p) as f:
                return f.read().strip()[:7], branch
        p = os.path.join(gitdir, "packed-refs")
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0][:7], branch
        return None, branch
    except OSError:
        return None, None


def _gib(n):
    from .vae_ops import fmt_bytes
    return fmt_bytes(n) if n is not None else "n/a"


def _env(name):
    return os.environ.get(name, "").strip()


def _source(names, value, builtin):
    """Where a global default comes from: the first of `names` set in the
    environment, else a runtime change (tests / bench), else built-in."""
    for n in names:
        if _env(n):
            return "env {}={}".format(n, _env(n))
    return "built-in" if value == builtin else "set at runtime"


def header():
    commit, branch = git_commit()
    where = "commit {}".format(commit) if commit else "commit unknown"
    if branch:
        where += ", branch {}".format(branch)
    lines = ["Monoload {} ({})".format(__version__, where)]
    if settings.disabled():
        lines.append("MONOLOAD_DISABLE=1: nothing installed, ComfyUI is native and the Monoload nodes pass their inputs through")
        return lines
    lines.append("master switch MONOLOAD: {} [{}]".format(
        "on" if settings.master() else "off: native ComfyUI unless a Monoload node enables it",
        _source(["MONOLOAD"], settings.master(), True)))
    return lines


def global_defaults():
    from . import hotpatch, release, vae
    from .vae_ops import fmt_bytes
    rows = []

    def row(label, value, src):
        rows.append("  {:<22s} {:<28s} [{}]".format(label, value, src))

    rows.append("installed: runtime LoRA merge {}, per-prompt LoRA release {}, VAE decode wrapper {}".format(
        "yes" if hotpatch.is_installed() else "no", "yes" if release.is_installed() else "no", "yes" if vae.is_installed() else "no"))
    rows.append("global defaults (a Monoload node's explicit choice overrides them for its model / VAE):")
    row("LoRA mode", "enable (Monoload)" if settings.master() else "native", _source(["MONOLOAD"], settings.master(), True))
    row("LoRA merge", "exact (bit-identical)" if settings.exact() else "fused", _source(["MONOLOAD_EXACT"], settings.exact(), False))
    row("LoRA after prompt", "keep" if settings.keep() else "release", _source(["MONOLOAD_KEEP_LORA"], settings.keep(), False))
    mode, var = vae.global_mode()
    row("VAE mode", {"layer2": "layer 2 only"}.get(mode, mode), "env {}".format(var) if var else "built-in")
    bud = vae.budget()
    row("VAE budget", fmt_bytes(bud) if bud else "none (default policy)", _source(["MONOLOAD_VAE_BUDGET"], bud, None))
    row("VAE GroupNorm scheme", "{}{}".format(vae.gn_scheme(), " (forced)" if vae.gn_scheme_forced() else ""),
        _source(["MONOLOAD_VAE_GN_SCHEME"], vae.gn_scheme_forced(), False))
    row("VAE stripe rows", str(vae.stripe_rows() or "auto"), _source(["MONOLOAD_VAE_STRIPE_ROWS"], vae.stripe_rows(), None))
    row("VAE workspace", fmt_bytes(vae.workspace()), _source(["MONOLOAD_VAE_WORKSPACE"], vae.workspace(), vae.DEFAULT_WORKSPACE))
    return rows


_SRC = {"node": "node", "env": "env", "default": "built-in"}


def describe_decode(r):
    """One line for a decode record (vae.decode_record)."""
    if r is None:
        return "last decode: not decoded yet (connect images from its VAE Decode to run this node after the decode)"
    ago = time.time() - r.get("when", time.time())
    head = "last decode ({:.0f} s ago): ".format(ago)
    strat = r.get("strategy")
    secs = r.get("seconds")
    t = ", {:.2f} s".format(secs) if secs is not None else ""
    if strat == "native":
        return head + "native ComfyUI decode ({}){}".format(r.get("reason"), t)
    if strat == "error":
        return head + "error: no decode fits the budget {}".format(_gib(r.get("budget")))
    if strat == "layer1":
        what = "layer 1 ({}), {} stripes of {} rows".format(r.get("adapter"), r.get("stripes"), r.get("rows"))
    elif strat == "layer2":
        what = "layer 2 (op-level chunking)"
    else:
        what = str(strat)
    est = (r.get("estimate") or {}).get("total")
    mem = r.get("mem") or {}
    measured = []
    if mem.get("reserved_peak") is not None:
        measured.append("reserved +{}".format(_gib(mem["reserved_peak"])))
    if mem.get("gtt_peak") is not None:
        measured.append("GTT +{}".format(_gib(mem["gtt_peak"])))
    return head + "{}, workspace {}, estimate {}, measured peak {}{}, OOM retries {}".format(
        what, _gib(r.get("workspace")), _gib(est), ", ".join(measured) or "n/a (no GPU)", t, r.get("retries", 0))


def vae_section(v):
    from . import vae, vae_overrides
    own = vae_overrides.overrides(v)
    lines = ["VAE ({}{}):".format(type(getattr(v, "first_stage_model", None)).__name__,
                                  ", a Monoload VAE Settings copy" if own else "")]
    if not vae.is_installed():
        lines.append("  the managed decode is not installed: ComfyUI's own decode")
    eff, src = vae.resolve_settings(v)
    scheme = eff["gn_scheme"] if eff["gn_forced"] or not eff["budget"] else "chosen by the budget"
    budget = _gib(eff["budget"]) if eff["budget"] else ("unlimited" if src["budget"] == "node" else "none (default policy)")
    mode_src = "env {}".format(eff["mode_env"]) if eff.get("mode_env") else _SRC[src["mode"]]
    lines.append("  settings: mode {} [{}], budget {} [{}], GroupNorm scheme {} [{}], stripe rows {} [{}]".format(
        {"layer2": "layer 2 only"}.get(eff["mode"], eff["mode"]), mode_src, budget, _SRC[src["budget"]],
        scheme, _SRC[src["gn_scheme"]], eff["stripe_rows"] or "auto", _SRC[src["stripe_rows"]]))
    lines.append("  " + describe_decode(vae.decode_record(v)))
    return lines


def _runtime_patches(model):
    return sum(1 for m in model.modules() for a in ("weight_function", "bias_function")
               for f in (m.__dict__.get(a) or ()) if getattr(f, "is_monoload_patch", False))


def model_section(p):
    from . import lora_overrides as lo
    lines = ["MODEL ({}):".format(type(getattr(p, "model", None)).__name__)]
    names = lo.lora_names(p)
    n_patched = len(getattr(p, "patches", {}) or {})
    n_hooks = len(getattr(p, "hook_patches", {}) or {})
    if names:
        lines.append("  LoRA: " + ", ".join("{} x {:g}".format(x["name"], x["strength"]) for x in names))
    elif n_patched:
        lines.append("  LoRA: none from the LoRA loader nodes (other patches present)")
    else:
        lines.append("  LoRA: none")
    lines.append("  patched weights: {}{}".format(n_patched, ", hook-LoRA groups: {}".format(n_hooks) if n_hooks else ""))
    eff, src = lo.resolve(p)
    lines.append("  " + lo.note(eff, src).replace("LoRA settings: ", "settings: "))
    state = "not loaded"
    try:
        import comfy.model_management as mm
        loaded = [x.model for x in mm.current_loaded_models if x.model is not None and x.model.model is p.model]
        if p in loaded:
            rp = _runtime_patches(p.model)
            bk = len(p.backup)
            if rp:
                state = "loaded; Monoload runtime merge on {} weights (no weight backups)".format(rp)
            elif bk:
                state = "loaded; native: LoRA baked into the weights ({} backups)".format(bk)
            else:
                state = "loaded; no LoRA in effect"
        elif loaded:
            state = "the shared model is loaded for another clone"
    except Exception:
        pass
    lines.append("  now: " + state)
    return lines


def report(vae=None, model=None):
    lines = header()
    if settings.disabled():
        return "\n".join(lines)
    if vae is None and model is None:
        lines += global_defaults()
    if vae is not None:
        lines += vae_section(vae)
    if model is not None:
        lines += model_section(model)
    return "\n".join(lines)
