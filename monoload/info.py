"""The text of the Monoload Info node: what Monoload is doing and why.

report(vae=None, model=None) builds it fresh on every call:
  always       Monoload version / commit, the master switch;
  always, last every global default with its source (an environment
               variable, a runtime change by tests / bench, or built-in);
  vae          the VAE's effective decode settings item by item with their
               sources, and the record of the last decode of THIS VAE object
               (a Monoload VAE Settings copy is its own object): layer,
               scheme, stripes and rows, workspace, estimate, measured peak
               reserved / GTT increase, time, OOM retries -- or that it has
               not been decoded yet; settings the mode does not use are
               marked (native: budget / scheme / rows; layer 2 only: scheme /
               rows);
  model        the LoRA files the loader nodes attached (name, strength),
               the patched weights, the mode / merge / after-prompt settings
               with their sources (merge marked unused in native mode), and the
               current state in memory.
"""

import os
import time

from . import __version__, settings
from .messages import msg

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
            return msg("info.src_env", name=n, value=_env(n))
    return msg("info.src_builtin") if value == builtin else msg("info.src_runtime")


def header():
    commit, branch = git_commit()
    where = msg("info.commit", commit=commit) if commit else msg("info.commit_unknown")
    if branch:
        where += msg("info.branch", branch=branch)
    lines = [msg("info.version", version=__version__, where=where)]
    if settings.disabled():
        lines.append(msg("info.disabled"))
        return lines
    lines.append(msg("info.master", state=msg("info.master_on") if settings.master() else msg("info.master_off"),
                     src=_source(["MONOLOAD"], settings.master(), True)))
    return lines


def global_defaults():
    from . import hotpatch, release, vae
    from .vae_ops import fmt_bytes
    rows = []

    def row(label, value, src):
        rows.append("  {:<22s} {:<28s} [{}]".format(label, value, src))

    yes = lambda b: msg("info.yes") if b else msg("info.no")   # noqa: E731
    rows.append(msg("info.installed", a=yes(hotpatch.is_installed()), b=yes(release.is_installed()), c=yes(vae.is_installed())))
    rows.append(msg("info.globals"))
    row(msg("info.row_lora_mode"), msg("info.enable_monoload") if settings.master() else msg("info.native"),
        _source(["MONOLOAD"], settings.master(), True))
    row(msg("info.row_lora_merge"), msg("info.exact") if settings.exact() else msg("info.fused"), _source(["MONOLOAD_EXACT"], settings.exact(), False))
    row(msg("info.row_lora_after"), msg("info.keep") if settings.keep() else msg("info.release"),
        _source(["MONOLOAD_KEEP_LORA"], settings.keep(), False))
    mode, var = vae.global_mode()
    row(msg("info.row_vae_mode"), msg("vae.layer2_only") if mode == "layer2" else msg("info.native") if mode == "native" else msg("vae.auto"),
        msg("info.src_envvar", var=var) if var else msg("info.src_builtin"))
    bud = vae.budget()
    row(msg("info.row_vae_budget"), fmt_bytes(bud) if bud else msg("info.budget_none"), _source(["MONOLOAD_VAE_BUDGET"], bud, None))
    row(msg("info.row_vae_scheme"), "{}{}".format(vae.gn_scheme(), msg("info.forced") if vae.gn_scheme_forced() else ""),
        _source(["MONOLOAD_VAE_GN_SCHEME"], vae.gn_scheme_forced(), False))
    row(msg("info.row_vae_rows"), str(vae.stripe_rows() or msg("vae.auto")), _source(["MONOLOAD_VAE_STRIPE_ROWS"], vae.stripe_rows(), None))
    row(msg("info.row_vae_ws"), fmt_bytes(vae.workspace()), _source(["MONOLOAD_VAE_WORKSPACE"], vae.workspace(), vae.DEFAULT_WORKSPACE))
    return rows


def _src(s):
    return {"node": msg("info.src_node"), "env": "env", "default": msg("info.src_builtin")}[s]


def describe_decode(r):
    """One line for a decode record (vae.decode_record)."""
    if r is None:
        return msg("info.not_decoded")
    head = msg("info.decode_head", ago=time.time() - r.get("when", time.time()))
    strat = r.get("strategy")
    secs = r.get("seconds")
    t = msg("info.secs", secs=secs) if secs is not None else ""
    if strat == "native":
        return head + msg("info.decode_native", reason=r.get("reason"), t=t)
    if strat == "error":
        return head + msg("info.decode_error", budget=_gib(r.get("budget")))
    if strat == "layer1":
        what = msg("info.decode_l1", adapter=r.get("adapter"), n=r.get("stripes"), rows=r.get("rows"))
    elif strat == "layer2":
        what = msg("info.decode_l2")
    else:
        what = str(strat)
    est = (r.get("estimate") or {}).get("total")
    mem = r.get("mem") or {}
    measured = []
    if mem.get("reserved_peak") is not None:
        measured.append("reserved +{}".format(_gib(mem["reserved_peak"])))
    if mem.get("gtt_peak") is not None:
        measured.append("GTT +{}".format(_gib(mem["gtt_peak"])))
    return head + msg("info.decode_line", what=what, ws=_gib(r.get("workspace")), est=_gib(est), measured=", ".join(measured) or msg("info.no_gpu"),
                      t=t, retries=r.get("retries", 0))


def vae_section(v):
    from . import vae, vae_overrides
    own = vae_overrides.overrides(v)
    lines = [msg("info.vae_head", model=type(getattr(v, "first_stage_model", None)).__name__, copy=msg("info.vae_copy") if own else "")]
    if not vae.is_installed():
        lines.append(msg("info.vae_not_installed"))
    eff, src = vae.resolve_settings(v)
    scheme = eff["gn_scheme"] if eff["gn_forced"] or not eff["budget"] else msg("vae.by_budget")
    budget = _gib(eff["budget"]) if eff["budget"] else (msg("vae.unlimited") if src["budget"] == "node" else msg("info.budget_none"))
    mode_src = msg("info.src_envvar", var=eff["mode_env"]) if eff.get("mode_env") else _src(src["mode"])
    mode = {"layer2": msg("vae.layer2_only"), "native": msg("info.native"), "auto": msg("vae.auto")}.get(eff["mode"], eff["mode"])
    rows = str(eff["stripe_rows"] or msg("vae.auto"))
    if eff["mode"] == "native":      # ComfyUI's own decode: none of the other items is used
        unused = msg("info.unused_native")
        budget, scheme, rows = budget + unused, scheme + unused, rows + unused
    elif eff["mode"] == "layer2":    # no stripes: scheme and stripe height do not apply
        unused = msg("info.unused_layer2")
        scheme, rows = scheme + unused, rows + unused
    lines.append(msg("info.vae_settings", mode=mode, mode_src=mode_src, budget=budget, budget_src=_src(src["budget"]), scheme=scheme,
                     scheme_src=_src(src["gn_scheme"]), rows=rows, rows_src=_src(src["stripe_rows"])))
    r = vae.decode_record(v)
    lines.append("  " + describe_decode(r))
    shown = None   # the scheme to explain: the one the last decode used, else the one the settings fix
    if r is not None and r.get("strategy") == "layer1" and r.get("gn_scheme"):
        shown = r["gn_scheme"]
    elif eff["mode"] != "native" and eff["mode"] != "layer2" and (eff["gn_forced"] or not eff["budget"]):
        shown = eff["gn_scheme"]
    if shown in ("A", "B", "C", "D"):
        lines.append(msg("info.scheme_hint", scheme=shown, hint=msg("scheme." + shown)))
    return lines


def _runtime_patches(model):
    return sum(1 for m in model.modules() for a in ("weight_function", "bias_function")
               for f in (m.__dict__.get(a) or ()) if getattr(f, "is_monoload_patch", False))


def model_section(p):
    from . import lora_overrides as lo
    lines = [msg("info.model_head", model=type(getattr(p, "model", None)).__name__)]
    names = lo.lora_names(p)
    n_patched = len(getattr(p, "patches", {}) or {})
    n_hooks = len(getattr(p, "hook_patches", {}) or {})
    if names:
        lines.append(msg("info.lora_list", items=", ".join("{} x {:g}".format(x["name"], x["strength"]) for x in names)))
    elif n_patched:
        lines.append(msg("info.lora_other"))
    else:
        lines.append(msg("info.lora_none"))
    lines.append(msg("info.patched", n=n_patched, hooks=msg("info.hooks", n=n_hooks) if n_hooks else ""))
    eff, src = lo.resolve(p)
    if eff["mode"] == "native":      # ComfyUI's own LoRA handling: the merge path is not used
        from .messages import label
        eff = dict(eff, merge=label(eff["merge"]) + msg("info.unused_native"))
    lines.append(msg("info.settings", note=lo.note(eff, src)))
    state = msg("info.state_not_loaded")
    try:
        import comfy.model_management as mm
        loaded = [x.model for x in mm.current_loaded_models if x.model is not None and x.model.model is p.model]
        if p in loaded:
            rp = _runtime_patches(p.model)
            bk = len(p.backup)
            if rp:
                state = msg("info.state_runtime", n=rp)
            elif bk:
                state = msg("info.state_baked", n=bk)
            else:
                state = msg("info.state_clean")
        elif loaded:
            state = msg("info.state_other")
    except Exception:
        pass
    lines.append(msg("info.state", state=state))
    return lines


def report(vae=None, model=None):
    lines = header()
    if settings.disabled():
        return "\n".join(lines)
    if vae is not None:
        lines += vae_section(vae)
    if model is not None:
        lines += model_section(model)
    if vae is not None or model is not None:
        lines.append("")
    lines += global_defaults()   # always, at the end
    return "\n".join(lines)
