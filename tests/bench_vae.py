"""Benchmark: native VAE.decode vs Monoload's managed decode (op-level chunking).

Runs, in one process with one loaded VAE, every resolution x mode:
  native    native comfy.sd.VAE.decode (Monoload's VAE wrapper uninstalled).
            An OOM, or native's fallback to tiled decoding, is detected and
            reported as OOM (the tiled fallback is aborted, never counted as a
            native result).
  monoload  Monoload's managed decode as the plugin runs it: layer 1 (stripe
            decoding) for recognized decoders (Wan 2.1 single frame), layer 2
            (conv row blocks + attention query blocks) for the rest; own memory
            estimate, OOM -> smaller blocks.
  monoload-l2  Monoload with layer 1 switched off: op-level chunking only
            (what monoload was in phase 1), for comparison.
  monoload-r<N>  monoload with the layer-1 stripe core height forced to N
            output rows (--stripe-rows adds these for a sweep).
  native2   native again: how much the GPU differs from itself.
and reports per run (printed as it happens) and in summary tables:
  time      cold = first run of that mode at that resolution, warm = median
            of --warm further runs (torch.cuda.synchronize around each);
  memory    torch.cuda max_memory_allocated / max_memory_reserved (peak stats
            reset before each run, empty_cache before each run), absolute and
            as increase over the value before the run; GTT (and VRAM) peak from
            a sampler thread reading /sys/class/drm/card*/device/mem_info_*
            every --sample-ms; cgroup memory.peak (reset per run where the
            kernel supports it) and the sampled memory.current peak;
  accuracy  monoload vs native and native2 vs native, when both succeeded:
            the raw decoder output (before process_output) and the clamped
            fp32 pixels: max|Δ|, mean|Δ|, RMSE, PSNR, p99|Δ|; the same for the
            output rows next to row-block boundaries (±--boundary-rows around
            every conv block boundary, mapped to output rows) vs all other
            rows; per-channel mean shift; NaN/Inf counts.
--fp32-ref: the same VAE loaded a second time in fp32 (same loader, vae_dtype
forced to fp32) is decoded once per resolution after the bf16 modes, as the
"true" value: native (bf16) vs fp32 and monoload (bf16) vs fp32 with the same
metrics (and the same block-boundary rows), so outliers of monoload vs native
can be told apart: bf16 noise (both equally far from fp32) or added by the
chunking (monoload farther). The fp32 decode is tried natively and with
Monoload's chunking (--fp32-impl); native fp32 needs about twice the bf16
memory (~90 GiB at 4K) and is skipped when it runs out of memory, the chunked
fp32 decode is then the reference; when both succeed, their difference is
printed as a sanity check. Also printed: the --outliers pixels where monoload
and native differ most (batch/row/column/channel) with the fp32, native and
monoload values there, and, over all pixels with |monoload - native| >
--outlier-threshold, how often monoload or native is closer to fp32.
--profile: one native decode (--profile-modes) per --profile-res under torch.profiler
(profile_memory=True, record_shapes, with_stack): the operators allocating the
most device memory and the largest single allocations with their shapes and
Python stacks; then one more native decode under the CUDA caching allocator's
memory history: the live allocations at the moment of the peak, largest first,
with their stacks (does im2col / Slow2d's columns dominate?).

Run inside the ComfyUI container, in a separate process; free the server's
models first:
  curl -X POST http://127.0.0.1:8188/free -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}'
  cd /opt/ComfyUI/custom_nodes/monoload
  python tests/bench_vae.py --checkpoint waiIllustriousSDXL_v170.safetensors
  python tests/bench_vae.py --vae ae.safetensors --profile
  python tests/bench_vae.py --vae qwen_image_vae.safetensors --res 1344x768,2688x1536
  python tests/bench_vae.py --checkpoint waiIllustriousSDXL_v170.safetensors --modes native,monoload --warm 0 --fp32-ref

ComfyUI launch args: $COMFY_ARGS if set, otherwise those of the container's
main process (PID 1), otherwise "--cpu".
"""

import argparse
import gc
import glob
import json
import logging
import math
import os
import re
import shlex
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if "COMFY_ARGS" not in os.environ:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from monoload import comfy_env as _ce
    _auto = _ce.pid1_comfy_args()
    os.environ["COMFY_ARGS"] = shlex.join(_auto) if _auto is not None else "--cpu"

import torch  # noqa: E402

from common import COMFY_ARGS, REPO, ROOT  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402
import nodes  # noqa: E402
from monoload import comfy_env, vae as mvae, vae_ops  # noqa: E402
from monoload.errors import MonoloadVAEOOMError  # noqa: E402

GIB = 1024 ** 3


def gib(n):
    return "   n/a" if n is None else "{:6.2f}".format(n / GIB)


# ---------------------------------------------------------------------------
# host-side memory: GTT / VRAM (amdgpu sysfs), cgroup v2
# ---------------------------------------------------------------------------

def drm_files(kind):
    """{device realpath: sysfs file} for every DRM card exposing mem_info_<kind> (no card number hard-coded)."""
    seen = {}
    for d in sorted(glob.glob("/sys/class/drm/card*")):
        if re.search(r"/card\d+$", d):
            p = os.path.join(d, "device", "mem_info_" + kind)
            if os.path.exists(p):
                seen.setdefault(os.path.realpath(os.path.join(d, "device")), p)
    return seen


def read_sum(paths):
    if not paths:
        return None
    t = 0
    for p in paths:
        try:
            with open(p) as f:
                t += int(f.read())
        except (OSError, ValueError):
            return None
    return t


def cgroup_dir():
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                if line.startswith("0::"):
                    d = "/sys/fs/cgroup" + line.strip()[3:]
                    if os.path.exists(os.path.join(d, "memory.current")):
                        return d
    except OSError:
        pass
    return "/sys/fs/cgroup" if os.path.exists("/sys/fs/cgroup/memory.current") else None


class Sampler:
    """Background thread: peak of GTT used, VRAM used, cgroup memory.current."""

    def __init__(self, interval):
        self.interval = interval
        self.gtt = list(drm_files("gtt_used").values())
        self.vram = list(drm_files("vram_used").values())
        cg = cgroup_dir()
        self.cg_current = os.path.join(cg, "memory.current") if cg else None
        self.cg_peak = os.path.join(cg, "memory.peak") if cg and os.path.exists(os.path.join(cg, "memory.peak")) else None

    def read(self):
        return {"gtt": read_sum(self.gtt), "vram": read_sum(self.vram), "cg": read_sum([self.cg_current] if self.cg_current else [])}

    def __enter__(self):
        self.base = self.read()
        self.peak = dict(self.base)
        self.n = 0
        self._stop = threading.Event()
        self._peak_fd = None
        self.cg_peak_reset = False
        if self.cg_peak:
            try:  # Linux >= 6.12: a write resets memory.peak for reads through this fd
                self._peak_fd = open(self.cg_peak, "r+")
                self._peak_fd.write("reset\n")
                self._peak_fd.flush()
                self.cg_peak_reset = True
            except OSError:
                if self._peak_fd is not None:
                    self._peak_fd.close()
                self._peak_fd = None
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            cur = self.read()
            self.n += 1
            for k, v in cur.items():
                if v is not None and (self.peak[k] is None or v > self.peak[k]):
                    self.peak[k] = v
            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        cur = self.read()
        for k, v in cur.items():
            if v is not None and (self.peak[k] is None or v > self.peak[k]):
                self.peak[k] = v
        self.cg_peak_value = None
        if self._peak_fd is not None:
            try:
                self._peak_fd.seek(0)
                self.cg_peak_value = int(self._peak_fd.read())
            except (OSError, ValueError):
                pass
            self._peak_fd.close()
        elif self.cg_peak:
            self.cg_peak_value = read_sum([self.cg_peak])  # lifetime peak of the cgroup (not reset)
        return False


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_vae(a):
    if a.checkpoint:
        path = folder_paths.get_full_path_or_raise("checkpoints", a.checkpoint)
        out = comfy.sd.load_checkpoint_guess_config(path, output_vae=True, output_clip=False, output_model=False,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))
        v = out[2]
        if v is None:
            raise SystemExit("checkpoint {} has no VAE".format(a.checkpoint))
        return v, "checkpoint " + a.checkpoint
    return nodes.VAELoader().load_vae(a.vae)[0], "VAELoader " + a.vae


def load_vae_fp32(a):
    """The same VAE through the same loader, with vae_dtype forced to fp32
    (VAE.__init__ asks model_management.vae_dtype when no dtype is given)."""
    orig = mm.vae_dtype
    mm.vae_dtype = lambda *args, **kwargs: torch.float32
    try:
        v, _ = load_vae(a)
    finally:
        mm.vae_dtype = orig
    if v.vae_dtype != torch.float32:
        raise SystemExit("could not load an fp32 copy of the VAE (got {})".format(v.vae_dtype))
    return v


def find_latent_file(name):
    cands = [name, os.path.join(folder_paths.get_input_directory(), name), os.path.join(folder_paths.get_output_directory(), name)]
    for c in cands:
        if os.path.isfile(c):
            return c
    raise SystemExit("latent file not found: {} (tried {})".format(name, cands))


def load_latent(path):
    """A SaveLatent .latent file, scaled exactly as the LoadLatent node does."""
    import safetensors.torch
    lat = safetensors.torch.load_file(path, device="cpu")
    mult = 1.0 if "latent_format_version_0" in lat else 1.0 / 0.18215
    return lat["latent_tensor"].float() * mult


def random_latent(vae, w, h, seed, std):
    r = vae.spacial_compression_decode()
    if w % r or h % r:
        raise SystemExit("resolution {}x{} is not a multiple of {}".format(w, h, r))
    shape = [1, vae.latent_channels]
    if vae.latent_dim == 3:
        shape.append(1)
    shape += [h // r, w // r]
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * std


# ---------------------------------------------------------------------------
# one decode, measured
# ---------------------------------------------------------------------------

class TiledFallback(Exception):
    pass


class UnloadCounter:
    """Counts the models comfy.model_management.free_memory unloads (native's
    memory_used_decode estimate can unload other models -- in this process:
    the VAE itself, which then loads again)."""

    def __enter__(self):
        self.n = 0
        self._orig = mm.free_memory
        me = self

        def free_memory(*a, **kw):
            out = me._orig(*a, **kw)
            me.n += len(out or [])
            return out
        mm.free_memory = free_memory
        return self

    def __exit__(self, *exc):
        mm.free_memory = self._orig
        return False


class NativeGuard:
    """Detects native's OOM -> tiled fallback and aborts it (never counted as a native result)."""

    NAMES = ("decode_tiled_", "decode_tiled_1d", "decode_tiled_3d", "_decode_tiled_owned")

    def __enter__(self):
        self.saved = {n: comfy.sd.VAE.__dict__[n] for n in self.NAMES if n in comfy.sd.VAE.__dict__}

        def abort(*a, **kw):
            raise TiledFallback("native decode ran out of memory and fell back to tiled decoding")
        for n in self.saved:
            setattr(comfy.sd.VAE, n, abort)
        return self

    def __exit__(self, *exc):
        for n, f in self.saved.items():
            setattr(comfy.sd.VAE, n, f)
        return False


def is_cuda():
    return mm.get_torch_device().type == "cuda"


def sync():
    if is_cuda():
        torch.cuda.synchronize()


def decode_once(vae, latent, capture, sampler_interval):
    """Returns (row dict, raw output on CPU or None, pixels on CPU or None)."""
    raw_store = []
    po = vae.process_output
    if capture:
        def capture_po(img):
            raw_store.append(img.to("cpu", dtype=torch.float32, copy=True))
            return po(img)
        vae.process_output = capture_po
    gc.collect()
    if is_cuda():
        sync()
        torch.cuda.empty_cache()
        base_alloc = torch.cuda.memory_allocated()
        base_res = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
    row = {"status": "ok"}
    out = None
    try:
        with Sampler(sampler_interval) as smp, NativeGuard(), UnloadCounter() as unl:
            t0 = time.perf_counter()
            try:
                with torch.inference_mode():
                    out = vae.decode(latent)
                sync()
            except TiledFallback as e:
                row["status"] = "OOM(tiled)"
                row["error"] = str(e)
            except MonoloadVAEOOMError as e:
                row["status"] = "OOM"
                row["error"] = str(e).splitlines()[0][:200]
            except Exception as e:
                if mm.is_oom(e):
                    row["status"] = "OOM"
                    row["error"] = "{}: {}".format(type(e).__name__, str(e).splitlines()[0][:200] if str(e) else "")
                else:
                    raise
            row["seconds"] = time.perf_counter() - t0
    finally:
        vae.process_output = po
    if is_cuda():
        row["alloc_peak"] = torch.cuda.max_memory_allocated()
        row["res_peak"] = torch.cuda.max_memory_reserved()
        row["alloc_delta"] = row["alloc_peak"] - base_alloc
        row["res_delta"] = row["res_peak"] - base_res
    for k in ("gtt", "vram", "cg"):
        b, p = smp.base[k], smp.peak[k]
        row[k + "_peak"] = p
        row[k + "_delta"] = (p - b) if (p is not None and b is not None) else None
    row["cg_mempeak"] = smp.cg_peak_value
    row["cg_mempeak_delta"] = (smp.cg_peak_value - smp.base["cg"]) if (smp.cg_peak_value is not None and smp.base["cg"] is not None and smp.cg_peak_reset) else None
    row["samples"] = smp.n
    row["unloaded"] = unl.n
    px = raw = None
    if out is not None and capture:
        px = out.to("cpu", dtype=torch.float32, copy=True)
        if raw_store:
            raw = torch.cat(raw_store, 0)
            raw = raw.movedim(1, -1)  # same layout as the returned pixels
    del out
    return row, raw, px


_DEFAULT_ROWS = mvae.stripe_rows()


def configure(mode):
    """Install / uninstall Monoload's VAE wrapper and set layer 1 for `mode`."""
    if not mode.startswith("monoload"):
        mvae.uninstall()
        return
    mvae.install()
    m = re.match(r"monoload-r(\d+)", mode)
    if mode.startswith("monoload-l2"):
        mvae.set_stripe(False)
        mvae.set_stripe_rows(_DEFAULT_ROWS)
    elif m:
        mvae.set_stripe(True)
        mvae.set_stripe_rows(int(m.group(1)))
    else:
        mvae.set_stripe(True)
        mvae.set_stripe_rows(_DEFAULT_ROWS)


def mono_line(m):
    """One line on what Monoload did in a run."""
    if m.get("strategy") == "layer1":
        e = m["estimate"]
        st = m.get("stats", {})
        return ("layer 1 ({}): {} stripes of {} rows, recompute {:.2f}x, checkpoint {}, {} (target {}), workspace {}, {} OOM retries; "
                "estimate {} GiB (prefix {} / stripes {} / persistent {}; native {}); {} Conv3d calls as conv2d, cache emptied {}x").format(
            m.get("adapter"), m["stripes"], m["rows"], m["recompute"], vae_ops.fmt_bytes(m["checkpoint_bytes"]), m.get("policy"), vae_ops.fmt_bytes(m["budget"]),
            vae_ops.fmt_bytes(m["workspace"]), m["retries"], gib(e["total"]).strip(), vae_ops.fmt_bytes(e["prefix"]), vae_ops.fmt_bytes(e["stripes"]),
            vae_ops.fmt_bytes(e["persistent"]), gib(m.get("native_estimate")).strip(), st.get("conv3d_as_2d"), st.get("cache_releases"))
    st = m.get("stats", {})
    return ("layer 2 ({}): estimate {} GiB (native {}), workspace {}, {} OOM retries; conv {} of {} calls in {} row blocks "
            "(largest block workspace {}, largest whole-conv workspace {}); attention {} call(s), query block {} of {} tokens").format(
        (m.get("layer1") or "")[:90], gib(m["estimate"]["total"]).strip(), gib(m.get("native_estimate")).strip(), vae_ops.fmt_bytes(m["workspace"]),
        m["retries"], st.get("conv_chunked"), st.get("conv_calls"), st.get("conv_blocks"), vae_ops.fmt_bytes(st.get("conv_ws_max_block")),
        vae_ops.fmt_bytes(st.get("conv_ws_max_full")), st.get("attn_calls"), st.get("attn_rows_min"), st.get("attn_tokens_max"))


def boundary_stats(m):
    """Rows where blocks meet, for the boundary / elsewhere statistics: layer-1
    stripe boundaries, or layer-2 conv block boundaries."""
    if not m:
        return None
    if m.get("strategy") == "layer1":
        return {"conv_boundaries": m.get("boundaries", [])}
    return m.get("stats")


def run_mode(vae, latent, mode, a, label):
    """cold + warm runs of one mode; returns summary dict and the captured tensors of the last successful run."""
    configure(mode)
    runs = []
    raw = px = None
    total = 1 + a.warm
    for i in range(total):
        capture = (i == total - 1) or a.warm == 0
        row, r, p = decode_once(vae, latent, capture, a.sample_ms / 1000.0)
        row["run"] = "cold" if i == 0 else "warm{}".format(i)
        if mode.startswith("monoload"):
            last = mvae.last_decode()
            row["mono"] = last
        runs.append(row)
        print("{:13s} {:>11s} {:5s} {:10s} {:8.3f}s | alloc peak {} (+{}) reserved {} (+{}) | GTT peak +{} | VRAM +{} | cgroup peak +{} (sampled +{}) GiB | "
              "models unloaded {}{}".format(
            mode, label, row["run"], row["status"], row["seconds"], gib(row.get("alloc_peak")), gib(row.get("alloc_delta")).strip(),
            gib(row.get("res_peak")), gib(row.get("res_delta")).strip(), gib(row.get("gtt_delta")).strip(), gib(row.get("vram_delta")).strip(),
            gib(row.get("cg_mempeak_delta")).strip(), gib(row.get("cg_delta")).strip(), row["unloaded"],
            ("  [" + row["error"] + "]") if row.get("error") else ""), flush=True)
        if mode.startswith("monoload") and row["status"] == "ok":
            m = row["mono"]
            print("{:13s} {:>11s}       {}; measured alloc +{} / reserved +{} GiB".format(
                "", "", mono_line(m), gib(row.get("alloc_delta")).strip(), gib(row.get("res_delta")).strip()), flush=True)
        if r is not None or p is not None:
            raw, px = r, p
        if row["status"] != "ok":
            break
    ok = [r for r in runs if r["status"] == "ok"]
    summary = {"mode": mode, "res": label, "status": runs[0]["status"] if not ok else ("ok" if len(ok) == len(runs) else "partial"),
               "cold": runs[0]["seconds"], "warm": statistics.median([r["seconds"] for r in runs[1:] if r["status"] == "ok"]) if len(ok) > 1 else None,
               "runs": runs}
    for k in ("alloc_peak", "alloc_delta", "res_peak", "res_delta", "gtt_delta", "vram_delta", "cg_delta", "cg_mempeak_delta", "unloaded"):
        vals = [r.get(k) for r in runs if r.get(k) is not None]
        summary[k] = max(vals) if vals else None
    if mode.startswith("monoload") and ok:
        summary["mono"] = ok[-1]["mono"]
    return summary, raw, px


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------

def err_stats(a, b, psnr_range):
    d = (a.double() - b.double())
    ad = d.abs()
    n = ad.numel()
    if n == 0:
        return None
    rmse = float(torch.sqrt((d * d).mean()))
    flat = ad.flatten().float()
    k = max(1, int(math.ceil(0.99 * n)))
    if n > 16_000_000:
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))[:16_000_000]
        p99 = float(torch.quantile(flat[idx], 0.99))
    else:
        p99 = float(flat.kthvalue(k).values)
    return {"max": float(ad.max()), "mean": float(ad.mean()), "rmse": rmse,
            "psnr": (20 * math.log10(psnr_range / rmse)) if rmse > 0 else float("inf"), "p99": p99}


def boundary_rows(stats, h_out, width):
    rows = set()
    for h_l, o0 in stats.get("conv_boundaries", []):
        r = int(round(o0 * h_out / h_l))
        for x in range(r - width, r + width):
            if 0 <= x < h_out:
                rows.add(x)
    return sorted(rows)


def compare(ref, other, mono_stats, width):
    """ref/other: (raw, px) on CPU in NHWC (raw may be None). Returns dict of metrics."""
    raw_r, px_r = ref
    raw_o, px_o = other
    res = {}
    # flatten frames of a 5D (B,T,H,W,C) output into the batch
    def nhwc(t):
        return None if t is None else t.reshape(-1, t.shape[-3], t.shape[-2], t.shape[-1])
    raw_r, px_r, raw_o, px_o = map(nhwc, (raw_r, px_r, raw_o, px_o))
    res["px"] = err_stats(px_o, px_r, 1.0)
    if raw_r is not None and raw_o is not None:
        res["raw"] = err_stats(raw_o, raw_r, 2.0)
        res["raw_nonfinite"] = (int((~torch.isfinite(raw_r)).sum()), int((~torch.isfinite(raw_o)).sum()))
        res["raw_ch_shift"] = [float(x) for x in (raw_o.double() - raw_r.double()).mean(dim=(0, 1, 2))]
    res["px_nonfinite"] = (int((~torch.isfinite(px_r)).sum()), int((~torch.isfinite(px_o)).sum()))
    res["px_ch_shift"] = [float(x) for x in (px_o.double() - px_r.double()).mean(dim=(0, 1, 2))]
    if mono_stats:
        h = px_r.shape[1]
        rows = boundary_rows(mono_stats, h, width)
        if rows:
            idx = torch.tensor(rows)
            mask = torch.zeros(h, dtype=torch.bool)
            mask[idx] = True
            res["boundary_rows"] = len(rows)
            res["px_boundary"] = err_stats(px_o[:, mask], px_r[:, mask], 1.0)
            res["px_interior"] = err_stats(px_o[:, ~mask], px_r[:, ~mask], 1.0) if (~mask).any() else None
            if raw_r is not None and raw_o is not None:
                res["raw_boundary"] = err_stats(raw_o[:, mask], raw_r[:, mask], 2.0)
                res["raw_interior"] = err_stats(raw_o[:, ~mask], raw_r[:, ~mask], 2.0) if (~mask).any() else None
    return res


def boundary_centers(stats, h_out):
    """Block boundaries of all chunked convs, mapped to output rows."""
    if not stats:
        return []
    return sorted({int(round(o0 * h_out / h_l)) for h_l, o0 in stats.get("conv_boundaries", [])})


def outliers(native, mono, ref, mono_stats, k, thr):
    """native / mono / ref: (raw, px) on CPU; ref = fp32 or None. The k pixels
    where monoload and native differ most, with the three values there, and
    over every pixel with |mono - native| > thr: how often each bf16 result is
    closer to the reference."""
    def nhwc(t):
        return None if t is None else t.reshape(-1, t.shape[-3], t.shape[-2], t.shape[-1])
    raw_n, px_n = map(nhwc, native)
    raw_m, px_m = map(nhwc, mono)
    raw_f, px_f = map(nhwc, ref) if ref is not None else (None, None)
    d = (px_m - px_n).abs()
    flat = d.flatten()
    vals, idx = torch.topk(flat, min(k, flat.numel()))
    N, H, W, C = px_n.shape
    centers = boundary_centers(mono_stats, H)
    rows = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        b, rem = divmod(i, H * W * C)
        y, rem = divmod(rem, W * C)
        x, c = divmod(rem, C)
        row = {"b": b, "y": y, "x": x, "c": c, "diff": v, "native": float(px_n[b, y, x, c]), "mono": float(px_m[b, y, x, c])}
        if raw_n is not None and raw_m is not None:
            row["raw_native"], row["raw_mono"] = float(raw_n[b, y, x, c]), float(raw_m[b, y, x, c])
        if px_f is not None:
            row["fp32"] = float(px_f[b, y, x, c])
            if raw_f is not None:
                row["raw_fp32"] = float(raw_f[b, y, x, c])
        row["boundary_dist"] = min((abs(y - r) for r in centers), default=None)
        rows.append(row)
    res = {"top": rows, "threshold": thr, "count": int((d > thr).sum())}
    if px_f is not None and res["count"]:
        m = d > thr
        en = (px_n - px_f).abs()[m]
        em = (px_m - px_f).abs()[m]
        res["mono_closer"] = int((em < en).sum())
        res["native_closer"] = int((en < em).sum())
        res["mean_err_native"] = float(en.double().mean())
        res["mean_err_mono"] = float(em.double().mean())
    return res


def print_outliers(o, label, has_ref):
    print("{:26s} {:>11s} the {} pixels where monoload and native differ most (pixel values in [0, 1]; raw in brackets){}".format(
        "outliers", label, len(o["top"]), "" if has_ref else " -- add --fp32-ref for the fp32 values"))
    print("{:26s} {:>11s}   {:>3s} {:>5s} {:>5s} {:>2s} | {:>8s} | {:>18s} {:>18s} {:>18s} | {:>8s} {:>8s} | {:>6s}".format(
        "", "", "b", "row", "col", "c", "|m-n|", "fp32", "native", "monoload", "|n-fp32|", "|m-fp32|", "bdist"))
    for r in o["top"]:
        def v(px, raw):
            if px is None:
                return "{:>18s}".format("n/a")
            return "{:>18s}".format("{:.4f} [{:+.4f}]".format(px, raw) if raw is not None else "{:.4f}".format(px))
        f = r.get("fp32")
        print("{:26s} {:>11s}   {:3d} {:5d} {:5d} {:2d} | {:8.4f} | {} {} {} | {:>8s} {:>8s} | {:>6s}".format(
            "", "", r["b"], r["y"], r["x"], r["c"], r["diff"], v(f, r.get("raw_fp32")), v(r["native"], r.get("raw_native")),
            v(r["mono"], r.get("raw_mono")), "{:.4f}".format(abs(r["native"] - f)) if f is not None else "n/a",
            "{:.4f}".format(abs(r["mono"] - f)) if f is not None else "n/a",
            str(r["boundary_dist"]) if r["boundary_dist"] is not None else "n/a"))
    line = "{:26s} {:>11s} {} pixel values with |monoload - native| > {}".format("", "", o["count"], o["threshold"])
    if "mono_closer" in o:
        line += ": monoload closer to fp32 in {}, native closer in {}; mean |error vs fp32| there: native {:.4g}, monoload {:.4g}".format(
            o["mono_closer"], o["native_closer"], o["mean_err_native"], o["mean_err_mono"])
    print(line)
    print("{:26s} {:>11s} (bdist = rows to the nearest conv block boundary mapped to the output)".format("", ""))


def print_compare(pair, label, c):
    print("{:26s} {:>11s} raw    {}".format(pair, label, fmt_err(c.get("raw"))))
    print("{:26s} {:>11s} pixels {}".format("", "", fmt_err(c.get("px"))))
    if "px_boundary" in c:
        print("{:26s} {:>11s} pixels near block boundaries ({} rows): {}".format("", "", c["boundary_rows"], fmt_err(c["px_boundary"])))
        print("{:26s} {:>11s} pixels elsewhere: {}".format("", "", fmt_err(c.get("px_interior"))))
    print("{:26s} {:>11s} channel mean shift (pixels) {} ; NaN/Inf ref/other: raw {} pixels {}".format(
        "", "", ["{:.2g}".format(x) for x in c["px_ch_shift"]], c.get("raw_nonfinite"), c["px_nonfinite"]))


def fmt_err(e):
    if e is None:
        return "n/a"
    return "max {:.3g} mean {:.3g} rmse {:.3g} psnr {:.1f} p99 {:.3g}".format(e["max"], e["mean"], e["rmse"], e["psnr"], e["p99"])


# ---------------------------------------------------------------------------
# --profile
# ---------------------------------------------------------------------------

def _frames_str(frames, n=8):
    out = []
    for f in frames:
        fn = f.get("filename", "")
        name = f.get("name", "")
        if not fn or ("/torch/" in fn and "nn/modules" not in fn and "functional" not in fn):
            # C++ frames (--profile-cpp): keep the ATen / allocator ones that tell which kernel allocated
            if not re.search(r"slow_conv|im2col|vol2col|conv|bmm|softmax|mm|empty|native::", name) or "python" in name.lower():
                continue
        out.append("{}:{} {}".format(fn.replace("/opt/ComfyUI/", ""), f.get("line"), f.get("name")))
        if len(out) >= n:
            break
    return out


def profile_decode(vae, latent, label, top, mode, cpp):
    configure(mode)
    print("\n=== profile: {} decode {} (torch.profiler, profile_memory=True) ===".format(mode, label), flush=True)
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU]
    if is_cuda():
        acts.append(ProfilerActivity.CUDA)
    gc.collect()
    if is_cuda():
        torch.cuda.empty_cache()
    status = "ok"
    try:
        with NativeGuard(), torch.inference_mode(), profile(activities=acts, profile_memory=True, record_shapes=True, with_stack=True) as prof:
            vae.decode(latent)
            sync()
    except TiledFallback:
        status = "OOM(tiled)"
    except Exception as e:
        if not mm.is_oom(e):
            raise
        status = "OOM"
    if status != "ok":
        print("{} decode {} under the profiler: {} (the table below covers the run up to that point)".format(mode, label, status))
    ka = prof.key_averages()
    sort_key = None
    for k in ("self_device_memory_usage", "self_cuda_memory_usage", "self_cpu_memory_usage"):
        try:
            ka.table(sort_by=k, row_limit=1)
            sort_key = k
            break
        except Exception:
            continue
    print("-- operators by self device memory allocated ({}) --".format(sort_key))
    print(ka.table(sort_by=sort_key, row_limit=top, max_name_column_width=60))

    def mem_of(e):
        for k in ("self_device_memory_usage", "self_cuda_memory_usage"):
            v = getattr(e, k, None)
            if v is not None:
                return v
        return 0
    evs = [e for e in prof.events() if mem_of(e) > 0]
    evs.sort(key=mem_of, reverse=True)
    print("-- largest single allocations (one operator call each) --")
    for e in evs[:top]:
        stack = [s for s in (e.stack or []) if "/torch/" not in s or "nn/modules" in s][:5]
        print("  {:>9s}  {:40s} shapes {}".format(vae_ops.fmt_bytes(mem_of(e)), e.name[:40], str(e.input_shapes)[:120]))
        for s in stack:
            print("             at {}".format(s.replace("/opt/ComfyUI/", "")))
    del prof, ka, evs

    if not is_cuda():
        print("(memory history needs a CUDA/HIP device; skipped)")
        return
    print("\n=== profile: {} decode {} (allocator memory history: live allocations at the peak) ===".format(mode, label), flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="all" if cpp else "python")
    except Exception as e:
        print("memory history not available on this build: {}".format(e))
        return
    status = "ok"
    try:
        with NativeGuard(), torch.inference_mode():
            vae.decode(latent)
            sync()
    except TiledFallback:
        status = "OOM(tiled)"
    except Exception as e:
        if not mm.is_oom(e):
            torch.cuda.memory._record_memory_history(enabled=None)
            raise
        status = "OOM"
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    if status != "ok":
        print("{} decode {}: {} (the trace covers the run up to that point)".format(mode, label, status))
    dev = torch.cuda.current_device()
    traces = snap.get("device_traces", [])
    trace = traces[dev] if dev < len(traces) else []
    # a tensor's memory counts as allocated until free_requested (= what max_memory_allocated counts)
    free_act = "free_requested" if any(ev.get("action") == "free_requested" for ev in trace) else "free_completed"
    # pass 1: when is the sum of live blocks largest
    live = {}
    cur = peak = 0
    peak_i = -1
    for i, ev in enumerate(trace):
        act = ev.get("action")
        if act == "alloc":
            live[ev["addr"]] = ev["size"]
            cur += ev["size"]
            if cur > peak:
                peak, peak_i = cur, i
        elif act == free_act:
            sz = live.pop(ev["addr"], None)
            if sz is not None:
                cur -= sz
    if peak_i < 0:
        print("no allocation events recorded")
        return
    # pass 2: the live set at that moment
    live = {}
    for ev in trace[:peak_i + 1]:
        act = ev.get("action")
        if act == "alloc":
            live[ev["addr"]] = ev
        elif act == free_act:
            live.pop(ev["addr"], None)
    blocks = sorted(live.values(), key=lambda e: e["size"], reverse=True)
    print("peak of tensors allocated during the decode: {} in {} live blocks (allocated before the decode -- e.g. the weights -- not included)".format(
        vae_ops.fmt_bytes(peak), len(blocks)))
    for ev in blocks[:top]:
        print("  {:>9s}".format(vae_ops.fmt_bytes(ev["size"])))
        for s in _frames_str(ev.get("frames", []), 12 if cpp else 8):
            print("             at {}".format(s))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def env_header(vae, source, a):
    info = comfy_env.comfyui_version_info(ROOT)
    mono = None
    try:
        g = os.path.join(REPO, ".git")
        with open(os.path.join(g, "HEAD")) as f:
            head = f.read().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            p = os.path.join(g, ref)
            mono = open(p).read().strip() if os.path.isfile(p) else ref
        else:
            mono = head
    except OSError:
        pass
    dev = mm.get_torch_device()
    print("ComfyUI args: {}".format(" ".join(COMFY_ARGS)))
    print("ComfyUI {} commit {} | Monoload {} | torch {} (git {}) hip {}".format(
        info.get("comfyui_version"), (info.get("comfyui_commit") or "?")[:10], (mono or "?")[:10], torch.__version__,
        (getattr(torch.version, "git_version", None) or "?")[:10], getattr(torch.version, "hip", None)))
    if is_cuda():
        props = torch.cuda.get_device_properties(dev)
        print("device {} ({}), total {} GiB | cudnn.enabled {} (False = MIOpen off: Slow2d im2col + GEMM)".format(
            dev, props.name, gib(props.total_memory).strip(), torch.backends.cudnn.enabled))
    gtt_tot = read_sum(list(drm_files("gtt_total").values()))
    print("GTT cards: {} | GTT total {} GiB | cgroup {}".format(list(drm_files("gtt_used").keys()) or "none", gib(gtt_tot).strip(), cgroup_dir()))
    attn = "xformers" if mm.xformers_enabled_vae() else ("pytorch" if mm.pytorch_attention_enabled_vae() else "split")
    print("VAE: {} -> {} / {}, dtype {}, device {}, output device {} ({}), attention: {}".format(
        source, type(vae.first_stage_model).__name__, type(getattr(vae.first_stage_model, "decoder", vae.first_stage_model)).__name__,
        vae.vae_dtype, vae.device, vae.output_device, vae.vae_output_dtype(), attn))
    print("Monoload VAE workspace {} (MONOLOAD_VAE_WORKSPACE / --workspace); env: {}".format(
        vae_ops.fmt_bytes(mvae.workspace()), {k: v for k, v in os.environ.items() if k.startswith(("MONOLOAD", "PYTORCH_", "TORCH_", "HIP", "GPU_", "COMFYUI_"))}))


def main():
    p = argparse.ArgumentParser(description="native vs Monoload VAE decode benchmark")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="checkpoint with a built-in VAE (models/checkpoints), e.g. an SDXL checkpoint")
    src.add_argument("--vae", help="VAE file loaded by VAELoader (models/vae), e.g. ae.safetensors, qwen_image_vae.safetensors")
    p.add_argument("--res", default="1344x768,2688x1536,3840x2160", help="output resolutions WxH, comma separated (random latents)")
    p.add_argument("--latent", action="append", help="SaveLatent .latent file instead of random latents (repeatable; path, or name in input/ or output/)")
    p.add_argument("--seed", type=int, default=0, help="seed of the random latents")
    p.add_argument("--latent-std", type=float, default=1.0, help="std of the random latents")
    p.add_argument("--modes", default="native,monoload,native2",
                   help="comma list run in order per resolution: native, monoload (layer 1 where recognized, else layer 2), "
                        "monoload-l2 (layer 2 only), monoload-r<N> (layer 1 with N-row stripes), native2")
    p.add_argument("--stripe-rows", help="comma list of layer-1 stripe core heights to sweep (adds a monoload-r<N> mode per value)")
    p.add_argument("--warm", type=int, default=3, help="warm runs after the cold one (median reported)")
    p.add_argument("--workspace", help="Monoload workspace for this run (e.g. 1G, 512M); default MONOLOAD_VAE_WORKSPACE or 1G")
    p.add_argument("--sample-ms", type=float, default=10.0, help="GTT / VRAM / cgroup sampling interval (5-20 ms)")
    p.add_argument("--boundary-rows", type=int, default=4, help="output rows on each side of a block boundary counted as 'boundary'")
    p.add_argument("--profile", action="store_true", help="profile one native decode per --profile-res (torch.profiler + allocator history)")
    p.add_argument("--profile-res", help="resolutions to profile (default: all of --res)")
    p.add_argument("--profile-only", action="store_true", help="only the profile, no benchmark")
    p.add_argument("--profile-modes", default="native", help="modes to profile: native, monoload, monoload-l2, monoload-r<N> (comma list)")
    p.add_argument("--profile-cpp", action="store_true", help="C++ frames in the allocator history (shows the ATen kernel; symbolizing can take minutes)")
    p.add_argument("--top", type=int, default=25, help="rows in the profile tables")
    p.add_argument("--fp32-ref", action="store_true", help="also decode with an fp32 copy of the VAE as reference (native vs fp32, monoload vs fp32)")
    p.add_argument("--fp32-impl", default="both", choices=("both", "monoload", "native"),
                   help="how the fp32 reference is decoded: native fp32 (falls back to chunked when it runs out of memory), Monoload-chunked fp32, or both (default)")
    p.add_argument("--outliers", type=int, default=10, help="pixels listed where monoload and native differ most")
    p.add_argument("--outlier-threshold", type=float, default=0.01, help="|monoload - native| above which a pixel value counts as an outlier")
    p.add_argument("--json", help="also write all results to this JSON file")
    a = p.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    if a.workspace:
        mvae.set_workspace(mvae.parse_size(a.workspace))

    vae, source = load_vae(a)
    env_header(vae, source, a)
    mm.load_models_gpu([vae.patcher])
    vae32 = None
    if a.fp32_ref:
        vae32 = load_vae_fp32(a)
        print("fp32 reference: same VAE loaded again with dtype {} ({}), decoded {}".format(
            vae32.vae_dtype, type(vae32.first_stage_model).__name__,
            {"both": "natively and chunked (native skipped if it runs out of memory)", "native": "natively (chunked if native runs out of memory)",
             "monoload": "with Monoload's chunking"}[a.fp32_impl]))

    inputs = []
    if a.latent:
        for name in a.latent:
            lat = load_latent(find_latent_file(name))
            r = vae.spacial_compression_decode()
            inputs.append(("{}x{}".format(lat.shape[-1] * r, lat.shape[-2] * r), lat, os.path.basename(name)))
    else:
        for res in a.res.split(","):
            w, h = (int(x) for x in res.lower().split("x"))
            inputs.append((res, random_latent(vae, w, h, a.seed, a.latent_std), "random seed {}".format(a.seed)))
    for label, lat, what in inputs:
        print("input {}: latent {} ({}), native estimate {}".format(label, list(lat.shape), what,
                                                                     vae_ops.fmt_bytes(mvae._native_estimate(vae, lat.shape))))

    if a.profile or a.profile_only:
        want = set(a.profile_res.split(",")) if a.profile_res else None
        for label, lat, _ in inputs:
            if want is None or label in want:
                for pm in a.profile_modes.split(","):
                    profile_decode(vae, lat, label, a.top, pm, a.profile_cpp)
        if a.profile_only:
            mvae.uninstall()
            return

    modes = a.modes.split(",")
    if a.stripe_rows:
        modes += ["monoload-r{}".format(int(r)) for r in a.stripe_rows.split(",") if r.strip()]
    results = []
    accuracy = []
    outlier_info = []
    for label, lat, _ in inputs:
        print("\n=== {} ===".format(label), flush=True)
        captured = {}
        for mode in modes:
            s, raw, px = run_mode(vae, lat, mode, a, label)
            results.append(s)
            if s["status"] in ("ok", "partial") and px is not None:
                captured[mode] = (raw, px, boundary_stats(s.get("mono")))
        ref = captured.get("native")
        mono_stats = captured.get("monoload", (None, None, None))[2]
        for other in modes:
            if other == "native" or other not in captured:
                continue
            if ref is None:
                accuracy.append({"res": label, "pair": other + " vs native", "note": "native failed (OOM): no reference"})
                continue
            c = compare((ref[0], ref[1]), (captured[other][0], captured[other][1]), captured[other][2], a.boundary_rows)
            c.update({"res": label, "pair": other + " vs native"})
            accuracy.append(c)
            print_compare(other + " vs native", label, c)

        fp32 = None
        if vae32 is not None:
            a0 = argparse.Namespace(**dict(vars(a), warm=0))
            got = {}
            for mode, want in (("native-fp32", a.fp32_impl in ("both", "native")), ("monoload-fp32", True)):
                if not want or (mode == "monoload-fp32" and a.fp32_impl == "native" and "native-fp32" in got):
                    continue
                s32, raw, px = run_mode(vae32, lat, mode, a0, label)
                results.append(s32)
                if s32["status"] == "ok" and px is not None:
                    got[mode] = (raw, px)
            if "native-fp32" in got and "monoload-fp32" in got:
                c = compare(got["native-fp32"], got["monoload-fp32"], mono_stats, a.boundary_rows)
                c.update({"res": label, "pair": "fp32 chunked vs fp32 native"})
                accuracy.append(c)
                print_compare("fp32 chunked vs fp32 nat.", label, c)
            fp32 = got.get("native-fp32") or got.get("monoload-fp32")
            which = "native" if "native-fp32" in got else ("chunked" if fp32 is not None else None)
            if fp32 is None:
                accuracy.append({"res": label, "pair": "* vs fp32", "note": "fp32 reference failed (OOM)"})
            else:
                for other in [m_ for m_ in modes if m_ in captured and m_ != "native2"]:
                    bst = captured[other][2] if other.startswith("monoload") else mono_stats
                    c = compare(fp32, (captured[other][0], captured[other][1]), bst, a.boundary_rows)
                    c.update({"res": label, "pair": "{} vs fp32 ({})".format(other, which)})
                    accuracy.append(c)
                    print_compare("{} vs fp32".format(other), label, c)
            del got
        if "native" in captured and "monoload" in captured and a.outliers > 0:
            o = outliers(captured["native"][:2], captured["monoload"][:2], fp32, mono_stats, a.outliers, a.outlier_threshold)
            o["res"] = label
            outlier_info.append(o)
            print_outliers(o, label, fp32 is not None)
        del fp32
        del captured
        gc.collect()
    mvae.uninstall()

    print("\n=== summary: time and memory (GiB; Δ = increase over the value right before the run; peaks = max over the runs) ===")
    hdr = "{:11s} {:13s} {:10s} {:>8s} {:>8s} | {:>10s} {:>10s} {:>8s} {:>8s} {:>9s} {:>9s} | {:>9s} {:>9s} {:>6s} | {}"
    print(hdr.format("res", "mode", "status", "cold s", "warm s", "alloc pk", "alloc Δ", "resv Δ", "GTT Δ", "cg peakΔ", "cg sampΔ", "estimate",
                     "native est", "unload", "layer"))
    for s in results:
        m = s.get("mono") or {}
        est = (m.get("estimate") or {}).get("total")
        nat = mvae._native_estimate(vae32 if s["mode"].endswith("fp32") else vae, next(l for lb, l, _ in inputs if lb == s["res"]).shape)
        if m.get("strategy") == "layer1":
            layer = "L1 {}x{} rows, recompute {:.2f}x, ckpt {}".format(m["stripes"], m["rows"], m["recompute"], vae_ops.fmt_bytes(m["checkpoint_bytes"]))
        elif m.get("strategy") == "layer2":
            layer = "L2"
        else:
            layer = ""
        print("{:11s} {:13s} {:10s} {:>8s} {:>8s} | {:>10s} {:>10s} {:>8s} {:>8s} {:>9s} {:>9s} | {:>9s} {:>9s} {:>6} | {}".format(
            s["res"], s["mode"], s["status"], "{:.3f}".format(s["cold"]), "{:.3f}".format(s["warm"]) if s["warm"] is not None else "n/a",
            gib(s.get("alloc_peak")), gib(s.get("alloc_delta")), gib(s.get("res_delta")), gib(s.get("gtt_delta")), gib(s.get("cg_mempeak_delta")),
            gib(s.get("cg_delta")), gib(est) if s["mode"].startswith("monoload") else "", gib(nat), s.get("unloaded") or 0, layer))
    print("\n=== summary: accuracy (raw = decoder output before process_output; pixels = clamped fp32 in [0, 1]; "
          "bound/inter = rows near the stripe (layer 1) or conv block (layer 2) boundaries of that monoload run -- of the monoload run for native -- / all other rows) ===")
    print("{:11s} {:28s} | {:>9s} {:>9s} {:>9s} | {:>9s} {:>9s} {:>9s} {:>7s} {:>9s} | {:>9s} {:>9s} | {:>9s}".format(
        "res", "pair", "raw max", "raw mean", "raw rmse", "px max", "px mean", "px rmse", "PSNR", "px p99", "bound max", "inter max", "NaN/Inf"))
    for c in accuracy:
        if "note" in c:
            print("{:11s} {:28s} | {}".format(c["res"], c["pair"], c["note"]))
            continue
        r = c.get("raw") or {}
        x = c["px"]
        f = lambda v: "{:9.3g}".format(v) if v is not None else "      n/a"
        print("{:11s} {:28s} | {} {} {} | {} {} {} {:7.1f} {} | {} {} | {:>9s}".format(
            c["res"], c["pair"], f(r.get("max")), f(r.get("mean")), f(r.get("rmse")), f(x["max"]), f(x["mean"]), f(x["rmse"]), x["psnr"], f(x["p99"]),
            f((c.get("px_boundary") or {}).get("max")), f((c.get("px_interior") or {}).get("max")),
            "{}/{}".format(c["px_nonfinite"][1], (c.get("raw_nonfinite") or ("-", "-"))[1])))
    print("\nalloc = torch.cuda.max_memory_allocated (tensors), resv = max_memory_reserved (allocator incl. cache), GTT = amdgpu GTT used "
          "(device-wide, every process), cg = container cgroup (memory.peak reset per run where supported / sampled memory.current).")
    print("layer: L1 = layer 1, stripe decoding (stripes x core rows, conv work relative to a whole-image decode, checkpoint size); "
          "L2 = layer 2, op-level chunking.")
    print("estimate = what Monoload passed to load_models_gpu; native est = memory_used_decode (what native passes); "
          "unload = models unloaded by load_models_gpu during the run (here only the VAE is loaded, so 1 = the VAE itself was unloaded and loaded again).")
    print("native2 vs native = the GPU's own run-to-run difference; monoload vs native should be of that order or small (no bit-exactness promised).")
    if outlier_info:
        print("\n=== summary: outliers (|monoload - native| > threshold) ===")
        for o in outlier_info:
            line = "{:11s} {} pixel values above {}, largest {:.4f}".format(o["res"], o["count"], o["threshold"], o["top"][0]["diff"] if o["top"] else 0.0)
            if "mono_closer" in o:
                line += "; closer to fp32: monoload {}, native {}; mean |error vs fp32| there: native {:.4g}, monoload {:.4g}".format(
                    o["mono_closer"], o["native_closer"], o["mean_err_native"], o["mean_err_mono"])
            print(line)
    if vae32 is not None:
        print("X vs fp32 = error of the bf16 result X against the fp32 decode of the same VAE (native fp32 if it fit, otherwise chunked fp32). "
              "If monoload vs fp32 is about as large as native vs fp32, the monoload/native outliers are bf16 noise, not chunking.")
    if a.json:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
                return str(o)
            return o
        with open(a.json, "w") as f:
            json.dump(clean({"results": results, "accuracy": accuracy, "outliers": outlier_info}), f, indent=1, default=str)
        print("results written to {}".format(a.json))


if __name__ == "__main__":
    main()
