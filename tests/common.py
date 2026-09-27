"""Shared helpers for Monoload tests. Run inside the locked ComfyUI image
(tests/docker_run.sh); every test script is a plain `python tests/xxx.py`.

ComfyUI launch args come from $COMFY_ARGS (default: --cpu --disable-mmap --fp16-unet).
"""

import json
import os
import shlex
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from monoload import comfy_env  # noqa: E402

COMFY_ARGS = shlex.split(os.environ.get("COMFY_ARGS", "--cpu --disable-mmap --fp16-unet"))
_TEST_ARGV = list(sys.argv)
ROOT, ARGS = comfy_env.setup(COMFY_ARGS)  # ComfyUI parses sys.argv here
sys.argv = _TEST_ARGV

import torch  # noqa: E402
import folder_paths  # noqa: E402
import nodes  # noqa: E402
import comfy.model_management  # noqa: E402

from monoload import fmt, rebuild, transfer  # noqa: E402
from monoload.nodes import MonoloadUNETLoader, register_model_folder  # noqa: E402
from monoload.errors import MonoloadError, MonoloadFormatError, MonoloadUnsupportedError  # noqa: E402

register_model_folder()
MODELS = folder_paths.models_dir
torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("[{}] {}{}".format("PASS" if ok else "FAIL", name, (" — " + detail) if detail else ""), flush=True)
    return ok


def finish(extra=None):
    failed = [r for r in RESULTS if not r[1]]
    print("\n== {} checks, {} failed".format(len(RESULTS), len(failed)))
    if extra is not None:
        print("RESULT_JSON " + json.dumps(extra, default=str))
    sys.exit(1 if failed else 0)


def expect_raises(name, exc_type, fn, *substrings):
    try:
        fn()
    except exc_type as e:
        msg = str(e)
        missing = [s for s in substrings if s not in msg]
        first = msg.strip().splitlines()[0] if msg.strip() else ""
        return check(name, not missing, "missing {} in: {}".format(missing, msg[:500]) if missing else first[:220])
    except Exception as e:
        return check(name, False, "wrong exception {}: {}".format(type(e).__name__, str(e)[:500]))
    return check(name, False, "no exception raised")


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------

def proc_status(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    return None


def reset_hwm():
    """Reset VmHWM to the current RSS (Linux: write 5 to clear_refs)."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def mapped_model_files(prefix=None):
    prefix = prefix or MODELS
    out = set()
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split(None, 5)
            if len(parts) == 6 and parts[5].strip().startswith(prefix):
                out.add(parts[5].strip())
    return out


def gtt_paths():
    """mem_info_gtt_used of each distinct amdgpu device (empty if none/unreadable)."""
    import glob
    import re
    seen = {}
    for d in glob.glob("/sys/class/drm/card*"):
        if not re.search(r"/card\d+$", d):
            continue
        p = os.path.join(d, "device", "mem_info_gtt_used")
        if os.path.exists(p):
            seen.setdefault(os.path.realpath(os.path.join(d, "device")), p)
    return list(seen.values())


def read_gtt(paths):
    total = 0
    for p in paths:
        with open(p) as f:
            total += int(f.read().strip())
    return total


def cgroup_memory_current():
    for p in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(p) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            pass
    return None


class MemWatch:
    """High-frequency RSS (+GTT, cgroup) sampler + mapping watcher (runs in a thread)."""

    def __init__(self, interval=0.002, maps_every=10):
        self.interval = interval
        self.maps_every = maps_every
        self.peak = 0
        self.samples = 0
        self.mapped = set()
        self._stop = threading.Event()
        self._page = os.sysconf("SC_PAGE_SIZE")
        self._gtt = gtt_paths()
        self.gtt_peak = None
        self.cg_peak = None

    def _rss(self):
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * self._page

    def _run(self):
        i = 0
        while not self._stop.is_set():
            r = self._rss()
            if r > self.peak:
                self.peak = r
            if self._gtt:
                g = read_gtt(self._gtt)
                self.gtt_peak = g if self.gtt_peak is None else max(self.gtt_peak, g)
            c = cgroup_memory_current()
            if c is not None:
                self.cg_peak = c if self.cg_peak is None else max(self.cg_peak, c)
            if i % self.maps_every == 0:
                self.mapped |= mapped_model_files()
            i += 1
            self.samples += 1
            time.sleep(self.interval)

    def __enter__(self):
        self.base = self._rss()
        self.peak = self.base
        self.gtt_base = read_gtt(self._gtt) if self._gtt else None
        self.gtt_peak = self.gtt_base
        self.cg_base = cgroup_memory_current()
        self.cg_peak = self.cg_base
        reset_hwm()
        self.hwm0 = proc_status("VmHWM")
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        self.mapped |= mapped_model_files()
        self.end = self._rss()
        self.hwm = proc_status("VmHWM")
        self.gtt_end = read_gtt(self._gtt) if self._gtt else None
        self.cg_end = cgroup_memory_current()

    def report(self):
        return {
            "rss_before": self.base,
            "rss_after": self.end,
            "peak_rss_sampled": self.peak,
            "vm_hwm": self.hwm,
            "peak_delta": max(self.peak, self.hwm or 0) - self.base,
            "mapped_model_files": sorted(self.mapped),
            "samples": self.samples,
            "gtt_before": self.gtt_base,
            "gtt_peak_delta": None if self.gtt_base is None else self.gtt_peak - self.gtt_base,
            "gtt_after_delta": None if self.gtt_base is None else self.gtt_end - self.gtt_base,
            "cgroup_before": self.cg_base,
            "cgroup_peak_delta": None if self.cg_base is None else self.cg_peak - self.cg_base,
        }


def gib(n):
    return "{:.3f} GiB".format(n / 2 ** 30)


# ---------------------------------------------------------------------------
# loading / comparing / sampling
# ---------------------------------------------------------------------------

def load_native(name):
    return nodes.UNETLoader().load_unet(name, "default")[0]


def load_monoload(name, buffer_mb=0):
    return MonoloadUNETLoader().load_unet(name, buffer_mb=buffer_mb)[0]


def byte_view(t):
    return t.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)


def tensors_bytes_equal(a, b):
    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(byte_view(a), byte_view(b))


def compare_state(model_a, model_b):
    sa = model_a.state_dict()
    sb = model_b.state_dict()
    problems = []
    if list(sa.keys()) != list(sb.keys()):
        problems.append("key lists differ: only_a={} only_b={}".format(sorted(set(sa) - set(sb))[:5], sorted(set(sb) - set(sa))[:5]))
    for k in sa:
        if k in sb and not tensors_bytes_equal(sa[k], sb[k]):
            problems.append("{} differs".format(k))
    return len(sa), problems


def compare_with_file(model, path):
    """Every parameter/buffer of `model` equals the bytes stored in the Monoload file."""
    header = fmt.read_header_path(path)
    fd = fmt.open_readonly(path)
    problems = []
    try:
        entries = rebuild.state_entries(model)
        for e in entries:
            info = header.by_name.get(e.name)
            if info is None:
                problems.append("{} not in file".format(e.name))
                continue
            buf = torch.empty(info.nbytes, dtype=torch.uint8)
            if info.nbytes:
                transfer.pread_into(fd, transfer.cpu_memoryview(buf), header.data_start + info.begin)
            if not torch.equal(byte_view(e.tensor), buf):
                problems.append("{} differs from file".format(e.name))
    finally:
        os.close(fd)
    return len(entries), problems


def sample(model, positive, negative, latent, seed=1234, steps=4, cfg=5.0, sampler="euler", scheduler="normal"):
    out = nodes.common_ksampler(model, seed, steps, cfg, sampler, scheduler, positive, negative, {"samples": latent.clone()}, denoise=1.0)
    return out[0]["samples"]


def diff_stats(a, b):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    return {"bit_exact": bool(torch.equal(a, b)), "max_abs": float(d.max()), "mean_abs": float(d.mean())}


def family_inputs(family, seed=0, model=None):
    """Deterministic fake conditioning + empty latent for a model family.
    `zimage` reads the caption dim / latent channels from `model` (a ModelPatcher)."""
    g = torch.Generator().manual_seed(seed)
    if family == "zimage":
        dim = model.model.model_config.unet_config.get("cap_feat_dim", 2560)
        ch = model.model.latent_format.latent_channels
        pos = [[torch.randn(1, 32, dim, generator=g), {}]]
        neg = [[torch.randn(1, 32, dim, generator=g) * 0.1, {}]]
        return pos, neg, torch.zeros(1, ch, 32, 32)
    if family == "sd15":
        pos = [[torch.randn(1, 77, 768, generator=g), {}]]
        neg = [[torch.randn(1, 77, 768, generator=g) * 0.1, {}]]
        latent = torch.zeros(1, 4, 32, 32)  # 256x256
        return pos, neg, latent
    if family == "anima":
        def c(scale):
            return [[torch.randn(1, 24, 1024, generator=g) * scale,
                     {"t5xxl_ids": torch.randint(0, 32000, (24,), generator=g),
                      "t5xxl_weights": torch.ones(24)}]]
        pos = c(1.0)
        neg = c(0.1)
        latent = torch.zeros(1, 16, 1, 32, 32)  # 256x256
        return pos, neg, latent
    raise ValueError(family)


def free_all():
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    import gc
    gc.collect()


def config_snapshot_eq(a, b):
    diffs = []
    for k in a:
        if a[k] != b[k]:
            diffs.append("{}: native={} monoload={}".format(k, a[k], b[k]))
    return not diffs, diffs
