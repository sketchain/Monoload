"""Replay a VAE decode through a model of PyTorch's caching allocator.

The CUDA / HIP caching allocator decides how much a decode really holds
(max_memory_reserved, i.e. GTT on the APU): freed blocks stay cached and are
reused only when a request fits, so the same tensors can need very different
reserved peaks depending on the order and sizes of the requests. This tool
reproduces that without a GPU:

  * AllocatorSim  the allocator's rules (c10/cuda/CUDACachingAllocator.cpp,
                  constants from c10/core/AllocatorConfig.h of the locked
                  image): sizes rounded to 512 B; <= 1 MiB from 2 MiB small
                  segments, 1-10 MiB from 20 MiB segments, larger rounded up to
                  2 MiB; best fit (smallest free block >= request, lowest
                  address); a large-pool block is split when more than 1 MiB
                  remains; freed blocks merge with free neighbours of the same
                  segment; empty_cache releases segments that are entirely free.
                  Defaults of the allocator config (no max_split_size, no
                  expandable segments, no garbage collection threshold).
  * Tracer        a TorchDispatchMode over meta tensors: every aten op's fresh
                  output storage is a malloc, a storage that died (all its
                  tensors and views gone) is a free; the buffers the GPU kernels
                  allocate internally are added by hand from the kernels' code:
                  Slow2d (4D conv without cuDNN) and SlowDilated3d (5D) copy a
                  non-contiguous input / weight, allocate the output, then the
                  im2col / vol2col columns (Slow2d skips them for 1x1 / stride 1 /
                  unpadded); upsampling copies a non-contiguous input.
  * decode_trace  builds the Wan 2.1 decoder (qwen_image_vae dimensions) on the
                  meta device and runs Monoload's layer 1 or layer 2 on it, with
                  the switches the earlier versions had (cache emptied between
                  prefix and stripes or not, stripe order, single-frame Conv3d as
                  conv2d or not), and reports the alloc / reserved deltas the
                  bench would print.

Validated against the CT 700 readings (DESIGN §9.13.4). Usage:

    python tests/alloc_sim.py                      # the validation table
    python tests/alloc_sim.py --res 3840x2160 --rows 32,128,155,512
"""

import argparse
import bisect
import contextlib
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MIB = 1 << 20
K_MIN_BLOCK = 512
K_SMALL_SIZE = 1 << 20
K_SMALL_BUFFER = 2 << 20
K_LARGE_BUFFER = 20 << 20
K_MIN_LARGE_ALLOC = 10 << 20
K_ROUND_LARGE = 2 << 20


class _Block:
    __slots__ = ("size", "addr", "prev", "next", "free", "small", "requested")

    def __init__(self, size, addr, small):
        self.size, self.addr, self.small = size, addr, small
        self.prev = self.next = None
        self.free = True
        self.requested = 0


class AllocatorSim:
    def __init__(self):
        self.pools = {True: [], False: []}   # small? -> sorted [(size, addr, block)]
        self.next_addr = 0
        self.reserved = self.allocated = 0
        self.peak_reserved = self.peak_allocated = 0
        self.segments = 0
        self.mallocs = 0        # device mallocs (new segments)
        self.live = set()
        self.peak_blocks = []   # sizes of the live blocks at the allocation peak
        self.tag = ""           # what is running (for the peak report)
        self.heads = []         # first block of every segment
        self.peak_segments = [] # segment layout when reserved peaked: [(segment size, [(block size, tag or None if free)])]

    @staticmethod
    def round_size(n):
        return K_MIN_BLOCK if n < K_MIN_BLOCK else K_MIN_BLOCK * (-(-n // K_MIN_BLOCK))

    @staticmethod
    def allocation_size(n):
        if n <= K_SMALL_SIZE:
            return K_SMALL_BUFFER
        if n < K_MIN_LARGE_ALLOC:
            return K_LARGE_BUFFER
        return K_ROUND_LARGE * (-(-n // K_ROUND_LARGE))

    def _insert(self, b):
        bisect.insort(self.pools[b.small], (b.size, b.addr, b))

    def _remove(self, b):
        pool = self.pools[b.small]
        i = bisect.bisect_left(pool, (b.size, b.addr))
        assert pool[i][2] is b
        pool.pop(i)

    def malloc(self, nbytes):
        size = self.round_size(max(int(nbytes), 1))
        small = size <= K_SMALL_SIZE
        pool = self.pools[small]
        i = bisect.bisect_left(pool, (size, -1))
        if i < len(pool):
            b = pool.pop(i)[2]
        else:
            seg = self.allocation_size(size)
            b = _Block(seg, self.next_addr, small)
            self.next_addr += seg
            self.reserved += seg
            self.segments += 1
            self.mallocs += 1
            self.heads.append(b)
            if self.reserved > self.peak_reserved:
                self.peak_reserved = self.reserved
                self.peak_segments = self._layout(extra=(b, size))
        rem = b.size - size
        if (small and rem >= K_MIN_BLOCK) or (not small and rem > K_SMALL_SIZE):
            r = _Block(rem, b.addr + size, small)
            r.prev, r.next = b, b.next
            if b.next is not None:
                b.next.prev = r
            b.next = r
            b.size = size
            self._insert(r)
        b.free = False
        b.requested = (nbytes, self.tag)
        self.allocated += b.size
        self.live.add(b)
        if self.allocated > self.peak_allocated:
            self.peak_allocated = self.allocated
            self.peak_blocks = sorted((x.size, x.requested[1]) for x in self.live)
        return b

    def _layout(self, extra=None):
        out = []
        for h in self.heads:
            blocks, x = [], h
            while x is not None:
                if extra is not None and x is extra[0]:
                    blocks.append((extra[1], "<request: " + self.tag + ">"))
                    if x.size > extra[1]:
                        blocks.append((x.size - extra[1], None))
                else:
                    blocks.append((x.size, None if x.free else x.requested[1]))
                x = x.next
            out.append((sum(sz for sz, _ in blocks), blocks))
        return out

    def free(self, b):
        assert not b.free
        self.live.discard(b)
        self.allocated -= b.size
        b.free = True
        for nb in (b.prev, b.next):
            if nb is not None and nb.free:
                self._remove(nb)
                if nb is b.prev:
                    nb.size += b.size
                    nb.next = b.next
                    if b.next is not None:
                        b.next.prev = nb
                    b = nb
                else:
                    b.size += nb.size
                    b.next = nb.next
                    if nb.next is not None:
                        nb.next.prev = b
        self._insert(b)

    def empty_cache(self):
        for small in (True, False):
            keep = []
            for item in self.pools[small]:
                b = item[2]
                if b.prev is None and b.next is None:
                    self.reserved -= b.size
                    self.segments -= 1
                    self.heads.remove(b)
                else:
                    keep.append(item)
            self.pools[small] = keep

    def reset_peak(self):
        self.peak_reserved, self.peak_allocated = self.reserved, self.allocated

    def cached(self):
        return self.reserved - self.allocated


# ---------------------------------------------------------------------------
# tracing a decode on the meta device
# ---------------------------------------------------------------------------

def _tensors(x):
    import torch
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _tensors(y)


CONV_OPS = ("aten::convolution", "aten::_convolution", "aten::conv2d", "aten::conv3d")


def _ints(v):
    return [int(x) for x in v] if isinstance(v, (list, tuple)) else [int(v)]


def make_tracer(sim):
    import torch
    from torch.multiprocessing.reductions import StorageWeakRef
    from torch.utils._python_dispatch import TorchDispatchMode

    aten = torch.ops.aten

    class Tracer(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.live = {}
            self.static = set()   # storages allocated outside the trace (weights): views of them are not allocations

        def poll(self):
            dead = [k for k, (w, _) in self.live.items() if w.expired()]
            for k in dead:
                sim.free(self.live.pop(k)[1])

        def _track(self, t):
            st = t.untyped_storage()
            k = st._cdata
            if k in self.live or k in self.static or st.nbytes() == 0:
                return
            self.live[k] = (StorageWeakRef(st), sim.malloc(st.nbytes()))

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            self.poll()
            temps, after = [], []
            name = func.name()
            sim.tag = name
            if name in CONV_OPS:
                x, w = args[0], args[1]
                e = x.element_size()
                if not x.is_contiguous():
                    temps.append(sim.malloc(x.numel() * e))
                if not w.is_contiguous():
                    temps.append(sim.malloc(w.numel() * e))
                stride = _ints(args[3] if len(args) > 3 else kwargs.get("stride", 1))
                padding = args[4] if len(args) > 4 else kwargs.get("padding", 0)
                padding = [0] if padding == "valid" else ([1] if isinstance(padding, str) else _ints(padding))
                one = list(w.shape[2:]) == [1] * (w.dim() - 2) and all(s == 1 for s in stride) and all(p == 0 for p in padding)
                after.append(("cols", x.dim() == 5 or not one, w))
            elif "upsample_nearest" in name and not args[0].is_contiguous():
                temps.append(sim.malloc(args[0].numel() * args[0].element_size()))
            out = func(*args, **kwargs)
            for t in _tensors(out):
                self._track(t)
            for kind, need, w in after:
                if need:
                    o = next(_tensors(out))
                    cols = w.shape[1] * w[0, 0].numel() * o[0, 0].numel() * o.element_size()
                    temps.append(sim.malloc(cols))
            for b in reversed(temps):
                sim.free(b)
            return out

    return Tracer()


WAN_QWEN = dict(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                temperal_downsample=[False, True, True], dropout=0.0)


class _MetaVAE:
    """The attributes of comfy.sd.VAE that layer 1 reads."""

    def __init__(self, fsm, dtype):
        import torch
        self.first_stage_model = fsm
        self.device = self.output_device = torch.device("meta")
        self.vae_dtype = dtype
        self.process_output = lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)

    def vae_output_dtype(self):
        import torch
        return torch.float32


_MODELS = {}


def meta_vae(dtype):
    import torch
    import comfy.ldm.wan.vae as wan
    if dtype not in _MODELS:
        with torch.device("meta"):
            fsm = wan.WanVAE(**WAN_QWEN)
        fsm.to(dtype).eval()
        _MODELS[dtype] = _MetaVAE(fsm, dtype)
    return _MODELS[dtype]


@contextlib.contextmanager
def _patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def decode_trace(w, h, dtype="bf16", rows=None, layer=1, ws=None, clear=False, order="largest", conv2d=True, arena=None, batch=1, out_first=True,
                 layer1_ws=None, contiguous=True):
    """alloc / reserved deltas (bytes) of one decode of a w x h image, as bench_vae measures them
    (cache emptied and peaks reset right before the decode, the weights already loaded).

    rows    stripe core height (None: the default policy of monoload.vae.choose_plan)
    clear   empty the cache between prefix and stripes (725a010 did)
    order   "largest" (largest stripe first) / "natural" (top to bottom, 4e54d20)
    conv2d  single-frame Conv3d as conv2d (SlowDilated3d otherwise)
    arena   None: the plan's arena (the current code); 0: none (the earlier versions); bytes: that size
    out_first  a row-blocked conv allocates its output before the first block (False: after it, up to 725a010)
    layer1_ws  monoload.vae.LAYER1_WORKSPACE for this decode (384 MiB up to 5d668b6, 128 MiB since)
    contiguous the row slice into a Resample is made contiguous first (False: up to 725a010)
    """
    import torch
    import comfy.model_management as mm
    from monoload import vae as mvae, vae_ops, vae_engine as eng

    dt = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[dtype] if isinstance(dtype, str) else dtype
    v = meta_vae(dt)
    fsm = v.first_stage_model
    sim = AllocatorSim()
    sim.tag = "weights"
    for p in fsm.parameters():           # the weights, as loaded before the decode (encoder too)
        sim.malloc(p.numel() * p.element_size())
    for b in fsm.buffers():
        sim.malloc(b.numel() * b.element_size())
    sim.empty_cache()
    sim.reset_peak()
    base_alloc, base_res = sim.allocated, sim.reserved
    lat = torch.empty(batch, 16, 1, h // 8, w // 8, device="meta")   # the bench's latent is on the CPU: not counted, its copy is
    tracer = make_tracer(sim)
    tracer.static.update(t.untyped_storage()._cdata for t in itertools.chain(fsm.parameters(), fsm.buffers(), [lat]))
    events = {"empty_cache": 0}

    def soft_empty_cache(force=False):
        tracer.poll()
        events["empty_cache"] += 1
        sim.empty_cache()

    info = {}
    orig_prefix = eng.run_prefix

    def run_prefix_clear(prefix, z):
        ckpt = orig_prefix(prefix, z)
        soft_empty_cache(True)
        return ckpt

    with contextlib.ExitStack() as es:
        es.enter_context(_patched(vae_ops, "slow_dilated3d", lambda x: conv2d))
        es.enter_context(_patched(vae_ops, "OUT_FIRST", out_first))
        es.enter_context(_patched(mm, "soft_empty_cache", soft_empty_cache))
        es.enter_context(_patched(eng, "arena_supported", lambda device: True))
        if layer1_ws:
            es.enter_context(_patched(mvae, "LAYER1_WORKSPACE", layer1_ws))
        if not contiguous:
            es.enter_context(_patched(eng, "CONTIGUOUS_INPUT", ()))
        if clear:
            es.enter_context(_patched(eng, "run_prefix", run_prefix_clear))
        if layer == 1:
            bound, why = mvae._select_layer1(v, lat, {})
            assert bound is not None, why
            outb = batch * 3 * h * w * 4
            if rows:
                ws_ = ws or mvae.layer1_workspace()
                plan = bound.plan(v, lat, 0, ws_, rows=rows, out_bytes=outb)
            else:
                plan, _, ws_, _ = mvae.choose_plan(v, lat, bound, outb)
            if order == "natural":
                plan.order = list(range(len(plan.stripes)))
            if arena is not None:
                plan.arena = arena
            info.update(plan=plan, stripes=len(plan.stripes), rows=max(b - a for a, b in plan.stripes), estimate=plan.estimate,
                        arena=plan.arena, live_peak=plan.live_peak)
            stats = vae_ops.OpStats()
            with torch.inference_mode(), tracer:
                out = bound.run(v, lat, plan, ws_, stats)
                del out
                tracer.poll()
            info["stats"] = stats
        else:
            ws_ = ws or mvae.workspace()
            stats = vae_ops.OpStats()
            with torch.inference_mode(), tracer:
                with vae_ops.OpChunking(fsm, ws_, stats):
                    z = lat.to(dt)
                    out = fsm.decode(z)
                    del z
                out = out.to(torch.float32)
                del out
                tracer.poll()
    info["peak_blocks"] = [x for x in sim.peak_blocks if x[1] != "weights"]
    info["peak_segments"] = [sg for sg in sim.peak_segments if not all(t == "weights" for _, t in sg[1] if t is not None) or len(sg[1]) == 1 and sg[1][0][1] is None and False]
    info.update(alloc=sim.peak_allocated - base_alloc, reserved=sim.peak_reserved - base_res, mallocs=sim.mallocs,
                empty_cache=events["empty_cache"])
    return info


# ---------------------------------------------------------------------------
# validation against CT 700
# ---------------------------------------------------------------------------

# (label, w, h, dtype, rows, layer, version, measured alloc GiB or None, measured reserved GiB)
# version "v1" = 4e54d20 (SlowDilated3d, no cache emptied, top-to-bottom), "v2" = 725a010, "v3" = 85a5c6f
# (arena, layer-1 workspace 384 MiB), "cur" = the current code (layer-1 workspace 128 MiB; GTT readings of
# the -w128 runs of the workspace experiment). rows None = the default plan.
MEASURED = [
    ("4K r32 v2", 3840, 2160, "bf16", 32, 1, "v2", 0.94, 1.07),
    ("4K r64 v2", 3840, 2160, "bf16", 64, 1, "v2", 0.94, 1.14),
    ("4K r128 v2", 3840, 2160, "bf16", 128, 1, "v2", 0.97, 1.36),
    ("4K r155 v2", 3840, 2160, "bf16", 155, 1, "v2", 1.04, 1.44),
    ("4K r256 v2", 3840, 2160, "bf16", 256, 1, "v2", 1.24, 1.46),
    ("4K r512 v2", 3840, 2160, "bf16", 512, 1, "v2", 1.70, 2.09),
    ("1344 6x128 v2", 1344, 768, "bf16", 128, 1, "v2", 0.57, 0.70),
    ("2688 12x128 v2", 2688, 1536, "bf16", 128, 1, "v2", 0.78, 1.02),
    ("fp32 1344 6x128 v2", 1344, 768, "fp32", 128, 1, "v2", 0.72, 0.96),
    ("fp32 2688 12x128 v2", 2688, 1536, "fp32", 128, 1, "v2", 1.07, 1.41),
    ("fp32 4K 13x167 v2", 3840, 2160, "fp32", 167, 1, "v2", 1.61, 2.07),
    ("4K r32 v1", 3840, 2160, "bf16", 32, 1, "v1", 0.94, 1.07),
    ("4K r64 v1", 3840, 2160, "bf16", 64, 1, "v1", 0.94, 1.07),
    ("4K r128 v1", 3840, 2160, "bf16", 128, 1, "v1", 0.97, 1.07),
    ("4K r256 v1", 3840, 2160, "bf16", 256, 1, "v1", 1.24, 1.44),
    ("4K r512 v1", 3840, 2160, "bf16", 512, 1, "v1", 1.70, 2.66),
    ("1344 1x768 v1", 1344, 768, "bf16", 768, 1, "v1", 1.11, 1.27),
    ("2688 3x512 v1", 2688, 1536, "bf16", 512, 1, "v1", 1.42, 2.17),
    ("fp32 1344 2x384 v1", 1344, 768, "fp32", 384, 1, "v1", 1.13, 1.80),
    ("fp32 2688 5x308 v1", 2688, 1536, "fp32", 308, 1, "v1", 1.67, 2.70),
    ("fp32 4K 10x216 v1", 3840, 2160, "fp32", 216, 1, "v1", 1.84, 3.22),
    ("4K default v3", 3840, 2160, "bf16", None, 1, "v3", 1.12, 1.13),
    ("2688 default v3", 2688, 1536, "bf16", None, 1, "v3", 0.86, 0.87),
    ("4K r32 v3", 3840, 2160, "bf16", 32, 1, "v3", 1.12, 1.13),
    ("4K r256 v3", 3840, 2160, "bf16", 256, 1, "v3", 1.35, 1.35),
    ("4K r512 v3", 3840, 2160, "bf16", 512, 1, "v3", 1.82, 1.82),
    ("fp32 4K default v3", 3840, 2160, "fp32", None, 1, "v3", 1.70, 1.70),
    ("1344 default cur", 1344, 768, "bf16", None, 1, "cur", None, 0.36),
    ("2688 default cur", 2688, 1536, "bf16", None, 1, "cur", None, 0.56),
    ("4K default cur", 3840, 2160, "bf16", None, 1, "cur", None, 0.87),
    ("1344 layer 2", 1344, 768, "bf16", None, 2, "v1", 1.82, 2.12),
    ("2688 layer 2", 2688, 1536, "bf16", None, 2, "v1", 3.80, 5.07),
    ("4K layer 2", 3840, 2160, "bf16", None, 2, "v1", 6.45, 9.61),
]


def _version(version):
    if version == "v1":
        return dict(clear=False, order="natural", conv2d=False, arena=0, out_first=False, layer1_ws=384 * MIB, contiguous=False)
    if version == "v2":
        return dict(clear=True, order="largest", conv2d=True, arena=0, out_first=False, layer1_ws=384 * MIB, contiguous=False)
    if version == "v3":
        return dict(layer1_ws=384 * MIB)
    return {}


def main():
    import common  # noqa: F401  (ComfyUI environment)
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=None)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--version", default="current", help="current / v1 / v2 / v3")
    ap.add_argument("--peak", action="store_true", help="list the live blocks at the allocation peak")
    ap.add_argument("--segments", action="store_true", help="the segments (>= 2 MiB) and their blocks when reserved peaked")
    a = ap.parse_args()
    G = float(1 << 30)
    if a.res:
        w, h = (int(x) for x in a.res.lower().split("x"))
        for r in ([int(x) for x in a.rows.split(",")] if a.rows else [None]):
            i = decode_trace(w, h, a.dtype, r, **_version(a.version))
            print("{} {} rows {}: {} x {} rows, estimate {:.2f} | sim alloc {:.2f} reserved {:.2f} GiB, {} device mallocs, cache emptied {}x".format(
                a.res, a.dtype, r or "default", i["stripes"], i["rows"], i["estimate"] / G, i["alloc"] / G, i["reserved"] / G, i["mallocs"], i["empty_cache"]))
            if a.segments:
                for size, blocks in i["peak_segments"]:
                    if size >= (2 << 20) and not any(t == "weights" for _, t in blocks):
                        print("   segment {:5.0f} MiB: {}".format(size / MIB, " | ".join(
                            "{:.0f} {}".format(sz / MIB, "free" if t is None else t.replace("aten::", "")) for sz, t in blocks if sz >= MIB)))
            if a.peak:
                print("   live at the alloc peak:", ", ".join("{:.0f} MiB {}".format(sz / MIB, t) for sz, t in sorted(i["peak_blocks"], reverse=True) if sz >= MIB))
        return
    print("{:24s} {:>13s} {:>13s} {:>9s}".format("case", "alloc m/sim", "resv m/sim", "estimate"))
    for label, w, h, dt, rows, layer, ver, ma, mr in MEASURED:
        i = decode_trace(w, h, dt, rows, layer=layer, **_version(ver))
        est = i.get("estimate")
        print("{:24s} {:>5s} / {:5.2f} {:5.2f} / {:5.2f} {:>9s}".format(label, "{:.2f}".format(ma) if ma is not None else "-", i["alloc"] / G,
                                                                    mr, i["reserved"] / G, "{:.2f}".format(est / G) if est else ""))


if __name__ == "__main__":
    main()
