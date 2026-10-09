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
        self.ctx = ""           # the layer-1 unit / prefix module running (decode_trace(units=True))
        self.heads = []         # first block of every segment
        self.peak_segments = [] # segment layout when reserved peaked: [(segment size, [(block size, tag or None if free)])]
        self.watch = None       # (start address, end address) of a segment whose high-water mark is tracked
        self.hwm = 0            # highest end offset of a block allocated in it
        self.hwm_at = None      # (request tag, size, [(offset, size, tag) of the blocks live in it]) when it was reached

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
        if self.watch is not None and self.watch[0] <= b.addr < self.watch[1] and b.addr + size - self.watch[0] > self.hwm:
            self.hwm = b.addr + size - self.watch[0]
            self.hwm_at = (self.tag, size, sorted(((x.addr - self.watch[0], x.size, x.requested[1]) for x in self.live
                                                   if self.watch[0] <= x.addr < self.watch[1]), key=lambda t: t[0]))
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
            sim.tag = (sim.ctx + ":" if sim.ctx else "") + name
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
# the LDM ddconfig comfy.sd.VAE builds for SDXL / SD1.5 (z 4, AutoencoderKL with post_quant_conv) and Flux ae /
# SD3 (z 16, AutoencodingEngine, no post_quant_conv)
LDM_DDCONFIG = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128,
                'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
# model name -> (latent channels, latent dims[, spatial ratio (default 8)])
LATENT = {"qwen": (16, 5), "sdxl": (4, 4), "flux": (16, 4), "flux2": (128, 4, 16)}


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


def _build(model):
    """The full-size first-stage model (on the meta device: no memory)."""
    if model == "qwen":
        import comfy.ldm.wan.vae as wan
        return wan.WanVAE(**WAN_QWEN)
    from comfy.ldm.models.autoencoder import AutoencoderKL, AutoencodingEngine
    if model == "sdxl":
        return AutoencoderKL(ddconfig=dict(LDM_DDCONFIG), embed_dim=4)
    if model == "flux2":
        # Flux 2: AutoencoderKL with a batch-norm latent (z 32, latent 128 channels at H/16), as comfy.sd.VAE builds it
        return AutoencoderKL(ddconfig=dict(LDM_DDCONFIG, z_channels=32, batch_norm_latent=True), embed_dim=32)
    if model == "flux":
        dd = dict(LDM_DDCONFIG, z_channels=16)
        return AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                  encoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Encoder", 'params': dd},
                                  decoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Decoder", 'params': dd})
    raise ValueError(model)


def meta_vae(dtype, model="qwen"):
    import torch
    key = (dtype, model)
    if key not in _MODELS:
        with torch.device("meta"):
            fsm = _build(model)
        fsm.to(dtype).eval()
        _MODELS[key] = _MetaVAE(fsm, dtype)
    return _MODELS[key]


@contextlib.contextmanager
def _patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def decode_trace(w, h, dtype="bf16", rows=None, layer=1, ws=None, clear=False, order="largest", conv2d=True, arena=None, batch=1, out_first=True,
                 layer1_ws=None, contiguous=True, model="qwen", scheme=None, arena_need=False, units=False, output_in_arena=False,
                 probe=False):
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
    model   "qwen" (Wan 2.1, qwen_image_vae), "sdxl" (LDM AutoencoderKL, z 4), "flux" (LDM AutoencodingEngine, z 16),
            "flux2" (LDM AutoencoderKL with a batch-norm latent, z 32, latent 128 x H/16)
    scheme  GroupNorm scheme of an LDM layer-1 decode (None: the current setting)
    units   tag every allocation with the layer-1 unit / prefix module that made it (for --peak)
    output_in_arena  the layer-1 layout up to 6324592 (vae_engine.OUTPUT_IN_ARENA): no cache emptied before / after the
            decode, the output buffer allocated after the first prefix, in the arena (False: the current layout, DESIGN §9.22)
    probe   run choose_budget's layer-2 shape probe (vae._probe, 8 x 8) right before a layer-1 decode, its blocks left
            in the cache: the first decode with a budget (DESIGN §9.21)
    info["stays"]  reserved after the decode with its output still alive and the cache emptied (check_selftest_mem's
            "stays reserved after empty_cache")
    arena_need  run in an arena 4x the plan's live peak and report (info["arena_need"]) the highest offset any
            block reached in it: the smallest arena with the same placements (best fit keeps choosing the same
            blocks while the arena's tail is the largest free block), i.e. the arena this plan needs
    """
    import torch
    import comfy.model_management as mm
    from monoload import vae as mvae, vae_ops, vae_engine as eng

    dt = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[dtype] if isinstance(dtype, str) else dtype
    v = meta_vae(dt, model)
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
    zc, nd, r = (LATENT[model] + (8,))[:3]
    # the bench's latent is on the CPU: not counted, its copy is. float64, so that the decode's .to(device, dtype) is a
    # real copy for every decode dtype (an fp32 latent on meta would be returned as is by .to(meta, fp32): review
    # 2026-10 item 15, the fp32 configurations missed the latent's copy)
    lat = torch.empty((batch, zc, 1, h // r, w // r) if nd == 5 else (batch, zc, h // r, w // r), device="meta", dtype=torch.float64)
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
        es.enter_context(_patched(eng, "OUTPUT_IN_ARENA", output_in_arena))
        es.enter_context(_patched(eng, "block_addr", lambda t: tracer.live[t.untyped_storage()._cdata][1].addr))
        es.enter_context(_patched(eng, "device_reserved", lambda device: sim.reserved))
        if probe:   # the probe's attention (not under OpChunking) asks for the free memory
            es.enter_context(_patched(mm, "get_free_memory", lambda dev=None, torch_free_too=False: (48 << 30, 48 << 30) if torch_free_too else 48 << 30))
        if layer1_ws:
            es.enter_context(_patched(mvae, "LAYER1_WORKSPACE", layer1_ws))
        if not contiguous:
            es.enter_context(_patched(eng, "CONTIGUOUS_INPUT", ()))
        if clear:
            es.enter_context(_patched(eng, "run_prefix", run_prefix_clear))
        if scheme:
            from monoload import vae_ldm
            es.enter_context(_patched(vae_ldm, "_SCHEME", [scheme]))
        if layer == 1:
            bound, why = mvae._select_layer1(v, lat, {})
            assert bound is not None, why
            if units:
                def tagged(fn, label):
                    def run(x):
                        old = sim.ctx
                        sim.ctx = label
                        try:
                            return fn(x)
                        finally:
                            sim.ctx = old
                    return run
                for u in bound.units:
                    u.module = tagged(u.module, u.name.replace("decoder.", ""))
                    for nr in u.norms:
                        if nr.partial is not None:
                            nr.partial.module = tagged(nr.partial.module, nr.partial.name.replace("decoder.", ""))
                bound.prefix_pass = tagged(bound.prefix_pass, "prefix")
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
            if arena_need:
                plan.arena = 4 * plan.live_peak
                orig_reserve = eng.reserve_arena

                def reserve(device, nbytes):
                    seg = sim.allocation_size(sim.round_size(nbytes))
                    sim.watch = (sim.next_addr, sim.next_addr + seg)
                    orig_reserve(device, nbytes)
                    sim.hwm = 0   # the arena block itself (freed before the next op's first allocation)
                es.enter_context(_patched(eng, "reserve_arena", reserve))
            info.update(plan=plan, stripes=len(plan.stripes), rows=max(b - a for a, b in plan.stripes), estimate=plan.estimate,
                        arena=plan.arena, live_peak=plan.live_peak)
            stats = vae_ops.OpStats()
            with torch.inference_mode(), tracer:
                if probe:
                    mvae._PROBES.clear()
                    mvae._probe(v, lat, {})
                    tracer.poll()
                out = bound.run(v, lat, plan, ws_, stats)
                tracer.poll()
                sim.empty_cache()
                info["stays"] = sim.reserved - base_res
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
                empty_cache=events["empty_cache"], arena_need=sim.hwm, arena_need_at=sim.hwm_at)
    return info


def selftest_trace(model, dtype="bf16", tail=True, info=None):
    """Reserved peak (bytes) of the first-use layer-1 self-test on the full-size decoder (meta device), the steps
    of vae_engine._self_test_run: the fp32 copy, the reference whole-image decode under layer-2 chunking
    (SELFTEST_REF_WORKSPACE), the forced small stripes (SELFTEST_ROWS, SELFTEST_WORKSPACE); the VAE's own weights
    loaded before; tail=True: also the comparison at the end (max|ref|, max|out - ref|, isfinite; info, a dict:
    what it allocates and adds to reserved, tail_alloc / tail_reserved, and reserved before it). Returns (peak
    reserved, reserved left after the copy and every tensor are freed, the bound's selftest_memory())."""
    import gc
    import torch
    import comfy.model_management as mm
    from monoload import vae as mvae, vae_ops, vae_engine as eng

    dt = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[dtype]
    v = meta_vae(dt, model)
    fsm = v.first_stage_model
    zc, nd, r = (LATENT[model] + (8,))[:3]
    lat = torch.empty((1, zc, 1, 64, 64) if nd == 5 else (1, zc, 64, 64), device="meta")
    bound, why = mvae._select_layer1(v, lat, {})
    assert bound is not None, why
    sim = AllocatorSim()
    sim.tag = "weights"
    for p in itertools.chain(fsm.parameters(), fsm.buffers()):
        sim.malloc(p.numel() * p.element_size())
    sim.empty_cache()
    sim.reset_peak()
    base = sim.reserved
    tracer = make_tracer(sim)
    tracer.static.update(t.untyped_storage()._cdata for t in itertools.chain(fsm.parameters(), fsm.buffers()))

    def soft_empty_cache(force=False):
        tracer.poll()
        sim.empty_cache()

    free = 48 << 30
    with contextlib.ExitStack() as es:
        es.enter_context(_patched(vae_ops, "slow_dilated3d", lambda x: True))
        es.enter_context(_patched(mm, "soft_empty_cache", soft_empty_cache))
        es.enter_context(_patched(mm, "get_free_memory", lambda dev=None, torch_free_too=False: (free, free) if torch_free_too else free))
        es.enter_context(torch.inference_mode())
        es.enter_context(tracer)
        dev = torch.device("meta")
        tb = bound.fp32_copy(dev)
        z = torch.empty(list(bound.selftest_latent(eng.SELFTEST_LATENT)), device=dev)
        with vae_ops.OpChunking(tb.module, eng.SELFTEST_REF_WORKSPACE, vae_ops.OpStats()):
            ref = tb.reference_decode(z)
        n = eng.SELFTEST_LATENT
        plan = eng.Plan(tb, n, n, eng.SELFTEST_ROWS, eng.SELFTEST_WORKSPACE, 4, 0, 0)
        out = torch.empty_like(ref)

        def write(o0, o1, rows):
            out.narrow(tb.hdim, o0, o1 - o0).copy_(rows)
        with vae_ops.OpChunking(tb.module, eng.SELFTEST_WORKSPACE, vae_ops.OpStats()):
            tb.stripe_pass({0: tb.prefix_pass(z)}, plan, write)
        if tail:
            # the comparison (vae_engine._self_test_run): max|ref|, max|out - ref|, isfinite(out), each statement's
            # temporaries freed before the next (float() / bool() of a meta tensor cannot be taken: the reductions only)
            tracer.poll()
            a0, r0, p_alloc, p_res = sim.allocated, sim.reserved, sim.peak_allocated, sim.peak_reserved
            sim.peak_allocated, sim.peak_reserved = a0, r0
            m = ref.abs().max()
            del m
            m = (out - ref).abs().max()
            del m
            m = torch.isfinite(out).all()
            del m
            tracer.poll()
            if info is not None:
                info.update(tail_alloc=sim.peak_allocated - a0, tail_reserved=sim.peak_reserved - r0, before_tail=r0 - base)
            sim.peak_allocated, sim.peak_reserved = max(p_alloc, sim.peak_allocated), max(p_res, sim.peak_reserved)
        del tb, z, ref, out, plan
        gc.collect()
        tracer.poll()
    sim.empty_cache()
    return sim.peak_reserved - base, sim.reserved - base, bound.selftest_memory()


# ---------------------------------------------------------------------------
# validation against CT 700
# ---------------------------------------------------------------------------

# (label, w, h, dtype, rows, layer, version, measured alloc GiB or None, measured reserved GiB)
# version "v1" = 4e54d20 (SlowDilated3d, no cache emptied, top-to-bottom), "v2" = 725a010, "v3" = 85a5c6f
# (arena, layer-1 workspace 384 MiB), "w128" = 5d668b6..6324592 (layer-1 workspace 128 MiB, the output in the
# arena; GTT readings of the -w128 runs of the workspace experiment), "cur" = the current code (layer 2: unchanged
# since 5d668b6). v1..w128 replay the layer-1 layout of their time (output_in_arena). rows None = the default plan.
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
    ("1344 default w128", 1344, 768, "bf16", None, 1, "w128", None, 0.36),
    ("2688 default w128", 2688, 1536, "bf16", None, 1, "w128", None, 0.56),
    ("4K default w128", 3840, 2160, "bf16", None, 1, "w128", None, 0.87),
    # Flux 2 layer 1 (README §10.5, 4fbaa22; GTT): X default (B), Y schemes at 4K, Z the budgets' choices
    ("F2 1344 default w128", 1344, 768, "bf16", None, 1, "w128", None, 0.62, dict(model="flux2")),
    ("F2 2688 default w128", 2688, 1536, "bf16", None, 1, "w128", None, 1.33, dict(model="flux2")),
    ("F2 4K default w128", 3840, 2160, "bf16", None, 1, "w128", None, 2.18, dict(model="flux2")),
    ("F2 4K A w128", 3840, 2160, "bf16", None, 1, "w128", None, 1.11, dict(model="flux2", scheme="A")),
    ("F2 4K D w128", 3840, 2160, "bf16", None, 1, "w128", None, 1.53, dict(model="flux2", scheme="D")),
    ("F2 4K C w128", 3840, 2160, "bf16", None, 1, "w128", None, 4.71, dict(model="flux2", scheme="C")),
    ("F2 1344 b3 w128", 1344, 768, "bf16", 768, 1, "w128", None, 1.98, dict(model="flux2", scheme="B", ws=384 * MIB)),
    ("F2 2688 b3 w128", 2688, 1536, "bf16", 384, 1, "w128", None, 2.74, dict(model="flux2", scheme="C", ws=128 * MIB)),
    ("F2 4K b3 w128", 3840, 2160, "bf16", 180, 1, "w128", None, 2.44, dict(model="flux2", scheme="B", ws=128 * MIB)),
    ("F2 1344 b1.5 w128", 1344, 768, "bf16", 384, 1, "w128", None, 1.05, dict(model="flux2", scheme="C", ws=192 * MIB)),
    ("F2 2688 b1.5 w128", 2688, 1536, "bf16", 96, 1, "w128", None, 1.22, dict(model="flux2", scheme="B", ws=128 * MIB)),
    ("F2 4K b1.5 w128", 3840, 2160, "bf16", 128, 1, "w128", None, 1.11, dict(model="flux2", scheme="A", ws=128 * MIB)),
    ("1344 layer 2", 1344, 768, "bf16", None, 2, "v1", 1.82, 2.12),
    ("2688 layer 2", 2688, 1536, "bf16", None, 2, "v1", 3.80, 5.07),
    ("4K layer 2", 3840, 2160, "bf16", None, 2, "v1", 6.45, 9.61),
    # SDXL / Flux layer 2 (README §10.1, af9abc6: the row-blocked conv allocated its output after the first block)
    ("SDXL 1344 layer 2", 1344, 768, "bf16", None, 2, "v1", 2.41, 3.72, dict(model="sdxl")),
    ("SDXL 2688 layer 2", 2688, 1536, "bf16", None, 2, "v1", 6.15, 9.25, dict(model="sdxl")),
    ("SDXL 4K layer 2", 3840, 2160, "bf16", None, 2, "v1", 11.17, 14.86, dict(model="sdxl")),
    ("Flux 1344 layer 2", 1344, 768, "bf16", None, 2, "v1", 2.41, 3.72, dict(model="flux")),
    ("Flux 2688 layer 2", 2688, 1536, "bf16", None, 2, "v1", 6.15, 9.24, dict(model="flux")),
    ("Flux 4K layer 2", 3840, 2160, "bf16", None, 2, "v1", 11.18, 14.89, dict(model="flux")),
    # SDXL layer 2, workspace experiment K (5d668b6; GTT, the reserved peak plus <= 0.02 GiB of the rest of the machine)
    ("SDXL 1344 l2 1G", 1344, 768, "bf16", None, 2, "cur", None, 3.72, dict(model="sdxl", ws=1024 * MIB)),
    ("SDXL 2688 l2 1G", 2688, 1536, "bf16", None, 2, "cur", None, 9.23, dict(model="sdxl", ws=1024 * MIB)),
    ("SDXL 4K l2 1G", 3840, 2160, "bf16", None, 2, "cur", None, 14.99, dict(model="sdxl", ws=1024 * MIB)),
    ("SDXL 1344 l2 512M", 1344, 768, "bf16", None, 2, "cur", None, 2.68, dict(model="sdxl", ws=512 * MIB)),
    ("SDXL 2688 l2 512M", 2688, 1536, "bf16", None, 2, "cur", None, 7.47, dict(model="sdxl", ws=512 * MIB)),
    ("SDXL 4K l2 512M", 3840, 2160, "bf16", None, 2, "cur", None, 15.86, dict(model="sdxl", ws=512 * MIB)),
    ("SDXL 1344 l2 256M", 1344, 768, "bf16", None, 2, "cur", None, 2.30, dict(model="sdxl", ws=256 * MIB)),
    ("SDXL 2688 l2 256M", 2688, 1536, "bf16", None, 2, "cur", None, 7.91, dict(model="sdxl", ws=256 * MIB)),
    ("SDXL 4K l2 256M", 3840, 2160, "bf16", None, 2, "cur", None, 15.42, dict(model="sdxl", ws=256 * MIB)),
    ("SDXL 1344 l2 128M", 1344, 768, "bf16", None, 2, "cur", None, 1.87, dict(model="sdxl", ws=128 * MIB)),
    ("SDXL 2688 l2 128M", 2688, 1536, "bf16", None, 2, "cur", None, 7.67, dict(model="sdxl", ws=128 * MIB)),
    ("SDXL 4K l2 128M", 3840, 2160, "bf16", None, 2, "cur", None, 15.32, dict(model="sdxl", ws=128 * MIB)),
]


def _version(version):
    if version == "v1":
        return dict(clear=False, order="natural", conv2d=False, arena=0, out_first=False, layer1_ws=384 * MIB, contiguous=False,
                    output_in_arena=True)
    if version == "v2":
        return dict(clear=True, order="largest", conv2d=True, arena=0, out_first=False, layer1_ws=384 * MIB, contiguous=False,
                    output_in_arena=True)
    if version == "v3":
        return dict(layer1_ws=384 * MIB, output_in_arena=True)
    if version == "w128":
        return dict(output_in_arena=True)
    return {}


def main():
    import common  # noqa: F401  (ComfyUI environment)
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=None)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--version", default="current", help="current / v1 / v2 / v3 / w128")
    ap.add_argument("--model", default="qwen", help="qwen / sdxl / flux / flux2")
    ap.add_argument("--scheme", default=None, help="GroupNorm scheme of an LDM layer-1 decode: A / B / C / D")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--ws", default=None, help="workspace MiB (layer 2: default 1024; layer 1: the layer-1 default)")
    ap.add_argument("--probe", action="store_true", help="layer 1: the budget's shape probe right before the decode")
    ap.add_argument("--peak", action="store_true", help="list the live blocks at the allocation peak")
    ap.add_argument("--segments", action="store_true", help="the segments (>= 2 MiB) and their blocks when reserved peaked")
    a = ap.parse_args()
    G = float(1 << 30)
    if a.res:
        w, h = (int(x) for x in a.res.lower().split("x"))
        for r in ([int(x) for x in a.rows.split(",")] if a.rows else [None]):
            kw = dict(_version(a.version), model=a.model, scheme=a.scheme, layer=a.layer, units=a.layer == 1)
            if a.probe:
                kw["probe"] = True
            if a.ws:
                kw["ws"] = int(float(a.ws) * MIB)
            i = decode_trace(w, h, a.dtype, r, **kw)
            if a.layer == 1:
                p = i["plan"]
                print("{} {} {} rows {}: {} x {} rows{}, recompute {:.2f}x, live {:.2f} arena {:.2f} estimate {:.2f} | sim alloc {:.2f} reserved {:.2f} GiB, "
                      "stays {:.2f} GiB, {} device mallocs, cache emptied {}x".format(
                          a.model, a.res, a.dtype, r or "default", i["stripes"], i["rows"],
                          ", {} statistics passes (scheme {})".format(len(p.passes), a.scheme or "default") if p.passes else "", p.recompute,
                          p.live_peak / G, p.arena / G, i["estimate"] / G, i["alloc"] / G, i["reserved"] / G, i["stays"] / G, i["mallocs"],
                          i["empty_cache"]))
            else:
                print("{} {} {} layer 2: sim alloc {:.2f} reserved {:.2f} GiB, {} device mallocs".format(
                    a.model, a.res, a.dtype, i["alloc"] / G, i["reserved"] / G, i["mallocs"]))
            if a.segments:
                for size, blocks in i["peak_segments"]:
                    if size >= (2 << 20) and not any(t == "weights" for _, t in blocks):
                        print("   segment {:5.0f} MiB: {}".format(size / MIB, " | ".join(
                            "{:.0f} {}".format(sz / MIB, "free" if t is None else t.replace("aten::", "")) for sz, t in blocks if sz >= MIB)))
            if a.peak:
                print("   live at the alloc peak:", ", ".join("{:.0f} MiB {}".format(sz / MIB, t) for sz, t in sorted(i["peak_blocks"], reverse=True) if sz >= MIB))
        return
    print("{:24s} {:>13s} {:>13s} {:>9s}".format("case", "alloc m/sim", "resv m/sim", "estimate"))   # resv m: reserved, or GTT where only that was read
    for label, w, h, dt, rows, layer, ver, ma, mr, *kw in MEASURED:
        i = decode_trace(w, h, dt, rows, layer=layer, **dict(_version(ver), **(kw[0] if kw else {})))
        est = i.get("estimate")
        print("{:24s} {:>5s} / {:5.2f} {:5.2f} / {:5.2f} {:>9s}".format(label, "{:.2f}".format(ma) if ma is not None else "-", i["alloc"] / G,
                                                                    mr, i["reserved"] / G, "{:.2f}".format(est / G) if est else ""))


if __name__ == "__main__":
    main()
