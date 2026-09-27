"""Unit test of the transfer layer (no ComfyUI needed).

Writes random tensors with the Monoload writer, then streams them back via
  * the direct CPU path,
  * the staged CPU path (tiny buffers -> packing and splitting),
  * the CUDA pipeline code path, driven on CPU through fake Stream/Event
    objects (this machine has no GPU; it checks chunking, buffer rotation,
    event bookkeeping and non-contiguous destinations, not real async H2D).
"""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monoload import fmt, transfer  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print("[{}] {}{}".format("PASS" if ok else "FAIL", name, (" — " + detail) if detail else ""))
    if not ok:
        FAILED.append(name)


def make_tensors(seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.int64, torch.uint8, torch.bool, torch.float64]
    sizes = [0, 1, 3, 17, 1000, 4096, 65537, 300000, 1 << 20, (1 << 20) + 5]
    i = 0
    for n in sizes:
        for dt in dtypes:
            if dt.is_floating_point:
                t = torch.randn(n, generator=g).to(dt)
            elif dt == torch.bool:
                t = torch.randint(0, 2, (n,), generator=g).bool()
            else:
                t = torch.randint(0, 100, (n,), generator=g).to(dt)
            if n == 300000:
                t = t.reshape(300, 1000)
            out.append(("t{}_{}_{}".format(i, n, str(dt)[6:]), t))
            i += 1
    out.append(("scalar", torch.tensor(3.5)))
    out.append(("conv", torch.randn(64, 32, 3, 3, generator=g).half()))
    return out


class FakeEvent:
    log = []

    def record(self, stream):
        self.recorded = True
        FakeEvent.log.append("record")

    def synchronize(self):
        assert getattr(self, "recorded", False), "synchronize before record"
        FakeEvent.log.append("sync")


class FakeStream:
    def __init__(self, device=None):
        pass

    def wait_stream(self, other):
        pass

    def synchronize(self):
        pass


class FakeStreamCtx:
    def __init__(self, s):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def run(path, header, tensors, dests_fn, buf_mb_bytes, mode, fake_cuda=False):
    dests = dests_fn()
    items = [(header.data_start + header.by_name[n].begin, header.by_name[n].nbytes, d) for (n, _), d in zip(tensors, dests)]
    items.sort(key=lambda x: x[0])
    os.environ["MONOLOAD_STAGING"] = mode
    saved = None
    if fake_cuda:
        saved = (transfer._is_cuda, torch.cuda.Stream, torch.cuda.Event, torch.cuda.stream, torch.cuda.current_stream, transfer._alloc_staging)
        transfer._is_cuda = lambda dev: True
        torch.cuda.Stream = FakeStream
        torch.cuda.Event = FakeEvent
        torch.cuda.stream = FakeStreamCtx
        torch.cuda.current_stream = lambda device=None: FakeStream()
        transfer._alloc_staging = lambda n, dev: (torch.empty(n, dtype=torch.uint8), False)
    fd = fmt.open_readonly(path)
    try:
        stats = transfer.stream_into_tensors(fd, items, torch.device("cpu"), buf_mb_bytes, path)
    finally:
        os.close(fd)
        if saved:
            (transfer._is_cuda, torch.cuda.Stream, torch.cuda.Event, torch.cuda.stream, torch.cuda.current_stream, transfer._alloc_staging) = saved
    ok = all(d.dtype == t.dtype and torch.equal(d.contiguous().reshape(-1).view(torch.uint8), t.contiguous().reshape(-1).view(torch.uint8))
             for (_, t), d in zip(tensors, dests))
    return ok, stats


def main():
    tensors = make_tensors()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.safetensors")
        hb, _ = fmt.build_header_bytes([(n, t.dtype, t.shape, t.numel() * t.element_size()) for n, t in tensors], {"k": "v"})
        transfer.write_tensor_file(path, hb, [t for _, t in tensors])
        header = fmt.read_header_path(path)
        check("header roundtrip: names/order/metadata", [t.name for t in header.tensors] == [n for n, _ in tensors] and header.metadata == {"k": "v"})
        sd = transfer.read_state_dict_cpu(path, header)
        check("read_state_dict_cpu equals written tensors", all(torch.equal(sd[n].reshape(-1).view(torch.uint8), t.reshape(-1).view(torch.uint8)) for n, t in tensors if t.numel()))
        check("read_state_dict_cpu keys sorted like safe_open", list(sd) == sorted(sd))

        def fresh():
            return [torch.empty(t.shape, dtype=t.dtype) for _, t in tensors]

        def fresh_noncontig():
            out = []
            for _, t in tensors:
                if t.dim() == 4:
                    out.append(torch.empty(t.shape, dtype=t.dtype).to(memory_format=torch.channels_last))
                elif t.dim() == 2:
                    out.append(torch.empty(tuple(reversed(t.shape)), dtype=t.dtype).t())
                else:
                    out.append(torch.empty(t.shape, dtype=t.dtype))
            return out

        ok, st = run(path, header, tensors, fresh, 512 << 20, "auto")
        check("direct CPU path", ok and st.path == "direct", str(st))
        for buf in (1 << 20, 4096 + 3, 8 << 20):
            ok, st = run(path, header, tensors, fresh, buf, "always")
            check("staged CPU path, buffer {} B".format(buf), ok and st.path == "staged", str(st))
            FakeEvent.log.clear()
            ok, st = run(path, header, tensors, fresh, buf, "always", fake_cuda=True)
            syncs = FakeEvent.log.count("sync")
            check("CUDA pipeline (fake stream), buffer {} B".format(buf), ok and syncs == max(0, st.chunks - transfer.NUM_BUFFERS),
                  "{}; event syncs {} for {} chunks".format(st, syncs, st.chunks))
        ok, st = run(path, header, tensors, fresh_noncontig, 8 << 20, "always", fake_cuda=True)
        check("non-contiguous destinations (channels_last / transposed)", ok, str(st))
        ok, st = run(path, header, tensors, fresh_noncontig, 512 << 20, "auto")
        check("non-contiguous destinations, direct path", ok, str(st))
        try:
            run(path, header, tensors, fresh_noncontig, 4096, "always")
            check("non-contiguous tensor larger than buffer is refused", False, "no error")
        except Exception as e:
            check("non-contiguous tensor larger than buffer is refused", "MONOLOAD_BUFFER_MB" in str(e), str(e)[:120])
    print("\n== {} failed".format(len(FAILED)))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
