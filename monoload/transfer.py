"""Bulk data movement between disk and tensors.

Rules enforced here:
  * files are only ever accessed with os.preadv / os.read / os.write — never mmap;
  * no full CPU copy of a model is ever assembled by the loader: data goes
    disk -> (pinned staging buffer ->) final parameter storage.
"""

import logging
import os
import time

import torch

from .errors import MonoloadError, MonoloadUnsupportedError

DEFAULT_BUFFER_MB = 512
NUM_BUFFERS = 2


def buffer_bytes_from(value_mb=None):
    """Resolve the staging buffer size: explicit value > $MONOLOAD_BUFFER_MB > 512."""
    mb = value_mb
    if not mb:
        env = os.environ.get("MONOLOAD_BUFFER_MB", "").strip()
        mb = int(env) if env else DEFAULT_BUFFER_MB
    mb = int(mb)
    if mb < 1:
        raise ValueError("MONOLOAD buffer size must be >= 1 MiB")
    return mb * 1024 * 1024


def staging_mode():
    """'auto' (default) or 'always' (force the staging path even for CPU targets; used by tests)."""
    mode = os.environ.get("MONOLOAD_STAGING", "auto").strip().lower() or "auto"
    if mode not in ("auto", "always"):
        raise ValueError("MONOLOAD_STAGING must be 'auto' or 'always'")
    return mode


def pread_into(fd, mv, offset, path="<file>"):
    """Fill the writable memoryview `mv` from `fd` at `offset` (loops on short reads)."""
    total = len(mv)
    got = 0
    while got < total:
        n = os.preadv(fd, [mv[got:]], offset + got)
        if n == 0:
            raise MonoloadError("[Monoload] {}: 读取到文件末尾（偏移 {}），文件被截断了？".format(path, offset + got))
        got += n


def byte_view(t):
    """A flat uint8 view sharing storage with contiguous tensor `t`."""
    return t.detach().reshape(-1).view(torch.uint8)


def cpu_memoryview(t):
    """Writable memoryview over the bytes of a contiguous CPU tensor."""
    return memoryview(byte_view(t).numpy()).cast("B")


def _is_cuda(device):
    return device.type == "cuda"


def _alloc_staging(nbytes, device):
    pin = False
    if _is_cuda(device):
        pin = True
    try:
        buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)
    except RuntimeError:
        if not pin:
            raise
        logging.warning("[Monoload] pinned staging buffer allocation failed; falling back to pageable memory")
        buf = torch.empty(nbytes, dtype=torch.uint8)
        pin = False
    return buf, pin


def release_host_cache():
    """Return cached pinned host memory to the OS (PyTorch keeps freed pinned blocks)."""
    for fn in (getattr(torch._C, "_host_emptyCache", None),
               getattr(getattr(torch.cuda, "memory", None), "empty_host_cache", None)):
        if fn is not None:
            try:
                fn()
                return
            except Exception:
                pass


class _Piece:
    __slots__ = ("buf_off", "dst", "dst_off", "n", "whole")

    def __init__(self, buf_off, dst, dst_off, n, whole):
        self.buf_off = buf_off
        self.dst = dst          # destination tensor
        self.dst_off = dst_off  # byte offset inside the destination
        self.n = n
        self.whole = whole      # piece covers the entire destination tensor


class _Chunk:
    __slots__ = ("file_off", "nbytes", "pieces")

    def __init__(self, file_off):
        self.file_off = file_off
        self.nbytes = 0
        self.pieces = []


def plan_chunks(items, buf_bytes):
    """items: [(abs_file_offset, nbytes, dst_tensor)] sorted by offset.
    Pack contiguous small tensors into one chunk (<= buf_bytes); split big ones."""
    chunks = []
    cur = None
    for off, n, dst in items:
        if n == 0:
            continue
        contiguous = dst.is_contiguous()
        if not contiguous and n > buf_bytes:
            raise MonoloadUnsupportedError(
                "non_contiguous_large_tensor",
                "目标张量不是连续内存且大于搬运缓冲（{} > {} 字节），请调大 MONOLOAD_BUFFER_MB".format(n, buf_bytes))
        pos = 0
        while pos < n:
            room = 0 if cur is None else buf_bytes - cur.nbytes
            need_new = cur is None or cur.file_off + cur.nbytes != off + pos or room == 0
            if not need_new and not contiguous and room < n:
                need_new = True  # a non-contiguous destination must arrive in one piece
            if need_new:
                cur = _Chunk(off + pos)
                chunks.append(cur)
                room = buf_bytes
            take = min(n - pos, room)
            cur.pieces.append(_Piece(cur.nbytes, dst, pos, take, take == n))
            cur.nbytes += take
            pos += take
    return chunks


def _copy_piece(buf_slice, piece, non_blocking):
    dst = piece.dst
    if dst.is_contiguous():
        byte_view(dst)[piece.dst_off:piece.dst_off + piece.n].copy_(buf_slice, non_blocking=non_blocking)
    else:
        # Only reached for small non-contiguous destinations (e.g. channels_last):
        # bounce through a same-device temporary with storage offset 0 so the
        # dtype view is always legal, then let copy_ handle the strides.
        tmp = torch.empty(piece.n, dtype=torch.uint8, device=dst.device)
        tmp.copy_(buf_slice, non_blocking=non_blocking)
        dst.copy_(tmp.view(dst.dtype).view(dst.shape))


class TransferStats:
    def __init__(self):
        self.bytes = 0
        self.chunks = 0
        self.seconds = 0.0
        self.path = ""
        self.buffer_bytes = 0
        self.pinned = False

    def __str__(self):
        gib = self.bytes / 1024 ** 3
        speed = gib / self.seconds if self.seconds > 0 else float("inf")
        extra = ""
        if self.path == "staged":
            extra = ", {} x {} MiB {}buffers".format(NUM_BUFFERS, self.buffer_bytes // (1024 * 1024), "pinned " if self.pinned else "")
        return "{:.2f} GiB in {:.2f}s ({:.2f} GiB/s, path={}, {} reads{})".format(gib, self.seconds, speed, self.path, self.chunks, extra)


def stream_into_tensors(fd, items, device, buf_bytes, path="<file>"):
    """Read file ranges straight into destination tensors living on `device`.

    CPU destinations are filled in place with preadv (zero staging). Other
    devices (and CPU with MONOLOAD_STAGING=always) use two staging buffers:
    while one buffer is being filled by preadv, the other one's H2D copies run
    asynchronously on a dedicated stream; events guard buffer reuse.
    """
    stats = TransferStats()
    stats.bytes = sum(n for _, n, _ in items)
    t0 = time.perf_counter()
    device = torch.device(device)

    if device.type == "cpu" and staging_mode() == "auto":
        stats.path = "direct"
        for off, n, dst in items:
            if n == 0:
                continue
            if dst.is_contiguous():
                pread_into(fd, cpu_memoryview(dst), off, path)
            else:
                tmp = torch.empty(n, dtype=torch.uint8)
                pread_into(fd, cpu_memoryview(tmp), off, path)
                dst.copy_(tmp.view(dst.dtype).view(dst.shape))
                del tmp
            stats.chunks += 1
        stats.seconds = time.perf_counter() - t0
        return stats

    stats.path = "staged"
    chunks = plan_chunks(items, buf_bytes)
    stats.chunks = len(chunks)
    largest = max((c.nbytes for c in chunks), default=0)
    size = min(buf_bytes, max(largest, 1))
    stats.buffer_bytes = size
    bufs = []
    mvs = []
    for _ in range(NUM_BUFFERS):
        b, pinned = _alloc_staging(size, device)
        bufs.append(b)
        mvs.append(cpu_memoryview(b))
        stats.pinned = pinned
    try:
        if _is_cuda(device):
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))  # allocations of the destinations first
            events = [None] * NUM_BUFFERS
            for i, chunk in enumerate(chunks):
                b = i % NUM_BUFFERS
                if events[b] is not None:
                    events[b].synchronize()  # previous H2D out of this buffer is done
                pread_into(fd, mvs[b][:chunk.nbytes], chunk.file_off, path)
                with torch.cuda.stream(stream):
                    for p in chunk.pieces:
                        _copy_piece(bufs[b][p.buf_off:p.buf_off + p.n], p, non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(stream)
                events[b] = ev
            stream.synchronize()
            torch.cuda.current_stream(device).wait_stream(stream)
        else:
            for i, chunk in enumerate(chunks):
                b = i % NUM_BUFFERS
                pread_into(fd, mvs[b][:chunk.nbytes], chunk.file_off, path)
                for p in chunk.pieces:
                    _copy_piece(bufs[b][p.buf_off:p.buf_off + p.n], p, non_blocking=False)
    finally:
        del mvs
        del bufs
        release_host_cache()
    stats.seconds = time.perf_counter() - t0
    return stats


# ---------------------------------------------------------------------------
# Converter helpers
# ---------------------------------------------------------------------------

def read_state_dict_cpu(path, header):
    """Equivalent of comfy.utils.load_torch_file(path) under --disable-mmap
    (a dict of independent CPU tensors, same keys/dtypes/shapes, same key order)
    but read with preadv instead of safetensors' mmap."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        tensors = {}
        for info in header.tensors:  # file order -> sequential IO
            t = torch.empty(info.shape, dtype=info.dtype)
            if info.nbytes:
                pread_into(fd, cpu_memoryview(t), header.data_start + info.begin, path)
            tensors[info.name] = t
    finally:
        os.close(fd)
    # safetensors' safe_open().keys() returns the names sorted.
    return {k: tensors[k] for k in sorted(tensors)}


def hash_file(path, chunk_bytes=16 * 1024 * 1024):
    """Streaming hash with plain read() calls. blake3 if available, else sha256."""
    try:
        import blake3
        h = blake3.blake3(max_threads=getattr(blake3.blake3, "AUTO", 1))
        algo = "blake3"
    except Exception:
        import hashlib
        h = hashlib.sha256()
        algo = "sha256"
    buf = bytearray(chunk_bytes)
    mv = memoryview(buf)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        while True:
            n = os.readv(fd, [mv])
            if n == 0:
                break
            h.update(mv[:n])
    finally:
        os.close(fd)
    return algo, h.hexdigest()


def write_tensor_file(path, header_bytes, tensors):
    """Write header + each tensor's bytes, one tensor at a time.
    GPU tensors are brought to the CPU one by one; nothing is concatenated."""
    tmp = path + ".partial"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0), 0o644)
    try:
        _write_all(fd, memoryview(header_bytes))
        for t in tensors:
            t = t.detach()
            if t.numel() == 0:
                continue
            if t.device.type != "cpu":
                t = t.to("cpu")
            t = t.contiguous()
            _write_all(fd, cpu_memoryview(t))
            del t
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.close(fd)
    os.replace(tmp, path)


def _write_all(fd, mv):
    done = 0
    while done < len(mv):
        done += os.write(fd, mv[done:])
