"""The Monoload file format: a plain safetensors file plus `monoload.*` metadata.

Only header parsing / writing lives here; bulk data movement is in
`transfer.py`. Nothing in this module ever mmaps a file.
"""

import enum
import importlib
import json
import os
import struct

import torch

from .errors import MonoloadFormatError

FORMAT_NAME = "monoload"
FORMAT_VERSION = 1
SUPPORTED_FORMAT_VERSIONS = (1,)
COMPONENT_DIFFUSION_MODEL = "diffusion_model"
SUPPORTED_COMPONENTS = (COMPONENT_DIFFUSION_MODEL,)  # future: "text_encoder", "vae"
QUANT_NONE = "none"
SUPPORTED_QUANTS = (QUANT_NONE,)  # future: "fp8_scaled", "gguf", ...

AUX_PREFIX = "__monoload_aux__."

META_FORMAT = "monoload.format"
META_FORMAT_VERSION = "monoload.format_version"
META_VERSION = "monoload.version"
META_COMPONENT = "monoload.component"
META_QUANT = "monoload.quant"
META_MODEL = "monoload.model"
META_SOURCE = "monoload.source"
META_ENV = "monoload.env"
META_CONVERT_LOG = "monoload.convert_log"

MAX_HEADER_BYTES = 256 * 1024 * 1024

_ST_TO_TORCH = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
for _name, _attr in (("U16", "uint16"), ("U32", "uint32"), ("U64", "uint64"),
                     ("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2"),
                     ("F8_E8M0", "float8_e8m0fnu")):
    if hasattr(torch, _attr):
        _ST_TO_TORCH[_name] = getattr(torch, _attr)
_TORCH_TO_ST = {v: k for k, v in _ST_TO_TORCH.items()}


def st_to_torch_dtype(name):
    return _ST_TO_TORCH.get(name)


def torch_to_st_dtype(dtype):
    try:
        return _TORCH_TO_ST[dtype]
    except KeyError:
        raise ValueError("dtype {} cannot be stored in safetensors".format(dtype))


def dtype_name(dtype):
    return None if dtype is None else str(dtype).replace("torch.", "")


def dtype_from_name(name):
    if name is None:
        return None
    dt = getattr(torch, name, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError("unknown torch dtype {}".format(name))
    return dt


# ---------------------------------------------------------------------------
# Tagged JSON: lossless encoding of the handful of non-JSON types that appear
# in ComfyUI model configs. Anything else is rejected loudly.
# ---------------------------------------------------------------------------

def class_path(cls):
    return "{}.{}".format(cls.__module__, cls.__qualname__)


def import_class_path(path):
    """Import `pkg.mod.Class` (qualname may contain dots)."""
    parts = path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        mod_name = ".".join(parts[:i])
        try:
            obj = importlib.import_module(mod_name)
        except ImportError:
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return None
        return obj
    return None


def to_tagged(obj, where="value"):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, torch.dtype):
        return {"__dtype__": dtype_name(obj)}
    if isinstance(obj, enum.Enum):
        return {"__enum__": class_path(type(obj)), "name": obj.name}
    if isinstance(obj, tuple):
        return {"__tuple__": [to_tagged(v, where) for v in obj]}
    if isinstance(obj, list):
        return [to_tagged(v, where) for v in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError("{}: non-string dict key {!r} cannot be recorded".format(where, k))
            if k.startswith("__") and k.endswith("__"):
                raise TypeError("{}: reserved key {!r}".format(where, k))
            out[k] = to_tagged(v, "{}.{}".format(where, k))
        return out
    raise TypeError("{}: value of type {} cannot be recorded in Monoload metadata".format(where, type(obj).__name__))


def from_tagged(obj):
    if isinstance(obj, list):
        return [from_tagged(v) for v in obj]
    if isinstance(obj, dict):
        if "__dtype__" in obj:
            return dtype_from_name(obj["__dtype__"])
        if "__tuple__" in obj:
            return tuple(from_tagged(v) for v in obj["__tuple__"])
        if "__enum__" in obj:
            cls = import_class_path(obj["__enum__"])
            if cls is None:
                raise ValueError("enum class {} not found".format(obj["__enum__"]))
            return cls[obj["name"]]
        return {k: from_tagged(v) for k, v in obj.items()}
    return obj


def dumps_tagged(obj, where):
    return json.dumps(to_tagged(obj, where), ensure_ascii=False, sort_keys=False)


def loads_tagged(s):
    return from_tagged(json.loads(s))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

class TensorInfo:
    __slots__ = ("name", "st_dtype", "dtype", "shape", "begin", "end")

    def __init__(self, name, st_dtype, dtype, shape, begin, end):
        self.name = name
        self.st_dtype = st_dtype
        self.dtype = dtype
        self.shape = tuple(shape)
        self.begin = begin  # relative to the data section
        self.end = end

    @property
    def nbytes(self):
        return self.end - self.begin

    def __repr__(self):
        return "TensorInfo({}, {}, {}, [{}, {}))".format(self.name, self.st_dtype, list(self.shape), self.begin, self.end)


class Header:
    """Parsed safetensors header. `tensors` is ordered by file offset."""

    def __init__(self, path, tensors, metadata, data_start, file_size, key_order):
        self.path = path
        self.tensors = tensors
        self.by_name = {t.name: t for t in tensors}
        self.metadata = metadata
        self.data_start = data_start
        self.file_size = file_size
        self.key_order = key_order  # names in the order they appear in the JSON header


def _pread_exact(fd, n, offset):
    out = bytearray(n)
    mv = memoryview(out)
    got = 0
    while got < n:
        r = os.preadv(fd, [mv[got:]], offset + got)
        if r == 0:
            raise EOFError
        got += r
    return bytes(out)


def read_header(fd, path):
    """Parse a safetensors header with pread (no mmap)."""
    file_size = os.fstat(fd).st_size
    try:
        (hlen,) = struct.unpack("<Q", _pread_exact(fd, 8, 0))
    except EOFError:
        raise MonoloadFormatError(path, "文件太短，不是 safetensors 文件")
    if hlen > MAX_HEADER_BYTES or 8 + hlen > file_size:
        raise MonoloadFormatError(path, "safetensors 文件头长度非法（{} 字节）".format(hlen))
    try:
        raw = json.loads(_pread_exact(fd, hlen, 8).decode("utf-8"))
    except Exception as e:
        raise MonoloadFormatError(path, "safetensors 文件头不是合法 JSON：{}".format(e))
    if not isinstance(raw, dict):
        raise MonoloadFormatError(path, "safetensors 文件头不是 JSON 对象")

    data_start = 8 + hlen
    metadata = raw.pop("__metadata__", None) or {}
    problems = []
    tensors = []
    key_order = []
    for name, info in raw.items():
        key_order.append(name)
        try:
            st_dtype = info["dtype"]
            shape = [int(s) for s in info["shape"]]
            begin, end = (int(x) for x in info["data_offsets"])
        except Exception:
            problems.append("{}: 条目格式错误 {!r}".format(name, info))
            continue
        dtype = st_to_torch_dtype(st_dtype)
        if dtype is None:
            problems.append("{}: 不认识的 dtype {}".format(name, st_dtype))
            continue
        numel = 1
        for s in shape:
            if s < 0:
                problems.append("{}: 负的维度 {}".format(name, shape))
            numel *= s
        itemsize = torch.empty((), dtype=dtype).element_size()
        if end < begin or end - begin != numel * itemsize:
            problems.append("{}: 数据区间 [{}, {}) 与 dtype {} / 形状 {} 不符".format(name, begin, end, st_dtype, shape))
            continue
        if data_start + end > file_size:
            problems.append("{}: 数据超出文件末尾（文件被截断？）".format(name))
            continue
        tensors.append(TensorInfo(name, st_dtype, dtype, shape, begin, end))
    if problems:
        raise MonoloadFormatError(path, "safetensors 文件头不一致", problems)
    tensors.sort(key=lambda t: (t.begin, t.end))
    for a, b in zip(tensors, tensors[1:]):
        if b.begin < a.end:
            raise MonoloadFormatError(path, "张量 {} 与 {} 的数据区间重叠".format(a.name, b.name))
    return Header(path, tensors, {str(k): str(v) for k, v in metadata.items()}, data_start, file_size, key_order)


def open_readonly(path):
    return os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))


def read_header_path(path):
    fd = open_readonly(path)
    try:
        return read_header(fd, path)
    finally:
        os.close(fd)


def build_header_bytes(entries, metadata):
    """entries: iterable of (name, dtype, shape, nbytes) in file order.
    Returns (header_bytes_including_length_prefix, list of (name, begin, end))."""
    header = {}
    layout = []
    off = 0
    for name, dtype, shape, nbytes in entries:
        if name in header or name == "__metadata__":
            raise ValueError("duplicate tensor name {}".format(name))
        header[name] = {"dtype": torch_to_st_dtype(dtype), "shape": [int(s) for s in shape], "data_offsets": [off, off + nbytes]}
        layout.append((name, off, off + nbytes))
        off += nbytes
    body = {"__metadata__": {str(k): str(v) for k, v in metadata.items()}}
    body.update(header)
    hb = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    return struct.pack("<Q", len(hb)) + hb, layout
