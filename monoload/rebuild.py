"""Record (at conversion) and replay (at load) everything ComfyUI infers from
the weights while building a diffusion model, without running detection."""

import copy

import torch

from . import fmt
from .errors import MonoloadFormatError, MonoloadUnsupportedError


# ---------------------------------------------------------------------------
# Model tensors: parameters + persistent buffers, in state_dict order.
# ---------------------------------------------------------------------------

class StateEntry:
    __slots__ = ("name", "module", "attr", "kind")

    def __init__(self, name, module, attr, kind):
        self.name = name
        self.module = module
        self.attr = attr
        self.kind = kind  # "param" | "buffer"

    @property
    def tensor(self):
        store = self.module._parameters if self.kind == "param" else self.module._buffers
        return store[self.attr]


def state_entries(module, prefix=""):
    """Same names and order as module.state_dict() (params, persistent buffers,
    then children), but yielding where each tensor lives so it can be replaced."""
    out = []
    _walk(module, prefix, out)
    return out


def _walk(module, prefix, out):
    for name, p in module._parameters.items():
        if p is not None:
            out.append(StateEntry(prefix + name, module, name, "param"))
    for name, b in module._buffers.items():
        if b is not None and name not in module._non_persistent_buffers_set:
            out.append(StateEntry(prefix + name, module, name, "buffer"))
    for name, child in module._modules.items():
        if child is not None:
            _walk(child, prefix + name + ".", out)


# ---------------------------------------------------------------------------
# Fingerprint / snapshots
# ---------------------------------------------------------------------------

def model_fingerprint(model):
    cfg = model.model_config
    dm = getattr(model, "diffusion_model", None)
    return {
        "model_class": fmt.class_path(type(model)),
        "diffusion_model_class": None if dm is None else fmt.class_path(type(dm)),
        "model_sampling_bases": [fmt.class_path(b) for b in type(model.model_sampling).__bases__],
        "model_type": model.model_type,
        "latent_format_class": fmt.class_path(type(model.latent_format)),
        "adm_channels": model.adm_channels,
        "concat_keys": tuple(model.concat_keys),
        "memory_usage_factor": model.memory_usage_factor,
        "manual_cast_dtype": model.manual_cast_dtype,
        "unet_dtype": cfg.unet_config.get("dtype", None),
    }


def config_snapshot(cfg):
    """Key model_config attributes, for equivalence tests and diagnostics."""
    return {
        "class": fmt.class_path(type(cfg)),
        "unet_config": dict(cfg.unet_config),
        "sampling_settings": dict(cfg.sampling_settings),
        "latent_format_class": fmt.class_path(type(cfg.latent_format)),
        "manual_cast_dtype": cfg.manual_cast_dtype,
        "supported_inference_dtypes": list(cfg.supported_inference_dtypes),
        "memory_usage_factor": cfg.memory_usage_factor,
        "optimizations": dict(cfg.optimizations),
        "custom_operations": cfg.custom_operations,
        "quant_config": cfg.quant_config,
    }


# ---------------------------------------------------------------------------
# get_model() state-dict access recording / replay
# ---------------------------------------------------------------------------

class RecordingStateDict(dict):
    """Wraps the real state dict during conversion and records what
    get_model()/model_type() look at."""

    def __init__(self, sd):
        super().__init__(sd)
        self.contains = {}
        self.got = []
        self.iterated = False
        self.mutated = []

    def __contains__(self, key):
        r = dict.__contains__(self, key)
        self.contains[key] = r
        return r

    def __getitem__(self, key):
        v = dict.__getitem__(self, key)
        if key not in self.got:
            self.got.append(key)
        return v

    def get(self, key, default=None):
        if dict.__contains__(self, key):
            return self[key]
        self.contains[key] = False
        return default

    def keys(self):
        self.iterated = True
        return dict.keys(self)

    def items(self):
        self.iterated = True
        return dict.items(self)

    def values(self):
        self.iterated = True
        return dict.values(self)

    def __iter__(self):
        self.iterated = True
        return dict.__iter__(self)

    def _mut(name):
        def f(self, *a, **k):
            self.mutated.append(name)
            return getattr(dict, name)(self, *a, **k)
        return f

    __setitem__ = _mut("__setitem__")
    __delitem__ = _mut("__delitem__")
    pop = _mut("pop")
    popitem = _mut("popitem")
    update = _mut("update")
    setdefault = _mut("setdefault")
    clear = _mut("clear")
    del _mut

    def record(self):
        key_meta = None
        if self.iterated:
            key_meta = {k: [list(v.shape), fmt.dtype_name(v.dtype)] for k, v in dict.items(self) if isinstance(v, torch.Tensor)}
        return {
            "contains": dict(self.contains),
            "values": list(self.got),
            "iterated": self.iterated,
            "key_meta": key_meta,
        }


class ReplayStateDict(dict):
    """What get_model() sees at load time: exactly the answers recorded at
    conversion. Any access that was not recorded is an error, never a guess."""

    def __init__(self, path, view, aux_values):
        super().__init__()
        self._path = path
        self._contains = view["contains"]
        self._values = aux_values
        self._iterated = view["iterated"]
        self._key_meta = view.get("key_meta") or {}
        if self._iterated:
            for k, (shape, dt) in self._key_meta.items():
                dict.__setitem__(self, k, torch.empty(shape, dtype=fmt.dtype_from_name(dt), device="meta"))
        for k, v in aux_values.items():
            dict.__setitem__(self, k, v)

    def _unrecorded(self, what):
        raise MonoloadFormatError(self._path, "模型构造时访问了转换时没有记录的 state dict 内容（{}）".format(what))

    def __contains__(self, key):
        if key in self._contains:
            return self._contains[key]
        if key in self._values:
            return True
        if self._iterated:
            return dict.__contains__(self, key)
        self._unrecorded("'{}' in state_dict".format(key))

    def __getitem__(self, key):
        if key in self._values or (self._iterated and dict.__contains__(self, key)):
            return dict.__getitem__(self, key)
        self._unrecorded("state_dict['{}']".format(key))

    def get(self, key, default=None):
        if key in self._values or (self._iterated and dict.__contains__(self, key)):
            return dict.__getitem__(self, key)
        if self._contains.get(key, None) is False:
            return default
        self._unrecorded("state_dict.get('{}')".format(key))

    def _need_iter(self):
        if not self._iterated:
            self._unrecorded("遍历 state_dict")

    def keys(self):
        self._need_iter()
        return dict.keys(self)

    def items(self):
        self._need_iter()
        return dict.items(self)

    def values(self):
        self._need_iter()
        return dict.values(self)

    def __iter__(self):
        self._need_iter()
        return dict.__iter__(self)


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------

def _fmt_dtype(d):
    return fmt.dtype_name(d) if isinstance(d, torch.dtype) or d is None else repr(d)


def rebuild_model(path, record, load_device, target_device, aux_values):
    """Construct model_config and the (uninitialised) BaseModel exactly as
    load_diffusion_model_state_dict would, from the recorded information."""
    import comfy.model_management as mm

    cls = fmt.import_class_path(record["config_class"])
    if cls is None:
        raise MonoloadFormatError(path, "找不到模型配置类 {}（ComfyUI 版本变了？）".format(record["config_class"]))
    cfg = cls(copy.deepcopy(record["detected_unet_config"]))

    if record.get("quant_config") is not None:
        raise MonoloadUnsupportedError("quantized", "v1 不支持量化模型")

    ds = record["dtype_selection"]
    if ds.get("model_options"):
        raise MonoloadUnsupportedError("model_options", "v1 只支持 UNETLoader 的 weight_dtype=default")
    supported = list(cfg.supported_inference_dtypes)
    now_dtype = mm.unet_dtype(model_params=ds["model_params"], supported_dtypes=supported, weight_dtype=ds["weight_dtype"])
    now_mc = mm.unet_manual_cast(now_dtype, load_device, cfg.supported_inference_dtypes)
    problems = []
    if list(ds["supported_dtypes"]) != supported:
        problems.append("supported_inference_dtypes: 文件 {} / 现在 {}".format(
            [_fmt_dtype(d) for d in ds["supported_dtypes"]], [_fmt_dtype(d) for d in supported]))
    if now_dtype != ds["unet_dtype"]:
        problems.append("推理 dtype: 文件按 {} 转换，当前启动参数下 ComfyUI 会选 {}".format(_fmt_dtype(ds["unet_dtype"]), _fmt_dtype(now_dtype)))
    if now_mc != ds["manual_cast_dtype"]:
        problems.append("manual_cast dtype: 文件 {} / 当前 {}（device={}）".format(_fmt_dtype(ds["manual_cast_dtype"]), _fmt_dtype(now_mc), load_device))
    if problems:
        raise MonoloadFormatError(path, "当前 ComfyUI 启动参数/设备下选出的 dtype 与转换时不同", problems)

    sid = record["set_inference_dtype"]
    cfg.set_inference_dtype(sid["dtype"], sid["manual_cast_dtype"], device=load_device)
    after = record["config_after"]
    got = {
        "unet_dtype": cfg.unet_config.get("dtype", None),
        "manual_cast_dtype": cfg.manual_cast_dtype,
        "memory_usage_factor": cfg.memory_usage_factor,
    }
    diffs = ["{}: 文件 {} / 重建 {}".format(k, after[k], got[k]) for k in got if after[k] != got[k]]
    if diffs:
        raise MonoloadFormatError(path, "set_inference_dtype 之后的模型配置与转换时不同", diffs)

    cfg.optimizations = dict(after["optimizations"])
    cfg.sampling_settings = dict(after["sampling_settings"])
    recorded_type = record["model_type"]
    # Instance-level override on Monoload's own config object only: the model
    # type was inferred from weights at conversion time and is replayed as-is.
    cfg.model_type = lambda state_dict=None, prefix="": recorded_type

    view = ReplayStateDict(path, record["get_model_view"], aux_values)
    model = cfg.get_model(view, "", device=target_device)

    fp_now = model_fingerprint(model)
    fp_rec = record["fingerprint"]
    diffs = ["{}: 文件 {} / 重建 {}".format(k, fp_rec.get(k), v) for k, v in fp_now.items() if fp_rec.get(k) != v]
    if model.model_type != recorded_type:
        diffs.append("model_type: 文件记录 {} / 重建 {}".format(recorded_type, model.model_type))
    if diffs:
        raise MonoloadFormatError(path, "重建出的模型与转换时的模型不一致", diffs)
    return cfg, model


def validate_tensor_set(path, entries, header):
    """Names, shapes and dtypes of the freshly built model must match the file exactly."""
    file_infos = {t.name: t for t in header.tensors if not t.name.startswith(fmt.AUX_PREFIX)}
    model_entries = {e.name: e for e in entries}
    problems = []
    for name in model_entries:
        if name not in file_infos:
            problems.append("模型有、文件里没有：{}".format(name))
    for name in file_infos:
        if name not in model_entries:
            problems.append("文件里有、模型没有：{}".format(name))
    for name, e in model_entries.items():
        info = file_infos.get(name)
        if info is None:
            continue
        t = e.tensor
        if tuple(t.shape) != info.shape:
            problems.append("{}: 形状 文件 {} / 模型 {}".format(name, list(info.shape), list(t.shape)))
        if t.dtype != info.dtype:
            problems.append("{}: dtype 文件 {} / 模型 {}".format(name, fmt.dtype_name(info.dtype), fmt.dtype_name(t.dtype)))
    if problems:
        raise MonoloadFormatError(path, "模型的参数/buffer 与文件中的张量不一致", problems)
    return file_infos
