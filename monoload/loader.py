"""Load a Monoload diffusion model: header -> rebuild -> one allocation -> stream."""

import logging
import os
import time

import torch

import comfy.memory_management
import comfy.model_management
import comfy.model_patcher

from . import __version__, fmt, rebuild, transfer
from .errors import MonoloadFormatError, MonoloadUnsupportedError
from .patcher import MonoloadModelPatcher


def check_supported_runtime():
    if comfy.memory_management.aimdo_enabled or comfy.model_patcher.CoreModelPatcher is not comfy.model_patcher.ModelPatcher:
        raise MonoloadUnsupportedError(
            "dynamic_vram",
            "ComfyUI 开启了 DynamicVRAM（comfy-aimdo）。它按 mmap 懒加载权重、由 ModelPatcherDynamic 管理，"
            "与 Monoload 的「一次分配 + pread 搬运」互斥，v1 不支持。请用 --gpu-only（目标配置）或 --disable-dynamic-vram 启动。")


def parse_metadata(path, header):
    md = header.metadata
    problems = []
    if md.get(fmt.META_FORMAT) != fmt.FORMAT_NAME:
        raise MonoloadFormatError(path, "不是 Monoload 转换出的文件（metadata 里没有 {}={}）".format(fmt.META_FORMAT, fmt.FORMAT_NAME))
    try:
        version = int(md.get(fmt.META_FORMAT_VERSION, ""))
    except ValueError:
        version = None
    if version not in fmt.SUPPORTED_FORMAT_VERSIONS:
        raise MonoloadFormatError(path, "格式版本 {!r} 不受支持（本版本 Monoload {} 支持 {}）".format(
            md.get(fmt.META_FORMAT_VERSION), __version__, list(fmt.SUPPORTED_FORMAT_VERSIONS)))
    component = md.get(fmt.META_COMPONENT)
    if component not in fmt.SUPPORTED_COMPONENTS:
        problems.append("组件类型 {!r} 不受支持（支持 {}）".format(component, list(fmt.SUPPORTED_COMPONENTS)))
    quant = md.get(fmt.META_QUANT)
    if quant not in fmt.SUPPORTED_QUANTS:
        problems.append("量化类型 {!r} 不受支持（支持 {}）".format(quant, list(fmt.SUPPORTED_QUANTS)))
    if problems:
        raise MonoloadFormatError(path, "文件类型不受支持", problems)
    try:
        record = fmt.loads_tagged(md[fmt.META_MODEL])
        env = fmt.loads_tagged(md.get(fmt.META_ENV, "{}"))
    except Exception as e:
        raise MonoloadFormatError(path, "metadata 解析失败：{}".format(e))
    return record, env


def _warn_env(path, env):
    try:
        import comfyui_version
        now = comfyui_version.__version__
    except Exception:
        now = None
    then = env.get("comfyui_version")
    if then is not None and now is not None and then != now:
        logging.warning("[Monoload] {} 是用 ComfyUI {} 转换的，当前是 {}。结构/dtype 校验已通过，继续加载；如出现异常请重新转换。".format(
            os.path.basename(path), then, now))


def _read_aux(fd, path, header, names):
    out = {}
    for key in names:
        info = header.by_name.get(fmt.AUX_PREFIX + key)
        if info is None:
            raise MonoloadFormatError(path, "缺少构造模型所需的辅助张量 {}".format(fmt.AUX_PREFIX + key))
        t = torch.empty(info.shape, dtype=info.dtype)
        if info.nbytes:
            transfer.pread_into(fd, transfer.cpu_memoryview(t), header.data_start + info.begin, path)
        out[key] = t
    return out


def _place_on_target(entries, target):
    """Make sure every tensor to be filled already lives on the target device.
    Tensors a module created without honouring `device=` get a fresh (empty)
    allocation there; their untouched CPU placeholder is dropped."""
    replaced = {}
    moved = 0
    moved_bytes = 0
    for e in entries:
        t = e.tensor
        if t.device == target:
            continue
        new = replaced.get(id(t))
        if new is None:
            data = torch.empty(t.shape, dtype=t.dtype, device=target)
            new = torch.nn.Parameter(data, requires_grad=t.requires_grad) if e.kind == "param" else data
            replaced[id(t)] = new
            moved += 1
            moved_bytes += data.numel() * data.element_size()
        if e.kind == "param":
            e.module._parameters[e.attr] = new
        else:
            e.module._buffers[e.attr] = new
    return moved, moved_bytes


def load_monoload_diffusion_model(path, model_options={}, disable_dynamic=False):
    """Monoload counterpart of comfy.sd.load_diffusion_model. Also used as the
    ModelPatcher's cached_patcher_init, hence the signature."""
    t0 = time.perf_counter()
    check_supported_runtime()
    load_device = model_options.get("load_device", comfy.model_management.get_torch_device())
    offload_device = comfy.model_management.unet_offload_device()
    target = torch.device(offload_device)
    buf_bytes = transfer.buffer_bytes_from(model_options.get("monoload_buffer_mb"))

    fd = fmt.open_readonly(path)
    try:
        header = fmt.read_header(fd, path)
        record, env = parse_metadata(path, header)
        _warn_env(path, env)
        aux = _read_aux(fd, path, header, record["get_model_view"]["values"])
        expected_aux = {fmt.AUX_PREFIX + k for k in record["get_model_view"]["values"]}
        stray = [t.name for t in header.tensors if t.name.startswith(fmt.AUX_PREFIX) and t.name not in expected_aux]
        if stray:
            raise MonoloadFormatError(path, "文件里有未登记的辅助张量", stray)

        cfg, model = rebuild.rebuild_model(path, record, load_device, target, aux)
        del aux
        entries = rebuild.state_entries(model)
        file_infos = rebuild.validate_tensor_set(path, entries, header)
        moved, moved_bytes = _place_on_target(entries, target)
        if moved:
            logging.info("[Monoload] {} tensors ({:.1f} MiB) were not created on {} by the model constructor; allocated there instead".format(
                moved, moved_bytes / 2 ** 20, target))
        items = [(header.data_start + file_infos[e.name].begin, file_infos[e.name].nbytes, e.tensor) for e in entries]
        items.sort(key=lambda x: x[0])
        with torch.no_grad():
            stats = transfer.stream_into_tensors(fd, items, target, buf_bytes, path)
    finally:
        os.close(fd)

    patcher = MonoloadModelPatcher(model, load_device=load_device, offload_device=offload_device)
    patcher.cached_patcher_init = (load_monoload_diffusion_model, (path, model_options))
    logging.info("[Monoload] loaded {} -> {}: {}; total {:.2f}s".format(
        os.path.basename(path), target, stats, time.perf_counter() - t0))
    return patcher
