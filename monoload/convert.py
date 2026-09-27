"""Convert a diffusion model into a Monoload file.

    python -m monoload.convert SOURCE.safetensors [-o OUT] [--force] [-- COMFYUI ARGS...]

Run it inside the ComfyUI container with the same ComfyUI launch arguments
used for inference (they decide the dtype). Without `-- ARGS` the arguments of
the container's main process (`python main.py ...` as PID 1) are reused.

The model is loaded once through ComfyUI's native path
(load_diffusion_model_state_dict: detection, key conversion, dtype selection
are all ComfyUI's). Everything inferred from the weights is recorded, then the
parameters and persistent buffers of the loaded model are written one tensor
at a time, in the order the Monoload loader reads them.
"""

import argparse
import contextlib
import copy
import datetime
import logging
import os
import sys
import time

from . import __version__, comfy_env


def _vm_status(field):
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class _Capture:
    def __init__(self):
        self.config_input = None
        self.config = None
        self.set_inference = None
        self.config_after_sid = None
        self.view = None
        self.view_mutated = None
        self.aux = {}
        self.get_model_prefix = None
        self.unet_dtype_call = None
        self.manual_cast_call = None
        self.log = []


class _LogCapture(logging.Handler):
    def __init__(self, sink):
        super().__init__(level=logging.WARNING)
        self.sink = sink

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        if len(msg) > 20000:
            msg = msg[:20000] + "...(truncated)"
        self.sink.append({"level": record.levelname, "message": msg})


def _instrument_config(inst, cap):
    from .rebuild import RecordingStateDict

    orig_sid = inst.set_inference_dtype
    orig_gm = inst.get_model

    def set_inference_dtype(dtype, manual_cast_dtype, **kwargs):
        cap.set_inference = {"dtype": dtype, "manual_cast_dtype": manual_cast_dtype}
        r = orig_sid(dtype, manual_cast_dtype, **kwargs)
        cap.config_after_sid = {
            "unet_dtype": inst.unet_config.get("dtype", None),
            "manual_cast_dtype": inst.manual_cast_dtype,
            "memory_usage_factor": inst.memory_usage_factor,
        }
        return r

    def get_model(state_dict, prefix="", device=None):
        rec = RecordingStateDict(state_dict)
        try:
            return orig_gm(rec, prefix, device=device)
        finally:
            cap.get_model_prefix = prefix
            cap.view = rec.record()
            cap.view_mutated = list(rec.mutated)
            cap.aux = {k: dict.__getitem__(rec, k) for k in rec.got}
            dict.clear(rec)  # drop references to the source tensors

    inst.set_inference_dtype = set_inference_dtype
    inst.get_model = get_model


@contextlib.contextmanager
def capture_native_load():
    """Observe (never alter) comfy.sd.load_diffusion_model_state_dict in this process."""
    import comfy.model_detection as model_detection
    import comfy.model_management as mm

    cap = _Capture()
    orig_mcu = model_detection.model_config_from_unet_config
    orig_ud = mm.unet_dtype
    orig_mc = mm.unet_manual_cast

    def model_config_from_unet_config(unet_config, state_dict=None, unet_key_prefix=""):
        inp = copy.deepcopy(unet_config)
        res = orig_mcu(unet_config, state_dict, unet_key_prefix)
        if res is not None:
            cap.config_input = inp
            cap.config = res
            _instrument_config(res, cap)
        return res

    def unet_dtype(*a, **k):
        r = orig_ud(*a, **k)
        if cap.unet_dtype_call is None:  # the call made by load_diffusion_model_state_dict
            cap.unet_dtype_call = (a, k, r)
        return r

    def unet_manual_cast(*a, **k):
        r = orig_mc(*a, **k)
        if cap.manual_cast_call is None:
            cap.manual_cast_call = (a, k, r)
        return r

    handler = _LogCapture(cap.log)
    root = logging.getLogger()
    model_detection.model_config_from_unet_config = model_config_from_unet_config
    mm.unet_dtype = unet_dtype
    mm.unet_manual_cast = unet_manual_cast
    root.addHandler(handler)
    try:
        yield cap
    finally:
        model_detection.model_config_from_unet_config = orig_mcu
        mm.unet_dtype = orig_ud
        mm.unet_manual_cast = orig_mc
        root.removeHandler(handler)
        if cap.config is not None:
            for attr in ("set_inference_dtype", "get_model"):
                cap.config.__dict__.pop(attr, None)


class ConvertError(RuntimeError):
    pass


MAX_AUX_BYTES = 256 * 1024 * 1024


def build_record(model, cap):
    from . import rebuild
    import torch

    cfg = model.model_config
    if cap.config is None or cap.config is not cfg:
        raise ConvertError("没能捕获到 ComfyUI 选中的模型配置（加载流程与预期不同）")
    if cfg.quant_config is not None:
        raise ConvertError("检测到量化权重（quant_config），Monoload v1 不支持量化模型")
    if cfg.custom_operations is not None:
        raise ConvertError("模型使用了 custom_operations，Monoload v1 不支持")
    if cap.view is None:
        raise ConvertError("没能捕获到 get_model() 调用")
    if cap.view_mutated:
        raise ConvertError("{}.get_model() 修改了 state dict（{}），Monoload v1 不支持".format(type(cfg).__name__, cap.view_mutated))
    if cap.get_model_prefix != "":
        raise ConvertError("get_model() 的 prefix 非空（{!r}），不符合预期".format(cap.get_model_prefix))
    if cap.unet_dtype_call is None or cap.manual_cast_call is None or cap.set_inference is None:
        raise ConvertError("没能捕获到 ComfyUI 的 dtype 选择（是否传了 dtype 覆盖？v1 只支持 weight_dtype=default）")
    aux_bytes = sum(t.numel() * t.element_size() for t in cap.aux.values())
    if aux_bytes > MAX_AUX_BYTES:
        raise ConvertError("get_model() 直接读取了 {:.1f} MiB 的源张量，超出 v1 的上限".format(aux_bytes / 2 ** 20))
    for k, t in cap.aux.items():
        if not isinstance(t, torch.Tensor):
            raise ConvertError("get_model() 读取的 {} 不是张量".format(k))

    ud_kwargs = cap.unet_dtype_call[1]
    return {
        "config_class": "{}.{}".format(type(cfg).__module__, type(cfg).__qualname__),
        "detected_unet_config": cap.config_input,
        "quant_config": None,
        "dtype_selection": {
            "model_params": int(ud_kwargs["model_params"]),
            "supported_dtypes": list(ud_kwargs["supported_dtypes"]),
            "weight_dtype": ud_kwargs["weight_dtype"],
            "unet_dtype": cap.unet_dtype_call[2],
            "manual_cast_dtype": cap.manual_cast_call[2],
            "model_options": {},
        },
        "set_inference_dtype": cap.set_inference,
        "config_after": dict(cap.config_after_sid,
                             optimizations=dict(cfg.optimizations),
                             sampling_settings=dict(cfg.sampling_settings)),
        "model_type": model.model_type,
        "get_model_view": cap.view,
        "fingerprint": rebuild.model_fingerprint(model),
    }


def convert(source, output, force=False, comfy_args=(), root=None):
    import torch
    import comfy.sd
    import comfy.model_management as mm

    from . import fmt, rebuild, transfer

    source = os.path.abspath(source)
    output = os.path.abspath(output)
    if os.path.exists(output) and not force:
        raise ConvertError("输出文件已存在：{}（加 --force 覆盖）".format(output))
    if os.path.realpath(source) == os.path.realpath(output):
        raise ConvertError("输出文件不能覆盖源文件")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    timings = {}
    t_all = time.perf_counter()

    st = os.stat(source)
    src_header = fmt.read_header_path(source)
    logging.info("[Monoload] hashing {} ({:.2f} GiB)...".format(source, st.st_size / 2 ** 30))
    t = time.perf_counter()
    hash_algo, digest = transfer.hash_file(source)
    timings["hash"] = time.perf_counter() - t
    logging.info("[Monoload] {} = {}".format(hash_algo, digest))

    # Same dict comfy.utils.load_torch_file would return under --disable-mmap,
    # but read with preadv: safetensors.safe_open mmaps the file.
    t = time.perf_counter()
    sd = transfer.read_state_dict_cpu(source, src_header)
    timings["read_source"] = time.perf_counter() - t
    src_metadata = dict(src_header.metadata) if src_header.metadata else None

    t = time.perf_counter()
    with capture_native_load() as cap:
        patcher = comfy.sd.load_diffusion_model_state_dict(sd, model_options={}, metadata=src_metadata)
    del sd
    timings["native_load"] = time.perf_counter() - t
    if patcher is None:
        raise ConvertError("ComfyUI 识别不了这个模型（UNETLoader 同样会失败）")
    model = patcher.model
    record = build_record(model, cap)

    entries = rebuild.state_entries(model)
    names = [e.name for e in entries]
    sd_keys = list(model.state_dict().keys())
    if names != sd_keys:
        raise ConvertError("模型的 state_dict() 与参数/buffer 清单不一致（自定义 _save_to_state_dict？），v1 不支持")

    missing = [m["message"] for m in cap.log if m["message"].startswith("unet missing")]
    if missing:
        logging.warning("[Monoload] 警告：ComfyUI 加载时报告了缺失的权重，这些参数保持未初始化的值（与原生行为相同）：\n  " + "\n  ".join(missing))

    tensors = []
    entries_meta = []
    for k, v in cap.aux.items():
        v = v.detach().to("cpu").contiguous()
        tensors.append(v)
        entries_meta.append((fmt.AUX_PREFIX + k, v.dtype, v.shape, v.numel() * v.element_size()))
    for e in entries:
        tt = e.tensor
        tensors.append(tt)
        entries_meta.append((e.name, tt.dtype, tt.shape, tt.numel() * tt.element_size()))

    env = comfy_env.comfyui_version_info(root) if root else {}
    env.update({
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "comfy_args": list(comfy_args),
        "load_device": str(mm.get_torch_device()),
        "offload_device": str(mm.unet_offload_device()),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    })
    metadata = {
        fmt.META_FORMAT: fmt.FORMAT_NAME,
        fmt.META_FORMAT_VERSION: str(fmt.FORMAT_VERSION),
        fmt.META_VERSION: __version__,
        fmt.META_COMPONENT: fmt.COMPONENT_DIFFUSION_MODEL,
        fmt.META_QUANT: fmt.QUANT_NONE,
        fmt.META_MODEL: fmt.dumps_tagged(record, "model"),
        fmt.META_SOURCE: fmt.dumps_tagged({
            "filename": os.path.basename(source),
            "size": st.st_size,
            "mtime": int(st.st_mtime),
            "hash_algo": hash_algo,
            "hash": digest,
            "tensor_count": len(src_header.tensors),
        }, "source"),
        fmt.META_ENV: fmt.dumps_tagged(env, "env"),
        fmt.META_CONVERT_LOG: fmt.dumps_tagged(cap.log, "convert_log"),
    }
    header_bytes, _ = fmt.build_header_bytes(entries_meta, metadata)
    t = time.perf_counter()
    with torch.no_grad():
        transfer.write_tensor_file(output, header_bytes, tensors)
    timings["write"] = time.perf_counter() - t
    total = sum(m[3] for m in entries_meta)
    timings["total"] = time.perf_counter() - t_all
    n_entries = len(entries)
    # Drop the loaded model now (not at interpreter shutdown).
    del tensors, entries
    patcher.detach(unpatch_all=False)
    del model, patcher
    return {
        "output": output,
        "tensors": n_entries,
        "aux_tensors": len(cap.aux),
        "bytes": total,
        "config_class": record["config_class"],
        "unet_dtype": str(record["dtype_selection"]["unet_dtype"]),
        "manual_cast_dtype": str(record["dtype_selection"]["manual_cast_dtype"]),
        "model_type": record["model_type"].name,
        "timings": timings,
        "vm_hwm": _vm_status("VmHWM"),
        "warnings": len(cap.log),
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    comfy_args = None
    if "--" in argv:
        i = argv.index("--")
        argv, comfy_args = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(prog="python -m monoload.convert", description=__doc__.split("\n\n")[0])
    p.add_argument("source", help="UNETLoader 能加载的扩散模型 .safetensors")
    p.add_argument("-o", "--output", help="输出文件（默认 <ComfyUI models>/monoload/<源文件名>）")
    p.add_argument("--force", action="store_true", help="覆盖已存在的输出文件")
    p.add_argument("--no-auto-args", action="store_true", help="不要自动沿用容器主进程（PID 1）的 ComfyUI 启动参数")
    a = p.parse_args(argv)

    if comfy_args is None:
        auto = None if a.no_auto_args else comfy_env.pid1_comfy_args()
        if auto is None:
            print("[Monoload] 警告：没有给出 ComfyUI 启动参数（-- ...），也没能从 PID 1 读到；按无参数启动处理。", file=sys.stderr)
            comfy_args = []
        else:
            comfy_args = auto
            print("[Monoload] 沿用容器主进程的 ComfyUI 启动参数：{}".format(" ".join(comfy_args)), file=sys.stderr)
    else:
        print("[Monoload] ComfyUI 启动参数：{}".format(" ".join(comfy_args)), file=sys.stderr)

    root, _args = comfy_env.setup(comfy_args)
    import folder_paths

    output = a.output or os.path.join(folder_paths.models_dir, "monoload", os.path.basename(a.source))
    try:
        info = convert(a.source, output, force=a.force, comfy_args=comfy_args, root=root)
    except ConvertError as e:
        print("[Monoload] 转换失败：{}".format(e), file=sys.stderr)
        return 2
    finally:
        import gc
        gc.collect()  # release the ModelPatcher before interpreter shutdown
    t = info["timings"]
    print("[Monoload] 完成：{}".format(info["output"]))
    print("  模型配置类     {}".format(info["config_class"]))
    print("  model_type     {}".format(info["model_type"]))
    print("  推理 dtype     {}   manual_cast {}".format(info["unet_dtype"], info["manual_cast_dtype"]))
    print("  张量           {} 个（另有 {} 个辅助张量），{:.3f} GiB".format(info["tensors"], info["aux_tensors"], info["bytes"] / 2 ** 30))
    print("  耗时           哈希 {:.1f}s  读源 {:.1f}s  原生加载 {:.1f}s  写出 {:.1f}s  合计 {:.1f}s".format(
        t["hash"], t["read_source"], t["native_load"], t["write"], t["total"]))
    if info["vm_hwm"]:
        print("  进程内存峰值   VmHWM {:.2f} GiB".format(info["vm_hwm"] / 2 ** 30))
    if info["warnings"]:
        print("  ComfyUI 警告   {} 条（已记录在 metadata 的 monoload.convert_log）".format(info["warnings"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
