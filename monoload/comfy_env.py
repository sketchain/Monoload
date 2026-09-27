"""Bootstrap ComfyUI inside a standalone process (the converter, tests).

ComfyUI parses sys.argv when comfy.cli_args is imported, so the ComfyUI
launch arguments must be in place before any `comfy` import.
"""

import logging
import os
import sys


def find_comfyui_root():
    env = os.environ.get("COMFYUI_PATH")
    candidates = [env] if env else []
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(6):
        candidates.append(d)
        d = os.path.dirname(d)
    candidates.append("/opt/ComfyUI")
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "main.py")) and os.path.isdir(os.path.join(c, "comfy")):
            return os.path.abspath(c)
    raise RuntimeError("找不到 ComfyUI 目录；请设置环境变量 COMFYUI_PATH")


def pid1_comfy_args():
    """ComfyUI args of the container's main process (`python main.py ...`), if any."""
    try:
        with open("/proc/1/cmdline", "rb") as f:
            argv = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
    except OSError:
        return None
    for i, a in enumerate(argv):
        if os.path.basename(a) == "main.py":
            return argv[i + 1:]
    return None


def setup(comfy_args, log_level=logging.INFO):
    root = find_comfyui_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    sys.argv = [os.path.join(root, "main.py")] + list(comfy_args)
    import comfy.options
    comfy.options.enable_args_parsing()
    from comfy.cli_args import args

    # Mirror the environment tweaks main.py applies before torch is imported.
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.cuda_device)
    if args.deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import cuda_malloc
        if "rocm" in cuda_malloc.get_torch_version_noimport():
            os.environ["OCL_SET_SVM_SIZE"] = "262144"
    except Exception:
        pass

    logging.basicConfig(level=log_level, format="%(message)s")
    logging.getLogger().setLevel(log_level)
    return root, args


def comfyui_version_info(root):
    info = {"comfyui_version": None, "comfyui_commit": None}
    try:
        import comfyui_version
        info["comfyui_version"] = comfyui_version.__version__
    except Exception:
        pass
    try:
        git = os.path.join(root, ".git")
        with open(os.path.join(git, "HEAD")) as f:
            head = f.read().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            p = os.path.join(git, ref)
            if os.path.isfile(p):
                with open(p) as f:
                    info["comfyui_commit"] = f.read().strip()
            else:
                with open(os.path.join(git, "packed-refs")) as f:
                    for line in f:
                        if line.strip().endswith(ref):
                            info["comfyui_commit"] = line.split()[0]
        else:
            info["comfyui_commit"] = head
    except Exception:
        pass
    return info
