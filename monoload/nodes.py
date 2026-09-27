"""ComfyUI nodes: a `monoload` model folder and a UNETLoader-like loader."""

import logging
import os

import folder_paths

from .loader import load_monoload_diffusion_model

FOLDER_NAME = "monoload"
EXTENSIONS = {".safetensors"}


def register_model_folder():
    default = os.path.join(folder_paths.models_dir, FOLDER_NAME)
    entry = folder_paths.folder_names_and_paths.get(FOLDER_NAME)
    if entry is None:
        folder_paths.folder_names_and_paths[FOLDER_NAME] = ([default], set(EXTENSIONS))
    else:
        paths, exts = entry
        if default not in paths:
            paths.append(default)
        if isinstance(exts, set):
            exts.update(EXTENSIONS)
        else:
            folder_paths.folder_names_and_paths[FOLDER_NAME] = (paths, set(exts) | EXTENSIONS)
    try:
        os.makedirs(default, exist_ok=True)
    except OSError as e:
        logging.info("[Monoload] could not create {}: {}".format(default, e))


class MonoloadUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list(FOLDER_NAME),),
            },
            "optional": {
                "buffer_mb": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 64, "advanced": True,
                                      "tooltip": "每块 pinned 搬运缓冲的大小（MiB，共两块）。0 = 用环境变量 MONOLOAD_BUFFER_MB，默认 512。"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "model/loaders"
    DESCRIPTION = "加载 Monoload 转换过的扩散模型：按参数从硬盘直接读进目标设备，全程只有一份权重。"

    def load_unet(self, unet_name, buffer_mb=0):
        path = folder_paths.get_full_path_or_raise(FOLDER_NAME, unet_name)
        model_options = {}
        if buffer_mb:
            model_options["monoload_buffer_mb"] = int(buffer_mb)
        return (load_monoload_diffusion_model(path, model_options=model_options),)


NODE_CLASS_MAPPINGS = {"MonoloadUNETLoader": MonoloadUNETLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MonoloadUNETLoader": "Load Diffusion Model (Monoload)"}
