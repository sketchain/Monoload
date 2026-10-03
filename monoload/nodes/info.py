"""Monoload Info: shows what Monoload is doing, as text in the node and as a
STRING output (monoload/info.py builds it).

All inputs are optional: nothing connected -> version, master switch and
every global default with its source; vae -> that VAE's effective decode
settings and the record of its last decode; model -> the LoRA attached to
it and how Monoload handles it. images is not read: connect a VAE Decode's
IMAGE output to make this node run after that decode. The node runs on every
prompt (never cached) and is an output node, so it runs with nothing
connected to its STRING output.

The text in the node box is drawn by web/monoload_info.js with the text
preview widget of ComfyUI's frontend (the one of the core "Preview as Text"
node), from the "text" entry of the node's UI output.
"""

from .. import info


class MonoloadInfo:
    DISPLAY_NAME = "Monoload Info"
    CATEGORY = "Monoload"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show"
    OUTPUT_NODE = True
    DESCRIPTION = ("What Monoload is doing: version, master switch and global defaults; with a VAE its decode settings and "
                   "last decode; with a MODEL its LoRA and how they are handled. Refreshed on every run.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {
            "vae": ("VAE", {"tooltip": "Show this VAE's decode settings (with sources) and its last decode."}),
            "model": ("MODEL", {"tooltip": "Show the LoRA on this model and how Monoload handles them."}),
            "images": ("IMAGE", {"tooltip": "Not read: connect the VAE Decode output so that this node runs after the decode."}),
        }}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")   # never equal to itself: run on every prompt

    def show(self, vae=None, model=None, images=None):
        text = info.report(vae=vae, model=model)
        return {"ui": {"text": [text]}, "result": (text,)}
