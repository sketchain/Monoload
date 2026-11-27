"""Group D: registers the streamed multi-frame decodes of Mochi and Cosmos 1.0 in monoload/vae_video.py at run time
(what the optional import there will do once merged; DESIGN §9.43). Import after ComfyUI is set up (common)."""

import comfy.ldm.cosmos.vae as cvae
import comfy.ldm.genmo.vae.model as genmo
from monoload import vae_cosmos, vae_mochi, vae_video


def register(on=True):
    """on: add the two chunk generators (if vae_video does not have them yet); off: leave them out (the whole-video
    layer 2 / native, as before). MULTIFRAME follows _CHUNKS."""
    for cls, mod in ((genmo.VideoVAE, vae_mochi), (cvae.CausalContinuousVideoTokenizer, vae_cosmos)):
        if on:
            vae_video._CHUNKS.setdefault(cls, mod.chunks)
        else:
            vae_video._CHUNKS.pop(cls, None)
    vae_video.MULTIFRAME = frozenset(vae_video._CHUNKS) | vae_video.WHOLE
