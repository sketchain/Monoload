"""Shared helpers for Monoload tests. Run inside the locked ComfyUI image
(tests/docker_run.sh); every test script is a plain `python tests/xxx.py`.

ComfyUI launch args come from $COMFY_ARGS (default: --cpu --fp16-unet).
"""

import hashlib
import json
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from monoload import comfy_env  # noqa: E402

COMFY_ARGS = shlex.split(os.environ.get("COMFY_ARGS", "--cpu --fp16-unet"))
_TEST_ARGV = list(sys.argv)
ROOT, ARGS = comfy_env.setup(COMFY_ARGS)  # ComfyUI parses sys.argv here
sys.argv = _TEST_ARGV

import torch  # noqa: E402
import folder_paths  # noqa: E402
import nodes  # noqa: E402
import comfy.model_management  # noqa: E402

from monoload import hotpatch  # noqa: E402
from monoload.errors import MonoloadError, MonoloadUnsupportedError  # noqa: E402,F401

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("[{}] {}{}".format("PASS" if ok else "FAIL", name, (" — " + detail) if detail else ""), flush=True)
    return ok


def finish(extra=None):
    failed = [r for r in RESULTS if not r[1]]
    print("\n== {} checks, {} failed".format(len(RESULTS), len(failed)))
    if extra is not None:
        print("RESULT_JSON " + json.dumps(extra, default=str))
    sys.exit(1 if failed else 0)


def expect_raises(name, exc_type, fn, *substrings):
    try:
        fn()
    except exc_type as e:
        msg = str(e)
        missing = [s for s in substrings if s not in msg]
        first = msg.strip().splitlines()[0] if msg.strip() else ""
        return check(name, not missing, "missing {} in: {}".format(missing, msg[:500]) if missing else first[:220])
    except Exception as e:
        return check(name, False, "wrong exception {}: {}".format(type(e).__name__, str(e)[:500]))
    return check(name, False, "no exception raised")


# ---------------------------------------------------------------------------
# mode switching / loading / comparing
# ---------------------------------------------------------------------------

def free_all():
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    import gc
    gc.collect()


def set_runtime(enabled):
    """Switch between native ComfyUI and Monoload (all models unloaded first)."""
    free_all()
    if enabled:
        hotpatch.install()
    else:
        hotpatch.uninstall()
    assert hotpatch.is_installed() == enabled


def load_checkpoint(name):
    model, clip, _vae = nodes.CheckpointLoaderSimple().load_checkpoint(name)
    return model, clip


def load_unet(name):
    return nodes.UNETLoader().load_unet(name, "default")[0]


def load_clip(name, type_="stable_diffusion"):
    return nodes.CLIPLoader().load_clip(name, type_)[0]


def apply_loras(model, clip, loras):
    """loras: list of (name, strength_model, strength_clip)."""
    for name, sm, sc in loras:
        model, clip = nodes.LoraLoader().load_lora(model, clip, name, sm, sc)
    return model, clip


def encode(clip, text):
    return nodes.CLIPTextEncode().encode(clip, text)[0]


def sample(model, positive, negative, latent, seed=1234, steps=2, cfg=5.0, sampler="euler", scheduler="normal"):
    out = nodes.common_ksampler(model, seed, steps, cfg, sampler, scheduler, positive, negative, {"samples": latent.clone()}, denoise=1.0)
    return out[0]["samples"]


def byte_view(t):
    return t.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)


def weight_digests(module):
    """Per-tensor blake2b of every param/buffer (bytes), without keeping a copy."""
    out = {}
    for k, v in module.state_dict().items():
        out[k] = (v.dtype, tuple(v.shape), hashlib.blake2b(byte_view(v).numpy().tobytes(), digest_size=16).hexdigest())
    return out


def digests_diff(a, b):
    return [k for k in set(a) | set(b) if a.get(k) != b.get(k)]


def cond_equal(c1, c2):
    """Conditioning lists from CLIPTextEncode: tensors and pooled outputs bit-equal."""
    if len(c1) != len(c2):
        return False
    for (t1, d1), (t2, d2) in zip(c1, c2):
        if not torch.equal(byte_view(t1), byte_view(t2)):
            return False
        p1, p2 = d1.get("pooled_output"), d2.get("pooled_output")
        if (p1 is None) != (p2 is None) or (p1 is not None and not torch.equal(byte_view(p1), byte_view(p2))):
            return False
    return True


def diff_stats(a, b):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    return {"bit_exact": bool(torch.equal(a, b)), "max_abs": float(d.max()), "mean_abs": float(d.mean())}
