"""Write a random-weight SD1.5 checkpoint (real structure: UNet, CLIP-L text
encoder, VAE; fp16, about 2 GiB) and a matching plain LoRA (rank 4 on every
Linear of the UNet and the text encoder), for tests that need ComfyUI's own
loader nodes without real models (tests/test_lora_node.py) and CPU smoke runs
of the check scripts. The images mean nothing.

    python tests/make_synthetic_checkpoint.py MODELS_DIR
  -> MODELS_DIR/checkpoints/synthetic_sd15.safetensors
     MODELS_DIR/loras/synthetic_sd15_lora.safetensors
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("COMFY_ARGS", "--cpu")
import common  # noqa: E402,F401  (ComfyUI environment)
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402  writing a test file, not loading a model
import comfy.lora  # noqa: E402
import comfy.model_detection  # noqa: E402
import comfy.sd1_clip  # noqa: E402
from comfy.ldm.models.autoencoder import AutoencoderKL  # noqa: E402
from make_synthetic_vaes import init_random  # noqa: E402

out = sys.argv[1]
cfg = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': None,
       'dtype': torch.float32, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
       'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 768, 'num_heads': 8,
       'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 'use_temporal_attention': False, 'use_temporal_resblock': False}
mc = comfy.model_detection.model_config_from_unet_config(cfg)
model = mc.get_model({})
init_random(model.diffusion_model, 1)
sd = {"model.diffusion_model." + k: v.half() for k, v in model.diffusion_model.state_dict().items()}
clip = comfy.sd1_clip.SDClipModel(dtype=torch.float32)
init_random(clip.transformer, 2)
for k, v in clip.transformer.state_dict().items():
    sd["cond_stage_model.transformer." + k] = v.half() if v.is_floating_point() else v
dd = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
vae = AutoencoderKL(ddconfig=dd, embed_dim=4)
init_random(vae, 3)
for k, v in vae.state_dict().items():
    sd["first_stage_model." + k] = v.half()
os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(out, "checkpoints", "synthetic_sd15.safetensors"))
g = torch.Generator().manual_seed(4)
lora = {}
km = comfy.lora.model_lora_keys_unet(model, {})
km = comfy.lora.model_lora_keys_clip(clip, km)
states = {**{"diffusion_model." + k: v for k, v in model.diffusion_model.state_dict().items()},
          **{"transformer." + k: v for k, v in clip.transformer.state_dict().items()}}
n = 0
for lk, mk in km.items():
    if not (lk.startswith("lora_unet_") or lk.startswith("lora_te_")) or not isinstance(mk, str):
        continue
    w = states.get(mk.replace("clip_l.", "")) if mk not in states else states[mk]
    if w is None or w.ndim != 2:
        continue
    o, i = w.shape
    lora[lk + ".lora_up.weight"] = (torch.randn(o, 4, generator=g) * 0.05).half()
    lora[lk + ".lora_down.weight"] = (torch.randn(4, i, generator=g) * 0.05).half()
    lora[lk + ".alpha"] = torch.tensor(4.0)
    n += 1
os.makedirs(os.path.join(out, "loras"), exist_ok=True)
save_file(lora, os.path.join(out, "loras", "synthetic_sd15_lora.safetensors"))
print("checkpoint keys", len(sd), "lora layers", n)
