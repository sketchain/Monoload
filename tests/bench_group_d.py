"""bench_vae.py with group D's Mochi / Cosmos streaming registered (tests/group_d_reg.py), until the registration is
merged into monoload/vae_video.py. Same arguments as bench_vae.py, e.g.

    python tests/bench_group_d.py --vae mochi_vae.safetensors --res 848x480 --frames 85 --modes native,monoload
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bench_vae  # noqa: E402  (sets up ComfyUI)
import group_d_reg  # noqa: E402

group_d_reg.register(True)

if __name__ == "__main__":
    bench_vae.main()
