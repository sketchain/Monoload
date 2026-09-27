# Monoload

**mono（单份）+ load。** 先把扩散模型离线转换成「ComfyUI 内部最终布局」的文件，再用自定义加载节点从硬盘按参数直接读进目标设备。整个加载过程中，任何时刻内存 + 显存里属于这个模型权重的字节数都 ≤ 1 份模型 + 搬运缓冲（默认 2×512MiB）+ 少量开销。

为 Strix Halo（gfx1151，统一内存，显存走 GTT）+ ROCm 设计，目标配置是 `--disable-mmap --gpu-only`。设计细节见 [docs/DESIGN.md](docs/DESIGN.md)。

锁定的参考环境：`docker.io/kyuz0/amd-strix-halo-comfyui@sha256:384aa1fecef6a841832e0d5552949977330308d8c25e212a94f5e8dfcc061cae`（ComfyUI 0.31.0，commit `62b3c94b`，torch 2.14.0a0+rocm7.15）。

---

## 1. 原理

原生 `UNETLoader` 在 `--disable-mmap` 下：`load_torch_file` 先 mmap 文件（`safe_open`），再把每个张量复制一份到 CPU 拼成完整 dict；`load_diffusion_model_state_dict` 建好模型、`model.to(offload_device)`，再用 `load_state_dict` 逐个复制。**dict 和参数同时存在**，GPU 上还会有 mmap 页被访问的问题。打 LoRA 时 `ModelPatcher` 还会把被改动的原权重**备份**一份。

Monoload 分两步：

1. **离线转换**（`python -m monoload.convert`）：在目标容器里、用和推理完全相同的 ComfyUI 启动参数，走 ComfyUI 原生的扩散模型加载流程把模型完整加载一遍（识别、key 转换、dtype 选择都由 ComfyUI 完成），记录下所有「从权重推断出来的东西」（模型配置类、原始 `unet_config`、dtype 选择、`model_type`、`sampling_settings` 等），然后把**加载完之后模型内部的参数和 persistent buffer**，按加载时的读取顺序逐个张量写成一个 safetensors 文件。
2. **加载**（节点 `Load Diffusion Model (Monoload)`）：只读文件头和 metadata，按记录直接重建 `model_config` 和模型（不跑识别和转换），参数在目标设备上**只分配一次**，然后用 `os.preadv` 把数据直接读进去：
   * 目标是 GPU（`--gpu-only`）：两块 pinned 缓冲双缓冲流水，一块在 `preadv` 读盘，另一块在做 `non_blocking` H2D 拷贝，用 stream/event 同步。连续的小张量打包成一次读，大张量分块。
   * 目标是 CPU（普通模式 / `--cpu`）：直接 `preadv` 进参数内存，零中转。
   * 全程不 mmap，不存在完整的 CPU 副本。

**LoRA** 走运行时临时合并：Monoload 的 `ModelPatcher` 不改权重、不备份，而是把 patch 挂到每层的 `weight_function` 上（ComfyUI 的 lowvram 模式本来就用这条路），这一层算到的那一刻才用 `comfy.lora.calculate_weight` 合并出临时权重，用完即丢。数值上复刻了原生「烘焙」LoRA 的路径，结果与原生**逐位一致**。LoRA / LoCon / LoHa / LoKr 等所有 ComfyUI 支持的类型都能用，Hook LoRA 也支持；直接用原生的 `LoraLoader` / `LoraLoaderModelOnly` 即可。

## 2. 目录约定

| 位置（宿主机） | 容器内 | 用途 |
|---|---|---|
| `/models/download/`（举例，以后可挪到 HDD） | `/models/download/`（只读挂载） | 原始模型，只用来下载和转换 |
| `/models/comfy/monoload/` | `/opt/ComfyUI/models/monoload/` | 转换好的文件，ComfyUI 只从这里读扩散模型 |
| `/srv/comfy/custom_nodes/monoload/`（本仓库） | `/opt/ComfyUI/custom_nodes/monoload/` | Monoload 节点 + 转换器 |

`models/monoload/` 已经在现有的 `/models/comfy` 挂载里面，不需要单独挂载；节点加载时会自动建这个目录。

## 3. compose 里要加的东西

```yaml
services:
  comfyui:
    # ...（原有配置不变）
    environment:
      - TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
      - TORCH_BLAS_PREFER_HIPBLASLT=1
      - MONOLOAD_BUFFER_MB=512          # 可选：每块 pinned 搬运缓冲的大小，共两块
    volumes:
      - /models/comfy:/opt/ComfyUI/models
      - /pictures/comfy:/opt/ComfyUI/output
      - /srv/comfy/user:/opt/ComfyUI/user
      - /srv/comfy/custom_nodes/monoload:/opt/ComfyUI/custom_nodes/monoload:ro   # 新增：Monoload
      - /models/download:/models/download:ro                                     # 新增：原始模型（给转换器读）
    command: >
      python main.py --listen 0.0.0.0 --port 8188
      --disable-mmap --gpu-only --bf16-vae
```

仓库以只读方式挂载即可（`__pycache__` 写不进去不影响运行）。

## 4. 转换

```bash
# 1) 先让 ComfyUI 卸载已加载的模型，给转换腾出 GTT（或者直接停掉服务）
curl -X POST http://127.0.0.1:8188/free -H 'Content-Type: application/json' \
     -d '{"unload_models": true, "free_memory": true}'

# 2) 转换（在运行中的容器里执行）
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui \
  python -m monoload.convert /models/download/krea2_turbo_bf16.safetensors
# 输出：/opt/ComfyUI/models/monoload/krea2_turbo_bf16.safetensors（宿主机 /models/comfy/monoload/）
```

* **ComfyUI 启动参数**：不写 `-- 参数` 时，转换器自动沿用容器主进程（PID 1，即 `python main.py ...`）的参数，并在开头打印出来。也可以显式给：`python -m monoload.convert SRC -- --disable-mmap --gpu-only --bf16-vae`。参数决定 dtype，**必须和推理时一致**。
* 其他选项：`-o 输出路径`、`--force`（覆盖已有文件）、`--no-auto-args`。
* 输入可以是任何 `UNETLoader` 能加载的扩散模型文件（包括整合包 checkpoint，UNETLoader 会从中取出扩散模型）。
* 转换结束会打印：配置类、`model_type`、推理 dtype / manual_cast、张量数和大小、各阶段耗时、**进程内存峰值（VmHWM）**。

### 转换时的内存

转换器按原生 CPU 路径加载：CPU 上是一份完整的源 dict，`--gpu-only` 下模型参数在 GTT 里（GTT 不计入 LXC 的 cgroup）。

实测（CPU 模式，模型参数也在 CPU 上）：

| 模型 | 源文件 | 转换进程 VmHWM | 其中：源 dict + 基础开销 |
|---|---|---|---|
| SD1.5 fp16 整合包 | 1.99 GiB | 4.45 GiB | ≈ 2.85 GiB |
| Anima base（bf16） | 3.89 GiB | 8.66 GiB | ≈ 4.77 GiB（= 源 + 0.88 GiB） |

`--gpu-only` 下模型那一份在 GTT，所以**转换进程的 CPU 峰值 ≈ 源文件大小 + 约 1 GiB**（ROCm 运行时本身再加 1～2 GiB）。

**Krea 2 Turbo bf16（26G）**：CPU 峰值约 **27～29 GiB**（Krea2 没有额外的 key 转换，不会产生额外副本）。**转换时建议把 CT 700 内存临时调到 32G**，GTT 需要留出约 26G 空闲（所以先卸载服务里的模型）。转换完可以调回去。

另外，源文件会被完整读两遍（一遍算 blake3 哈希，一遍加载），放在 HDD 上时会慢一些。

## 5. 使用

* 节点：`model/loaders` → **Load Diffusion Model (Monoload)**（`MonoloadUNETLoader`），列出 `models/monoload/` 里的文件，输出 `MODEL`，用法和 `UNETLoader` 一样。
* 高级参数 `buffer_mb`：每块搬运缓冲的大小（MiB），0 表示用环境变量 `MONOLOAD_BUFFER_MB`（默认 512）。
* LoRA：直接接原生 `LoraLoaderModelOnly` / `LoraLoader`；Hook LoRA（`Create Hook LoRA` 等）也可以。
* 文本编码器和 VAE：v1 继续用原生 `CLIPLoader` / `VAELoader`（理由见 DESIGN.md §5.1）。注意原生 `CLIPLoader` 加载 `qwen3vl_4b_bf16` 时，CPU 上会短暂有一份完整 dict（约 8G），CT 内存要按这个量留余量。

日志里每次加载会有一行：

```
[Monoload] loaded krea2_turbo_bf16.safetensors -> cuda:0: 24.xx GiB in N s (X GiB/s, path=staged, K reads, 2 x 512 MiB pinned buffers); total N s
```

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MONOLOAD_BUFFER_MB` | 512 | 每块 pinned 缓冲大小（共两块） |
| `MONOLOAD_STAGING` | `auto` | `always`：CPU 目标也走双缓冲路径（测试用） |

## 6. 什么情况下需要重新转换

加载器会**严格校验**，下面任何一项不满足都会报错，并提示用当前参数重新转换：

* 格式版本、组件类型、量化类型不受支持（Monoload 升级了格式版本时）。
* 记录的模型配置类找不到（ComfyUI 升级删改了模型类）。
* **dtype 选择变了**：用当前的启动参数和设备重算 ComfyUI 会选的推理 dtype / manual_cast，与文件不同就报错。例如换了 `--bf16-unet`/`--fp16-unet`/`--fp32-unet`/`--force-fp16`、从 GPU 换到 `--cpu`、换了 GPU。
* 重建出的模型与转换时不一致（模型类、`model_type`、`latent_format`、`memory_usage_factor` 等）。
* 模型的参数 + persistent buffer 与文件里的张量在名字集合、形状、dtype 上有任何差异（ComfyUI 升级改了网络结构或 key 转换时）。
* 文件被截断、文件头损坏、metadata 损坏。

**只有 ComfyUI 版本号和转换时不同**时只给警告、照常加载。建议升级 ComfyUI（换镜像 digest）后把模型都重新转换一遍。源模型更新了（比如微调出了新版本）当然也要重新转换；metadata 里记录了源文件的大小和 blake3，可以核对。

## 7. 限制（v1）

* 只转换扩散模型。文本编码器、VAE 继续走原生加载；`CheckpointLoaderSimple` 这类整合包加载器不走 Monoload（但可以把整合包当作转换器的输入）。
* 不支持量化格式（fp8 scaled、GGUF 等）、`UNETLoader` 的 fp8 `weight_dtype` 选项、`custom_operations`。转换器遇到会直接拒绝。
* 不支持 DynamicVRAM（comfy-aimdo）：检测到开启会明确报错。`--gpu-only` 会关掉它。
* LoRA 以下三种情况**报错，绝不退回改权重 + 备份**，报错信息写明情况和 key：
  * `force_patch_weights`：要求把 LoRA 烘焙进权重，常见于 `ModelSave` / `CheckpointSave` / 保存合并后的模型；
  * patch 修改的参数不属于 `comfy.ops` 层（没有运行时合并路径）；
  * patch 会改变权重形状。
* LoRA 每一步、每一层都要重新合并一次，代价是每层一次 LoRA 矩阵乘；换来的是没有备份、不改权重。
* LoRA 文件由原生 `LoraLoader` 读取（`load_torch_file`，内部 `safe_open` 会短暂 mmap LoRA 文件再复制）；LoRA 文件本身常驻内存，这是允许的。
* 非 `--gpu-only` 时，模型待机时和原生一样放在 CPU（`unet_offload_device()`），采样前由 ComfyUI 移到 GPU。
* 不用 O_DIRECT（ZFS `primarycache=metadata` 下 `pread` 本来就不留缓存）。
* 本仓库的测试都在无 GPU 的机器上用 `--cpu` 完成；GPU 上的双缓冲流水只用假 stream 做过逻辑测试，**需要按第 9 节在 CT 700 上真机验收**。

## 8. 测试

所有测试都在锁定 digest 的镜像里用 `--cpu` 跑，测试脚本是普通 Python（镜像里没有 pytest）。

```bash
# 需要的文件（放在 $MODELS 下）：
#   diffusion_models/v1-5-pruned-emaonly-fp16.safetensors
#       https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors
#   loras/rubber_duck.safetensors        (Norod78/SD15-Rubber-Duck-LoRA, 普通 LoRA)
#   loras/lycoris_annalise.safetensors   (pmczip/SD1.5_LyCORIS_Models, LyCORIS LoCon, 含卷积层)
#   loras/synthetic_{lokr,loha}_sd15.safetensors
#       由 tests/make_synthetic_loras.py 生成（HF 上找不到 SD1.5 的 LoKr，按真实 LoRA 的层名/形状合成）
#   monoload/  （空目录，可写）
MODELS=/path/to/models tests/run_all.sh
```

| 脚本 | 内容 |
|---|---|
| `tests/test_transfer_unit.py` | 文件读写、直读 / 双缓冲 / CUDA 流水分支（假 stream）、极小缓冲分块、非连续目标 |
| `tests/test_equivalence.py` | 原生 `UNETLoader` vs Monoload：全部参数和 buffer 逐字节比较、model_config 关键属性、模型指纹、固定种子采样逐位比较、`cached_patcher_init` 重建 |
| `tests/test_memory.py` | 加载峰值（`/proc/self/statm` 2ms 采样 + `VmHWM`）、`/proc/self/maps` 检查是否映射模型文件、LoRA 采样峰值和备份；有 GPU 时同时采样 GTT 和 cgroup |
| `tests/test_validation.py` | 篡改文件（稀疏文件，不占磁盘）：改名、改形状、删张量、多张量、改 dtype、改格式版本、去掉格式标记、组件/量化类型、类名、dtype 记录、model_type、unet_config、metadata 损坏、截断；非 Monoload 文件；ComfyUI 版本不同只警告；DynamicVRAM 报错 |
| `tests/test_lora.py` | 4 种 LoRA 与原生逐位比较、无备份、LoRA 生效期间和撤掉后权重与文件逐字节一致；带 keyframe 的 Hook LoRA；三种拒绝场景 |

### 实测结果（CPU，无 GPU）

**正确性**

| 模型 | 参数和 buffer | model_config / 指纹 | 采样（固定种子） |
|---|---|---|---|
| SD1.5，`--fp16-unet` | 688 个逐字节一致 | 一致（SD15 / EPS / fp16，manual_cast fp32） | 4 步，逐位一致（max_abs = 0） |
| SD1.5，`--bf16-unet` | 688 个逐字节一致 | 一致 | 4 步，逐位一致 |
| Anima base，`--bf16-unet` | 686 个逐字节一致 | 一致（Anima / FLOW / bf16，shift 3.0） | 3 步，逐位一致 |
| SD1.5，双缓冲 + 4 MiB 缓冲（大张量被拆成多块） | 688 个逐字节一致 | 一致 | — |

**加载内存（进程 RSS 峰值增量）**

| 模型 | 原生 UNETLoader | Monoload 直读 | Monoload 双缓冲 2×512MiB |
|---|---|---|---|
| SD1.5 UNet（1.60 GiB） | 3.95 GiB（2.46×），**映射了模型文件** | 1.58 GiB（0.99×），无映射 | 2.58 GiB（模型 + 1 GiB 缓冲），无映射 |
| Anima（3.89 GiB） | 7.78 GiB（2.00×），**映射了模型文件** | 3.88 GiB（1.00×），无映射 | 4.88 GiB（模型 + 1 GiB 缓冲），无映射 |

原生 SD1.5 的 2.46× 里还包括整合包里的 CLIP/VAE（dict 里是整个文件）。

**LoRA（SD1.5，fp16）**

| LoRA | 被 patch 的 key | 与原生 LoRA 的差异 | Monoload 备份 | 原生备份 |
|---|---|---|---|---|
| Rubber Duck（LoRA） | 192 | 逐位一致 | 0 | 192 项 |
| Annalise（LyCORIS LoCon） | 278 | 逐位一致 | 0 | 278 项，1.60 GiB |
| 合成 LoKr | 160 | 逐位一致 | 0 | 160 项 |
| 合成 LoHa | 160 | 逐位一致 | 0 | 160 项 |
| Hook LoRA（Rubber Duck，keyframe 1.0 → 0.4） | — | 逐位一致 | `hook_backup` / `cached_hook_patches` 均为 0 | — |

* LoRA 生效期间以及撤掉之后，模型的 688 个张量都与转换文件逐字节一致；撤掉 LoRA 后的采样结果与打 LoRA 之前逐位一致。
* 采样内存：Annalise LoRA（文件 162 MiB）下，Monoload 采样峰值比不打 LoRA 高 0.42 GiB。这部分是常驻的 LoRA 文件加上当前层（最大一层 fp32 约 110 MiB）的临时合并量。原生则多出 1.60 GiB 常驻备份。
* 三种拒绝场景（`force_patch_weights`、非 comfy.ops 参数、改变形状）都报出了明确的错误和 key。

## 9. 真机验收清单（CT 700）

以下命令都在 PVE 宿主机或 CT 700 里执行；`comfyui` 是容器名。准备工作：

```bash
# 在 CT 700 里：把仓库放到 /srv/comfy/custom_nodes/monoload，按第 3 节改 compose，重建容器
docker compose up -d --force-recreate comfyui
docker logs comfyui 2>&1 | grep -i monoload      # 应该看到 custom_nodes/monoload 被加载，没有报错

# 另开一个终端做内存监视（容器内看 GTT 和容器 cgroup；Ctrl-C 结束）
docker exec comfyui sh /opt/ComfyUI/custom_nodes/monoload/tools/watch_mem.sh
# 在 PVE 宿主机上看 CT 700 的 cgroup（cgroup v2）：
CGROUP=/sys/fs/cgroup/lxc/700 sh /path/to/monoload/tools/watch_mem.sh
cat /sys/fs/cgroup/lxc/700/memory.peak
```

`watch_mem.sh` 每 0.5 秒打印一次 GTT 占用（所有 `/sys/class/drm/card*/device/mem_info_gtt_used` 之和）、cgroup 的 `memory.current`，以及二者相对启动时的峰值增量。

### 9.1 转换

```bash
curl -X POST http://127.0.0.1:8188/free -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}'
# CT 700 内存临时调到 32G（Krea 2 需要，见第 4 节），然后：
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python -m monoload.convert /models/download/krea2_turbo_bf16.safetensors
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python -m monoload.convert /models/download/<Anima 微调>.safetensors
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python -m monoload.convert /models/download/<LucidDreamer Z>.safetensors
```

**要看的数**：开头打印的 ComfyUI 参数必须和 compose 的 `command` 一致；「推理 dtype」应为 `torch.bfloat16`、manual_cast 为 `None`；记下「进程内存峰值 VmHWM」和 `memory.peak`。

**通过标准**：转换成功；VmHWM ≈ 源文件大小 + 1～3 GiB；`dmesg | grep -iE "oom|killed process"` 没有新记录。

### 9.2 加载：GTT、容器内存、耗时

1. 重启容器（或 `/free` 卸载所有模型），开着 `watch_mem.sh`。
2. 在 UI 里用 **Load Diffusion Model (Monoload)** 选 `krea2_turbo_bf16.safetensors`，加上原生 `CLIPLoader`（`qwen3vl_4b_bf16`，type `krea2`）、`VAELoader`（`qwen_image_vae`），KSampler 8 步、CFG 1、euler + simple，出图。
3. 记录：
   * **GTT**：加载 Monoload 节点前后的 GTT 读数，以及 `watch_mem.sh` 的 GTT 峰值增量。
   * **容器内存**：`watch_mem.sh` 的 cgroup 峰值增量，宿主机上 `memory.peak`，或者 `docker stats comfyui`。
   * **耗时**：`docker logs comfyui 2>&1 | grep "\[Monoload\] loaded"` 那一行（GiB、秒、GiB/s、`path=staged`、`pinned buffers`）。
   * `dmesg | grep -iE "oom|killed process|amdgpu.*(fault|timeout)"`。

**通过标准**：
* 扩散模型加载阶段（看 Monoload 那一行日志前后），GTT 峰值增量 ≈ 模型大小（Krea 2 约 24.x GiB），**不超过模型大小 + 1 GiB**，加载后稳定在 ≈ 模型大小。
* 这一阶段容器 CPU 内存峰值增量**在 1～2 GiB 量级**（2×512MiB pinned 缓冲 + 少量开销），远低于模型大小；加载结束后回落，pinned 缓冲会被释放。整个工作流的峰值里还包含原生 `CLIPLoader` 加载文本编码器的 dict（约 8G），它出现在 TE 加载阶段，分开看。
* 日志里是 `path=staged … pinned buffers`；没有 OOM，没有 amdgpu fault，没有卡死。

也可以直接用测试脚本（会新起一个进程，和服务共用 GTT，先 `/free`）：

```bash
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui env COMFY_ARGS="--disable-mmap --gpu-only --bf16-vae" \
  python tests/test_memory.py --mode monoload --name krea2_turbo_bf16.safetensors
```

它会打印进程 RSS 峰值增量、GTT 峰值增量、cgroup 峰值增量，并检查：没有映射模型文件；RSS 峰值 ≤ 2 块缓冲 + 1 GiB；GTT 峰值 ≤ 模型 + 1 GiB。

### 9.3 正确性：Krea 2 原生 vs Monoload

1. 用原生 `UNETLoader`（`diffusion_models/krea2_turbo_bf16.safetensors`）出一张图。**原生加载会在 CPU 上拼完整 dict（26G），这一步要把 CT 内存临时调到 ≥ 32G。**
2. 同一个工作流、同一个种子，只把加载器换成 Monoload，再出一张。
3. 对比：`docker exec comfyui python /opt/ComfyUI/custom_nodes/monoload/tools/compare_images.py /opt/ComfyUI/output/A.png /opt/ComfyUI/output/B.png`

**通过标准**：输出 `identical`。如果不完全一致，用原生加载器同种子再出一张 A2，比较 A 和 A2：GPU 上某些 kernel 本身不确定时，要求 A 与 B 的差异不大于 A 与 A2 的差异。

较小的两个模型可以做逐字节的权重比较（同一进程里同时放原生和 Monoload 两份，需要约 2 倍模型大小的 GTT，加上 1 份模型大小的 CPU 内存）：

```bash
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui env COMFY_ARGS="--disable-mmap --gpu-only --bf16-vae" \
  python tests/test_equivalence.py --family anima --source <Anima 微调>.safetensors --converted <Anima 微调>.safetensors
```

LucidDreamer Z 用 `--family zimage`（假条件的维度从模型配置里取；这个 family 在开发机上没跑过，因为磁盘放不下 Z-Image。如果采样那一步因为条件格式报错，加 `--no-sample`，只比较权重和配置）。

通过标准：`all N params+buffers byte-identical`、`model_config key attributes equal`、`model fingerprint equal` 都是 PASS。采样那一项在 GPU 上如果不是逐位一致，看打印的 max_abs，并按 9.3 的方法和「原生跑两次」的差异比较。显存不够同时放两份时加 `--no-sample` 或分开跑。

### 9.4 LoRA

1. Krea 2（Monoload）+ `LoraLoaderModelOnly`（任选一个 Krea 2 LoRA）出图，开着 `watch_mem.sh`。
2. 同一工作流去掉 LoRA 再出一张，确认 LoRA 生效（两图不同）；再加回 LoRA，出图应与第 1 张一致。

**通过标准**：
* 挂 LoRA 后 GTT 只增加 **LoRA 文件大小 + 采样时的临时量**（几百 MiB 量级），**不会出现一份被 patch 层大小的备份**。对照：原生 `UNETLoader` + 同一个 LoRA 会多出与被 patch 权重等大的一份（若干 GiB）。
* 日志里没有 Monoload 的 `不支持` / `内部错误`。
* 可选：用原生 `UNETLoader` + 同一个 LoRA、同种子出图，与 Monoload + LoRA 的结果用 `compare_images.py` 比较，应为 `identical`（标准同 9.3）。

### 9.5 验收用模型汇总

| 模型 | 转换 | 9.2 内存/耗时 | 9.3 正确性 | 9.4 LoRA |
|---|---|---|---|---|
| Krea 2 Turbo bf16 | ✓ | ✓ | 出图对比 | ✓ |
| Anima 2.9B 微调 | ✓ | ✓（`test_memory.py`） | `test_equivalence.py --family anima` | 可选 |
| LucidDreamer Z | ✓ | ✓（`test_memory.py`） | `test_equivalence.py --family zimage` | 可选 |

## 10. 仓库结构

```
__init__.py              ComfyUI 入口：注册 monoload 模型目录和节点
monoload/
  fmt.py                 文件格式：safetensors 头读写（pread）、metadata、带标签的 JSON
  transfer.py            搬运：直读、pinned 双缓冲流水、逐张量写文件、哈希
  rebuild.py             记录/重放重建模型所需信息，严格校验
  convert.py             转换器 CLI（python -m monoload.convert）
  loader.py              加载函数 load_monoload_diffusion_model（也是 cached_patcher_init）
  patcher.py             MonoloadModelPatcher：LoRA / Hook LoRA 运行时临时合并
  nodes.py               MonoloadUNETLoader 节点
  comfy_env.py           在独立进程里按指定参数启动 ComfyUI 环境
tests/                   测试（见第 8 节）
tools/                   真机验收用的小工具
docs/DESIGN.md           设计说明
```
