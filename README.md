# Monoload

**ComfyUI 插件：打 LoRA 时不改原权重、不做任何备份，每一层在计算的那一刻才临时合并 LoRA，用完即丢。** 装上后对所有模型生效（`UNETLoader`、`CheckpointLoaderSimple`、`CLIPLoader` 加载出的模型都算），原生的 `LoraLoader` / `LoraLoaderModelOnly` / Hook LoRA 节点照常使用，工作流不用改。结果与原生 ComfyUI **逐位一致**。

为 Strix Halo（gfx1151，统一内存，显存走 GTT）+ ROCm、`--gpu-only` 的配置设计。在这种配置下，原生 ComfyUI 打 LoRA 时会把被改动的原权重在 GTT 里再备份一份，LoRA 改到的权重越多，备份就越大。

设计细节见 [docs/DESIGN.md](docs/DESIGN.md)。v1（离线转换 + pread 直读）已经废弃，代码保留在 tag `v1-converter`（远端归档分支 `archive/v1-converter`）。

参考环境：`docker.io/kyuz0/amd-strix-halo-comfyui@sha256:384aa1fecef6a841832e0d5552949977330308d8c25e212a94f5e8dfcc061cae`（ComfyUI 0.31.0，commit `62b3c94b`）。

---

## 1. 原理（一段话版）

原生 ComfyUI 在加载模型时把 LoRA「合并」进权重：先把原权重备份，再原地写入合并结果；换 LoRA 组合时写回备份，再重新合并。ComfyUI 在 lowvram 模式下本来就有另一条路：`comfy.ops` 的层只要挂了 `weight_function`，`forward` 时就会先拷一份临时权重，调用这些函数，算完就丢。Monoload 在 `ModelPatcher` 类上替换了几个方法，让「本该合并进权重」的 patch 改为挂成这样的 weight function，数值上复刻原生合并的每一步，所以结果逐位一致。换 LoRA 组合只是换 patch，不再有「写回备份 → 重新合并」。详见 DESIGN.md §2–§4。

## 2. 推荐的 compose

```yaml
services:
  comfyui:
    image: docker.io/kyuz0/amd-strix-halo-comfyui@sha256:384aa1fecef6a841832e0d5552949977330308d8c25e212a94f5e8dfcc061cae
    container_name: comfyui
    restart: unless-stopped
    devices:
      - /dev/kfd
      - /dev/dri
    security_opt:
      - seccomp:unconfined
    shm_size: 8g
    environment:
      - TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
      - TORCH_BLAS_PREFER_HIPBLASLT=1
      - GPU_PINNED_MIN_XFER_SIZE=65536        # 新增，见下
      # - MONOLOAD_DISABLE=1                  # 需要完全原生的行为时打开
    volumes:
      - /models/comfy:/opt/ComfyUI/models
      - /pictures/comfy:/opt/ComfyUI/output
      - /srv/comfy/user:/opt/ComfyUI/user
      - /srv/comfy/custom_nodes/monoload:/opt/ComfyUI/custom_nodes/monoload:ro   # 新增：Monoload
    ports:
      - "8188:8188"
    working_dir: /opt/ComfyUI
    command: >
      python main.py --listen 0.0.0.0 --port 8188
      --gpu-only --bf16-vae
```

**为什么去掉 `--disable-mmap`、加上 `GPU_PINNED_MIN_XFER_SIZE=65536`：**

* 之前用 mmap 加载权重会慢约 300 倍、还会随机卡死（ROCm issue #6530）。根因在 ROCm 的 rocclr：H2D 拷贝超过 1 MiB 时，它会选择「临时 pin 源内存」，在 APU 上这一步要经过 KFD HMM，逐页处理。上游 **ROCm/clr `3ccb59f`（2026-09-23）** 已经修复：在统一内存设备上不再做这一步 pin。
* 修复进入正式版 ROCm 之前，设 `GPU_PINNED_MIN_XFER_SIZE=65536` 提高 rocclr 走「临时 pin」这条路的门槛，效果与该修复等价。镜像换成带修复的 ROCm 之后可以去掉这个变量。
* 有了它，原生加载不开 `--disable-mmap` 就够快了。`--disable-mmap` 反而会让 ComfyUI 先把整个模型复制成一份 CPU 上的 dict（计入 LXC 的 cgroup），然后再拷进 GTT，所以去掉。
* mmap 读进来的页面在页缓存里，内存紧张时内核会自动回收。

Monoload 本身和 mmap 无关：它只接管 LoRA 怎么打，不改模型怎么加载。

## 3. 安装

```bash
# CT 700 里
git clone https://github.com/sketchain/Monoload /srv/comfy/custom_nodes/monoload
# 按第 2 节修改 compose，然后
docker compose up -d --force-recreate comfyui
docker logs comfyui 2>&1 | grep -i monoload
# 应看到：[Monoload] runtime LoRA merge installed on ModelPatcher (no in-place LoRA, no weight backups)
```

没有额外依赖，不注册新节点（节点列表里不会出现 Monoload）。以只读方式挂载即可。

## 4. 开关

| 设置 | 效果 |
|---|---|
| 默认（装上即生效） | 所有 `ModelPatcher` 走运行时合并 |
| `MONOLOAD_DISABLE=1`（也接受 `true`/`yes`/`on`） | 插件不做任何事，行为与原生完全一致；日志里是 `MONOLOAD_DISABLE is set: ... NOT installed` |

改完要重启容器。

## 5. 支持的范围

* **加载入口**：`UNETLoader`、`CheckpointLoaderSimple`（MODEL 和 CLIP）、`CLIPLoader` / `DualCLIPLoader` 等（文本编码器的 `CLIP.patcher`）；凡是 ComfyUI 原生的 `ModelPatcher` 都覆盖。
* **LoRA 节点**：`LoraLoader`（MODEL + CLIP 强度）、`LoraLoaderModelOnly`、多个 LoRA 叠加、Hook LoRA（`Create Hook LoRA`、`Set CLIP Hooks`、条件上的 hooks、keyframe 强度调度）。
* **LoRA 类型**：LoRA、LoCon、LoHa、LoKr，以及 `comfy.lora.calculate_weight` 支持的其他类型（GLoRA、OFT、BOFT、diff、set…）。
* **加载模式**：`--gpu-only` / `--highvram`（全量加载）是目标配置；普通模式下的部分加载也兼容，这时被卸载的层原生本来就用 `LowVramPatch`，Monoload 不去碰。

## 6. 会直接报错的情况（绝不退回到「改权重 + 备份」）

报错信息写明是哪种情况和对应的 key，形如 `[Monoload] 不支持（<kind>） key=<key>: 说明`。

| kind | 情况 | 怎么办 |
|---|---|---|
| `dynamic_vram` | 开了 DynamicVRAM（comfy-aimdo）的同时打 LoRA | 用 `--gpu-only`（目标配置），或 `MONOLOAD_DISABLE=1` |
| `force_patch_weights` | 有节点要求把 LoRA 合并进权重：`ModelSave`、`CheckpointSave`、保存合并后的模型等 | 做这类操作时设 `MONOLOAD_DISABLE=1` 重启 |
| `lora_non_comfy_ops_param` | LoRA 改到的参数不属于 `comfy.ops` 层（没有运行时合并路径） | 设 `MONOLOAD_DISABLE=1` |
| `lora_shape_change` | patch 会改变权重形状 | 同上 |
| `lora_quantized_param` | LoRA 改到的是量化参数（fp8 scaled 等） | v2 暂不支持，设 `MONOLOAD_DISABLE=1` |

## 7. 限制

* **每一步都有合并开销。** 原生是「加载时合并一次」，Monoload 是「每步、每个被 patch 的层都合并一次」。没被 patch 的层完全不受影响。开销和可选的放宽方案见 DESIGN.md §5，**GPU 上的实际数字请用第 9 节的基准脚本测**。
* 自己实现了 `patch_weight_to_device` 的第三方 `ModelPatcher` 子类（例如 ComfyUI-GGUF）不归 Monoload 管，保持它们自己的行为；日志里会对这个类打一次警告。
* LoRA 文件读进来后常驻 CPU 内存（原生也是这样）。另外，被用到的 LoRA 张量会在计算设备上缓存一份（只缓存比权重小的低秩因子等，模型合并带进来的整层权重不缓存），避免每步每层都做一次 H2D，模型卸载时释放。
* 模型合并节点（`ModelMergeSimple` 等）同样走运行时合并，每一步都会重新混合一遍，开销比 LoRA 大；需要反复使用合并结果时，建议先用 `MONOLOAD_DISABLE=1` 保存成新模型。
* 本仓库的测试都在无 GPU 的机器上用 `--cpu` 跑；GPU 上的一致性和性能要按第 9 节在 CT 700 上确认。

## 8. 测试（CPU，锁定镜像）

```bash
# $MODELS 下需要：
#   checkpoints/v1-5-pruned-emaonly-fp16.safetensors
#       https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors
#   diffusion_models/v1-5-pruned-emaonly-fp16.safetensors   （同一个文件，硬链接即可；UNETLoader 会从中取出 UNet）
#   text_encoders/clip_l.safetensors
#       https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
#   loras/rubber_duck.safetensors          Norod78/SD15-Rubber-Duck-LoRA（LoRA，UNet + TE）
#   loras/lycoris_annalise.safetensors     pmczip/SD1.5_LyCORIS_Models（LyCORIS LoCon，含卷积层，UNet + TE）
#   loras/synthetic_{lokr,loha}_sd15.safetensors
#       tests/make_synthetic_loras.py 生成（HF 上找不到 SD1.5 的 LoKr，按真实 LoRA 的层名和形状合成，UNet + TE）
MODELS=/path/to/models tests/run_all.sh
```

| 脚本 | 内容 |
|---|---|
| `tests/test_entry.py` | 按 ComfyUI 的方式加载插件：默认会替换 `ModelPatcher` 的方法（`CoreModelPatcher` 也覆盖到）；`MONOLOAD_DISABLE=1` 时一个方法都不动 |
| `tests/test_dtype_paths.py` | 参数 dtype × 计算 dtype × lora dtype（含 gfx1151 上的 fp16）× {基础 LoRA、hook、两者都有}，共 54 种组合，逐位对比原生合并的数值 |
| `tests/test_lora_hot.py` | 两条管线：`CheckpointLoaderSimple`；`UNETLoader` + `CLIPLoader`。同一串 LoRA 组合先用原生跑、再装上 Monoload 连续切换着跑，比较 TE 输出和采样结果；加上 Hook LoRA 和报错场景 |

**实测结果**（`--cpu --fp16-unet`，SD1.5，每个组合采样 2 步，hook 3 步）：

* 两条管线各切换 8 次组合，顺序是 duck → locon → duck+locon → duck → none → lokr → loha → duck+locon：
  * 文本编码器输出（cond 和 pooled）、采样得到的 latent 全部与原生**逐位一致**（max_abs = 0）；
  * MODEL 和 CLIP 的 `backup` / `hook_backup` / `cached_hook_patches` 始终为 0。作为对照，原生在同样的组合下有 192/72（duck）、278/72（locon）项备份（UNet/TE）。
* 切换组合期间、以及撤掉 LoRA 之后，MODEL 和 CLIP 的全部权重与加载时逐字节一致（逐张量 blake2b）；撤掉 LoRA 之后，模块上不残留任何 weight function，出图与从没打过 LoRA 的结果逐位一致。
* Hook LoRA（keyframe 强度 1.0 → 0.4，`Set CLIP Hooks` + 条件上的 hooks）：TE 输出和 latent 都与原生逐位一致，没有任何备份或缓存。
* dtype 矩阵：54 种组合全部逐位一致。另外故意把「fp32→bf16」这类有损转换当作无损，测试会报出 6 个失败，说明它能发现这类问题。
* 报错：`force_patch_weights`（MODEL 和 CLIP 各一次）、非 comfy.ops 参数、形状改变、DynamicVRAM，都给出了预期的 kind 和 key。
* `ModelMergeSimple`（模型与挂了 LoRA 的自身克隆按 0.5 混合，patch 是整层大小的张量）：与原生逐位一致，无备份。
* `tests/run_all.sh` 合计 139 项检查（入口 3 + 2，dtype 矩阵 54，LoRA 80），0 失败；另在真实 ComfyUI 服务里通过 API 跑通了 CheckpointLoaderSimple + LoraLoader + KSampler + VAEDecode + SaveImage 的完整工作流。

**CPU 上的基准参考**（`tests/bench_lora.py`，SD1.5，256×256，3 步，只能看相对比例，不代表 GPU）：

| 组合 | 每步 原生 → Monoload | 切换组合（patch）原生 → Monoload | 结果差异 |
|---|---|---|---|
| Rubber Duck（192 层） | 7.13s → 15.74s（2.21×） | 6.57s → 0.08s | 0 |
| Annalise LoCon（278 层） | 9.13s → 24.23s（2.65×） | 12.55s → 0.08s | 0 |
| 两个叠加 | 9.37s → 26.56s（2.83×） | 16.64s → 0.08s | 0 |
| 不打 LoRA | 7.85s → 8.34s | 0.05s → 0.06s | 0 |

**要注意：** 运行时合并的每步开销，约等于每次模型调用都重做一遍原生那次性的合并。所以切换组合几乎零成本，但每一步都变慢了。GPU 上的比例要靠 9.1 的基准来测；分析和几种放宽方案见 DESIGN.md §5.3（没有实现，等你决定）。

## 9. CT 700 真机验收

准备：按第 2、3 节部署，确认日志里有 `runtime LoRA merge installed`。以下命令都在 CT 700 里执行，`comfyui` 是容器名。基准脚本在服务所在的同一个容器里另起一个进程，先让服务把模型卸掉：

```bash
curl -X POST http://127.0.0.1:8188/free -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}'
```

### 9.1 正确性 + 性能：基准脚本

同一个进程里依次跑「原生」「Monoload」「原生第二遍」，用的是同一份已经加载好的模型，然后逐组合比较结果和耗时。脚本会自动沿用容器主进程的 ComfyUI 启动参数。

```bash
# SDXL 整合包（CheckpointLoaderSimple），LoRA 放在 models/loras/
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_lora.py \
  --checkpoint <sdxl>.safetensors --width 1024 --height 1024 --steps 20 --cfg 6 --scheduler normal \
  --combo <loraA>.safetensors --combo <loraB>.safetensors --combo <loraA>.safetensors+<loraB>.safetensors --combo none \
  --modes native,monoload,native2 --repeat 2

# Krea 2（UNETLoader + CLIPLoader 分开加载）
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_lora.py \
  --unet krea2_turbo_bf16.safetensors --clip qwen3vl_4b_bf16.safetensors --clip-type krea2 \
  --width 1024 --height 1024 --steps 8 --cfg 1 --sampler euler --scheduler simple \
  --combo <krea2 LoRA>.safetensors --combo none --modes native,monoload,native2 --repeat 2
```

`--combo` 的写法：`a.safetensors[:模型强度[:CLIP强度]]`，多个 LoRA 用 `+` 连起来叠加，`none` 表示不打 LoRA。原生模式下 LoRA 会带着备份常驻 GTT，Krea 2 这类大模型要留意 GTT 余量。

**看这些数：**

* 每个组合一行：`keys` 是被 patch 的 UNet/TE key 数，`backups` 是备份数；`lora` / `encode` / `patch` 分别是 LoRA 节点耗时、文本编码耗时（含 TE 打 patch）、扩散模型加载耗时（原生是「写回上一组备份 + 合并新组合」，也就是切换 LoRA 组合的代价）；`step1` 和 `step` 是第一步和中位数的每步耗时，最后是当时的 GTT。
* 汇总表：每步耗时（native vs monoload，以及比值）、patch 耗时、encode 耗时、`max|Δ|`（两种模式最终 latent 的最大差异）。
* 最后的 layer probe：一次模型调用中，所有被 patch 层的临时拷贝耗时、逐位一致合并耗时、放宽合并耗时（放宽合并只在这里测量，插件并不使用）。

**通过标准：**

* `monoload vs native` 的 `max|Δ|` = 0。如果不是 0，看 `native2 vs native`：原生自己跑两次都有差异时，要求 monoload 的差异不超过它。
* Monoload 那几行的 `backups` 全是 0，原生那几行在有 LoRA 时不是 0。
* 切换组合时，Monoload 的 `patch` 明显小于原生。
* **把两张汇总表和 layer probe 的输出发给我**：用来判断每步开销是否可以接受，以及要不要采用 DESIGN.md §5.3 里的放宽方案。

### 9.2 显存（GTT）：没有备份

1. 开一个终端看内存：`docker exec comfyui sh /opt/ComfyUI/custom_nodes/monoload/tools/watch_mem.sh`（每 0.5 秒打印 GTT 和容器 cgroup 内存，以及相对启动时的峰值增量）。
2. 在 UI 里：Krea 2（`UNETLoader`）+ `LoraLoader`（模型和 CLIP 强度都不为 0）+ `CLIPLoader`（`qwen3vl_4b_bf16`，type `krea2`）+ `VAELoader`（`qwen_image_vae`），8 步、CFG 1、euler + simple。先不打 LoRA 出一张，记下 GTT；再打上 LoRA 出一张，记下 GTT。
3. 设 `MONOLOAD_DISABLE=1` 重启容器，重复第 2 步作为对照。

**通过标准：** Monoload 下打 LoRA 后，GTT 只多出 LoRA 大小加几百 MiB 的临时量；原生对照下会多出一份被 LoRA 改到的那些权重的大小（`--gpu-only` 时备份就在 GTT 里）。

### 9.3 撤掉 LoRA / 切换组合

在 UI 里依次出图：无 LoRA（A）→ LoRA 1 → LoRA 1+2 → LoRA 2 → 无 LoRA（B），种子相同。

```bash
docker exec comfyui python /opt/ComfyUI/custom_nodes/monoload/tools/compare_images.py /opt/ComfyUI/output/<A>.png /opt/ComfyUI/output/<B>.png
```

**通过标准：** A 和 B 输出 `identical`（打过 LoRA 再撤掉，结果与从来没打过一样）。可选：每个组合再和 `MONOLOAD_DISABLE=1` 下同种子的图比较，也应当是 `identical`，GPU 本身不确定时按 9.1 的办法判断。

### 9.4 日志

`docker logs comfyui 2>&1 | grep -iE "monoload|traceback"`，以及 `dmesg | grep -iE "oom|killed process|amdgpu.*(fault|timeout)"`：不应出现 `不支持`、`内部错误`、OOM、amdgpu fault。

## 10. 仓库结构

```
__init__.py                 ComfyUI 入口：没有设 MONOLOAD_DISABLE 时调用 hotpatch.install()
monoload/hotpatch.py        运行时合并的全部实现（MonoloadRuntimePatch + ModelPatcher 方法替换）
monoload/errors.py          报错类型
monoload/comfy_env.py       在独立进程里按指定参数启动 ComfyUI 环境（测试、基准用）
tests/                      测试和基准脚本（见第 8、9 节）
tools/watch_mem.sh          GTT / cgroup 内存监视
tools/compare_images.py     两张图逐像素比较
docs/DESIGN.md              设计说明
```
