# Monoload

**ComfyUI 插件：打 LoRA 时不改原权重、不做任何备份，每一层在计算的那一刻才临时合并 LoRA，用完即丢；每个 prompt 结束后自动释放 LoRA，底模继续常驻。** 装上后对所有模型生效（`UNETLoader`、`CheckpointLoaderSimple`、`CLIPLoader` 加载出的模型都算），原生的 `LoraLoader` / `LoraLoaderModelOnly` / Hook LoRA 节点照常使用，工作流不用改。结果与原生 ComfyUI **逐位一致**；唯一的例外是 fp8 这类量化层，那里有意做得比原生更精确（见 §5）。

为 Strix Halo（gfx1151，统一内存，显存走 GTT）+ ROCm、`--gpu-only` 的配置设计。在这种配置下，原生 ComfyUI 打 LoRA 时会把被改动的原权重在 GTT 里再备份一份，LoRA 改到的权重越多，备份就越大。

设计细节见 [docs/DESIGN.md](docs/DESIGN.md)。v1（离线转换 + pread 直读）已经废弃，代码保留在 tag `v1-converter`（远端归档分支 `archive/v1-converter`）。

参考环境：`docker.io/kyuz0/amd-strix-halo-comfyui@sha256:384aa1fecef6a841832e0d5552949977330308d8c25e212a94f5e8dfcc061cae`（ComfyUI 0.31.0，commit `62b3c94b`）。

---

## 1. 原理（一段话版）

原生 ComfyUI 在加载模型时把 LoRA「合并」进权重：先把原权重备份，再原地写入合并结果；换 LoRA 组合时写回备份，再重新合并。ComfyUI 在 lowvram 模式下本来就有另一条路：`comfy.ops` 的层只要挂了 `weight_function`，`forward` 时就会先拷一份临时权重，调用这些函数，算完就丢。Monoload 在 `ModelPatcher` 类上替换了几个方法，让「本该合并进权重」的 patch 改为挂成这样的 weight function，数值上复刻原生合并的每一步，所以结果逐位一致。换 LoRA 组合只是换 patch，不再有「写回备份 → 重新合并」。详见 DESIGN.md §2–§4。

另外，每个 prompt 执行完后，Monoload 会把 LoRA 从常驻模型和执行缓存里清掉（DESIGN.md §7）。具体包括：模型上的运行时 patch、计算设备上的 LoRA 副本、LoRA 节点缓存的 LoRA 文件，以及缓存里挂着 LoRA 的输出。底模和加载节点的缓存原样保留，下一个工作流直接用，不重新加载。

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
      # - MONOLOAD_KEEP_LORA=1                # 不在每个 prompt 结束后释放 LoRA
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
# 应看到：
#   [Monoload] runtime LoRA merge installed on ModelPatcher (no in-place LoRA, no weight backups)
#   [Monoload] LoRA is released after every prompt (base models stay loaded)
```

没有额外依赖，不注册新节点（节点列表里不会出现 Monoload）。以只读方式挂载即可。

## 4. 开关

| 设置 | 效果 |
|---|---|
| 默认（装上即生效） | 所有 `ModelPatcher` 走运行时合并；每个 prompt 结束后释放 LoRA，日志里有 `[Monoload] released LoRA after prompt: ...` |
| `MONOLOAD_KEEP_LORA=1` | 运行时合并照常，但 prompt 结束后不释放 LoRA（LoRA 节点缓存和挂着 LoRA 的 clone 保留，重复跑同一个 LoRA 工作流时省掉读 LoRA 文件） |
| `MONOLOAD_DISABLE=1` | 插件不做任何事，行为与原生完全一致；日志里是 `MONOLOAD_DISABLE is set: ... NOT installed` |

开关都接受 `1` / `true` / `yes` / `on`。

改完要重启容器。

## 5. 支持的范围

* **加载入口**：`UNETLoader`、`CheckpointLoaderSimple`（MODEL 和 CLIP）、`CLIPLoader` / `DualCLIPLoader` 等（文本编码器的 `CLIP.patcher`）；凡是 ComfyUI 原生的 `ModelPatcher` 都覆盖。
* **LoRA 节点**：`LoraLoader`（MODEL + CLIP 强度）、`LoraLoaderModelOnly`、多个 LoRA 叠加、Hook LoRA（`Create Hook LoRA`、`Set CLIP Hooks`、条件上的 hooks、keyframe 强度调度）。
* **LoRA 类型**：LoRA、LoCon、LoHa、LoKr，以及 `comfy.lora.calculate_weight` 支持的其他类型（GLoRA、OFT、BOFT、diff、set…）。
* **加载模式**：`--gpu-only` / `--highvram`（全量加载）是目标配置；普通模式下的部分加载也兼容，这时被卸载的层原生本来就用 `LowVramPatch`，Monoload 不去碰。
* **量化模型**（fp8 scaled 等，ComfyUI 的 mixed precision 层）：放宽处理。LoRA 直接合并在反量化出来的临时权重上，用完就丢，**不再量化回 fp8**。gfx1151 上原生本来就是每次 forward 先反量化（`supports_fp8_compute()` 为 False），所以没有额外的速度损失。结果比原生更精确，因此与原生不逐位一致；它与「先把这些层反量化成高精度、再走逐位一致的运行时合并」逐位一致。详见 DESIGN.md §3.3。
* **自动释放**：每个 prompt 结束后执行（成功、失败、中断都会）。原生 LoRA 节点（含 `LoraLoaderBypass`）、Hook LoRA 节点、第三方 LoRA 节点都覆盖，判断依据是缓存里的值是否带 LoRA patch 或 bypass LoRA 注入。三种执行缓存（默认的 RAM pressure、`--cache-classic`、`--cache-lru`）都支持。

## 6. 会直接报错的情况（绝不退回到「改权重 + 备份」）

报错信息写明是哪种情况和对应的 key，形如 `[Monoload] 不支持（<kind>） key=<key>: 说明`。

| kind | 情况 | 怎么办 |
|---|---|---|
| `dynamic_vram` | 开了 DynamicVRAM（comfy-aimdo）的同时打 LoRA | 用 `--gpu-only`（目标配置），或 `MONOLOAD_DISABLE=1` |
| `force_patch_weights` | 有节点要求把 LoRA 合并进权重：`ModelSave`、`CheckpointSave`、保存合并后的模型等 | 做这类操作时设 `MONOLOAD_DISABLE=1` 重启 |
| `lora_non_comfy_ops_param` | LoRA 改到的参数不属于 `comfy.ops` 层（没有运行时合并路径） | 设 `MONOLOAD_DISABLE=1` |
| `lora_shape_change` | patch 会改变权重形状 | 同上 |

## 7. 限制

* **每一步都有合并开销。** 原生是「加载时合并一次」，Monoload 是「每步、每个被 patch 的层都合并一次」。没被 patch 的层完全不受影响。开销和可选的放宽方案见 DESIGN.md §5，**GPU 上的实际数字请用第 9 节的基准脚本测**。
* 自己实现了 `patch_weight_to_device` 的第三方 `ModelPatcher` 子类（例如 ComfyUI-GGUF）不归 Monoload 管，保持它们自己的行为；日志里会对这个类打一次警告。
* LoRA 文件读进来后常驻 CPU 内存（原生也是这样）。另外，被用到的 LoRA 张量会在计算设备上缓存一份（只缓存比权重小的低秩因子等，模型合并带进来的整层权重不缓存），避免每步每层都做一次 H2D，模型卸载时释放。
* 模型合并节点（`ModelMergeSimple` 等）同样走运行时合并，每一步都会重新混合一遍，开销比 LoRA 大；需要反复使用合并结果时，建议先用 `MONOLOAD_DISABLE=1` 保存成新模型。合并结果也属于「带权重 patch」，会在 prompt 结束后一起释放，下次用到时重新执行合并节点。
* 自动释放之后，再跑同一个带 LoRA 的工作流时，要重新读一遍 LoRA 文件（通常只有几十到几百 MB）。如果下游节点已经命中缓存，LoRA 节点根本不会执行。不想要这个开销就设 `MONOLOAD_KEEP_LORA=1`。
* 非 `--gpu-only`、模型处于部分加载状态时，释放 LoRA 改走原生卸载（权重回到 offload 设备，下次采样再搬回来）。目标配置 `--gpu-only` 下是就地释放，不搬任何权重。
* 本仓库的测试都在无 GPU 的机器上用 `--cpu` 跑；GPU 上的一致性和性能要按第 9 节在 CT 700 上确认。

## 8. 测试（CPU，锁定镜像）

```bash
# $MODELS 下需要：
#   checkpoints/v1-5-pruned-emaonly-fp16.safetensors
#       https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors
#   diffusion_models/v1-5-pruned-emaonly-fp16.safetensors   （同一个文件，硬链接即可；UNETLoader 会从中取出 UNet）
#   diffusion_models/sd15_unet_fp8_scaled.safetensors      tests/make_fp8_unet.py 生成（SD1.5 UNet 的 Linear 层转成 fp8 scaled）
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
| `tests/test_entry.py` | 按 ComfyUI 的方式加载插件：默认替换 `ModelPatcher` 的方法（`CoreModelPatcher` 也覆盖到）并包装 `PromptExecutor.execute_async`；`MONOLOAD_KEEP_LORA=1` 时不包装执行器；`MONOLOAD_DISABLE=1` 时两者都不动 |
| `tests/test_dtype_paths.py` | 参数 dtype × 计算 dtype × lora dtype（含 gfx1151 上的 fp16）× {基础 LoRA、hook、两者都有}，54 种组合逐位对比原生合并的数值；再加 fp8 参数的 54 种组合，对比「先反量化」参照 |
| `tests/test_lora_hot.py` | 两条管线：`CheckpointLoaderSimple`；`UNETLoader` + `CLIPLoader`。同一串 LoRA 组合先用原生跑、再装上 Monoload 连续切换着跑，比较 TE 输出和采样结果；加上 Hook LoRA、模型合并和报错场景 |
| `tests/test_quant.py` | fp8 scaled UNet + LoRA：与「先反量化被改到的层 + 逐位一致路径」逐位一致、无备份、fp8 权重不变；与原生的误差只报告 |
| `tests/test_release.py` | 用真正的 `PromptExecutor` 连续跑 LoRA → 无 LoRA → Hook LoRA → 无 LoRA → bypass LoRA → 无 LoRA → LoRA（换种子）：弱引用确认 LoRA 全部释放、底模不重新加载、结果与从没见过 LoRA 的进程逐位一致；RAM pressure / classic / LRU 三种缓存各一遍，外加 `MONOLOAD_KEEP_LORA=1` |

**实测结果**（`--cpu --fp16-unet`，SD1.5，每个组合采样 2 步，hook 3 步）：

* 两条管线各切换 8 次组合，顺序是 duck → locon → duck+locon → duck → none → lokr → loha → duck+locon：
  * 文本编码器输出（cond 和 pooled）、采样得到的 latent 全部与原生**逐位一致**（max_abs = 0）；
  * MODEL 和 CLIP 的 `backup` / `hook_backup` / `cached_hook_patches` 始终为 0。作为对照，原生在同样的组合下有 192/72（duck）、278/72（locon）项备份（UNet/TE）。
* 切换组合期间、以及撤掉 LoRA 之后，MODEL 和 CLIP 的全部权重与加载时逐字节一致（逐张量 blake2b）；撤掉 LoRA 之后，模块上不残留任何 weight function，出图与从没打过 LoRA 的结果逐位一致。
* Hook LoRA（keyframe 强度 1.0 → 0.4，`Set CLIP Hooks` + 条件上的 hooks）：TE 输出和 latent 都与原生逐位一致，没有任何备份或缓存。
* dtype 矩阵：54 种组合全部逐位一致。另外故意把「fp32→bf16」这类有损转换当作无损，测试会报出 6 个失败，说明它能发现这类问题。
* 报错：`force_patch_weights`（MODEL 和 CLIP 各一次）、非 comfy.ops 参数、形状改变、DynamicVRAM，都给出了预期的 kind 和 key。
* `ModelMergeSimple`（模型与挂了 LoRA 的自身克隆按 0.5 混合，patch 是整层大小的张量）：与原生逐位一致，无备份。
* fp8（量化层放宽合并）：Rubber Duck（160 个 fp8 层被 patch）、Annalise LoCon（182 个）都与「先反量化」参照逐位一致，无备份，fp8 权重（qdata + scale）不变；与原生 fp8 LoRA（合并后重新量化成 fp8）的 latent 差异 max_abs 4.76 / 1.88（LoRA 本身的影响 max_abs 34 / 26）。另外发现：不打 LoRA 时，把整个模型都反量化也会和 fp8 模型差 3e-4，因为原生没挂 LoRA 的 fp8 层直接用 `QuantizedTensor` 做 `F.linear`，所以参照只反量化被改到的层（DESIGN.md §3.3）。
* 自动释放（三种缓存都一样）：每个 LoRA prompt 结束后，从 `models/loras` 读出的全部张量都已回收（到三个 LoRA prompt 结束时累计登记 792 / 1584 / 2376 个，弱引用全部失效），没有残留的带 patch 的 patcher、运行时 patch、设备缓存、`loaded_lora`；接下来的无 LoRA prompt 没有再执行 `CheckpointLoaderSimple`，`ModelPatcher.load()` 调用 0 次，结果与从没见过 LoRA 的进程逐位一致；释放后再用 LoRA 也与全新进程逐位一致。`MONOLOAD_KEEP_LORA=1` 时 LoRA 保留。
* `tests/run_all.sh` 合计 322 项检查（入口 11，dtype 矩阵 108，LoRA 80，fp8 8，释放 2 + 32×3 + 17），0 失败。

**CPU 上的基准参考**（`tests/bench_lora.py`，SD1.5，256×256，3 步，只能看相对比例，不代表 GPU）：

| 组合 | 每步 原生 → Monoload | 切换组合（patch）原生 → Monoload | 结果差异 |
|---|---|---|---|
| Rubber Duck（192 层） | 7.13s → 15.74s（2.21×） | 6.57s → 0.08s | 0 |
| Annalise LoCon（278 层） | 9.13s → 24.23s（2.65×） | 12.55s → 0.08s | 0 |
| 两个叠加 | 9.37s → 26.56s（2.83×） | 16.64s → 0.08s | 0 |
| 不打 LoRA | 7.85s → 8.34s | 0.05s → 0.06s | 0 |

同一台 CPU 上，ComfyUI 自带的 bypass LoRA（`--modes ...,bypass`，只作对照）：Rubber Duck 8.56s → 10.27s（1.20×），Annalise LoCon 9.10s → 13.54s（1.49×）；与原生合并的 latent 差异 max_abs 0.033 / 0.155（LoRA 本身的影响约 34）。同一轮里 Monoload 是 16.90s / 24.60s。

**要注意：** 运行时合并的每步开销，约等于每次模型调用都重做一遍原生那次性的合并。所以切换组合几乎零成本，但每一步都变慢了。GPU 上的比例要靠 9.1 的基准来测；分析和几种放宽方案见 DESIGN.md §5.3（没有实现，等你决定）。

## 9. CT 700 真机验收

准备：按第 2、3 节部署，确认日志里有 `runtime LoRA merge installed`。以下命令都在 CT 700 里执行，`comfyui` 是容器名。基准脚本在服务所在的同一个容器里另起一个进程，先让服务把模型卸掉：

```bash
curl -X POST http://127.0.0.1:8188/free -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}'
```

### 9.1 正确性 + 性能：基准脚本

同一个进程里、用同一份已经加载好的模型，依次跑几种模式：`native`（原生，Monoload 卸下）、`monoload`、`bypass`（ComfyUI 自带的 bypass LoRA，`comfy.sd.load_bypass_lora_for_models`，只作对照）、`native2`（原生再跑一遍，量 GPU 自身的不确定性），然后逐组合比较结果和耗时。脚本会自动沿用容器主进程的 ComfyUI 启动参数。

```bash
# SDXL 整合包（CheckpointLoaderSimple），LoRA 放在 models/loras/
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_lora.py \
  --checkpoint <sdxl>.safetensors --width 1024 --height 1024 --steps 20 --cfg 6 --scheduler normal \
  --combo <loraA>.safetensors --combo <loraB>.safetensors --combo <loraA>.safetensors+<loraB>.safetensors --combo none \
  --modes native,monoload,bypass,native2 --repeat 2

# Krea 2（UNETLoader + CLIPLoader 分开加载）
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_lora.py \
  --unet krea2_turbo_bf16.safetensors --clip qwen3vl_4b_bf16.safetensors --clip-type krea2 \
  --width 1024 --height 1024 --steps 8 --cfg 1 --sampler euler --scheduler simple \
  --combo <krea2 LoRA>.safetensors --combo none --modes native,monoload,bypass,native2 --repeat 2
```

`--combo` 的写法：`a.safetensors[:模型强度[:CLIP强度]]`，多个 LoRA 用 `+` 连起来叠加，`none` 表示不打 LoRA。原生模式下 LoRA 会带着备份常驻 GTT，Krea 2 这类大模型要留意 GTT 余量。

**看这些数：**

* 每个组合一行：`keys` 是被 patch 的 UNet/TE key 数，`backups` 是备份数；`lora` / `encode` / `patch` 分别是 LoRA 节点耗时、文本编码耗时（含 TE 打 patch）、扩散模型加载耗时（原生是「写回上一组备份 + 合并新组合」，也就是切换 LoRA 组合的代价）；`step1` 和 `step` 是第一步和中位数的每步耗时，最后是当时的 GTT。
* 汇总表（每种模式对 `native` 各一张）：每步耗时和比值、patch 耗时、encode 耗时、`max|Δ|`（最终 latent 的最大差异）。`bypass` 按设计就和合并路径的数值不同，它的 `max|Δ|` 只作参考。
* 最后的 layer probe：一次模型调用中，所有被 patch 层的临时拷贝耗时、逐位一致合并耗时、放宽合并耗时（放宽合并只在这里测量，插件并不使用）。

**通过标准：**

* `monoload vs native` 的 `max|Δ|` = 0。如果不是 0，看 `native2 vs native`：原生自己跑两次都有差异时，要求 monoload 的差异不超过它。
* Monoload 那几行的 `backups` 全是 0，原生那几行在有 LoRA 时不是 0。
* 切换组合时，Monoload 的 `patch` 明显小于原生。
* **把所有汇总表和 layer probe 的输出发给我**：用来判断每步开销是否可以接受，以及要不要采用 DESIGN.md §5.3 里的放宽方案（`bypass` 那张表就是方案 D 的实测数据）。

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

### 9.4 每个 prompt 结束后释放 LoRA

1. 重启容器，开着 `watch_mem.sh`。
2. UI 里跑一次**不带** LoRA 的 Krea 2 工作流，记下 GTT 和容器内存（G0 / C0）。
3. 同一个工作流加上 `LoraLoader`（模型和 CLIP 强度都不为 0），跑完后记下 GTT / 容器内存（G1 / C1）。日志里应有一行 `[Monoload] released LoRA after prompt: N loaded model(s) back to base, ...`。
4. 再跑一次第 2 步那个不带 LoRA 的工作流，记下 GTT / 容器内存（G2 / C2），出图记为 B；和第 2 步的图（A）比较：`tools/compare_images.py A.png B.png`。

**通过标准：**
* G1 ≈ G0、C1 ≈ C0（差值在几百 MiB 以内，也就是临时量和分配器缓存的量级，远小于 LoRA 文件大小加上被改层的大小）：LoRA 在 GPU 和 CPU 上都已释放。对照组：设 `MONOLOAD_KEEP_LORA=1` 重启后重复第 3 步，C1 会高出约一个 LoRA 文件的大小。
* 第 4 步的日志里**没有** `Requested to load` / `loaded completely` 这类重新加载底模的行，这一步的耗时和第 2 步第二次运行时相当；A 和 B `identical`。
* 也可以用测试脚本在 GPU 上跑同样的检查（需要 SD1.5 整合包和 `rubber_duck.safetensors`，见第 8 节）：
  ```bash
  docker exec -w /opt/ComfyUI/custom_nodes/monoload -e COMFY_ARGS="--gpu-only --bf16-vae" comfyui python tests/test_release.py --save-reference /tmp/ref.pt
  docker exec -w /opt/ComfyUI/custom_nodes/monoload -e COMFY_ARGS="--gpu-only --bf16-vae" comfyui python tests/test_release.py --reference /tmp/ref.pt
  ```
  GPU 上「与参照逐位一致」那几项如果只差一点点，说明 GPU 自身的计算不确定，用 9.1 里 `native2` 的差异来判断即可；其余检查（弱引用全部失效、没有 `load()` 调用、底模没有重新加载）与 GPU 无关，必须全部通过。

### 9.5 fp8 模型（有的话）

对一个 fp8 scaled 的扩散模型挂 LoRA 出图，确认：没有报错；GTT 不多出一份备份；图正常。它和 `MONOLOAD_DISABLE=1` 下的图会有细微差别（原生把合并结果重新量化成 fp8，Monoload 不会），这是预期行为。

### 9.6 日志

`docker logs comfyui 2>&1 | grep -iE "monoload|traceback"`，以及 `dmesg | grep -iE "oom|killed process|amdgpu.*(fault|timeout)"`：不应出现 `不支持`、`内部错误`、OOM、amdgpu fault。

## 10. 仓库结构

```
__init__.py                 ComfyUI 入口：按 MONOLOAD_DISABLE / MONOLOAD_KEEP_LORA 调用 hotpatch.install()、release.install()
monoload/hotpatch.py        运行时合并的全部实现（MonoloadRuntimePatch + ModelPatcher 方法替换，含量化层的放宽合并）
monoload/release.py         每个 prompt 结束后释放 LoRA（包装 PromptExecutor.execute_async）
monoload/errors.py          报错类型
monoload/comfy_env.py       在独立进程里按指定参数启动 ComfyUI 环境（测试、基准用）
tests/                      测试和基准脚本（见第 8、9 节）
tools/watch_mem.sh          GTT / cgroup 内存监视
tools/compare_images.py     两张图逐像素比较
docs/DESIGN.md              设计说明
```
