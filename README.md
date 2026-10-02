# Monoload

**ComfyUI 插件：打 LoRA 时不改原权重、不做任何备份，每一层在计算的那一刻才临时合并 LoRA，用完即丢；每个 prompt 结束后自动释放 LoRA，底模继续常驻。** 装上后对所有模型生效（`UNETLoader`、`CheckpointLoaderSimple`、`CLIPLoader` 加载出的模型都算），原生的 `LoraLoader` / `LoraLoaderModelOnly` / Hook LoRA 节点照常使用，工作流不用改。

合并默认走快速路径：普通 LoRA / LoCon 直接用一次融合的 fp16 `addmm_` 加进临时权重，其他类型在计算 dtype 下合并。它与原生合并的差异在 fp16 舍入量级（CT 700 实测：合并后权重的差异是 LoRA 改动量的 0.25%，单个元素最多差 1 个 ulp，见 §5.1）。设 `MONOLOAD_EXACT=1` 时，结果与原生 ComfyUI **逐位一致**，但每步更慢。fp8 这类量化层在两种模式下都有意做得比原生更精确（见 §5）。

为 Strix Halo（gfx1151，统一内存，显存走 GTT）+ ROCm、`--gpu-only` 的配置设计。在这种配置下，原生 ComfyUI 打 LoRA 时会把被改动的原权重在 GTT 里再备份一份，LoRA 改到的权重越多，备份就越大。

**另一项功能：VAE 解码降峰值（第一阶段，§12）。** 接管 `VAE.decode`：卷积按输出行分块、注意力按 query 分块，限制 im2col 展开缓冲和注意力分数矩阵；自己给 `load_models_gpu` 报真实的内存需求，不再用原生约 92 GiB（SDXL 4K）的估算去卸载别的模型；OOM 时只缩小分块重试，绝不退回 tiled 近似解码。结果与整图解码数学等价（只差浮点误差）。与 LoRA 部分相互独立，可以单独关掉。

设计细节见 [docs/DESIGN.md](docs/DESIGN.md)。v1（离线转换 + pread 直读）已经废弃，代码保留在 tag `v1-converter`（远端归档分支 `archive/v1-converter`）。

参考环境：`docker.io/kyuz0/amd-strix-halo-comfyui@sha256:384aa1fecef6a841832e0d5552949977330308d8c25e212a94f5e8dfcc061cae`（ComfyUI 0.31.0，commit `62b3c94b`）。

---

## 1. 原理（一段话版）

原生 ComfyUI 在加载模型时把 LoRA「合并」进权重：先把原权重备份，再原地写入合并结果；换 LoRA 组合时写回备份，再重新合并。ComfyUI 在 lowvram 模式下本来就有另一条路：`comfy.ops` 的层只要挂了 `weight_function`，`forward` 时就会先拷一份临时权重，调用这些函数，算完就丢。Monoload 在 `ModelPatcher` 类上替换了几个方法，让「本该合并进权重」的 patch 改为挂成这样的 weight function。默认用融合的 `addmm_` 把 LoRA 加进那份临时权重；`MONOLOAD_EXACT=1` 时则数值上复刻原生合并的每一步，结果逐位一致。换 LoRA 组合只是换 patch，不再有「写回备份 → 重新合并」。详见 DESIGN.md §2–§4。

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
      # - MONOLOAD_EXACT=1                    # 与原生逐位一致的合并（每步更慢），默认是融合 fp16 addmm
      # - MONOLOAD_KEEP_LORA=1                # 不在每个 prompt 结束后释放 LoRA
      # - MONOLOAD_DISABLE_VAE=1              # 只关掉 VAE 解码管理（§12），LoRA 部分照常
      # - MONOLOAD_VAE_WORKSPACE=1G           # VAE 分块的工作区预算（默认 1G）
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
#   [Monoload] runtime LoRA merge installed on ModelPatcher (no in-place LoRA, no weight backups), merge: fused fp16 addmm / relaxed (set MONOLOAD_EXACT=1 for bit-exact)
#   [Monoload] LoRA is released after every prompt (base models stay loaded)
#   [Monoload] VAE decode managed: op-level chunking (conv row blocks, attention query blocks), workspace 1.00 GiB ...
```

没有额外依赖，不注册新节点（节点列表里不会出现 Monoload）。以只读方式挂载即可。

## 4. 开关

| 设置 | 效果 |
|---|---|
| 默认（装上即生效） | 所有 `ModelPatcher` 走运行时合并，合并用快速路径（普通 LoRA/LoCon 融合 fp16 `addmm_`，其他类型在计算 dtype 下合并）；每个 prompt 结束后释放 LoRA，日志里有 `[Monoload] released LoRA after prompt: ...` |
| `MONOLOAD_EXACT=1` | 合并改走逐位一致路径：结果与原生 ComfyUI 完全相同，每步多出的合并开销更大（CT 700 上约是默认路径的 3 倍：0.26s 对 0.08s，见 §5.1）。用于验收、对比，或需要和原生出图完全一致的时候。VAE 解码也走原生（分块会改变 GEMM 形状，不保证逐位一致） |
| `MONOLOAD_KEEP_LORA=1` | 运行时合并照常，但 prompt 结束后不释放 LoRA（LoRA 节点缓存和挂着 LoRA 的 clone 保留，重复跑同一个 LoRA 工作流时省掉读 LoRA 文件） |
| `MONOLOAD_DISABLE_VAE=1` | 只关掉 VAE 解码管理（§12），`VAE.decode` 保持原生；LoRA 部分不受影响 |
| `MONOLOAD_VAE_WORKSPACE` | VAE 分块的工作区预算，默认 `1G`；写法 `1G`、`512M`、`768`（纯数字按 MiB）。决定卷积每块的行数和注意力每块的 query 数（§12） |
| `MONOLOAD_DISABLE=1` | 插件不做任何事，行为与原生完全一致（LoRA 和 VAE 都不接管）；日志里是 `MONOLOAD_DISABLE is set: ... NOT installed` |

开关都接受 `1` / `true` / `yes` / `on`。

改完要重启容器。

## 5. 支持的范围

* **加载入口**：`UNETLoader`、`CheckpointLoaderSimple`（MODEL 和 CLIP）、`CLIPLoader` / `DualCLIPLoader` 等（文本编码器的 `CLIP.patcher`）；凡是 ComfyUI 原生的 `ModelPatcher` 都覆盖。
* **LoRA 节点**：`LoraLoader`（MODEL + CLIP 强度）、`LoraLoaderModelOnly`、多个 LoRA 叠加、Hook LoRA（`Create Hook LoRA`、`Set CLIP Hooks`、条件上的 hooks、keyframe 强度调度）。
* **LoRA 类型**：LoRA、LoCon、LoHa、LoKr，以及 `comfy.lora.calculate_weight` 支持的其他类型（GLoRA、OFT、BOFT、diff、set…）。默认路径下，普通 LoRA / LoCon 走融合 `addmm_`，其余类型（含 DoRA、带 Tucker mid 的 LoCon、模型合并）在计算 dtype 下走 `calculate_weight`，数值与原生 lowvram 的 `LowVramPatch` 相同。
* **加载模式**：`--gpu-only` / `--highvram`（全量加载）是目标配置；普通模式下的部分加载也兼容，这时被卸载的层原生本来就用 `LowVramPatch`，Monoload 不去碰。
* **量化模型**（fp8 scaled 等，ComfyUI 的 mixed precision 层）：放宽处理，两种模式都一样。LoRA 直接合并在反量化出来的临时权重上，用完就丢，**不再量化回 fp8**。gfx1151 上原生本来就是每次 forward 先反量化（`supports_fp8_compute()` 为 False），所以没有额外的速度损失。结果比原生更精确，因此与原生不逐位一致；它与「先把这些层反量化成高精度、再走同一条合并路径」逐位一致。详见 DESIGN.md §3.3。
* **自动释放**：每个 prompt 结束后执行（成功、失败、中断都会）。原生 LoRA 节点（含 `LoraLoaderBypass`）、Hook LoRA 节点、第三方 LoRA 节点都覆盖，判断依据是缓存里的值是否带 LoRA patch 或 bypass LoRA 注入。三种执行缓存（默认的 RAM pressure、`--cache-classic`、`--cache-lru`）都支持。

### 5.1 默认路径与原生的差异

默认路径和原生合并的差别只在舍入：原生把 LoRA 改动先用 fp32 算好，加回权重后再舍入一次回 fp16；默认路径在 fp16 下用一个 GEMM（`addmm_`）直接加回，累加和舍入发生在 GEMM 内部。

CT 700 实测（WAI v17 SDXL + Smooth Booster，788 层，4.77 GiB 被 patch 的 fp16 权重，`bench_sdxl_v2`）：

| 合并方式 | 每次模型调用的合并开销（不含 0.038s 临时拷贝） | 合并后权重与原生的差异：‖Δw‖ / ‖LoRA 改动‖ | 单元素最大差异 |
|---|---|---|---|
| 逐位一致（`MONOLOAD_EXACT=1`） | 0.221s | 0 | 0 |
| 默认路径（融合 addmm，C） | 0.040s | 2.54e-3 | 1.22e-4（1 个 fp16 ulp，量级 0.125–0.25 的权重） |
| 仅放宽合并（A，不采用） | 0.117s | 3.56e-5 | 6.1e-5（1 个 ulp） |

也就是说，合并后的权重最多在个别元素上差 1 个 fp16 ulp，总量是 LoRA 改动本身的 0.25%。

每步耗时实测（`bench_sdxl_v3`）：原生约 0.64s；默认路径 1.13× / 1.10× / 叠加 1.18×；逐位一致 1.39× / 1.33× / 1.67×。

**出图会有可见的细节差异。** SDXL 20 步采样会把权重上 1 个 ulp 的差异放大：最终 latent 与原生的 mean|Δ| 是 LoRA 本身作用的 4–15%，量级与 ComfyUI 自带的 bypass LoRA 相当。这是采样放大造成的，已经实测确认：权重误差小 70 倍的放宽合并（A，也就是原生 lowvram 的数值），latent 差异只小 1.2–2.5 倍。

同种子出图对比（Smooth Booster，seed 42）：默认路径和原生的图构图、风格一致，只是细节位置有偏移，肉眼看不出画质差别，但 97.5% 的像素不同（mean abs diff 5.29）。LoRA 的效果强度不变。需要和原生出图完全一致时用 `MONOLOAD_EXACT=1`。详见 DESIGN.md §5.5。

测试里的容差（DESIGN.md §5.5）：`‖Δw‖ ≤ u·(‖W‖ + 20·‖LoRA 改动‖)`，u 是比较精度的单位舍入（fp16 为 2⁻¹¹）。上面 2.54e-3 只用掉「20·u = 9.8e-3」这一项的约四分之一。

## 6. 会直接报错的情况（绝不退回到「改权重 + 备份」）

报错信息写明是哪种情况和对应的 key，形如 `[Monoload] 不支持（<kind>） key=<key>: 说明`。

| kind | 情况 | 怎么办 |
|---|---|---|
| `dynamic_vram` | 开了 DynamicVRAM（comfy-aimdo）的同时打 LoRA | 用 `--gpu-only`（目标配置），或 `MONOLOAD_DISABLE=1` |
| `force_patch_weights` | 有节点要求把 LoRA 合并进权重：`ModelSave`、`CheckpointSave`、保存合并后的模型等 | 做这类操作时设 `MONOLOAD_DISABLE=1` 重启 |
| `lora_non_comfy_ops_param` | LoRA 改到的参数不属于 `comfy.ops` 层（没有运行时合并路径） | 设 `MONOLOAD_DISABLE=1` |
| `lora_shape_change` | patch 会改变权重形状 | 同上 |

## 7. 限制

* **每一步都有合并开销。** 原生是「加载时合并一次」，Monoload 是「每步、每个被 patch 的层都合并一次」。没被 patch 的层完全不受影响。CT 700（SDXL，1344×768）实测：默认路径每步 1.10–1.18×，逐位一致路径 1.33–1.67×。分析见 DESIGN.md §5。
* 自己实现了 `patch_weight_to_device` 的第三方 `ModelPatcher` 子类（例如 ComfyUI-GGUF）不归 Monoload 管，保持它们自己的行为；日志里会对这个类打一次警告。
* LoRA 文件读进来后常驻 CPU 内存（原生也是这样）。另外，被用到的 LoRA 张量会在计算设备上缓存一份（只缓存比权重小的低秩因子等，模型合并带进来的整层权重不缓存），避免每步每层都做一次 H2D，模型卸载时释放。
* 模型合并节点（`ModelMergeSimple` 等）同样走运行时合并，每一步都会重新混合一遍，开销比 LoRA 大；需要反复使用合并结果时，建议先用 `MONOLOAD_DISABLE=1` 保存成新模型。合并结果也属于「带权重 patch」，会在 prompt 结束后一起释放，下次用到时重新执行合并节点。
* 自动释放之后，再跑同一个带 LoRA 的工作流时，要重新读一遍 LoRA 文件（通常只有几十到几百 MB）。如果下游节点已经命中缓存，LoRA 节点根本不会执行。不想要这个开销就设 `MONOLOAD_KEEP_LORA=1`。
* 非 `--gpu-only`、模型处于部分加载状态时，释放 LoRA 改走原生卸载（权重回到 offload 设备，下次采样再搬回来）。目标配置 `--gpu-only` 下是就地释放，不搬任何权重。
* 默认路径与原生合并不逐位一致（§5.1）。需要和原生出图完全一致时设 `MONOLOAD_EXACT=1`。
* 本仓库的测试都在无 GPU 的机器上用 `--cpu` 跑；GPU 上的一致性和性能要按第 9 节在 CT 700 上确认。
* VAE 解码管理的限制见 §12.5。

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
#   loras/synthetic_{lokr,loha,unet_only}_sd15.safetensors
#       tests/make_synthetic_loras.py 生成（HF 上找不到 SD1.5 的 LoKr，按真实 LoRA 的层名和形状合成，UNet + TE；
#       unet_only 是 Rubber Duck 去掉 TE key 的版本）
MODELS=/path/to/models tests/run_all.sh
```

`run_all.sh` 先跑 5 遍入口测试和 VAE 测试，然后把下面每个 LoRA 功能测试在两种合并路径下各跑一遍：先 `MONOLOAD_EXACT=1`（逐位一致），再默认路径。只想跑 VAE 部分时：`MODELS=/path/to/models tests/docker_run.sh python tests/test_vae.py`（不需要任何模型文件，`$MODELS` 可以是空目录）。

| 脚本 | 内容 |
|---|---|
| `tests/test_entry.py` | 按 ComfyUI 的方式加载插件：默认替换 `ModelPatcher` 的方法（`CoreModelPatcher` 也覆盖到）、包装 `PromptExecutor.execute_async` 和 `VAE.decode`；`MONOLOAD_KEEP_LORA=1` 时不包装执行器；`MONOLOAD_DISABLE_VAE=1` 或 `MONOLOAD_EXACT=1` 时 `VAE.decode` 保持原生；`MONOLOAD_DISABLE=1` 时都不动；合并路径与 `MONOLOAD_EXACT` 一致；`decode_tiled` 从不被替换 |
| `tests/test_dtype_paths.py` | `EXACT=1`：参数 dtype × 计算 dtype × lora dtype（含 gfx1151 上的 fp16）× {基础 LoRA、hook、两者都有}，54 种组合逐位对比原生合并的数值。默认路径：同样的 dtype 组合 × 8 种 patch 组合（LoRA、两个 LoRA、strength_model ≠ 1、LoHa、LoRA+LoHa、diff、hook、LoRA+hook），共 144 项；每项要求与独立实现的参照（`addmm_` / 原生 `LowVramPatch`）逐位一致，并且与原生合并的差异在容差以内。两种模式都再加 fp8 参数的 54 种组合，对比「先反量化 + 同一路径」 |
| `tests/test_lora_hot.py` | 两条管线：`CheckpointLoaderSimple`；`UNETLoader` + `CLIPLoader`。同一串 LoRA 组合先用原生跑、再装上 Monoload 连续切换着跑。`EXACT=1`：TE 输出和采样结果与原生逐位一致。默认路径：UNet 和 TE 所有被 patch 的权重与原生合并的差异在容差以内（DESIGN.md §5.5），同样的 key 上重算逐位一致路径与原生逐位一致；latent 差异只报告。两种模式都检查无备份、权重不变、撤掉 LoRA 后逐位一致，加上 Hook LoRA、模型合并和报错场景 |
| `tests/test_quant.py` | fp8 scaled UNet + LoRA：与「先反量化被改到的层 + 同一合并路径」逐位一致、无备份、fp8 权重不变；与原生的误差只报告 |
| `tests/test_vae.py` | VAE 解码管理（§12），不需要模型文件：分块卷积 / 分块注意力与不分块的结果一致（含各种 kernel、stride、dilation、groups、padding、cast 路径 + weight_function、Wan CausalConv3d）；用 ComfyUI 自己的 LDM `Decoder` / `WanVAE`（小通道、随机权重）构造 SDXL 式、Flux 式、Qwen 式 VAE，受管理的解码与原生 `VAE.decode` 比较；多帧交给原生、OOM 缩小分块重试、下限时报错、绝不调用 tiled |
| `tests/test_release.py` | 用真正的 `PromptExecutor` 连续跑 LoRA → 无 LoRA → 只改 UNet 的 LoRA → 无 LoRA → Hook LoRA → 无 LoRA → bypass LoRA → 无 LoRA → LoRA（换种子）：弱引用确认 LoRA 全部释放、底模不重新加载、结果与从没见过 LoRA 的进程逐位一致；RAM pressure / classic / LRU 三种缓存各一遍，外加 `MONOLOAD_KEEP_LORA=1` |

**实测结果**（`--cpu --fp16-unet`，SD1.5，每个组合采样 2 步，hook 3 步）。以下逐位一致的结论都是 `MONOLOAD_EXACT=1` 下的：

* 两条管线各切换 8 次组合，顺序是 duck → locon → duck+locon → duck → none → lokr → loha → duck+locon：
  * 文本编码器输出（cond 和 pooled）、采样得到的 latent 全部与原生**逐位一致**（max_abs = 0）；
  * MODEL 和 CLIP 的 `backup` / `hook_backup` / `cached_hook_patches` 始终为 0。作为对照，原生在同样的组合下有 192/72（duck）、278/72（locon）项备份（UNet/TE）。
* 切换组合期间、以及撤掉 LoRA 之后，MODEL 和 CLIP 的全部权重与加载时逐字节一致（逐张量 blake2b）；撤掉 LoRA 之后，模块上不残留任何 weight function，出图与从没打过 LoRA 的结果逐位一致。
* Hook LoRA（keyframe 强度 1.0 → 0.4，`Set CLIP Hooks` + 条件上的 hooks）：TE 输出和 latent 都与原生逐位一致，没有任何备份或缓存。
* dtype 矩阵：54 种组合全部逐位一致。另外故意把「fp32→bf16」这类有损转换当作无损，测试会报出 6 个失败，说明它能发现这类问题。
* 报错：`force_patch_weights`（MODEL 和 CLIP 各一次）、非 comfy.ops 参数、形状改变、DynamicVRAM，都给出了预期的 kind 和 key。
* `ModelMergeSimple`（模型与挂了 LoRA 的自身克隆按 0.5 混合，patch 是整层大小的张量）：与原生逐位一致，无备份。
* fp8（量化层放宽合并）：Rubber Duck（160 个 fp8 层被 patch）、Annalise LoCon（182 个）都与「先反量化」参照逐位一致，无备份，fp8 权重（qdata + scale）不变；与原生 fp8 LoRA（合并后重新量化成 fp8）的 latent 差异 max_abs 4.76 / 1.88（LoRA 本身的影响 max_abs 34 / 26）。另外发现：不打 LoRA 时，把整个模型都反量化也会和 fp8 模型差 3e-4，因为原生没挂 LoRA 的 fp8 层直接用 `QuantizedTensor` 做 `F.linear`，所以参照只反量化被改到的层（DESIGN.md §3.3）。
* 自动释放（三种缓存都一样）：每个 LoRA prompt 结束后，从 `models/loras` 读出的全部张量都已回收（每个 LoRA prompt 结束时，此前登记的全部弱引用都已失效），没有残留的带 patch 的 patcher、运行时 patch、设备缓存、`loaded_lora`；接下来的无 LoRA prompt 没有再执行 `CheckpointLoaderSimple`，`ModelPatcher.load()` 调用 0 次，结果与从没见过 LoRA 的进程逐位一致；释放后再用 LoRA 也与全新进程逐位一致。`MONOLOAD_KEEP_LORA=1` 时 LoRA 保留。
* 默认路径：
  * dtype 矩阵 144 项全部与独立参照逐位一致，并且与原生合并的差异都在容差以内（最坏的一项用掉容差的 57%）。
  * 两条管线、8 个组合：UNet 和 TE 的权重与原生合并比较，‖Δw‖/‖LoRA 改动‖ 最大 2.8e-4（容差 0.03–0.27）；LoKr、LoHa、模型合并按 fp16 比较是 0。latent 与原生的 mean|Δ| 是 0.005–0.012，LoRA 本身的作用是 4.9–7.3（LoHa 0.39），hook 是 0.024 对 7.51。数字见 DESIGN.md §5.5。
  * 在同样的 key 上重算逐位一致路径，与原生全部逐位一致。
  * 备份、权重不变、撤掉 LoRA、报错、fp8、自动释放（三种缓存 + KEEP）这些检查与逐位一致路径完全相同，全部通过。
* VAE 解码管理（`tests/test_vae.py`，131 项，0 失败）：分块卷积 53 项、分块注意力 18 项与不分块一致（卷积相对误差最大 4.6e-7，注意力 3.5e-7，多数为 0）；SDXL 式 / Flux 式 / Qwen 式 decoder 在 16 KiB 和 256 KiB 预算下（绝大多数卷积被分块）与原生 `VAE.decode` 的 raw 输出差 ≤ 7.2e-6、像素差 ≤ 3.2e-6（fp32；bf16 为 0）；预算足够大时与原生逐位一致；多帧交给原生；OOM 缩小分块重试、下限报错，tiled 从未被调用。加上 VAE 之后 LoRA 部分的 `test_dtype_paths.py`（198 / 108 项）和入口测试照旧全部通过。下面 826 项的合计是加入 VAE 之前的完整运行。
* `tests/run_all.sh` 合计 826 项检查（入口 18；`MONOLOAD_EXACT=1`：dtype 108、LoRA 80、fp8 8、释放 2 + 42×3 + 22；默认路径：dtype 198、LoRA 106、fp8 8、释放 2 + 42×3 + 22），0 失败。

**CPU 上的基准参考**（`tests/bench_lora.py`，SD1.5，256×256，3 步，只能看相对比例，不代表 GPU）：

| 组合 | 每步 原生 → Monoload | 切换组合（patch）原生 → Monoload | 结果差异 |
|---|---|---|---|
| Rubber Duck（192 层） | 7.13s → 15.74s（2.21×） | 6.57s → 0.08s | 0 |
| Annalise LoCon（278 层） | 9.13s → 24.23s（2.65×） | 12.55s → 0.08s | 0 |
| 两个叠加 | 9.37s → 26.56s（2.83×） | 16.64s → 0.08s | 0 |
| 不打 LoRA | 7.85s → 8.34s | 0.05s → 0.06s | 0 |

同一台 CPU 上，ComfyUI 自带的 bypass LoRA（`--modes ...,bypass`，只作对照）：Rubber Duck 8.56s → 10.27s（1.20×），Annalise LoCon 9.10s → 13.54s（1.49×）；与原生合并的 latent 差异 max_abs 0.033 / 0.155（LoRA 本身的影响约 34）。同一轮里 Monoload 是 16.90s / 24.60s。

**要注意：** 运行时合并的每步开销，约等于每次模型调用都重做一遍原生那次性的合并。所以切换组合几乎零成本，但每一步都变慢了。上表是逐位一致路径（当时唯一的路径）。GPU 上的比例要靠 9.1 的基准来测，分析见 DESIGN.md §5.3–§5.5。

## 9. CT 700 真机验收

准备：按第 2、3 节部署，确认日志里有 `runtime LoRA merge installed`。以下命令都在 CT 700 里执行，`comfyui` 是容器名。基准脚本在服务所在的同一个容器里另起一个进程，先让服务把模型卸掉：

```bash
curl -X POST http://127.0.0.1:8188/free -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}'
```

### 9.1 正确性 + 性能：基准脚本

同一个进程里、用同一份已经加载好的模型，依次跑 `--modes` 里的几种模式，然后逐组合比较结果和耗时。脚本会自动沿用容器主进程的 ComfyUI 启动参数。

| 模式 | 内容 |
|---|---|
| `native`、`native2` | 原生（Monoload 卸下）。`native2` 是原生再跑一遍，用来量 GPU 自身的不确定性 |
| `monoload` | Monoload 默认路径（融合 fp16 addmm / 放宽合并），也就是插件装上后的实际行为 |
| `monoload-exact` | Monoload 逐位一致路径，等于设了 `MONOLOAD_EXACT=1` |
| `bypass` | ComfyUI 自带的 bypass LoRA（`comfy.sd.load_bypass_lora_for_models`），只作对照 |

```bash
# SDXL 整合包（CheckpointLoaderSimple），LoRA 放在 models/loras/
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_lora.py \
  --checkpoint <sdxl>.safetensors --width 1024 --height 1024 --steps 20 --cfg 6 --scheduler normal \
  --combo <loraA>.safetensors --combo <loraB>.safetensors --combo <loraA>.safetensors+<loraB>.safetensors --combo none \
  --modes native,monoload,monoload-exact,native2 --repeat 2

# Krea 2（UNETLoader + CLIPLoader 分开加载）
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_lora.py \
  --unet krea2_turbo_bf16.safetensors --clip qwen3vl_4b_bf16.safetensors --clip-type krea2 \
  --width 1024 --height 1024 --steps 8 --cfg 1 --sampler euler --scheduler simple \
  --combo <krea2 LoRA>.safetensors --combo none --modes native,monoload,monoload-exact,native2 --repeat 2
```

`--combo` 的写法：`a.safetensors[:模型强度[:CLIP强度]]`，多个 LoRA 用 `+` 连起来叠加，`none` 表示不打 LoRA。原生模式下 LoRA 会带着备份常驻 GTT，Krea 2 这类大模型要留意 GTT 余量。

**看这些数：**

* 每个组合一行：`keys` 是被 patch 的 UNet/TE key 数，`backups` 是备份数；`lora` / `encode` / `patch` 分别是 LoRA 节点耗时、文本编码耗时（含 TE 打 patch）、扩散模型加载耗时（原生是「写回上一组备份 + 合并新组合」，也就是切换 LoRA 组合的代价）；`step1` 和 `step` 是第一步和中位数的每步耗时，最后是当时的 GTT。
* `seconds per step` 表：每个组合一行，各模式的每步耗时并排列出，括号里是相对 `native` 的比值。`monoload`（默认路径）和 `monoload-exact`（逐位一致）分开两列。
* 汇总表（每种模式对 `native` 各一张）：每步耗时和比值、patch 耗时、encode 耗时、`max|Δ|` / `mean|Δ|`（最终 latent 的差异），以及两种模式下各自的 `effect`（这个组合相对同模式 `none` 的变化，也就是 LoRA 本身的作用）。差异要和 effect 对比着看。`bypass` 按设计就和合并路径的数值不同，它的差异只作参考。
* bypass 模式下：多个 LoRA 叠加时，脚本把各个 LoRA 的 bypass 注入合并成一个（装的时候正序、卸的时候倒序）。ComfyUI 自带的 `load_bypass_lora_for_models` 连续调用时，后一个会覆盖前一个，所以不能直接用。`keys` 列显示实际挂上的 bypass hook 数（UNet / TE）。`[BypassLoRA] Adapter key not in model state_dict: clip_...` 这类警告只是挂 UNet 时遍历到 TE 的 key 打出的噪音，TE 的 adapter 会单独挂上，脚本只在最后汇总一行。
* 最后的 layer probe：一次模型调用中，所有被 patch 层的临时拷贝、逐位一致合并、默认路径（插件实际用的融合 addmm / 放宽合并）、只用放宽合并（A，对照）各自的耗时，以及默认路径和 A 相对逐位一致结果（= 原生合并）的权重差异：`||Δw|| / ||ΔW_lora||`、换算成 `u·||W||` 的值、单元素最大差异，并标出是否在 DESIGN.md §5.5 的容差以内。另有一行 `mm-add`（`torch.mm` 后 `add_`，两个 kernel 的写法），只作诊断。
* 诊断模式 `monoload-relaxed`：默认路径关掉融合、全部走放宽合并。用来判断出图差异来自采样放大还是来自融合 addmm 的精度（DESIGN.md §5.5）。

**通过标准：**

* `monoload-exact vs native` 的 `max|Δ|` = 0。如果不是 0，看 `native2 vs native`：原生自己跑两次都有差异时，要求 monoload-exact 的差异不超过它。
* `monoload vs native`（默认路径）：layer probe 里默认路径那一行标为 `within tolerance`；latent 的 `mean|Δ|` 远小于 `effect` 的 mean（CPU 测试里约为 0.1%）。
* Monoload 那几行的 `backups` 全是 0，原生那几行在有 LoRA 时不是 0。
* 切换组合时，Monoload 的 `patch` 明显小于原生。
* **把 `seconds per step` 表、各汇总表和 layer probe 的输出发给我。**

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

VAE 解码管理（§12）每次解码打一行，形如 `[Monoload] VAE decode 1x4x270x480 -> op-level chunking (workspace 1.00 GiB): 23 of 40 conv call(s) in row blocks, attention 1 call(s) in query blocks of 2071 / 129600 tokens; memory estimate 17.91 GiB (native 91.86 GiB), 12.3s`；交给原生时是 `[Monoload] VAE decode left native: <原因>`。不应出现 `Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding`（那是原生的 tiled 回退，受管理的解码不会走到）。

### 9.7 VAE 解码：峰值、耗时、精度（`tests/bench_vae.py`）

在服务所在的容器里另起一个进程运行，自动沿用 PID 1 的 ComfyUI 启动参数。先让服务把模型卸掉（`/free`，同 9.1），否则服务进程占着的 GTT 会算进 GTT 和 cgroup 的读数里，原生解码也更容易 OOM。

```bash
curl -X POST http://127.0.0.1:8188/free -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}'
# SDXL：取 checkpoint 内置的 VAE（只加载 VAE，不加载 UNet / CLIP）
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --checkpoint waiIllustriousSDXL_v170.safetensors --profile --json /opt/ComfyUI/output/bench_vae_sdxl.json 2>&1 | tee bench_vae_sdxl.txt
# Flux ae / qwen_image_vae：models/vae 下的文件，走 VAELoader
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae ae.safetensors --profile --json /opt/ComfyUI/output/bench_vae_flux.json 2>&1 | tee bench_vae_flux.txt
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae qwen_image_vae.safetensors --profile --json /opt/ComfyUI/output/bench_vae_qwen.json 2>&1 | tee bench_vae_qwen.txt
```

参数：

| 参数 | 含义 |
|---|---|
| `--checkpoint X` / `--vae X` | VAE 来源：checkpoint 内置的 VAE（`models/checkpoints`），或 `VAELoader` 加载的文件（`models/vae`）。二选一 |
| `--res` | 输出分辨率列表 `宽x高`，默认 `1344x768,2688x1536,3840x2160`；latent 是固定种子的随机数（`--seed`，默认 0；`--latent-std`，默认 1） |
| `--latent F` | 改用 ComfyUI `SaveLatent` 节点存下的 `.latent` 文件（路径，或 `input/`、`output/` 下的文件名；可重复），缩放与 `LoadLatent` 节点相同；这时 `--res` 不用。耗时和显存与 latent 内容无关，精度用真实 latent 更有代表性 |
| `--modes` | 默认 `native,monoload,native2`，每个分辨率按这个顺序跑：`native` 原生 `VAE.decode`（Monoload 的 VAE 部分卸下）；`monoload` 第二层（逐算子分块）；`native2` 原生再跑一次，作为 GPU 自身不确定性的基线 |
| `--warm N` | 冷启动（该模式、该分辨率的第一次）之后再跑 N 次热启动，取中位数，默认 3 |
| `--workspace` | 这次运行用的 Monoload 预算（如 `512M`），默认取 `MONOLOAD_VAE_WORKSPACE` 或 1G |
| `--sample-ms` | GTT / VRAM / cgroup 采样间隔，默认 10 ms |
| `--boundary-rows` | 块边界上下各多少行算「边界附近」，默认 4 |
| `--profile` | 先对每个分辨率（或 `--profile-res` 指定的）跑一次 torch.profiler + 分配器历史，见下 |
| `--profile-modes` | 要 profile 的模式，默认 `native`；加上 `monoload` 可以看分块后峰值时刻剩下的是什么 |
| `--profile-cpp` | 分配器历史里带 C++ 栈（能直接看到 `slow_conv2d` / `im2col` 这类 ATen 函数），符号化可能要几分钟 |
| `--profile-only` | 只做 profile，不跑基准 |
| `--fp32-ref` | fp32 参照：同一个 VAE 用同一个加载方式再加载一份 fp32 的（把 `vae_dtype` 强制为 fp32），每个分辨率在 bf16 各模式之后解码一次作为「真值」，输出 `native vs fp32`、`monoload vs fp32` 两组与上面相同的误差指标（块边界行用同一次 monoload 运行的） |
| `--fp32-impl` | fp32 参照怎么解码：`both`（默认：原生 fp32 和分块 fp32 都跑；原生 fp32 显存约为 bf16 的两倍，4K 约 90 GiB，OOM 时跳过，改用分块的结果作参照；两者都成功时另外输出两者之差，作为分块在 fp32 下是否精确的旁证）、`native`（原生，OOM 时退到分块）、`monoload`（只用分块） |
| `--outliers N` | 列出 monoload 与 native 差得最多的 N 个像素（默认 10）：batch、行、列、通道，三方（fp32、native、monoload）在这些点上的像素值和 raw 值，各自离 fp32 多远，离最近的块边界几行 |
| `--outlier-threshold` | 统计 \|monoload − native\| 超过这个值（默认 0.01）的所有像素值：其中 monoload 和 native 各有多少个更接近 fp32，以及两者在这些点上对 fp32 的平均误差 |
| `--json F` | 全部结果另存一份 JSON |

**原生 OOM 的处理：** 原生解码真 OOM 时会退回 tiled，脚本把 tiled 路径换成「立即中止」，这一行标为 `OOM(tiled)`，不当作原生结果，也不参与精度比较；直接抛出的 OOM 标为 `OOM`；monoload 在最小预算下仍 OOM 时（`MonoloadVAEOOMError`）同样标为 `OOM`。脚本不会因此退出，后面的模式和分辨率照常跑。

**输出：**

* 逐行：每个模式、每个分辨率、每次运行一行：耗时；`alloc peak (+Δ)` = `torch.cuda.max_memory_allocated`（运行前 reset）及相对运行前的增量；`reserved` 同理（含分配器缓存，每次运行前 `empty_cache`）；`GTT peak +` = 后台线程每 10 ms 读 `mem_info_gtt_used`（所有 card，不写死卡号）得到的峰值增量，`VRAM +` 同理；`cgroup peak +` = `memory.peak`（内核支持时每次运行前 reset），括号里是采样 `memory.current` 得到的峰值增量；`models unloaded` = 这次运行里 `load_models_gpu` 卸载了几个模型（脚本进程里只有 VAE，所以 1 表示原生的大估算把 VAE 自己卸掉又重新加载）。monoload 行下面还有一行：Monoload 给 `load_models_gpu` 的估算（和原生 `memory_used_decode` 对比）、实际预算、OOM 重试次数、被分块的卷积调用数和总块数、最大的块工作区和最大的整层工作区、注意力每块 query 数。
* 精度（两边都成功时）：`monoload vs native`、`native2 vs native` 各一组。`raw` 是 `process_output` 之前的 decoder 原始输出，`pixels` 是 clamp 后的 fp32 像素；指标是 max|Δ|、mean|Δ|、RMSE、PSNR（像素按 [0,1] 取 20·log10(1/RMSE)，raw 按范围 2）、p99|Δ|。另外：卷积块边界映射到输出行、上下各 4 行的「边界附近」与其余行分开统计（第二层的块边界在每层分辨率上都不同，所以这一块可能覆盖相当多的行）；每个通道的均值偏移；两边的 NaN/Inf 个数。
* 最后两张汇总表：时间和内存（每个分辨率 × 模式；峰值取各次运行的最大值；`estimate` 是 Monoload 的估算，`native est` 是原生的估算）；精度。
* `--profile`：① torch.profiler（`profile_memory=True`、`record_shapes`、`with_stack`）跑一次原生解码：按自身显存分配排序的算子表，以及分配最大的单次算子调用（名字、输入形状、Python 调用栈），用来看 `aten::slow_conv2d_forward` / `im2col` 这类展开操作占多少；② 打开 CUDA 分配器的内存历史再跑一次：回放分配/释放事件，找到这次解码中已分配张量总量最大的时刻，列出那一刻存活的最大分配和各自的调用栈（只统计解码期间分配的，权重等事先存在的不算）。

**通过标准 / 看什么：**

* 峰值主因：原生的 profile 里，峰值时刻最大的存活分配是否是卷积的展开缓冲（栈落在 `conv.py:_conv_forward`，算子表里 `slow_conv2d` / `im2col` 一类分配最大）。
* `monoload` 不出现 `OOM`；`alloc Δ` 明显低于原生（4K 原生预计数十 GiB），并且不超过 `estimate`；GTT 和 cgroup 的增量同步下降。
* 精度：`monoload vs native` 没有 NaN/Inf；PSNR 足够高（初步目标：像素 RMSE ≤ 1e-3，即 PSNR ≥ 60 dB，max|Δ| ≤ 1e-2）；边界附近与其余行的误差同一量级（没有系统性的边界误差）；通道均值偏移接近 0。`native2 vs native` 是 GPU 本身的基线。
* 耗时：monoload 的热启动与原生相比慢多少（算量约 1 倍，多出来的是块的启动和拷贝开销）。
* **把三份完整输出（`bench_vae_*.txt`）和 JSON 发给我。**

**fp32 参照（判断 monoload 的 max|Δ| 离群点是 bf16 固有噪声，还是分块额外引入的；CT 700 的结果见 §10.1：是 bf16 噪声）：** 峰值和耗时第一轮已经测过，这一轮只看精度，所以去掉 `native2`、不跑热启动：

```bash
curl -X POST http://127.0.0.1:8188/free -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}'
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --checkpoint waiIllustriousSDXL_v170.safetensors --modes native,monoload --warm 0 --fp32-ref \
  --json /opt/ComfyUI/output/bench_vae_fp32_sdxl.json 2>&1 | tee bench_vae_fp32_sdxl.txt
# Flux / Qwen：把 --checkpoint ... 换成 --vae ae.safetensors / --vae qwen_image_vae.safetensors，输出文件名相应改成 _flux / _qwen
```

看什么：

* `native vs fp32` 和 `monoload vs fp32` 两行（PSNR、RMSE、max、p99）是否相当。相当，就说明 monoload 和 native 只是各自带着 bf16 的舍入噪声，离群点是噪声在两个结果里落到了不同位置；monoload 明显更差，才说明分块额外引入了误差。
* `outliers` 表：前 10 个点上 `|n-fp32|` 和 `|m-fp32|` 哪个大，`bdist` 是否集中在 0（块边界上）；下面一行统计里 monoload 和 native 各有多少个点更接近 fp32。
* `fp32 chunked vs fp32 native`（只有原生 fp32 放得下的分辨率才有，通常只有 1344×768）：分块在 fp32 下应当只差 1e-6 量级。
* 同样把三份输出和 JSON 发给我。

## 10. 真机验收结果（CT 700，2026-10）

* **9.1 第一轮**（WAI v17 SDXL，1344×768，20 步，CFG 6）。当时插件只有逐位一致路径，这一条里的 Monoload 数字都是逐位一致路径，也就是现在的 `MONOLOAD_EXACT=1`，不是现在的默认路径：
  * `monoload vs native`（逐位一致路径）的 max|Δ| 全部为 0；Monoload 的 backups 全部为 0，原生是 788 / 986 / 1052。
  * 切换组合的 patch 耗时：原生 0.43–0.65s，Monoload 0.09–0.11s。
  * 每步耗时：原生约 0.64s，Monoload（逐位一致路径）0.85–1.06s（1.33× / 1.39× / 叠加 1.66×）。
  * GTT：原生 13.9G，Monoload 8.6G。`native2 vs native` 全部为 0，GPU 计算是确定的。
  * 那一轮 bypass 叠加组合的数据无效（只挂上了第二个 LoRA），已在脚本里修正。
* **9.4 释放：**
  * GTT 依次为 7.20 → 7.37 → 7.37，cgroup 一直是 1.40；不带 LoRA 的第三个工作流，出图与重启后的参照图 identical。
  * 发现的问题：Smooth Booster 没有 TE key，`LoraLoader` 产出的 CLIP clone 不带 patch，却换了 uuid。释放后它被回收，模型上的 uuid 还是它的，导致下一个工作流的 CLIP 又 `load()` 了一次。已修复（DESIGN.md §7.3 第 5 步），并加了只改 UNet 的 LoRA 的测试。
* **9.1 第二轮**（`bench_sdxl_v2`，更新后的基准脚本，同样设置）：
  * 每步：原生 0.638 / 0.640 / 0.643s；Monoload（逐位一致）0.891 / 0.855 / 1.064s（1.40× / 1.34× / 1.66×），与原生 max|Δ| 全部为 0；bypass（叠加已修正）0.870 / 0.799 / 1.024s。
  * layer probe：临时拷贝 0.038s、逐位一致合并 0.221s、放宽合并 0.117s、融合 addmm 0.040s。与原生合并的权重差异：放宽 3.56e-5、融合 2.54e-3（‖Δw‖/‖LoRA 改动‖），单元素最大差 1 个 fp16 ulp。
  * 据此决定：默认用融合 addmm（C），C 不支持的类型用放宽合并（A），`MONOLOAD_EXACT=1` 切回逐位一致（§5.1，DESIGN.md §5.4）。
* **9.1 第三轮**（`bench_sdxl_v3`，默认路径已实现）：
  * 每步：原生 0.639 / 0.645 / 0.640s；默认路径 0.723 / 0.710 / 0.758s（1.13× / 1.10× / 1.18×）；逐位一致 0.891 / 0.855 / 1.065s（1.39× / 1.33× / 1.67×）。不打 LoRA 时三者相同。
  * 逐位一致路径与原生 max|Δ| 全部为 0；`native2` 与原生全部为 0。
  * layer probe：默认路径合并 0.052s（+ 拷贝 0.038s），与原生合并的权重差异 2.54e-3，在容差 0.0308 以内；关掉 fp16 GEMM 的降精度累加后耗时和误差都不变。
  * 默认路径与原生的 latent：mean|Δ| 0.556 / 0.189 / 0.519，LoRA 本身的作用 mean 3.81 / 4.71 / 4.77（见 §5.1 和 DESIGN.md §5.5）。
* **9.1 第四轮**（`bench_sdxl_v4`，同样设置，euler + normal，模式 native / monoload / monoload-relaxed）：
  * 可复现：默认路径与原生的 latent mean|Δ| 仍是 0.556 / 0.189 / 0.519，effect 与第三轮一位不差。
  * 每步：默认路径（C）1.13× / 1.11× / 1.19×；只用放宽合并（A）1.23× / 1.19× / 1.38×；不打 LoRA 1.01×。GTT：Monoload 8.39G（A 8.52G），原生 13.90G。backups 全部为 0。
  * A 与原生的 latent mean|Δ| 是 0.363 / 0.0768 / 0.438，为 C 的 65% / 41% / 84%。A 的权重误差比 C 小 70 倍，latent 差异只小 1.2–2.5 倍：出图差异主要来自采样放大，已确认（DESIGN.md §5.5）。A 的数值就是原生 lowvram 的数值。
  * layer probe 的 `mm-add`（`torch.mm` 后 `add_`）：权重误差与 A 完全相同（3.56e-5，max 6.1e-5），合并 0.084s（C 0.052s，A 0.118s）。所以 C 多出来的权重误差来自 hipBLASLt `addmm_`（beta=1）内部的累加或 epilogue。
* **同种子出图对比**（API 直接提交，WAI v17 + Smooth Booster 1.0/1.0，seed 42，设置同上；D = 默认路径，E = `MONOLOAD_EXACT=1`，即与原生逐位一致）：
  * `compare_images`：97.5% 的像素不同，mean abs diff 5.29，max 255。
  * 构图和风格一致，只是细节位置有偏移，肉眼看不出画质差别。
* **决定：** 默认路径维持 C，`MONOLOAD_EXACT=1` 继续作为与原生逐位一致的选项；`mm-add` 不采用（每步多约 5%，出图差异只降到 A 的水平，类别不变）。
* **9.5 fp8**（WAI 生成的 fp8 UNet）：采样时挂 LoRA 只多出 0.58G GTT，没有备份，没有报错，出图正常。
* **9.6 日志：** 没有 traceback / 不支持 / 内部错误，没有 OOM。

### 10.1 VAE 解码第一阶段（`bench_vae_*`，2026-10，af9abc6）

设置：`--gpu-only --bf16-vae`，`cudnn.enabled = False`，split 注意力；随机 latent（种子 0）；每档 1 次冷启动 + 3 次热启动；先 `/free`，脚本进程里只有这一个 VAE。完整分析见 DESIGN.md §9.12。

* **峰值主因已确认：Slow2d 的 im2col 展开缓冲。** 原生解码峰值时刻最大的存活分配是最后几层 3×3 卷积的 columns：SDXL / Flux 依次 4.43 / 17.72 / 35.60 GiB（占峰值 5.68 / 22.70 / 45.61 GiB 的 78%），Qwen 3.32 / 13.29 / 26.70 GiB（峰值 3.98 / 15.92 / 31.99 GiB），与调研估算一致。原生 4K 每次都先在 HIPCachingAllocator 报 OOM（申请 38.2 GB，free 只有 11.8 GB），清空缓存重试后才成功，已经在边缘。
* **显存（GiB；原生 → monoload；1344×768 / 2688×1536 / 3840×2160）：**

| VAE | alloc 增量 | GTT 峰值增量 | monoload 估算 | 原生估算 |
|---|---|---|---|---|
| SDXL | 5.68→2.41 / 22.70→6.15 / 45.61→11.17 | 8.36→3.72 / 42.84→9.25 / 52.48→14.86 | 3.98 / 9.92 / 17.91 | 11.43 / 45.73 / 91.86 |
| Flux `ae` | 5.68→2.41 / 22.71→6.15 / 45.61→11.18 | 8.36→3.72 / 42.84→9.24 / 52.50→14.89 | 3.98 / 9.92 / 17.92 | 11.43 / 45.73 / 91.86 |
| `qwen_image_vae` | 3.98→1.82 / 15.92→3.80 / 31.99→6.46 | 6.10→2.12 / 29.37→5.05 / 59.14→9.59 | 3.49 / 7.95 / 13.96 | 4.23 / 16.92 / 33.99 |

  4K 峰值降到原来的 1/4～1/6。**估算是有效上界：** 9 个场景里 monoload 的估算都大于实测的 reserved / GTT 增量（例如 SDXL 4K 估 17.9、实测 14.9）。`unload` 全为 0，cgroup 采样增量 ≤ 0.1 GiB（GTT 不计入容器 cgroup）。
* **耗时（热启动中位数，秒；原生 → monoload）：**

| VAE | 1344×768 | 2688×1536 | 3840×2160 |
|---|---|---|---|
| SDXL | 0.878 → 0.908 | 4.720 → 4.106 | 12.157 → 10.184 |
| Flux `ae` | 0.872 → 0.906 | 4.741 → 4.102 | 12.142 → 10.108 |
| `qwen_image_vae` | 0.664 → 0.670 | 3.436 → 3.036 | 8.064 → 7.338 |

  低分辨率慢 1–4%，高分辨率反而快 9–17%：原生要现场 hipMalloc 几十 GiB 的大块（Flux 4K 的 profile 里 hipMalloc 占 4.25 s CPU 时间，4K 还多一次 OOM → 清缓存 → 重试），分块后每块 ≤ 1 GiB，可以复用缓存。注意 bench 每次运行前都 `empty_cache`，原生的这部分开销在服务里会因缓存状态而不同（缓存留着时又会多占几十 GiB GTT）。
* **精度（monoload vs native，clamp 后的像素）：** `native2 vs native` 全部为 0，GPU 是确定的。

| VAE | PSNR (dB) | RMSE | p99 | max\|Δ\|（块边界附近 / 其余行） |
|---|---|---|---|---|
| SDXL | 68.3 / 62.5 / 61.2 | 3.8e-4 / 7.5e-4 / 8.7e-4 | 0.00098 / 0.0024 / 0.0025 | 0.0024 / 0.0386；0.0108 / 0.0959；0.0362 / 0.0528 |
| Flux `ae` | 67.5 / 64.9 / 61.5 | 4.2e-4 / 5.7e-4 / 8.4e-4 | 0.0020 / 0.0020 / 0.0024 | 0.0039 / 0.0083；0.0083 / 0.0208；0.0283 / 0.0464 |
| `qwen_image_vae` | 63.5 / 60.5 / 60.3 | 6.7e-4 / 9.4e-4 / 9.6e-4 | 0.0020 / 0.0024 / 0.0024 | 0.0059 / 0.0107；0.0217 / 0.0337；0.0176 / 0.0547 |

  RMSE 全部 ≤ 9.6e-4（PSNR ≥ 60 dB），通道均值偏移 ≤ 2e-4，没有 NaN/Inf。**没有接缝：** 块边界附近的 RMSE 与其余行相同（例如 SDXL 4K 8.6e-4 对 8.7e-4），最大误差也不在边界上。像素 max|Δ| 是零星单点，为 0.008–0.096，超过调研里 max ≤ 1e-2 的研发目标；p99 只有 1–2.5 个 bf16 ulp 的量级。这些离群点后来用 fp32 参照确认是 bf16 固有噪声（见下）。
* **fp32 参照（`bench_vae_fp32_*`，76e56a1，`--modes native,monoload --warm 0 --fp32-ref`）：离群点是 bf16 固有噪声，不是分块引入的。**

| VAE | 输出 | fp32 分块 vs fp32 整图：max\|Δ\| / PSNR | 原生(bf16) vs fp32：RMSE / PSNR / max | monoload(bf16) vs fp32：RMSE / PSNR / max | \|m−n\| > 0.01 的像素里更接近 fp32 的（monoload / 原生） |
|---|---|---|---|---|---|
| SDXL | 1344×768 | 7.8e-6 / 133.5 dB | 1.14e-3 / 58.8 / 0.050 | 1.14e-3 / 58.9 / 0.049 | 34 / 36 |
| SDXL | 2688×1536 | 9.7e-6 / 133.0 dB | 8.91e-4 / 61.0 / 0.067 | 8.25e-4 / 61.7 / 0.072 | 180 / 188 |
| SDXL | 3840×2160 | （原生 fp32 OOM） | 1.08e-3 / 59.3 / 0.058 | 1.06e-3 / 59.5 / 0.064 | 690 / 441 |
| Flux `ae` | 1344×768 | 2.8e-6 / 138.8 dB | 1.28e-3 / 57.8 / 0.054 | 1.28e-3 / 57.8 / 0.054 | （没有离群点） |
| Flux `ae` | 2688×1536 | 7.6e-6 / 133.9 dB | 1.07e-3 / 59.4 / 0.051 | 1.07e-3 / 59.4 / 0.054 | 12 / 3 |
| Flux `ae` | 3840×2160 | （原生 fp32 OOM） | 1.07e-3 / 59.4 / 0.067 | 1.07e-3 / 59.4 / 0.077 | 302 / 354 |
| `qwen_image_vae` | 1344×768 | 2.8e-6 / 138.7 dB | 9.72e-4 / 60.2 / 0.017 | 9.73e-4 / 60.2 / 0.023 | 1 / 1 |
| `qwen_image_vae` | 2688×1536 | 7.6e-6 / 136.1 dB | 9.69e-4 / 60.3 / 0.030 | 9.69e-4 / 60.3 / 0.022 | 128 / 105 |
| `qwen_image_vae` | 3840×2160 | （原生 fp32 OOM） | 9.71e-4 / 60.3 / 0.028 | 9.72e-4 / 60.2 / 0.035 | 239 / 288 |

  * **分块在 fp32 下与整图等价**：fp32 分块 vs fp32 整图，像素 max|Δ| 2.8e-6～9.7e-6，PSNR 133～139 dB（1344 和 2688 两档；4K 的原生 fp32 放不下，退回 tiled，bench 正确标为 `OOM(tiled)`，4K 改用分块 fp32 作参照）。
  * **和 fp32 真值比，monoload 与原生同样准**：两者对 fp32 的 RMSE / PSNR / p99 几乎相同（例如 SDXL 4K 原生 1.08e-3 / 59.3 dB，monoload 1.06e-3 / 59.5 dB）。原生自己对 fp32 的 max 误差就有 0.017–0.067，同样超过 1e-2，所以 max ≤ 1e-2 对 bf16 解码本身就不现实。
  * |monoload − native| > 0.01 的像素里，monoload 和原生各有约一半更接近 fp32；离群点离块边界的距离（`bdist`）不集中在 0。SDXL 1344 和 2688 最大的离群点都在图像最右边几列（W 方向根本不分块），而且在这些点上原生离 fp32 更远。
  * 显存：Qwen 原生 fp32 在 4K 时 GTT 一度到 60.3 GiB（上限 62.5）后 OOM，monoload fp32 是 20.4 GiB。
  * **原生估算会挤掉其他模型（实测）**：这一轮进程里同时加载了 bf16 和 fp32 两份 VAE。原生的过大估算（bf16 4K 91.86 GiB，fp32 2688 91.45 GiB、4K 183.72 GiB；Qwen fp32 4K 67.98 GiB）让 `load_models_gpu` 把另一份 VAE 卸载了（原生这几行 `unload = 1`，下一次运行时再重新加载），monoload 的行全部是 0。服务里常驻 UNet / CLIP 时，被挤掉的就是它们。
* 日志计时：第一轮发现插件日志里的解码秒数没有等 GPU 同步、比实际短（4K 显示 0.84 s，实际 10.2 s），已修复：计时前后各同步一次设备。
* **结论：第一阶段验收通过。** 峰值主因确认；4K 峰值降到原来的 1/4～1/6；速度不降反升；没有接缝；估算是有效上界；精度与原生同为 bf16 的水平，分块在 fp32 下与整图等价。

## 11. 仓库结构

```
__init__.py                 ComfyUI 入口：按 MONOLOAD_DISABLE / MONOLOAD_KEEP_LORA / MONOLOAD_DISABLE_VAE / MONOLOAD_EXACT
                            调用 hotpatch.install()、release.install()、vae.install()
monoload/hotpatch.py        运行时合并的全部实现（MonoloadRuntimePatch + ModelPatcher 方法替换；默认的融合 addmm / 放宽合并、
                            MONOLOAD_EXACT=1 的逐位一致路径、量化层在反量化临时权重上的合并）
monoload/release.py         每个 prompt 结束后释放 LoRA（包装 PromptExecutor.execute_async）
monoload/errors.py          报错类型
monoload/comfy_env.py       在独立进程里按指定参数启动 ComfyUI 环境（测试、基准用）
monoload/vae.py             VAE 解码管理入口（包装 VAE.decode：内存估算、逐张、OOM 缩小分块重试、覆盖范围；第一层的接口 STRIPE_ADAPTERS）
monoload/vae_ops.py         逐算子分块（卷积按输出行、注意力按 query；受管理期间的实例属性替换）
tests/                      测试和基准脚本（见第 8、9 节；VAE：test_vae.py、bench_vae.py、make_synthetic_vaes.py）
tools/watch_mem.sh          GTT / cgroup 内存监视
tools/compare_images.py     两张图逐像素比较
docs/DESIGN.md              设计说明
```

## 12. VAE 解码降峰值（第一阶段）

### 12.1 要解决的问题

调研（DESIGN.md §9.1）的结论，待真机数据确认：

* 禁用 MIOpen（ComfyUI 在 AMD 上设 `torch.backends.cudnn.enabled = False`）后，3×3 卷积走 Slow2d：先 im2col 展开再 GEMM。4K 输出时最后一层 Conv256→256 的展开缓冲约 35.6 GiB，SDXL 原版 4K 解码峰值估计约 45 GiB，Qwen 约 31 GiB。**峰值主要来自展开缓冲，不是激活本身。**
* mid block 的全局注意力在 4K 时 N ≈ 13 万，一张 bf16 分数矩阵约 31 GiB；原生按空闲内存自动切片，峰值不固定。
* 原生用 `memory_used_decode` 估算显存：AMD 上 SDXL 4K 估约 92 GiB，并按它调用 `load_models_gpu`（`--gpu-only` 下也可能卸载别的模型）、切分 batch。
* 真 OOM 后原生自动退回 tiled 解码：每个 tile 各自做 GroupNorm 和注意力，结果和整图不等价。

### 12.2 原理

三层设计（DESIGN.md §9.2），本阶段实现第二层和兜底：

* **第二层：逐算子分块。** decoder 自己的 forward 原样运行，只在受管理的解码期间、只在这个 VAE 的模块上替换两类重算子：
  * 卷积（Conv2d、Conv3d，含 comfy.ops 各变体）：整层的 im2col 展开估算超过预算时，按输出行分块，每块只取它需要的输入行（含上下 halo），块与块之间用真实的相邻行，zero padding 只出现在图像真实的上下边缘；输出预先分配好，逐块写入。接在 `_conv_forward` 上，所以 `cast_bias_weight` / weight_function（包括 Monoload 的运行时 LoRA）照常生效。
  * 注意力：按 query 分块，K/V 保持完整，每个 query 仍对全图做 softmax；每块的运算与原生实现相同。
  * GroupNorm、上采样等其余算子保持原生。整图语义不变，算量约 1 倍。
* **管理入口（包装 `VAE.decode`）：** 自己按形状估算内存上界交给 `load_models_gpu`（SDXL 4K 约 17.9 GiB，原生约 92 GiB，估算方法见 DESIGN.md §9.4）；batch 逐张；输出的设备、dtype、`process_output`、NHWC 与原生完全一致。
* **OOM：** 只缩小分块（预算减半）重试，降到 64 MiB 仍 OOM 就抛 `MonoloadVAEOOMError`，**绝不调用 tiled 解码**。
* **兜底：** 不在覆盖范围内的解码走原生整图解码，打一条日志。
* **第一层（条带解码，后两阶段）：** 认得的结构按输出条带倒推重算、GroupNorm 全局统计逐层求得，进一步去掉高分辨率激活。第二阶段做 Qwen，第三阶段做 SDXL / Flux。

### 12.3 开关

| 设置 | 效果 |
|---|---|
| 默认 | 启用，预算 1 GiB |
| `MONOLOAD_VAE_WORKSPACE=512M` | 预算（卷积每块的展开缓冲、注意力每块的分数矩阵 + softmax 结果）。写法 `1G`、`512M`、`768`（纯数字按 MiB）。越小峰值越低、块越多 |
| `MONOLOAD_DISABLE_VAE=1` | 只关 VAE 部分，LoRA 部分不受影响 |
| `MONOLOAD_EXACT=1` | VAE 走原生（分块会改变 GEMM 形状，不保证逐位一致） |
| `MONOLOAD_DISABLE=1` | 什么都不装 |

启动日志写明状态和预算：`[Monoload] VAE decode managed: ... workspace 1.00 GiB ...`，或 `[Monoload] VAE decode: native (...)`。

### 12.4 覆盖范围

* 接管：图像解码，即 4D latent（SDXL、Flux `ae` 等 2D VAE；给 2D VAE 的 5D latent 与原生一样取第一帧），以及 T=1 的 5D latent（`qwen_image_vae` / Wan 2.1 VAE 等）。
* 交给原生（打日志）：多帧视频 latent——**这只是第一阶段暂时不做，不是永远不做**；1D / 音频 latent；自己往预分配输出里写的 VAE（`comfy_has_chunked_io`，如 LTX）。
* 不经过 `VAE.decode` 的路径保持原生：用户显式使用的 `VAEDecodeTiled` 节点 / `VAE.decode_tiled`（用户自己选了 tiled 的语义），以及直接调用 `first_stage_model.decode` 的第三方代码。
* 第二层不挑结构：任何 VAE 的 `torch.nn.Conv2d/Conv3d` 和 ComfyUI 自带的三种 VAE 注意力（split / pytorch / xformers）都会被分块；不认识的注意力实现保持原样（日志里列出）。

### 12.5 限制

* 与原生不逐位一致（块的 GEMM 形状不同）。CT 700（bf16）上与原生的像素 RMSE ≤ 9.6e-4、PSNR ≥ 60 dB，零星单点的 max|Δ| 到 0.096；fp32 参照确认这是 bf16 固有噪声：对 fp32 真值，monoload 与原生同样准，分块在 fp32 下与整图只差 1e-5 以内（§10.1）。
* 第二层只去掉展开缓冲和分数矩阵，不减少激活本身：4K 时 SDXL 全尺寸的一张激活约 4 GiB，实测峰值 alloc 11.2 GiB、GTT 14.9 GiB（§10.1）。
* 块多了会有额外的启动和拷贝开销（实测 1344×768 慢 1–4%，更高分辨率反而快 9–17%，见 §10.1）。
* 每个 VAE 第一次受管理解码时，会先用 8×8 的 latent 跑一次小解码来量激活大小（结果缓存，RNG 状态不变）。

### 12.6 第一阶段的状态

* 已完成：管理入口、第二层（卷积行分块 + 注意力 query 分块）、OOM 策略、开关、CPU 测试（`tests/test_vae.py`）、真机测量脚本（`tests/bench_vae.py`，9.7）。
* 已在 CT 700 上测完（§10.1）：峰值主因是 im2col 展开缓冲；4K 峰值降到原来的 1/4～1/6（SDXL / Flux alloc 45.6 → 11.2 GiB，Qwen 32.0 → 6.5 GiB）；高分辨率反而更快；PSNR ≥ 60 dB，没有接缝；估算是有效上界。
* 精度已用 fp32 参照确认：max|Δ| 离群点是 bf16 固有噪声，分块在 fp32 下与整图等价（§10.1）。**第一阶段验收通过。**
* 下一步：第二、三阶段（第一层条带解码）的优先级和预算待定；第二层之后 4K 峰值已在 15 GiB 以内，第一层主要的收益是进一步去掉全尺寸激活。
* 实现细节、行区间推导和后两阶段计划见 DESIGN.md §9。
