# Monoload 交接说明（第二阶段结束时，2026-10）

给下一个对话用：当前做到了哪里、必须遵守的规矩、第三阶段（LDM decoder）从哪里下手。细节分别在 README.md（使用、真机结果、bench 命令）和 docs/DESIGN.md（§9：VAE 解码降峰值的设计）。

## 1. 当前状态

仓库 github.com/sketchain/Monoload，开发分支 `dev`（不碰 `main`）。ComfyUI 插件，两部分功能，互相独立：

* **LoRA 运行时合并**（早期工作，本轮没动）：所有 `ModelPatcher` 走运行时合并，不做原地 LoRA、不留权重备份；每个 prompt 结束后释放 LoRA；`MONOLOAD_EXACT=1` 时与原生逐位一致。**这部分的行为不能改。**
* **VAE 解码降峰值**（本轮的工作）：包装 `comfy.sd.VAE.decode`。
  * 第二层（逐算子分块，所有 VAE）：卷积按输出行分块（im2col 缓冲 ≤ 工作区），注意力按 query 分块；整图语义不变。
  * 第一层（条带解码，目前只有 Wan 2.1 VAE 单帧 = `qwen_image_vae`）：前缀在 H/8 上整图算出存档，之后按输出行条带倒推重算；整个解码在一个缓存分配器 arena 里运行。
  * 管理入口：自己的内存估算交给 `load_models_gpu`、batch 逐张、OOM 缩块重试、从不调用 tiled。

CT 700（Strix Halo，gfx1151，统一内存 62.5 GiB GTT）上 4K（3840×2160）解码的 GTT 增量：

| VAE | 原生 | 第二层 | 第一层 |
|---|---|---|---|
| SDXL / Flux `ae` | 52.5 GiB | 14.9 GiB | （第三阶段） |
| `qwen_image_vae` | 59.2 GiB | 9.6 GiB | 1.13 GiB（估算 1.52，8.24 s，原生 7.89 s） |

第二阶段已验收（85a5c6f，README §10.2）。还没做的：

1. workspace 实验（README 9.7 的 J、K；DESIGN §9.13.11）：数据回来之前不改默认值。
2. 第三阶段：SDXL / Flux 的 LDM decoder 走第一层（下面第 5 节）。
3. 多帧 Wan 视频 latent 目前交给原生（不是永远不做）。

## 2. 提交记录（`dev` 上的合并提交 ← 功能提交）

| 阶段 | 合并 | 功能提交 | 内容 |
|---|---|---|---|
| VAE 第一阶段 | af9abc6 | 89758a2 | 管理入口、第二层逐算子分块、开关、`bench_vae.py`、测试 |
| fp32 参照 | 76e56a1 | f08c986 | bench 的 fp32 参照和离群点列表；第一阶段真机结果 |
| 第一阶段验收 | 191f59b | e3ab1e6 | 日志计时前后同步设备；fp32 结论，第一阶段通过 |
| 第二阶段 | 4e54d20 | 6a50513 | 第一层：Wan 2.1 单帧条带解码、自检、预算 |
| 第二阶段调整 | 725a010 | 0826ff9 | 默认条带策略、按 forward 数的内存模型、单帧 Conv3d 改走 conv2d |
| 第二阶段第三轮 | 85a5c6f | 45a0b6b | arena、估算 = arena + largest、分块卷积先分配输出、`tests/alloc_sim.py` |
| 收尾 | 本文件所在的合并 | — | 实测值写进文档、workspace 实验命令（bench `-w<MiB>`）、本交接说明 |

LoRA 部分的历史：3c473fa … 5bfbc8e（v1 文件格式 → v2 全局运行时合并 → 释放、fp8、默认融合 addmm 路径），见 `git log --first-parent dev`。

## 3. 必须遵守的规矩

**行为：**

* **绝不退回 tiled**：不调用 `decode_tiled_`，也不让原生的 OOM → tiled 回退发生在受管理的解码里（tile 局部的 GroupNorm / 注意力与整图不等价）。用户显式用 `VAEDecodeTiled` 节点时保持原生。
* **OOM：缩块缩条带重试，最后抛 `MonoloadVAEOOMError`**。第二层：工作区减半到 64 MiB；第一层：条带高度和工作区一起减半到 8 行 / 64 MiB。第一层 OOM 不退回第二层（第二层峰值更高）。报错信息是中文，写明怎么办。
* **自己算的估算传给 `load_models_gpu`**（`memory_required=`），不用原生的 `memory_used_decode`（AMD 上 SDXL 4K 约 92 GiB，会挤掉其他模型）。估算必须是 reserved（GTT）的上界；第一层的上界由 `tests/alloc_sim.py` 在 384 个计划上验证。
* **预算可以用环境变量覆盖，显式设置时严格执行**：`MONOLOAD_VAE_BUDGET` 设了就取估算不超过它的最高条带，放不下直接抛 `MonoloadError`（写明需要多少），不悄悄放宽；`MONOLOAD_VAE_STRIPE_ROWS` 强制条带高度（优先于默认策略和预算）。其他开关：`MONOLOAD_VAE_WORKSPACE`、`MONOLOAD_DISABLE_VAE_STRIPE=1`、`MONOLOAD_DISABLE_VAE=1`、`MONOLOAD_EXACT=1`（VAE 走原生）、`MONOLOAD_DISABLE=1`。
* **原模型的模块实例原样调用**（条带里是调用在行切片上），`comfy.ops` 的 cast / `weight_function`（包括 Monoload 的运行时 LoRA）照常生效；替换只做在实例属性上、只在受管理的解码期间，退出时恢复（`OpChunking`）。
* **按真实结构识别，不看文件名**；不认识就走第二层并打一条日志说明原因。
* **第一层每种结构在本进程第一次使用前做 fp32 自检**（与原生整图解码比，相对误差 ≤ 1e-4），不通过就对这种结构禁用第一层、醒目警告、走第二层。
* **精度**：不承诺与原生逐位一致（GEMM 形状不同），但 fp32 下与整图等价（≤ 1e-5），bf16 下对 fp32 真值与原生同一水平；用 bench 的 `--fp32-ref` 判断离群点是不是 bf16 噪声。
* **不改 LoRA 部分的行为。**

**流程：**

* 从 `dev` 开 feature 分支，做完合回 `dev`（`git merge --no-ff`），push `dev`。不碰 `main`，不开 PR（除非用户要求）。
* **只跑小测试，不跑 `tests/run_all.sh`。** VAE 相关的小测试：`tests/test_vae_stripe.py`、`tests/test_vae.py`、`tests/test_entry.py`（几种开关组合）、`tests/test_dtype_paths.py`（默认和 `MONOLOAD_EXACT=1`）。
* **CT 700 上的 bench 代码由我们写，容器操作（拉代码、重建镜像、`/free`、跑命令、切开关）由用户做。** 报告里不写容器层面的步骤，只给 `docker exec ... python tests/bench_vae.py ...` 命令和要看的指标。
* 文档和报告用中文。报告写明合并提交、小测试结果、与要求不同之处及原因。冲突时以用户的最新要求为准。
* 提交信息结尾加 Co-Authored-By / Claude-Session 两行（见会话里的 attribution 提示）。

## 4. 环境与工具

* **锁定镜像**：`docker.io/kyuz0/amd-strix-halo-comfyui@sha256:384aa1fecef6a841832e0d5552949977330308d8c25e212a94f5e8dfcc061cae`，ComfyUI 0.31.0 / 62b3c94，torch 2.14.0a0 + ROCm 7.15。云端容器里 docker daemon 可能没起来：`sudo dockerd > /tmp/dockerd.log 2>&1 &`。
* **跑测试**：`MODELS=<目录> tests/docker_run.sh python tests/test_vae_stripe.py`（把仓库只读挂进镜像的 custom_nodes）。VAE 测试不需要真实模型；需要 VAE 文件的（bench 的 CPU 冒烟）用 `python tests/make_synthetic_vaes.py OUT_DIR [--full]` 生成随机权重的 SDXL / Flux / Wan 结构文件（`--full` 是真实宽度），放在 `$MODELS/vae`。
* **CT 700 的事实**：`--gpu-only --bf16-vae`；ComfyUI 在 AMD 上设 `cudnn.enabled = False`，所以 4D 卷积走 Slow2d（im2col + GEMM，im2col 缓冲就是原生峰值的主因），5D 卷积走 SlowDilated3d；VAE 注意力是 split（`normal_attention`）；统一内存，显存占用看 GTT。
* **`tests/bench_vae.py`**（README 9.7）：`--checkpoint` / `--vae`；`--res`；`--modes native,monoload,monoload-l2,monoload-r<N>,native2`，模式名加 `-w<MiB>` 只对该模式改工作区；`--stripe-rows`；`--warm`；`--fp32-ref`；`--profile-only --profile-modes ...`；`--no-arena`；`--json`。输出每次运行的 alloc / reserved / GTT / 耗时、估算、精度（对原生、对 fp32）、条带 / 分块边界附近的误差、离群点。
* **`tests/alloc_sim.py`**（DESIGN §9.13.10）：在 meta 设备上跑全尺寸 Wan decoder，按 PyTorch 缓存分配器的规则重放分配 / 释放，复现了 24 个 CT 700 读数（≤ 0.02 GiB）。`python tests/alloc_sim.py --res 3840x2160 --rows 32,144,512 [--version v1|v2] [--peak] [--segments]`。改了内存相关的代码先用它看，再让用户上真机。

## 5. 代码地图

* `monoload/vae.py`：入口和策略。`install()` 包 `VAE.decode` → `_decode`（覆盖范围判断，不管的交给原生）→ `_managed_decode`（选层、自检）→ `_decode_layer1`（`choose_plan`、`load_models_gpu`、OOM 循环、日志、`last_decode()`）或 `_decode_layer2`（形状探测、`estimate()`、OOM 循环）。设置读环境变量（`workspace()`、`budget()`、`layer1_workspace()`、`stripe_rows()`）。`STRIPE_ADAPTERS` 是第一层适配器的注册表。
* `monoload/vae_ops.py`：第二层的引擎，完全通用。`OpChunking`（实例级替换 `_conv_forward` 和 `optimized_attention`）、`_ConvChunker`（行分块、只在真实边缘补零、先分配输出、单帧 Conv3d 改走 conv2d）、三种注意力的 query 分块、`OpStats`。
* `monoload/vae_stripe.py`：第一层，通用部分和 Wan 专用部分混在一起（见下表）。
* `tests/test_vae_stripe.py`（69 项）、`tests/test_vae.py`（131 项）、`tests/alloc_sim.py`、`tests/bench_vae.py`、`tests/make_synthetic_vaes.py`。

## 6. 第三阶段入口：LDM decoder（SDXL / Flux）

只做分析，没有实现。

### 6.1 结构和难点

`comfy.ldm.modules.diffusionmodules.model.Decoder`（SDXL 的 AutoencoderKL 和 Flux 的 `ae` 都是它，ch 128、ch_mult [1,2,4,4]、num_res_blocks 2，latent 4 / 16 通道；4D 张量 [B, C, H, W]）：

```
conv_in → mid.block_1 (ResnetBlock) → mid.attn_1 (AttnBlock，全图注意力) → mid.block_2
→ up[3]: 3 × ResnetBlock (512 ch, H/8) → Upsample (nearest ×2 + 3×3 conv)
→ up[2]: 3 × ResnetBlock (512, H/4) → Upsample
→ up[1]: 3 × ResnetBlock (256, H/2) → Upsample
→ up[0]: 3 × ResnetBlock (128, H)
→ norm_out (GroupNorm 32) → swish → conv_out
ResnetBlock: GroupNorm → swish → conv3×3 → GroupNorm → swish → dropout → conv3×3，+ x（通道变时 nin_shortcut 1×1）
```

和 Wan 的区别只有一个，但是根本性的：**归一化是 GroupNorm（32 组，每组在整张图上求均值和方差），不是逐位置的 RMS norm**。条带里任何一个 GroupNorm 都需要它的输入在全图上的统计量，而这个输入又依赖前面所有 GroupNorm 的统计量，所以不能像 Wan 那样「切片上原样调用模块」。按 Wan 的拆法（前缀到 H/8 的 3 个残差块为止），条带部分有 3 级 × 3 个残差块 × 2 + `norm_out` = 19 个 GroupNorm，形成一条依赖链。

其余部分与 Wan 同构：halo（残差块 2、3×3 卷积 1、上采样 `[⌊a/2⌋, ⌈b/2⌉)`）、存档（H/8，512 通道，4K bf16 时约 133 MB）、前缀里的全图注意力（继续用第二层的 query 分块）都能直接沿用。

### 6.2 现有代码里哪些是通用引擎、哪些是 Wan 专用

| 位置 | 通用（引擎） | Wan 专用（适配器） |
|---|---|---|
| `vae_stripe.py` 区间 | `Unit`、`need_in`、`valid_out`、`stripe_needs`、`split_rows`（只依赖 kind / halo / scale） | `Unit.kind` 的取值和 halo 由 `build_units` 按 Wan 结构给出 |
| 结构识别 | `_hooks`（forward hook 检查）、`_conv_io` | `wan_structure`、`_causal_conv`、`_rms`、`_check_residual`、`_check_resample`、`build_units`、`signature`、`match` |
| 内存模型 | `conv_extra`（Slow2d 的 columns / 块拷贝）、`arena_bytes`、`Plan` 的骨架（条带、needs、顺序、重算、arena、估算公式） | `res_peak` / `up_peak` / `unit_peak` / `unit_largest` / `prefix_peak` / `prefix_largest` / `_macs_prefix_module`（按 Wan 的 forward 数的：RMS 的临时量、Resample、AttentionBlock） |
| 执行 | `run_prefix`、`run_stripes`（逐单元调用、精确行检查、裁剪）、`arena_supported`、`reserve_arena`、`CONTIGUOUS_INPUT` | `HDIM = 3`（5D 的行维；LDM 是 4D，行维是 2）、`WanStripe.run` 里输出的形状 `(n, C, T, H, W)` |
| 自检 | 流程（fp32 副本、小 latent、强制小条带、和整图比、按结构缓存、`fork_rng`、结束后清缓存） | `_fp32_copy`（按 Wan 的构造参数重建 `Decoder3d`）、`_Shim`、`_is_wan`、参照用 `wan.WanVAE.decode` |
| `vae.py` | `choose_plan`、`_decode_layer1`（OOM 循环、日志、`last_decode`）、`_select_layer1` | `_layer1_self_test` 直接调 `vae_stripe.self_test`；`_selftest_memory` 读 `bound.fsm.decoder` / `conv2`；`_out_bytes` 按 5D 取 T |
| `vae_ops.py` | 全部通用（第二层，任何 VAE） | — （`slow_dilated3d` / conv2d 改道只对单帧 Conv3d 起作用） |
| `tests/alloc_sim.py` | `AllocatorSim`、`Tracer`（含卷积 / 上采样的内核内部缓冲） | `WAN_QWEN`、`meta_vae`、`decode_trace` 里用 `vs.match` |

### 6.3 打算抽出的 engine / adapter 边界（重构清单）

1. 拆 `vae_stripe.py`：
   * `monoload/vae_engine.py`：区间（`Unit`、`need_in`、`valid_out`、`stripe_needs`、`split_rows`）、`Plan`（骨架 + 适配器给的代价模型）、`run_prefix` / `run_stripes`（行维 `hdim` 作为参数，不再是模块常量）、arena、自检的流程框架。
   * `monoload/vae_wan.py`：现有的 Wan 适配器（结构识别、`build_units`、代价模型、fp32 副本、输出形状）。
   * `monoload/vae_ldm.py`：第三阶段的 LDM 适配器。
2. 适配器接口（现在 `vae.py` 注释里的约定再加几项）：`match`、`key`、`name`、`prefix`、`units`、`ckpt_channels`、`hdim`、`output_shape(n, plan)`；代价模型 `unit_peak / unit_largest / prefix_peak / prefix_largest / macs`；`fp32_copy()` / `reference_decode()`（自检用）；`selftest_memory()`；以及第三阶段新增的「统计量依赖」描述（哪些单元里有 GroupNorm，见 6.4）。
3. `vae.py`：`_layer1_self_test` 改调 `bound.self_test()`；`_selftest_memory` 改调适配器；`_out_bytes` 用适配器给的输出形状。`choose_plan`、OOM 循环、日志不变。
4. `Plan` 的估算公式（live → arena → arena + largest + 16 MiB）保持在引擎里，代价模型由适配器提供。新的适配器先用 `alloc_sim` 验证「reserved ≤ 估算」再上真机。
5. `tests/alloc_sim.py`：模型构造按适配器分（加一个 LDM 的 meta 构造，SDXL / Flux 的维度），`decode_trace` 不再直接用 `vae_stripe.match`。有了 LDM 构造，第二层的 SDXL 也能模拟（现在只能实测）。
6. 测试拆成引擎测试（区间、arena、分配器）和各适配器的测试（结构识别、精确行、整个 decoder 与原生比）。

### 6.4 GroupNorm 统计量跨条带调度（要在第三阶段定的方案）

要求不变：结果与整图解码数学等价（只差浮点误差），不能用 tile 局部统计量近似。

* **统计量本身**：每个 GroupNorm 需要其输入在全图上每组的均值和方差。条带可以各自累加每组的 Σx、Σx²（或按 Welford 合并的 count / mean / M2），全部条带跑完后得到全图统计量。要用 fp32 累加；与 torch 的 `group_norm` 内核（bf16 输入时内部也用 fp32）不是逐位一致，但应在自检容差内。
* **依赖链**：第 k 个 GroupNorm 的输入依赖第 1 … k−1 个 GroupNorm 的统计量。最直接的做法是一个 GroupNorm 一遍：第 k 遍从存档跑所有条带到第 k 个 GroupNorm 的输入为止，只累加统计量。19 个 GroupNorm 就是 19 遍，越靠后的遍越长，卷积重算大约是整图的 10 倍量级，太慢。
* **分级存档**：一级（同一分辨率的 3 个残差块）里的 GroupNorm 统计量都算完后，把这一级的输出整张存下来当作下一级的存档。下一级只需要为自己的 6 个 GroupNorm 跑遍数，每遍只从本级存档开始。代价是存档变大：4K bf16 时 H/4 × 512 通道约 0.53 GB，H/2 × 256 通道约 1.06 GB，全分辨率 × 128 通道约 2.1 GB。全分辨率那一级的存档本身就是整张激活。
* **折中方案**：在某一级停止升级存档（例如只到 H/2），全分辨率级用逐 GroupNorm 的遍数。也可以把每遍算出的中间结果按条带缓存一部分，减少重复。总遍数、每遍的重算和存档大小之间的权衡，要用重算比例和 `alloc_sim` 的峰值一起算，再上真机扫参。
* **怎么把统计量用上**：GroupNorm 不能在切片上原样调用（它会用切片自己的统计量）。需要在受管理的解码期间对 GroupNorm 实例做实例级替换（与 `OpChunking` 替换 `_conv_forward` 同一手法）：已知统计量时按 `(x − mean) / sqrt(var + eps) × weight + bias` 计算，统计遍时只累加、不需要输出后面的部分。这是对「模块原样调用」原则的一个有意偏离，要在设计里写清楚，并由自检（fp32、与原生整图比）兜底。
* **注意力**：SDXL / Flux decoder 的注意力只在 mid（H/8），在前缀里整图算，与 Wan 相同，不进条带。
* **预期**：SDXL 4K 的前缀与 Qwen 量级相近（H/8 × 512 通道，P ≈ 133 MB，注意力约 6P + 分数块），条带部分取决于 6.4 的方案。目标是把第二层的 14.9 GiB 降到几个 GiB。具体数字等方案定了再用 `alloc_sim` 算。

### 6.5 第三阶段开始前建议先做的

1. 读 DESIGN §9.11（当初的第三阶段计划）、§9.13（第一层的全部设计，第三阶段会照搬它的大部分）。
2. 先做 6.3 的重构，确保 Wan 的行为和数字不变（`test_vae_stripe.py` 全过、`alloc_sim` 的校验表不变），合进 `dev`，再开始 LDM 适配器。
3. 在 `alloc_sim` 里加 LDM 的 meta 构造，先模拟 SDXL 第二层，与 README §10.1 的真机读数（4K alloc 11.17、GTT 14.86 GiB）对上，再用它评估 6.4 的各个方案。
