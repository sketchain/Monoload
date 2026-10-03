# Monoload 交接说明（第三阶段实现完成、待真机时，2026-10）

给下一个对话用：当前做到了哪里、必须遵守的规矩、第三阶段（LDM decoder）的状态和待决定的事。细节分别在 README.md（使用、真机结果、bench 命令）和 docs/DESIGN.md（§9：VAE 解码降峰值的设计；§9.14：第三阶段）。

## 1. 当前状态

仓库 github.com/sketchain/Monoload，开发分支 `dev`（不碰 `main`）。ComfyUI 插件，两部分功能，互相独立：

* **LoRA 运行时合并**（早期工作，没动）：所有 `ModelPatcher` 走运行时合并，不做原地 LoRA、不留权重备份；每个 prompt 结束后释放 LoRA；`MONOLOAD_EXACT=1` 时与原生逐位一致。**这部分的行为不能改。**
* **VAE 解码降峰值**：包装 `comfy.sd.VAE.decode`。
  * 第二层（逐算子分块，所有 VAE）：卷积按输出行分块（im2col 缓冲 ≤ 工作区），注意力按 query 分块；整图语义不变。
  * 第一层（条带解码）：前缀在 H/8 上整图算出存档，之后按输出行条带倒推重算；整个解码在一个缓存分配器 arena 里运行。引擎 `vae_engine.py` + 每种 decoder 一个适配器：Wan 2.1 单帧（`qwen_image_vae`，第二阶段，已验收）；**LDM decoder（SDXL / SD1.5 / SD3 / Flux `ae`，第三阶段，已实现，待真机）**，GroupNorm 的整图统计量用统计遍跨条带求得。
  * 管理入口：自己的内存估算交给 `load_models_gpu`、batch 逐张、OOM 缩块重试、从不调用 tiled。

CT 700（Strix Halo，gfx1151，统一内存 62.5 GiB GTT）上 4K（3840×2160）解码的 GTT 增量：

| VAE | 原生 | 第二层 | 第一层 |
|---|---|---|---|
| SDXL / Flux `ae` | 52.5 GiB | 14.9 GiB | **模拟 1.10 GiB**（方案 A，估算 1.48，卷积算量约 12 倍；待实测） |
| `qwen_image_vae` | 59.2 GiB | 9.6 GiB | 0.87 GiB（估算 1.16，8.44 s，原生 7.89 s） |

还没做的 / 要用户决定的：

1. **第三阶段的真机测试**：README 9.7 的命令 L–P（每条后面有模拟器的预测）。
2. **LDM 的默认 GroupNorm 方案和条带规则**（下面 6.3）：现在默认 A（峰值最低），条带规则沿用第二阶段（128 行条带的峰值为目标）。看真机的峰值和耗时再定。
3. 多帧 Wan 视频 latent 目前交给原生（不是永远不做）。
4. Flux 2 的 `batch_norm_latent`、带注意力的 up 级等 LDM 变体目前走第二层，可以以后按需加。

## 2. 提交记录（`dev` 上的合并提交 ← 功能提交）

| 阶段 | 合并 | 功能提交 | 内容 |
|---|---|---|---|
| VAE 第一阶段 | af9abc6 | 89758a2 | 管理入口、第二层逐算子分块、开关、`bench_vae.py`、测试 |
| fp32 参照 | 76e56a1 | f08c986 | bench 的 fp32 参照和离群点列表；第一阶段真机结果 |
| 第一阶段验收 | 191f59b | e3ab1e6 | 日志计时前后同步设备；fp32 结论，第一阶段通过 |
| 第二阶段 | 4e54d20 | 6a50513 | 第一层：Wan 2.1 单帧条带解码、自检、预算 |
| 第二阶段调整 | 725a010 | 0826ff9 | 默认条带策略、按 forward 数的内存模型、单帧 Conv3d 改走 conv2d |
| 第二阶段第三轮 | 85a5c6f | 45a0b6b | arena、估算 = arena + largest、分块卷积先分配输出、`tests/alloc_sim.py` |
| 收尾 | 5d668b6 | 9387168 | 实测值写进文档、workspace 实验命令（bench `-w<MiB>`）、交接说明 |
| 第一层工作区 128 MiB | 13d2384 | b742b16 | `LAYER1_WORKSPACE` 384 → 128 MiB；workspace 实验结果 |
| 第三阶段 3a | aa07431 | a68161b | 拆出 `vae_engine.py` / `vae_wan.py`，行为和数字不变 |
| 第三阶段 3b + 3c | 本文件所在的合并（`git log --first-parent dev` 最上面一条） | 见合并 | `alloc_sim` 的 LDM 构造（复现第二层 18 个读数）；`vae_ldm.py`；引擎的统计遍、GroupNorm 替换、存档布局；`test_vae_ldm.py`；bench 的 `-g<S>` / `--gn-schemes` / `--fp32-chunked` |

LoRA 部分的历史：3c473fa … 5bfbc8e（v1 文件格式 → v2 全局运行时合并 → 释放、fp8、默认融合 addmm 路径），见 `git log --first-parent dev`。

## 3. 必须遵守的规矩

**优先级（用户定）：峰值内存优先，多算几遍可以接受，速度是次要的。** 取舍时先比峰值（reserved / GTT），再看耗时；例如第一层工作区选 128 MiB（峰值降 23–45%，耗时多 2–7%），默认条带高度取「不明显变慢的最低峰值」。

**行为：**

* **绝不退回 tiled**：不调用 `decode_tiled_`，也不让原生的 OOM → tiled 回退发生在受管理的解码里（tile 局部的 GroupNorm / 注意力与整图不等价）。用户显式用 `VAEDecodeTiled` 节点时保持原生。
* **OOM：缩块缩条带重试，最后抛 `MonoloadVAEOOMError`**。第二层：工作区减半到 64 MiB；第一层：条带高度和工作区一起减半到 8 行 / 64 MiB。第一层 OOM 不退回第二层（第二层峰值更高）。报错信息是中文，写明怎么办。
* **自己算的估算传给 `load_models_gpu`**（`memory_required=`），不用原生的 `memory_used_decode`（AMD 上 SDXL 4K 约 92 GiB，会挤掉其他模型）。估算必须是 reserved（GTT）的上界；第一层的上界由 `tests/alloc_sim.py` 验证（Wan 384 个计划，LDM 202 个计划）。
* **预算可以用环境变量覆盖，显式设置时严格执行**：`MONOLOAD_VAE_BUDGET` 设了就取估算不超过它的最高条带，放不下直接抛 `MonoloadError`（写明需要多少），不悄悄放宽；`MONOLOAD_VAE_STRIPE_ROWS` 强制条带高度（优先于默认策略和预算）。其他开关：`MONOLOAD_VAE_GN_SCHEME`（LDM 的 GroupNorm 方案 A / D / B / C，默认 A）、`MONOLOAD_VAE_WORKSPACE`、`MONOLOAD_DISABLE_VAE_STRIPE=1`、`MONOLOAD_DISABLE_VAE=1`、`MONOLOAD_EXACT=1`（VAE 走原生）、`MONOLOAD_DISABLE=1`。
* **原模型的模块实例原样调用**（条带里是调用在行切片上），`comfy.ops` 的 cast / `weight_function`（包括 Monoload 的运行时 LoRA）照常生效；替换只做在实例属性上、只在受管理的解码期间，退出时恢复（`OpChunking`）。唯一的有意例外：LDM 第一层期间条带部分的 GroupNorm 实例换成「用冻结的整图统计量」的 forward（`vae_engine.GlobalNorms`，权重仍走模块自己的 cast / weight_function），由 fp32 自检兜底（DESIGN §9.14.4）。
* **按真实结构识别，不看文件名**；不认识就走第二层并打一条日志说明原因。
* **第一层每种结构在本进程第一次使用前做 fp32 自检**（与原生整图解码比，相对误差 ≤ 1e-4），不通过就对这种结构禁用第一层、醒目警告、走第二层。
* **精度**：不承诺与原生逐位一致（GEMM 形状不同），但 fp32 下与整图等价（≤ 1e-5），bf16 下对 fp32 真值与原生同一水平；用 bench 的 `--fp32-ref` 判断离群点是不是 bf16 噪声。
* **不改 LoRA 部分的行为。**

**流程：**

* 从 `dev` 开 feature 分支，做完合回 `dev`（`git merge --no-ff`），push `dev`。不碰 `main`，不开 PR（除非用户要求）。
* **只跑小测试，不跑 `tests/run_all.sh`。** VAE 相关的小测试：`tests/test_vae_ldm.py`、`tests/test_vae_stripe.py`、`tests/test_vae.py`、`tests/test_entry.py`（8 种开关组合，见文件头）、`tests/test_dtype_paths.py`（默认和 `MONOLOAD_EXACT=1`）。
* **CT 700 上的 bench 代码由我们写，容器操作（拉代码、重建镜像、`/free`、跑命令、切开关）由用户做。** 报告里不写容器层面的步骤，只给 `docker exec ... python tests/bench_vae.py ...` 命令和要看的指标。
* 文档和报告用中文。报告写明合并提交、小测试结果、与要求不同之处及原因。冲突时以用户的最新要求为准。
* 提交信息结尾加 Co-Authored-By / Claude-Session 两行（见会话里的 attribution 提示）。

## 4. 环境与工具

* **锁定镜像**：`docker.io/kyuz0/amd-strix-halo-comfyui@sha256:384aa1fecef6a841832e0d5552949977330308d8c25e212a94f5e8dfcc061cae`，ComfyUI 0.31.0 / 62b3c94，torch 2.14.0a0 + ROCm 7.15。云端容器里 docker daemon 可能没起来：`sudo dockerd > /tmp/dockerd.log 2>&1 &`。
* **跑测试**：`MODELS=<目录> tests/docker_run.sh python tests/test_vae_stripe.py`（把仓库只读挂进镜像的 custom_nodes）。VAE 测试不需要真实模型；需要 VAE 文件的（bench 的 CPU 冒烟）用 `python tests/make_synthetic_vaes.py OUT_DIR [--full]` 生成随机权重的 SDXL / Flux / Wan 结构文件（`--full` 是真实宽度），放在 `$MODELS/vae`。
* **CT 700 的事实**：`--gpu-only --bf16-vae`；ComfyUI 在 AMD 上设 `cudnn.enabled = False`，所以 4D 卷积走 Slow2d（im2col + GEMM，im2col 缓冲就是原生峰值的主因），5D 卷积走 SlowDilated3d；VAE 注意力是 split（`normal_attention`）；统一内存，显存占用看 GTT。
* **`tests/bench_vae.py`**（README 9.7）：`--checkpoint` / `--vae`；`--res`；`--modes native,monoload,monoload-l2,monoload-r<N>,native2`，模式名加 `-w<MiB>` 只对该模式改工作区、加 `-g<S>` 选 GroupNorm 方案；`--stripe-rows`；`--gn-schemes ADBC`（与 `--stripe-rows` 一起时扫「方案 × 高度」）；`--fp32-chunked l2`（fp32 参照只用第二层）；`--warm`；`--fp32-ref`；`--profile-only --profile-modes ...`；`--no-arena`；`--json`。输出每次运行的 alloc / reserved / GTT / 耗时、估算、精度（对原生、对 fp32）、条带 / 分块边界附近的误差、离群点。
* **`tests/alloc_sim.py`**（DESIGN §9.13.10、§9.14.6）：在 meta 设备上跑全尺寸 decoder（Wan / SDXL / Flux），按 PyTorch 缓存分配器的规则重放分配 / 释放，复现了 51 个 CT 700 读数（Wan 33、SDXL / Flux 第二层 18，≤ 0.02 GiB）。`python tests/alloc_sim.py --res 3840x2160 --rows 32,144,512 [--model qwen|sdxl|flux] [--scheme A|B|C|D] [--layer 2 --ws 1024] [--version v1|v2] [--peak] [--segments]`（`--peak` 时每块标出是哪个单元分配的）；`decode_trace(..., arena_need=True)` 给出这个计划实际需要的 arena。改了内存相关的代码先用它看，再让用户上真机。

## 5. 代码地图

* `monoload/vae.py`：入口和策略。`install()` 包 `VAE.decode` → `_decode`（覆盖范围判断，不管的交给原生）→ `_managed_decode`（选层、自检）→ `_decode_layer1`（`choose_plan`、`load_models_gpu`、OOM 循环、日志、`last_decode()`）或 `_decode_layer2`（形状探测、`estimate()`、OOM 循环）。设置读环境变量（`workspace()`、`budget()`、`layer1_workspace()`、`stripe_rows()`、`gn_scheme()`）。`STRIPE_ADAPTERS = [vae_wan, vae_ldm]`。
* `monoload/vae_ops.py`：第二层的引擎，完全通用。`OpChunking`（实例级替换 `_conv_forward` 和 `optimized_attention`）、`_ConvChunker`、三种注意力的 query 分块、`OpStats`。
* `monoload/vae_engine.py`：第一层的引擎，与 decoder 无关。适配器接口写在模块注释里。区间（`Unit`、`need_in`、`valid_out`、`stripe_needs`、`split_rows`）；`Plan`（最后一遍的条带、统计遍 `Pass`、存档布局「分开 / 池」、存活量 → arena → 估算、重算倍数）；执行（`run_prefix`、`run_chain`、`run_stripes`、`run_passes`）；GroupNorm（`NormRef`、`Moments`、`group_norm_frozen`、`GlobalNorms`）；arena；`StripeAdapter` 基类；自检流程。
* `monoload/vae_wan.py`：Wan 2.1 单帧适配器（没有需要整图统计的归一化，所以没有统计遍；数字与第二阶段相同）。
* `monoload/vae_ldm.py`：LDM 适配器：识别（`ldm_structure`）、单元（残差块带 norm1 / norm2 的 `NormRef`，norm2 的「部分单元」）、方案的存档位置（`scheme_positions`）、按 forward 数的内存模型、fp32 副本（按配置重建 `Decoder` + `post_quant_conv`）。
* 测试：`tests/test_vae_ldm.py`（77 项）、`tests/test_vae_stripe.py`（73 项）、`tests/test_vae.py`（131 项）、`tests/alloc_sim.py`、`tests/bench_vae.py`、`tests/make_synthetic_vaes.py`。

**以后加一种新 VAE**：写一个适配器（`match`，以及 `StripeAdapter` 的子类：结构、单元、代价模型、fp32 副本 / 参照解码），注册到 `STRIPE_ADAPTERS`；有需要整图统计的 GroupNorm 就在单元上挂 `NormRef`，引擎自动安排统计遍；先用 `alloc_sim` 加一个 meta 构造，验证 reserved ≤ 估算，再上真机。

## 6. 第三阶段：LDM decoder（SDXL / Flux）

### 6.1 做了什么（细节 DESIGN §9.14）

* **3a**：`vae_stripe.py` 拆成 `vae_engine.py` + `vae_wan.py`，Wan 的模拟校验表和测试输出逐字不变（单独合并 aa07431）。
* **3b**：`alloc_sim` 加了 SDXL（AutoencoderKL，z 4）和 Flux（AutoencodingEngine，z 16）的 meta 构造；第二层的 18 个真机读数（README §10.1 和 workspace 实验 K）全部复现到 0.02 GiB 以内，包括 4K「workspace 越小 reserved 越高」的碎片。§10.1 和实验 K 在 4K 上差 0.13 GiB，是 85a5c6f 改了分块卷积的输出分配顺序（模拟器两种顺序都复现）。
* **3c**：GroupNorm 统计遍（19 个），方案 A / D / B / C 是同一个机制的参数（存档位置），`MONOLOAD_VAE_GN_SCHEME` 选择，默认 A；统计量 fp32，shifted data + 块内 `var_mean` + Chan 合并；GroupNorm 实例级替换 + fp32 自检；识别只认确认过的结构。
* 模拟器顺带查出并修掉了三处 LDM 特有的内存问题：`nin_shortcut`（1×1，不分块）的整份拷贝没算进模型；这份拷贝在块末尾才发生、放不进被切碎的 arena（改成调用前先连续化）；越来越大的存档放不进前面的洞（「分开 / 池」两种布局，取 arena 小的）。

### 6.2 HANDOFF 粗估与模拟的对比（SDXL 4K bf16，默认条带策略）

| 方案 | 粗估峰值 / 重算 | 模拟 reserved / 估算 / 重算（整个 decoder） |
|---|---|---|
| A | 约 1 GB / 约 13× | 1.10 / 1.48 GiB / 12.1× |
| D | 约 1.5 GB / 9–10× | 1.52 / 2.03 / 8.1× |
| B | 约 2–2.5 GB / 约 6× | 2.17 / 3.18 / 5.3× |
| C | 约 4.5 GB / 约 4.5× | 4.70 / 8.67 / 5.7×（存档池让早期统计遍的条带变矮） |

### 6.3 要用户看真机数据后决定的

* **默认方案。** 峰值优先选的是 A。但模拟显示 4K 的峰值下限约 1.05 GiB（整图前缀决定），A 用 32–96 行条带或 **D 用 32 行条带**都能到，而 D 的算量更少（10.3× 对 A 默认 12.1×）。如果真机上 D-r32 的峰值和 A 一样、明显更快，可以考虑「LDM 默认 D + 更矮的条带」，或者把默认条带规则改成「峰值最低的方案 × 高度里算量最少的」。命令 N / O 就是为这个准备的。
* **速度能否接受。** A 约 12 倍卷积算量；如果 4K 慢到不能接受，B（2.17 GiB、5.3×）是折中。
* 第二层的 SDXL / Flux 估算现在只在不认识的 LDM 变体上用到。
