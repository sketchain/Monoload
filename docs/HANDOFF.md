# Monoload 交接说明（VAE 三阶段、预算、VAE 节点已验收；进行中：总开关 → LoRA 节点 → 信息节点 → 多语言四个分支，2026-10）

给下一个对话用：当前做到了哪里、必须遵守的规矩、第三阶段（LDM decoder）的状态和待决定的事。细节分别在 README.md（使用、真机结果、bench 命令）和 docs/DESIGN.md（§9：VAE 解码降峰值的设计；§9.14：第三阶段）。

## 1. 当前状态

仓库 github.com/sketchain/Monoload，开发分支 `dev`（不碰 `main`）。ComfyUI 插件，两部分功能，互相独立，由总开关 `MONOLOAD` 统一管（§8）：

* **LoRA 运行时合并**（早期工作）：所有 `ModelPatcher` 走运行时合并，不做原地 LoRA、不留权重备份；每个 prompt 结束后释放 LoRA；`MONOLOAD_EXACT=1` 时与原生逐位一致。**默认（总开关开、没有节点）时这部分的行为不能改**；`MONOLOAD=0` 时钩子直通，与原生逐位一致。
* **VAE 解码降峰值**：包装 `comfy.sd.VAE.decode`。
  * 第二层（逐算子分块，所有 VAE）：卷积按输出行分块（im2col 缓冲 ≤ 工作区），注意力按 query 分块；整图语义不变。
  * 第一层（条带解码）：前缀在 H/8 上整图算出存档，之后按输出行条带倒推重算；整个解码在一个缓存分配器 arena 里运行。引擎 `vae_engine.py` + 每种 decoder 一个适配器：Wan 2.1 单帧（`qwen_image_vae`，第二阶段，已验收）；LDM decoder（SDXL / SD1.5 / SD3 / Flux `ae`，第三阶段，7059bdd 已验收），GroupNorm 的整图统计量用统计遍跨条带求得，默认方案 B。
  * **选择策略**：不设预算时用默认规则（第一层，128 行条带的 arena 为目标）；设了 `MONOLOAD_VAE_BUDGET` 时「预算内最快」（`vae.choose_budget`：第二层能放下就用第二层，否则第一层各方案 × 条带高度里耗时模型预测最快的；DESIGN §9.14.10）。
  * 管理入口：自己的内存估算交给 `load_models_gpu`、batch 逐张、OOM 缩块重试、从不调用 tiled。
  * **节点 Monoload VAE Settings**（插件的第一个节点，`monoload/nodes/`）：输入一个 VAE，输出共享权重的副本，副本带自己的预算 / 方案 / 条带高度 / 模式；逐项「节点 > 环境变量 > 默认值」；总开关分支之后环境变量只是全局默认，节点的 `mode auto` 能为这个 VAE 打开管理（DESIGN §9.15、§10，README §4.1）。

CT 700（Strix Halo，gfx1151，统一内存 62.5 GiB GTT）上 4K（3840×2160）解码的 GTT 增量：

| VAE | 原生 | 第二层 | 第一层 |
|---|---|---|---|
| SDXL / Flux `ae` | 52.5 GiB / 11.9 s | 15.0 GiB / 9.9 s | 默认方案 B 2.17 GiB / 42.1 s（A 1.10 / 75.0，D 1.52 / 57.8，C 4.70 / 35.5） |
| `qwen_image_vae` | 59.2 GiB / 7.89 s | 9.6 GiB / 6.9 s | 0.87 GiB / 8.46 s（估算 1.16） |

还没做的 / 要用户决定的：

1. **四个分支（用户 2026-10 的要求，按顺序，每个合回 `dev` 后再开下一个）**：① settings-master-switch（7482eec，§8）；② lora-settings-node（48c9e54，§9）；③ info-node（a7e187c，§10）；④ i18n（界面中英文、后端消息表 `MONOLOAD_LANG`，§11）。四个都已合并，待真机：命令 V、README §9.9 的网页检查。
   待真机：README §9.8 的命令 V（`tests/check_lora_node.py`）。
2. 第二层「总是比第一层快」是 CT 700 的实测结论，预算策略把它写成了固定优先级；换硬件时要复核。
3. 多帧 Wan 视频 latent 目前交给原生（不是永远不做）。
4. Flux 2 的 `batch_norm_latent`、带注意力的 up 级等 LDM 变体目前走第二层，可以以后按需加。

9b30154 的真机结果（用户确认通过）：命令 T（SDXL 三档 × 3G / 1.5G）全部 GTT ≤ 估算 ≤ 预算、61–63 dB，4K 3G 选中 B 12×180（2.43 GiB / 41.1 s）；命令 U（节点）副本按节点预算选 B 12×180，原 VAE 仍是默认 B 17×128（2.17 GiB），共享权重、原 VAE 不带设置。

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
| 第三阶段 3b + 3c | 7059bdd | 96daa32 | `alloc_sim` 的 LDM 构造（复现第二层 18 个读数）；`vae_ldm.py`；引擎的统计遍、GroupNorm 替换、存档布局；`test_vae_ldm.py`；bench 的 `-g<S>` / `--gn-schemes` / `--fp32-chunked` |
| 默认 B、预算内最快 | 01377c4 | 0a64213 | LDM 默认方案 B；`choose_budget`；耗时模型 `vae_ldm.TIME_COEF`；一条带不跑统计遍；预算内最高条带的搜索在最矮处不单调时继续找；bench 的 `-b<GiB>` / `--budgets`、预算放不下记为 `over budget`；7059bdd 的实测写进文档 |
| 估算收紧、耗时模型第二版 | cc3b9d9 | cfa5a56、fb919af、68ba014、7127617 | 检查点前置 + `saves_fit`（有保证的存档不算进 largest）；耗时模型加卷积调用数和前缀注意力；README §10.4、命令 T |
| Monoload VAE Settings 节点 | 9b30154 | 316b22c 等 | `monoload/nodes/`（注册表 + 节点）、`vae_overrides.py`、`vae.resolve_settings` / `_Applied`；`test_vae_node.py`、`test_entry.py` 的注册检查、`check_vae_node.py`；README §4.1、DESIGN §9.15 |

| 总开关 `MONOLOAD`、高级变量变成全局默认 | 7482eec | 01d212c 等 | `monoload/settings.py`；hotpatch / release / vae 总是装上、关闭时直通；VAE 节点预算下拉框；`test_master_switch.py`；README §4 / §13、DESIGN §10 |
| Monoload LoRA Settings 节点 | 48c9e54 | 3fdfbae 等 | `lora_overrides.py`；hotpatch 按 patcher 决定接管和合并路径；release 按 patcher 释放；`nodes/lora_settings.py`；`test_lora_node.py`、`make_synthetic_checkpoint.py`、`check_lora_node.py`；README §4.2、§9.8，DESIGN §11 |
| Monoload Info 节点 | a7e187c | 53ecadb 等 | `monoload/info.py`、`nodes/info.py`、`web/monoload_info.js`；`vae.decode_record` / `_MemProbe`；`lora_overrides.install_names`；`test_info_node.py`；README §4.3、DESIGN §12 |
| 多语言 | 本文件所在的合并（`git log --first-parent dev` 最上面一条） | 见合并 | `locales/{en,zh}/nodeDefs.json`、`web/monoload_i18n.js`；`monoload/messages.py`（`MONOLOAD_LANG`）；`test_messages.py`；README §4 / §9.9 / §13、DESIGN §13 |

LoRA 部分的历史：3c473fa … 5bfbc8e（v1 文件格式 → v2 全局运行时合并 → 释放、fp8、默认融合 addmm 路径），见 `git log --first-parent dev`。

## 3. 必须遵守的规矩

**优先级（用户定）：峰值内存优先，多算几遍可以接受，速度是次要的。** 取舍时先比峰值（reserved / GTT），再看耗时；例如第一层工作区选 128 MiB（峰值降 23–45%，耗时多 2–7%），默认条带高度取「不明显变慢的最低峰值」。

**行为：**

* **绝不退回 tiled**：不调用 `decode_tiled_`，也不让原生的 OOM → tiled 回退发生在受管理的解码里（tile 局部的 GroupNorm / 注意力与整图不等价）。用户显式用 `VAEDecodeTiled` 节点时保持原生。
* **OOM：缩块缩条带重试，最后抛 `MonoloadVAEOOMError`**。第二层：工作区减半到 64 MiB；第一层：条带高度和工作区一起减半到 8 行 / 64 MiB。第一层 OOM 不退回第二层（第二层峰值更高）。报错信息写明怎么办；默认英文，`MONOLOAD_LANG=zh` 中文（消息表 `monoload/messages.py`，加新消息时两种语言都要写）。
* **自己算的估算传给 `load_models_gpu`**（`memory_required=`），不用原生的 `memory_used_decode`（AMD 上 SDXL 4K 约 92 GiB，会挤掉其他模型）。估算必须是 reserved（GTT）的上界；第一层的上界由 `tests/alloc_sim.py` 验证（Wan 384 个计划，LDM 202 个计划）。
* **预算可以用环境变量覆盖，显式设置时严格执行**：`MONOLOAD_VAE_BUDGET` 设了就在估算不超过它的做法里选预计最快的（第二层，或第一层的方案 × 条带高度；第一层仍取预算内最高的条带），一个都放不下直接抛 `MonoloadError`（写明各需要多少），不悄悄放宽；不认识的 decoder 只能走第二层，放不下也报错。强制设置优先于预算：`MONOLOAD_VAE_STRIPE_ROWS`（强制条带高度，方案仍按预算选，放不下也照跑并注明）、`MONOLOAD_VAE_GN_SCHEME`（强制方案，默认 B，高度仍按预算取）、`MONOLOAD_DISABLE_VAE_STRIPE=1`（第二层）。其他开关：`MONOLOAD_VAE_WORKSPACE`、`MONOLOAD_DISABLE_VAE=1` / `MONOLOAD_EXACT=1`（VAE 全局默认原生）、`MONOLOAD=0`（全局原生）、`MONOLOAD_DISABLE=1`（什么都不装）。
* **原模型的模块实例原样调用**（条带里是调用在行切片上），`comfy.ops` 的 cast / `weight_function`（包括 Monoload 的运行时 LoRA）照常生效；替换只做在实例属性上、只在受管理的解码期间，退出时恢复（`OpChunking`）。唯一的有意例外：LDM 第一层期间条带部分的 GroupNorm 实例换成「用冻结的整图统计量」的 forward（`vae_engine.GlobalNorms`，权重仍走模块自己的 cast / weight_function），由 fp32 自检兜底（DESIGN §9.14.4）。
* **按真实结构识别，不看文件名**；不认识就走第二层并打一条日志说明原因。
* **优先级（用户定，2026-10）：环境变量 = 全局默认值；节点上明确选的值只对那一个模型 / VAE 生效，总是压过全局。** 逐项「节点上明确选的 > 高级环境变量 > 内置默认」，节点上选「跟随全局」的项继承全局。`MONOLOAD=0` 时没有节点的工作流与原版 ComfyUI 逐位一致、开销接近零；`MONOLOAD_DISABLE=1` 什么都不装，节点原样透传。节点不修改输入（返回共享权重的副本）。
* **第一层每种结构在本进程第一次使用前做 fp32 自检**（与原生整图解码比，相对误差 ≤ 1e-4），不通过就对这种结构禁用第一层、醒目警告、走第二层。
* **精度**：不承诺与原生逐位一致（GEMM 形状不同），但 fp32 下与整图等价（≤ 1e-5），bf16 下对 fp32 真值与原生同一水平；用 bench 的 `--fp32-ref` 判断离群点是不是 bf16 噪声。
* **不改 LoRA 部分的默认行为**（总开关开、没有节点时）。

**流程：**

* 从 `dev` 开 feature 分支，做完合回 `dev`（`git merge --no-ff`），push `dev`。不碰 `main`，不开 PR（除非用户要求）。
* **只跑小测试，不跑 `tests/run_all.sh`。** 小测试：`tests/test_messages.py`、`tests/test_info_node.py`、`tests/test_lora_node.py`（先 `python tests/make_synthetic_checkpoint.py $MODELS`）、`tests/test_master_switch.py`、`tests/test_vae_node.py`、`tests/test_vae_ldm.py`、`tests/test_vae_stripe.py`、`tests/test_vae.py`、`tests/test_entry.py`（各种开关组合，含 `MONOLOAD=0`，见文件头）、`tests/test_dtype_paths.py`（默认和 `MONOLOAD_EXACT=1`）。
* **CT 700 上的 bench 代码由我们写，容器操作（拉代码、重建镜像、`/free`、跑命令、切开关）由用户做。** 报告里不写容器层面的步骤，只给 `docker exec ... python tests/bench_vae.py ...` 命令和要看的指标。
* 文档和报告用中文。报告写明合并提交、小测试结果、与要求不同之处及原因。冲突时以用户的最新要求为准。
* 提交信息结尾加 Co-Authored-By / Claude-Session 两行（见会话里的 attribution 提示）。

## 4. 环境与工具

* **锁定镜像**：`docker.io/kyuz0/amd-strix-halo-comfyui@sha256:384aa1fecef6a841832e0d5552949977330308d8c25e212a94f5e8dfcc061cae`，ComfyUI 0.31.0 / 62b3c94，torch 2.14.0a0 + ROCm 7.15。云端容器里 docker daemon 可能没起来：`sudo dockerd > /tmp/dockerd.log 2>&1 &`。
* **跑测试**：`MODELS=<目录> tests/docker_run.sh python tests/test_vae_stripe.py`（把仓库只读挂进镜像的 custom_nodes）。VAE 测试不需要真实模型；需要 VAE 文件的（bench 的 CPU 冒烟）用 `python tests/make_synthetic_vaes.py OUT_DIR [--full]` 生成随机权重的 SDXL / Flux / Wan 结构文件（`--full` 是真实宽度），放在 `$MODELS/vae`。
* **CT 700 的事实**：`--gpu-only --bf16-vae`；ComfyUI 在 AMD 上设 `cudnn.enabled = False`，所以 4D 卷积走 Slow2d（im2col + GEMM，im2col 缓冲就是原生峰值的主因），5D 卷积走 SlowDilated3d；VAE 注意力是 split（`normal_attention`）；统一内存，显存占用看 GTT。
* **`tests/bench_vae.py`**（README 9.7）：`--checkpoint` / `--vae`；`--res`；`--modes native,monoload,monoload-l2,monoload-r<N>,native2`，模式名加 `-w<MiB>` 只对该模式改工作区、加 `-g<S>` 强制 GroupNorm 方案、加 `-b<GiB>` 设预算（顺序 `-g`、`-b`、`-w`）；`--stripe-rows`；`--budgets 20,3,1.5`；`--gn-schemes ADBC`（与 `--stripe-rows` 一起时扫「方案 × 高度」）；`--fp32-chunked l2`（fp32 参照只用第二层）；`--warm`；`--fp32-ref`；`--profile-only --profile-modes ...`；`--no-arena`；`--json`。输出每次运行的 alloc / reserved / GTT / 耗时、估算、精度（对原生、对 fp32）、条带 / 分块边界附近的误差、离群点。
* **`tests/alloc_sim.py`**（DESIGN §9.13.10、§9.14.6）：在 meta 设备上跑全尺寸 decoder（Wan / SDXL / Flux），按 PyTorch 缓存分配器的规则重放分配 / 释放，复现了 51 个 CT 700 读数（Wan 33、SDXL / Flux 第二层 18，≤ 0.02 GiB）。`python tests/alloc_sim.py --res 3840x2160 --rows 32,144,512 [--model qwen|sdxl|flux] [--scheme A|B|C|D] [--layer 2 --ws 1024] [--version v1|v2] [--peak] [--segments]`（`--peak` 时每块标出是哪个单元分配的）；`decode_trace(..., arena_need=True)` 给出这个计划实际需要的 arena。改了内存相关的代码先用它看，再让用户上真机。

## 5. 代码地图

* `monoload/vae.py`：入口和策略。`install()` 包 `VAE.decode` → `_decode`（覆盖范围判断，不管的交给原生）→ `_managed_decode`（有预算 → `_decode_budget` / `choose_budget`：候选、排序、自检、报错；没有 → 选层、自检）→ `_decode_layer1`（`choose_plan` 或预算选好的计划、`load_models_gpu`、OOM 循环、日志、`last_decode()`）或 `_decode_layer2`（`_layer2_estimate`：形状探测 + `estimate()`；OOM 循环）。设置读环境变量（`workspace()`、`budget()`、`layer1_workspace()`、`stripe_rows()`、`gn_scheme()` / `gn_scheme_forced()`）。`STRIPE_ADAPTERS = [vae_wan, vae_ldm]`。
* `monoload/vae_ops.py`：第二层的引擎，完全通用。`OpChunking`（实例级替换 `_conv_forward` 和 `optimized_attention`）、`_ConvChunker`、三种注意力的 query 分块、`OpStats`。
* `monoload/vae_engine.py`：第一层的引擎，与 decoder 无关。适配器接口写在模块注释里。区间（`Unit`、`need_in`、`valid_out`、`stripe_needs`、`split_rows`）；`Plan`（最后一遍的条带、统计遍 `Pass`、存档布局「分开 / 池」、存活量 → arena → 估算、重算倍数）；执行（`run_prefix`、`run_chain`、`run_stripes`、`run_passes`）；GroupNorm（`NormRef`、`Moments`、`group_norm_frozen`、`GlobalNorms`）；arena；`StripeAdapter` 基类（`plan` 找预算内最高条带、`smallest_plan`、`variants` / `predict_seconds` 给预算策略用）；`Plan.work_levels`（各分辨率级的卷积 MAC，耗时模型的输入）；自检流程。
* `monoload/vae_wan.py`：Wan 2.1 单帧适配器（没有需要整图统计的归一化，所以没有统计遍；数字与第二阶段相同）。
* `monoload/settings.py`：总开关 `master()` / `set_master()`、全局默认 `exact()` / `keep()`（`MONOLOAD_EXACT` / `MONOLOAD_KEEP_LORA`）、`disabled()`、`env_flag()`，不导入 torch / ComfyUI。
* `monoload/lora_overrides.py`：单个模型的 LoRA 设置（`model_options["monoload_lora"]`）、`resolve()`（逐项取值和来源）、`enabled()` / `merge_exact()` / `wants_release()`（hotpatch 和 release 用）。
* `monoload/nodes/`：ComfyUI 节点。`__init__.py` 是注册表（`NODES` → `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`，插件入口导出，任何开关下都注册），`vae_settings.py` 是 Monoload VAE Settings。加节点：写模块、把类加进 `NODES`。
* `monoload/vae_overrides.py`：单个 VAE 的设置（副本上的属性、`with_settings`），不导入 torch / ComfyUI；`vae.py` 的 `resolve_settings`（逐项取设置）、`_Applied`（解码期间换进全局设置、结束换回）。
* `monoload/vae_ldm.py`：LDM 适配器：识别（`ldm_structure`）、单元（残差块带 norm1 / norm2 的 `NormRef`，norm2 的「部分单元」）、方案的存档位置（`scheme_positions`）、按 forward 数的内存模型、fp32 副本（按配置重建 `Decoder` + `post_quant_conv`）、`variants()`（每个方案一个，默认方案在前）、耗时模型 `TIME_COEF` / `predict_seconds`。
* 测试：`tests/test_messages.py`（15 项）、`tests/test_info_node.py`（16 项）、`tests/test_lora_node.py`（19 项，需要 `make_synthetic_checkpoint.py` 生成的合成 SD1.5）、`tests/test_master_switch.py`（18 项）、`tests/test_vae_node.py`（22 项）、`tests/test_vae_ldm.py`（100 项）、`tests/test_vae_stripe.py`（74 项）、`tests/test_vae.py`（131 项）、`tests/alloc_sim.py`、`tests/bench_vae.py`、`tests/make_synthetic_vaes.py`。

**以后加一种新 VAE**：写一个适配器（`match`，以及 `StripeAdapter` 的子类：结构、单元、代价模型、fp32 副本 / 参照解码），注册到 `STRIPE_ADAPTERS`；有需要整图统计的 GroupNorm 就在单元上挂 `NormRef`，引擎自动安排统计遍；先用 `alloc_sim` 加一个 meta 构造，验证 reserved ≤ 估算，再上真机。

## 6. 第三阶段：LDM decoder（SDXL / Flux）

### 6.1 做了什么（细节 DESIGN §9.14）

* **3a**：`vae_stripe.py` 拆成 `vae_engine.py` + `vae_wan.py`，Wan 的模拟校验表和测试输出逐字不变（单独合并 aa07431）。
* **3b**：`alloc_sim` 加了 SDXL（AutoencoderKL，z 4）和 Flux（AutoencodingEngine，z 16）的 meta 构造；第二层的 18 个真机读数（README §10.1 和 workspace 实验 K）全部复现到 0.02 GiB 以内，包括 4K「workspace 越小 reserved 越高」的碎片。§10.1 和实验 K 在 4K 上差 0.13 GiB，是 85a5c6f 改了分块卷积的输出分配顺序（模拟器两种顺序都复现）。
* **3c**：GroupNorm 统计遍（19 个），方案 A / D / B / C 是同一个机制的参数（存档位置），`MONOLOAD_VAE_GN_SCHEME` 选择，当时默认 A（验收后改成 B）；统计量 fp32，shifted data + 块内 `var_mean` + Chan 合并；GroupNorm 实例级替换 + fp32 自检；识别只认确认过的结构。
* 模拟器顺带查出并修掉了三处 LDM 特有的内存问题：`nin_shortcut`（1×1，不分块）的整份拷贝没算进模型；这份拷贝在块末尾才发生、放不进被切碎的 arena（改成调用前先连续化）；越来越大的存档放不进前面的洞（「分开 / 池」两种布局，取 arena 小的）。

### 6.2 HANDOFF 粗估与模拟的对比（SDXL 4K bf16，默认条带策略）

| 方案 | 粗估峰值 / 重算 | 模拟 reserved / 估算 / 重算（整个 decoder） |
|---|---|---|
| A | 约 1 GB / 约 13× | 1.10 / 1.48 GiB / 12.1× |
| D | 约 1.5 GB / 9–10× | 1.52 / 2.03 / 8.1× |
| B | 约 2–2.5 GB / 约 6× | 2.17 / 3.18 / 5.3× |
| C | 约 4.5 GB / 约 4.5× | 4.70 / 8.67 / 5.7×（存档池让早期统计遍的条带变矮） |

### 6.3 真机结果与决定（7059bdd，README §10.3，DESIGN §9.14.10）

* 内存与模拟器一致（≤ 0.01 GiB），所有行 ≤ 估算；精度与原生同一水平（4K PSNR 约 61.2 dB，各方案、各高度相同）。
* 耗时 4K：A 75.0、D 57.8、B 42.1、C 35.5 s，第二层 9.9 s；A / D 的矮条带只更慢（D r32 69.1 s）。
* 用户决定：1 GiB 和 2 GiB 的差别不重要，要能自定义。默认方案 B；设了预算时「预算内最快」；强制设置优先于预算。

### 6.4 预算内最快（DESIGN §9.14.10）

* **耗时模型**：耗时 = Σ 各分辨率级的系数 × 该级卷积 MAC，系数（秒 / 10¹² MAC）前缀 1.35、H/4 0.113、H/2 0.129、H 0.269；12 个 CT 700 读数拟合，最大误差 2.3%。它解释了 C（5.7×）比 B（5.3×）快。只用来给同一次解码的第一层配置排序。
* **第二层固定优先**（放得下就选），依据是实测（SDXL 4K 9.9 s 对 ≥ 35.5 s，Qwen 6.9 对 8.4 s）。
* **第一层工作区**依次试 预算/8、128 MiB、64 MiB：预算/8 的工作区会把 B 挤出 3G（4K）。
* **两处引擎改动**：一条带不跑统计遍（整图就是整图统计量）；「预算内最高条带」在最矮的高度放不下时翻倍继续找（B 在 8 / 16 行比 32 行需要更多：统计遍被挤到 1–3 行、存档用池）。
* **真机（01377c4）**：选择都与预测一致，GTT ≤ 估算 ≤ 预算；但 4K 3G 选中的 D（7×309，64 MiB）实测 63.2 s，比默认 B（42.0 s）慢、峰值也更高。原因两个：B 的估算里含 1 GiB 的存档（3.17 > 3G）；耗时模型看不到工作区（64 MiB 的卷积调用数翻倍）。下面 6.5 修了这两处。

### 6.5 收紧估算、耗时模型第二版（DESIGN §9.14.11）

* **检查点前置**：有存档的计划，前缀之前先分配检查点缓冲（arena 的最前面），前缀输出拷进去。之后的长寿命分配（检查点、输出缓冲、各存档 / 池）是确定的序列，`vae_engine.saves_fit` 按缓存分配器的 best fit 规则重放它；每个存档都放得下，就说明它不会被挤出 arena，估算的 largest 不再算它（`Plan.saves_guaranteed`）。4K 默认 B 的估算 3.17 → 2.68 GiB，C 8.67 → 5.21，arena 不变。
* **耗时模型第二版**（`vae_ldm.TIME_COEF` / `TIME_PER_CALL` / `TIME_ATTN`）：H/4 / H/2 / H 的 MAC 系数 0.0922 / 0.1294 / 0.2464 秒 / 10¹²，每次卷积 GEMM 调用 0.237 ms（`unit_calls` 按工作区算行块，`Plan.conv_calls`），前缀注意力 0.239 × 2·C·token² / 10¹²。22 个读数拟合，最大误差 5.3%（整图一条带 −17%）。
* **很高的条带（老问题，扩大的网格里发现）**：4K 方案 B 540 / 768 行时 reserved 超过估算（死掉的存档留下的洞放不下全分辨率的大临时量，一次「挤出」变成好几次）。arena 规则补了两条（有存档时）：`Plan.front_arena`（洞用不上时，临时量放在存活存档的上面）；`arena_bytes` 在 ≤ 3 条带时用 live/8、工作区 > 128 MiB 时余量至少一个工作区。默认计划不变。403 个计划 reserved 全部 ≤ 估算（DESIGN §9.14.11）。
* **按预算选的新结果**（SDXL）：3G → 1344 整图一条带、2688 C 4×384、4K B 12×180（预测 2.43 GiB / 41 s）；1.5G → 1344 C、2688 B、4K A。验证命令 README 9.7 的 T。

## 7. Monoload VAE Settings 节点（DESIGN §9.15，README §4.1）

* **做什么**：输入一个 VAE，输出一个副本（`copy.copy`，共享 `first_stage_model` 和 `patcher`），副本带自己的设置：`mode`（default = 跟随全局 / auto = 为这个 VAE 打开 / layer 2 only / native）、`budget`（default / unlimited / custom，总开关分支加的下拉框，排在最后以兼容旧工作流）+ `budget_gib`（只在 custom 时用）、`gn_scheme`（default / A / B / C / D）、`stripe_rows`（0 = 跟随全局）。
* **优先级**：逐项「节点 > 环境变量 > 默认值」（`vae.resolve_settings`）；串联的节点，下游没设的项沿用上游。每次解码的日志末尾写明各项的值和来源（node / env / default），`last_decode()` 有 `settings` / `settings_source`。
* **怎么生效**：`_decode` 取设置后，在 `_Applied` 里换进全局的 `_SETTINGS` 和 `vae_ldm` 的方案，解码结束（含报错）换回；解码路径其余部分不变。依赖「ComfyUI 一次只执行一个 prompt」。
* **全局开关**：总开关分支之后，`MONOLOAD=0` / `MONOLOAD_DISABLE_VAE` / `MONOLOAD_EXACT` 只让「没有自己模式」的 VAE 走原生，节点 `mode auto` 照样打开；`MONOLOAD_DISABLE=1` 时节点原样返回输入，日志说明一次。节点在任何开关下都注册（保存的工作流能加载）。
* **测试**：`tests/test_vae_node.py`（22 项）；`tests/test_entry.py` 在 8 种开关组合下检查 ComfyUI 的加载器注册了节点。
* **真机**：README 9.7 的命令 U（`tests/check_vae_node.py`），9b30154 已通过。

## 8. 总开关 `MONOLOAD`（DESIGN §10，README §4、§13）

* **总开关**：`MONOLOAD` 不设 / `1` = 开启（默认策略，和以前一样）；`0` = 全局原生：hotpatch / release / VAE 包装照装，但直通原生——没有节点的工作流与原版逐位一致（`tests/test_master_switch.py` 用卸掉钩子的原版对照，整体加载、lowvram、Hook LoRA、VAE 解码都 `torch.equal`，备份数也一样）。只有节点明确开启的模型 / VAE 走 Monoload（VAE：节点 `mode auto`；LoRA：分支二的节点）。`MONOLOAD_DISABLE=1` 仍然什么都不装。
* **高级变量都变成全局默认**：`MONOLOAD_EXACT`、`MONOLOAD_KEEP_LORA`、`MONOLOAD_DISABLE_VAE`、`MONOLOAD_DISABLE_VAE_STRIPE`、`MONOLOAD_VAE_BUDGET`、`MONOLOAD_VAE_GN_SCHEME`、`MONOLOAD_VAE_STRIPE_ROWS`、`MONOLOAD_VAE_WORKSPACE`。以前 `MONOLOAD_DISABLE_VAE` / `MONOLOAD_EXACT` 不装 VAE 包装、`MONOLOAD_KEEP_LORA` 不装释放；现在都装，运行时按全局默认决定。
* **实现要点**：hotpatch `_active()` 先看 `_enabled()`（这一版 = 总开关，分支二改成按 patcher），`patch_weight_to_device`、`ModelPatcherDynamic.load` 也看；`unpatch_model` 按模型上的 `_monoload_runtime` 标记去掉运行时 patch。release `enabled()` 每个 prompt 判断。vae `global_mode()`；全局原生的解码只在 DEBUG 记日志。
* **README**：正文（§4）只讲 `MONOLOAD`、`MONOLOAD_DISABLE` 和节点；其他变量和预算例子在 §13「高级选项」。bench 和测试照旧用环境变量。
* **VAE 节点**：`budget` 下拉框（default / unlimited / custom）+ `budget_gib`；旧工作流里 `budget_gib > 0` 的要改成 custom 才生效（README §4.1）。

## 9. Monoload LoRA Settings 节点（DESIGN §11，README §4.2、§9.8）

* **做什么**：输入 MODEL（必接）和 CLIP（可选），输出 clone（CLIP 没接时输出空）；`mode`（default / enable / native）、`merge`（default / fused / exact）、`after_prompt`（default / release / keep），default 跟随全局。
* **设置放在 `model_options`**：每次 clone 都复制，所以放在 `LoraLoader` 前后都行；设置和上游不同时换 `patches_uuid`，同一底模不同设置的 clone 切换时 ComfyUI 会先还原再按各自方式加载。
* **hotpatch 按 patcher 决定**：`_enabled()` 看 patcher 的 mode，没有就看总开关；`MonoloadRuntimePatch.exact` 是 patcher 的合并路径（None = 全局）。没有节点时行为不变（`test_release.py` 用合成 SD1.5 跑了默认 / KEEP / EXACT，42 / 22 / 42 项全过）。
* **release 按 patcher**：已加载模型和输出缓存按各自设置；原生模型默认保留；全局默认保留且没用过节点时直接返回。
* **测试**：`tests/test_lora_node.py`（19 项）；合成模型 `tests/make_synthetic_checkpoint.py $MODELS`（约 2 GiB，云端测试时放在 scratchpad 的 models 目录）。
* **待真机**：命令 V（`tests/check_lora_node.py --lora <文件>`，默认 checkpoint `waiIllustriousSDXL_v170.safetensors`）。

## 10. Monoload Info 节点（DESIGN §12，README §4.3）

* **输入都可选**（vae、model、images），输出 STRING，同样的文字显示在节点框里；输出节点、每次运行都刷新。
* **节点框里的文字**：`web/monoload_info.js`（插件导出 `WEB_DIRECTORY`）用前端自带的 `window.comfyAPI.textPreviewWidgets`（核心 Preview as Text 的控件）；在锁定镜像起服务、用 Chromium 打开真实前端核对过。改前端相关代码时，用同样的办法看（`docker run -p 127.0.0.1:8188:8188 ... python main.py --cpu --listen 0.0.0.0`，Playwright 在 `/opt/node-tools/node_modules/playwright`）。
* **内容**：不接输入 → 版本 / commit、总开关、每项全局默认和来源；vae → 设置和来源、这个 VAE 对象上一次解码的记录（`vae.decode_record`，含实测 reserved / GTT 峰值增量）；model → LoRA 名字和强度（`LoraLoader` 包装记在 `model_options`）、设置和来源、当前状态。
* **测试**：`tests/test_info_node.py`（16 项）。

## 11. 多语言（DESIGN §13）

* **界面**：`locales/en|zh/nodeDefs.json`（ComfyUI 官方机制），跟随 ComfyUI 的语言；节点名、输入名、输出名、提示由前端翻译。**下拉选项的显示文字前端 1.48.7 不翻译**，由 `web/monoload_i18n.js` 用 combo 控件的 `getOptionLabel`（只改显示）补上；存进工作流和 prompt 的值始终是英文（浏览器里核对过）。加新节点 / 新输入时两份 nodeDefs.json 都要加（`test_messages.py` 会查）。
* **后端**：所有用户可见的日志、报错、Info 文字在 `monoload/messages.py`；`MONOLOAD_LANG=zh` 中文，默认英文。内部诊断（internal error）保持英文。
* **文档**：仍是中文。

