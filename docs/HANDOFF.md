# Monoload 交接说明（第四阶段：VAE 解码管理推广到全部 VAE；4a 盘点、4b-0 修缺口、4b-1 Flux 2 第一层已完成并验收，2026-10）

给下一个对话用。到这里为止的功能都已在 CT 700 上验收；这份文档讲：现在是什么状态（§1）、节点和设置体系（§2）、必须遵守的规矩（§3）、第四阶段的盘点结果和下一步（§4）、环境和工具（§5）、代码地图（§6）、提交记录（§7）。细节在 README.md（使用、真机结果、bench 命令）和 docs/DESIGN.md（§9 VAE 设计，§10–§13 总开关 / LoRA 节点 / Info 节点 / 多语言）。

## 1. 当前状态

仓库 github.com/sketchain/Monoload，开发分支 `dev`（不碰 `main`，`main` 停在 5bfbc8e）。ComfyUI 插件，两部分功能互相独立：

* **LoRA 运行时合并**（`monoload/hotpatch.py`、`release.py`）：所有 `ModelPatcher` 的 LoRA 不写进权重、不留备份，每一层计算时在临时权重上合并；默认融合 fp16 `addmm_`，可选逐位一致；每个 prompt 结束后释放 LoRA，底模常驻。CT 700：SDXL 比原生少用约 4.7 GiB（8.2 对 12.9 GiB），每步 fused 0.73 s、exact 0.90 s、原生 0.654 s（19d7694，命令 V）。
* **VAE 解码降峰值**（`monoload/vae*.py`）：包装 `comfy.sd.VAE.decode`。
  * 第二层（所有 VAE 的图像解码）：卷积按输出行分块，注意力按 query 分块，整图语义不变。
  * 第一层（认得的 decoder）：H/8 前缀整图算出存档，之后按输出行条带倒推重算；整个解码在一个 arena 里。适配器：Wan 2.1 单帧（`qwen_image_vae`）、LDM decoder（SDXL / SD1.5 / SD3 / Flux `ae`，GroupNorm 整图统计量用统计遍求，方案 A / D / B / C，默认 B）。
  * 不设预算：默认条带策略；设了预算（环境变量或节点）：预算内预计最快（第二层放得下就用第二层）。
  * 自己的内存估算交给 `load_models_gpu`；OOM 只缩块重试；从不退回 tiled。
* **三个节点**（分类 `Monoload`）：Monoload LoRA Settings、Monoload VAE Settings、Monoload Info（§2）。
* **多语言**：节点界面中 / 英跟随 ComfyUI；日志和报错默认英文，`MONOLOAD_LANG=zh` 中文。

CT 700（Strix Halo，gfx1151，62.5 GiB 统一内存）4K 解码的 GTT 增量：

| VAE | 原生 | 第二层 | 第一层 |
|---|---|---|---|
| SDXL / Flux `ae` | 52.5 GiB / 11.9 s | 15.0 GiB / 9.9 s | 默认 B 2.17 GiB / 42 s（A 1.10 / 75，D 1.52 / 58，C 4.70 / 36） |
| `qwen_image_vae` | 59.2 GiB / 7.9 s | 9.6 GiB / 6.9 s | 0.87 GiB / 8.5 s |

验收情况：VAE 三个阶段、预算策略、VAE 节点、总开关、LoRA 节点、Info 节点、多语言都已在 CT 700 上通过（最近一次 19d7694：命令 V / U、网页检查）。polish-after-ui-test（05ded8d）修了 UI 实测发现的六处问题，待 CT 700 复测（README §9.10）。

**第四阶段 4a（vae-inventory，575fc46；命令 W 的识别修正 87e6262）**：只加了盘点脚本和文档，插件行为没变。结论：用户实际在用的三个 VAE（SDXL、Flux `ae`、`qwen_image_vae`）的图像解码**已经全部走第一层**；其余 VAE 的结构、现在的路、模拟峰值和建议顺序见 §4 和 DESIGN §9.16；发现现有代码的 4 处缺口（SVD 结果被改变、2D latent 的音频 VAE 被管理且 ACE 的估算约 1 PiB、像素空间被管理、TAESD 小图第二层反而更高）。用户已定顺序（§4.0）。

**4b-1（vae-flux2-layer1，4fbaa22，CT 700 验收通过）**：Flux 2 VAE（`AutoencoderKL` + batch-norm latent）走第一层，复用 LDM 适配器：latent 的反归一化和 2×2 还原是前缀第一个模块（`vae_ldm.LatentUnpatch`），引擎加 `decoder_hw` 钩子在 decoder 的分辨率上做计划。模拟与 Flux `ae` 相同（4K 默认 B 2.18 GiB）。DESIGN §9.18，测试 `tests/test_vae_flux2.py`，真机命令 README §9.12。

**4b-0（vae-coverage-fixes）**：盘点的缺口 1–3 已修（DESIGN §9.17）：SVD 这类在 batch 各帧之间混合的 decoder、2D latent 的音频 VAE、没有可分块算子的像素空间 VAE 都走原生；`_native_reason` 的理由改走消息表。缺口 4（TAESD 小图）不在范围内。

## 2. 节点和设置体系

**规则（用户定）：环境变量 = 全局默认值；节点上明确选的值只对那一个模型 / VAE 生效，而且总是压过全局。** 逐项判断：节点上明确选的 > 高级环境变量 > 内置默认；节点上选「跟随全局」（存储值 `default`）的项继承全局。

* **总开关 `MONOLOAD`**（`settings.py`）：不设 / `1` = 默认策略；`0` = 全局原生，钩子照装但直通，没有节点的工作流与原版逐位一致（`test_master_switch.py`），只有节点明确开启的模型 / VAE 走 Monoload。`MONOLOAD_DISABLE=1` 什么都不装，节点原样透传。
* **高级环境变量**（README §13）：`MONOLOAD_EXACT`、`MONOLOAD_KEEP_LORA`、`MONOLOAD_DISABLE_VAE`、`MONOLOAD_DISABLE_VAE_STRIPE`、`MONOLOAD_VAE_BUDGET`、`MONOLOAD_VAE_GN_SCHEME`、`MONOLOAD_VAE_STRIPE_ROWS`、`MONOLOAD_VAE_WORKSPACE`、`MONOLOAD_LANG`。bench 和测试仍用它们配置。
* **Monoload LoRA Settings**（`lora_overrides.py`、`nodes/lora_settings.py`）：MODEL（必接）+ CLIP（可选）→ clone；`mode`（default / enable / native）、`merge`（default / fused / exact）、`after_prompt`（default / release / keep）。设置在 `model_options["monoload_lora"]`，每次 clone 都复制，所以放在 LoRA 加载器前后都行；设置和上游不同的 clone 换 `patches_uuid`。hotpatch 按 patcher 决定接管和合并路径（`_enabled`、`MonoloadRuntimePatch.exact`），release 按 patcher 决定释放。原生模型默认保留。
* **Monoload VAE Settings**（`vae_overrides.py`、`nodes/vae_settings.py`）：VAE → 共享权重的副本（`copy.copy`，属性 `_monoload_vae_settings`）；控件顺序 `budget`（default / unlimited / custom）、`budget_gib`（精度 0.01，只在 custom 时用）、`gn_scheme`、`stripe_rows`、`mode`（default / auto / layer 2 only / native）。`vae.resolve_settings` 逐项取值和来源，`_Applied` 在解码期间换进全局设置（含来源 `_SETTINGS["src"]`，日志和报错据此写「来源：节点 / 环境变量」并按来源给建议）。dev 不做旧工作流兼容（用户定）。
* **Monoload Info**（`info.py`、`nodes/info.py`、`web/monoload_info.js`）：输入 vae / model / images 都可选，文字显示在节点框里并从 STRING 输出；每次运行都刷新。内容：版本 / commit / 总开关；VAE 的设置和来源（当前模式下不生效的项会标出来）、这个 VAE 对象上一次解码的记录（`vae.decode_record`：层、方案及一句说明、条带、工作区、估算、实测 reserved / GTT 峰值增量，第一次含自检时注明）；模型的 LoRA 名字和强度（`LoraLoader` 包装记在 `model_options`）、设置和来源、内存里的状态；最后总是全局默认值表。
* **界面翻译**：`locales/{en,zh}/nodeDefs.json`（官方机制）；下拉的显示文字由 `web/monoload_i18n.js` 用 combo 的 `getOptionLabel` 补（前端 1.48.7 不做），存储值始终英文。**加节点或输入时两份 nodeDefs.json 都要加**（`test_messages.py` 检查）。
* **后端消息**：所有用户可见的日志、报错、Info 文字都在 `monoload/messages.py`（`msg(key, **字段)`，英文 / 中文两列，字段一致）。**加消息时两种语言都要写。** 内部诊断（internal error）保持英文。

## 3. 必须遵守的规矩

**优先级（用户定）：峰值内存优先，多算几遍可以接受，速度次要。** 取舍时先比峰值（reserved / GTT），再看耗时。

**行为：**

* **绝不退回 tiled**：不调用 `decode_tiled_`，也不让原生的 OOM → tiled 回退发生在受管理的解码里。用户显式用 `VAEDecodeTiled` 时保持原生。
* **OOM：缩块缩条带重试，最后抛 `MonoloadVAEOOMError`**（写明怎么办）。第一层 OOM 不退回第二层。
* **自己的估算传给 `load_models_gpu`**（`memory_required=`），必须是 reserved（GTT）的上界；第一层的上界由 `tests/alloc_sim.py` 验证。
* **显式预算严格执行**：放不下就报错并写明各需要多少，按来源给建议，不悄悄放宽。强制设置（条带高度、方案、只用第二层）优先于预算。
* **原模型的模块实例原样调用**，`comfy.ops` 的 cast / `weight_function`（包括 Monoload 的运行时 LoRA）照常生效；替换只做在实例属性上、只在受管理的解码期间，退出时恢复。唯一的例外：LDM 第一层期间 GroupNorm 用冻结的整图统计量，由 fp32 自检兜底。
* **按真实结构识别，不看文件名**；不认识就走第二层并打一条日志。
* **第一层每种结构在本进程第一次使用前做 fp32 自检**（相对误差 ≤ 1e-4），不通过就对这种结构禁用第一层、醒目警告、走第二层。
* **精度**：不承诺与原生逐位一致，但 fp32 下与整图等价（≤ 1e-5），bf16 下对 fp32 真值与原生同一水平。
* **LoRA 默认行为不变**（总开关开、没有节点时）；`MONOLOAD=0` 且没有节点时与原版逐位一致、开销接近零。
* **节点不修改输入**（返回 clone / 副本），任何开关下都注册。
* **日志默认英文**（用户定）；中文在 `MONOLOAD_LANG=zh`。

**流程：**

* 从 `dev` 开 feature 分支，每项单独提交，做完 `git merge --no-ff` 回 `dev` 并 push。不碰 `main`，不开 PR。
* **只跑小测试，不跑 `tests/run_all.sh`**（清单见 §6）。
* **CT 700 上的 bench / 检查脚本由我们写，容器操作由用户做。** 报告只给 `docker exec ... python tests/...` 命令、每行的预期和要看的指标。
* 文档和报告用中文。报告写明合并提交、小测试结果、与要求不同之处及原因、要用户决定的事。冲突时以用户的最新要求为准。
* 提交信息结尾加 Co-Authored-By / Claude-Session 两行（见会话里的 attribution 提示）。

## 4. 第四阶段：VAE 解码管理推广到全部 VAE

### 4.0 4a 盘点的结果（详见 DESIGN §9.16）

* **方法**：`tests/vae_inventory.py` 在 meta 上建出 sd.py 能构造的每种 VAE（交给 `comfy.sd.VAE` 本身识别），统计结构、判断 Monoload 现在的路、用 alloc_sim 追踪原生 / 第二层 / 第一层的 reserved 峰值。追踪加了 60 GiB 设备上限（OOM 时先释放缓存再试），SDXL 4K 原生模拟 52.48 GiB = 实测。`tests/probe_vae_gaps.py` 用随机权重的小解码确认缺口和第二层在多帧上的精度。`tests/check_models.py`（命令 W）给 CT 700 识别模型文件用哪个 VAE。
* **用户在用的**：SDXL（checkpoint 内置）、Flux `ae`（Z-Image 也用）、`qwen_image_vae`（Krea 2、Anima）——图像解码都已是第一层。`novaAnimeAM`、`luciddreamerZ` 等命令 W 的结果确认。
* **还走第二层的图像 VAE**：Flux 2（4K 原生 52.5 → 第二层 15.05，第一层预计 2.18，与 Flux `ae` 同一个 decoder，**最容易**）、Wan 2.2 单帧（4K 34.2 → 19.8，第二层不够）、HunyuanImage 2.1（38.1 → 11.1）、HunyuanImage 2.1 Refiner（1344 原生 50.3 → 4.4）、HunyuanVideo 1.0 / 1.5 单帧、SeedVR2、TAE 系列、Stage A / C、Mage、像素空间。
* **还走原生的多帧视频**：第二层在多帧上**数值精确**（Wan 2.1 / 2.2、HunyuanVideo 1.0 / 1.5、CogVideoX 的小解码，相对误差 ≤ 2.3e-6），模拟峰值：Wan 2.1 480p 81 帧 8.7 → 5.0、Wan 2.2 704p 121 帧 39.4 → 10.4、HunyuanVideo 1.0 480p 73 帧 62（原生 OOM → tiled）→ 15.2、HunyuanVideo 1.5 720p 121 帧 113（OOM）→ 21.6、CogVideoX 35.0 → 11.7、Cosmos 30.2 → 13.0、Mochi 362 → 59.3（第二层不够）、LTX 30.6 / 59.3（模拟碎片，待确认）→ 6.7 / 5.0（要走 `output_buffer`）。要做的是 `_native_reason` 放开、`_probe` / `estimate` 认识 5D 多帧。
* **现有缺口**（这一步只报告）：① SVD 的 `VideoDecoder` 以 batch 为时间轴，第二层逐样本解码改变了结果；② ACE-Step / LTX 2 音频 / MiniMax 音频（2D latent）被管理，ACE 的估算约 1 PiB（`load_models_gpu` 会卸载一切）；③ 像素空间被管理（估算 2 GiB）；④ TAESD 1344 第二层 2.66 > 原生 1.77 GiB。
* **建议顺序**：0 修缺口 ① – ③ → 1 Flux 2 第一层 → 2 多帧视频第二层（通用）→ 3 Wan 2.2 单帧第一层 → 4 HunyuanImage 2.1 第一层 → 5 视频第一层（等有需要）。不建议做：TAE 系列、Stage A / C、Mage、MiniMax 视频、SeedVR2 第一层、音频、3D。
* **用户定的（4a 之后）**：顺序 ① 修缺口 1–3 → ② Flux 2 第一层（一定会用）→ ③ 视频等用户定了再说（不是不做，往后放）。SVD 先走原生（并进 ①），以后做视频第二层时把 SVD 整批第二层一起做。每项一个分支，分开提交，合进 dev。
* **进度**：① 完成（4b-0，DESIGN §9.17）；② 完成并在 CT 700 上**验收通过**（4b-1，DESIGN §9.18，实测 README §10.5：W / X / Y / Z / check_vae_node 都与预测一致）；③ 视频等用户定。
* **待用户决定**：第一次使用时的自检峰值不在估算里（DESIGN §9.19：自检约 0.74 GiB，小图 / 紧预算时第一次会超出估算；4K 节点副本多出的 0.24 GiB 推断是进程里第一批 GPU 计算的一次性开销）。可选改法 A / B / C1 / C2 / C3，建议 B + A；可选诊断命令 AA（README §9.12）。

### 4.1 入口和做法（4a 之前写的，仍然适用）

**现在的覆盖范围**（`vae._native_reason`、`STRIPE_ADAPTERS`）：

| ComfyUI 的 VAE（`comfy/sd.py` 按 state dict 识别） | 现在走哪条路 |
|---|---|
| SD1.5 / SDXL / SD3 / Flux `ae`（LDM `Decoder`，`AutoencoderKL` / `AutoencodingEngine`） | 第一层（`vae_ldm`） |
| Wan 2.1 / `qwen_image_vae` 单帧（5D，T=1） | 第一层（`vae_wan`） |
| 其他 2D 图像 VAE（Flux 2 的 `batch_norm_latent` 变体、带注意力的 up 级、TAESD、Stable Cascade Stage A / C、Mage-VAE、SeedVR2 等） | 第二层（逐算子分块），或第一层识别不通过时第二层 |
| 多帧视频 latent：Wan 2.1 / 2.2、Hunyuan 系（3D 卷积 `AutoencoderKL` / `AutoencodingEngine`）、Mochi、Cosmos、CogVideoX、MiniMax H3、TAEHV 等 | **原生**（`_native_reason`：multi-frame video latent，第一阶段暂不做） |
| 自己往预分配输出写的（`comfy_has_chunked_io`，如 LTX） | 原生 |
| 1D / 音频（Stable Audio、ACE、MMAudio、LTX Audio 等） | 原生 |

**入口和做法：**

1. **先盘点**：在锁定镜像里列出 `comfy/sd.py` 能构造的每种 `first_stage_model`、解码的 latent 形状和 decoder 结构（哪些有 GroupNorm 这类整图统计、哪些有时间维的因果缓存、哪些有注意力），以及用户实际用的模型。用户的优先级决定先做哪几种。
2. **第二层先行**：对还走原生的 VAE，先看第二层（`vae_ops.OpChunking`，实例级替换 `_conv_forward` 和注意力）能不能直接覆盖：多帧视频要处理 Conv3d 的时间维和因果缓存（`comfy.ldm.wan.vae` 的 `feat_cache`），`_native_reason` 里放开相应的情况，`estimate()` / `_probe` 要认识 5D 多帧的形状。
3. **第一层适配器**：每种结构一个适配器（接口在 `vae_engine.py` 的模块注释）：`match`（按真实结构识别）、`StripeAdapter` 子类（单元、`need_in` / `valid_out`、按 forward 数的内存模型、fp32 副本 / 参照解码、`variants` / `predict_seconds`），注册到 `vae.STRIPE_ADAPTERS`；需要整图统计的归一化在单元上挂 `NormRef`，引擎自动安排统计遍。
4. **先模拟再上真机**：在 `tests/alloc_sim.py` 加 meta 构造，确认 reserved ≤ 估算、预测峰值，然后写 bench 命令让用户在 CT 700 上测；耗时模型用实测拟合。
5. **测试**：照 `tests/test_vae_ldm.py` / `test_vae_stripe.py` 的结构写（随机权重的小 decoder，与原生整图比较，OOM 注入，自检失败注入），不需要模型文件；bench 的 CPU 冒烟用 `tests/make_synthetic_vaes.py` 的做法生成结构文件。
6. **节点和 Info 不用改**：新适配器自动受 VAE Settings 的各项设置管；方案类设置只对有方案的适配器有意义（`bound.schemes`）。新增的用户可见消息加进 `messages.py`（两种语言）。

## 5. 环境与工具

* **锁定镜像**：`docker.io/kyuz0/amd-strix-halo-comfyui@sha256:384aa1fecef6a841832e0d5552949977330308d8c25e212a94f5e8dfcc061cae`，ComfyUI 0.31.0 / 62b3c94，前端 1.48.7，torch 2.14.0a0 + ROCm 7.15。云端容器里 docker daemon 可能没起来：`sudo dockerd > /tmp/dockerd.log 2>&1 &`。
* **跑测试**：`MODELS=<目录> tests/docker_run.sh python tests/<test>.py`（仓库只读挂进镜像的 custom_nodes；`docker_run.sh` 传 `MONOLOAD*` 环境变量）。
* **合成模型**：`tests/make_synthetic_vaes.py OUT [--full]`（SDXL / Flux / Wan 结构的随机权重 VAE）；`tests/make_synthetic_checkpoint.py MODELS`（随机权重 SD1.5 checkpoint + UNet / TE LoRA，约 2 GiB，`test_lora_node.py` 用）。真实 SD1.5 不在云端：`test_release.py` 可以用合成 checkpoint 冒充（硬链接成 `v1-5-pruned-emaonly-fp16.safetensors` / `rubber_duck.safetensors`）。
* **网页实测**：在锁定镜像里起服务（`docker run -p 127.0.0.1:8188:8188 ... python main.py --cpu --listen 0.0.0.0`），Playwright（`/opt/node-tools/node_modules/playwright`，Chromium `/opt/pw-browsers/chromium-1194`）打开真实前端，或用 `/prompt` API 跑工作流看日志（第 5 项的泄漏就是这样复现的）。
* **CT 700 的事实**：`--gpu-only --bf16-vae`；AMD 上 `cudnn.enabled = False`，4D 卷积走 Slow2d（im2col + GEMM），5D 走 SlowDilated3d；VAE 注意力是 split；统一内存，看 GTT。用户终端是 `LANG=C`，中文日志显示成下划线（字节是正确的 UTF-8，不用改）。
* **`tests/bench_vae.py`**（README §9.7）：`--checkpoint` / `--vae`；`--res`；`--modes native,monoload,monoload-l2,monoload-r<N>,native2`，模式名后缀 `-g<S>`（方案）、`-b<GiB>`（预算）、`-w<MiB>`（工作区）；`--stripe-rows`；`--budgets`；`--gn-schemes`；`--fp32-ref`；`--profile-only`；`--json`。
* **`tests/alloc_sim.py`**（DESIGN §9.13.10、§9.14.6）：meta 设备上跑全尺寸 decoder，按缓存分配器的规则重放，复现了 51 个 CT 700 读数（≤ 0.02 GiB）。改内存相关代码先用它看。
* **盘点脚本**（DESIGN §9.16）：`tests/vae_inventory.py`（`--only` / `--no-trace` / `--json`；每种 VAE 的结构、路径、原生 / 第二层 / 第一层模拟峰值，带 60 GiB 设备上限；新适配器的 meta 构造可以从这里的 `KINDS` 抄）、`tests/probe_vae_gaps.py`（缺口和多帧第二层精度的小解码）、`tests/check_models.py`（CT 700 的命令 W）。
* **检查脚本**：`tests/check_vae_node.py`（命令 U）、`tests/check_lora_node.py`（命令 V，不带 `--lora` 时列文件）。

## 6. 代码地图

* `__init__.py`：插件入口；导出三张表（节点、显示名、`WEB_DIRECTORY`）；`MONOLOAD_DISABLE` 以外装 hotpatch / release / VAE 包装和 `LoraLoader` 名字包装；启动日志。
* `monoload/settings.py`：总开关、全局默认 `exact()` / `keep()`、`disabled()`。不导入 torch / ComfyUI。
* `monoload/messages.py`：消息表、`msg()` / `label()`、`MONOLOAD_LANG`。
* `monoload/hotpatch.py`：`ModelPatcher` 方法替换、`MonoloadRuntimePatch`（融合 / 逐位一致）、按 patcher 的 `_enabled` / `_active`、`_monoload_runtime` 模型标记。
* `monoload/release.py`：每个 prompt 结束后按 patcher 释放；`_ancestor_chains` / `_repoint_orphans`（两层不带 patch 的 clone 一起被回收时，把已加载模型指回活着的祖先，修 CT 700 的「memory leak」警告）。
* `monoload/lora_overrides.py`：LoRA 节点的设置、`resolve()`、`enabled()` / `merge_exact()` / `wants_release()`、`install_names()`。
* `monoload/vae.py`：VAE 入口和策略（`_decode` → `_managed_decode` → 预算 `choose_budget` / 默认 `choose_plan` → `_decode_layer1` / `_decode_layer2`）；`resolve_settings` / `settings_note` / `_Applied` / `_from`（来源）；`global_mode()`；`decode_record()` / `_MemProbe`（测量前先清分配器缓存）；`STRIPE_ADAPTERS`。
* `monoload/vae_ops.py`：第二层引擎（`OpChunking`、`_ConvChunker`、注意力 query 分块、`OpStats`）。
* `monoload/vae_engine.py`：第一层引擎（区间、`Plan`、统计遍、arena、`StripeAdapter` 基类、自检 `_SELFTEST`）。
* `monoload/vae_wan.py`、`monoload/vae_ldm.py`：两个适配器（LDM 的含 Flux 2 batch-norm latent，`LatentUnpatch`；`vae_ldm.scheme_positions`：A 不存，D 存 H/4 级输出，B 存 H/4 和 H/2，C 再加全分辨率各块的输入；耗时模型 `TIME_COEF`）。
* `monoload/vae_overrides.py`：VAE 节点的设置（不导入 torch / ComfyUI）。
* `monoload/info.py`：Info 节点的文字。
* `monoload/nodes/`：注册表 + 三个节点。`web/`：`monoload_info.js`、`monoload_i18n.js`。`locales/`：界面翻译。
* **小测试**（都不需要真实模型）：`test_entry.py`（开关组合，见文件头）、`test_master_switch.py`、`test_messages.py`、`test_info_node.py`、`test_release_chain.py`、`test_vae_node.py`、`test_vae.py`、`test_vae_ldm.py`、`test_vae_flux2.py`、`test_vae_stripe.py`、`test_dtype_paths.py`（默认和 `MONOLOAD_EXACT=1`）、`test_lora_node.py`（要合成 checkpoint）。需要真实 SD1.5 的：`test_lora_hot.py`、`test_quant.py`、`test_release.py`。

## 7. 提交记录（`dev` 上的合并）

| 内容 | 合并 |
|---|---|
| VAE 第一阶段（管理入口、第二层）/ fp32 参照 / 验收 | af9abc6 / 76e56a1 / 191f59b |
| 第二阶段（Wan 单帧第一层）及调整、arena、alloc_sim、收尾 | 4e54d20 / 725a010 / 85a5c6f / 5d668b6 / 13d2384 |
| 第三阶段（拆引擎、LDM 第一层） | aa07431 / 7059bdd |
| 默认 B、预算内最快 / 估算收紧 | 01377c4 / cc3b9d9 |
| VAE 节点 | 9b30154 |
| 总开关 / LoRA 节点 / Info 节点 / 多语言 | 7482eec / 48c9e54 / a7e187c / 19d7694 |
| UI 实测后的修正（控件顺序和精度、Info 全局表和不生效标注、方案说明、预算来源措辞、泄漏警告、首次解码测量） | 05ded8d |
| 第四阶段 4a：全部 VAE 的盘点（只有脚本和文档）/ 命令 W 识别修正 | 575fc46 / 87e6262 |
| 4b-0：SVD、2D latent 音频、像素空间走原生 | c89542a |
| 4b-1：Flux 2 VAE 走第一层 | 4fbaa22 |
| Flux 2 验收结果、自检峰值的分析和诊断脚本 | 本文件所在的合并（`git log --first-parent dev` 最上面一条） |

LoRA 部分更早的历史：3c473fa … 5bfbc8e（v1 文件格式 → v2 运行时合并 → 释放、fp8、融合 addmm），见 `git log --first-parent dev`。各阶段的设计和真机数据：DESIGN §9.12–§9.15、README §10。
