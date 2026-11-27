# Monoload 交接说明（video 分支：适配 ComfyUI v0.46.3、视频 VAE 推广，2026-11；之前：第四阶段 VAE 推广到全部图像 VAE、第五轮，2026-10）

给下一个对话用。到这里为止的功能都已在 CT 700 上验收；这份文档讲：现在是什么状态（§1）、节点和设置体系（§2）、必须遵守的规矩（§3）、第四阶段的盘点结果和下一步（§4）、环境和工具（§5）、代码地图（§6）、提交记录（§7）。细节在 README.md（使用、真机结果、bench 命令）和 docs/DESIGN.md（§9 VAE 设计，§10–§13 总开关 / LoRA 节点 / Info 节点 / 多语言）。

## 0. video 分支（2026-11，先读这一节）

**分支**：长期分支 `video`（从 dev 339b754 开出）。每项工作一个功能分支，从 video 开、`git merge --no-ff` 合回 video 并推送；不开 PR，不碰 `main` / `dev`（以后有空再从 video 往 main 提 PR）。每合进 video 一项就更新本节。

**底座**：ComfyUI **v0.46.3**（cab5f4e，2026-11-24），前端 1.59.6，comfy-aimdo 0.6.0，comfy-kitchen 0.2.41；ComfyUI-GGUF 用 kyuz0/ComfyUI-GGUF-H3 6d33de2。**不再兼容 0.31（62b3c94）**，下面 §1–§7 里 62b3c94 / 锁定镜像的说法是 dev 时代的历史。
真机：LXC 里的容器 `comfyui-latest`，仓库挂在 `/opt/ComfyUI/custom_nodes/monoload`，端口 8190；真机命令写成 `docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui-latest python tests/...`，由专门的验证线程跑。GPU 稳定可用约 **55 GiB**（60 GiB OOM，卡在宿主机 `ttm.pages_limit` 默认值）；LXC 内存只给 48 GiB（运行时什么都不经过内存，§3）。
云端：venv `/home/user/venv`（torch 2.15 CPU 可用），`COMFYUI_PATH=/home/user/ComfyUI`（v0.46.3）；只跑小测试、一次一个，不跑 `run_all.sh`。

**进度**
* **阶段 A：适配 v0.46.3（comfy-0.46-adapt，合并 8d89069）**——小测试全过，待真机冒烟。改了什么：
  * `VAE.decode` 现在把 `load_models_gpu` 和输出分配包在 `comfy.model_prefetch.pause_malloc_graph()` 里：Monoload 的四处 `load_models_gpu`（自检、第一层、探测、第二层）和输出分配照做（`vae_ops.malloc_graph_paused`）。
  * Flux 2 decoder 新的 `fused_norm` 路径（`Decoder.forward_fused` + `strip_conv`，CUDA + comfy-kitchen fp16/bf16 时启用；HIP 上只有 fp16 + fp16_conv3d，CT 700 的 `--bf16-vae` 不走）：托管解码期间临时关掉（`vae_ops.unfused_norms`），原生不动——融合内核绕过冻结统计量和 OpChunking，第一层的峰值也比它低。测试 `test_vae_flux2` 第 8 节。
  * 新的 `VAE.clone()`（VAE 精度节点、Select VAE Device 用）只复制固定的属性：Monoload 包了 `clone`，把 VAE Settings 节点的设置带到副本上。
  * 形状探测缓存的键加上 VAE dtype（`_probe_key`：通道、维数、dtype），换精度的副本不共用探测结果。
  * Wan 2.2 适配器支持 `patch_size` 1 / 2 和 4D latent：**Qwen Image 2.1 的 VAE**（`WanVAE` patch_size 1、temporal_kernel 1、z 64、RGBA 输出）也走第一层（`test_vae_wan22` 第 9 节）；结构签名不再含时间核（fp32 副本只留一片）。
  * v0.46.3 的 `pytorch_attention_enabled_vae` 在 AMD 上读 `gcnArchName`，gfx11 返回 False：**CT 700 上原生 VAE 注意力是 split**（Monoload 三种都分块，不受影响）；`vae_inventory` / `check_models` 模拟 gfx1151。
  * 测试桩跟着改：CLIP 的 `tokenizer_metadata`、SeedVR 新的 `Attention(query_dim, norm_num_groups, eps)`。
  * 核对过不用改的：`LowVramPatch` 新的 `prepare` / `memory_required`（`MonoloadRuntimePatch` 早已覆盖）；`load()` 现在会把形状变化的 key 烤进权重，Monoload 照旧报 `lora_shape_change`（没改）；"config" patch 变成 object patch；前端 1.59.6 仍不翻译 combo 选项，`monoload_i18n.js` 还需要。
* **LoRA：量化 safetensors / GGUF / 视频小参数（lora-quant-formats、lora-gguf、lora-video-models 三次合并）**——CPU 小测试全过，待真机。设计和进度在 `docs/GGUF-QUANT.md`。
  * 各种量化 safetensors（fp8 e4m3 / e5m2、int8_tensorwise、nvfp4、mxfp8、convrot_w4a4、asym_w4a8_int8、w6a8_int8；fp8 的三种来源：文件内 `comfy_quant` / 头部 `_quantization_metadata` / 旧 `scaled_fp8` 标记）：hotpatch 不用改，`tests/test_quant_formats.py`（66 项）锁住「在反量化临时权重上合并、不量化回去、无备份」。
  * ComfyUI-GGUF（UnetLoaderGGUF / CLIPLoaderGGUF，按结构找，`monoload/gguf.py`）：量化层合并进 GGMLLayer 反量化出的临时权重（exact = 原生 GGUF 的算法逐位一致，默认 fused），F16 / F32 层也运行时合并、不备份（原生会备份）；hook LoRA 也支持（原生 GGUF 对量化层不应用 hook）；保存 GGUF 模型报 `gguf_save`；Info 节点多一行 GGUF 统计。每个 prompt 开始重新找一次（自定义节点可能比 Monoload 后加载）。`tests/test_gguf_lora.py`（38 项）。
  * 不是 comfy.ops 层 weight / bias 的参数（Wan `modulation`、LTX `scale_shift_table` 等、纯 torch 层）：`MonoloadParamSwap`，所属模块 forward 期间换上合并好的临时参数，出来换回，无备份，原生算法逐位一致。**原生 v0.46.3 加载时静默丢弃这些 key**（`_load_list` 跳过非叶子模块），所以带这类 key 的 LoRA 和原生比 max|Δ| 不为 0（已定：不加「跟原生一样丢掉」的开关）。`tests/test_lora_param_swap.py`（44 项）；`test_lora_hot` 里原来的「纯 torch 层报错」改成「换参数」。
  * 真机：`test_quant.py` GPU 复测默认 / `MONOLOAD_EXACT=1` 各 8 项全过、逐位一致，容差没放宽。之前 GPU 上的失败是测试参照的问题（fp8 参照在 CPU 反量化，CPU / GPU 的反量化 kernel 末位舍入不同），参照改成在计算设备上反量化（Merge video-quant-ref，GGUF 线程 611a7a6）；`bench_lora.py` 逐层探测像 `cast_bias_weight` 一样先反量化量化参数；新增 `tests/make_model_lora.py`。
  * K-quant 等云端造不出的 GGUF 类型：`tests/check_gguf_kquant.py`（真机逐层对原生），计划见 `docs/GGUF-QUANT.md` §6.2。GGUF VAE 没有加载器（H3 和核心都没有），H7 已取消。
  * `bench_lora.py` 加了 `--unet-gguf` / `--clip-gguf` / `--latent C,T,H,W` / `--batch`，每个组合报 GTT 峰值（20 ms 采样）。
* **H1 / H3 / H6：多帧第二层入口（video-l2-entry，合并见 §7）**——Wan 2.1 / 2.2 和 LTX（0.9.x 卷积 VAE 的 chunked io、LTX 2.5 扩散 decoder `CausalDiffusionVAE`）的 5D latent 走第二层，峰值靠实测探测 `vae._probe_mf` 定上界（16、32 两个 latent 尺寸在 64 MiB 工作区上的 allocated 连线 × `MF_MARGIN` 1.25，加上在解码自己工作区上再跑一次量到的碎片），输出块直接写进预分配的输出缓冲；F 组的扩展点 `vae_ops.EXTRA_CHUNKERS`；C 组的 CogVideoX 登记（`vae_video`，可选导入 `vae_cogvideo`，没合进来时照旧原生）；`bench_vae --frames N`。真机：Wan 2.1 720p×81 原生 reserved 18.90 → 9.84 GiB，估算 13.77，像素 max 0.0166（PSNR 60.8），热解码慢 9%。最终探测的真机读数在排队（DESIGN §9.35 表）。
* **多帧探测按 latent 截断；LTX-2.5 扩散 decoder 等 F 组（video-probe-clip，合并见 §7）**——最终版探测的真机读数（DESIGN §9.35 表）：Wan 2.2 1280×704×121 原生 reserved 39.68 → 10.65 GiB，估算 11.96 是上界，热跑快 20%；LTX-2 卷积 768×512×97 原生 59.29（碎片）→ 8.33，但冷跑高过估算 7.86（多出的是 32 × 32 的探测本身，比 16 × 24 的真解码还大）；LTX-2.5 扩散 decoder 第二层什么都切不到，冷跑 28.14 是原生 9.91 的 3 倍。改了两处：探测尺寸按解码自己的 latent 截断（截断形状进缓存 key）；`CausalDiffusionVAE` 只有 F 组的 `vae_ltx`（可选导入）在时才托管，没有时多帧原生。CPU 回归全过；LTX-2 截断后的真机复测在排队。H4 的 GPU 测试（`test_vae_norm_g`、`check_norm_reduce --pixelnorm`）真机全过（DESIGN §9.37）。三份原生基线写进 `/srv/comfy-latest/results/native/`。
* **H2 / H5：引擎的时间轴（video-engine-time，合并见 §7）**——`vae_engine` 认帧：单元的 `tscale`、适配器的 `tdim` / `ckpt_planes` 决定每行平面数；适配器给 `time_chunks` 就按「条带为主」跑时间块，每条带带着自己的因果卷积缓存跨块；统计遍不能和时间块同用（B 组的 HunyuanVideo 1.0 按块统计自己实现）；视频自检至少 3 个时间块。现有图片适配器不受影响，CPU 回归全过。各组第一层接上（DESIGN §9.36）。
* **H4：方案 G 扩到 Refiner 的 RMS_norm 和 LTX 的 PixelNorm（video-norm-g，合并见 §7）**——`OpChunking` 里把通道 > 510 的 `vae_refiner.RMS_norm` 和每个 `PixelNorm`（按调用判断通道数）换成分组归约，躲开 ROCm 的 32 KiB / 位置归约临时缓冲；小通道逐位相同。现在只第二层用到，各组写第一层时照 Wan 2.2 把 `grouped_norm_bytes` 加进单元（DESIGN §9.37）。H7 取消（PLAN-video §5）。
* **盘点（video-inventory，合并见 §7）**——`tests/vae_inventory.py` 跟到 v0.46.3（新 VAE 四种、托管多帧走 Monoload 自己的路和 H1 估算），结果 DESIGN §9.39：托管的 Wan 2.1 / 2.2、LTX 0.9.0 / 2 的估算都是上界；LTX 扩散 decoder、RelayFlash 在 meta 上追踪不了，要真机。
* **E 组 SVD：整批第二层不合，保持原生**——模拟里只省 20%（25.71 → 20.72 GiB），而且探测估算 20.03 给不出上界（整批时碎片随尺寸涨）。代码留在分支 video-svd-l2（DESIGN §9.38）。
* **G 组：自带分块的 transformer VAE（video-g-selftiled）**——MiniMax H3（视频原版 `ViT3DDecoder`、Sol `NonCausalViT3DDecoder`、MiniMax-H3-Image / P-H3-Image）、SeedVR2、RelayVSR Flash 都声明 `comfy_has_chunked_io`，`_native_reason` 先判这一条，一律原生，Monoload 只透传，**不做第一层 / 第二层**。理由和真机数据在 `docs/PLAN-video.md` G 组：MiniMax 原版 1080p×107 帧总 reserved 11.77 GiB（权重 4.85、输出 2.50，块内工作量 <1 GiB 且不随帧数涨），开/关 Monoload 逐位相同。新增 `tests/test_vae_selftiled.py`（锁路由，含 H1 放开 5D 以后；真机 19 项全过）、`tests/bench_selftiled.py`（原生峰值、RSS/GTT、SeedVR2 的缓存放哪、`--gguf` 反量化加载 GGUF VAE、`--relay`）。Sol、SeedVR2、RelayVSR 也已真机测过原生峰值，结论不变：Sol 1080p×107 帧 ≤ 14.63 GiB，SeedVR2（bf16）1080p×33 帧 reserved 43.11 GiB、4K 由 ComfyUI 直接改走它自己的分块，RelayVSR x4 4K ≤ 7.70 GiB。现在 `_native_reason` 里 `chunked_io` 的判断排在多帧白名单之前，只有 `chunked_io_managed`（LTX `VideoVAE`，按类型精确匹配）放行；**以后改这里要保持这几个类不被放行**，路由测试会拦。
* **视频第一层 A 组：Wan 2.1 / 2.2 多帧（video-wan-l1，合并见 §7；DESIGN §9.40）**——`monoload/vae_wan_video.py`：T > 1 的 Wan 2.1 VAE（16 ch、8×）和 Wan 2.2 VAE（48 ch、16×）走第一层，条带为主：`conv2` 和前缀按时间块整段算进存档 `[1, C, T, h, w]`，每个输出行条带带着自己的 `feat_cache` 依次跑全部时间块（原生的模块实例和调用方式，Wan 2.1 的 2 帧切片照做），每块直接写进预分配的 5D 输出。用引擎的时间轴（H2：`time_chunks` / `chunk_planes` / `stripe_state` / `call_unit` / `unit_cache`，引擎的 `Plan` 和条带），单元 / halo / 条带划分沿用单帧适配器；Wan 2.1 时间上采样之后不再切 2 帧一片（引擎的链是线性的，数值相同，计划按 8 帧算）。自检跑真正的视频路径（fp32、3 个时间块）。小测试 57 项全过（对原生相对误差 ≤ 1.7e-6，自检、注入错误、未注册时逐位一致）。真机（DESIGN §9.40 有表）：第一层对第二层，Wan 2.2 在 1280×704×121 上 3.54 对 8.91 GiB、不慢，估算全是上界；Wan 2.1 同档 4.34 对 8.00 GiB，但慢 20%，720p 估算少算 0.65–1.0 GiB。81 → 121 帧第一层只涨 0.1 GiB 以内。**决定**：Wan 2.2 多帧走第一层，Wan 2.1 多帧先留第二层（`WAN21_ENABLED = False`；第二层 1080p 约 22 GiB 也够，查清估算之后再开，办法见 §9.40）。
  * **已注册（video-wan-reg，合并见 §7）**：`vae.py` 照 C 组的可选 import 把 `vae_wan_video` 加进 `STRIPE_ADAPTERS`（只接 T > 1）。真机冒烟 Wan 2.2 1280×704×121（`MONOLOAD_VAE_FIT_CAP=55G`）：走第一层，6 条 × 118 行、3 个时间块，首次自检 5.4e-06 通过（75.7 s）；冷跑含自检 alloc +4.32 / reserved +4.93 GiB，低于自检上界 7.83（不含自检的估算 4.03），原生估算 13.43。Wan 2.1 多帧仍关（`WAN21_ENABLED=False`）。
  * 真机命令：`docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui-latest python tests/test_vae_wan_video.py --wan21 /opt/ComfyUI/models/vae/wan_2.1_vae.safetensors --wan22 /opt/ComfyUI/models/vae/wan2.2_vae.safetensors --sizes 832x480,1280x720 --frames 81,121 --native-dir /srv/comfy-latest/results/native --no-native --no-small --route 1280x704x121 --route-l2 2.1,2.2`（`--mem-history` 打印分配器轨迹）。预期：Wan 2.2 全部 PASS；Wan 2.1 的 720p 估算现在会 FAIL（已知，所以默认关着）。
* **B 组：Hunyuan（分支 claude/project-thread-glsomb，合并见 §7；DESIGN §9.41）**——HunyuanVideo 1.0（Kandinsky 5 同一个）/ 1.5 的多帧第二层：按 decoder 结构认（first-stage 是通用的 `AutoencoderKL` / `AutoencodingEngine`），按原生的 2 帧一块交出（c8f481e）。HunyuanVideo 1.5 / HunyuanImage 2.1 Refiner 官方文件的第一层 `monoload/vae_hunyuan15.py`（4e348fa）：前缀整段跑到 `up[0]`，之后条带为主按 2 帧时间块跑，carry 每条带每单元一份；单帧（Refiner）没有时间块。真机 480p×81 第二层 10.43 GiB 不慢、第一层 2.54 慢 56%，所以第一层只作后备（`prefer_layer2 = True`，配合 video-l2-frag 的路由；估算少算的 carry 碎片归引擎修，video-carry-store）。Refiner 2048² 第二层 10.06 GiB 15.0 s、第一层 0.77 22.6 s（估算是上界），单帧也第二层优先。**已注册**：`vae.py` 照可选 import 把 `vae_hunyuan15` 加进 `STRIPE_ADAPTERS`（video-b-reg，只接 5D）；B 组真机 `test_vae_hunyuan15` 42 项全过（600885c）。`prefer_layer2` 的路由在 video-l2-frag 合入之前不生效（之前第一层照常可用、只是慢）。自检上界补上参考解码留在缓存里的空段（dc91365，真机复测中）。HunyuanVideo 1.0 / Kandinsky 5：**第二层已够，不做第一层**（原生 480p×81 就 OOM；第二层 480p×81 16.90、720p×129 37.66 GiB，探测估算两档都偏低，归 video-l2-frag 核对）。**待**：HunyuanVideo 1.5 720p×81 真机（第一次跑遇到 GPU 硬件异常，已重排）（DESIGN §9.41）。
* **多帧探测落盘（video-probe-cache，合并见 §7；DESIGN §9.44）**——第二层多帧探测（`_probe_mf`）的结果写进 `<user>/monoload/probe_cache.json`（`MONOLOAD_PROBE_CACHE` 改路径，0 关），key 含权重指纹、latent 布局、workspace、显卡、torch 版本、Monoload 源码哈希。重启后不再冷跑探测（Wan 2.1 832×480×33 冷跑 21.0 → 14.3 s）。LTX-2.5 扩散 decoder 的第二层估计直接用 `vae_ltx.plan()`，不探测（768×512×97 冷跑 109.7 → 29.9 s，估算 6.10 ≥ reserved 5.59）。「第二层估计 ≥ 原生估计就走原生」不做：原生估计在视频 VAE 上低好几倍，反例在 §9.44。
* **视频条带 carry 缓冲（video-carry-store，合并见 §7；DESIGN §9.45）**——`vae_engine.CarryStore`：每遍条带一块缓冲装各单元的 carry，碎片消掉，估算成了上界：HunyuanVideo 1.5 832×480×81 第一层 2.54 → 2.41 GiB（估 2.69）、Wan 2.1 1280×720×81 3.93 → 3.57（估 4.12）。HV1.5 的首次自检峰值仍超它的上界（+5.46 对 4.70，改前 +5.20），归 B 组复核；Wan 2.1 多帧要不要打开由 A 组按这个结果再定。
* **F 组：LTX（分支 claude/project-thread-b4e366，合并见 §7；DESIGN §9.42、PLAN-video F 组）**——卷积 VAE（LTX 0.9.x / LTX-2）**只走第二层，不做第一层**：LTX-2 1280×704×97 原生 reserved 56.76 → 7.99、1920×1088×193 55.95 → 16.14 GiB，时间不变。LTX-2.5 扩散 decoder（`CausalDiffusionVAE`）由新的 `monoload/vae_ltx.py` 经 `vae_ops.EXTRA_CHUNKERS` 接管（`vae_video` 已可选导入它，导入成功才把这个类放进多帧白名单）：算子级邻域注意力按 T 切等长段（光环 k_t − 1，结果和整块相同）、SwiGLU 按工作区分块、精简的 decoder forward。真机 768×512×97 走正常入口 reserved 9.84 → 5.52，热跑 17.9 → 31.3 s（冷跑 109.7 s，含两次探测解码）；1920×1088×97 50.15 → 26.07，106 → 191 s；1920×1088×193 reserved 39.23（主线 55 以内），367 s；fp32 自测 ≤ 5.6e-6。（`comfyui-latest` 容器的环境里没设 `MONOLOAD`，没设就是开（`settings._master_from_env`）；bench 的 monoload 档日志里应有「[Monoload] VAE decode … -> layer」那一行。）
  * 真机命令：`docker exec -e MONOLOAD=1 -w /opt/ComfyUI/custom_nodes/monoload comfyui-latest python tests/bench_ltx.py --vae ltx-2.5-video-vae-bf16.safetensors --res 768x512 --frames 97 --modes monoload,native --repeat 2 --compare`（`--modes lean` 直接装 `vae_ltx`，不经探测）；精度 `python tests/check_ltx_dd.py --vae ltx-2.5-video-vae-bf16.safetensors`（fp32，默认三档小尺寸，估算低于实测 reserved 会打 `!!`）。
* **D 组 Mochi / Cosmos 1.0：逐帧流式的第二层（分支 claude/project-thread-gjyfzb，待适配线程 --no-ff 合；DESIGN §9.43）**——`vae_mochi.py` / `vae_cosmos.py` 的 `chunks` 按帧深度优先、因果卷积带缓存，登记已在 video（855469a，可选导入）。两种都不做第一层。真机（2026-11-27，reserved Δ）：Cosmos 1280×704×121 原生 30.18 → **2.96 GiB**（估算 4.58，warm 8.5 → 10.4 s，像素 max 0.0215 / PSNR 54.8，bf16 噪声级）；Mochi（VAE 是 fp16）848×480×85 原生 OOM → tiled，Monoload **5.16 GiB**（估算 6.83，warm 67 s，0 次 OOM 重试）；`test_vae_stream` 22/22（fp32 相对误差 ≤ 3.4e-6）；`check_group_d_zero` 4/4（MONOLOAD=0 两种都对原生逐位一致）。真机命令：`docker exec -e MONOLOAD=1 -w <仓库> comfyui-latest python tests/test_vae_stream.py`、`tests/check_group_d_zero.py`、`tests/bench_vae.py --vae cosmos_cv8x8x8_1.0.safetensors --res 1280x704 --frames 121 --modes native,monoload`（Mochi：`mochi_vae.safetensors --res 848x480 --frames 85`）。
* **阶段 B**：**`docs/PLAN-video.md` 已写（合并见 §7）**——共用钩子 H1–H7、各 VAE 的第一层方案和 A–H 分组、模型清单与单次峰值（> 55 只有 MiniMax H3 完整版 bf16 DiT）、开工顺序。`docs/PLAN-video.md`（每种视频 VAE + 新图像 VAE 的第一层方案，分组可并行，含 safetensors / GGUF；覆盖 v0.46.3 里量化能跑的全部视频模型，扩散模型和文本编码器分开算；单次峰值 ≤ 55G 为主线，> 55G 单列「等宿主机调 ttm 后再验」；超大模型不分批会 OOM 时才把 TE 和视频模型分开加载；不做 MAGI-2）。
* **阶段 C（进行中）**：多帧第二层入口已合（上面 H1）；剩下feat_cache 和因果时间卷积；SVD 整批第二层；LTX 分块 IO 的输出缓冲（音频仍原生）；`bench_vae --frames`；共用的第一层视频钩子（参考 vae-refiner-layer1 f51a6bc）。每项一个功能分支。

**待办（视频这一轮之后）**：形状变化的 LoRA 的运行时支持——协调者替用户定：这次冲刺先保持报错（原生 v0.46.3 会烤进权重），视频做完再考虑。

**未决（等用户定）**：Refiner 第一层分支 vae-refiner-layer1；polish-after-ui-test 的复测（换底座后一起测）。

## 1. 当前状态

仓库 github.com/sketchain/Monoload，开发分支 `dev`（不碰 `main`，`main` 停在 5d9ac63，即 PR #2 合并的 dev）。ComfyUI 插件，两部分功能互相独立：

* **LoRA 运行时合并**（`monoload/hotpatch.py`、`release.py`）：所有 `ModelPatcher` 的 LoRA 不写进权重、不留备份，每一层计算时在临时权重上合并；默认融合 fp16 `addmm_`，可选逐位一致；每个 prompt 结束后释放 LoRA，底模常驻。CT 700：SDXL 比原生少用约 4.7 GiB（8.2 对 12.9 GiB），每步 fused 0.73 s、exact 0.90 s、原生 0.654 s（19d7694，命令 V）。
* **VAE 解码降峰值**（`monoload/vae*.py`）：包装 `comfy.sd.VAE.decode`。
  * 第二层（所有 VAE 的图像解码）：卷积按输出行分块，注意力按 query 分块，整图语义不变。
  * 第一层（认得的 decoder）：低分辨率前缀整图算出存档，之后按输出行条带倒推重算；整个解码在一个 arena 里。适配器：Wan 2.1 单帧（`qwen_image_vae`，`vae_wan`）、Wan 2.2 单帧（`vae_wan22`，已验收；> 510 通道的 RMS_norm 在托管解码里分组求范数，DESIGN §9.28 / §9.29）、LDM decoder（SDXL / SD1.5 / SD3 / Flux `ae` / Flux 2，`vae_ldm`，GroupNorm 整图统计量用统计遍求，方案 A / D / B / C，默认 B）、HunyuanImage 2.1（`vae_hunyuan`，同样的方案，已验收）。
  * 不设预算：默认条带策略；设了预算（环境变量或节点）：预算内预计最快（第二层放得下就用第二层）。
  * 自己的内存估算交给 `load_models_gpu`；OOM 只缩块重试；从不退回 tiled。
* **三个节点**（分类 `Monoload`）：Monoload LoRA Settings、Monoload VAE Settings、Monoload Info（§2）。
* **多语言**：节点界面中 / 英跟随 ComfyUI；日志和报错默认英文，`MONOLOAD_LANG=zh` 中文。

CT 700（Strix Halo，gfx1151，62.5 GiB 统一内存）4K 解码的 GTT 增量：

| VAE | 原生 | 第二层 | 第一层 |
|---|---|---|---|
| SDXL / Flux `ae` | 52.5 GiB / 11.9 s | 15.0 GiB / 9.9 s | 默认 B 2.17 GiB / 42 s（A 1.10 / 75，D 1.52 / 58，C 4.70 / 36） |
| `qwen_image_vae` | 59.2 GiB / 7.9 s | 9.6 GiB / 6.9 s | 0.87 GiB / 8.5 s |

验收情况：VAE 三个阶段、预算策略、VAE 节点、总开关、LoRA 节点、Info 节点、多语言都已在 CT 700 上通过（第二轮审查之后 e924a77，README §10.8；图像补齐最近一次 0593f27：Wan 2.2 方案 G 之后的 GA–GK 和 `run_all.sh` 43 组 1614 项，README §10.13）。polish-after-ui-test（05ded8d）修了 UI 实测发现的六处问题，待 CT 700 复测（README §9.10）。

**第四阶段 4a（vae-inventory，575fc46；命令 W 的识别修正 87e6262）**：只加了盘点脚本和文档，插件行为没变。结论：用户实际在用的三个 VAE（SDXL、Flux `ae`、`qwen_image_vae`）的图像解码**已经全部走第一层**；其余 VAE 的结构、现在的路、模拟峰值和建议顺序见 §4 和 DESIGN §9.16；发现现有代码的 4 处缺口（SVD 结果被改变、2D latent 的音频 VAE 被管理且 ACE 的估算约 1 PiB、像素空间被管理、TAESD 小图第二层反而更高）。用户已定顺序（§4.0）。

**第二轮代码审查（基准 main 5d9ac63，ComfyUI 62b3c94，2026-10）：已完成并在 CT 700 上验收**（8c5cb37，README §10.8）。清单 01–20 全部改完；21、23 按用户的决定做了，22 不做（§4.0）。最要紧的是 01：默认设置下 `CheckpointSave` / `ModelSave` / `CLIPSave` 以前会把 LoRA 悄悄丢掉、存成底模；CT 700 上 SDXL + Smooth Booster 存出的 checkpoint 与原生 2515 个张量逐个相同（也与整进程 `MONOLOAD=0` 相同）。验收之后又改了：`run_all` 的「VAE 接口变了」那组在容器里没拿到 `TEST_BROKEN_VAE_API`、实际跑的是普通入口测试，已修（eee564c）；`bench_vae` 没跑原生时写「native not run」（17880b8）；21 / 23 见 §4.0。**这些也已在 CT 700 上复查通过**（e924a77，README §10.8 复查）：`run_all.sh` 41 组 1490 项 0 失败 0 跳过，「VAE 接口变了」那组 11 项、有 NOT managed 警告；命令 AE 不变；命令 Z（Flux 2 预算 3G / 1.5G）六档选择与 §10.5 相同，每档第一次（探测 + 自检 + 解码）都 ≤ 估算。

**4b-1（vae-flux2-layer1，4fbaa22，CT 700 验收通过）**：Flux 2 VAE（`AutoencoderKL` + batch-norm latent）走第一层，复用 LDM 适配器：latent 的反归一化和 2×2 还原是前缀第一个模块（`vae_ldm.LatentUnpatch`），引擎加 `decoder_hw` 钩子在 decoder 的分辨率上做计划。模拟与 Flux `ae` 相同（4K 默认 B 2.18 GiB）。DESIGN §9.18，测试 `tests/test_vae_flux2.py`，真机命令 README §9.12。

**图像补齐（2026-10）——CT 700 结果（dev 189741e，README §10.9 / §10.10；Wan 2.2 方案 G 之后 0593f27，§10.13）：两种都已验收**：
* **HunyuanImage 2.1：验收通过**（README §10.9）。XH / YH / ZH 与预测一致，8 次自检通过、峰值 1.43 ≤ 上界 1.49 GiB。
* **Wan 2.2：验收通过**（CT 700，dev 0593f27，README §10.13）。第一次测（README §10.10）默认 128 行和更矮的条带超估算，来源是 ROCm 上 `F.normalize` → `torch.linalg.vector_norm` 的归约内核：每输出跨多块时申请 32 KiB / 位置的临时缓冲（ROCm 把每输出的块数至少设成 64），1024 通道、位置数是 4 的倍数且 ≤ 40960（gfx1151 target 320）时触发（命令 N，README §10.12）。**方案 G**（用户定）：托管解码期间（`OpChunking`：第一层前缀 / 条带、自检、第二层；原生不动）把通道 > 510 的 Wan `RMS_norm` 实例 forward 换成分组求范数（每组 ≤ 510 通道，fp32 里合并）；Wan 2.1（384 通道）不换；临时量 1 MiB + (4G + 4 + 2e) B / 位置算进模型；alloc_sim 的跟踪器模拟这个缓冲（不用 G 时复现旧读数）。复测 GA–GK：第一层 4K 默认 1.04 / 1.34 / 估算 1.35（以前 1.49 / 3.27），2688 0.72 / 0.92 / 0.94，1344 0.45 / 0.45 / 0.59，fp32、各条带高度、预算、节点、D1 全部 ≤ 估算、与模拟一致；精度（PSNR 对原生 57–59 dB）、耗时不变；Wan 2.1 / HunyuanImage / SDXL 不变；`run_all` 43 组 1614 项 0 失败。注意力块的模型比实测多算 196 MiB（偏保守，记录在案，不改）。

图像补齐（实现，2026-10）：
* **Wan 2.2 单帧（vae-wan22-layer1，合并 d6473d3）**：`vae2_2.WanVAE`；带 `DupUp3D` 捷径的上采样块整块是一个 ×2 单元（halo 13，逐块测过是精确边界），最后 `unpatchify` 也是单元。4K 模拟：原生 34.2 → 第二层 19.8 → 第一层 1.34 GiB。自检的 fp32 副本只留每个 CausalConv3d 时间核的最后一片（单帧只读它；0.79 GiB 而不是 2.2 GB）。顺带修了第二层估算（`decode_retained`：`WanVAE.decode` 单帧也把每个卷积的输入克隆进 feat_cache，4K 13.5 GiB；以前估算 10.0 < 模拟 19.8）。DESIGN §9.23，测试 `test_vae_wan22.py`（50 项），真机命令 README §9.16（W / XW / YW / ZW）。
* **HunyuanImage 2.1（vae-hunyuan21-layer1，合并 838f7fc）**：LDM 的块和 GroupNorm 统计遍，多了第一步 `conv_in + repeat`、`PixelUnshuffle2D`（depth-to-space ×2，halo 2）、6 级；方案按结构定义（D = H/16，B = H/16…H/2）。4K（3840×2176）模拟：原生 38.1 → 第二层 11.1 → 第一层 B 2.13（A 0.74、C 4.67）GiB。自检几何按适配器（`selftest_size` 12）；第二层估算按 6 份激活（`activation_copies`；以前 4K 10.06 < 模拟 11.10）。Info 节点有 HunyuanImage 自己的方案说明，`gn_scheme` tooltip 改成按结构描述。DESIGN §9.24，测试 `test_vae_hunyuan.py`（57 项），真机命令 README §9.17（W / XH / YH / ZH，4K 用 3840×2176）。
* **HunyuanImage 2.1 Refiner（第五轮，见下面「第五轮」）：** §9.25 的分析对错了 sd.py 的分支——官方文件是 `refiner_vae=True` 的 decoder（与 HunyuanVideo 1.5 相同，单张图 1 帧、RMS norm），不是分析的 `refiner_vae=False`（DESIGN §9.31）。
* **用户定的（两种新 VAE 共同，接受的行为）**：自检不改。fp32 自检峰值（Wan 2.2 1.51、HunyuanImage 1.41 GiB）比多数尺寸的解码本身大；预算低于自检上界（Wan 2.2 1.61、HunyuanImage 2.1 1.49 GiB）时第一次解码报错是接受的行为。**任何方案都不许放到 CPU 上。**
* **Refiner：先不动**，等这一轮（Wan 2.2 显存）结束再说。（第五轮已做，见下。）

**第五轮（2026-10，基于 dev 907ef25）：**
* **一 Refiner 第二层估算（vae-refiner-l2-estimate，合并 f7e1aa2）：** `vae_hunyuan.decoder_terms` 对 `vae_refiner.Decoder(refiner_vae=False)` 取 6 份激活（alloc_sim `--model hyr`：2048² 模拟 23.44 / 以前估算 18.05，扫 50 个尺寸 × 4 种工作区 + fp32 最多要 5.67 份）。**之后发现官方 Refiner 文件不是这一支**（docs-refiner-file，合并 b426588）：官方文件的 decoder 是 `refiner_vae=True`，第二层估算本来就是上界（alloc_sim `--model hv15`：2048² 10.46 / 14.43，最多要 2.72 份），不用改；6 份那条只对 `refiner_vae=False` 生效。DESIGN §9.30 / §9.31，README §9.21（命令 W / RA / RB）。
* **二 Refiner 第一层：按用户的规格（`refiner_vae=False`、条带带 4 帧、跨帧 GroupNorm 方案、引擎的「每行平面数」）做完、测试过（`test_vae_refiner` 69 项，alloc_sim 48 个配置 reserved ≤ 估算），但官方文件用不上它，所以留在分支 `vae-refiner-layer1`（f51a6bc），没有合并，等用户定**（分支上的 DESIGN §9.34）。官方文件的第一层只做了分析（DESIGN §9.31）：和 Wan 2.2 单帧同类（无整图统计、无方案），要先定方案 G 是否扩到 `vae_refiner.RMS_norm`。
* **三 视频（video-analysis，合并 565e73b，只有文档）：** DESIGN §9.32（每种视频 VAE 的解码路径、原生峰值来源、第二层怎么套、alloc_sim 的原生 / 第二层、现有估算不是上界、方案 G 的影响、建议顺序），README §9.22（文件清单、bench 命令；bench 要先加 `--frames`，没加）。
* **四 Flux 2 dev（flux2-dev-run，合并 2219844）：** `tests/run_flux2_dev.py`，README §9.23 / DESIGN §9.33；CPU 冒烟测试过，没在真模型上跑。
* **用户定的（第五轮）：** Refiner 自检约 3 GiB：预算低于自检上界时第一次解码报错，**暂时接受（先不管，不是定死）**；不许放到 CPU 上。默认方案和条带目标等真机数据。（这两条当时针对 `refiner_vae=False` 的设想；官方文件的第一层如果做，自检预计约 1.5 GiB，DESIGN §9.31。）

**第二轮其余各项（2026-10，基于 dev 189741e 的真机日志；CT 700 复查通过，dev 1dc61b2，README §10.11：`run_all` 43 组 1603 项 0 失败 0 跳过；方案 G 之后 0593f27：43 组 1614 项，README §10.13）**：
* 三 `test_entry.py` 的适配器列表改成 `vae_wan, vae_wan22, vae_ldm, vae_hunyuan`（`run_all.sh` 的 9 组入口测试本地都过：17 / 17 / 17 / 8 / 17 / 17 / 17 / 17 / 11 项）。
* 四 `check_vae_node.py` 先 `load_models_gpu` 再测（以前第一行含整个权重：Wan 2.2 1.31、HunyuanImage 0.755 GiB）。
* 五 `bench_vae.py` 汇总表的内存各列和估算取自同一次运行（增量最大的那次），是冷启动那次时标 `[cold]` / `[cold: incl. the first-use self-test]`。
* 六 没有 GroupNorm 方案的 decoder（Wan 2.1 / 2.2、第一层不认的）：解码日志的设置不写方案这一项，节点 / 环境变量设了时写「not applicable」，Info 节点标「不适用」，不显示方案说明（`vae.gn_schemes_apply`，适配器的 `has_schemes`）。HunyuanImage 的方案数字改成实测。
* 七 第二层估算的 retained / copies 按 decoder 的类取（适配器的 `decoder_terms`），第一层拒绝（forward hook、`vae_options`）时也算：以前 Wan 2.2 4K 少算 13.47 GiB、HunyuanImage 少 2 份激活（DESIGN §9.27）。

**4b-0（vae-coverage-fixes）**：盘点的缺口 1–3 已修（DESIGN §9.17）：SVD 这类在 batch 各帧之间混合的 decoder、2D latent 的音频 VAE、没有可分块算子的像素空间 VAE 都走原生；`_native_reason` 的理由改走消息表。缺口 4（TAESD 小图）不在范围内。

## 2. 节点和设置体系

**规则（用户定）：环境变量 = 全局默认值；节点上明确选的值只对那一个模型 / VAE 生效，而且总是压过全局。** 逐项判断：节点上明确选的 > 高级环境变量 > 内置默认；节点上选「跟随全局」（存储值 `default`）的项继承全局。

* **总开关 `MONOLOAD`**（`settings.py`）：不设 / `1` = 默认策略；`0` = 全局原生，钩子照装但直通，没有节点的工作流与原版逐位一致（`test_master_switch.py`），只有节点明确开启的模型 / VAE 走 Monoload。`MONOLOAD_DISABLE=1` 什么都不装，节点原样透传。
* **高级环境变量**（README §13）：`MONOLOAD_EXACT`、`MONOLOAD_KEEP_LORA`、`MONOLOAD_DISABLE_VAE`、`MONOLOAD_DISABLE_VAE_STRIPE`、`MONOLOAD_VAE_BUDGET`、`MONOLOAD_VAE_GN_SCHEME`、`MONOLOAD_VAE_STRIPE_ROWS`、`MONOLOAD_VAE_WORKSPACE`、`MONOLOAD_LANG`。bench 和测试仍用它们配置。
* **Monoload LoRA Settings**（`lora_overrides.py`、`nodes/lora_settings.py`）：MODEL（必接）+ CLIP（可选）→ clone；`mode`（default / enable / native）、`merge`（default / fused / exact）、`after_prompt`（default / release / keep）。设置在 `model_options["monoload_lora"]`，每次 clone 都复制，所以放在 LoRA 加载器前后都行；设置和上游不同的 clone 换 `patches_uuid`。hotpatch 按 patcher 决定接管和合并路径（`_enabled`、`MonoloadRuntimePatch.exact`），release 按 patcher 决定释放。原生模型默认保留。
* **Monoload VAE Settings**（`vae_overrides.py`、`nodes/vae_settings.py`）：VAE → 共享权重的副本（`copy.copy`，属性 `_monoload_vae_settings`）；控件顺序 `budget`（default / unlimited / custom）、`budget_gib`（精度 0.01，只在 custom 时用）、`gn_scheme`、`stripe_rows`、`mode`（default / auto / layer 2 only / native）。`vae.resolve_settings` 逐项取值和来源，`_Applied` 在解码期间换进全局设置（含来源 `_SETTINGS["src"]`，日志和报错据此写「来源：节点 / 环境变量」并按来源给建议）。dev 不做旧工作流兼容（用户定）。
* **Monoload Info**（`info.py`、`nodes/info.py`、`web/monoload_info.js`）：输入 vae / model / images 都可选，文字显示在节点框里并从 STRING 输出；每次运行都刷新。内容：版本 / commit / 总开关；VAE 的设置和来源（当前模式下不生效的项会标出来）、这个 VAE 对象上一次经 `VAE.decode` 的解码记录（不含 `VAEDecodeTiled`：`decode_tiled` 不经过 `VAE.decode`；`vae.decode_record`：层、方案及一句说明、条带、工作区、估算、实测 reserved / GTT 峰值增量，第一次含自检时注明）；模型的 LoRA 名字和强度（`LoraLoader` 包装记在 `model_options`）、设置和来源、内存里的状态；最后总是全局默认值表。
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
* **还走第二层的图像 VAE**（4a 盘点时的状态；Flux 2 已在 4b-1 改走第一层）：Flux 2（4K 原生 52.5 → 第二层 15.05，第一层预计 2.18，与 Flux `ae` 同一个 decoder，**最容易**）、Wan 2.2 单帧（4K 34.2 → 19.8，第二层不够）、HunyuanImage 2.1（38.1 → 11.1）、HunyuanImage 2.1 Refiner（1344 原生 50.3 → 4.4）、HunyuanVideo 1.0 / 1.5 单帧、SeedVR2、TAE 系列、Stage A / C、Mage、像素空间。
* **还走原生的多帧视频**：第二层在多帧上**数值精确**（Wan 2.1 / 2.2、HunyuanVideo 1.0 / 1.5、CogVideoX 的小解码，相对误差 ≤ 2.3e-6），模拟峰值：Wan 2.1 480p 81 帧 8.7 → 5.0、Wan 2.2 704p 121 帧 39.4 → 10.4、HunyuanVideo 1.0 480p 73 帧 62（原生 OOM → tiled）→ 15.2、HunyuanVideo 1.5 720p 121 帧 113（OOM）→ 21.6、CogVideoX 35.0 → 11.7、Cosmos 30.2 → 13.0、Mochi 362 → 59.3（第二层不够）、LTX 30.6 / 59.3（模拟碎片，待确认）→ 6.7 / 5.0（要走 `output_buffer`）。要做的是 `_native_reason` 放开、`_probe` / `estimate` 认识 5D 多帧。
* **现有缺口**（这一步只报告）：① SVD 的 `VideoDecoder` 以 batch 为时间轴，第二层逐样本解码改变了结果；② ACE-Step / LTX 2 音频 / MiniMax 音频（2D latent）被管理，ACE 的估算约 1 PiB（`load_models_gpu` 会卸载一切）；③ 像素空间被管理（估算 2 GiB）；④ TAESD 1344 第二层 2.66 > 原生 1.77 GiB。
* **建议顺序**：0 修缺口 ① – ③ → 1 Flux 2 第一层 → 2 多帧视频第二层（通用）→ 3 Wan 2.2 单帧第一层 → 4 HunyuanImage 2.1 第一层 → 5 视频第一层（等有需要）。不建议做：TAE 系列、Stage A / C、Mage、MiniMax 视频、SeedVR2 第一层、音频、3D。
* **用户定的（4a 之后）**：顺序 ① 修缺口 1–3 → ② Flux 2 第一层（一定会用）→ ③ 视频等用户定了再说（不是不做，往后放）。SVD 先走原生（并进 ①），以后做视频第二层时把 SVD 整批第二层一起做。每项一个分支，分开提交，合进 dev。
* **进度**：① 完成（4b-0，DESIGN §9.17）；② 完成并在 CT 700 上**验收通过**（4b-1，DESIGN §9.18，实测 README §10.5：W / X / Y / Z / check_vae_node 都与预测一致）；③ 视频等用户定。之后用户定的「图像补齐」：Wan 2.2 单帧、HunyuanImage 2.1 第一层完成并验收（Wan 2.2 经方案 G，见 §1）；Refiner 见 §1「第五轮」；视频（多帧）第五轮只做了分析（DESIGN §9.32），路由不变。
* **自检峰值（DESIGN §9.19 → §9.20）**：用户选 B + A，已实现（vae-selftest-budget）：参照解码在第二层分块下跑（自检 0.74 → 约 0.38 GiB），自检上界（430 MiB）算进第一次解码的估算和预算比较。
* **第一次解码多 0.13 GiB / arena 被输出钉住（DESIGN §9.21 → §9.22）**：用户选 ① a、② a + 结束时清缓存，已实现（vae-arena-output）：`StripeAdapter.run` 先清缓存、再分配输出（单独一段）、再预留 arena，解码完清缓存；估算 = 输出段 + arena + largest + 16 MiB。布局变了使方案 A（无存档）4K 128 MiB 工作区高了 0.12 GiB（前缀在检查点前留下的洞以前被输出填上），加了 `move_low`（检查点挪进下面的洞，只在洞放得下时）。alloc_sim：37 个配置峰值都不升、≤ 估算（手工跑的结果；自动测试只断言其中一部分），解码后只留输出；63 个 CT 700 读数（含新加的 Flux 2 的 12 个）按旧布局重放都对得上；预算选择不变。
* **代码审查（基准 4f140ea，ComfyUI 62b3c94）**，每项一个分支：
  * 已改：03 解码失败的记录（`vae-record-errors`）；04 `check_selftest_mem` 自检预算选中的方案（`check-selftest-scheme`）；05 `run_all.sh` 加上 `test_vae_ldm` / `test_vae_flux2` / `test_vae_node`（`run-all-vae`）；07 自检模拟补上最后的比较，上界不变（`selftest-trace-tail`）；08 文档（`review08-docs`）。
  * 用户定了之后改的：01 每个模型一个绑定对象，所有运行时 patch 读它，`partially_load` 调原函数之前绑定指向当前 patcher（`lora-runtime-binding`，`tests/test_lora_clone_binding.py`）；02 选 B，强制的第一层配置自检失败时（首次或已缓存，决定相同）第二层也要放得下预算，否则报错（`vae-selftest-fail-budget`）；06 加保险，重试计划的估算超过第一次就跳过这一档（`vae-retry-guard`）；分析 01 时发现的 lowvram + hook 普通 LoRA 加两次，单独修（`lora-lowvram-hook`）。
  * **审查修正全部完成，已在 CT 700 上验收**（b4c33d5，README §10.7）：`run_all.sh` 39 组 1451 项 0 失败；LoRA 每步耗时与 9.1 第三轮相同（默认 1.13 / 1.10 / 1.19×，逐位一致 1.39 / 1.33 / 1.66×，逐位一致 max|Δ| 0，备份 0）。
* **第二轮代码审查（基准 main 5d9ac63，ComfyUI 62b3c94；报告的修复清单 01–23）**，每项一个分支，从 dev 开、`--no-ff` 合回 dev：
  * **LoRA / 释放**：01 保存时 LoRA 被丢掉（`lora-save-state-dict`；接管 `model_state_dict_for_saving`，调用期间把这个 patcher 被 patch 的层的 `comfy_patched_weights` 暂设 False，这些层走原生 `LazyCastingParam` 精确合并；`tests/test_lora_save.py`，真机脚本 `tests/check_lora_save.py`）；02 有原生备份的 patcher 不同步 uuid（`release-sync-backup`；先用 `PromptExecutor` 证实 clone 在那次释放里被回收、下个 prompt 带着 LoRA，max|Δ| 26.8）；03 原生 patcher 被节点释放时，从 offload 设备还原的参数移回原设备（`release-native-offload`，选了「移回」而不是 `model_unload`：底模仍常驻，`--gpu-only` 时什么都不做）；04 删掉 `_drop_shadowed_runtime_patches`（`lora-drop-shadowed-hooks`）；05 摘运行时 patch 时不删 `comfy_patched_weights`（`lora-hook-keep-flag`；比清单多一步：清单只保留 `keep is None` 时删，但只有 hook、没有普通 LoRA 的模型每次采样结束也走 `keep=None`，那样仍会删光，所以整个去掉）。
  * **VAE**：06 自检 OOM 包成 `MonoloadVAEOOMError`（用户选 (a)，不缓存、不退第二层；预算路径不再把它记成「预算不够」）；07 arena 加完余量再按 2 MiB 取整（alloc_sim 21 个受影响的计划 reserved 不变、≤ 估算）；08 第一层 OOM 报错写最后真跑过的计划，跳过的档另列；09 自检失败转第二层时记录含自检；10 预算报错的小数位数加到估算和预算看得出差别为止（2–6 位），节点预算按字节四舍五入；11 ComfyUI 升级使 VAE 部分导入失败时节点和 LoRA 部分照常（`TEST_BROKEN_VAE_API=1` 的入口测试，进了 `run_all.sh`）；12 第二层也分块 SeedVR2 的 `optimized_vae_attention`。
  * **测试 / Info / 文档**：13 `test_check_selftest_mem` 前置条件不满足时退出码 1；14 `run_all.sh` 缺模型文件跳过的组计数、退出码 = 失败 + 跳过；15 alloc_sim 的 fp32 配置也拷贝 latent（两条上界断言的配置不变；fp32 重放 v2 4K 2.07 → 2.09、v1 4K 3.21 → 3.23）；16 alloc_sim 第二层直接跑 `vae._run`（21 个第二层读数不变）；17 Info 中文下来源「环境变量」；18 全选 default 的节点副本也标「副本」；19 `gn_scheme` tooltip 加 Flux 2（`test_messages` 多查英文 locale 与节点自己的 tooltip 一致）；20 文档（另外把方案 C「全分辨率每个块的输入」改成「输出」，tooltip、两份 locale、消息表、README、DESIGN、HANDOFF 都有这个错）。
  * **21–23（先分析，用户定了之后）**：21 **做了**（`vae-probe-cache`，56b9e93）：预算模式第一次解码时 8×8 形状探测的块留在缓存里紧接自检，alloc_sim 上 Wan（`qwen_image_vae`）fp32 自检到 426 / 430 MiB；现在 `vae._shape_probe` 探测完清一次缓存，所有结构 bf16 / fp32 的自检峰值都与单独自检相同（Wan fp32 410，余量 4 → 20 MiB），`alloc_sim.selftest_trace(probe=True)` 和 `test_vae_flux2` 检查；CT 700 命令 Z 复查：每档第一次都 ≤ 估算（README §10.8）。22 **不做**（用户定）：`place()` 补「剩余 ≤ 1 MiB 不切分」后 3780 个计划完全相同，DESIGN §9.14.11 记了一句（a0ce191）。23 **做了**（`vae-probe-memreq`，c9e6bcf）：8×8 探测交给 `load_models_gpu` 的 `memory_required` 不再带 2 × 工作区（以前 2.02–2.13 GiB，探测实际只占 18–238 MiB；现在是探测自己的小估算，`load_models_gpu` 另有约 0.8 GiB 的 `minimum_inference_memory` 兜底），`test_vae_ldm` 检查；CT 700 复查通过（命令 Z，选择和数字不变）。
  * **已在 CT 700 上验收**（8c5cb37，README §10.8）：AC / AD 三种保存与原生逐张量相同；AE SDXL 4K 默认 2.17 GiB / 估算 2.68，与以前相同；`run_all.sh` 41 组 1494 项 0 失败 0 跳过（其中「VAE 接口变了」那组实际没进入该模式）。**复查**（e924a77，含 21 / 23 和两处测试工具的修正）：`run_all.sh` 41 组 1490 项（那一组 17 → 11，21 / 23 在 `test_vae_flux2` / `test_vae_ldm` 各加 1 项）0 失败 0 跳过；AE 不变；命令 Z 不变，每档第一次 ≤ 估算。
* **已在 CT 700 上验收**（4f140ea，B + A 和这次的改动一起，命令 AA / X / Y / Z，README §10.6）：自检峰值 0.37（加 warm-up，≤ 上界 0.42）/ 0.45（不加，含进程第一批 GPU 计算约 0.07，不在上界里）；decode 1 = decode 2 = 2.43，解码后只留 0.09；X / Y / Z / check_vae_node 的选择和数字都与预测一致。

### 4.1 入口和做法（4a 之前写的，仍然适用）

**现在的覆盖范围**（`vae._native_reason`、`STRIPE_ADAPTERS`；当前代码，4b-0 / 4b-1 之后）：

| ComfyUI 的 VAE（`comfy/sd.py` 按 state dict 识别） | 现在走哪条路 |
|---|---|
| SD1.5 / SDXL / SD3 / Flux `ae` / Flux 2（LDM `Decoder`，`AutoencoderKL` / `AutoencodingEngine`；Flux 2 带 `batch_norm_latent`，4b-1） | 第一层（`vae_ldm`） |
| Wan 2.1 / `qwen_image_vae` 单帧（5D，T=1） | 第一层（`vae_wan`） |
| Wan 2.2 单帧（`vae2_2.WanVAE`，5D，T=1） | 第一层（`vae_wan22`） |
| HunyuanImage 2.1（`AutoencodingEngine` + `hunyuan_video.vae.Decoder`，4D，32x） | 第一层（`vae_hunyuan`） |
| 其他 2D 图像 VAE（带注意力的 up 级、TAESD、Stable Cascade Stage A / C、Mage-VAE 等） | 第二层（逐算子分块），或第一层识别不通过时第二层 |
| 3D VAE 的单帧 latent（5D，T=1）：Hunyuan 系（含 HunyuanImage 2.1 Refiner）、Mochi、Cosmos、CogVideoX、TAEHV 等，Wan 2.1 / 2.2 以外 | 第二层（SeedVR2 在 v0.46.3 声明了 `comfy_has_chunked_io`，单帧也走原生，见下面一行） |
| 多帧 Wan 2.1 / 2.2（T>1，`wan.WanVAE` / `vae2_2.WanVAE`） | 第一层（`vae_wan_video`，**注册到 `STRIPE_ADAPTERS` 之后**）；没注册时第二层（H1，`vae_video.MULTIFRAME`，实测探测定上界） |
| 多帧 LTX（0.9.x 卷积 VAE、LTX 2.5 扩散 decoder） | 第二层（H1）；扩散 decoder 由 `vae_ltx` 经 `EXTRA_CHUNKERS` 换算子（F 组），`vae_ltx` 导入失败时扩散 decoder 原生 |
| 多帧视频 latent（T>1），`vae_video.MULTIFRAME` 之外：Hunyuan 系（3D 卷积 `AutoencoderKL` / `AutoencodingEngine`）、Mochi、Cosmos、CogVideoX、MiniMax H3、TAEHV 等 | **原生**（`_native_reason`：multi-frame video latent，第一阶段暂不做） |
| 自己往预分配输出写的（`comfy_has_chunked_io`）：MiniMax H3 视频 / Sol / 图像、SeedVR2（`VideoAutoencoderKLWrapper`）、RelayVSR Flash | 原生，单帧也不管（`tests/test_vae_selftiled.py`；G 组，PLAN-video）。例外：LTX 卷积 VAE（`ltx.VideoVAE`）由 H1 接管（`vae_video.chunked_io_managed`，按类型精确匹配） |
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
* **跑测试**：`MODELS=<目录> tests/docker_run.sh python tests/<test>.py`（仓库只读挂进镜像的 custom_nodes；`docker_run.sh` 传 `MONOLOAD*` 环境变量）。镜像拉不下来时（第二轮审查就是这样做的）：venv 里装 CPU 版 torch（2.14.1+cpu）+ ComfyUI 62b3c94 的 requirements（另加 torchsde，torchvision / torchaudio 用 CPU 源），`COMFYUI_PATH=<ComfyUI 目录> python tests/<test>.py`；`test_entry.py` / `test_release.py` 要求 `/opt/ComfyUI/custom_nodes/monoload` 指向仓库（软链接即可）；`test_release.py` 的 LoRA 要是 `models/loras` 里的真文件（硬链接，不能是指到别处的软链接，否则弱引用统计是 0）。不要同时跑两个 torch 测试（4 核时互相拖慢十倍以上）。
* **合成模型**：`tests/make_synthetic_vaes.py OUT [--full]`（SDXL / Flux / Wan 结构的随机权重 VAE）；`tests/make_synthetic_checkpoint.py MODELS`（随机权重 SD1.5 checkpoint + UNet / TE LoRA，约 2 GiB，`test_lora_node.py` 用）。真实 SD1.5 不在云端：`test_release.py` 可以用合成 checkpoint 冒充（硬链接成 `v1-5-pruned-emaonly-fp16.safetensors` / `rubber_duck.safetensors`）。
* **网页实测**：在锁定镜像里起服务（`docker run -p 127.0.0.1:8188:8188 ... python main.py --cpu --listen 0.0.0.0`），Playwright（`/opt/node-tools/node_modules/playwright`，Chromium `/opt/pw-browsers/chromium-1194`）打开真实前端，或用 `/prompt` API 跑工作流看日志（第 5 项的泄漏就是这样复现的）。
* **CT 700 的事实**：`--gpu-only --bf16-vae`；AMD 上 `cudnn.enabled = False`，4D 卷积走 Slow2d（im2col + GEMM），5D 走 SlowDilated3d；VAE 注意力是 split；统一内存，看 GTT。用户终端是 `LANG=C`，中文日志显示成下划线（字节是正确的 UTF-8，不用改）。
* **`tests/bench_vae.py`**（README §9.7）：`--checkpoint` / `--vae`；`--res`；`--modes native,monoload,monoload-l2,monoload-r<N>,native2`，模式名后缀 `-g<S>`（方案）、`-b<GiB>`（预算）、`-w<MiB>`（工作区）；`--stripe-rows`；`--budgets`；`--gn-schemes`；`--fp32-ref`；`--profile-only`；`--json`。
* **`tests/alloc_sim.py`**（DESIGN §9.13.10、§9.14.6；模型 `qwen` / `sdxl` / `flux` / `flux2` / `wan22` / `hy21` / `hyr`（sd.py 的 `refiner_vae=False` 那一支）/ `hv15`（官方 Refiner 文件和 HunyuanVideo 1.5 的 decoder），`hy21` 的宽高要是 32 的倍数，4K 用 3840×2176）：meta 设备上跑全尺寸 decoder，按缓存分配器的规则重放，复现了 63 个 CT 700 读数（手工跑全表时 ≤ 0.02 GiB；自动测试只抽查其中一部分，阈值 0.03 GiB（`test_vae_stripe` / `test_vae_flux2`）/ 0.05 GiB（`test_vae_ldm`）；第一层的旧布局用 `output_in_arena` 重放，DESIGN §9.22）。改内存相关代码先用它看。
* **方案 G**（`vae_ops.GROUP_NORM_MAX` / `_GroupedRMSNorm` / `grouped_norm_bytes`，DESIGN §9.29）：改 Wan 的 RMS_norm 或 `OpChunking` 时看 `test_vae_wan22` 第 7、8 节。
* **`tests/check_norm_reduce.py`**（DESIGN §9.28）：纯 PyTorch，量 `F.normalize` 在各种通道 / 位置数 / 布局下的归约临时缓冲，按 `Reduce.cuh` 的启动配置预测（`--predict` 不要 GPU），并对比分组求范数的候选方案。
* **`tests/check_l1_units.py`**（DESIGN §9.26）：第一层逐单元的真机显存——每次调用自己的峰值、模型对同一次调用的计数、调用中新开的段；`--sim MODEL` 用 alloc_sim 出同样的表（无 GPU）对照。`bench_vae.py --profile` 另有 reserved 峰值时的段和开段请求的栈。
* **盘点脚本**（DESIGN §9.16）：`tests/vae_inventory.py`（`--only` / `--no-trace` / `--json`；每种 VAE 的结构、路径、原生 / 第二层 / 第一层模拟峰值，带 60 GiB 设备上限；新适配器的 meta 构造可以从这里的 `KINDS` 抄）、`tests/probe_vae_gaps.py`（缺口和多帧第二层精度的小解码）、`tests/check_models.py`（CT 700 的命令 W）。
* **检查脚本**：`tests/check_vae_node.py`（命令 U）、`tests/check_lora_node.py`（命令 V，不带 `--lora` 时列文件）。

## 6. 代码地图

* `__init__.py`：插件入口；导出三张表（节点、显示名、`WEB_DIRECTORY`）；`MONOLOAD_DISABLE` 以外装 hotpatch / release / VAE 包装和 `LoraLoader` 名字包装；启动日志。
* `monoload/settings.py`：总开关、全局默认 `exact()` / `keep()`、`disabled()`。不导入 torch / ComfyUI。
* `monoload/messages.py`：消息表、`msg()` / `label()`、`MONOLOAD_LANG`。
* `monoload/hotpatch.py`：`ModelPatcher` 方法替换（含保存用的 `model_state_dict_for_saving`）、`MonoloadRuntimePatch`（融合 / 逐位一致）、按 patcher 的 `_enabled` / `_active`、`_monoload_runtime` 模型标记。
* `monoload/release.py`：每个 prompt 结束后按 patcher 释放；`_ancestor_chains` / `_repoint_orphans`（两层不带 patch 的 clone 一起被回收时，把已加载模型指回活着的祖先，修 CT 700 的「memory leak」警告）。
* `monoload/lora_overrides.py`：LoRA 节点的设置、`resolve()`、`enabled()` / `merge_exact()` / `wants_release()`、`install_names()`。
* `monoload/vae.py`：VAE 入口和策略（`_decode` → `_managed_decode` → 预算 `choose_budget` / 默认 `choose_plan` → `_decode_layer1` / `_decode_layer2`）；`resolve_settings` / `settings_note` / `_Applied` / `_from`（来源）；`global_mode()`；`decode_record()` / `_MemProbe`（测量前先清分配器缓存）；`STRIPE_ADAPTERS`。
* `monoload/vae_ops.py`：第二层引擎（`OpChunking`、`_ConvChunker`、注意力 query 分块、`OpStats`）。
* `monoload/vae_engine.py`：第一层引擎（区间、`Plan`、统计遍、arena、`StripeAdapter` 基类、自检 `_SELFTEST`）。
* `monoload/vae_wan22.py`：Wan 2.2 单帧适配器（复合单元 `DUPUP`、`UNPATCH`、`decode_retained`、只留时间核最后一片的 fp32 副本；DESIGN §9.23）。`monoload/vae_hunyuan.py`：HunyuanImage 2.1 适配器（复用 `vae_ldm` 的块、内存模型和方案，`ConvIn`、`PU` 单元、`selftest_size` 12、`activation_copies` 6；DESIGN §9.24）。引擎为它们加了：`need_in` 按 `scale > 1`、`Unit.extra`、`skeleton()`、`StripeAdapter.decode_retained` / `activation_copies` / `selftest_size`；`vae._decoder_terms` 把后两者给第二层估算。
* `monoload/vae_refiner.py`（**只在分支 vae-refiner-layer1 上**）：`vae_refiner.Decoder(refiner_vae=False)` 的适配器（条带带 4 帧、`Unit.tscale`、跨帧 GroupNorm 方案，DESIGN §9.34）。dev 上 `vae_hunyuan.decoder_terms` 给这一支 6 份激活（§9.30）。
* `monoload/vae_wan.py`、`monoload/vae_ldm.py`：两个适配器（LDM 的含 Flux 2 batch-norm latent，`LatentUnpatch`；`vae_ldm.scheme_positions`：A 不存，D 存 H/4 级输出，B 存 H/4 和 H/2，C 再加全分辨率各块的输出（块 1、块 2 和 norm_out 的输入；块 0 的输入不存）；耗时模型 `TIME_COEF`）。
* `monoload/vae_ltx.py`（F 组）：LTX-2.5 扩散 decoder 的第二层扩展（`chunker` 挂进 `vae_ops.EXTRA_CHUNKERS`）：`_NASlabs`（邻域注意力按 T 等长段，`t_blocks` / `na_block_frames` / `na_slab_bytes`）、SwiGLU 分块、`_DecoderForward`（精简 forward）、`plan()`（reserved 口径的诊断估算）/ `native_estimate()`。测试：`tests/check_ltx_dd.py`（fp32 精度 + 估算对照）、`tests/bench_ltx.py`（卷积 / 扩散两种，`--modes native,lean,monoload,l2x`，`--compare`）。
* `monoload/vae_overrides.py`：VAE 节点的设置（不导入 torch / ComfyUI）。
* `monoload/info.py`：Info 节点的文字。
* `monoload/nodes/`：注册表 + 三个节点。`web/`：`monoload_info.js`、`monoload_i18n.js`。`locales/`：界面翻译。
* **小测试**（都不需要真实模型）：`test_entry.py`（开关组合，见文件头；其中 `MONOLOAD_VAE_BUDGET` + `MONOLOAD_VAE_STRIPE_ROWS`、`MONOLOAD_VAE_GN_SCHEME` 两种不在 `run_all.sh` 里，要手动跑）、`test_master_switch.py`、`test_messages.py`、`test_info_node.py`、`test_release_chain.py`、`test_vae_node.py`、`test_vae.py`、`test_vae_ldm.py`、`test_vae_flux2.py`、`test_vae_wan22.py`、`test_vae_hunyuan.py`、`test_vae_refiner.py`（`refiner_vae=False` 的第二层估算和官方 Refiner 文件的 decoder）、`test_vae_stripe.py`、`test_dtype_paths.py`（默认和 `MONOLOAD_EXACT=1`）、`test_lora_node.py` 和 `test_lora_save.py`（要合成 checkpoint）。CT 700 检查脚本另有 `check_lora_save.py`（README §9.15）、`run_flux2_dev.py`（Flux 2 dev 整套，README §9.23）。需要真实 SD1.5 的：`test_lora_hot.py`、`test_quant.py`、`test_release.py`。

## 7. 提交记录（`dev` 上的合并；video 分支的在表末）

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
| Flux 2 验收结果、自检峰值的分析和诊断脚本 | 0a4e177 |
| 自检缩小、算进第一次的估算和预算；第一次解码 / arena 钉住的分析 | 6324592 |
| 解码前后清缓存、输出单独一段、检查点下移；合并验收清单 / 验收结果 | 4f140ea / 65306c0 |
| 审查 05 / 04 / 03 / 08 / 07 | 68ed231 / 4f40a69 / 9c70cab / 3730e6d / 2d36fcd |
| 审查进度记录 | 01f3943 |
| 审查 06 / lowvram + hook / 02 / 01 | 0b5311c / 75f5b6e / 2c28532 / c49960c |
| 审查修正的文档和真机清单 | b4c33d5 |
| 审查修正的真机结果 | f70e485 |
| 第二轮审查 01 / 02 / 03 / 04 / 05 | 0e110a1 / 4cb4ce8 / 275f400 / c0b6c7b / f327c9d |
| 第二轮审查 06 / 14 / 08 / 09 / 07 | 07c1787 / a74a6a3 / d044df1 / d42689c / d03f72f |
| 第二轮审查 13 / 19 / 17 / 18 / 11 / 12 | 5c1d8c2 / d88a0cb / 1d71488 / 9cadb98 / 2960272 / c99fb53 |
| 第二轮审查 10 / 20 / 16 / 01 的真机脚本 / 15 | 5e6be4a / c9e4ea0 / 349e641 / e5003cc / 84627fb |
| 第二轮审查的文档和真机清单 | 8c5cb37 |
| 验收后：`TEST_BROKEN_VAE_API` 传进容器 / bench_vae「native not run」/ 21 / 22 / 23 | eee564c / 17880b8 / 56b9e93 / a0ce191 / c9e6bcf |
| 第二轮审查的真机结果 | e924a77 |
| 复查结果（21 / 23 / 测试工具修正后） | 531bf32 |
| 图像补齐：Wan 2.2 单帧第一层 / HunyuanImage 2.1 第一层 | d6473d3 / 838f7fc |
| 图像补齐的文档、Refiner 分析 | 189741e |
| HunyuanImage 2.1 验收结果 / Wan 2.2 矮条带超估算的分析和诊断 | 8c432bd / c1e8be6 |
| test_entry 适配器列表 / check_vae_node 先装权重 / bench 汇总同一次 / 无方案不写方案 / 第二层估算按类型 | 5dd96b7 / 161155d / 403c366 / 3c6786b / 23391cd |
| 第二轮的 HANDOFF | 1dc61b2 |
| 第 3–7 项复查结果 / check_vae_node 不打印 Wan 的方案 / F.normalize 归约临时缓冲的分析 | 3825aea / 86a966f / 2711c3e |
| 第三轮的 HANDOFF | 2c24452 |
| check_norm_reduce 的汇总口径 / 命令 N 的结果 / 方案 G（分组求范数）/ 复测清单 | 9bc1c4c / e2feb57 / f1f3fc1 / cf1b7c3 |
| 第四轮的 HANDOFF | 0593f27 |
| Wan 2.2 验收结果（README §10.13） | d714185 |
| Wan 2.2 验收那一轮的 HANDOFF | 907ef25 |
| Refiner 第二层估算（`refiner_vae=False`）/ 官方 Refiner 文件的更正 | f7e1aa2 / b426588 |
| 视频分析 / Flux 2 dev 整套脚本 | 565e73b / 2219844 |
| （未合并）`refiner_vae=False` 的第一层：分支 vae-refiner-layer1 | f51a6bc |
| **video 分支**：适配 ComfyUI v0.46.3（阶段 A） | 8d89069 |
| video：PLAN-video.md | d58707f |
| **video 分支**：LoRA 量化 safetensors 测试 / GGUF / 视频小参数 | 见 `git log --first-parent video`（Merge lora-quant-formats / lora-gguf / lora-video-models） |
| **video 分支**：H1 / H3 / H6 多帧第二层入口、LTX 进第二层、`EXTRA_CHUNKERS` | 5fddc22 |
| **video 分支**：H2 / H5 引擎时间轴 | 0e1b913 |
| **video 分支**：H4 方案 G（Refiner RMS_norm、LTX PixelNorm） | Merge video-norm-g（`git log --first-parent video`） |
| video：G 组（自带分块的 transformer VAE）保持原生、路由测试、bench_selftiled | 见 `git log --first-parent video`（Merge video-g-selftiled） |
| **video 分支**：多帧探测按 latent 截断；LTX-2.5 扩散 decoder 等 F 组的 `vae_ltx` 再托管 | Merge video-probe-clip（`git log --first-parent video`） |
| **video 分支**：A 组 `vae_wan_video` 注册进 `STRIPE_ADAPTERS`（可选 import），Wan 2.2 多帧走第一层 | Merge video-wan-reg（`git log --first-parent video`） |
| **video 分支**：F 组 LTX（卷积只走第二层；扩散 decoder 由 `vae_ltx` 接管） | Merge claude/project-thread-b4e366（`git log --first-parent video`） |
| **video 分支**：多帧探测落盘、LTX-2.5 扩散 decoder 用 `plan()` 估计 | Merge video-probe-cache（`git log --first-parent video`） |
| **video 分支**：视频条带 carry 放进一块缓冲（`CarryStore`） | Merge video-carry-store（`git log --first-parent video`） |
| **video 分支**：D 组 Cosmos 1.0 / Mochi 逐帧流式第二层（`vae_cosmos`、`vae_mochi`） | Merge video-groupd（`git log --first-parent video`） |
| **video 分支**：B 组 Hunyuan（HunyuanVideo 1.0 / 1.5 多帧第二层、`vae_hunyuan15` 第一层，已注册进 `STRIPE_ADAPTERS`） | Merge video-b-reg（`git log --first-parent video`） |
| **video 分支**：`test_quant` / `test_quant_formats` 的 fp8 参照改在计算设备反量化（GPU 逐位），`bench_lora` 逐层探测先反量化 | Merge video-quant-ref（`git log --first-parent video`） |
| video：视频第一层 A 组 Wan 2.1 / 2.2 多帧（`vae_wan_video`） | 见 `git log --first-parent video`（Merge video-wan-l1） |
| 本文件 | 本文件所在的提交（`git log --first-parent video` 最上面几条） |

LoRA 部分更早的历史：3c473fa … 5bfbc8e（v1 文件格式 → v2 运行时合并 → 释放、fp8、融合 addmm），见 `git log --first-parent dev`。各阶段的设计和真机数据：DESIGN §9.12–§9.15、README §10。
