# Monoload 视频计划（video 分支，ComfyUI v0.46.3，2026-11）

给并行开工的各线程用：每种视频 VAE（和 v0.46.3 新增的图像 VAE）第一层怎么做、按什么分组、各组依赖哪些共用钩子；以及要覆盖的视频模型、单次峰值怎么估、真机按什么顺序验。分析的依据是 DESIGN §9.32（62b3c94 时代的视频分析；v0.46.3 里 Wan 2.1、两个 Hunyuan、CogVideoX、Cosmos、Mochi、LTX 卷积 VAE 的解码代码没有变，结论仍然适用）加上 v0.46.3 的新代码（`git diff 62b3c94 v0.46.3`：`wan/vae2_2.py`、`seedvr/vae.py` 重写、`minimax/vae.py`、新的 `lightricks/vae/na_diffusion_decoder.py`、`wan/model_vsr.py`）。

## 0. 规矩（§3 之外，这一轮新加的）

* **单次峰值 ≤ 55 GiB 是主线**：CT 700 现在 GPU 稳定可用约 55 GiB（60 GiB OOM，卡在宿主机 `ttm.pages_limit` 默认值）。单次峰值 = 同一时刻装在 GPU 上的最大的那一份：扩散模型（DiT）、文本编码器（TE）、VAE 解码（权重 + Monoload 估算），三者取最大（ComfyUI 换模型时会卸前一个；Monoload 的估算交给 `load_models_gpu`，它按需卸别的）。**> 55 GiB 的单列一组，标「等宿主机调 ttm 后再验」**，代码照写、照做小测试，真机等调完再跑。
* **分批加载只在需要时**：只有超大模型、TE 和 DiT 同时在 GPU 上会 OOM 时，才分开（先编码、卸 TE、再加载 DiT）；怎么分由做那一组的线程判断，写进它的真机命令。
* **运行时什么都不经过内存**（HANDOFF §3）：LXC 只给了 48 GiB 内存，GTT 不算在里面；任何方案都不许放到 CPU 上（包括自检、缓存、输出列表）。
* **safetensors 和 GGUF 都要做**：DiT / TE 的 GGUF 由 GGUF 线程负责（LoRA 运行时合并到 GGUF 权重上）；VAE 这边见 §5。
* **不做 MAGI-2。**
* 每个第一层适配器：fp32 自检（相对误差 ≤ 1e-4）、alloc_sim 扫尺寸证明 reserved ≤ 估算、`MONOLOAD=0` 与原生逐位相同、默认行为不变；不认得的结构走第二层或原生，不报错。

## 1. 共用钩子（阶段 C 先做，各组依赖它们）

| 钩子 | 内容 | 谁用 | 状态 |
|---|---|---|---|
| **H1 多帧入口** | `_native_reason` 放开 T > 1 的 5D latent（按 decoder 类型白名单逐个放开，没放开的仍原生）；形状探测和 `estimate` 认 5D；`decoder_terms` 加多帧项：每卷积缓存（`retained`）、carry、**输出列表 + `torch.cat`（2 × 输出）**、CogVideoX 一直活着的第一段；alloc_sim 的 5D 跟踪（`decode_trace` / `vi.trace` 带 T），扫尺寸确认估算是上界（DESIGN §9.32：现有估算对多帧基本都不是上界）| 全部多帧第二层、全部视频第一层 | 阶段 C |
| **H2 引擎的时间轴** | 从分支 vae-refiner-layer1（f51a6bc，DESIGN §9.34 分支版）只取引擎部分：`Unit.tscale`、`StripeAdapter.tdim` / `ckpt_planes`、`Plan` 的平面数（每行元素数 = 宽 × 平面数，区间倒推不变）。再加**按时间块推进**：适配器给出时间块序列（Wan 2.1：1 + 2 + 2…；Wan 2.2：1 + 1…；Hunyuan：2 帧一块），引擎「条带为主」地跑：对每个输出行条带，依次跑全部时间块，**每个条带自己的每卷积缓存 / carry 只有条带那么大**（含 halo 行），时间块之间留在条带里；低分辨率前缀对整段视频算一次存档 | 视频第一层（A、B、C、D 组）| 阶段 C |
| **H3 直接写输出** | 每个时间块 / 条带的输出直接写进预分配的 5D 输出（`vae_dtype` 或 fp32，和原生 `VAE.decode` 一样在 `output_device`），不留 Python 列表、不 `cat`（原生的 2 × 输出这一项没了）；LTX / RelayFlash / MiniMax 的 `output_buffer` 走同一个接口 | 全部视频第一层；LTX 第二层 | 阶段 C |
| **H4 方案 G 推广** | 分组求范数（≤ 510 通道一组，fp32 合并）扩到 `vae_refiner.RMS_norm`（没有 `channel_first` / `bias`，要一个变体）和 LTX 的 PixelNorm（`mean(x²)`，1024 通道）；**替用户定：扩**（理由同 Wan 2.2：ROCm 归约内核每位置 32 KiB 的临时缓冲落在缓存外、开新段，DESIGN §9.28 / §9.32 第 2 点）| B 组（HunyuanVideo 1.5 / Refiner）、F 组（LTX）| 阶段 C |
| **H5 视频自检** | fp32 自检用小尺寸、少量时间块（≥ 3 块，让缓存跨块）：副本只留每个 `CausalConv3d` 用得到的时间核切片；自检上界算进第一次的估算（同图像）| 全部视频第一层 | 阶段 C |
| **H6 测量** | `bench_vae --frames N`（随机 latent 带时间维，按 VAE 的 `downscale_ratio` 算 T）；`vae_inventory` 的视频用例；README 给每组的命令模板 | 全部 | 阶段 C |

**做法要点（H2）**：图像第一层是「前缀整图 → 存档 → 按输出行条带倒推重算」。视频多一个时间轴，但 Wan / Hunyuan / CogVideoX 第二段的 decoder 本来就**按时间块因果地**跑、块之间只传每卷积最后 2 帧（`feat_cache` / carry），这些缓存在空间上是**逐像素局部**的（3×3 卷积的 halo 之外不读别的行）。所以条带为主：存档（整段视频的低分辨率前缀，例如 Wan 2.1 的 r 级：81 帧 832×480 约 100 MB）→ 对每个条带，从存档切出 halo 行、依次跑全部时间块、条带自己的缓存跟着走 → 写进输出。峰值 = 存档 + 一个条带一个时间块的激活 + 条带的缓存 + 输出，与帧数基本无关（只有存档和输出随帧数涨）。halo 按单元倒推（和图像一样），缓存的 halo 行要和条带一起保存（缓存存的是卷积**输入**，halo 已经包含）。

## 2. 每种 VAE 的方案和分组

图例：**原生** = 现在的路；**L2** = 多帧第二层（阶段 C 的 H1）；**L1** = 视频第一层（各组线程）。峰值数字是 DESIGN §9.32 的 alloc_sim（GiB，reserved，原生 > 60 记 OOM；第二层工作区 1 GiB）。

### A 组：Wan VAE（价值最大，先开）

| VAE | 用在哪些模型 | 832×480×81 原生 → L2 | 720p×81 原生 → L2 |
|---|---|---|---|
| Wan 2.1 VAE（`wan/vae.py`，16 ch，8×）| Wan 2.1 全系、**Wan 2.2 A14B**（T2V / I2V / S2V / Animate / Camera）、Wan Animate 2、VACE、FunControl、HuMo、SCAIL / SCAIL2、FlowRVS、CausalAR、WanDancer、Cosmos Predict 2 视频；图像：Qwen-Image、Krea 2、Anima、JoyImage | 8.72 → 4.96 | 20.29 → 8.99 |
| Wan 2.2 VAE（`vae2_2.py`，48 ch，16×）| Wan 2.2 TI2V 5B；图像：PAN2 | 17.90 → 5.20 | 40.49 → 9.47 |

* **L1 方案**：扩展现有的 `vae_wan` / `vae_wan22` 单帧适配器到多帧，走 H2 的条带为主。时间块：Wan 2.1 第一块 1 个 latent 帧、之后每块 2 帧，`Resample` 的时间上采样（`time_conv`）第一块跳过；Wan 2.2 每块 1 帧，第一块不做时间卷积、`DupUp3D` 丢掉多出的帧，缓存总是补成 2 帧（`clone`）。单元沿用单帧的划分（halo 不变：时间核不影响空间 halo），每个单元多一个「条带缓存」状态。注意 v0.46.3 的 Wan 2.2：`feat_cache is None` 时 `ResidualBlock` / 上采样 / `DupUp3D` 会自己按行分块（`strip_apply`），多帧路径总是带缓存，不触发；我们的单元调用时传缓存，也不触发。
* 方案 G 已覆盖 Wan 2.2 的 1024 / 512 通道 RMS；Wan 2.1 最多 384 通道，不用分组。
* **进度**：已做（`monoload/vae_wan_video.py`，DESIGN §9.41，分支 video-wan-l1）。用 H2 的引擎时间轴；和上面的差别：单帧适配器不动、另起一个只接 T > 1 的适配器；Wan 2.1 时间上采样之后不切 2 帧一片（引擎的链是线性的，数值相同）。真机：1280×704×121 第一层对第二层 Wan 2.2 3.54 对 8.91 GiB（不慢、估算是上界），Wan 2.1 4.34 对 8.00 GiB（慢 20%、720p 估算少算），所以只开 Wan 2.2，Wan 2.1 多帧留第二层（`WAN21_ENABLED`）。等改 `vae.py` 的线程注册到 `STRIPE_ADAPTERS`。
* 预计 L1：与帧数基本无关的 1–2 GiB + 存档 + 输出（720p×81 fp32 输出 0.9 GiB）。

### B 组：Hunyuan VAE

| VAE | 用在哪些模型 | 832×480×81 原生 → L2 | 720p×81 原生 → L2 |
|---|---|---|---|
| HunyuanVideo 1.0（LDM `AutoencoderKL` conv3d，`CarriedConv3d`，GroupNorm）| HunyuanVideo T2V / I2V / SkyReels I2V、Kandinsky 5 视频 | 74.1（OOM）→ 14.98 | 171（OOM）→ 37.88 |
| HunyuanVideo 1.5（`vae_refiner`，`refiner_vae=True`，RMS norm）| HunyuanVideo 1.5、1.5 SR distilled；图像：HunyuanImage 2.1 Refiner 官方文件 | 50.01 → 9.14 | 110（OOM）→ 22.95 |

* **难点：mid 的全局注意力跨全部帧**（tokens = T·h·w），在全部 latent 帧上一次算完，然后 `torch.split` 成 2 帧一块跑上采样路径。L1：前缀 = `conv_in` + mid（整段视频，latent 分辨率；注意力用第二层的 query 分块）+ 存档；之后条带为主跑各块。
* HunyuanVideo 1.0 的 GroupNorm：原生是**每个时间块各自统计**（块内全部帧、整幅空间）→ L1 要每块一个统计遍（LDM 的方案 B 机制按块做，H2 的平面数正好描述「每行 2 帧」）。HunyuanVideo 1.5 是 RMS norm，没有统计遍（像 Wan 2.2），但要 H4（方案 G 扩到 `vae_refiner.RMS_norm`）。
* 顺带：HunyuanImage 2.1 Refiner 官方文件的单帧第一层（DESIGN §9.31，和 Wan 2.2 单帧同类）在这一组做掉。分支 vae-refiner-layer1 的 `refiner_vae=False` 适配器仍不合（官方文件用不上）。
* **进度（分支 claude/project-thread-glsomb）：** 两种 decoder 的多帧第二层入口（`vae_video` 按 decoder 结构认，逐块交出）；HunyuanVideo 1.5 / Refiner 第一层 `vae_hunyuan15.py`（前缀整段到 `up[0]`，条带为主 2 帧一块，carry 每条带每单元一份，单帧没有时间块；方案 G 的临时量在每个 norm 的计数里）。真机 480p×81：原生 50.09 GiB，第二层 10.43（不慢），第一层 2.54（慢 56%）→ 视频的第一层只作后备（`prefer_layer2`：第二层放得下就走第二层）。Refiner 2048²：原生 52.67，第二层 10.06（15.0 s），第一层 0.77（22.6 s，估算是上界）→ 单帧也第二层优先。待注册。HunyuanVideo 1.0 的第一层看第二层真机数，够就写「第二层已够」（DESIGN §9.41）。

### C 组：CogVideoX

| VAE | 模型 | 720×480×49 | 832×480×81 原生 → L2 | 720p×81 原生 → L2 |
|---|---|---|---|---|
| `cogvideo/vae.py`（GroupNorm、`conv_caches`）| CogVideoX 2B / 5B、I2V、Inpaint | — | 53.12 → 19.28 | 121（OOM）→ 45.41 |

* 原生：第一段**整段视频**一起过 `conv_in`、mid、前两级上采样（时间 ×2），第二段每 4 帧一块过后两级（`conv_caches`）；各块 `.cpu()` 存主机最后 `cat`——**这违反「不经过内存」**，Monoload 接管时第二段的输出要直接写设备上的输出（H3）。
* L1：第一段的 GroupNorm 统计量跨整段视频 → 统计遍按「每行全部帧」（H2 平面数 = T）；第一段的低分辨率部分进存档；第二段条带为主、按 4 帧块推进、条带自己的 `conv_caches`。
* L2 的探测现在会失败（读 meta 张量的值），H1 里要修。

### D 组：Cosmos 1.0、Mochi（整段一次、没有时间块）

| VAE | 模型 | 832×480 原生 → L2 | 720p 原生 → L2 |
|---|---|---|---|
| Cosmos CV8x8x8（因式分解卷积、每帧 1 组 GroupNorm、空间注意力 + 每像素时间注意力、haar）| Cosmos 1.0 T2V / I2V（7B / 14B）| 11.63 → 4.15 | 21.27 → 11.05 |
| Mochi（`PConv3d` 时间复制补边、每帧 GroupNorm、`DepthToSpaceTime`）| Mochi 1 | 337（OOM）→ 54.36（79 帧）| 762（OOM）→ 106（OOM）|

* **结果（D 组做完，DESIGN §9.43）：不做第一层，改成逐帧流式的第二层。** 上面「时间卷积非因果、没有缓存可借」的判断不对：v0.46.3 里两种 decoder 跨帧的只有**因果**时间卷积（Mochi `PConv3d` 前面补首帧，Cosmos `CausalConv3d` 首帧重复）和时间上采样，Cosmos 的时间注意力在第一次时间上采样之前（latent 分辨率，整段照原生跑）。所以按帧深度优先、每个因果卷积带前 k−1 帧缓存，就能一帧一帧地流过去（`vae_mochi.py` / `vae_cosmos.py`）。
* 真机（reserved Δ）：Cosmos 1280×704×121 原生 30.18 → **2.96 GiB**；Mochi 848×480×85 原生 OOM → tiled，Monoload **5.16 GiB**（模拟 1280×720×85 是 11.45）。都远低于 DiT，第一层换不来决定性的差别。

### E 组：SVD（4D，batch 是时间轴）

* 阶段 C 做**整批 L2**（与原生同样的子批划分一次进 `OpChunking`；原生按空闲内存分子批，结果随空闲内存变——Monoload 选的子批要与原生在同样空闲内存下相同，或者整批，文档写清）。只省约 20%（GroupNorm 跨整批帧）。L1 以后再说（统计遍按「每行全部帧」，H2 可描述）。
* 模型：SVD / SVD-XT、SV3D（图像转 3D 轨道视频，同 VAE）。

### F 组：LTX

| VAE | 模型 | 1280×704×97 原生 → L2（模拟）|
|---|---|---|
| LTX 0.9.0 卷积 VAE（`causal_video_autoencoder`，`comfy_has_chunked_io`，PixelNorm）| LTX-Video 0.9.0 | 51.35 → 6.40 |
| LTX 0.9.5+ / LTX-2 卷积 VAE（version 2）| LTX-Video 0.9.5+、LTX-2（LTXAV 的视频部分）| 59.33 → 6.94 |
| **LTX 2.4 扩散 decoder**（`na_diffusion_decoder.CausalDiffusionVAE`，v0.46.3 新增；就是 LTX-2.5 官方视频 VAE，sd.py 叫它 2.4，加它的提交叫 2.5）| LTX-2.5（官方 VAE `ltx-2.5-video-vae`；`ltx-2.5-video-vae-conv` 是卷积 VideoVAE，归上一行）| 未模拟（真机见 HANDOFF §4）|
| LTX-2 音频 VAE | LTXAV 的音频 | 原生（音频不管）|

* 卷积 VAE（**定案：只走第二层，不做第一层**）：原生按**总**显存定块大小（32–128 MiB）沿 T 切块递归、写进 `output_buffer`。真机 LTX-2（2026-11-26，`tests/bench_ltx.py`，基线在 `results/native/`）：1280×704×97 原生 reserved 56.76 / alloc 9.93 GiB → 第二层 7.99 / 7.40，11.7 → 11.4 s；1920×1088×193 原生 55.95 / 18.56 → 第二层 16.14 / 13.50，51.5 → 52.4 s。原生的 reserved 是碎片（ROCm 上 cudnn 关着，im2col 一次要 5.8 GiB，OOM 后清缓存才跑完），第二层的卷积行块把它消掉了，时间不变；第二层已经比要求的低得多，第一层（PixelNorm 空间条带）收益不值，不做。和原生的 max|d| 0.014 / 0.033 来自 GEMM 分组和 batch>1 时解码噪声的已知差别。
* **LTX 2.4 扩散 decoder（定案：算子级 NA3D 按 T 切块 + 光环 kernel−1，不走条带引擎）**：没有卷积和已知注意力（邻域注意力 `comfy_kitchen.na3d` + SwiGLU MLP），第 5 段（256 通道、像素 1/4 分辨率、8 个块、窗口 11³）原生峰值约 6 × S（S = 第 5 段一份激活；整份 q/k/v/注意力输出 + x + context），随帧数和分辨率线性涨。做法在 `monoload/vae_ltx.py`，经 `vae_ops.EXTRA_CHUNKERS` 由第二层的 `OpChunking` 装上：
  * 每个 `NeighborhoodAttention3D` 按 T 分段：一段查询帧 [a, b) 配一段 n = (b − a) + 2(k_t − 1) 帧的 q/k/v，贴边时整段往里挪（不截短）。NATTEN 语义是窗口正好 k_t 帧、居中、在边界处往里挪，窗口离它的查询最远 k_t − 1 帧，所以查询块两边各有 k_t − 1 帧（或碰到边界）时每个查询在段里的窗口和整块里的完全相同。一次调用的所有段等长、共用一份 q/k/v（分配器反复用同一块，不碎）。只切 T、H/W 不切 → RoPE 直接用全局位置表按 T 切片；HIP 的 na3d 是融合 WMMA kernel，自己没有大 workspace，段的 T × H 还要 ≤ 65535（kernel 的 grid 上限）。kernel 对段里每一帧都算注意力，光环那部分算完扔掉，所以查询块取「4n 帧的 q/k/v/输出放得进 max(工作区, 半份激活)」的最大值、且至少 2(k_t − 1) 帧（注意力最多是原生的两倍）——第一版按 1 GiB 工作区定块，第 5 段只剩 1 帧、段 21 帧，真机慢了约 14 倍，段长不一还把 reserved 碎到 alloc 的 3 倍。残差原地写回 x：一段读完它的帧以后，下一段起点之前的结果立刻写，剩下的挂起。进第 5 段前清一次缓存。
  * SwiGLU 按 workspace 限 token 数（原生最少一整帧）；`NADiffusionDecoder.forward` 换成同样步骤的精简版：最后一次上采样只算原生保留的帧（原生用视图丢掉开头重复帧和末尾 16 个 padding 帧，整块仍被拖着），`conv_in_x_t` / `norm_out` / `conv_out` 按 T 分块直接写结果，最后一步的块跑完就放掉 context；`x_t` 噪声仍整块用同一个 generator 生成，和原生逐位相同。
  * 峰值 ≈ x + context + 一段 q/k/v/输出 + 挂起结果 + 像素张量，193 帧约 3 × S。和原生的差别只来自 GEMM 行分组不同（fp32 自测 ≤ 1e-4，`tests/check_ltx_dd.py`）。
  * **为什么不走条带引擎**：条带的光环要把各层窗口半径累加——第 5 段 8 层 × 每层 10 帧（k_t − 1）= 80 帧，加上前 4 段，一个条带的重算量比它本身还大；而注意力本身逐层就能精确切开（每层只要 k_t − 1 帧光环、不累加），切完以后剩下的整份张量只有 x 和 context 两份，已把 1920×1088×193 压进 55 GiB 主线（真机 reserved 39.23）。真要更大（x + context 本身超了）才需要跨层流式，到时再议。
  * **真机（2026-11-27，`tests/bench_ltx.py`，`ltx-2.5-video-vae-bf16`，`--bf16-vae`，1 GiB 工作区）**：768×512×97 走 Monoload 正常入口（适配线程的探测定上界 + `vae_ltx`）reserved 9.84 → 5.59（冷）/ 5.52（热），热跑 17.9 → 31.3 s，冷跑 109.7 s（多出来的约 78 s 是两次探测解码）；1920×1088×97 直接装 `vae_ltx`（lean）reserved 50.15 → 26.07（alloc 38.82 → 25.12），106.3 → 190.9 s；1920×1088×193 lean reserved 39.23 / alloc 38.28，367.2 s（原生超 55，不跑）。和原生 max|d| 0.002 / 0.027（bf16 的 GEMM 分组差别；fp32 自测 ≤ 5.6e-6）。**定案：合，扩散 decoder 交给 `vae_ltx`**（峰值约减半，慢约 1.8 倍：多出来的是光环帧的重复注意力，块已取到预算内最大）。
  * `plan()` 的总数是 reserved 口径：allocated 上界 × 1.05 + 256 MiB（1920×1088×97 估 25.17 → 26.68 ≥ 实测 26.07；fp32 小尺寸 2.55 → 2.93 ≥ 2.70）。路由用的仍是适配线程的探测，`plan()` 给诊断和 `check_ltx_dd.py` 对照。
* H4：PixelNorm 的分组归约已在第二层里（video-norm-g）。

### G 组：本来就自己分块的 transformer VAE（结论：原生已够，Monoload 只透传，不做 L1 / L2）

五个类都声明 `comfy_has_chunked_io`（往 `VAE.decode` 预分配的输出里写），`vae._native_reason` 在看帧数之前就把它们判成原生（`vae.nr_chunked_io`），图像和视频、单帧和多帧都一样；`tests/test_vae_selftiled.py` 锁住这条路由（不加载权重，只用类，19 项，真机 2026-11-26 全过）。**H1 放开多帧时，`comfy_has_chunked_io` 的判断要留在多帧白名单之前**（或白名单里不放这五个类），否则第二层的 8×8 探测、按面积外推会套到它们的 Linear / 自带注意力上，只添乱。

为什么不做：它们自己把显存限住了，峰值 ≈ 权重 + 输出 + 一个块的工作量，工作量和帧数基本无关。

| VAE（类） | 自己怎么分块 | 原生峰值（CT 700 实测，reserved） | 结论 |
|---|---|---|---|
| MiniMax H3 视频，原版 decoder（`MiniMaxH3VideoVAE` + `ViT3DDecoder`） | 时间上 5 + 2 个 latent 一块，空间 256 px tile（一次最多 4 块），边解边写输出 | 见下表：1080p 107 帧总共 11.8 GiB（权重 4.85 + 输出 2.5 + 工作量 < 2.5）| 原生已够，只透传 |
| MiniMax H3 视频，Sol decoder（`NonCausalViT3DDecoder`） | 整段的残差流 + 5 个 latent 一块的 q/k/v 环（ring 2）| 1080p 73 帧 ≤ 12.73 GiB、107 帧 ≤ 14.63 GiB（权重 3.35 + 输出 1.70 / 2.50 + 工作量；alloc 多占 ≤ 6.32 / ≤ 7.44）。按代码：残差流每输出像素·帧 4 B（2048 维 bf16 ÷ 16×16×4 像素），只有 fp32 输出（12 B）的三分之一；q/k/v/注意力输出的环 1080p 约 1.3 GiB、单块 qkv / MLP 临时量约 1 GiB，与帧数无关——实测 34 帧多出的 1.1 GiB alloc 就是输出 0.8 + 残差流 0.3 | 只透传：随帧数涨的部分以 ComfyUI 预分配的输出为主，第一层省不掉它；残差流整段在显存里是这个 decoder 的设计（不重算时间重叠），拆开就不是同一个结果 |
| MiniMax H3 Image / P-H3 Image（`H3ImageVAE` 子类） | 同原版的 256 px tile | 没有单独的 Image 模型文件（Comfy-Org/MiniMax-H3 只有 fl2va / ref2va），按原版的数推 | 只透传 |
| SeedVR2（`VideoAutoencoderKLWrapper`） | 按空闲显存定每片几个 latent 帧；卷积 / GroupNorm 0.5 GiB 一块；放不下时把因果缓存放到锁页内存（`CausalMemoryCache(offload=True)`，每输出像素 6200 B）| 1080p 33 帧：alloc 31.35 GiB、reserved 43.11 GiB（GGUF F16 反量化后按 `--bf16-vae` 跑，权重 0.47，输出 0.76，耗时 160 s；因果缓存放到了锁页内存）。4K 33 帧：原生估算 123.5 GiB，ComfyUI 直接改走分块解码（`preferred_decode_tile`，最大 768 px），分块峰值没测（Monoload 不改这条路）。估算来源：`comfy_memory_used_decode` 按实测拟合（`comfy_decode_estimate_is_reliable`），每输出像素 3150 B，**bf16 再 ×5**（只有 fp16 channels-last 走拟合的快路径） | 只透传：峰值由它自己按空闲显存规划、估算可靠、放不下自动分块，第一层要重写它的因果缓存机制（4a 盘点就不建议）。能省的是精度：SeedVR2 VAE 用 fp16 比 bf16 少约 5 倍，这是用户侧的选择，不是 Monoload 的事 |
| RelayVSR Flash（`RelayFlash`） | 逐帧解码，窗口注意力 48×48 / 24×24，`HostWriter` 直接写输出 | x4：1080p 31 帧 2.02 GiB、4K 31 帧 ≤ 7.70 GiB（权重 0.06，输出 0.72 / 2.87，4K 耗时 21 s）。按代码：一帧的工作量，和帧数无关 | 只透传 |

**MiniMax H3 原版 decoder 实测**（`minimax_h3_video_vae_fp16.safetensors`，`--gpu-only --bf16-vae`，权重 4.85 GiB；随机 latent；`tests/bench_selftiled.py`；原生和装了 Monoload 的输出逐位相同）：

| 尺寸 | 帧（latent） | 耗时 | 解码多占（alloc / reserved）| 其中输出（fp32）| 总 reserved |
|---|---|---|---|---|---|
| 1280×720 | 22（7） | 13 s | 0.84 / 1.49 GiB | 0.23 | 6.47 |
| 1280×720 | 107（32） | 76 s | ≤ 2.20 / ≤ 3.26 | 1.10 | ≤ 8.24 |
| 1920×1088 | 22（7） | 28 s | 1.35 / 2.40 | 0.51 | 7.38 |
| 1920×1088 | 107（32） | 159 s | ≤ 5.00 / ≤ 6.79 | 2.50 | ≤ 11.77 |

「≤」：这一轮的脚本在读峰值前把输出（`movedim` 过、不连续）在显存里拷了一份再算哈希，多出一份输出大小，所以长片段的数是上界（d3f7707 已改）；短片段的数不受影响（拷贝比解码本身的峰值小），由它可知工作量约 0.6 GiB（720p）/ 0.85 GiB（1080p），不随帧数涨。ComfyUI 自己的估算（`memory_used_decode`：720p 0.58–1.02、1080p 1.05–1.66 GiB）不含输出，只拿去 `load_models_gpu` 腾地方，不影响结果。单次峰值由 DiT / TE 决定（§3），VAE 不是瓶颈。上面 Sol、SeedVR2、RelayVSR 三行也是同一版脚本测的，带「≤」的数同样含这份拷贝。

**GGUF VAE**：社区有 MiniMax H3 视频 VAE Q8_0（2.94 GB）和 SeedVR2 VAE F16（0.50 GB）的 GGUF，但 ComfyUI 核心的 `VAELoader` 不列 `.gguf`，ComfyUI-GGUF（kyuz0 H3）只有 UNet / CLIP 加载器——**没有能把它们当 VAE 加载的节点**（真机核实）。`bench_selftiled --gguf` 用 gguf 包读出来、反量化成 fp16 再构造 `comfy.sd.VAE`（SeedVR2 F16：250 个张量，正常构造），解码就是 fp16 VAE 的解码，峰值同 safetensors 版。

真机命令（在容器里另起进程、拿 GPU 锁、先让服务器卸模型；结果写进 `/srv/comfy-latest/results/native/`）：

```bash
/srv/comfy/gpulock run "G组: MiniMax 原生峰值" docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui-latest python tests/bench_selftiled.py --vae minimax_h3_video_vae_fp16.safetensors --res 1920x1088 --frames 22,107 --modes native,monoload
/srv/comfy/gpulock run "G组: SeedVR2 原生峰值" docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui-latest python tests/bench_selftiled.py --gguf seedvr2_ema_vae_f16.gguf --res 1920x1080,3840x2160 --frames 33 --modes native
/srv/comfy/gpulock run "G组: RelayVSR 原生峰值" docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui-latest python tests/bench_selftiled.py --vae relayvsr_flash_vae_x4_fp32.safetensors --relay --res 1920x1080,3840x2160 --frames 31 --modes native
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui-latest python tests/test_vae_selftiled.py   # 不占 GPU（--cpu），预期 19 项全过
```

预期：`bitwise native == monoload: True`；MiniMax 1080p 22 帧解码多占 ≈ 1.35 GiB（alloc）；没有 `TILED_FALLBACK` / `OOM`。

### H 组：预览用的小 VAE（不做）

TAESD / TAEF1 / TAEF2、**Qwen Image 2.1 TAESD（64 ch，16×，RGBA，v0.46.3 新增）**、TAEHV / lighttaew2.1 / TAEW2.2 / lighttae_hv15、MiniMax H3 的 TAEHV：几 MB 的权重，峰值本来就低；保持原生 / 第二层现状。

### v0.46.3 的新图像 VAE

| VAE | 状态 |
|---|---|
| Qwen Image 2.1 VAE（`vae2_2.WanVAE`，patch_size 1、temporal_kernel 1、z 64、RGBA，4D latent）| **阶段 A 已做**：走 Wan 2.2 适配器的第一层（`test_vae_wan22` 第 9 节），待真机 |
| Qwen Image 2.1 TAESD | H 组，不做 |
| MiniMax H3 Image / P-H3 Image | G 组 |
| Trellis2 shape / texture VAE、Hunyuan3D、TripoSplat | 3D，不是图像 / 视频解码，原生 |
| MiniMax Music3 DAV、YuE2 Oobleck、MiniMax H3 音频、Stable Audio 3 等 | 音频，原生 |

## 3. 模型清单和单次峰值

（见下表；尺寸来自 Hugging Face 文件页，按 ComfyUI 实际在用的量化形式。单次峰值 = max(DiT, TE, VAE 解码)；VAE 解码按 Monoload 管理后的估算。）

单位：Hugging Face 文件页的 GB（10⁹ 字节；GiB ≈ GB × 0.93）。「?」= 没查到。来源仓库：Comfy-Org 各 *_repackaged、city96 / QuantStack 的 GGUF、Lightricks、Kijai、kandinskylab、mcmonkey、numz、zai-org（2026-11-26 查）。**采样时的单次峰值还要加 DiT 的激活**（720p×81 量级几 GiB，由各组线程在真机上量），表里只算权重。

| 家族 / 变体 | DiT（bf16 · fp8 · GGUF Q8 · Q4_K_M）| TE | VAE（文件）→ 组 | 单次最大（权重）| 分组 |
|---|---|---|---|---|---|
| Wan 2.1 T2V 14B | 28.58 · 14.29 · 15.88 · 10.12 | umt5_xxl 11.37 · fp8 6.74 | wan_2.1_vae 0.25 → A | DiT | ≤ 55 |
| Wan 2.1 I2V / FLF2V 14B（+ clip_vision_h 1.26）| 32.79 · 16.40 · 18.14 · 11.34 | 同上 | 同上 → A | DiT | ≤ 55 |
| Wan 2.1 1.3B（T2V / Fun / Camera / VACE 1.3B）| 2.84–4.31 | 同上 | 同上 → A | TE | ≤ 55 |
| Wan 2.1 VACE 14B / Fun Camera 14B / MagRef | 34.68 / 33.80 / 32.79（无 fp8）| 同上 | 同上 → A | DiT | ≤ 55 |
| Wan 2.1 HuMo / FlowRVS / SCAIL / SCAIL2 / CausalAR | ?（14B 级，约 28–35 · 14–17）| 同上 | 同上 → A | DiT | ≤ 55（待核实）|
| Wan 2.2 T2V / I2V A14B（高噪 + 低噪两个，分阶段只驻留一个）| 每个 28.58 · 14.29 · 15.41 · 9.65 | 同上 | wan_2.1_vae → A | DiT | ≤ 55 |
| Wan 2.2 TI2V 5B / Fun 5B | 10.00 · — · 5.40 · 3.43 | 同上 | wan2.2_vae 1.41 → A | TE | ≤ 55 |
| Wan 2.2 S2V 14B（+ wav2vec2 0.63）| 32.59 · 16.39 | 同上 | wan_2.1_vae → A | DiT | ≤ 55 |
| Wan 2.2 Animate 14B | 34.55 · int8 18.41 · 18.72 · 11.50 | 同上 | 同上 → A | DiT | ≤ 55 |
| Wan 2.2 Fun Camera / Control / Inpaint / VACE 14B、ChronoEdit | 28.58–34.68 · 14.29–17.35 | 同上 | 同上 → A | DiT | ≤ 55 |
| Wan 2.2 RelayVSR / WanDancer、Wan Animate 2 | ? | 同上 | RelayFlash → G / wan_2.1_vae → A | ? | 待核实 |
| HunyuanVideo 1.0 T2V / I2V / SkyReels | 25.64 · ? · 13.97 · 7.88 | llava_llama3 16.07 · fp8 9.09（+ clip_l 0.25）| hunyuan_video_vae_bf16 0.49 → B | DiT | ≤ 55 |
| HunyuanVideo 1.5（T2V / I2V，480p / 720p，蒸馏版）、SR distilled | 16.65 · 8.33 | qwen_2.5_vl_7b 16.58 · fp8 9.38（+ byt5 0.44）| hunyuanvideo15_vae_fp16 2.52 → B | DiT / TE | ≤ 55 |
| Kandinsky 5 Video Lite 2B | 4.57 | qwen_2.5_vl_7b（同上）+ clip_l | hunyuan_video_vae → B | TE | ≤ 55 |
| Kandinsky 5 Video Pro 19B | 43.39（只有 bf16）| 同上 | 同上 → B | DiT 40.4 GiB | ≤ 55（激活余量小）|
| LTX-Video 0.9.x 2B / 13B | 5.72–9.37 / 28.58 · 15.69 | t5xxl 9.79 · fp8 5.16 | 内嵌（卷积 VAE）→ F | DiT | ≤ 55 |
| LTX-2 19B（音视频）| 43.29 · 27.08 · fp4 19.99 | gemma_3_12B 24.38 · fp8 13.21（+ text_projection 2.31）| LTX-2 卷积 VAE + 音频 → F | DiT | ≤ 55 |
| LTX-2.3 22B（音视频）| 完整 46.15（含 VAE）· transformer-only bf16 42.02 · fp8 23.47–25.23 · int8 21.51 | 同上 | LTX23_video_vae 1.45 + audio 0.36 → F | DiT | ≤ 55（bf16 激活余量小）|
| LTX-2.4 / 2.5 | ?（下载线程在下 unsloth 的 GGUF 镜像，尺寸以它为准）| ? | 2.4 扩散 decoder → F | ? | 待核实 |
| MiniMax H3 pruned（视频 + 音频）| 40.23 · 20.96 · int8 20.97 · w6a8 15.98 | qwen3vl_32b 51.51 · int8 27.14 · nvfp4 15.69 | video 5.21（int8 2.81）/ sol 3.60 / audio 0.61 → G | **TE bf16 48 GiB** | ≤ 55（TE bf16 时 TE 和 DiT 必须分开加载）|
| MiniMax H3 完整（fl2va / ref2va）| **66.28** · int8 34.04 | 同上 | 同上 → G | **DiT bf16 61.7 GiB** | **bf16：> 55，等宿主机调 ttm 后再验**；int8：≤ 55 |
| MiniMax H3 Image | ? | 同上 | H3 图像 VAE → G | ? | 待核实 |
| Mochi 1 | 20.06 · 10.03 | t5xxl 9.79 · 5.16 | mochi_vae 0.92 → D（流式第二层，720p 模拟 11.45）| DiT | ≤ 55（VAE 原生 OOM）|
| Cosmos 1.0 7B / 14B | 14.47 / 28.53 | oldt5_xxl ? | CV8x8x8 ? → D | DiT | ≤ 55 |
| Cosmos Predict 2 视频 2B / 14B | 3.91 / 28.53 | oldt5_xxl ? | wan_2.1_vae（未核实）→ A | DiT | ≤ 55 |
| CogVideoX 2B / 5B / I2V / Inpaint | 5B 11.15（2B ?）| t5xxl 9.79 · 5.16 | 0.86 → C | DiT | ≤ 55 |
| SVD / SVD-XT、SV3D | 9.56 | 无（CLIP vision）| SD VAE（内嵌）→ E | DiT | ≤ 55 |
| SeedVR2 3B / 7B（视频超分）| 6.78 · 3.39 / 16.48 · 8.24 | 无 | ema_vae_fp16 0.50 → G | DiT | ≤ 55 |

**结论：**
* **> 55 GiB 只有一项：MiniMax H3 完整版的 bf16 DiT（61.7 GiB）** → 单列「等宿主机调 ttm 后再验」；它的 int8（31.7 GiB）和 pruned 各档都在主线。
* **要分开加载 TE 和 DiT 的**：MiniMax H3 用 qwen3vl_32b bf16（48 GiB）时，任何 DiT 都放不进剩下的 7 GiB；其余家族 TE + DiT 同时驻留也多在 55 以内（最大的组合 LTX-2.3 bf16 42 + gemma bf16 22.7 ≈ 65 GiB 也超，ComfyUI 本来就会按需卸 TE，真机看日志确认）。
* **VAE 解码在 Monoload 管理下都远小于 DiT**（§2 的 L2 数字 4–45 GiB，L1 预计 1–3 GiB + 输出），不会决定单次峰值；Mochi 原来是例外（整段 L2 在 720p > 60），D 组的流式第二层已降到 11.45（模拟）；CogVideoX 720p（L2 45 GiB，接近线）。原生解码有好几种超过 55（HunyuanVideo、CogVideoX、Mochi 的 720p），这正是要 Monoload 的地方。
* 主线验证的模型按价值排：Wan 2.2 A14B（fp8 / GGUF）、Wan 2.2 TI2V 5B、HunyuanVideo 1.5、LTX-2.3、HunyuanVideo 1.0、Kandinsky 5、CogVideoX、Cosmos Predict 2、Mochi、MiniMax H3（pruned / int8）、SeedVR2、SVD；Cosmos 1.0 与 LTX 0.9.x 只做冒烟。

## 4. 并行开工的顺序

1. **阶段 C（本线程）先出 H1 + H3 + H6**（多帧第二层的入口、直接写输出、`--frames`）：A / B / C / F 组的第二层立刻能用，真机能测原生基线。
2. 同时开：**A 组（Wan）L1**、**G 组测量**（只要 H6）、**F 组的原生基线**（只要 H6）。
3. H2 + H4 + H5 合进 video 后开：**B 组（Hunyuan）L1**、**C 组（CogVideoX）L1**。
4. 最后：E 组的 L1、F 组 L1（看基线）。D 组（Mochi / Cosmos）已用流式第二层解决，不做 L1（DESIGN §9.43）。

每组一个线程、一个功能分支（从 video 开，合回 video），各自写真机命令（`docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui-latest python tests/...`，带预期和看哪项）交给验证线程。改 `vae_engine.py` / `vae.py` 的共用部分只在阶段 C 做；各组线程只加适配器文件和测试，需要改共用部分时先发给本线程。

## 5. GGUF

* DiT / TE：GGUF 线程（量化 LoRA 运行时合并）。
* VAE：**不做（原计划的钩子 H7 已取消，2026-11-26）。** ComfyUI v0.46.3 核心和 kyuz0/ComfyUI-GGUF-H3 6d33de2 都没有 GGUF VAE 的加载器（GGUF 线程核实，见 `docs/GGUF-QUANT.md`）；现成的视频 VAE GGUF 文件只有 MiniMax-H3 Q8_0、SeedVR2 F16（社区），加载不了。另外原设想的「每模块反量化一次、缓存到解码结束」等于同时多留一份权重，违反「权重只留一份」，也不该做。以后真有 GGUF VAE 加载器再议。
