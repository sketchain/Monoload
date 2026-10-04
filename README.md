# Monoload

**ComfyUI 插件：打 LoRA 时不改原权重、不做任何备份，每一层在计算的那一刻才临时合并 LoRA，用完即丢；每个 prompt 结束后自动释放 LoRA，底模继续常驻。** 装上后对所有模型生效（`UNETLoader`、`CheckpointLoaderSimple`、`CLIPLoader` 加载出的模型都算），原生的 `LoraLoader` / `LoraLoaderModelOnly` / Hook LoRA 节点照常使用，工作流不用改。

合并默认走快速路径：普通 LoRA / LoCon 直接用一次融合的 fp16 `addmm_` 加进临时权重，其他类型在计算 dtype 下合并。它与原生合并的差异在 fp16 舍入量级（CT 700 实测：合并后权重的差异是 LoRA 改动量的 0.25%，单个元素最多差 1 个 ulp，见 §5.1）。设高级选项 `MONOLOAD_EXACT=1`（§13）时，结果与原生 ComfyUI **逐位一致**，但每步更慢。fp8 这类量化层在两种模式下都有意做得比原生更精确（见 §5）。

为 Strix Halo（gfx1151，统一内存，显存走 GTT）+ ROCm、`--gpu-only` 的配置设计。在这种配置下，原生 ComfyUI 打 LoRA 时会把被改动的原权重在 GTT 里再备份一份，LoRA 改到的权重越多，备份就越大。

**另一项功能：VAE 解码降峰值（§12）。** 接管 `VAE.decode`。认得的结构（目前是 `qwen_image_vae` / Wan 2.1 VAE 的单帧解码）走第一层：从低分辨率存档按输出行条带倒推重算，默认在不明显变慢的前提下取最低的峰值（4K 约 0.9 GiB，原生约 59 GiB）；其余 VAE 走第二层：卷积按输出行分块、注意力按 query 分块，限制 im2col 展开缓冲和注意力分数矩阵；自己给 `load_models_gpu` 报真实的内存需求，不再用原生约 92 GiB（SDXL 4K）的估算去卸载别的模型；OOM 时只缩小分块重试，绝不退回 tiled 近似解码。结果与整图解码数学等价（只差浮点误差）。与 LoRA 部分相互独立，可以单独关掉。

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
      # - MONOLOAD=0                         # 总开关：全局原生，只有节点上明确开启的模型 / VAE 走 Monoload（§4）
      # - MONOLOAD_DISABLE=1                  # 什么都不装，完全原生
      # 其余 MONOLOAD_* 是高级选项（全局默认值），见 §13
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
#   [Monoload] master switch MONOLOAD: on
#   [Monoload] LoRA: runtime merge (no in-place LoRA, no weight backups), merge fused fp16 addmm / relaxed (MONOLOAD_EXACT=1 for bit-exact); released after every prompt (base models stay loaded)
#   [Monoload] VAE decode managed: op-level chunking (conv row blocks, attention query blocks), workspace 1.00 GiB ...
#   [Monoload] VAE layer 1 (stripe decoding) on for recognized decoders (Wan 2.1 / qwen_image_vae single frame; ...): default stripe policy: the peak of 128-row stripes, tallest stripes within it ...
```

`MONOLOAD=0` 时第一行是 `master switch MONOLOAD: off (MONOLOAD=0): native ComfyUI unless a Monoload node enables Monoload for its model / VAE`，VAE 一行是 `VAE decode: native by default (MONOLOAD=0) ...`。

没有额外依赖。节点在分类 `Monoload` 下（§4.1）。以只读方式挂载即可。

## 4. 开关与节点

只有一个总开关 `MONOLOAD`，外加完全不装的 `MONOLOAD_DISABLE`：

| 设置 | 效果 |
|---|---|
| 不设，或 `MONOLOAD=1`（默认） | 所有模型和 VAE 用 Monoload 的默认策略：LoRA 运行时合并（快速路径，§5.1），每个 prompt 结束后释放 LoRA（日志 `[Monoload] released LoRA after prompt: ...`），VAE 解码管理（§12） |
| `MONOLOAD=0` | **全局原生**：插件照常装上，但钩子把每个调用原样交给 ComfyUI。工作流里没有 Monoload 节点时，LoRA 和 VAE 的结果都与原版 ComfyUI **逐位一致**（权重原地合并 + 备份也是原生自己的），额外开销是每次调用多一次判断（实测 `patch_weight_to_device` 每次 +0.2 µs，`VAE.decode` 每次约 3 µs）。只有节点上**明确开启**的那个模型 / VAE（LoRA 节点的 `mode` 选 `enable`，VAE 节点的 `mode` 选 `auto`）走 Monoload |
| `MONOLOAD_DISABLE=1` | 什么都不装（没有任何钩子），就是原版 ComfyUI。Monoload 节点仍然注册（保存的工作流照常能打开），但原样输出输入，日志里说明一次 |

**优先级（逐项判断）：节点上明确选的 > 高级选项（环境变量，§13） > 内置默认。** 环境变量是全局默认值；节点上明确选的值只对那一个模型 / VAE 生效，而且总是压过全局。节点上留在 `default`（跟随全局）的项继承全局设置。例如 compose 里设了 `MONOLOAD_EXACT=1`（VAE 全局走原生），某个 VAE 节点的 `mode` 选了 `auto`：这个 VAE 照样走 Monoload 的解码管理，其他 VAE 原生。

高级选项（`MONOLOAD_EXACT`、`MONOLOAD_KEEP_LORA`、`MONOLOAD_DISABLE_VAE`、`MONOLOAD_VAE_*` 等）见 §13，一般不用设。开关接受 `1` / `true` / `yes` / `on`，`MONOLOAD` 另外接受 `0` / `false` / `no` / `off`（其他值按开启处理并警告）。改完要重启容器。

**界面语言：** 三个节点的名字、输入名、下拉选项的显示文字、提示跟随 ComfyUI 的语言设置（设置 → Comfy → Locale，中文 / English）；存进工作流的下拉取值始终是英文（`default`、`custom` 等），换语言不影响已保存的工作流。日志和报错默认英文，设 `MONOLOAD_LANG=zh` 改成中文（§13）；Monoload Info 节点的文字跟随 `MONOLOAD_LANG`。下文的选项按存储的英文取值写，中文界面上显示的是对应的中文（`default` = 跟随全局）。

### 4.1 节点用法：Monoload VAE Settings（单独设置某个 VAE）

全局设置（总开关和高级选项）对所有 VAE 生效。要让某一个 VAE 用不同的设置，在工作流里加 **Monoload VAE Settings** 节点（分类 `Monoload`）：输入一个 VAE，输出一个带设置的 VAE 副本，把副本接到 VAE Decode 等节点上。

* **副本共享权重**，不多占内存，加载 / 卸载仍由 ComfyUI 的模型管理统一处理；**输入的 VAE 不变**，工作流里其他直接用它的分支照旧按环境变量和默认值解码。
* 任何对副本调用 `vae.decode` 的节点都按节点的设置解码；`VAE Decode (Tiled)` 按现有规则保持原生；encode 不受影响。
* 一个工作流里可以有多个这样的节点（多个副本），各用各的设置。节点可以串联：下游节点没设的项沿用上游节点的值。

| 选项 | 取值 | 含义 |
|---|---|---|
| `budget` | `default` / `unlimited` / `custom` | 峰值预算。`default`：跟随全局（`MONOLOAD_VAE_BUDGET`，没设就是默认策略）；`unlimited`：这个 VAE 不限预算（默认策略），即使全局设了预算；`custom`：用 `budget_gib`——估算不超过它的做法里选预计最快的，一个都放不下就报错并写明各需要多少 |
| `budget_gib` | GiB，精度 0.01 | 只在 `budget` 选 `custom` 时生效（这时必须大于 0）；选别的时填了也不用，日志里说明 |
| `gn_scheme` | `default` / `A` / `B` / `C` / `D` | 强制 LDM decoder（SDXL / SD1.5 / SD3 / Flux `ae`）第一层的 GroupNorm 方案（§12.8）；`default` 跟随全局 |
| `stripe_rows` | 输出行数，`0` = 跟随全局 | 强制第一层的条带高度 |
| `mode` | `default` / `auto` / `layer 2 only` / `native` | `default`：跟随全局（默认开启时是 `auto`；`MONOLOAD=0`、`MONOLOAD_DISABLE_VAE=1`、`MONOLOAD_EXACT=1` 时是原生；`MONOLOAD_DISABLE_VAE_STRIPE=1` 时是 `layer 2 only`）；`auto`：**为这个 VAE 打开**解码管理，认得的 decoder 走第一层（条带），其余走第二层——全局关着（包括 `MONOLOAD=0`）也打开；`layer 2 only`：只用第二层；`native`：这个 VAE 用 ComfyUI 自己的解码 |

**GroupNorm 方案（`gn_scheme`）是什么：** 只对 LDM decoder（SDXL / SD1.5 / SD3 / Flux `ae`）的第一层（条带解码）有意义。GroupNorm 要用整张图的统计量，所以第一层在出图之前要先跑几遍「统计遍」；方案决定把哪些中间结果整张存下来，让这几遍从存档出发，而不必每遍都从 H/8 的存档重算：

| 方案 | 存什么 | 取舍 | SDXL 4K 实测（GTT / 耗时） |
|---|---|---|---|
| A | 什么都不存（只有 H/8 存档） | 内存最低，重算最多，最慢 | 约 1.1 GiB / 75 s |
| D | H/4 级的输出 | 介于 A 和 B 之间 | 约 1.5 GiB / 58 s |
| B | H/4 和 H/2 级的输出 | **内置默认** | 约 2.2 GiB / 42 s |
| C | B 再加上全分辨率每个块的输入 | 内存最高，最快 | 约 4.7 GiB / 36 s |

`default` = 跟随全局：看 `MONOLOAD_VAE_GN_SCHEME`；没设时用 B；设了预算（节点或环境变量）时，在放得下预算的方案里选预计最快的。选了具体方案就强制用它（预算只决定条带高度）。界面上的提示和 Monoload Info 里都有同样的一句说明。

节点上留在 `default` / `0` 的项跟随全局设置（高级选项，§13），全局也没设就用内置默认。例如 compose 里设了 `MONOLOAD_VAE_GN_SCHEME=D`，节点只把 `budget` 设成 `custom`、`budget_gib = 3`：预算来自节点，方案来自环境变量（强制 D），条带高度和模式用默认值。`MONOLOAD_DISABLE=1` 时节点原样输出输入的 VAE。

**控件顺序：** `budget` 下拉框紧挨在 `budget_gib` 前面。`budget_gib` 精度 0.01 GiB（填 0.25 就存 0.25）。dev 不做旧工作流兼容：之前存的用到这个节点的工作流，控件值会错位，要重新设一次。

**日志**：每次解码的那一行末尾写明设置和来源，例如 `settings: budget 3.00 GiB (node), GroupNorm scheme chosen by the budget (default), stripe rows auto (default), mode auto (default)`；`last_decode()` 里是 `settings` / `settings_source`。

**例子：** 同一个 SDXL checkpoint，生成 4K 时想把 VAE 解码的峰值压在 3 GiB 以内，其他工作流不变：

```
Load Checkpoint ──VAE──> Monoload VAE Settings (budget custom, budget_gib 3) ──VAE──> VAE Decode ──> Save Image
                └─VAE──> VAE Encode（原 VAE，不受影响）
```

4K 的这次解码按预算选第一层方案 B、12 条 180 行（预测约 2.4 GiB、约 41 s，§13 的表；CT 700 实测 2.43 GiB、41.0 s）；工作流里直接接原 VAE 的解码仍是默认的方案 B、17 条 128 行（2.17 GiB）。

### 4.2 节点用法：Monoload LoRA Settings（单独设置某个模型的 LoRA）

全局设置对所有模型生效。要让某一个模型（和它的 CLIP）的 LoRA 用不同的处理方式，在工作流里加 **Monoload LoRA Settings** 节点（分类 `Monoload`）。典型接法：

```
Load Checkpoint ──> Load LoRA（可以串好几个）──MODEL/CLIP──> Monoload LoRA Settings ──MODEL──> KSampler
                                                                               └─CLIP──> CLIP Text Encode
```

* 输入 MODEL（必接）、CLIP（可选）；输出 MODEL 和 CLIP。没接 CLIP 时 CLIP 输出是空的（`LoraLoaderModelOnly` 后面只接 MODEL 就行，不报错）。
* 输出是 ComfyUI 的 clone：**共享权重**、不多占内存，**输入不变**，工作流里其他直接用输入的分支照旧按全局设置。
* **放在 LoRA 加载器之前也可以**：设置存在模型的 `model_options` 里，ComfyUI 每次 clone（包括 `LoraLoader`、`LoraLoaderModelOnly` 内部的 clone 和 `CLIP.clone()`）都会复制它，所以 `Load Checkpoint → Monoload LoRA Settings → Load LoRA → KSampler` 和接在后面效果一样。节点可以串联：下游节点留在 `default` 的项沿用上游节点的值。

| 选项 | 取值 | 含义 |
|---|---|---|
| `mode` | `default` / `enable` / `native` | `default`：跟随全局（默认开启；`MONOLOAD=0` 时原生）；`enable`：这个模型用 Monoload 的运行时合并（不改权重、不留备份），全局关着（包括 `MONOLOAD=0`）也打开；`native`：这个模型用 ComfyUI 自己的 LoRA 处理（权重原地合并 + 备份） |
| `merge` | `default` / `fused` / `exact` | `default`：跟随全局（`MONOLOAD_EXACT`，没设就是 `fused`）；`fused`：默认的快速路径（普通 LoRA 一次融合的 fp16 `addmm_`，与原生差在 fp16 舍入量级，§5.1）；`exact`：与原生 ComfyUI **逐位一致**（每步更慢）。只在这个模型走 Monoload 时有意义 |
| `after_prompt` | `default` / `release` / `keep` | `default`：跟随全局（走 Monoload 的模型释放，`MONOLOAD_KEEP_LORA=1` 时保留；原生的模型保留，和原版一样）；`release`：每个 prompt 结束后释放这个模型的 LoRA（底模继续常驻）；`keep`：保留到下一个 prompt |

**优先级（逐项判断）：节点上明确选的 > 高级选项（§13） > 内置默认**，同 §4。例子：compose 里设了 `MONOLOAD=0`（全局原生），某个工作流的 SDXL 想用 Monoload 并且逐位一致：节点 `mode = enable`、`merge = exact`，`after_prompt` 留 `default`（这个模型走 Monoload，所以按全局默认释放）。

* **按模型生效的合并方式**：运行时 patch 在装上时记下这个模型的 `merge`，同一个进程里一个模型用 `exact`、另一个用 `fused` 互不影响。
* **同一个底模的两个 clone 设置不同时**（比如一个分支 `native`、一个分支 `enable`），设置不同的 clone 会换一个 `patches_uuid`，ComfyUI 在两者之间切换时会把权重还原再按各自的方式加载（测试确认交替加载各得各的结果，最后底模权重逐位还原）。
* 日志：节点执行时打一行 `[Monoload] Monoload LoRA Settings: LoRA settings: mode enable (node), merge exact (node), after prompt release (default)`（括号里是来源：`node` / `env MONOLOAD=0` 等 / `default`）。
* `MONOLOAD_DISABLE=1` 时节点原样输出输入的 MODEL / CLIP，日志说明一次。
* 不在范围内的照旧：bypass LoRA（`LoraLoaderBypass`）不经过运行时合并；`force_patch_weights`（保存合并后的模型）在走 Monoload 的模型上照旧报错——这时把那个模型的 `mode` 设成 `native` 即可，不用重启（§6）。

### 4.3 节点用法：Monoload Info（看 Monoload 在做什么）

**Monoload Info**（分类 `Monoload`）把当前状态写成文字，**显示在节点框里**，同时从 `text` 输出同样的 STRING（可以接到保存文本之类的节点）。三个输入都可选，什么都不接也不报错；每次运行都重新生成（不走缓存）；它是输出节点，`text` 不接也会运行。

| 接了什么 | 显示什么 |
|---|---|
| 总是显示 | 开头：Monoload 版本和 commit、总开关状态；**最后**：装上了哪些钩子，以及每一项全局默认值和来源（`env 变量=值` / `built-in` / `set at runtime`）——接了 vae / model 时也有 |
| `vae` | 这个 VAE 实际用的解码设置（逐项，带来源：`node` / `env ...` / `built-in`），以及**这个 VAE 对象**上一次解码的记录：第几层、方案、条带数和行数、工作区、估算、实测峰值（reserved 和 GTT 的增量）、耗时、OOM 重试次数；还没解码过就写 `not decoded yet`；当前模式下不生效的设置会标出来（`native` 时预算、方案、条带高度标「native 模式下不使用」，`layer 2 only` 时方案和条带高度标「只用第二层时不使用」）；所用的 GroupNorm 方案附一句说明 |
| `model` | 挂在这个模型上的 LoRA（文件名 × 强度，按 `LoraLoader` / `LoraLoaderModelOnly` 加载的顺序）、被改动的权重数、mode / merge / after prompt 三项设置和来源（模式是 native 时 merge 标「不使用」）、当前内存里的状态（Monoload 运行时合并 / 原生烘焙 + 备份数 / 没加载） |
| `images` | 不读内容，只用来**排顺序**：把 VAE Decode 的 IMAGE 接过来，Info 就在这次解码之后运行，显示的就是这次解码 |

```
Load Checkpoint ─VAE─> Monoload VAE Settings ─VAE─┬─> VAE Decode ─IMAGE─┬─> Save Image
                                                  └──────────vae──────┐ └─images─┐
Load LoRA ─MODEL─> Monoload LoRA Settings ─MODEL─┬─> KSampler          │          │
                                                 └──────model──────> Monoload Info <┘
```

* 解码记录按 VAE 对象分开：节点做的副本和原 VAE 是两个对象，各记各的，不会混。
* LoRA 名字来自 `LoraLoader` 的包装（只在克隆出的 patcher 的 `model_options` 里记下文件名和强度，不改 patch、uuid 和数值）。别的插件的 LoRA 加载器打的 patch 只显示「被改动的权重数」。
* 实测峰值只在 GPU 上有（CPU 上写 `n/a`）：reserved 是 PyTorch 的 `max_memory_reserved` 相对解码前的增量（起点之前先清一次分配器缓存，否则采样留下的缓存块会被解码复用，增量偏低）；这个结构在本进程里第一次解码时含首次自检的时间和内存，Info 会注明「含首次自检」；GTT 是解码期间每 20 ms 读一次 amdgpu 的 `mem_info_gtt_used` 得到的峰值增量。
* 显示方式：插件带一个前端扩展 `web/monoload_info.js`，用 ComfyUI 前端自带的文字预览控件（核心节点「Preview as Text」用的那个，`window.comfyAPI.textPreviewWidgets`）。锁定版本（ComfyUI 0.31.0，前端 1.48.7）上用浏览器实测过；前端没有这个控件时退回一个只读的文本控件。

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
| `force_patch_weights` | 有节点要求把 LoRA 合并进权重：`ModelSave`、`CheckpointSave`、保存合并后的模型等 | 在这个模型前面加 Monoload LoRA Settings，`mode` 选 `native`（§4.2）；或设 `MONOLOAD=0` / `MONOLOAD_DISABLE=1` 重启 |
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

`run_all.sh` 先跑 7 遍入口测试、总开关测试和两个 VAE 测试，然后把下面每个 LoRA 功能测试在两种合并路径下各跑一遍：先 `MONOLOAD_EXACT=1`（逐位一致），再默认路径。只想跑 VAE 部分时：`MODELS=/path/to/models tests/docker_run.sh python tests/test_vae.py`（不需要任何模型文件，`$MODELS` 可以是空目录）。

| 脚本 | 内容 |
|---|---|
| `tests/test_entry.py` | 按 ComfyUI 的方式加载插件：默认替换 `ModelPatcher` 的方法（`CoreModelPatcher` 也覆盖到）、包装 `PromptExecutor.execute_async` 和 `VAE.decode`；除 `MONOLOAD_DISABLE=1`（都不动）外每种开关组合都装上；`MONOLOAD=0` 时总开关关闭、VAE 全局默认原生；`MONOLOAD_KEEP_LORA=1` 时全局默认保留 LoRA；`MONOLOAD_DISABLE_VAE=1` / `MONOLOAD_EXACT=1` / `MONOLOAD_DISABLE_VAE_STRIPE=1` 时 VAE 全局默认分别是原生 / 原生 / 只用第二层；合并路径与 `MONOLOAD_EXACT` 一致；`decode_tiled` 从不被替换 |
| `tests/test_dtype_paths.py` | `EXACT=1`：参数 dtype × 计算 dtype × lora dtype（含 gfx1151 上的 fp16）× {基础 LoRA、hook、两者都有}，54 种组合逐位对比原生合并的数值。默认路径：同样的 dtype 组合 × 8 种 patch 组合（LoRA、两个 LoRA、strength_model ≠ 1、LoHa、LoRA+LoHa、diff、hook、LoRA+hook），共 144 项；每项要求与独立实现的参照（`addmm_` / 原生 `LowVramPatch`）逐位一致，并且与原生合并的差异在容差以内。两种模式都再加 fp8 参数的 54 种组合，对比「先反量化 + 同一路径」 |
| `tests/test_lora_hot.py` | 两条管线：`CheckpointLoaderSimple`；`UNETLoader` + `CLIPLoader`。同一串 LoRA 组合先用原生跑、再装上 Monoload 连续切换着跑。`EXACT=1`：TE 输出和采样结果与原生逐位一致。默认路径：UNet 和 TE 所有被 patch 的权重与原生合并的差异在容差以内（DESIGN.md §5.5），同样的 key 上重算逐位一致路径与原生逐位一致；latent 差异只报告。两种模式都检查无备份、权重不变、撤掉 LoRA 后逐位一致，加上 Hook LoRA、模型合并和报错场景 |
| `tests/test_quant.py` | fp8 scaled UNet + LoRA：与「先反量化被改到的层 + 同一合并路径」逐位一致、无备份、fp8 权重不变；与原生的误差只报告 |
| `tests/test_vae.py` | VAE 解码管理（§12），不需要模型文件：分块卷积 / 分块注意力与不分块的结果一致（含各种 kernel、stride、dilation、groups、padding、cast 路径 + weight_function、Wan CausalConv3d）；用 ComfyUI 自己的 LDM `Decoder` / `WanVAE`（小通道、随机权重）构造 SDXL 式、Flux 式、Qwen 式 VAE，受管理的解码与原生 `VAE.decode` 比较；多帧交给原生、OOM 缩小分块重试、下限时报错、绝不调用 tiled |
| `tests/test_vae_stripe.py` | VAE 第一层（§12.7），不需要模型文件：区间倒推对照暴力依赖展开；每个单元（残差块、上采样、head 卷积）在任意切片上「有效行」与整图逐行一致、紧邻的下一行不一致；用 ComfyUI 的 `WanVAE`（小通道、随机权重）在 fp32 下比较第一层与原生整图解码（条带 1 行、不整除、等于整图、按预算自动、默认策略、奇数尺寸、很小的 latent、batch 2、条带内再分块、bf16），块边界附近的误差不比其他区域大；识别（Dropout 训练态、forward hook、开关）；故意少算一行 halo 时自检能抓到并回退第二层；预算报错；OOM 缩小条带、到下限报错、不退回 tiled 也不退回第二层；默认策略（128 行条带的估算为目标）和开关的优先级；内存模型单调、最大的条带先跑；自检后和前缀 / 条带之间清空缓存；SDXL / Flux 结构不归 Wan 适配器（归 LDM 适配器）；单帧 Conv3d 改走 conv2d 与模块原样一致、只在该改的时候改 |
| `tests/test_master_switch.py` | 总开关和全局默认（§4），不需要模型文件：`MONOLOAD` 的解析；`MONOLOAD=0` 且没有节点时，挂 LoRA 的模型（整体加载、lowvram 部分加载、Hook LoRA）和 VAE 解码与卸掉钩子的原版 ComfyUI 逐位一致，备份也和原生一样；释放不运行；包装每次调用的开销；`MONOLOAD=0` / `MONOLOAD_DISABLE_VAE` / `MONOLOAD_EXACT` 下节点 `mode auto` 打开管理、`default` 原生；预算下拉框；`MONOLOAD_DISABLE` 时节点原样透传 |
| `tests/test_lora_node.py` | Monoload LoRA Settings 节点（§4.2），需要 `tests/make_synthetic_checkpoint.py $MODELS` 生成的随机权重 SD1.5 checkpoint 和 LoRA（约 2 GiB）：接口；逐项优先级和来源；clone 共享权重、输入不变、设置不同时换 uuid；节点放在 `LoraLoader` 前面设置也保留；串联；`LoraLoader` / `LoraLoaderModelOnly`（不接 CLIP）/ 两个串联的 `LoraLoader`：`exact` 和 `native` 与卸掉钩子的原版逐位一致（`native` 的备份数也一样），`fused` = 不加节点的默认路径；同一底模的不同设置交替加载各得各的结果、底模权重逐位还原；`MONOLOAD=0` 下 `enable`；prompt 结束后的释放 / 保留；`MONOLOAD_DISABLE` 透传 |
| `tests/test_info_node.py` | Monoload Info 节点（§4.3），不需要模型文件：接口（输出节点、输入都可选、不缓存、界面文字 = STRING 输出）；不接输入时的版本 / 总开关 / 全局默认和来源；VAE 的设置来源、`not decoded yet`、解码记录归属（副本和原 VAE 不混）、原生解码也有记录、每次刷新；模型的 LoRA 名字和强度（串联、只接 MODEL、强度 0）、设置来源、加载后的状态；各种输入组合和 `MONOLOAD_DISABLE` 不报错 |
| `tests/test_messages.py` | 消息和翻译，不需要模型文件：消息表每一项都有英文和中文、格式字段一致；`MONOLOAD_LANG` 的解析；`zh` 时预算报错、OOM 报错、不支持 LoRA 的报错、节点的报错和日志、Info 的文字都是中文；代码里（消息表以外）没有中文；`locales/en`、`locales/zh` 的 `nodeDefs.json` 覆盖每个节点、输入（名字、提示）、下拉选项（键 = 存储的英文值）、输出 |
| `tests/test_vae_node.py` | Monoload VAE Settings 节点（§4.1），不需要模型文件：注册表和接口；按 ComfyUI 的方式调用节点；预算（下拉框 + `budget_gib`）/ 方案 / 条带高度 / 模式逐项判断「节点 > 环境变量 > 默认值」；副本共享权重和 patcher、输入的 VAE 不变、ComfyUI 的模型管理里只有一个已加载模型、串联、encode 一致；预算报错、强制方案和高度、只用第二层、原生；解码后（包括报错后）全局设置复原；多个副本交替解码互不干扰；日志写明来源；包装没装上时副本走原生；总开关和全局默认下的节点行为在 `test_master_switch.py`。ComfyUI 加载器注册节点的检查在 `test_entry.py`（各种开关组合） |
| `tests/test_vae_ldm.py` | VAE 第一层的 LDM decoder（§12.8），不需要模型文件：GroupNorm 统计量（Moments 对 fp64，含均值远大于标准差；冻结统计量的 GroupNorm 对 `F.group_norm`；实例替换走 weight_function、退出复原）；识别（11 种不认的结构走第二层、与原生一致）；整个 decoder（SDXL 式 / Flux 式，四种方案，不同条带高度、奇数 / 很小的 latent、batch 2、宽组、很小的工作区）与原生整图解码比；bf16 对 fp32 真值与原生同一水平；自检抓住注入的错误（条带局部统计量、丢一条带、halo 少一行）；全尺寸 SDXL 4K 的计划；OOM；开关；分配器模拟 |
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
* VAE 第一层（`tests/test_vae_stripe.py`，74 项，0 失败）：区间倒推与暴力展开 993 条条带全部一致；5 种单元在每个切片上的精确行与整图一致（≤ 3.6e-7）、halo 不多不少；整个 Wan decoder 在条带高度 1 / 7 / 40 / 整图 / 按预算 / 默认策略、奇数和很小的 latent、batch 2、条带内再分块下，与原生整图解码的 raw 差 ≤ 2.1e-6（fp32），条带边界附近不比其他区域差；halo 少算一行时自检两种方式都能抓到并回退第二层；SDXL / Flux 继续走第二层；默认策略选出「128 行条带的估算」之内最高的条带（320 行的图 3 条 107 行，再高一档就超过目标），`MONOLOAD_VAE_STRIPE_ROWS` / `MONOLOAD_VAE_BUDGET` 能覆盖它（设了预算而第二层放得下时走第二层）；估算随条带高度单调、最大的条带先跑；自检后清空 1 次缓存、batch 2 在前缀和条带之间清空 2 次；单帧 Conv3d 改走 conv2d 时与模块原样输出一致（≤ 4.8e-7），T=3 和已被别人替换过 `_conv_forward` 的模块不改走；arena 每次解码只分配一次、CPU 和非默认分配器配置下不用；分配器模拟器复现 9 个真机读数（4e54d20、725a010、85a5c6f 和工作区 128 MiB 的默认计划，误差 ≤ 0.03 GiB），本版 9 个计划（1344 … 8K、bf16 / fp32）的模拟 reserved ≤ 估算。
* VAE 第一层的 LDM decoder（`tests/test_vae_ldm.py`，100 项，0 失败）：统计量对 fp64 相对误差 ≤ 1.9e-7（均值 1000、标准差 0.01 时也是，朴素的 E[x²]−E[x]² 在这里差 1.9e3 倍）；冻结统计量的 GroupNorm 与 `F.group_norm` 差 ≤ 7.2e-7（fp32）/ 0（bf16），行切片上也一样；11 种不认的结构都走第二层并与原生一致；SDXL 式 / Flux 式整个 decoder 在四种方案、条带 1 / 7 / 40 / 默认、奇数和很小的 latent、batch 2、ch 64、16 KiB 工作区下与原生整图解码的 raw 差 ≤ 4.6e-6（fp32），条带边界附近不比其他区域差；bf16 对 fp32 真值的 RMSE 与原生相同；自检误差 9e-7，三种注入的错误（条带局部统计量、丢一条带、halo 少一行）都被抓到并回退第二层；全尺寸 SDXL 4K 计划各方案的峰值与重算顺序；OOM 缩条带、到下限报错、不退回 tiled / 第二层；`MONOLOAD_VAE_GN_SCHEME`（强制与默认 B）；预算内最快（第二层放得下就选第二层；否则预测最快的方案，一条带时不跑统计遍、取默认方案；选中的方案自检失败时选下一个；强制高度 / 方案 / 第二层优先于预算；放不下时报错并列出各需要多少；全尺寸 SDXL 三档 × 20 / 3 / 1.5 GiB 选中 README §13 表里的配置）；估算收紧（`saves_fit` 的单元检查、4K 默认计划的存档有保证、largest 不是存档）；很高的 B 条带模拟 reserved ≤ 估算；耗时模型对 4K D 309 行 / 64 MiB 的实测 63.2 s 误差 ≤ 8%；模拟器复现 5 个第二层真机读数、5 个第一层计划 reserved ≤ 估算。「预算内最快」之后：`test_vae_stripe.py` 74 项、`test_vae.py` 131 项、入口测试 8 种开关组合（含 `MONOLOAD_VAE_GN_SCHEME=D` 和预算 + 强制高度）、`test_dtype_paths.py`（198 / 108 项）全过；`tests/alloc_sim.py` 的 51 个真机读数照旧复现（≤ 0.02 GiB）。
* 总开关（`tests/test_master_switch.py`，18 项，0 失败）：`MONOLOAD=0`、没有节点时，LoRA 模型的整体加载 / lowvram 加载 / Hook LoRA / 撤掉 hook 四种输出与原版 ComfyUI（钩子卸掉）`torch.equal`，备份数相同（2 对 2），撤掉后权重逐位还原；VAE 解码 `torch.equal`，不打 INFO 日志；释放不运行；开销 `patch_weight_to_device` 每次 +0.2 µs、`VAE.decode` 包装每次 3.4 µs；对照组（开启）无备份，与原生差 1.2e-4（快速路径的舍入）。
* Monoload VAE Settings 节点（`tests/test_vae_node.py`，22 项，0 失败）：逐项优先级（四项各自「节点 > 环境变量 > 默认值」）；副本与原 VAE 共享模型和 patcher、原 VAE 不受影响（预算 1024 GiB 的副本走第二层，原 VAE 紧接着走第一层方案 B）、模型管理里只有一个已加载模型、串联、encode 一致；预算报错并写明各需要多少、报错后全局设置复原；强制 D + 24 行、只用第二层、原生；全局 `MONOLOAD_DISABLE_VAE_STRIPE=1` 时节点的 `auto` 仍走第一层；三个副本交替解码各用各的设置；包装没装上时（ComfyUI 接口不符）副本走原生、不报预算错。`test_entry.py` 各种开关组合下 ComfyUI 的加载器都注册了这个节点（`MONOLOAD_DISABLE=1` 也注册）。
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
| `--modes` | 默认 `native,monoload,native2`，每个分辨率按这个顺序跑：`native` 原生 `VAE.decode`（Monoload 的 VAE 部分卸下）；`monoload` 插件的实际行为（认得的结构走第一层，其余走第二层）；`monoload-l2` 强制只用第二层（逐算子分块），便于对比；`monoload-r<N>` 第一层、条带核心高度强制为 N 行；`native2` 原生再跑一次，作为 GPU 自身不确定性的基线 |
| `--stripe-rows` | 第一层条带高度扫参，例如 `32,64,128,256`：每个值加一个 `monoload-r<N>` 模式，各自报告峰值、耗时、重算比例和精度 |
| `--warm N` | 冷启动（该模式、该分辨率的第一次）之后再跑 N 次热启动，取中位数，默认 3 |
| `--workspace` | 这次运行用的 Monoload 预算（如 `512M`），默认取 `MONOLOAD_VAE_WORKSPACE` 或 1G |
| `--sample-ms` | GTT / VRAM / cgroup 采样间隔，默认 10 ms |
| `--boundary-rows` | 块边界上下各多少行算「边界附近」，默认 4 |
| `--profile` | 先对每个分辨率（或 `--profile-res` 指定的）跑一次 torch.profiler + 分配器历史，见下 |
| `--profile-modes` | 要 profile 的模式，默认 `native`；可以写 `monoload`（第一层或第二层，看峰值时刻剩下的是什么）、`monoload-l2`、`monoload-r<N>` |
| `--profile-cpp` | 分配器历史里带 C++ 栈（能直接看到 `slow_conv2d` / `im2col` 这类 ATen 函数），符号化可能要几分钟 |
| `--profile-only` | 只做 profile，不跑基准 |
| `--fp32-ref` | fp32 参照：同一个 VAE 用同一个加载方式再加载一份 fp32 的（把 `vae_dtype` 强制为 fp32），每个分辨率在 bf16 各模式之后解码一次作为「真值」，输出 `native vs fp32`、`monoload vs fp32` 两组与上面相同的误差指标（块边界行用同一次 monoload 运行的） |
| `--fp32-impl` | fp32 参照怎么解码：`both`（默认：原生 fp32 和分块 fp32 都跑；原生 fp32 显存约为 bf16 的两倍，4K 约 90 GiB，OOM 时跳过，改用分块的结果作参照；两者都成功时另外输出两者之差，作为分块在 fp32 下是否精确的旁证）、`native`（原生，OOM 时退到分块）、`monoload`（只用分块） |
| `--outliers N` | 列出 monoload 与 native 差得最多的 N 个像素（默认 10）：batch、行、列、通道，三方（fp32、native、monoload）在这些点上的像素值和 raw 值，各自离 fp32 多远，离最近的块边界几行 |
| `--outlier-threshold` | 统计 \|monoload − native\| 超过这个值（默认 0.01）的所有像素值：其中 monoload 和 native 各有多少个更接近 fp32，以及两者在这些点上对 fp32 的平均误差 |
| `--json F` | 全部结果另存一份 JSON |

**原生 OOM 的处理：** 原生解码真 OOM 时会退回 tiled，脚本把 tiled 路径换成「立即中止」，这一行标为 `OOM(tiled)`，不当作原生结果，也不参与精度比较；直接抛出的 OOM 标为 `OOM`；monoload 在最小预算下仍 OOM 时（`MonoloadVAEOOMError`）同样标为 `OOM`。脚本不会因此退出，后面的模式和分辨率照常跑。

**输出：**

* monoload 各模式的每次运行下面多一行，写明用了哪一层：第一层是条带数 × 核心行数、重算比例（卷积算量相对整图解码）、存档大小、预算、工作区、OOM 重试次数、估算（前缀 / 条带 / 常驻三部分），后面是实测的 alloc / reserved 增量，便于对比估算和实测；第二层是没用第一层的原因和分块统计。汇总表最后一列 `layer` 同样标出 L1（条带数 × 行数、重算比例、存档）或 L2。第一层的「块边界附近」统计按条带边界算。
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

**第二阶段（第一层，Qwen 条带解码）的真机测试：** 每条命令前都先 `/free`（同上）。

```bash
# A. Qwen 三档：原生 / 插件实际行为（第一层）/ 只用第二层，带 fp32 参照
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae qwen_image_vae.safetensors --modes native,monoload,monoload-l2 --warm 2 --fp32-ref \
  --json /opt/ComfyUI/output/bench_vae_l1_qwen.json 2>&1 | tee bench_vae_l1_qwen.txt
# B. 4K 条带高度扫参
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae qwen_image_vae.safetensors --res 3840x2160 --modes native,monoload --stripe-rows 32,64,128,256,512 --warm 2 \
  --json /opt/ComfyUI/output/bench_vae_l1_sweep.json 2>&1 | tee bench_vae_l1_sweep.txt
# C. 4K 第一层的峰值构成（profile），只做 profile
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae qwen_image_vae.safetensors --res 3840x2160 --profile-only --profile-modes monoload 2>&1 | tee bench_vae_l1_profile.txt
# D. SDXL 回归：仍走第二层、行为不变
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --checkpoint waiIllustriousSDXL_v170.safetensors --res 3840x2160 --modes native,monoload --warm 2 \
  --json /opt/ComfyUI/output/bench_vae_l1_sdxl.json 2>&1 | tee bench_vae_l1_sdxl.txt
```

看什么：A 的 monoload 行写的是 `layer 1`，4K 的 alloc / GTT 增量在 2–4 GiB 量级并且不超过估算，`monoload vs fp32` 与 `native vs fp32` 相当，条带边界附近的误差不比其余行大，`monoload-l2` 与第一阶段的数字一致；第一次解码前日志里有 `VAE layer 1 (Wan 2.1 stripes) self-test passed`。B 看峰值、耗时、重算比例随条带高度的变化（选默认预算的依据）。C 看峰值时刻剩下的分配是什么（前缀的注意力、条带内的卷积、存档）。D 的 monoload 行写的是 `layer 2 (first-stage model is AutoencoderKL, not ...)`，数字与第一阶段一致。

上面 A–D 已在 4e54d20 上测完（§10.2）。

**第二阶段调整后的复测**（默认策略、内存模型、缓存处理、单帧 Conv3d 改走 conv2d；每条命令前都先 `/free`）：

```bash
# E. Qwen 三档：原生 / 第一层（新默认策略）/ 只用第二层，带 fp32 参照
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae qwen_image_vae.safetensors --modes native,monoload,monoload-l2 --warm 2 --fp32-ref \
  --json /opt/ComfyUI/output/bench_vae_l1b_qwen.json 2>&1 | tee bench_vae_l1b_qwen.txt
# F. 4K 条带高度扫参复核（monoload = 默认策略，155 行就是它在 4K 选的高度）
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae qwen_image_vae.safetensors --res 3840x2160 --modes native,monoload --stripe-rows 32,64,128,155,256,512 --warm 2 \
  --json /opt/ComfyUI/output/bench_vae_l1b_sweep.json 2>&1 | tee bench_vae_l1b_sweep.txt
# G. 4K 第一层的 profile：aten::fill_ 次数、峰值构成
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae qwen_image_vae.safetensors --res 3840x2160 --profile-only --profile-modes monoload 2>&1 | tee bench_vae_l1b_profile.txt
```

看什么：E 的 monoload 行是 `layer 1`，计划为 1344 → 6 条 128 行、2688 → 12 条 128 行、4K → 14 条 155 行，行尾写着 `default: peak of 128-row stripes`、`N Conv3d calls as conv2d`（N > 0）、`cache emptied 1x`；**每一行（冷启动也算）的 reserved / GTT 增量都不超过估算**（1344 估 0.74 GiB、2688 估 0.95、4K 估 1.25），4K 的 reserved 约 1.07；耗时与第一版（0.67 / 3.16 / 8.12 s）相比不明显变慢；`monoload vs fp32` 仍与 `native vs fp32` 相当。F 看 reserved 是否不再随条带高度叠加（512 行时第一版是 2.66，现在应接近 alloc 1.70 加余量），以及估算是否覆盖每个高度的 reserved。G 里 `aten::fill_` 应从 15.9 万次降到几百次。

上面 E–G 已在 725a010 上测完（§10.2）。

**第二阶段第三轮复测**（arena、分块卷积先分配输出、估算 = arena + largest；每条命令前都先 `/free`）：

```bash
# H. Qwen 三档：原生 / 第一层（默认）/ 只用第二层；--fp32-ref 可选（加上它多跑三档 fp32，4K 的原生 fp32 约 55 s）
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae qwen_image_vae.safetensors --modes native,monoload,monoload-l2 --warm 2 --fp32-ref \
  --json /opt/ComfyUI/output/bench_vae_l1c_qwen.json 2>&1 | tee bench_vae_l1c_qwen.txt
# I. 4K 条带高度扫参（monoload = 默认，144 行就是它在 4K 选的高度）
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae qwen_image_vae.safetensors --res 3840x2160 --modes native,monoload --stripe-rows 32,64,128,144,256,512 --warm 2 \
  --json /opt/ComfyUI/output/bench_vae_l1c_sweep.json 2>&1 | tee bench_vae_l1c_sweep.txt
```

看什么：

* H 的 monoload 行：计划 1344 → 6 条 128 行、2688 → 12 条 128 行、4K → 15 条 144 行（fp32 4K → 14 条 155 行），行尾写着 `arena ...`（不再有 `cache emptied`）。
* **每一行（冷启动也算）的 reserved / GTT 增量 ≤ 估算。** 按模拟，bf16 的 reserved 约 0.66 / 0.86 / 1.13 GiB（估算 1.05 / 1.25 / 1.52）；fp32 约 0.80 / 1.16 / 1.70 GiB（估算 1.19 / 1.55 / 2.27）。
* alloc 列现在约等于 arena（arena 这个块本身算作一次分配），张量本身的峰值要加 `--no-arena` 才看得到（那时 reserved 不代表插件的行为）。
* 耗时与 725a010 相同（4K 约 8.2 s）；精度数字不变。
* I：r32 / r64 / r128 / r144 的 reserved 都约 1.13 GiB，r256 约 1.35，r512 约 1.82；每个高度 reserved ≤ 估算。

上面 H、I 已在 85a5c6f 上测完，与模拟一致（§10.2），第二阶段验收通过。

**workspace 实验**（`MONOLOAD_VAE_WORKSPACE` 调小对峰值和速度的影响；已在 5d668b6 上测完，结果和决定见 §10.2：第一层默认改成 128 MiB，第二层保持 1 GiB）。bench 的模式名后面加 `-w<MiB>` 就是只对这个模式把 workspace 设成那么大（`monoload-w128`、`monoload-l2-w256`；不加就是默认的 1 GiB，第一层实际用 384 MiB）。每条命令前都先 `/free`：

```bash
# J. Qwen 三档：第一层 384（当时的默认）/ 256 / 128，第二层 1G（默认）/ 384 / 128
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae qwen_image_vae.safetensors --modes native,monoload,monoload-w256,monoload-w128,monoload-l2,monoload-l2-w384,monoload-l2-w128 --warm 2 \
  --json /opt/ComfyUI/output/bench_vae_ws_qwen.json 2>&1 | tee bench_vae_ws_qwen.txt
# K. SDXL（只走第二层）：1G（默认）/ 512 / 256 / 128
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --checkpoint waiIllustriousSDXL_v170.safetensors --modes native,monoload,monoload-w512,monoload-w256,monoload-w128 --warm 2 \
  --json /opt/ComfyUI/output/bench_vae_ws_sdxl.json 2>&1 | tee bench_vae_ws_sdxl.txt
```

看什么（都看汇总表 `summary: time and memory` 和 `summary: accuracy`）：

* J 的第一层（`monoload`、`-w256`、`-w128`）：`GTT Δ` / `resv Δ` 和热启动耗时 `warm s`。按模拟，reserved 是 0.66 / 0.51 / 0.36 GiB（1344）、0.87 / 0.71 / 0.56（2688）、1.13 / 1.00 / 0.87（4K）；要量的是耗时涨了多少（384 MiB 时是 0.67 / 3.21 / 8.24 s）。每行 reserved 仍应 ≤ `estimate` 列。
* J 的第二层（`monoload-l2`、`-w384`、`-w128`）：模拟显示 Qwen 第二层的 reserved 几乎不随 workspace 变（4K 约 9.6 GiB，峰值是整图激活），主要看耗时有没有变差，用来判断第二层要不要跟着调。
* K：SDXL 第二层 4K 上次是 GTT 14.86 GiB、9.88 s。看 workspace 从 1G 降到 512 / 256 / 128 时 GTT 降多少、耗时涨多少（SDXL 不在模拟器里，只能实测）。
* 两条命令的 `vs native` 精度应与默认 workspace 时同一水平（PSNR 约 60 dB），workspace 只改变分块大小。

**第三阶段：SDXL / Flux（LDM decoder）走第一层**（GroupNorm 整图统计量跨条带调度，DESIGN.md §9.14）。每条命令前都先 `/free`。bench 的模式名加 `-g<S>` 选 GroupNorm 方案（`monoload-gD`），`--gn-schemes ADBC` 一条命令加上 `monoload-gA … monoload-gC`，与 `--stripe-rows` 同时给时是「方案 × 高度」的全部组合（`monoload-r32-gD`）；`--fp32-chunked l2` 让分块的 fp32 参照只用第二层（与条带、统计量都无关，算量 1 倍）。方案 A 的卷积算量约 12 倍，4K 一次可能要一两分钟，所以热启动只跑 1 次。

```bash
# L. SDXL 三档：原生 / 第一层（默认 = 方案 A）/ 只用第二层，带 fp32 参照
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --checkpoint waiIllustriousSDXL_v170.safetensors --modes native,monoload,monoload-l2 --warm 1 --fp32-ref --fp32-chunked l2 \
  --json /opt/ComfyUI/output/bench_vae_l3_sdxl.json 2>&1 | tee bench_vae_l3_sdxl.txt
# M. Flux ae 三档，同上
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae ae.safetensors --modes native,monoload,monoload-l2 --warm 1 --fp32-ref --fp32-chunked l2 \
  --json /opt/ComfyUI/output/bench_vae_l3_flux.json 2>&1 | tee bench_vae_l3_flux.txt
# N. SDXL 4K：四种 GroupNorm 方案（默认条带策略）
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --checkpoint waiIllustriousSDXL_v170.safetensors --res 3840x2160 --modes native --gn-schemes ADBC --warm 1 \
  --json /opt/ComfyUI/output/bench_vae_l3_schemes.json 2>&1 | tee bench_vae_l3_schemes.txt
# O. SDXL 4K：方案 A、D × 条带高度 32 / 64 / 96（看矮条带能不能用更少的重算达到同样的峰值）
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --checkpoint waiIllustriousSDXL_v170.safetensors --res 3840x2160 --modes native --gn-schemes AD --stripe-rows 32,64,96 --warm 1 \
  --json /opt/ComfyUI/output/bench_vae_l3_rows.json 2>&1 | tee bench_vae_l3_rows.txt
# P. Qwen 回归（默认配置）：拆分引擎 / 适配器之后数字不变
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae qwen_image_vae.safetensors --modes native,monoload --warm 2 \
  --json /opt/ComfyUI/output/bench_vae_l3_qwen.json 2>&1 | tee bench_vae_l3_qwen.txt
```

**模拟器对每一行的预测**（`tests/alloc_sim.py`，GiB；第一层的 reserved 就是 arena；「估算」是交给 `load_models_gpu` 的数，实测 reserved / GTT 应当不超过它；耗时没法模拟，给的是卷积算量相对整图解码的倍数）：

| 命令 | 行 | 1344×768 | 2688×1536 | 3840×2160 |
|---|---|---|---|---|
| L（SDXL） | `native` | 实测（§10.1）GTT 8.36 | 42.84 | 52.48（先 OOM 再重试） |
| | `monoload`（A） | 6 条 128 行，reserved 0.47，估算 0.61，12×| 12 条 128 行，0.79，0.99，12× | 17 条 128 行，1.10，1.48，12× |
| | `monoload-l2` | 3.71 | 9.21 | 14.97 |
| | `monoload-l2-fp32`（参照） | 5.36 | 14.92 | 31.67 |
| M（Flux） | `monoload`（A） | 0.48，估算 0.62 | 0.79，0.99 | 1.10，1.49 |
| | `monoload-l2` | 3.71 | 9.23 | 15.00 |
| | `monoload-l2-fp32` | 5.38 | 14.94 | 31.72 |

| 命令 N / O（SDXL 4K） | 计划 | reserved | 估算 | 重算 |
|---|---|---|---|---|
| `monoload-gA`（= 默认） | 17 条 128 行，不存 | 1.10 | 1.48 | 12.1× |
| `monoload-gD` | 17 条 128 行，存 H/4（0.49 GiB） | 1.52 | 2.03 | 8.1× |
| `monoload-gB` | 17 条 128 行，存 H/4 + H/2（分开） | 2.17 | 3.18 | 5.3× |
| `monoload-gC` | 13 条 167 行，存 5 个（池） | 4.70 | 8.67 | 5.7× |
| `monoload-r32-gA` / `-r64-gA` / `-r96-gA` | 68×32 / 34×64 / 23×96 | 1.06 / 1.06 / 1.06 | 1.44 | 13.1× / 12.5× / 12.3× |
| `monoload-r32-gD` / `-r64-gD` / `-r96-gD` | 同上 | 1.09 / 1.22 / 1.36 | 1.59 / 1.72 / 1.87 | 10.3× / 9.0× / 8.5× |

P（Qwen）：计划和数字应当与 workspace 实验之后的默认相同：6 条 128 行 / 12 条 128 行 / 14 条 155 行，GTT 0.36 / 0.56 / 0.87 GiB（模拟同），热启动 0.71 / 3.45 / 8.44 s，估算 0.50 / 0.71 / 1.16。

看什么：

* L / M 的 `monoload` 行写的是 `layer 1 (LDM stripes, GroupNorm scheme A): ... 19 GroupNorm statistics passes ...`；第一次解码前日志有 `VAE layer 1 (LDM stripes, GroupNorm scheme A) self-test passed`。**每一行（冷启动也算）reserved / GTT ≤ 估算**，并且接近上表的预测。
* 精度：`monoload vs fp32` 与 `native vs fp32` 相当（RMSE / PSNR / p99），条带边界附近不比其余行差，没有整体亮度或通道均值的偏移（统计量错了会表现为这种偏移）。fp32 参照来自第二层，与条带和统计量无关。
* 耗时：第一层（方案 A，约 12 倍算量）比原生慢多少——这是峰值优先的代价，决定默认方案的主要依据之一。
* N / O：每个方案、每个高度的峰值和耗时。模拟显示 4K 的峰值下限约 1.05 GiB（由整图前缀决定）：A 用 32–96 行条带、或 D 用 32 行条带都能到这里，D 的算量更少（10.3× 对 12.3–13.1×）。
* 把输出和 JSON 发给我。

上面 L–P 已在 7059bdd 上测完，与模拟一致（§10.3）。之后的决定：LDM 默认方案改成 B，设了 `MONOLOAD_VAE_BUDGET` 时改成「预算内最快」（§13 的例子，DESIGN.md §9.14.10）。

**默认方案 B 与「按预算选」**。bench 的模式名加 `-b<GiB>` 就是只对这个模式设 `MONOLOAD_VAE_BUDGET`（`monoload-b3`、`monoload-b1.5`），`--budgets 20,3,1.5,1` 一条命令加上这些模式；`-g<S>` 仍是强制方案（`monoload-gA-b2`：强制 A、预算 2G）。预算放不下时这一行的状态是 `over budget`，后面附上报错（写明各需要多少），不算失败。每条命令前都先 `/free`。

```bash
# Q. SDXL / Flux 三档：原生 / 默认（方案 B）
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --checkpoint waiIllustriousSDXL_v170.safetensors --modes native,monoload --warm 1 \
  --json /opt/ComfyUI/output/bench_vae_b_sdxl.json 2>&1 | tee bench_vae_b_sdxl.txt
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae ae.safetensors --modes native,monoload --warm 1 \
  --json /opt/ComfyUI/output/bench_vae_b_flux.json 2>&1 | tee bench_vae_b_flux.txt
# R. SDXL 三档 × 预算 20G / 3G / 1.5G / 1G（按预算选）
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --checkpoint waiIllustriousSDXL_v170.safetensors --modes native --budgets 20,3,1.5,1 --warm 1 \
  --json /opt/ComfyUI/output/bench_vae_budget_sdxl.json 2>&1 | tee bench_vae_budget_sdxl.txt
# S. Qwen 三档 × 预算 20G / 3G / 1.5G（Wan：第二层对第一层，第一层取预算内最高的条带）
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --vae qwen_image_vae.safetensors --modes native --budgets 20,3,1.5 --warm 1 \
  --json /opt/ComfyUI/output/bench_vae_budget_qwen.json 2>&1 | tee bench_vae_budget_qwen.txt
```

**Q 的逐行预测**（GiB；reserved 是模拟值，实测 GTT 应当接近它、不超过估算；耗时是耗时模型的预测，DESIGN.md §9.14.10）：

| 行 | 1344×768 | 2688×1536 | 3840×2160 |
|---|---|---|---|
| `monoload`（B，SDXL） | 6 条 128 行，reserved 0.62，估算 0.76，5.1×，约 5.1 s | 12 条 128 行，1.33，1.83，5.1×，约 20.7 s | 17 条 128 行，2.17，3.17，5.3×，约 42.4 s（7059bdd 实测 2.17 / 42.1） |
| `monoload`（B，Flux） | 0.62，0.76，约 5.1 s | 1.33，1.84，约 20.7 s | 2.18，3.18，约 42.4 s |

日志里是 `layer 1 (LDM stripes, GroupNorm scheme B): ... 19 statistics passes, saves 506 MiB+1012 MiB`（4K），启动日志写 `scheme B (default)`。

**R 的逐行预测**（SDXL；选中的配置，估算 / 模拟 reserved GiB，耗时：第二层是实测，第一层是模型预测）：

| 行 | 1344×768 | 2688×1536 | 3840×2160 |
|---|---|---|---|
| `monoload-b20` | 第二层，3.98 / 3.71，0.88 s | 第二层，9.92 / 9.21，4.0 s | 第二层，17.91 / 14.97，9.9 s |
| `monoload-b3` | 第一层 1 条 768 行（不跑统计遍，算量 1×），2.48 / 1.97，约 1.3 s | B，4 条 384 行，2.73 / 2.20，约 19.8 s | D，7 条 309 行（工作区 64 MiB），2.94 / 2.33，约 53.6 s |
| `monoload-b1.5` | C，2 条 384 行，1.16 / 0.90，约 3.3 s | D，8 条 192 行，1.45 / 1.15，约 27.3 s | A，15 条 144 行，1.48 / 1.10，约 73.4 s |
| `monoload-b1` | B，3 条 256 行，0.95 / 0.76，约 5.0 s | D，22 条 70 行，1.00 / 0.74，约 30.3 s | `over budget`：第二层要 17.91，方案 A 最少 1.38，D 1.53，B 3.01，C 8.02 |

**S 的逐行预测**（Qwen；第一层没有耗时模型，参照以前的扫参）：

| 行 | 1344×768 | 2688×1536 | 3840×2160 |
|---|---|---|---|
| `monoload-b20` | 第二层，估算 3.49 / 模拟 2.10，约 0.66 s | 第二层，7.95 / 5.05，约 2.9 s | 第二层，13.96 / 9.59，约 6.9 s |
| `monoload-b3` | 第一层 1 条 768 行，1.70 / 1.31（4e54d20 的 1 条 768 行 0.67 s） | 2 条 768 行，2.72 / 1.96 | 4 条 540 行，2.87 / 2.09（r512 约 7.8 s） |
| `monoload-b1.5` | 1 条 768 行（工作区 192 MiB），1.45 / 1.06 | 4 条 384 行，1.48 / 1.08 | 9 条 240 行，1.49 / 1.12（r256 约 8.0 s） |

看什么：

* 每一行的日志先有一行 `VAE budget ... (from ...) -> ...: <为什么>; others: <其他候选各要多少>`，选中的配置与上表一致；**reserved / GTT ≤ 估算 ≤ 预算**。
* 耗时：第一层各行的实测与「约 x s」对比，看耗时模型在 B / C / D 和 1344 / 2688 上准不准（它只用 4K 和 A 的三档拟合）。如果某个预算下实测比没被选中的候选慢很多，就是模型排错了。
* 精度：所有行 `vs native` 与以前同一水平（PSNR 约 60 dB）。1344 的 `-b3` 只有一条带、不跑统计遍，就是整图解码。
* 把输出和 JSON 发给我。

上面 Q–S 已在 01377c4 上测完，与预测一致（§10.4）。之后收紧了有存档时的估算、耗时模型加了工作区（DESIGN.md §9.14.11），R 表里 3G / 1.5G / 1G 的选择随之变了，以下面 T 的预测为准。

**收紧估算之后的按预算选**（每条命令前先 `/free`）：

```bash
# T. SDXL 三档 × 预算 3G / 1.5G
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/bench_vae.py \
  --checkpoint waiIllustriousSDXL_v170.safetensors --modes native --budgets 3,1.5 --warm 1 \
  --json /opt/ComfyUI/output/bench_vae_budget2_sdxl.json 2>&1 | tee bench_vae_budget2_sdxl.txt
```

**T 的逐行预测**（选中的配置；估算 / 模拟 reserved GiB；耗时是新耗时模型的预测）：

| 行 | 1344×768 | 2688×1536 | 3840×2160 |
|---|---|---|---|
| `monoload-b3` | 第一层整图 1 条 768 行（不跑统计遍，工作区 384 MiB），2.48 / 1.97，约 0.8 s（01377c4 实测 1.00 s） | C，4 条 384 行（128 MiB），2.99 / 2.73，约 12.0 s | **B**，12 条 180 行（128 MiB），2.96 / 2.43，约 41.1 s（以前选 D：2.33 GiB / 63.2 s） |
| `monoload-b1.5` | C，2 条 384 行（192 MiB），1.25 / 1.05，约 2.8 s（以前 1.16 / 0.90，实测 2.89 s） | B，16 条 96 行（128 MiB），1.48 / 1.22，约 20.6 s（以前 D 8×192：29.0 s） | A，17 条 128 行（128 MiB），1.48 / 1.10，约 74.1 s（以前 A 15×144 / 64 MiB：77.9 s） |

看什么：

* 日志里选中的配置与上表一致；GTT ≤ 估算 ≤ 预算。4K `-b3` 选 B 的 180 行条带（GTT 约 2.43 GiB，比默认 128 行的 2.17 多 0.26 GiB），耗时模型预测比默认快约 1 s。1344 `-b1.5` 的 C 2 条 384 行和以前是同一个计划，但只有 2 条带时 arena 现在多留余量（下面），GTT 预计 1.05（以前实测 0.90）。
* 耗时与「约 x s」的差：新模型在 22 个读数上最大误差 5.3%（整图一条带 −17%）。
* 精度照旧 60–62 dB。
* 把输出和 JSON 发给我。

**Monoload VAE Settings 节点**（§4.1）：真实的 SDXL VAE，经过节点设置预算 3G 后解码 4K latent，再用原来的 VAE 解码同一个 latent（先 `/free`）：

```bash
# U. 节点：SDXL 4K，节点预算 3G 的副本 / 原 VAE / 副本再一次
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/check_vae_node.py \
  --checkpoint waiIllustriousSDXL_v170.safetensors --res 3840x2160 --budget 3 2>&1 | tee check_vae_node.txt
```

预期：

| 行 | 选中的配置 | 设置来源 | GTT / 估算 GiB | 耗时 |
|---|---|---|---|---|
| `node copy` | 第一层，方案 B，12 条 180 行，工作区 128 MiB | budget 3.00 GiB [node]，其余 [default] | 约 2.43 / 2.96 | 约 41 s（第一次多一次 B 的自检） |
| `original` | 第一层，方案 B，17 条 128 行，工作区 128 MiB | 全部 [default] | 约 2.17 / 2.68 | 约 42 s |
| `node copy again` | 同第一行 | 同第一行 | 同第一行 | 同第一行 |

* 开头一行写明副本共享权重（`True`）、原 VAE 不带设置（`True`）。
* 解码日志末尾有 `settings: budget 3.00 GiB (node), GroupNorm scheme chosen by the budget (default), ...`（副本）和 `settings: budget none (default), ...`（原 VAE）。
* 把输出发给我。

### 9.8 节点：Monoload LoRA Settings（`tests/check_lora_node.py`）

真实的 SDXL checkpoint（默认 `waiIllustriousSDXL_v170.safetensors`）+ 一个 LoRA（`--lora`，经 `LoraLoader`，强度 1.0）。先卸掉钩子用原版 ComfyUI 采样一次作参照，然后每一行用不同的节点设置采同一个种子（1024×1024，8 步，CFG 6）。不带 `--lora` 时列出 `models/loras` 和 `models/checkpoints` 后退出。

```bash
# V. LoRA 节点：路径、来源、每步耗时、与原生的差异、prompt 结束后的处理
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/check_lora_node.py --lora <models/loras 下的文件> 2>&1 | tee check_lora_node.txt
```

预期（每行打印实际路径、设置和来源、step1 / 每步耗时、GTT、与原生的差异、prompt 结束后释放还是保留）：

| 行 | 路径 | 与原生 | prompt 结束后 |
|---|---|---|---|
| `reference` | 原版 ComfyUI（钩子卸掉），有备份 | — | — |
| `no node` | Monoload 运行时合并，0 备份；设置全 `(default)` | 有小差异（fp16 舍入量级） | 释放 |
| `node enable` | 同上；mode `(node)` | 与 `no node` 逐位相同 | 释放 |
| `node exact` | Monoload，0 备份；merge exact `(node)` | **IDENTICAL**；每步比 fused 慢（约 1.3–1.7 倍，§10） | 释放 |
| `node native` | 原生：LoRA 合并进权重，备份数同 `reference` | **IDENTICAL**；每步同原生 | 保留 |
| `node keep` | Monoload，0 备份 | 同 `no node` | 保留 |
| `node exact before LoRA` | 节点在 checkpoint 和 `LoraLoader` 之间；Monoload，merge exact `(node)` | **IDENTICAL** | 释放 |
| `MONOLOAD=0, no node` | 原生，备份数同 `reference`；mode native `(env MONOLOAD=0)` | **IDENTICAL** | 保留 |
| `MONOLOAD=0, node enable` | Monoload，0 备份 | 与 `no node` 逐位相同 | 释放 |

* 最后一行打印 `'no node' == 'node enable' == 'MONOLOAD=0, node enable' bit for bit: True`。
* GTT：原生的行比 Monoload 的行多出备份的大小（§10 第一轮：13.9 对 8.6 GiB）。
* 把输出发给我。

### 9.9 网页界面手动检查

拉代码、重启容器后在浏览器里打开 ComfyUI（强制刷新一次，让前端扩展 `monoload_info.js`、`monoload_i18n.js` 生效）：

1. **节点能搜到**：双击画布打开搜索，输入 `Monoload`：应列出 Monoload LoRA Settings、Monoload VAE Settings、Monoload Info（中文界面下是「Monoload LoRA 设置」「Monoload VAE 设置」「Monoload 信息」）；节点库里有 `Monoload` 分类。
2. **中英切换**：设置 → Comfy → Locale 切到中文：节点标题、输入名（如「模式」「合并方式」「prompt 结束后」「预算 (GiB)」）、下拉显示（「跟随全局」「启用」「逐位一致」「不限」「自定义」……）、鼠标悬停的提示都是中文；切回 English 是英文（下拉显示 `default (follow global)` 等）。切换后**保存工作流**再看 JSON（或导出）：下拉的值仍是 `default` / `custom` 等英文。
3. **旧工作流**：打开以前存的、用到 Monoload VAE Settings 的工作流，不报错，各控件值对得上（新加的 `budget` 是 `default`）。
4. **Info 节点内容**：
   * 单独放一个 Monoload Info（什么都不接）→ 运行：节点框里出现文字，第一行 `Monoload 0.2.0 (commit …, branch dev)`，然后总开关、每项全局默认值和来源（compose 里设了的显示 `env 变量=值`，其余 `built-in`）。
   * 搭 `Load Checkpoint → Load LoRA → Monoload LoRA Settings → KSampler → VAE Decode`，VAE 经过 Monoload VAE Settings（预算 `custom` 3）；Info 的 `vae` 接 VAE Settings 的输出、`model` 接 LoRA Settings 的输出、`images` 接 VAE Decode 的输出 → 运行：VAE 段显示设置和来源、这次解码的层 / 方案 / 条带 / 工作区 / 估算 / 实测峰值（reserved、GTT）/ 耗时；MODEL 段显示 LoRA 文件名和强度、三项设置和来源、`now: loaded; Monoload runtime merge on … weights`（或原生时 `baked … backups`）。
   * 再运行一次（不改任何东西）：Info 仍然重新执行，「… s ago」变了。
   * `text` 输出接到一个显示文本的节点：内容与节点框里相同。
5. **`MONOLOAD_LANG=zh`**（compose 里加上并重启）：启动日志、解码日志、报错（例如把 VAE 节点的预算设成 `custom` 0.5 解码 4K）都是中文；Info 节点的文字是中文。

### 9.10 UI 实测之后的修正：复测（只测改动到的地方）

拉代码、重启容器、浏览器强制刷新一次。

1. **VAE 设置节点的控件**：新放一个 Monoload VAE Settings：控件顺序是 `budget`、`budget_gib`、`gn_scheme`、`stripe_rows`、`mode`；`budget_gib` 填 `0.25`，保存 / 导出工作流，JSON 里是 `0.25`（不是 0.3）。鼠标停在 `gn_scheme` 上：提示里有 A / D / B / C 各存什么、SDXL 4K 的内存和耗时、`default` 的含义（中文界面是中文）；下拉显示带简短说明（如「B（内置默认）」）。
2. **预算来源的措辞**：用你上次的工作流（VAE 设置节点 `custom` 0.3）解码，报错应是 `does not fit the peak budget 307 MiB (from the Monoload VAE Settings node; ...)`，最后的建议是「在节点上调大预算，或改成 default / unlimited」，**不再出现 `MONOLOAD_VAE_BUDGET`**。`MONOLOAD_LANG=zh` 时是「预算 307 MiB（来源：Monoload VAE 设置节点）」。把预算改成 `custom` 3 正常解码，日志那一行是 `VAE budget 3.00 GiB (from the Monoload VAE Settings node) -> ...`。
3. **泄漏警告**：同一个工作流（Checkpoint → LoraLoader → Monoload LoRA Settings（接 CLIP）→ CLIPTextEncode ×2 / KSampler → VAE 设置 custom 0.3 → VAEDecode → PreviewImage / Monoload Info），先跑一次（解码报错），紧接着再跑一次：日志里**不应再有** `Potential memory leak` / `WARNING, memory leak with model SDXLClipModel`。第一次的 release 日志末尾应有 `1 orphaned loaded model(s) re-pointed`。再把预算改成 3 正常跑两次，也没有这个警告。
4. **Info 节点**：接上 vae / model / images 运行：VAE 段和 MODEL 段之后，最后还有全局默认值表；VAE 段里有一行 `GroupNorm scheme B: keeps the H/4 and H/2 level outputs ...`。把 VAE 设置节点的 `mode` 改成 `native`：预算、方案、条带高度后面标 `(not used in native mode)`。LoRA 设置节点 `mode` 改成 `native`：merge 后面标同样的话。
5. **第一次解码的测量**：重启后第一次 1024×1024、预算 3G 的解码（一条带），Info 的 `measured peak reserved` 应在 arena（约 1.97 GiB）上下或略高，并写着 `(includes the first-use self-test)`；第二次约 1.97 GiB，不再有这句注明。

把第 2、3、5 步的日志 / Info 文字发给我。

### 9.11 第四阶段盘点：机器上的模型各用哪个 VAE（`tests/check_models.py`）

只读 safetensors 文件头（和 64 KiB 以下的小张量），在 meta 设备上识别，不占内存、不碰 GPU，可以在 ComfyUI 正常运行时跑；不用重启容器（只需要 `git pull` 后的脚本）。

```
# W. 每个模型文件：是什么模型、要哪种 latent / 哪个 VAE、Monoload 现在怎么解码
docker exec -w /opt/ComfyUI/custom_nodes/monoload comfyui python tests/check_models.py
```

预期（每个文件一段）：

* `== models/vae`：`ae.safetensors` → `VAE: AutoencodingEngine / decoder Decoder | latent 16 ch, latent_dim 2, x8`，1344×768 和 3840×2160 都是 `Monoload layer 1 (LDM stripes, GroupNorm scheme B)`；`qwen_image_vae.safetensors` → `WanVAE / decoder Decoder3d | latent 16 ch, latent_dim 3, x8`，两个图像尺寸 `layer 1 (Wan 2.1 stripes)`，81 帧视频 `native: multi-frame video latent ...`。
* `== models/checkpoints`：`waiIllustriousSDXL_v170` → `model: SDXL (latent format SDXL)`，内置 VAE `AutoencoderKL`，`layer 1`。
* `== models/diffusion_models`：每个文件一行 `model: <ComfyUI 的模型配置> | latent format <...> | VAE files that fit: <models/vae 里对得上的文件>`。预期 `krea2_turbo_bf16` → `Krea2`、`Wan21`、`qwen_image_vae.safetensors`；`smoothmixUltimateAnima_animaV20` → `Anima`、`Wan21`、`qwen_image_vae`；`wai_v17_fp8_test` → `SDXL`（`models/vae` 里没有对得上的文件，VAE 来自 checkpoint）。**要看的是 `novaAnimeAM_v5029B` 和两个 `luciddreamerZ_*`**：文件名看不出来，推测是 Anima（→ `qwen_image_vae`）和 Z-Image（`Lumina2` / `ZImage`、`Flux` 格式 → `ae`）。
* 出现 `failed: ...` 或 `model not detected` 的文件，把那几行发给我。

云端用的两个盘点脚本（不需要模型文件，锁定镜像里跑）：`tests/vae_inventory.py`（每种 VAE 的结构、现在的路、原生 / 第二层 / 第一层的模拟峰值）、`tests/probe_vae_gaps.py`（现有缺口的小解码、第二层在多帧视频上的精度）。结果见 DESIGN.md §9.16。

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

### 10.2 VAE 解码第二阶段：第一层条带解码（`bench_vae_l1_*`，2026-10，4e54d20）

设置同 §10.1（`--gpu-only --bf16-vae`，`cudnn.enabled = False`，split 注意力，随机 latent 种子 0，1 次冷启动 + 2 次热启动，先 `/free`）。这一轮测的是第一版的默认策略（3 GiB 预算内取最高条带）；之后按这些数据调整了默认策略和内存模型（见下面「调整」和 §12.7）。

* **显存（`qwen_image_vae`，GiB，热启动；原生 → 第一层 → 只用第二层）：**

| 输出 | 第一层计划 | alloc 增量 | reserved / GTT 增量 | 第一层估算 | 耗时 (s) |
|---|---|---|---|---|---|
| 1344×768 | 1 条 768 行 | 4.06 → 1.11 → 1.82 | 6.10 → 1.27 → 2.12 | 1.88 | 0.655 → 0.671 → 0.657 |
| 2688×1536 | 3 条 512 行，1.05× | 15.92 → 1.42 → 3.80 | 29.39 → 2.17 → 5.07 | 2.36 | 3.39 → 3.16 → 2.96 |
| 3840×2160 | 5 条 432 行，1.07× | 31.99 → 1.70 → 6.45 | 59.16 → 2.68 → 9.61 | 2.78 | 7.87 → 8.12 → 7.14 |

  4K 的 GTT 从 59.2 GiB 降到 2.7 GiB（第二层是 9.6），耗时与原生相同（第二层快 10%）。估算在热启动下都大于实测 reserved；冷启动的第一次解码 reserved 是 +3.04 GiB（4K），超过估算 2.78：自检的 fp32 副本和中间量留在了缓存里。fp32 解码（`monoload-fp32`，10 条 216 行）reserved 3.22，也超过了估算 2.93。
* **精度：** 第一层 vs fp32 与原生 vs fp32 相同（4K：RMSE 9.73e-4 对 9.71e-4，PSNR 60.2 对 60.3，p99 0.00287 对 0.00286）；fp32 下条带解码与整图只差 7e-6～8e-6（PSNR 136 dB）。|monoload − native| > 0.01 的像素里，monoload 更接近 fp32 的占一半多（4K 299 对 269），在这些点上 monoload 对 fp32 的平均误差更小（0.0069 对 0.0075）。**没有接缝：** 条带边界附近的 RMSE 与其余行相同（4K 9.69e-4 对 9.72e-4），最大误差也不在边界上。自检通过，误差 2.6e-6～3e-6，0.47 s。
* **4K 条带高度扫参（热启动；顺序 / 温度带来约 8% 的噪声）：**

| 核心行数 | 32 | 64 | 128 | 240 | 432 |
|---|---|---|---|---|---|
| 耗时 (s) | 11.7 | 9.8 | 8.8 | 9.2 | 8.1–8.9 |
| alloc / reserved 增量 (GiB) | 0.94 / 1.07 | 0.94 / 1.07 | 0.97 / 1.07 | 1.24 / 1.44 | 1.70 / 2.66 |
| 当时的估算 (GiB) | 1.77 | 1.77 | 1.77 | 1.99 | 2.78 |
| 重算 | 2.09× | 1.54× | 1.26× | 1.13× | 1.07× |

  约 128 行时峰值就到了下限（整图前缀决定），速度与 432 行相同；64 行起明显变慢。
* **峰值构成（profile，432 行）：** 最后一个 Resample 的上采样结果 630 MiB、它的卷积输出 315 MiB、im2col columns 384 MiB（工作区）、块拷贝 43 + 21 MiB、输入 160 MiB、存档 95 MiB、输出 95 MiB，合计 1.70 GiB。另有 `aten::fill_` 被调用 159308 次（GPU 时间不受影响）。
* **SDXL 回归：** 仍走第二层，4K alloc 11.17 GiB、GTT 14.86 GiB、9.88 s，与第一阶段一致。
* 日志里的解码秒数与 bench 一致（计时修正生效）。
* **结论：第二阶段验收通过**，并按数据做了以下调整（见 §12.7、DESIGN.md §9.13.4，真机复测命令见 9.7）：
  1. 默认策略改成「在不明显变慢的前提下尽量低峰值」：以 128 行条带的估算为目标，取不超过它的最高条带。4K 选 14 条 155 行，估算 1.25 GiB（第一版 2.78）。
  2. 自检结束后释放它的全部张量并清空缓存；前缀算完、条带开始前也清空一次缓存（第三轮去掉了，见下），先跑最大的条带。reserved 偏高的原因是前缀留在缓存里的块与条带要的块尺寸不同，条带阶段只能另外分配，两者叠加（432 行：前缀约 1.0 + 条带 1.6 ≈ 2.66）。
  3. `aten::fill_`：来自 PyTorch 的 SlowDilated3d 后端（CUDA/HIP 上关掉 cuDNN 后 Conv3d 走它），每次调用按输出通道逐个 `fill_(bias[n])`。单帧的 Conv3d 改用 conv2d（Slow2d：一次 `copy_` 设 bias，columns 和 GEMM 与 3D 路径相同）。
  4. 内存模型按 forward 代码重新数了一遍，与实测 alloc 逐项吻合（±5 MiB）；每个阶段加 `x/6 + 64 MiB` 的分配器余量（第三轮改成 arena，见下）。短条带时 4K 估算 1.25 GiB（第一版 1.77，实测 reserved 1.07）。

**725a010 复测（`bench_vae_l1b_*`，2026-10）：** 设置同上。

* **计划和功能：** 与预期一致：1344 → 6 条 128 行、2688 → 12 条 128 行、4K → 14 条 155 行；日志里有 `default`、`Conv3d as conv2d`（132 / 346 / 604 次）、`cache emptied 1x`。
* **显存（GTT 增量，GiB；原生 / 4e54d20 / 725a010）：** 1344 6.10 / 1.27 / 0.69，2688 29.4 / 2.17 / 1.02，4K 59.2 / 2.68 / 1.44。
* **速度（热启动，s）：** 第一层 0.67 / 3.24 / 8.25（原生 0.66 / 3.39 / 7.86）。4K 扫参 r32 10.65、r64 9.16、r128 8.38、r155 8.26、r256 8.00、r512 7.78，各高度都比 4e54d20 快 5–12%；第二层 4K 7.14 → 6.91、2688 2.96 → 2.85。`aten::fill_` 159308 → 9396 次（CUDA 合计约 57 ms）。
* **精度：** fp32 下第一层与整图差 7.3e-6 / 8.1e-6；第二层的精度数字与 4e54d20 逐位相同（conv2d 改道不改变结果）；对 fp32 真值 monoload 与原生相同（4K RMSE 9.73e-4 对 9.71e-4），离群点上更接近 fp32 的 monoload / native 为 17/12、207/139、302/231。
* **profile（4K 默认，alloc 1.04 GiB）：** columns 384 MiB、上采样结果 242、卷积输出 121、存档 95、输出 95，其余 63 + 43 + 21。
* **问题 1：估算不再是 reserved 的上界**（alloc 都在估算之内）：bf16 2688 估 0.95 / reserved 1.02，4K 1.25 / 1.44；fp32 1344 0.89 / 0.96，2688 1.27 / 1.41，4K 1.90 / 2.07；4K 扫参 r128 1.25 / 1.36、r155 1.25 / 1.44、r512 2.02 / 2.09。
* **问题 2：前缀后清缓存对矮条带有害**：4K reserved（4e54d20 → 725a010）r32 1.07 → 1.07、r64 1.07 → 1.14、r128 1.07 → 1.36、r256 1.44 → 1.46、r512 2.66 → 2.09。默认 4K 实测 1.44，而预期是约 1.07。

**第三轮调整（原因用分配器模拟器查清，见 DESIGN.md §9.13.4 / §9.13.10）：**

* 写了 `tests/alloc_sim.py`：在 meta 设备上跑全尺寸的 Wan decoder，记下每一次分配 / 释放（含卷积内核内部的 columns、输入拷贝），按 PyTorch 缓存分配器的规则重放。它复现了上面 24 个真机读数（两版的 4K 扫参、三档、fp32、第二层），alloc 和 reserved 的误差都 ≤ 0.02 GiB。
* 查清的原因：缓存分配器只能在同一个 segment 内合并空闲块。4e54d20 不清缓存，矮条带能从前缀留下的 384 / 96 MiB segment 里切出所有块，所以停在 1.07；高条带要的块（上采样结果 630 MiB 等）放不进去，只能叠加新 segment（2.66）。725a010 清缓存后条带阶段从零开 segment，而分块卷积的输出缓冲是在第一块之后才分配的，正好落进第一块 columns 释放的洞里，下一块的 384 MiB columns 放不下，只能再开一个 segment（r128：1.36）。
* **改法：** (1) 解码开始时先在缓存里占出一个 arena（分配一个大块后立刻释放），前缀和条带的所有张量都在这一个 segment 里分配和合并，不再清缓存。arena = 张量存活峰值 + 1/32 + 64 MiB（一条条带时 1/8），也就是实际的 reserved。(2) 分块卷积在第一块之前就分配好输出（结果不变）。(3) 传给上采样单元的行切片先做连续拷贝，免得它在内部再拷一份。(4) 估算 = arena + 这次解码最大的单个分配 + 16 MiB：模拟里会溢出 arena 的高条带计划，每次都只是一个请求（一块 columns 或一张全分辨率平面）找不到连续的洞，单独开了一个 segment。384 个计划（512² … 8K、bf16 / fp32、工作区 384 / 192 / 128 MiB、默认和 32 … 整图的强制高度）里 reserved 全部 ≤ 估算。
* 默认规则不变（128 行条带的峰值为目标，取不超过它的最高条带），只是「峰值」改用 arena 来比。

**85a5c6f 复测（`bench_vae_l1c_*`，2026-10）：与模拟器的预测一致，第二阶段验收通过。**

| 输出 | 默认计划 | GTT Δ（实测） | 模拟预测 | 估算 | 热启动耗时（原生） |
|---|---|---|---|---|---|
| 1344×768 | 6 条 128 行 | 0.71 GiB（reserved 0.70） | 0.66 | 1.05 | 0.672 s（0.654） |
| 2688×1536 | 12 条 128 行 | 0.87 GiB | 0.86 | 1.25 | 3.216 s（3.396） |
| 3840×2160 | 15 条 144 行 | 1.13 GiB | 1.13 | 1.52 | 8.241 s（7.889） |
| fp32 1344 / 2688 / 4K | 6×128 / 12×128 / 14×155 | 0.80 / 1.16 / 1.70 GiB | 0.80 / 1.16 / 1.70 | 1.19 / 1.55 / 2.27 | — |

* **所有行（含冷启动）的 reserved / GTT 都在估算以内。** 1344 的 GTT 比 reserved 多 0.01 GiB（GTT 是整机读数）。
* **4K 扫参：** r32 / r64 / r128 / r144 全是 1.13 GiB，r256 1.35，r512 1.82（4e54d20 2.66，725a010 2.09）；耗时 r32 10.60、r64 9.12、r128 8.35、r144 8.23、r256 7.99、r512 7.81 s，与 725a010 持平。
* **144 行以下内存不再下降的原因：** 峰值是 max(前缀, 存档 + 条带)。4K 时整图前缀（全局注意力的 q/k/v、注意力输出、分数块）的存活峰值是 954 MiB，144 行条带的「存档 + 条带」是 938 MiB，已经低于前缀；再矮的条带（128 行 898、64 行 741、32 行 662 MiB）只降低条带部分，峰值仍是前缀 954 MiB（加常驻 99 MiB = 1.03 GiB 存活，arena 1.12 GiB）。瓶颈在前缀。
* **精度：** 与之前相同：4K 对 fp32 的 RMSE monoload 9.73e-4、原生 9.71e-4；fp32 条带解码与整图的差在 1e-5 以下。
* 4K 的原生 fp32 OOM 后退回 tiled 是对照组自己放不下，不是插件的问题。
* alloc 列约等于 arena（arena 这个块本身算一次分配），所以与 reserved 几乎相同。

**workspace 实验（`bench_vae_ws_*`，5d668b6，2026-10）：** 与模拟一致，所有行 reserved ≤ 估算。

* **第一层**（Qwen，GTT Δ GiB / 热启动 s；workspace 384 → 128 MiB）：1344 0.66 / 0.67 → 0.36 / 0.71；2688 0.87 / 3.21 → 0.56 / 3.45；4K 1.13 / 8.24 → 0.87 / 8.44（4K 默认计划变成 14 条 155 行）。峰值降 23–45%，耗时多 2–7%。
* **第二层，Qwen 4K：** 调小内存不降（reserved 9.59 → 9.63–9.69 GiB，峰值是整图激活），反而变慢（6.94 → 7.83 / 9.07 s）。
* **第二层，SDXL 4K：** 调小后 alloc 降了，但 reserved 反而升高（14.99 → 15.32–15.86 GiB，碎片），也更慢（9.89 → 13.15 s）。中低分辨率调小有收益，但那里不是峰值瓶颈。
* **决定：** 第一层默认 workspace 384 → **128 MiB**（`LAYER1_WORKSPACE`；峰值优先、多算可以接受）。不选 64 MiB，是为了给 OOM 重试留一档可以缩（重试下限仍是 64 MiB）。设了 `MONOLOAD_VAE_BUDGET` 时的 `max(64 MiB, 预算/8)` 不变。第二层保持 1 GiB（SDXL 在第三阶段改走第一层）。新默认就是这次测过的 `-w128` 配置，不需要真机重测。
* 新默认下 4K 的峰值仍由前缀决定：前缀存活 698 MiB（qkv 等 6P + 128 MiB 分数块），155 行条带低于它；加常驻 99 MiB、arena 余量后 0.87 GiB。

### 10.3 VAE 解码第三阶段：SDXL / Flux 走第一层（`bench_vae_l3_*`，2026-10，7059bdd）

命令 L–P（§9.7）。设置同 §10.2。完整分析见 DESIGN.md §9.14.10。

* **显存与模拟器一致**（≤ 0.01 GiB），所有行热启动 GTT ≤ 估算。
* **SDXL（GTT 增量 GiB / 热启动 s）：**

| | 1344×768 | 2688×1536 | 3840×2160 |
|---|---|---|---|
| 原生 | 8.36 / 0.86 | 42.8 / 4.65 | 52.5 / 11.9 |
| 第二层 | 3.72 / 0.88 | 9.21 / 4.01 | 15.0 / 9.9 |
| 第一层，方案 A（当时的默认；Flux 相同） | 0.47 / 8.9 | 0.79 / 37.0 | 1.10 / 75.0 |

* **4K 的方案与条带高度：** A 1.10 / 75.0，D 1.52 / 57.8，B 2.17 / 42.1，C 4.70 / 35.5；A r32 / r64 / r96 1.06 / 79.3、1.06 / 76.5、1.06 / 75.7；D r32 / r64 / r96 1.09 / 69.1、1.22 / 61.5、1.36 / 58.9。C 的算量（5.7×）比 B（5.3×）多却更快：全分辨率每次乘加的耗时约是低分辨率的两倍，C 把重算挪到了低分辨率（耗时模型，DESIGN.md §9.14.10）。
* **精度：** 4K 对 fp32 的 RMSE：SDXL 0.00105（原生 0.00108），Flux 0.00107（0.00107）；各方案、各高度 PSNR 都约 61.2 dB；通道均值偏移与原生同一水平；条带边界附近不比其他行差。
* **Qwen 回归：** GTT 0.36 / 0.56 / 0.87 GiB，热启动 0.72 / 3.43 / 8.46 s，不变。
* **决定：** 1 GiB 和 2 GiB 的差别不重要，重要的是能自定义。LDM 默认方案改成 **B**（默认条带规则不变）；设了 `MONOLOAD_VAE_BUDGET` 时改成「预算内最快」（第二层和第一层各「方案 × 条带高度」里，估算不超过预算、预计最快的那个；§13）；`MONOLOAD_VAE_GN_SCHEME` / `MONOLOAD_VAE_STRIPE_ROWS` / `MONOLOAD_DISABLE_VAE_STRIPE` 仍可强制，且优先于预算。验证命令是 §9.7 的 Q–S。

### 10.4 默认方案 B 与按预算选（`bench_vae_b_*`、`bench_vae_budget_*`，2026-10，01377c4）

命令 Q–S（§9.7）。基本符合预测，验收通过。

* **默认 B（GTT GiB / 热启动 s）：** SDXL 0.62 / 4.6、1.33 / 19.5、2.17 / 42.0；Flux 0.62 / 4.7、1.33 / 19.6、2.18 / 41.9。精度对原生 61–62 dB。
* **R（SDXL 按预算选）：** 每档选中的配置与预测一致，GTT ≤ 估算 ≤ 预算，精度 60–62 dB；4K 1G 的报错信息清楚。热启动：1344 `-b3` 1.00 s（1 条带）、`-b1.5` 2.89（C）、`-b1` 4.86（B，64 MiB）；2688 `-b3` 19.7（B）、`-b1.5` 29.0（D，64 MiB）、`-b1` 31.2（D，64 MiB）；4K `-b3` 63.2（D 7×309，64 MiB）、`-b1.5` 77.9（A，64 MiB）。第一版耗时模型在 1344 / 2688 上 ±10% 以内，4K A +6%，4K D 309 行 / 64 MiB −15%。
* **S（Qwen）：** GTT 与预测逐档一致（第二层 2.12 / 5.05 / 9.59，`-b3` 1.31 / 1.96 / 2.09，`-b1.5` 1.06 / 1.08 / 1.12）。
* **问题与处理：** 4K 3G 选中的 D 比默认 B 更慢（63.2 对 42.0 s）、峰值更高（2.33 对 2.17）。B 落选是因为估算 3.17 GiB 里含了 1 GiB 的存档。已收紧估算（存档位置有保证时不算进 largest，4K B 2.68 GiB），耗时模型加了工作区（卷积调用数）和前缀注意力项（DESIGN.md §9.14.11）；验证命令 T。
* **用户确认：** 不认识的 decoder 在预算连第二层都放不下时报错，保持现状。

## 11. 仓库结构

```
__init__.py                 ComfyUI 入口：按 MONOLOAD_DISABLE / MONOLOAD_KEEP_LORA / MONOLOAD_DISABLE_VAE / MONOLOAD_EXACT
                            调用 hotpatch.install()、release.install()、vae.install()；注册节点（monoload/nodes，任何开关下都注册）
monoload/hotpatch.py        运行时合并的全部实现（MonoloadRuntimePatch + ModelPatcher 方法替换；默认的融合 addmm / 放宽合并、
                            MONOLOAD_EXACT=1 的逐位一致路径、量化层在反量化临时权重上的合并）
monoload/release.py         每个 prompt 结束后释放 LoRA（包装 PromptExecutor.execute_async）
monoload/errors.py          报错类型
monoload/comfy_env.py       在独立进程里按指定参数启动 ComfyUI 环境（测试、基准用）
monoload/vae.py             VAE 解码管理入口（包装 VAE.decode：内存估算、逐张、OOM 缩小分块重试、覆盖范围；第一层的接口 STRIPE_ADAPTERS）
monoload/vae_ops.py         逐算子分块（卷积按输出行、注意力按 query；受管理期间的实例属性替换）
monoload/vae_engine.py      第一层的引擎（与 decoder 无关）：区间倒推、执行计划和估算、条带执行、arena、自检流程、适配器基类
monoload/vae_wan.py         第一层的 Wan 2.1 VAE 单帧适配器（结构识别、单元、按 forward 数的内存模型、fp32 副本）
monoload/vae_ldm.py         第一层的 LDM decoder 适配器（SD1.5 / SDXL / SD3 / Flux ae；结构识别、单元、GroupNorm 方案、内存模型、fp32 副本）
monoload/settings.py        总开关 MONOLOAD、MONOLOAD_DISABLE（不导入 torch / ComfyUI）
monoload/lora_overrides.py  单个模型的 LoRA 设置（存在 model_options 里，逐项取值和来源；LoraLoader 的包装记下 LoRA 文件名；不导入 torch / ComfyUI）
monoload/messages.py        所有用户可见的日志 / 报错 / Info 文字的消息表（英文默认，MONOLOAD_LANG=zh 中文）
locales/{en,zh}/nodeDefs.json  节点界面的翻译（ComfyUI 官方的 locales 机制）
monoload/info.py            Monoload Info 节点的文字（版本、全局默认、VAE 设置和解码记录、模型的 LoRA）
web/monoload_info.js        前端扩展：把 Monoload Info 的文字显示在节点框里
web/monoload_i18n.js        前端扩展：下拉选项的显示文字按语言翻译（只改显示，存储值不变）
monoload/vae_overrides.py   单个 VAE 的设置（节点做的副本带的设置；不导入 torch / ComfyUI）
monoload/nodes/             ComfyUI 节点：__init__.py 是注册表（NODES → NODE_CLASS_MAPPINGS），lora_settings.py = Monoload LoRA Settings，vae_settings.py = Monoload VAE Settings，info.py = Monoload Info
tests/                      测试和基准脚本（见第 8、9 节；总开关：test_master_switch.py；LoRA 节点：test_lora_node.py、check_lora_node.py、make_synthetic_checkpoint.py；VAE：test_vae.py、test_vae_stripe.py、test_vae_ldm.py、test_vae_node.py、
                            bench_vae.py、check_vae_node.py、make_synthetic_vaes.py、alloc_sim.py = 缓存分配器模拟，DESIGN.md §9.13.10；
                            第四阶段盘点：vae_inventory.py、probe_vae_gaps.py、check_models.py，DESIGN.md §9.16）
tools/watch_mem.sh          GTT / cgroup 内存监视
tools/compare_images.py     两张图逐像素比较
docs/DESIGN.md              设计说明
docs/HANDOFF.md             交接说明（当前状态、提交记录、规矩、第三阶段的状态和待决定的事）
```

## 12. VAE 解码降峰值

### 12.1 要解决的问题

调研（DESIGN.md §9.1）的结论，待真机数据确认：

* 禁用 MIOpen（ComfyUI 在 AMD 上设 `torch.backends.cudnn.enabled = False`）后，3×3 卷积走 Slow2d：先 im2col 展开再 GEMM。4K 输出时最后一层 Conv256→256 的展开缓冲约 35.6 GiB，SDXL 原版 4K 解码峰值估计约 45 GiB，Qwen 约 31 GiB。**峰值主要来自展开缓冲，不是激活本身。**
* mid block 的全局注意力在 4K 时 N ≈ 13 万，一张 bf16 分数矩阵约 31 GiB；原生按空闲内存自动切片，峰值不固定。
* 原生用 `memory_used_decode` 估算显存：AMD 上 SDXL 4K 估约 92 GiB，并按它调用 `load_models_gpu`（`--gpu-only` 下也可能卸载别的模型）、切分 batch。
* 真 OOM 后原生自动退回 tiled 解码：每个 tile 各自做 GroupNorm 和注意力，结果和整图不等价。

### 12.2 原理

三层设计（DESIGN.md §9.2）。第一阶段实现了第二层和兜底；第二阶段实现了 Qwen 的第一层（§12.7）：

* **第二层：逐算子分块。** decoder 自己的 forward 原样运行，只在受管理的解码期间、只在这个 VAE 的模块上替换两类重算子：
  * 卷积（Conv2d、Conv3d，含 comfy.ops 各变体）：整层的 im2col 展开估算超过预算时，按输出行分块，每块只取它需要的输入行（含上下 halo），块与块之间用真实的相邻行，zero padding 只出现在图像真实的上下边缘；输出预先分配好，逐块写入。接在 `_conv_forward` 上，所以 `cast_bias_weight` / weight_function（包括 Monoload 的运行时 LoRA）照常生效。
  * 注意力：按 query 分块，K/V 保持完整，每个 query 仍对全图做 softmax；每块的运算与原生实现相同。
  * GroupNorm、上采样等其余算子保持原生。整图语义不变，算量约 1 倍。
* **管理入口（包装 `VAE.decode`）：** 自己按形状估算内存上界交给 `load_models_gpu`（SDXL 4K 约 17.9 GiB，原生约 92 GiB，估算方法见 DESIGN.md §9.4）；batch 逐张；输出的设备、dtype、`process_output`、NHWC 与原生完全一致。
* **OOM：** 只缩小分块（预算减半）重试，降到 64 MiB 仍 OOM 就抛 `MonoloadVAEOOMError`，**绝不调用 tiled 解码**。
* **兜底：** 不在覆盖范围内的解码走原生整图解码，打一条日志。
* **第一层（条带解码）：** 认得的结构按输出条带倒推重算，进一步去掉高分辨率激活。第二阶段已做 Qwen（Wan 2.1 VAE 单帧，§12.7）；第三阶段做 SDXL / Flux（LDM decoder，GroupNorm 的整图统计量跨条带调度，§12.8，待真机）。

### 12.3 开关

总开关和节点见 §4 / §4.1：默认开启；`MONOLOAD=0` 时 VAE 全局原生，节点 `mode` 选 `auto` 的 VAE 照样管理。VAE 相关的高级选项（`MONOLOAD_VAE_WORKSPACE`、`MONOLOAD_VAE_BUDGET`、`MONOLOAD_DISABLE_VAE_STRIPE`、`MONOLOAD_VAE_STRIPE_ROWS`、`MONOLOAD_VAE_GN_SCHEME`、`MONOLOAD_DISABLE_VAE`）见 §13。

启动日志写明状态和策略：`[Monoload] VAE decode managed: ... workspace 1.00 GiB ...` 加上 `[Monoload] VAE layer 1 (stripe decoding) on ...: default stripe policy: the peak of 128-row stripes, tallest stripes within it ...`（设了预算时是 `peak budget ... (MONOLOAD_VAE_BUDGET)`；或 `... off`），或 `[Monoload] VAE decode: native by default (MONOLOAD=0) ...`（全局原生时每次解码只在 DEBUG 级别记一行）。每次解码的日志写明用了哪一层：`-> layer 1 (Wan 2.1 stripes): 14 stripes of 155 rows (core), recompute 1.21x, checkpoint 95 MiB; default: peak of 128-row stripes, workspace 128 MiB; arena 886 MiB, memory estimate 1.16 GiB ...` 或 `-> layer 2, op-level chunking ...`。

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
* 第二阶段（Qwen 的第一层）已实现并通过真机验收（85a5c6f，§10.2、§12.7）。第三阶段（SDXL / Flux 的第一层）已实现、待真机（§12.8），交接说明见 docs/HANDOFF.md。
* 实现细节、行区间推导和第三阶段计划见 DESIGN.md §9。

### 12.7 第二阶段：第一层条带解码（`qwen_image_vae` / Wan 2.1 VAE 单帧）

**原理。** 在 62b3c94 里，Wan 2.1 VAE 解码单帧（T=1）就是一串模块依次调用：`conv2 → decoder.conv1 → middle（残差块、全局注意力、残差块）→ upsamples（每个分辨率 3 个残差块 + 上采样）→ head`，没有时间缓存，卷积等效于 2D，RMS 归一化是逐位置的，没有全图统计。所以：

* **前缀整图算，存一个低分辨率存档：** `conv2`、`conv1`、middle（注意力继续用第二层的 query 分块）和最低分辨率的 3 个残差块都在 H/8 上整图计算，结果就是存档（H/8，384 通道，4K 时约 95 MiB）。
* **之后按输出条带倒推重算：** 对每条输出条带 `[o0, o1)`，从 head 往回逐个单元推出需要的输入行：3×3 卷积两侧各多 1 行，残差块（两个 3×3 卷积）各多 2 行，nearest×2 上采样 `[⌊a/2⌋, ⌈b/2⌉)`，逐点运算不变，与图像边界求交（4K 时存档上每侧约 7 行）。然后从存档上取这些行，依次调用**原模型里的模块实例**（comfy.ops 的 cast / weight_function 照常，条带内的大卷积照样受第二层工作区约束），每个单元只保留需要且精确的那些行，最后只把核心行写进预先分配的输出。
* **zero padding 只在真实边缘起作用：** 模块对切片照常补零，但凡是受切片内部边界补零影响的输出行（每侧恰好 halo 行），都按全局坐标算出来并丢掉，不会进入后续计算；每个单元都检查「需要的行 ⊆ 精确的行」，不满足就是内部错误。图像真正的上下边缘处补零与整图完全一样。
* **内存：arena + 估算。** 按形状事先算出张量的存活峰值（前缀、存档、输出缓冲、每条带每个单元；按 forward 代码逐个数同时存活的张量，与真机 alloc 吻合到 ±5 MiB）。解码开始时在 PyTorch 的缓存分配器里先占出一个 arena（存活峰值 + 1/32 + 64 MiB；只有一条条带时 + 1/8）：分配一个这么大的块再立刻释放，之后所有张量都从这一个 segment 里切出、释放后在里面合并，所以实际占用（reserved / GTT）就是 arena。交给 `load_models_gpu` 的估算 = arena + 这次解码里最大的单个分配 + 16 MiB：高条带时偶尔会有一个请求（一块 columns 或一张全分辨率平面）被碎片挤出 arena、单独开一个 segment，估算把它算进去。分块卷积在第一块之前就分配好输出、传给上采样的切片先连续化，都是为了让 arena 里少出碎片。arena 的大小和估算用分配器模拟器（`tests/alloc_sim.py`，复现了 24 个真机读数）在 384 个计划上验证过：reserved 全部 ≤ 估算。自检结束后释放它的全部张量并清空缓存。
* **默认条带高度：在不明显变慢的前提下尽量低峰值。** 以 128 行条带的峰值（arena）为目标，取不超过它的最高条带（真机扫参：4K 时 128 行已经到了峰值下限、速度与 432 行相同，64 行起明显变慢）。条带阶段的峰值低于整图前缀时（大图），条带可以更高而不多占内存；小图就是 128 行；不足 128 行的图一条整图。条带内的工作区 128 MiB（workspace 实验后从 384 改小，§10.2；`MONOLOAD_VAE_WORKSPACE` 更小时取它）。默认计划（Qwen，bf16）：

| 输出 | 默认计划 | 重算 | GTT Δ（实测，workspace 128） | 估算 | 热启动耗时 | workspace 384 时（85a5c6f） |
|---|---|---|---|---|---|---|
| 1344×768 | 6 条 128 行 | 1.23× | 0.36 GiB | 0.50 GiB | 0.71 s | 0.66 GiB / 0.67 s |
| 2688×1536 | 12 条 128 行 | 1.25× | 0.56 GiB | 0.71 GiB | 3.45 s | 0.87 / 3.21 |
| 3840×2160 | 14 条 155 行 | 1.21× | 0.87 GiB | 1.16 GiB | 8.44 s | 1.13 / 8.24（15 条 144 行） |
| 7680×4320 | 13 条 333 行 | 1.10× | 3.36 GiB（模拟） | 4.01 GiB | — | 3.14（模拟） |

  4K 的峰值由前缀决定：workspace 384 时前缀存活 954 MiB，已高于 144 行条带的 938 MiB，所以 144 行以下不再下降；workspace 128 时前缀降到 698 MiB（分数块从 384 变成 128 MiB），默认条带 155 行，峰值 0.87 GiB（§10.2）。8K 的默认计划在模拟里有一个请求被挤出 arena（arena 2.88、reserved 3.36 GiB），仍在估算以内。

  `MONOLOAD_VAE_BUDGET` 设了就改为「估算不超过预算的最高条带」（工作区 = 预算/8，至少 64 MiB），`MONOLOAD_VAE_STRIPE_ROWS` 强制条带高度，两者都优先于默认规则。
* **单帧 Conv3d 改走 conv2d：** 关掉 cuDNN/MIOpen 后，PyTorch 在 GPU 上用 SlowDilated3d 跑 Conv3d，每次调用按输出通道逐个 `fill_(bias)`（真机 profile 里 15.9 万次 `aten::fill_`）。单帧、时间 kernel 实际为 1 的 Conv3d 调用改成在第 0 帧上调 `F.conv2d`（Slow2d：一次 `copy_` 设 bias，im2col 和 GEMM 与 3D 路径相同），只在 GPU + cuDNN 关闭时生效，第二层的单帧 Wan 解码也一样。
* **自检：** 每种结构在本进程第一次使用时，用 fp32 副本（只复制 conv2 和 decoder，测完释放）、24×24 的 latent、强制 40 行的条带（5 条，最后一条较短，条带内的大卷积再分块），和原生 `WanVAE.decode` 整图对比，相对误差 ≤ 1e-4 才启用第一层。结果按结构缓存，不改变随机数状态。不通过就对这种结构禁用第一层，改走第二层，日志里醒目地警告。
* **识别（按真实结构，不看文件名）：** `first_stage_model` 是 `comfy.ldm.wan.vae.WanVAE`、decoder 是 `Decoder3d`；upsamples 里只有 ResidualBlock 和 Resample（upsample2d / upsample3d，nearest(-exact)×2 + 3×3 卷积），没有 AttentionBlock；归一化都是 RMS_norm；卷积 stride 1、kernel 3 或 1、padding 符合预期；Dropout 处于 eval 或 p=0；模块上没有 forward hook 或实例级 forward 替换；latent 是 T=1、通道数对得上；没有 vae_options。任何一项不符就走第二层，并对这个 VAE 打一条日志说明哪里不符。Wan 2.2（48 通道）等其他结构都不匹配。
* **OOM：** 条带高度和工作区一起减半重试，到 8 行 / 64 MiB 仍不行就抛 `MonoloadVAEOOMError`。不退回 tiled，也不退回第二层（第二层的峰值更高）。
* **预算放不下：** 设了 `MONOLOAD_VAE_BUDGET` 而 8 行的条带也放不下时直接报错，写明需要多少。默认规则没有这个问题：它的目标本身就是能达到的峰值（不低于整图前缀）。

**限制：** 只覆盖 Wan 2.1 VAE 的单帧解码；SDXL / Flux 等 LDM decoder 见第三阶段（§12.8）。前缀仍然整图计算，4K 时它（全局注意力的 q/k/v、注意力输出和分数块）决定了约 0.78 GiB（存活）/ 0.87 GiB（arena）的峰值下限（workspace 128），8K 约 2.7 GiB。条带越矮重算越多（4K：128 行约 1.26 倍，32 行约 2.1 倍）。arena 和估算是用模拟器验证的，不是数学证明；估算比实际占用高一个「最大的单个分配」（默认计划 0.14–0.3 GiB），只影响交给 `load_models_gpu` 的数。配置了 `max_split_size_mb` / `expandable_segments` 时不用 arena。

**状态：第二阶段完成。** 4K GTT 原生 59.2 → 0.87 GiB（workspace 128），估算是上界，耗时 8.44 s（原生 7.89 s），精度与原生同为 bf16 水平，没有接缝（§10.2）。经过：4e54d20（第一版，4K 2.68 GiB）→ 725a010（默认策略、conv2d 改道；1.44 GiB，估算低于 reserved）→ 85a5c6f（arena、估算 = arena + largest；1.13 GiB，验收通过）→ 第一层 workspace 384 → 128 MiB（0.87 GiB）。第三阶段（LDM decoder）的入口见 docs/HANDOFF.md。细节见 DESIGN.md §9.13。

### 12.8 第三阶段：SDXL / SD1.5 / SD3 / Flux `ae`（LDM decoder）走第一层

**状态：已验收（7059bdd，§10.3）。之后默认方案改成 B，并加了「预算内最快」（命令 Q–S 待实测）。**

**难点。** LDM decoder 和 Wan 的结构几乎一样（前缀整图算到 H/8，之后按输出条带倒推重算），只差一点：归一化是 GroupNorm，每组的均值和方差要在**整张图**上算。条带上原样调用 GroupNorm 会用条带自己的统计量——这正是 tiled 解码不等价的原因，Monoload 不做这种近似。

**做法（DESIGN.md §9.14）：**

* **统计遍。** 条带部分有 19 个 GroupNorm（9 个残差块各两个 + `norm_out`），它们一个依赖一个。按顺序对每个 GroupNorm 跑一遍条带：只算到它的输入为止，把每条带「自己负责的那些行」的统计量累加起来，算完冻结；最后一遍按输出条带解码，所有 GroupNorm 都用冻结的整图统计量。结果与整图解码数学等价（fp32 测试里差 ≤ 4e-6）。
* **统计量的数值。** fp32；每组先减去一个平移量（第一块的均值），块内用 `torch.var_mean`，块与块、条带与条带用 Chan 合并（数值稳定，不做 `E[x²] − E[x]²` 这种会抵消的减法）。应用时与 ATen 的 GroupNorm 内核同一算法（`x·a + b`，fp32 算、回到原 dtype）。
* **GroupNorm 的实例级替换。** 只在受管理的第一层解码期间、只对条带部分的 19 个 GroupNorm 实例设实例属性 `forward`，退出时删掉；权重仍走模块自己的 cast / weight_function（包括 Monoload 的运行时 LoRA）。这是对「模块原样调用」的有意偏离，由 fp32 自检兜底（与原模型自己的整图 `decode` 比，≤ 1e-4）。
* **方案 A / D / B / C 是同一个机制**，只是「哪些位置把中间结果整张存下来」不同，每遍从最近的存档出发：A 不存（峰值最低、算得最多），D 存 H/4 级的输出，B 再存 H/2 级的输出，C 再存全分辨率每个残差块的输出。存档在统计遍里顺便写好，不需要额外的遍。**默认 B**（4K 2.17 GiB / 42 s；7059bdd 时默认 A：1.10 GiB / 75 s）；`MONOLOAD_VAE_GN_SCHEME` 强制别的方案；设了 `MONOLOAD_VAE_BUDGET` 时方案和条带高度按「预算内最快」选（§13）。只有一条带时（条带高度 ≥ 图高）不跑统计遍，就是整图解码。
* 其余沿用第二阶段：识别按真实结构（类型、kernel、padding、没有注意力 / 时间维 / tanh_out 等非默认分支、没有 hook），不认就走第二层并在日志写原因；每种结构 + 方案第一次使用前 fp32 自检；默认条带策略、预算、强制高度、OOM 缩条带和工作区、绝不退回 tiled、估算交给 `load_models_gpu`；整个解码在一个 arena 里。

**模拟器的预测（SDXL / Flux，bf16，GiB）：**

| 输出 | 原生（实测） | 第二层（实测） | 第一层 A（模拟 reserved / 估算） | D | B | C |
|---|---|---|---|---|---|---|
| 1344×768 | 8.36 | 3.72 | 0.47 / 0.61（12×） | 0.54（8×） | 0.62（5.1×） | 0.79（5.9×） |
| 2688×1536 | 42.84 | 9.25 | 0.79 / 0.99（12×） | 1.01（8×） | 1.33（5.1×） | 2.44（5.2×） |
| 3840×2160 | 52.48 | 14.86 | 1.10 / 1.48（12×） | 1.52（8×） | 2.17（5.3×） | 4.70（5.7×） |

（倍数是卷积算量相对整图解码。7059bdd 的实测与这张表一致，≤ 0.01 GiB，§10.3。）分配器模拟器先复现了 SDXL / Flux 第二层的 18 个真机读数（误差 ≤ 0.02 GiB，包括 4K「工作区越小 reserved 反而越高」的碎片现象），再验证了 202 个第一层计划（1024² … 8K、bf16 / fp32、四种方案、默认和强制高度）reserved 全部 ≤ 估算。

**代价：速度。** 第一层要重算：4K 方案 A 75 s、D 58 s、B 42 s、C 35.5 s，第二层 9.9 s，原生 11.9 s（§10.3）。峰值和速度之间怎么取，用 `MONOLOAD_VAE_BUDGET` 自己定：预算够放下第二层就是第二层的速度。

**限制：** 只认 2D 图像的 LDM decoder；3D / 视频 decoder、up 级带注意力、`tanh_out`、Flux 2 的 `batch_norm_latent` 走第二层。统计量与原生 GroupNorm 内核不逐位一致（累加顺序不同）。arena 和估算是模拟器验证的，不是证明。

## 13. 高级选项（环境变量）

一般不用设。它们都是**全局默认值**：`MONOLOAD` 开启（默认）时作用于所有模型 / VAE；`MONOLOAD=0` 时只作用于节点开启了的那个模型 / VAE 上节点没设的项。节点上明确选的值对那一个模型 / VAE 总是压过它们（§4 的优先级）。基准和测试脚本仍然用这些环境变量来配置。

| 设置 | 效果 |
|---|---|
| `MONOLOAD_EXACT=1` | 合并改走逐位一致路径：结果与原生 ComfyUI 完全相同，每步多出的合并开销更大（CT 700 上约是默认路径的 3 倍：0.26s 对 0.08s，见 §5.1）。用于验收、对比，或需要和原生出图完全一致的时候。VAE 解码的全局默认也变成原生（分块会改变 GEMM 形状，不保证逐位一致）；节点 `mode` 选 `auto` 的 VAE 仍然管理 |
| `MONOLOAD_KEEP_LORA=1` | 运行时合并照常，但 prompt 结束后不释放 LoRA（LoRA 节点缓存和挂着 LoRA 的 clone 保留，重复跑同一个 LoRA 工作流时省掉读 LoRA 文件） |
| `MONOLOAD_DISABLE_VAE=1` | VAE 解码的全局默认变成原生（§12），LoRA 部分不受影响；节点 `mode` 选 `auto` 的 VAE 仍然管理 |
| `MONOLOAD_VAE_WORKSPACE` | VAE 分块的工作区（卷积每块的 im2col 展开缓冲、注意力每块的分数矩阵的上限），默认 `1G`；写法 `1G`、`512M`、`768`（纯数字按 MiB）。第二层直接用它；第一层用 min(它, 128M)，所以只有设得比 128M 小才影响第一层（§12、DESIGN.md §9.13.11） |
| `MONOLOAD_VAE_BUDGET` | VAE 解码的峰值预算，写法同上（`3G`、`1.5G`、`2560M`）。不设时用默认策略（第一层，128 行条带的峰值内最高的条带；LDM 用方案 B）。设了就在**估算不超过预算**的做法里选**预计最快**的：第二层能放下就用第二层（每个卷积只算一次，最快）；放不下就在第一层的「GroupNorm 方案 × 条带高度」里按耗时模型挑最快的；一个都放不下就报错并写明各需要多少。例子见下 |
| `MONOLOAD_DISABLE_VAE_STRIPE=1` | 只关掉第一层，所有受管理的 VAE 解码都走第二层（设了预算也一样，超出预算时日志注明） |
| `MONOLOAD_VAE_STRIPE_ROWS` | 强制第一层的条带核心高度（输出行数），用于扫参和调试，优先于默认策略和预算（有预算时方案仍按预算选；放不下也照跑，日志注明） |
| `MONOLOAD_LANG` | 日志、报错和 Monoload Info 节点文字的语言：不设 / `en` = 英文（默认），`zh` = 中文。节点界面不看它，跟随 ComfyUI 自己的语言设置 |
| `MONOLOAD_VAE_GN_SCHEME` | 强制 SDXL / SD1.5 / SD3 / Flux `ae`（LDM decoder）第一层的 GroupNorm 整图统计量方案：`A`（不存中间结果，峰值最低、重算最多）、`D`、`B`（**默认**）、`C`（存得越多峰值越高、越快），见 §12.8。设了它，有预算时也只用这个方案（条带高度仍按预算取） |

**`MONOLOAD_VAE_BUDGET` 的例子**（SDXL / Flux `ae`，bf16；「估算」是交给 `load_models_gpu` 的上界，峰值是模拟的 reserved（实测与它一致）；耗时：第二层和默认 B 是实测，其余是 CT 700 上耗时模型的预测；DESIGN.md §9.14.10–11）：

| 预算 | 1344×768 | 2688×1536 | 3840×2160 |
|---|---|---|---|
| 不设（默认） | 方案 B，6 条 128 行：峰值 0.62 GiB，4.6 s | B，12 条 128 行：1.33 GiB，19.5 s | B，17 条 128 行：2.17 GiB，42.0 s |
| `20G` | 第二层：3.7 GiB，0.9 s | 第二层：9.2 GiB，4.0 s | 第二层：15.0 GiB，9.9 s |
| `3G` | 第一层整图 1 条：2.0 GiB，约 0.8 s | C，4 条 384 行：2.7 GiB，约 12 s | B，12 条 180 行：2.4 GiB，约 41 s |
| `2G` | C，2 条 384 行：1.2 GiB，约 2.8 s | B，11 条 140 行：1.7 GiB，约 19 s | D，18 条 120 行：1.5 GiB，约 59 s |
| `1.5G` | C，2 条 384 行：1.05 GiB，约 2.8 s | B，16 条 96 行：1.2 GiB，约 21 s | A，17 条 128 行：1.1 GiB，约 74 s |
| `1G` | C，2 条 384 行（工作区 128 MiB）：0.86 GiB，约 4 s | D，23 条 67 行：0.80 GiB，约 31 s | 报错：最少要约 1.38 GiB（方案 A），第二层要 17.9 GiB |

* 预算比的是**估算**（上界），所以实际峰值通常比预算低 10–30%。有存档的方案（B / C / D）的估算在 vae-estimate-fix 之后收紧了：存档在 arena 里的位置有保证，不再当成「可能被挤出 arena 的那一块」算进估算（4K 默认 B 3.17 → 2.68 GiB），所以 4K 设 `3G` 能选 B 了（以前选 D，实测更慢、峰值更高）。
* `qwen_image_vae`（只有一种第一层配置）：`20G` → 第二层（2.1 / 5.1 / 9.6 GiB）；`3G` → 第一层预算内最高的条带（1 条 768 行 / 2 条 768 行 / 4 条 540 行，1.3 / 2.0 / 2.1 GiB）；`1.5G` → 1.06 / 1.08 / 1.12 GiB；`1G` → 0.58 / 0.72 GiB，4K 报错（最少约 1.09 GiB）。
* 日志写明选了什么、为什么、其他候选各要多少：`[Monoload] VAE budget 3.00 GiB (from environment variable MONOLOAD_VAE_BUDGET) -> layer 1 scheme B 180 rows (workspace 128 MiB) 2.96 GiB, ~41.1 s: the fastest predicted that fits; others: layer 2 17.91 GiB (over); ...`。
* 预算的来源写在括号里：`(from environment variable MONOLOAD_VAE_BUDGET)` 或 `(from the Monoload VAE Settings node)`；放不下时的报错也按来源给建议（来自节点：改节点上的预算，或改成跟随全局 / 不限；来自环境变量：改 `MONOLOAD_VAE_BUDGET`）。强制的条带高度、方案、「只用第二层」同样写明来源。
* 工作区也是候选的一维：依次试 预算/8、128 MiB、64 MiB，取耗时模型预测最快的（工作区越小分块越多、越慢，但估算也越低）。
* 想固定某个方案或高度：再设 `MONOLOAD_VAE_GN_SCHEME` / `MONOLOAD_VAE_STRIPE_ROWS`，它们优先于预算。
