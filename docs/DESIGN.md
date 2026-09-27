# Monoload 设计说明（v1）

参考源码：锁定镜像 `kyuz0/amd-strix-halo-comfyui@sha256:384aa1fe…` 里的 `/opt/ComfyUI`，
ComfyUI 0.31.0，commit `62b3c94bd45154f6486c7abf1b9efcacee96ea69`。下文的函数/行为都按这份源码核实过。

## 0. 对需求里「原生为什么做不到」的核实

| 说法 | 核实结果 |
|---|---|
| `load_torch_file` 在 `--disable-mmap` 下逐张量 `copy=True` 拼完整 dict | 属实（`comfy/utils.py:133-139`）。而且它先用 `safetensors.safe_open` 打开文件，**safe_open 本身就是 mmap**，`--disable-mmap` 只是在 mmap 之后再复制一份。 |
| `load_diffusion_model_state_dict` 先 `get_model` 建模型，`model.to(offload_device)`，再 `load_state_dict(assign=False)` | 属实（`comfy/sd.py:2211-2216`），dict 和参数同时存在。 |
| 只有 DynamicVRAM 才 `assign=True` | 属实：`assign=model_patcher.is_dynamic()`。DynamicVRAM 由 `main.py` 在 `enables_dynamic_vram()` 且（`--enable-dynamic-vram` 或 NVIDIA）时打开；`--gpu-only`/`--highvram`/`--cpu` 都会关掉它。这个版本在 AMD 上默认不开。 |
| 默认打 LoRA 时备份原权重 | 属实：`ModelPatcher.patch_weight_to_device` 把原权重放进 `self.backup`（`model_patcher.py:906-907`）。`--gpu-only` 时 offload_device 就是 GPU，备份也在 GTT 里。 |

## 1. 文件格式

转换产物是**合法的 safetensors 文件**（任何 safetensors 工具都能读），额外约定：

* 张量顺序 = 加载顺序 = 重建出的模型 `named_modules()` 遍历顺序（参数、persistent buffer 依次排列），数据区首尾相接，没有空洞（safetensors 本身也不允许空洞）。
* 张量名 = **ComfyUI 内部最终名字**，即 `BaseModel` 下的完整路径（`diffusion_model.xxx`、`model_sampling.sigmas` …），不是源文件里的名字。key 转换、`process_unet_state_dict`（拆分/合并权重）都已经在转换时由 ComfyUI 做完。
* 保留命名空间 `__monoload_aux__.*`：见 §2.3，存放 `get_model()` 构造模型时需要读取的少量源张量。
* 不做 4K 对齐（v1 不要求）。加载器不依赖对齐：GPU 路径按字节搬运，非连续目标才走带临时张量的拷贝。

`__metadata__`（safetensors 规定只能是 str→str），全部以 `monoload.` 开头：

| key | 内容 |
|---|---|
| `monoload.format` | 固定 `"monoload"` |
| `monoload.format_version` | `"1"`，加载器只接受自己支持的版本 |
| `monoload.version` | 写文件的 Monoload 版本 |
| `monoload.component` | v1 只有 `diffusion_model`；预留 `text_encoder` / `vae` |
| `monoload.quant` | v1 只有 `none`；预留 `fp8_scaled` / `gguf` 等 |
| `monoload.model` | JSON：重建模型所需的全部信息（§2） |
| `monoload.source` | JSON：源文件名、大小、mtime、blake3（没有 blake3 时 sha256） |
| `monoload.env` | JSON：ComfyUI 版本/commit、torch 版本、转换时的 ComfyUI 启动参数、load/offload device |
| `monoload.convert_log` | JSON：转换时 ComfyUI 打出的 missing/unexpected keys 等警告 |

JSON 里非原生类型用带标签的编码：`torch.dtype` → `{"__dtype__": "bfloat16"}`，tuple → `{"__tuple__": [...]}`，枚举 → `{"__enum__": "comfy.model_base.ModelType", "name": "V_PREDICTION"}`。遇到无法编码的类型，转换器直接报错，不会悄悄丢信息。

## 2. 重建模型需要的信息

原生流程（`load_diffusion_model_state_dict`）里，所有「从权重推断」的东西都集中在这几处：

1. `model_detection.detect_unet_config(sd)` → `unet_config`（形状推断出的结构参数，含 `image_model` 等）
2. `model_config_from_unet_config()` → 按顺序匹配 `supported_models.models`，选中类（`matches()` 还可能看 `required_keys`）
3. `detect_layer_quantization()` → `quant_config`（v1 若检测到量化则拒绝转换）
4. dtype 选择：`unet_dtype(model_params, supported_dtypes, weight_dtype)`、`unet_manual_cast(...)` → `set_inference_dtype(dtype, manual_cast, device)`。注意有的子类会**改写**它（`Anima` 会改 `memory_usage_factor`，`CosmosI2VPredict2` 会根据设备改 manual_cast）
5. `get_model(sd)`：`model_type(sd)`（SDXL 系看 `v_pred` / `ztsnr` / `edm_*` 这些 key，还可能**改写 `sampling_settings`**）；部分类在 `get_model` 里直接读 state dict 的值或形状（`Stable_Zero123` 的 `cc_projection`、`StableAudio3` 的 padding embedding、`PixelDiT` 的形状）
6. `process_unet_state_dict()`：key 改名/拆分（在 `load_model_weights` 里做）

### 2.1 转换时怎么采集

转换器**原样调用** `comfy.sd.load_diffusion_model_state_dict`（这正是 `UNETLoader` → `load_diffusion_model` 的主体），只在转换进程里、只在这一次调用期间挂几个观测用的包装：

* 包 `model_detection.model_config_from_unet_config`：记录**传给配置类构造函数的原始 `unet_config`**（深拷贝）和选中的类（`模块.限定名`）。
* 在选中的 config **实例**上包 `set_inference_dtype`：记录传入参数（dtype、manual_cast、device），以及调用后的结果（`unet_config["dtype"]`、`manual_cast_dtype`、`memory_usage_factor`）。
* 在同一实例上包 `get_model`：传入一个「访问记录 dict」，记录 `get_model`/`model_type` 读了哪些 key（`in` 的结果、`[]`/`get` 取了哪些值、有没有遍历整个 dict）。取过值的 key 以 `__monoload_aux__.<key>` 存进文件；遍历过的话把全部 key 的形状/dtype 写进 metadata。如果 `get_model` 修改了 state dict，转换直接报错（v1 不支持）。
* 从加载完的模型上读：`model_type`、`get_model` 之后的 `sampling_settings`、`optimizations`、`latent_format` 类、模型类 / `diffusion_model` 类 / `model_sampling` 类、`adm_channels`、`concat_keys`（inpaint 等）、`memory_usage_factor`。
* 同时记录原生 dtype 选择的输入：源的参数量、源的主 dtype、`model_options`（v1 只支持 `UNETLoader` 的 `default`）。

写出去的张量就是**加载完之后** `BaseModel` 的全部参数 + persistent buffer（包含 `model_sampling.*`），逐个张量写文件：CPU 张量直接写字节，GPU 张量逐个 `.cpu()` 后写，不再拼第二份 dict。先写 `*.partial`，fsync 后 rename。

### 2.2 转换器怎么读源文件、内存峰值

按确认的方案：**走 ComfyUI 原生的 CPU 加载路径，不为省内存做特殊处理**。

* 原生 `load_torch_file` 内部用 `safetensors.safe_open`，而它本身就是 mmap（`--disable-mmap` 只是 mmap 之后再 `copy=True`）。为了守住「任何环节都不 mmap 模型文件」，转换器用一个几十行的 `read_state_dict_cpu()` 代替它：同样的 key（同样按名字排序）、同样的 dtype/形状、每个张量一块独立的 CPU 内存，只是用 `preadv` 读。之后原样交给 `comfy.sd.load_diffusion_model_state_dict`。测试里验证了转换结果与原生 `UNETLoader` 加载的模型逐字节一致。
* 源文件哈希单独顺序读一遍（blake3，没有就 sha256），同样不 mmap。
* 内存：`--gpu-only` 下模型参数在 GPU（GTT，不计入 cgroup），CPU 上是完整的源 dict。**转换进程的 CPU 峰值 ≈ 源文件大小 + 约 1 GiB 基础开销**；GTT 里是一份模型。具体实测数字和 Krea 2 的估算见 README。

### 2.3 加载时怎么重建

1. 读文件头（`pread`，不 mmap），校验 `format`、`format_version`、`component`、`quant`。
2. `import` 记录的配置类（找不到 → 报错）；`cls(原始 unet_config)` 构造（与原生完全相同的构造路径，`unet_extra_config`、子类 `__init__` 都会照常执行）。
3. **重算原生会选的 dtype**：用记录的参数量/主 dtype 和当前进程的启动参数、当前设备，调用 `model_management.unet_dtype` / `unet_manual_cast`。与记录值不同（换了 `--bf16-unet` 之类或换了设备）→ 报错要求重新转换，因为文件里存的就是那个 dtype 的权重。
4. 用记录的参数调用 `set_inference_dtype`，结果与记录不一致 → 报错。
5. 恢复 `optimizations`、`sampling_settings`；在这个 config **实例**上把 `model_type` 换成返回记录值的函数（只影响 Monoload 自己的实例）。
6. `get_model(视图 dict, device=目标设备)`：视图 dict 只回答转换时记录过的访问（`in` 结果、aux 张量、遍历时的 meta 张量），出现未记录的访问 → 报错。
7. 比对指纹：模型类、`diffusion_model` 类、`model_sampling` 类、`model_type`、`latent_format` 类、`adm_channels`、`concat_keys`、`memory_usage_factor`、`manual_cast_dtype`。不一致 → 报错。
8. 严格校验张量清单：模型的参数 + persistent buffer 名字集合必须与文件完全一致（多了、少了都列出来），每一项形状、dtype 一致。
9. 搬运（§3），包成 `MonoloadModelPatcher`，`cached_patcher_init = (monoload.loader.load_monoload_diffusion_model, (path, model_options))`。

只有 ComfyUI 版本/commit 与转换时不同才是警告；上面任何一项失败都是 `MonoloadError`，信息里带「请重新转换」和具体差异。

### 2.4 「参数只分配一次」

* 目标设备 = 原生的 `unet_offload_device()`：`--gpu-only`（HIGH_VRAM）时是 GPU，普通模式是 CPU——和原生模型「待机时住在哪」一致。
* `get_model(..., device=目标设备)`：`comfy.ops` 的层用 `torch.empty(device=…)` 直接在目标设备分配，**这就是唯一的一份**；不依赖「CPU 上未触碰的 empty 页 + `.to()`」。
* 少数没把 `device` 传下去的模块会在 CPU 上建参数：加载器检查每个要从文件读的张量，不在目标设备上就在目标设备上重新 `empty` 一个替换进去，原来那个随即释放，并在日志里统计这类张量的数量和字节数。实际遇到的主要是 `model_sampling.*`：它们在构造时于 CPU 上算出，只有几 KB，而且文件里有同样的值，会原样读回。
* DynamicVRAM（comfy-aimdo）开启时 `comfy.ops` 会延迟建参数、并依赖 mmap：v1 **明确报错不支持**，提示用 `--gpu-only` 或 `--disable-dynamic-vram`。

## 3. 搬运

* 打开文件用 `os.open(O_RDONLY)`，读用 `os.preadv`，全程无 mmap。
* 计划：按文件偏移排序，把**连续**的小张量打包进一块缓冲一次读完；大于缓冲的张量拆成缓冲大小的片段。
* **GPU 目标**：两块 `torch.empty(size, uint8, pin_memory=True)`（默认 512MiB×2，环境变量 `MONOLOAD_BUFFER_MB` 或节点参数可调）。单线程流水：
  等 `event[b]`（上一次从缓冲 b 发出的 H2D 完成）→ `preadv` 读进缓冲 b → 在专用 copy stream 上对每段 `dst_bytes.copy_(buf[b][..], non_blocking=True)` → `event[b].record()`。于是读缓冲 B 的同时，GPU 在搬缓冲 A。目标张量按字节视图（`view(uint8)`）拷贝，与对齐无关。开始前 copy stream 等待默认流（保证参数分配已完成），结束后同步。
* **CPU 目标**（普通模式 / `--cpu` 测试）：目标内存就是最终位置，直接 `preadv` 进参数的内存，零中转。设 `MONOLOAD_STAGING=always` 可强制走双缓冲路径（测试用；无 GPU 时缓冲不 pin）。
* 缓冲在加载结束后释放，并调用 `torch._C._host_emptyCache()`（存在时）把 PyTorch 缓存的 pinned 内存还给系统。
* 峰值：CPU 内存 ≈ 缓冲（2×512MiB）+ 少量；GPU ≈ 1 份模型。

## 4. LoRA：运行时临时合并

ComfyUI 已有的机制：`comfy.ops` 的层在 `forward` 里只要 `weight_function`/`bias_function` 非空就走 `cast_bias_weight`，它会先 `copy=True` 拿一份临时权重，再依次调用这些函数——lowvram 模式就是把 `LowVramPatch`（内部调 `comfy.lora.calculate_weight`）挂到这里。

Monoload 的接入点只有一个：`MonoloadModelPatcher(ModelPatcher)` 重写 `patch_weight_to_device()`。原生 `load()` 在全量加载时对每个参数调用它来「烘焙」LoRA 并备份；Monoload 版本对有 patch 的 key：

* **不备份、不改权重**，而是把 `MonoloadRuntimePatch`（`LowVramPatch` 子类）插到该层 `weight_function`/`bias_function` 的**最前面**（原生全量加载时 LoRA 先于 `weight_wrapper_patches` 生效，顺序保持一致）。
* `MonoloadRuntimePatch.__call__` 复刻原生 `patch_weight_to_device` 的数值路径：直接取**层上的原参数**（而不是 `cast_bias_weight` 传进来的、已转成计算 dtype 的副本），`cast_to_device(param, lora_compute_dtype)` → `calculate_weight(patches, w, key)`（intermediate 默认 fp32）→ `stochastic_rounding` 回参数 dtype（同样的 seed）→ 转成计算 dtype 返回。于是结果与原生「烘焙」路径**逐位一致**（测试实测 max_abs = 0），而不是原生 lowvram 路径那种在计算 dtype 下算 LoRA 的近似。
* 部分加载（非 `--gpu-only`、显存不够）时原生 `load()`/`partially_unload()` 会给 lowvram 层挂普通 `LowVramPatch`；Monoload 在它们之后把这些换成 `MonoloadRuntimePatch`，数值同样与烘焙一致。
* 因为 `cast_bias_weight` 每层用完临时权重即丢，同一时刻只有正在计算的那一层有临时权重。
* 走的是通用的 `calculate_weight`，所以 LoRA/LoHa/LoKr/GLoRA/OFT/BOFT、diff、set 等 ComfyUI 支持的所有类型都一样支持。原生 `LoraLoader` / `LoraLoaderModelOnly` 不用改。

配套的重写：

* `unpatch_model()`：原生只在 lowvram 时清 `weight_function`；Monoload 在卸载权重时清掉自己挂的运行时 patch，保证撤掉 LoRA 后模型与文件逐字节一致。
* `load(force_patch_weights=True)` 且有 patch（模型保存/合并「烘焙」时才会这样）→ 报错：Monoload 模型不支持把 LoRA 烘焙进权重。
* 某个被 patch 的参数不属于 `comfy.ops` 层（没有运行时路径），或 patch 会改变形状 → 报错，而不是退回备份。
* 原生 `load()` 会清空全量加载层的 `weight_function`，但跳过已标记 `comfy_patched_weights` 的层（原生里它们已烘焙）。Monoload 在 `load()` 前清掉被 patch 层的这个标记，保证重复加载时运行时 patch 一定会重新挂上。
* 加载结束后断言 `backup` / `hook_backup` 为空，出现即报内部错误。

### 4.1 Hook LoRA（v1 已实现）

原生做法：`patch_hooks(hooks)` 把当前 hook 组的 patch 写进权重（先 `hook_backup` 备份），MaxSpeed 模式还把每个 hook 组的合并结果缓存在 `cached_hook_patches`。

Monoload：
* `MonoloadModelPatcher` 有一个 `monoload_hook_state`（key → 当前生效的 hook patch 列表），所有运行时 patch 共享它。
* `patch_hooks(hooks)` 只做状态切换：用原生 `get_combined_hook_patches(hooks)` 算出组合（含 keyframe 强度），写进 `monoload_hook_state`；只被 hook 改到、还没有运行时 patch 的层补挂一个；不再生效的 hook-only patch 摘掉。**不写权重、不备份、不缓存。**
* `unpatch_hooks()` 清空状态。`patch_hook_weight_to_device` / `patch_cached_hook_weights` 若被调用即报内部错误。
* 数值：先按 §4 算基础 LoRA 并舍入到参数 dtype，再 `cast_to_device(w, float32)` → `calculate_weight(hook_patches, w, key, original_weights={key: [(原参数, identity)] + 基础 patch})` → `stochastic_rounding`。与原生 `patch_hook_weight_to_device`（在已烘焙基础 LoRA 的权重上转 fp32 再算）一致；测试中带 keyframe 的 hook LoRA 与原生逐位一致。
* 采样时正/负条件可能挂不同的 hook 组，每一步会来回切换；在 Monoload 里这只是换一个 dict，代价可以忽略。
* `return_weight=True`（只读取合并结果、不写回）仍交给原生实现，不产生备份。

## 5. 与其他加载路径的关系

Monoload 不调用 `comfy.sd.load_diffusion_model`、`load_torch_file`、`Module.load_state_dict`，不改任何全局函数（转换器里的观测包装只存在于转换进程、只在那一次调用期间）；ComfyUI 服务里只注册模型目录 `monoload` 和节点 `MonoloadUNETLoader`。原生 `UNETLoader` 等照常可用。

## 5.1 文本编码器和 VAE：v1 继续走 ComfyUI 原生加载

决定：v1 只转换扩散模型，TE/VAE 留到下一版。理由：

1. **路径差异大、测试矩阵大。** TE 走 `load_clip`：类型由用户在 `CLIPLoader` 里选（`krea2`、`qwen_image`…），可能多文件合并，还带 tokenizer 和 `CLIP` 包装对象；VAE 的构造函数按几十种 key 模式分支建子模型。每个组件都要单独写一套「采集 + 重建」和对应的等价测试，放进 v1 会显著推迟扩散模型这条主线。
2. **收益相对小。** 目标配置里最大的是扩散模型（Krea 2 为 26G）；`qwen3vl_4b` bf16 约 8G，VAE 不到 1G。原生加载 TE 时 CPU 上短暂有一份完整 dict（约 8G）+ GTT 里一份，CT 内存可以按这个量设。
3. **格式和搬运层已经为它们留好位置。** `monoload.component` 字段、`fmt`/`transfer` 模块与组件无关，下一版只需加 `text_encoder`、`vae` 两套采集/重建。

## 5.2 「单份权重」的口径

* 允许：LoRA 文件本身常驻内存；正在计算的那一层短暂多出临时副本（含 LoRA 按 fp32/lora_compute_dtype 计算的中间量）；搬运缓冲（默认 2×512MiB）。
* 不允许：整个模型、或一批层同时存在原权重和合并结果两份；任何形式的备份（`backup`、`hook_backup`、`cached_hook_patches`）。

## 6. 以后的扩展点

* `component`：`text_encoder` / `vae` 各自一套「采集 + 重建」即可，文件格式和搬运层通用。
* `quant`：量化张量（fp8 scaled 的 scale、GGUF 块）作为额外张量写入，重建时由对应 ops 接管；v1 碰到量化直接拒绝。
* O_DIRECT：搬运层已经按大块顺序读，届时只需在张量间补齐对齐（需要格式版本 2）。
