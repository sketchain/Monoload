# Monoload 设计说明（v2：运行时 LoRA 合并）

参考源码：锁定镜像 `kyuz0/amd-strix-halo-comfyui@sha256:384aa1fe…` 里的 `/opt/ComfyUI`（ComfyUI 0.31.0，commit `62b3c94b`）。下文提到的函数和行为都按这份源码核实过。

## 0. 转型

v1（tag `v1-converter`，远端归档分支 `archive/v1-converter`）做的是「离线转换 + pread 直读」，为的是绕开 mmap 的慢和卡死。后来查到根因在 ROCm 的 rocclr：H2D 拷贝超过 1 MiB 时，rocclr 会临时 pin 源内存，在 APU 上这一步走 KFD HMM，逐页处理。上游 ROCm/clr `3ccb59f`（2026-09-23）已经改成「统一内存设备上不做 pin」；修复进入正式版之前，设 `GPU_PINNED_MIN_XFER_SIZE=65536` 效果等价。于是原生加载改成「不开 `--disable-mmap` + 这个环境变量」就足够快，v1 那套不再需要。

v2 只做一件事：**在 ComfyUI 原生加载出来的模型上打 LoRA 时，不改原权重、不做任何备份，而是在每一层计算的那一刻临时合并。** 对工作流透明：原生 `UNETLoader`、`CheckpointLoaderSimple`、`CLIPLoader`、`LoraLoader`、`LoraLoaderModelOnly`、Hook LoRA 节点照常使用。

「单份」的口径不变：

* 允许：LoRA 文件本身常驻内存；LoRA 张量在计算设备上的一份副本（见 §5.2）；正在计算的那一层短暂多出的临时副本（含 LoRA 按 fp32 计算的中间量）。
* 不允许：整个模型、或一批层同时存在原权重和合并结果两份；任何形式的备份（`backup`、`hook_backup`、`cached_hook_patches`）。

## 1. 原生是怎么打 LoRA 的

* `LoraLoader` → `comfy.sd.load_lora_for_models` → `ModelPatcher.add_patches()`：只记录 patch（key → `(strength, adapter, strength_model, offset, function)` 列表），不碰权重。MODEL 和 CLIP（`clip.patcher`）各有一个 `ModelPatcher`。
* 真正改权重在加载时：`model_management.load_models_gpu` → `LoadedModel.model_load` → `ModelPatcher.patch_model` → `load()`。全量加载时，`load()` 对每个参数调用 `patch_weight_to_device(key)`：先把原权重放进 `self.backup`，再用 `comfy.lora.calculate_weight` 算出合并结果，**原地替换参数**。
* 切换 LoRA 组合：新的克隆和旧的共享同一个 `model`。换组合时 `unpatch_model()` 把备份写回，新组合再重新合并一遍。
* Hook LoRA：`patch_hooks(hooks)` 把当前 hook 组合并进权重（先写 `hook_backup`）；MaxSpeed 模式还会把每个 hook 组的合并结果缓存进 `cached_hook_patches`。
* lowvram（部分加载）时，被卸到 CPU 的层不合并，而是在 `weight_function` 上挂 `LowVramPatch`：`comfy.ops` 的层在 `forward` 里只要 `weight_function` 非空就走 `cast_bias_weight`，先 `copy=True` 拷一份临时权重，再依次调用这些函数。**这条运行时路径是 Monoload 复用的机制。**

## 2. 挂载方式：替换 `ModelPatcher` 类上的方法

`install()`（插件被 ComfyUI 导入时执行）直接在 `comfy.model_patcher.ModelPatcher` **这个类**上替换 8 个方法，原函数保存起来，由替换函数在需要时调用：

| 方法 | Monoload 版本做什么 |
|---|---|
| `patch_weight_to_device` | 对有 patch 的 key：**不备份、不改权重**，把 `MonoloadRuntimePatch` 插到该层 `weight_function` / `bias_function` 的最前面。没有 patch 的 key，以及只取合并结果、不写回的 `return_weight=True`，交给原函数。 |
| `load` | 检查 DynamicVRAM 和 `force_patch_weights`，清掉被 patch 层的 `comfy_patched_weights` 标记（见 §4），调用原 `load()`，最后断言没有产生备份。 |
| `partially_unload` | 调用原函数后，去掉被原生 `LowVramPatch` 取代的那些运行时 patch（见 §4）。 |
| `unpatch_model` | 调用原函数；卸载权重时摘掉所有运行时 patch，清空设备上的 LoRA 缓存。 |
| `patch_hooks` / `unpatch_hooks` | 只切换「当前生效的 hook patch」这个状态，不写权重（§3.2）。 |
| `patch_hook_weight_to_device` / `patch_cached_hook_weights` | 不应再被调用，被调用即报内部错误。 |

为什么选这种方式：

* **覆盖面正好。** ComfyUI 里所有模型包装都是 `ModelPatcher` 的实例：`UNETLoader`、`CheckpointLoaderSimple` 的 MODEL、`CLIP.patcher`（文本编码器）、VAE、ControlNet 等。`CoreModelPatcher` 在没开 DynamicVRAM 时就是 `ModelPatcher` 本身。方法在类上查找，所以装上之前已经建好的实例、以及没有重写这些方法的子类也都生效。
* **侵入小。** 不替换加载函数，不换类，不包装节点，不改 `comfy.ops` 和 `cast_bias_weight`，也不修改 ComfyUI 源码。工作流里的节点都不用换。
* **时机确定。** ComfyUI 在启动时（`init_extra_nodes`）导入 custom node，这时还没有执行任何 prompt，也就没有加载任何模型。
* **可以关。** 设了 `MONOLOAD_DISABLE=1`，插件不调用 `install()`，行为与原生完全一致。`uninstall()` 恢复原方法（测试里用来在同一进程中对比原生和 Monoload；调用前要先卸载所有模型）。

子类的处理：

* 自己重写了 `patch_weight_to_device` 的子类（例如 ComfyUI-GGUF 的 `GGUFModelPatcher`，它对量化权重有自己的一套 patch 机制，而且 `load` 时强制 `force_patch_weights=True`）不归 Monoload 管：所有替换函数先判断 `type(self).patch_weight_to_device` 是否还是 Monoload 的版本，不是就原样调用原函数，并对这个类打一次警告日志。
* `ModelPatcherDynamic`（DynamicVRAM）重写了 `load`，所以它的 `load` 也被包了一层，只做 DynamicVRAM 检查（§6）。

## 3. 运行时合并

### 3.1 `MonoloadRuntimePatch`

它是 `LowVramPatch` 的子类，挂在层的 `weight_function` 上。`cast_bias_weight` 在该层计算时给它一份**私有的**临时权重（因为存在 weight function，`cast_to(..., copy=True)`，再转成计算 dtype），它返回合并后的权重，用完即丢。同一时刻只有正在计算的那一层有临时权重。

数值上复刻原生的合并路径，所以结果与原生**逐位一致**（测试里 max_abs = 0）：

```
基础 LoRA（复刻 patch_weight_to_device）:
    W  -> lora_compute_dtype(device)          # 原生：cast_to_device(param, device, lora_dtype, copy=True)
       -> calculate_weight(patches, W, key)   # intermediate_dtype 默认 fp32
       -> stochastic_rounding(参数 dtype, seed=string_to_seed(key))
Hook LoRA（复刻 patch_hook_weight_to_device，在已合并基础 LoRA 的权重上）:
       -> float32 -> calculate_weight(hook_patches, W, key, original_weights={key: [(原参数, identity)] + 基础 patch})
       -> stochastic_rounding(参数 dtype, seed)
最后 -> 转成 cast_bias_weight 要的计算 dtype
```

插在 `weight_function` 的**最前面**：原生全量加载时 LoRA 已经合并进权重，先于 `weight_wrapper_patches` 生效，顺序保持一致。

走的是通用的 `calculate_weight`，所以 LoRA / LoCon / LoHa / LoKr / GLoRA / OFT / BOFT、diff、set 等 ComfyUI 支持的类型都能用。

### 3.2 Hook LoRA

每个 patcher 有一份 `hook_patches`（key → 当前生效的 hook patch 列表），它的所有运行时 patch 共享这一份。`patch_hooks(hooks)` 用原生的 `get_combined_hook_patches(hooks)` 算出组合（包括 keyframe 强度），写进这份状态；只被 hook 改到、还没有运行时 patch 的层补挂一个，不再生效的 hook-only patch 摘掉。**不写权重、不备份、不缓存。** 采样时正/负条件可能挂着不同的 hook 组，每一步会来回切换；在这里只是换一个 dict。CLIP 的 `SetClipHooks`（`forced_hooks`）走同一条路。

## 4. 与原生状态机的配合

* **重复加载。** 原生 `load()` 会清空所有全量加载层的 `weight_function`，但跳过已标记 `comfy_patched_weights` 的层（原生里它们已经合并好了）。Monoload 的 patch 并没有合并进权重，所以 `load()` 之前先清掉被 patch 层的这个标记，保证这些层会重新走一遍 `patch_weight_to_device`。
* **切换组合 / 卸载。** 原生 `unpatch_model()` 只在 lowvram 时清 `weight_function`；Monoload 在卸载权重时摘掉自己挂的所有运行时 patch。所以撤掉 LoRA 后，权重与加载时逐字节一致（本来也从未改过），也没有残留的 weight function。
* **部分加载（非 `--gpu-only`、显存不够）。** 原生对被卸载的层本来就用 `LowVramPatch`，而且不备份。Monoload **不改这部分**，只接管原生会合并进权重的那些层，所以在部分加载下结果也和原生逐位一致。原生 `partially_unload()` 会给已经合并过的层追加 `LowVramPatch`（原生里是先写回备份）；这时同一层会同时挂着 Monoload 的 patch 和原生的 `LowVramPatch`，Monoload 摘掉自己那个，得到的结果和原生一样。
* 每次 `load()` / `partially_unload()` 之后都断言 `backup` / `hook_backup` 为空。

## 5. 效率

### 5.1 代价在哪

原生是「每次加载合并一次，之后每步零开销」；运行时合并是「每步、每个被 patch 的层都合并一次」。没被 patch 的层完全不受影响：`weight_function` 为空，`forward` 走原来的快路径。被 patch 的层每次计算要多做这些事：

1. `cast_bias_weight` 拷一份临时权重（有 weight function 时它必须 `copy=True`）；
2. 如果 `lora_compute_dtype` 和权重 dtype 不同，就要做一次转换，最后再转回来（逐位一致需要）；
3. `calculate_weight`：LoRA 低秩矩阵乘（intermediate 是 fp32），再加回权重；
4. `stochastic_rounding` 回到参数 dtype（bf16/fp16 时就是一次 `.to()`），再转成计算 dtype（相同时不做任何事）。

### 5.2 已经省掉的

* **不再从参数重新读一遍。** `cast_bias_weight` 传进来的 `weight` 已经是一份私有副本。只要它和参数逐位相同（dtype 相同，或者是 fp16/bf16→fp32 这类无损加宽），就直接用它当原生路径里的 `temp`，不再 `cast_to_device(param, …, copy=True)`。如果它的 dtype 恰好就是 `lora_compute_dtype`，就原地在它上面合并，一次额外拷贝都没有。只有计算 dtype 比参数窄的时候（有损），才退回去从参数读，以保证逐位一致。
* **LoRA 张量只搬一次。** 原生 `calculate_weight` 每次都 `cast_to_device(LoRA 张量, 权重设备)`；LoRA 文件是读到 CPU 上的，在运行时合并下就变成每步每层一次 H2D。Monoload 在每个 patcher 里按张量缓存一份计算设备上的副本（同 dtype 搬运，数值不变），卸载时释放。代价是计算设备上多一份被用到的 LoRA 张量（LoRA 大小，属于允许的范围）。
* seed（`string_to_seed(key)`）在挂 patch 时算好；基础 patch 列表按设备缓存好的结构复用；没有 patch 的 key 直接返回。

### 5.3 实测开销（CPU）与可选的放宽（未实现，等拍板）

**CPU 参考数字**（锁定镜像，`--cpu --fp16-unet`，SD1.5，256×256，3 步，`tests/bench_lora.py`，第二遍的数字）。这台机器是 4 核 CPU，256px 下模型本身的计算量很小，所以合并开销显得特别突出，**不代表 GPU**：

| 组合（UNet 被 patch 的层） | 每步 原生 | 每步 Monoload | 比值 | 切换组合（patch）原生 | Monoload | 结果差异 |
|---|---|---|---|---|---|---|
| Rubber Duck（192 层，0.50 GiB） | 7.13s | 15.74s | 2.21× | 6.57s | 0.08s | 0（逐位一致） |
| Annalise LoCon（278 层） | 9.13s | 24.23s | 2.65× | 12.55s | 0.08s | 0 |
| 两个叠加（278 层） | 9.37s | 26.56s | 2.83× | 16.64s | 0.08s | 0 |
| 不打 LoRA | 7.85s | 8.34s | 1.06× | 0.05s | 0.06s | 0 |

layer probe（Rubber Duck，一次模型调用、全部 192 层合计）：临时拷贝 0.075s，逐位一致合并 8.19s，放宽合并（方案 A）6.07s。最大的一层（1280→10240，fp16）拆开来看：`calculate_weight` 64ms（其中低秩矩阵乘 17ms，其余是几遍逐元素运算），舍入回 fp16 再转回 fp32 19ms，临时拷贝 11ms。

结论：

1. **运行时合并的每步开销 ≈ 每次模型调用都重做一遍原生那次性的合并。** 在 GPU 上可以直接用基准脚本里原生的 `patch` 列来估算。一步通常只有一次模型调用：CFG 的正负条件在显存允许时会合批；Krea 2 的 CFG=1 时只有正条件。
2. 这部分开销主要是对被 patch 权重的几遍逐元素读写（拷贝、加上 delta、dtype 往返），低秩矩阵乘本身不贵。所以在带宽受限的 APU 上，它和「被 LoRA 改到的权重有多大」成正比，和分辨率无关。分辨率越高，模型本身的计算越多，占比就越小。
3. 逐位一致带来的额外开销（dtype 往返 + 舍入）在 CPU 上约占合并本身的 35%。GPU 上 bf16/fp16 转换很便宜，但 gfx1151 上的 `lora_compute_dtype` 是 fp16，而权重是 bf16，所以每层每步要多两遍转换。GPU 上的数字请看 layer probe。

可选的放宽方案（都**没有实现**）：

* **A. 按原生 lowvram 的数值合并：** 直接在计算 dtype 下 `calculate_weight(patches, weight, key, intermediate_dtype=计算 dtype)`，省掉 dtype 往返和舍入。与原生合并不再逐位一致，但与原生 lowvram 路径一致。CPU 上比逐位一致快约 25%。
* **B. 计算 dtype 下合并，intermediate 保持 fp32：** 省掉 dtype 往返，精度介于 A 和逐位一致之间。
* **C. 融合的 `addmm_`：** 把「低秩乘 → 缩放 → 转 dtype → 加回」合成一次 `weight.addmm_(up, down, alpha=scale)`，少几遍逐元素读写。只适用于普通 LoRA/LoCon，舍入顺序和原生不同。
* **D. bypass（低秩前向）：** `y = W·x + scale·up(down(x))`，完全不物化合并后的权重，每层的额外开销从「对整块权重做几遍逐元素运算」变成「两个很瘦的矩阵乘」（与 token 数 × rank 成正比）。在带宽受限的 APU 上，这很可能是唯一能把开销降一个量级的办法。ComfyUI 已经自带实现（`comfy/weight_adapter/bypass.py`、节点 `LoraLoaderBypass`），但数值与合并路径不同，而且不是所有 adapter 都支持。

建议先在 CT 700 上跑基准，看 GPU 上的每步比值再决定。如果 Krea 2 这类大模型的开销不可接受，A/B/C 大概只能省掉一部分，D 才可能根本解决问题。

## 6. 报错（绝不退回「改权重 + 备份」）

| 情况 | kind | 何时 |
|---|---|---|
| DynamicVRAM（comfy-aimdo）开启且模型有 LoRA/hook patch | `dynamic_vram` | 加载该模型时 |
| 要求把 patch 合并进权重（`force_patch_weights`，常见于 `ModelSave` / `CheckpointSave` / 模型合并后保存） | `force_patch_weights` | 加载时 |
| 被 patch 的参数不属于 `comfy.ops` 层（没有 `comfy_cast_weights`，没有运行时路径） | `lora_non_comfy_ops_param` | 挂 patch 时 |
| patch 会改变权重形状 | `lora_shape_change` | 挂 patch 时 |
| 被 patch 的参数是量化张量（有 `set_*`/`convert_*`，例如 fp8 scaled） | `lora_quantized_param` | 挂 patch 时 |

报错信息里写明是哪种情况和对应的 key，例如：

```
[Monoload] 不支持（lora_shape_change） key=diffusion_model.input_blocks.1.1.proj_in.weight: patch 会把权重形状从 [320, 320, 1, 1] 改成 [328, 320, 1, 1]，运行时合并无法支持
```

最后一种（量化参数）是这次新增的：原生加载的模型可能带 fp8 scaled 之类的量化层，原生会把合并结果重新量化后写回，并且备份。要做到运行时逐位一致，得复刻「反量化 → 合并 → 以同样的 seed 重新量化 → 再反量化」，v2 先不做，明确报错。需要时设 `MONOLOAD_DISABLE=1`。

## 7. 限制

* 只接管没有重写 `patch_weight_to_device` 的 `ModelPatcher`（ComfyUI 自带的加载器都属于这种）。GGUF 等自带 patch 机制的插件保持原生行为，日志里会提示。
* 每步都有合并开销（§5）。LoRA 越多、改的层越大，开销越明显；没被 patch 的层没有影响。
* LoRA 文件由原生 `LoraLoader` 读取，读进来之后常驻 CPU 内存（原生也是这样）；计算设备上另有一份被用到的 LoRA 张量缓存。
* 本仓库的测试都在无 GPU 的机器上用 `--cpu` 跑；GPU 上的数值一致性和耗时要按 README 的真机验收步骤确认。
