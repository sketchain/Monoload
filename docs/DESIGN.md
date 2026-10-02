# Monoload 设计说明（v2：运行时 LoRA 合并）

参考源码：锁定镜像 `kyuz0/amd-strix-halo-comfyui@sha256:384aa1fe…` 里的 `/opt/ComfyUI`（ComfyUI 0.31.0，commit `62b3c94b`）。下文提到的函数和行为都按这份源码核实过。

## 0. 转型

v1（tag `v1-converter`，远端归档分支 `archive/v1-converter`）做的是「离线转换 + pread 直读」，为的是绕开 mmap 的慢和卡死。后来查到根因在 ROCm 的 rocclr：H2D 拷贝超过 1 MiB 时，rocclr 会临时 pin 源内存，在 APU 上这一步走 KFD HMM，逐页处理。上游 ROCm/clr `3ccb59f`（2026-09-23）已经改成「统一内存设备上不做 pin」；修复进入正式版之前，设 `GPU_PINNED_MIN_XFER_SIZE=65536` 效果等价。于是原生加载改成「不开 `--disable-mmap` + 这个环境变量」就足够快，v1 那套不再需要。

v2 只做一件事：**在 ComfyUI 原生加载出来的模型上打 LoRA 时，不改原权重、不做任何备份，而是在每一层计算的那一刻临时合并。** 对工作流透明：原生 `UNETLoader`、`CheckpointLoaderSimple`、`CLIPLoader`、`LoraLoader`、`LoraLoaderModelOnly`、Hook LoRA 节点照常使用。

在此基础上还有三点：

* **合并有两条路径（§3.1）。** 默认路径：普通 LoRA / LoCon 用一次融合的 `addmm_` 直接加进计算 dtype 的临时权重（方案 C），其他类型在计算 dtype 下走 `calculate_weight`（方案 A，与原生 lowvram 的数值相同）。这条路径与原生合并不逐位一致，误差界见 §5.5。设 `MONOLOAD_EXACT=1` 时走逐位一致路径，结果与原生完全相同。
* 量化层（fp8 scaled 等）上的 LoRA 合并在反量化出来的临时权重上，不再重新量化（§3.3）。
* 每个 prompt 结束后释放 LoRA，底模继续常驻（§7）。

「单份」的口径不变：

* 允许：LoRA 文件本身常驻内存；LoRA 张量在计算设备上的一份副本（见 §5.2）；正在计算的那一层短暂多出的临时副本（逐位一致路径下还包括 LoRA 按 fp32 计算的中间量）。
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

合并有两条路径，由 `MONOLOAD_EXACT` 选择。插件导入时读取这个环境变量；测试和基准脚本用 `hotpatch.set_exact()` 在进程内切换，下一次层计算就生效，不需要重新加载。

#### 3.1.1 默认路径：融合 addmm（C），其余类型放宽合并（A）

直接在 `cast_bias_weight` 给的那份计算 dtype 临时权重上**原地**合并，patch 逐个处理：

```
对 key 的每个 patch (strength, adapter, strength_model, offset, function)：
  普通 LoRA / LoCon（LoRAAdapter，没有 Tucker mid、DoRA、reshape；offset、function 为空；strength_model == 1）:
      C: W.view(out, -1).addmm_(up.flatten(1), down.flatten(1), alpha=strength * alpha / rank)
         # 计算 dtype 下一个 GEMM，累加和加回在同一个 kernel 里完成，不产生整层大小的中间量
  其他（LoHa、LoKr、DoRA、diff、模型合并、带 offset/function/strength_model 的 patch ……）:
      A: W = calculate_weight([patch], W, key, intermediate_dtype=W.dtype)
         # 与原生 lowvram 的 LowVramPatch 完全相同的算法
Hook LoRA：在基础 patch 之后，同样逐个 patch 处理，A 带上 original_weights（与原生 hook 路径相同的参数）
```

* `calculate_weight` 本来就是对 patch 列表逐个独立处理的，所以 C 和 A 按 patch 混用，运算顺序和原生一致。一个 key 上全是不能融合的 patch 时，结果与原生 lowvram 的 `LowVramPatch` **逐位一致**（测试里对照）。
* 不做 `lora_compute_dtype` 的往返，也不做 `stochastic_rounding`。临时权重本身就是计算 dtype，合并完直接用。
* 计算 dtype 不是 fp32/fp16/bf16 时（实际上不会出现），先转成 fp32 合并，最后再转回去。
* 量化参数：反量化 dtype 与计算 dtype 相同时，直接用 `cast_bias_weight` 反量化出来的临时权重；不同时，从参数重新反量化再转成计算 dtype（等于「先反量化好的模型」交给 weight function 的那份）。

与原生的差异只来自计算 dtype 下的舍入，误差界和真机数据见 §5.5。

#### 3.1.2 `MONOLOAD_EXACT=1`：逐位一致路径

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

两条路径都插在 `weight_function` 的**最前面**：原生全量加载时 LoRA 已经合并进权重，先于 `weight_wrapper_patches` 生效，顺序保持一致。

两条路径都以通用的 `calculate_weight` 为基础（默认路径只把普通 LoRA/LoCon 换成等价的融合运算），所以 LoRA / LoCon / LoHa / LoKr / GLoRA / OFT / BOFT、diff、set 等 ComfyUI 支持的类型都能用。

### 3.2 Hook LoRA

每个 patcher 有一份 `hook_patches`（key → 当前生效的 hook patch 列表），它的所有运行时 patch 共享这一份。`patch_hooks(hooks)` 用原生的 `get_combined_hook_patches(hooks)` 算出组合（包括 keyframe 强度），写进这份状态；只被 hook 改到、还没有运行时 patch 的层补挂一个，不再生效的 hook-only patch 摘掉。**不写权重、不备份、不缓存。** 采样时正/负条件可能挂着不同的 hook 组，每一步会来回切换；在这里只是换一个 dict。CLIP 的 `SetClipHooks`（`forced_hooks`）走同一条路。

### 3.3 量化参数：在反量化的临时权重上合并，不重新量化

原生对 fp8 scaled 这类量化层（`mixed_precision_ops`，权重是 `QuantizedTensor`）打 LoRA 时：先反量化，合并，然后用 `set_weight` 以随机舍入**重新量化回 fp8** 写回，并且备份原权重。

Monoload 的做法：量化层挂上 weight function 后，`forward` 会走 `cast_bias_weight`，先拷一份量化张量，转成计算 dtype，再 `dequantize()`，然后才交给 weight function。Monoload 直接在这份**反量化出来的临时权重**上合并 LoRA，用完就丢，**不再量化回 fp8**。它把「参数 dtype」当作反量化后的 dtype（`QuantizedTensor.dtype`），其余步骤与当前的合并路径（§3.1.1 或 §3.1.2）完全相同。计算 dtype 与反量化 dtype 不同时，从参数重新反量化一次，而不是复用临时权重。这一节与 `MONOLOAD_EXACT` 无关：两条路径都不重新量化。

* **速度：** gfx1151 上 `supports_fp8_compute()` 为 False（`torch._scaled_mm` 只支持 MI300+），原生对 fp8 模型本来就是每次 forward 先反量化再算，所以放宽合并不会带来额外的速度损失。
* **精度：** 少了一次 fp8 重新量化，比原生更精确，但与原生**不逐位一致**。
* **正确性的判定标准：** 与「先把这些层反量化成高精度参数，再走同一条合并路径」逐位一致。两种模式下都这样测。
* **一个实测发现：** 没挂 weight function 的 fp8 层，在原生 ComfyUI 里根本不反量化，而是把 `QuantizedTensor` 直接交给 `F.linear`，由 comfy_kitchen 的算子计算，运算顺序与「先反量化再乘」不同。所以把整个模型都反量化，即使不打 LoRA，结果也和 fp8 模型不一样（CPU 上 latent 的 max_abs 约 3e-4）。参照模型因此只反量化**被 LoRA 改到的那些层**，其余层保持 fp8，这与 Monoload 实际做的事一一对应。

实测（`tests/test_quant.py`，SD1.5 UNet 的 184 个 Linear 转成 fp8 scaled，CPU，采样 2 步，`MONOLOAD_EXACT=1`）：

| LoRA | 被 patch 的 key | 其中 fp8 层 | 与「先反量化」参照 | 与原生 fp8 LoRA（合并 + 重新量化 + 备份） |
|---|---|---|---|---|
| Rubber Duck | 192 | 160 | 逐位一致 | max_abs 4.76，mean_abs 0.70（LoRA 本身的影响 max_abs 34.2） |
| Annalise LoCon | 278 | 182 | 逐位一致 | max_abs 1.88，mean_abs 0.34（LoRA 本身的影响 max_abs 26.4） |

与原生的差异来自原生的 fp8 重新量化，只报告，不作判定。另外 dtype 矩阵测试里，对 fp8 参数的 54 种组合（反量化 dtype × 计算 dtype × lora dtype × 基础 / hook）逐一对照「先反量化」参照，全部逐位一致。

## 4. 与原生状态机的配合

* **重复加载。** 原生 `load()` 会清空所有全量加载层的 `weight_function`，但跳过已标记 `comfy_patched_weights` 的层（原生里它们已经合并好了）。Monoload 的 patch 并没有合并进权重，所以 `load()` 之前先清掉被 patch 层的这个标记，保证这些层会重新走一遍 `patch_weight_to_device`。
* **切换组合 / 卸载。** 原生 `unpatch_model()` 只在 lowvram 时清 `weight_function`；Monoload 在卸载权重时摘掉自己挂的所有运行时 patch。所以撤掉 LoRA 后，权重与加载时逐字节一致（本来也从未改过），也没有残留的 weight function。
* **部分加载（非 `--gpu-only`、显存不够）。** 原生对被卸载的层本来就用 `LowVramPatch`，而且不备份。Monoload **不改这部分**，只接管原生会合并进权重的那些层，所以在部分加载下结果也和原生逐位一致。原生 `partially_unload()` 会给已经合并过的层追加 `LowVramPatch`（原生里是先写回备份）；这时同一层会同时挂着 Monoload 的 patch 和原生的 `LowVramPatch`，Monoload 摘掉自己那个，得到的结果和原生一样。
* 每次 `load()` / `partially_unload()` 之后都断言 `backup` / `hook_backup` 为空。

## 5. 效率

### 5.1 代价在哪

原生是「每次加载合并一次，之后每步零开销」；运行时合并是「每步、每个被 patch 的层都合并一次」。没被 patch 的层完全不受影响：`weight_function` 为空，`forward` 走原来的快路径。被 patch 的层在逐位一致路径（`MONOLOAD_EXACT=1`）下每次计算要多做这些事：

1. `cast_bias_weight` 拷一份临时权重（有 weight function 时它必须 `copy=True`）；
2. 如果 `lora_compute_dtype` 和权重 dtype 不同，就要做一次转换，最后再转回来（逐位一致需要）；
3. `calculate_weight`：LoRA 低秩矩阵乘（intermediate 是 fp32），再加回权重；
4. `stochastic_rounding` 回到参数 dtype（bf16/fp16 时就是一次 `.to()`），再转成计算 dtype（相同时不做任何事）。

默认路径（§3.1.1）只剩第 1 步的拷贝，加上每个普通 LoRA patch 一次 `addmm_`（读一遍、写一遍权重）。

### 5.2 已经省掉的

* **不再从参数重新读一遍。** `cast_bias_weight` 传进来的 `weight` 已经是一份私有副本。只要它和参数逐位相同（dtype 相同，或者是 fp16/bf16→fp32 这类无损加宽），就直接用它当原生路径里的 `temp`，不再 `cast_to_device(param, …, copy=True)`。如果它的 dtype 恰好就是 `lora_compute_dtype`，就原地在它上面合并，一次额外拷贝都没有。只有计算 dtype 比参数窄的时候（有损），才退回去从参数读，以保证逐位一致。
* **LoRA 张量只搬一次。** 原生 `calculate_weight` 每次都 `cast_to_device(LoRA 张量, 权重设备)`。LoRA 文件读在 CPU 上，放在运行时合并里，就成了每步每层一次 H2D。Monoload 在每个 patcher 里按张量缓存一份计算设备上的副本（同 dtype 搬运，数值不变），卸载时释放。**只缓存比被 patch 的权重小的张量**（LoRA 的低秩因子、alpha 等）；整层大小的 patch 张量（例如 `ModelMergeSimple` 等模型合并节点带进来的另一个模型的权重、完整的 diff）照原生的做法，每次临时搬运、用完即丢，所以不会让另一份模型常驻计算设备。缓存的代价是计算设备上多一份被用到的 LoRA 张量，大小不超过 LoRA 本身，在允许的范围内。
* seed（`string_to_seed(key)`）在挂 patch 时算好；基础 patch 列表按设备缓存好的结构复用；没有 patch 的 key 直接返回。

### 5.3 实测开销（CPU，逐位一致路径）与放宽方案

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

`bypass` 对照（同一台 CPU、同样设置、机器空闲时重跑；Monoload 的数字也是这一轮的）：

| 组合 | 每步 原生 | Monoload（逐位一致） | ComfyUI bypass | bypass 与原生的 latent max\|Δ\| |
|---|---|---|---|---|
| Rubber Duck | 8.56s | 16.90s（1.97×） | 10.27s（1.20×） | 0.033 |
| Annalise LoCon | 9.10s | 24.60s（2.70×） | 13.54s（1.49×） | 0.155 |
| 不打 LoRA | 8.40s | 9.09s | 8.96s | 0 |

（作为参照，这两个 LoRA 本身对 latent 的影响 max_abs 约 34。）

当时列出的放宽方案（这些是逐位一致路径时期的分析；C 和作为后备的 A 后来成了默认路径，见 §5.4）：

* **A. 按原生 lowvram 的数值合并：** 直接在计算 dtype 下 `calculate_weight(patches, weight, key, intermediate_dtype=计算 dtype)`，省掉 dtype 往返和舍入。与原生合并不再逐位一致，但与原生 lowvram 路径一致。CPU 上比逐位一致快约 25%。
* **B. 计算 dtype 下合并，intermediate 保持 fp32：** 省掉 dtype 往返，精度介于 A 和逐位一致之间。
* **C. 融合的 `addmm_`：** 把「低秩乘 → 缩放 → 转 dtype → 加回」合成一次 `weight.addmm_(up, down, alpha=scale)`，少几遍逐元素读写。只适用于普通 LoRA/LoCon，舍入顺序和原生不同。
* **D. bypass（低秩前向）：** `y = W·x + scale·up(down(x))`，完全不物化合并后的权重，每层的额外开销从「对整块权重做几遍逐元素运算」变成「两个很瘦的矩阵乘」（与 token 数 × rank 成正比）。ComfyUI 已经自带实现（`comfy/weight_adapter/bypass.py`、节点 `LoraLoaderBypass`），但数值与合并路径不同，不是所有 adapter 都支持，而且连续调用 `load_bypass_lora_for_models` 叠加多个 LoRA 时，后一个会覆盖前一个（它们的注入都存在同一个 key `bypass_lora` 下）。GPU 实测见 §5.4。

`tests/bench_lora.py` 带 `bypass` 模式，直接调用 ComfyUI 自带的 `comfy.sd.load_bypass_lora_for_models`，只提供对照数据，设计上不做改动。

### 5.4 真机数据（CT 700，gfx1151）与决定：默认用 C

**实测**（WAI v17，SDXL 整合包，1344×768，20 步，CFG 6，`--gpu-only`；`bench_sdxl_v2`，第二遍的数字）。权重、计算 dtype、`lora_compute_dtype` 都是 fp16；`native2 vs native` 全部逐位一致，说明 GPU 计算是确定的。

| 组合 | 原生每步 | Monoload 逐位一致 | ComfyUI bypass | 切换组合（patch）原生 → Monoload | GTT 原生 → Monoload |
|---|---|---|---|---|---|
| Smooth Booster（788 层 UNet） | 0.638s | 0.891s（1.40×） | 0.870s（1.36×） | 0.42s → 0.10s | 13.9G → 8.6G |
| S1 Dramatic Lighting（722 UNet + 264 TE） | 0.640s | 0.855s（1.34×） | 0.799s（1.25×） | 0.66s → 0.11s | |
| 两者叠加 | 0.643s | 1.064s（1.66×） | 1.024s（1.59×） | 0.62s → 0.11s | |

（这一轮的 bypass 叠加已经修正为两个 LoRA 都挂上。逐位一致路径与原生的 latent max|Δ| 全部为 0。）

layer probe（Smooth Booster，一次模型调用、788 层、4.77 GiB）：

| | 耗时（不含临时拷贝 0.038s） | 与原生合并的权重差异 ‖Δw‖/‖ΔW_lora‖ | max\|Δw\| |
|---|---|---|---|
| 逐位一致 | 0.221s | 0 | 0 |
| A. 放宽合并（`calculate_weight`，fp16 intermediate） | 0.117s | 3.56e-5 | 6.1e-5 |
| C. 融合 addmm（788 层全部可融合） | 0.040s | 2.54e-3 | 1.22e-4 |

0.038 + 0.221 = 0.259s，基本等于实测每步多出的 0.25s：CFG 的正负条件合成一批，每步只有一次模型调用，开销就是一次完整的合并。逐位一致路径的开销主要来自 `calculate_weight` 默认的 fp32 intermediate：生成整层大小的 fp32 delta，乘系数、转回 fp16、再加回权重，对 4.77 GiB 的权重做好几遍逐元素读写。C 只剩一次 GEMM 带加回，耗时和一次拷贝相当。

**决定（2026-10）：** 默认采用 C；C 不支持的类型用 A；设 `MONOLOAD_EXACT=1` 回到逐位一致路径（§3.1）。D（bypass）不采用：在这个分辨率下不比 A 快，token 越多越慢，ComfyUI 自带实现叠加多个 LoRA 时只保留最后一个，也不是所有 adapter 都支持。

**实测（`bench_sdxl_v3`，同样设置，第二遍）：**

| 组合 | 原生 | 默认路径（C/A） | 逐位一致（`MONOLOAD_EXACT=1`） | native2 |
|---|---|---|---|---|
| Smooth Booster | 0.639s | 0.723s（1.13×） | 0.891s（1.39×） | 0.645s |
| S1 Dramatic Lighting | 0.645s | 0.710s（1.10×） | 0.855s（1.33×） | 0.649s |
| 两者叠加 | 0.640s | 0.758s（1.18×） | 1.065s（1.67×） | 0.648s |
| 不打 LoRA | 0.644s | 0.643s | 0.647s | 0.647s |

与预估（约 0.72s / 1.12×，叠加约 1.2×）一致；`bench_sdxl_v4` 复测为 1.13× / 1.11× / 1.19×。layer probe 里默认路径的合并是 0.052s（+ 拷贝 0.038s），比上一轮单独测的 `addmm_`（0.040s）多 12ms，这是每层的 Python 开销（判断能否融合、取因子），约 15 µs/层。GTT 8.4G（原生 13.9G）。

### 5.5 默认路径与原生的精度差异、容差测试

**差异从哪来。** 原生（= 逐位一致路径）先把 delta 用 fp32 算好，W + delta 在 fp32 下求和，最后舍入一次回 fp16。C 在 fp16 下做 `addmm_`：低秩乘的累加和加回在 GEMM 内部完成，再舍入成 fp16。真实的和若落在两个 fp16 值的中点附近，只要累加顺序或中间精度稍有不同，就会被舍入到相邻的那个值。所以差异的形态是：个别元素差 1 个 ulp。CT 700 上 max|Δw| = 1.22e-4，正好是 [0.125, 0.25) 区间内 fp16 的 1 个 ulp；A 的 6.1e-5 是 [0.0625, 0.125) 的 1 个 ulp。C 的差异总量比 A 大约 70 倍。

* `bench_sdxl_v3` 里关掉 `allow_fp16_reduced_precision_reduction` 后，耗时和误差都完全不变（0.052s，2.54e-3），所以原因不是 PyTorch 的这个开关。
* `bench_sdxl_v4` 的 layer probe 测了 `mm-add`：先 `torch.mm` 出 fp16 的 delta，再 `add_(delta, alpha=scale)`，两个 kernel。它的权重误差与 A 完全相同（3.56e-5，max 6.1e-5），合并 0.084s（C 0.052s，A 0.118s）。所以 C 多出来的误差来自 hipBLASLt `addmm_`（beta=1）内部的累加或 epilogue，拆开写就没有了。
* `mm-add` 不采用：每步多约 5%，出图差异只能降到 A 的水平（见下），类别不变。

**指标（与 layer probe 相同）：**

```
rel = ‖Δw‖ / ‖ΔW_lora‖        Δw = w_default − w_native，ΔW_lora = w_native − w_base
```

对一个模型所有被 patch 的权重整体求和。w_native 是原生 `patch_weight_to_device(return_weight=True)` 的结果，也就是原生烘焙进去的权重。

**比较精度。** 两边先舍入到参数 dtype、计算 dtype、`lora_compute_dtype` 三者中最粗的那个，再比较。原生合并的精度受这三者里最粗的那个限制：例如 CPU 测试里是 fp16 参数、fp32 计算，原生会把合并结果舍入回 fp16，默认路径则在 fp32 下合并、不舍入，比原生更精确，这部分不算误差。gfx1151 上三者都是 fp16，等于直接比较。

**阈值：**

```
‖Δw‖ ≤ u · (1 · ‖W‖ + 20 · ‖ΔW_lora‖)，即  rel ≤ u · (‖W‖ / ‖ΔW_lora‖ + 20)
u = 比较精度的单位舍入：fp16 2⁻¹¹ ≈ 4.9e-4，bf16 2⁻⁸ ≈ 3.9e-3，fp32 2⁻²⁴
```

* 第一项是最终舍入：合并时求和顺序不同，就可能落到相邻的 ulp。每个变了的元素差 1 个 ulp（≤ 2u·|w|），所以整体 ‖Δw‖ ≤ u·‖W‖，相当于最多约四分之一的元素各差 1 个 ulp。
* 第二项是在计算 dtype 下算 delta 带来的误差，按 delta 本身的 20 个单位舍入计。
* 不用固定的 rel 阈值：rel 的下限就是 W 的舍入，LoRA 改动相对 W 越小，rel 就越大。纯舍入就可能让 LoHa 这种改动极小的 patch 得到 rel ≈ 1。
* gfx1151 上 C 的 2.54e-3，只占第二项（20u = 9.8e-3）的约四分之一，与 ‖W‖/‖ΔW_lora‖ 无关，都在阈值以内。Smooth Booster 实测 ‖W‖/‖ΔW_lora‖ = 44.3，阈值 0.0308；换算成 u·‖W‖ 是 0.117（A 是 0.0017）。

**出图（latent）层面的差异：来自采样放大（已确认）。** 权重只差到 1 个 ulp，但 SDXL 20 步 euler、CFG 6 采样下来，最终 latent 与原生的差异并不小。数据来自 `bench_sdxl_v3` / `bench_sdxl_v4`，两轮一位不差：

| 组合 | 默认路径（C）vs 原生 max\|Δ\| / mean\|Δ\| | 只用放宽合并（A）vs 原生 | A / C（mean） | LoRA 本身的作用 max / mean | 参照：bypass vs 原生（v2） |
|---|---|---|---|---|---|
| Smooth Booster | 21.2 / 0.556 | 17.2 / 0.363 | 65% | 30.4 / 3.81 | 23.6 / 0.524 |
| S1 Dramatic Lighting | 19.8 / 0.189 | 18.2 / 0.0768 | 41% | 32.1 / 4.71 | 19.9 / 0.149 |
| 两者叠加 | 27.9 / 0.519 | 27.2 / 0.438 | 84% | 33.7 / 4.77 | 26.5 / 0.431 |

* A 的权重误差比 C 小 70 倍（3.56e-5 对 2.54e-3），latent 差异却只小 1.2–2.5 倍。所以出图差异主要来自多步采样把微小扰动放大，而不是 C 比 A 多出来的那部分精度损失。
* A 的数值就是原生 lowvram（`LowVramPatch`）的数值，所以原生自己在 lowvram 下与全量加载相比，也是这个量级的差异。
* LoRA 的作用本身大小不变：effect 30.8/3.79 对原生 30.4/3.81。
* CPU 上采样 2 步时，同样量级的权重差异只造成 0.1%，也说明差异随采样步数放大。
* 同种子出图对比（API 直接提交，WAI v17 + Smooth Booster 1.0/1.0，seed 42，1344×768，20 步，CFG 6，euler + normal；默认路径对 `MONOLOAD_EXACT=1`，后者与原生逐位一致）：`compare_images` 97.5% 的像素不同，mean abs diff 5.29，max 255。构图和风格一致，只是细节位置有偏移，肉眼看不出画质差别。

**结论（2026-10）：** 默认路径维持 C；需要与原生出图完全一致时用 `MONOLOAD_EXACT=1`；`mm-add` 不采用。

**测试**（全部在两种模式下各跑一遍，见 `tests/run_all.sh`）：

* `MONOLOAD_EXACT=1`：原有的逐位一致测试（dtype 矩阵 108 项、两条管线的 TE 输出和 latent、hook、模型合并），结果必须逐位一致。
* 默认路径：
  * `test_dtype_paths.py`：参数 × 计算 × lora dtype × 8 种 patch 组合（LoRA、两个 LoRA、strength_model ≠ 1、LoHa、LoRA+LoHa、diff、hook、LoRA+hook），共 144 项。每项检查两件事：一是与独立实现的参照（普通 LoRA 用 `addmm_`，其余用原生 `LowVramPatch`）**逐位一致**，二是与原生合并的差异在上述阈值以内。
  * `test_lora_hot.py`：两条管线、8 个组合，UNet 和 TE 所有被 patch 的权重整体与原生合并比较，必须在阈值以内。同时在同样的 key 上重算逐位一致路径，必须与原生逐位一致。模型合并同样检查。latent 与原生的差异只报告；hook LoRA 要求 latent 的 mean|Δ| 小于 hook 本身作用的 10%。
  * `test_quant.py`：fp8 与「先反量化 + 同一路径」逐位一致。
CPU 实测（`--cpu --fp16-unet`：UNet 参数 fp16、计算 fp32，所以按 fp16 比较；SD1.5，采样 2 步，两条管线结果相同）：

| 组合 | UNet：rel / 阈值 | TE：rel / 阈值 | latent 与原生 mean\|Δ\|（LoRA 本身的作用 mean） |
|---|---|---|---|
| Rubber Duck | 7.3e-5 / 0.056 | 2.8e-4 / 0.27 | 0.0092（7.28） |
| Annalise LoCon | 2.6e-5 / 0.031 | 3.9e-5 / 0.065 | 0.0074（4.92） |
| 两者叠加 | 1.5e-5 / 0.039 | 4.7e-5 / 0.080 | 0.0081（5.95） |
| LoKr（A） | 0 / 0.016 | 0 / 0.012 | 0.0055（4.86） |
| LoHa（A） | 0 / 0.085 | 0 / 0.034 | 0.0069（0.39） |
| ModelMergeSimple（A） | 0 / 0.14（686 个 key） | | 0.012 |
| Hook LoRA | （由 dtype 矩阵覆盖） | | 0.024（7.51） |

CPU 上计算 dtype 是 fp32，默认路径在 fp32 下合并、不舍入回 fp16，比原生更精确。所以权重按 fp16 比较时几乎完全一样（LoKr / LoHa / 模型合并是 0），latent 的差异主要来自原生那次舍入回 fp16。gfx1151 上三种 dtype 都是 fp16，差异的形态见上面的 2.54e-3。

* 两种模式都要通过的行为测试：没有备份、权重逐字节不变、撤掉 LoRA 后与没打过逐位一致、报错、每个 prompt 结束后的释放（三种缓存加 KEEP）。

## 6. 报错（绝不退回「改权重 + 备份」）

| 情况 | kind | 何时 |
|---|---|---|
| DynamicVRAM（comfy-aimdo）开启且模型有 LoRA/hook patch | `dynamic_vram` | 加载该模型时 |
| 要求把 patch 合并进权重（`force_patch_weights`，常见于 `ModelSave` / `CheckpointSave` / 模型合并后保存） | `force_patch_weights` | 加载时 |
| 被 patch 的参数不属于 `comfy.ops` 层（没有 `comfy_cast_weights`，没有运行时路径） | `lora_non_comfy_ops_param` | 挂 patch 时 |
| patch 会改变权重形状 | `lora_shape_change` | 挂 patch 时 |

报错信息里写明是哪种情况和对应的 key，例如：

```
[Monoload] 不支持（lora_shape_change） key=diffusion_model.input_blocks.1.1.proj_in.weight: patch 会把权重形状从 [320, 320, 1, 1] 改成 [328, 320, 1, 1]，运行时合并无法支持
```

量化参数（fp8 scaled 等）不报错，走放宽合并，见 §3.3。

## 7. 每个 prompt 结束后释放 LoRA（底模常驻）

目的：一个工作流用完 LoRA 后，LoRA 相关的内存（CPU 和 GPU）全部还回去，底模（Checkpoint / UNET / CLIP 加载节点的输出）原样留在内存和执行缓存里，下一个工作流直接复用，不重新加载。设 `MONOLOAD_KEEP_LORA=1` 时不启用。

### 7.1 时机

`release.install()` 包装 `execution.PromptExecutor.execute_async`：原函数跑完之后（不管成功、失败还是被中断，写在 `finally` 里），调用 `release_after_prompt(executor)`。`main.py` 的 prompt worker 是一个接一个执行 prompt 的，这个点正好在「上一个 prompt 的所有节点都执行完」和「下一个 prompt 开始」之间，执行器的缓存也在手上（`executor.caches`）。释放过程中出错只记错误日志，不影响服务。

### 7.2 怎么判断哪些东西属于 LoRA

按**内容**判断，不按节点类型判断。原生 LoRA 节点、Hook LoRA 节点，以及第三方 LoRA 加载节点，只要产出的是带 patch 的 clone，就都覆盖到。一个值「带 LoRA」，指它（递归地，在 list/tuple/dict 里）包含下面任何一种：

* 带权重 patch 或 hook patch 的 `ModelPatcher`（`patches` 或 `hook_patches` 非空），或者带 bypass LoRA 注入的 `ModelPatcher`（`injections["bypass_lora"]`，来自 `LoraLoaderBypass` / `load_bypass_lora_for_models`）；
* `patcher` 属性是这样一个 `ModelPatcher` 的对象（`CLIP`），或者带有 `apply_hooks_to_conds` 的 `CLIP`（`SetClipHooks` 的输出）；
* 含有权重 hook 的 `HookGroup` / `Hook`（`CreateHookLora` 的输出）；
* 挂着这类 hooks 的 conditioning（conditioning 的 dict 里有 `hooks`）。

底模加载节点的输出是**没有** patch 的 patcher，因此不受影响。`ModelSamplingDiscrete` 这类只带 object patch 的 clone 也不受影响。带权重 patch 的模型合并结果按同样的规则释放，下次用到时重新执行合并节点即可，这一步很便宜。

### 7.3 释放什么、怎么释放

1. **已加载的模型**（`model_management.current_loaded_models`）：对 patcher 带 LoRA、或模块上还挂着运行时 patch 的 `LoadedModel`：
   * 就地 `unpatch_hooks()` + `unpatch_model(device_to=None, unpatch_weights=True)`。Monoload 版本会摘掉所有运行时 patch，清空设备上的 LoRA 缓存；**权重一个字节都不搬**，因为本来就没被改过。
   * 原生 `unpatch_model` 即使不搬权重，也会把模型标成「未加载」（`model_loaded_weight_memory = 0`，删掉 `comfy_patched_weights`）。权重其实还在原位，所以卸之前先记下这些状态，卸完原样恢复。
   * 沿 `patcher.parent` 往上找到第一个没有权重 patch 的祖先，也就是底模的 patcher（`LoraLoader` 的输出是它的 clone）。把 `LoadedModel` 切到这个 patcher（ComfyUI 自己在 patcher 被回收时也用 `_set_model` 做同样的事），并把 `model.current_weight_patches_uuid` 设为底模的。这样在 ComfyUI 看来，现在「已加载的就是底模、而且没有 patch」：下一个不带 LoRA 的工作流会直接复用，不调用 `ModelPatcher.load()`；下一个带 LoRA 的工作流因为 uuid 不同，会照常重新挂 patch。
   * 找不到干净的祖先时，把 uuid 设成一个新的随机值，强制下次使用时重新评估。
   * 部分加载（`model_lowvram`，只在非 `--gpu-only` 时出现）时，就地卸会连被卸载层的原生 lowvram 状态一起清掉，所以改用原生的 `LoadedModel.model_unload()`，权重回到 offload 设备。
2. **输出缓存**（`caches.outputs`，包括子图的 subcache；CLASSIC / LRU / RAM_PRESSURE 三种都支持）：删掉值带 LoRA 的条目，同时清理 LRU / RAM_PRESSURE 的附属字典（`used_generation`、`children`、`timestamps`）。下游节点的输出（latent、图片、普通 conditioning）不含 LoRA，照常保留。下次跑同一个工作流时，如果下游已经命中缓存，LoRA 节点就根本不会被执行。
3. **节点实例缓存**（`caches.objects`）：清掉节点实例上的 `loaded_lora`。`LoraLoader`、`LoraLoaderModelOnly`、`CreateHookLora`、`LoraLoaderBypass` 都用这个属性缓存读进来的 LoRA 文件。
4. `gc.collect()`：被丢掉的 LoRA clone 在这里被回收。
5. **同步不带 patch 的 clone 的 uuid**（真机验收时发现的问题）。`LoraLoader` 总会克隆 CLIP；而 `add_patches()` 不管有没有匹配到 key，都会换一个新的 `patches_uuid`。所以对 Smooth Booster 这种没有 TE key 的 LoRA，会得到一个「没有任何 patch、但 uuid 不同」的 CLIP clone，它被当作已加载模型。第 1 步只处理带 patch 的 patcher，没有处理它。缓存清掉之后它被回收，ComfyUI 的 finalizer 把 `LoadedModel` 切回父 patcher（底模的 CLIP），但模型上的 `current_weight_patches_uuid` 还是那个 clone 的，于是下一个工作流的 CLIP 又完整 `load()` 了一次（日志里只有 `loaded completely`，没有 `Requested to load`）。修法：gc 之后再检查一遍已加载模型，凡是 patcher 不带任何 patch 或 bypass 注入、模型上也没有运行时 patch 的，就把模型的 uuid 同步成这个 patcher 的。Monoload 下权重从不被修改，所以「没有 patch 的模型」与任何一个没有 patch 的 patcher 状态等价。
6. `soft_empty_cache()`，日志里打一行 `[Monoload] released LoRA after prompt: ...`（含 `N clean clone(s) re-synced`）。

到这一步，LoRA 张量已经没有任何引用：LoRA 文件读出的 dict、clone 上的 patch、运行时 patch 和它在计算设备上的副本、hook 组都已被回收。测试里用弱引用逐个确认（§7.4）。

### 7.4 验证（`tests/test_release.py`，真正的 `PromptExecutor`）

连续执行 API 格式的工作流：LoRA → 无 LoRA → 只改 UNet 的 LoRA（没有 TE key）→ 无 LoRA → Hook LoRA → 无 LoRA → bypass LoRA → 无 LoRA → LoRA（换一个种子），三种缓存模式各跑一遍，并在 `MONOLOAD_KEEP_LORA=1` 下再跑一遍：

* 每个 LoRA prompt 结束后：从 `models/loras` 读出的**每一个**张量的弱引用都已失效（CPU 上被回收；GPU 上的副本只挂在这些对象上，也一起被回收）；没有仍带 patch 的 patcher；模块上没有运行时 patch；设备缓存为空；节点实例上没有 `loaded_lora`。
* 接下来的无 LoRA prompt：`CheckpointLoaderSimple` 没有再次执行，底模对象是同一个，`ModelPatcher.load()` 一次也没被调用；输出与**另一个从没见过 LoRA 的进程**的输出逐位一致。
* 释放后再用 LoRA：结果与全新进程逐位一致。
* `MONOLOAD_KEEP_LORA=1`：LoRA 保留。

## 8. 限制

* 只接管没有重写 `patch_weight_to_device` 的 `ModelPatcher`（ComfyUI 自带的加载器都属于这种）。GGUF 等自带 patch 机制的插件保持原生行为，日志里会提示。
* 每步都有合并开销（§5）。LoRA 越多、改的层越大，开销越明显；没被 patch 的层没有影响。
* LoRA 文件由原生 `LoraLoader` 读取，在一个 prompt 执行期间常驻 CPU 内存；计算设备上另有一份被用到的 LoRA 张量缓存。prompt 结束后两者都会释放（§7），设了 `MONOLOAD_KEEP_LORA=1` 则保留，和原生一样。
* 量化层上的 LoRA 与原生不逐位一致（§3.3，有意为之）。
* 本仓库的测试都在无 GPU 的机器上用 `--cpu` 跑；GPU 上的数值一致性和耗时要按 README 的真机验收步骤确认。
