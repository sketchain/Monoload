# Monoload 设计说明（v2：运行时 LoRA 合并）

参考源码：锁定镜像 `kyuz0/amd-strix-halo-comfyui@sha256:384aa1fe…` 里的 `/opt/ComfyUI`（ComfyUI 0.31.0，commit `62b3c94b`）。下文提到的函数和行为都按这份源码核实过。

## 0. 转型

v1（tag `v1-converter`，远端归档分支 `archive/v1-converter`）做的是「离线转换 + pread 直读」，为的是绕开 mmap 的慢和卡死。后来查到根因在 ROCm 的 rocclr：H2D 拷贝超过 1 MiB 时，rocclr 会临时 pin 源内存，在 APU 上这一步走 KFD HMM，逐页处理。上游 ROCm/clr `3ccb59f`（2026-09-23）已经改成「统一内存设备上不做 pin」；修复进入正式版之前，设 `GPU_PINNED_MIN_XFER_SIZE=65536` 效果等价。于是原生加载改成「不开 `--disable-mmap` + 这个环境变量」就足够快，v1 那套不再需要。

v2 只做一件事：**在 ComfyUI 原生加载出来的模型上打 LoRA 时，不改原权重、不做任何备份，而是在每一层计算的那一刻临时合并。** 对工作流透明：原生 `UNETLoader`、`CheckpointLoaderSimple`、`CLIPLoader`、`LoraLoader`、`LoraLoaderModelOnly`、Hook LoRA 节点照常使用。

在此基础上还有四点：

* **VAE 解码降峰值（§9）。** 接管 `comfy.sd.VAE.decode`：自己估算内存、batch 逐张；认得的结构（Wan 2.1 VAE 单帧）按输出条带倒推重算（第一层，§9.13），其余的卷积按输出行分块、注意力按 query 分块（第二层）；结果与整图解码数学等价；OOM 只缩小分块重试，绝不退回 tiled 近似解码。与 LoRA 部分相互独立。
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
* VAE 部分的限制见 §9.9。

## 9. VAE 解码降峰值（第一阶段：解码管理入口 + 逐算子分块 + 测量工具）

依据：《ComfyUI VAE 条带解码调研报告》（2026-10-02，源码基线同上）。报告和需求冲突的地方以需求为准；报告里的峰值是推导值，已用 `tests/bench_vae.py` 在 CT 700 上确认（§9.12）。

### 9.1 问题

**调用链**（按 62b3c94 核实）：`VAEDecode.decode` → `comfy.sd.VAE.decode(samples)`：2D VAE 收到 5D 输入时取第一帧；`memory_used = memory_used_decode(shape, vae_dtype)`；`load_models_gpu([patcher], memory_required=memory_used)`；`batch_number = int(free / memory_used)`；逐批 `first_stage_model.decode(samples)` → `.to(output_device, intermediate_dtype, copy=True)` → 写进 `pixel_samples` → `process_output`（`(x+1)/2` 再 clamp 到 [0,1]，原地）；最后 `movedim(1,-1)` 成 NHWC。SDXL 是 `AutoencoderKL`（先 `post_quant_conv`），Flux `ae` 是 `AutoencodingEngine`，两者的 decoder 都是 `comfy.ldm.modules.diffusionmodules.model.Decoder`；`qwen_image_vae` 是 `comfy.ldm.wan.vae.WanVAE`，先 `conv2` 再 `Decoder3d`，单帧时 `feat_map=None`。

**峰值从哪来**（报告 C、E 节的推导，未实测）：

* 禁用 MIOpen 后（`torch.backends.cudnn.enabled = False`，在 `model_management.py` 的 AMD 分支，`COMFYUI_ENABLE_MIOPEN=1` 可跳过），普通 3×3 卷积走 Slow2d：先 im2col 展开成 `[Cin·k·k, Hout·Wout]` 的 columns，再 GEMM。bf16 下 columns 是 `18·Cin·Hout·Wout` 字节。4K 输出时最后一层 Conv256→256 的 columns 约 35.6 GiB，加上输入输出，SDXL 原版 4K 解码峰值约 45 GiB；Qwen 最大的是全尺寸 Resample Conv192→96，约 31 GiB。**峰值主要来自展开缓冲，而不是激活本身**（同一层的激活只有 2–4 GiB）。
* 全局注意力只在最低分辨率（H/8）的 mid block，但 4K 时 N ≈ 13 万，一张 bf16 分数矩阵约 31 GiB。原生 `slice_attention` 按当前空闲内存选切片数，峰值不固定。AMD 上 `pytorch_attention_enabled_vae()` 返回 False，所以实际走 split（`normal_attention`）。
* `memory_used_decode`：AMD 上 `VAE_KL_MEM_RATIO = 2.73`，SDXL 4K 估约 92 GiB，Wan/Qwen 约 34 GiB。`load_models_gpu` 按这个值腾内存，`--gpu-only` 下也会走卸载流程（`free_memory` 没有「HIGH_VRAM 不卸载」的保护），batch 也按它切。
* 真 OOM 后原生退回 `decode_tiled_`：每个 tile 各自做 GroupNorm、各自做注意力，靠重叠融合。正确的全局方差是 `Σ p_i·[var_i + (mean_i − mean)²]`，tile 内统计漏了组间项，任何固定 halo 也恢复不了全局注意力，所以**结果和整图不等价**。这违反 Monoload 的原则：不悄悄退回近似路径，也不悄悄退回高占用路径。

### 9.2 三层设计

目标：结果和原版整图解码**数学等价**（只差浮点误差，不要求逐位一致），峰值尽量低，允许多花算量换显存。

| 层 | 适用 | 做法 | 状态 |
|---|---|---|---|
| 第一层：条带解码 | 认得的结构（LDM `Decoder`、Wan `Decoder3d` 单帧） | 低分辨率前缀（含 mid 全局注意力）整图算；只在每个分辨率阶段末尾存档；按输出条带倒推每层所需的输入行区间并重算；GroupNorm 的全局统计逐层空跑求得（条带内 fp32 Welford，跨条带 Chan 合并）；按峰值预算自动选条带高度和存档方案，满足不了就报错 | **第二阶段已实现 Wan `Decoder3d` 单帧（§9.13）**；第三阶段做 LDM |
| 第二层：逐算子分块 | 不认识结构也能用 | 原 forward 原样运行，只在受管理的解码过程中替换重算子：卷积按输出行分块，限制 im2col 工作区；注意力按 query 分块，K/V 完整，每个 query 仍对全图做 softmax。整图语义不变，算量约 1 倍 | 第一阶段实现（真机验收通过，§9.12） |
| 兜底 | 前两层都处理不了 | 原生整图解码（结果本身正确），打一条日志 | 第一阶段实现 |

第二层不减少激活本身（每层的完整输入输出仍然存在），只去掉 columns 和分数矩阵这两类「临时大块」；第一层再去掉高分辨率激活。报告 E.1 的建议也是先做第二层这个 baseline：如果它已经满足 62.5G GTT 的目标，第一层可以按需推进。

### 9.3 接入点

**VAE.decode（管理入口，`monoload/vae.py`）。** 与 hotpatch 相同的做法：`install()` 保存 `comfy.sd.VAE.__dict__["decode"]`，在类上换成 Monoload 的版本，`uninstall()` 恢复。安装前检查签名和依赖的接口：`VAE.decode(self, samples_in, vae_options)`、comfy Conv3d / torch Conv2d 的 `_conv_forward(self, input, weight, bias, ...)`、`load_models_gpu(memory_required=...)`、三个注意力函数、`raise_non_oom` 等；任何一项不符就打警告、不安装，VAE 保持原生。选这个入口的理由（报告 B.5）：只有它能同时控制内存估算、batch、输出和 OOM；只换 `Decoder.forward` 会被原生的整图估算和自动 tiled 回退包住；替换 `first_stage_model` 要重建模型树，容易破坏 state_dict 前缀和 patcher 引用。

**卷积。** 在 62b3c94 上核实的调用链：

```
comfy.ops.disable_weight_init.Conv2d.forward
  ├─ comfy_cast_weights 或有 weight_function / bias_function（manual_cast、lowvram、Monoload 的运行时 LoRA）:
  │    forward_comfy_cast_weights(input)
  │      with CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):   # cast_bias_weight：dtype/设备转换、weight_function
  │          return self._conv_forward(input, weight, bias)
  └─ 否则: torch.nn.Conv2d.forward(input) → self._conv_forward(input, self.weight, self.bias)
comfy Conv3d 多一个 autopad 参数，并重写了 _conv_forward：
  _conv_forward(input, weight, bias, autopad=None): autopad == "causal_zero" 时 weight = weight[:, :, -T:]；
  NVIDIA 的 cudnn workaround；否则 super()._conv_forward → F.conv3d(input, weight, bias, stride, self.padding, dilation, groups)
```

两条路径最后都落在 `self._conv_forward(input, 最终权重, bias, ...)`。所以接入点选它：受管理的解码开始时，给 VAE 模型里每个 `torch.nn.Conv2d` / `Conv3d` 实例（comfy.ops 的各个变体都是它们的子类）设一个**实例属性** `_conv_forward`，解码结束（含异常）时删掉。实例属性优先于类方法，所以：

* 覆盖 cast 路径和非 cast 路径，拿到的是 `cast_bias_weight` 处理完的权重，不绕过 weight_function（实测一次卷积调用只调用一次 weight_function，而不是每块一次）；
* 每块仍然调用**原来的** `_conv_forward`（autopad 截断、后端 workaround、`padding_mode` 处理都照旧），只是 H 方向的 padding 临时设为 0，由 Monoload 在真实边缘补零（§9.5）；
* 只作用于这一个 VAE 的模块，只在受管理的解码期间存在，UNet / CLIP 完全不受影响；不改类，也不改全局函数。

**注意力。** `AttnBlock`（LDM）和 Wan 的 `AttentionBlock` 都在 `__init__` 里把 `vae_attention()` 的结果存成实例属性 `optimized_attention`。受管理期间，若它是 ComfyUI 的三个 VAE 注意力函数之一（`normal_attention` / `pytorch_attention` / `xformers_attention`），就换成对应的分块版本，结束时还原；不认识的实现（第三方替换过的）保持原样，并在日志里列出。

### 9.4 内存估算

管理入口给 `load_models_gpu` 报的是这个上界（`vae.estimate()`）：

```
estimate = 4 × A_max + 2 × workspace + 输出缓冲 + 一个 latent 样本
  A_max     ：一个样本解码过程中最大的单个激活张量（字节）
  workspace ：MONOLOAD_VAE_WORKSPACE（默认 1 GiB）
  输出缓冲  ：整个 batch 的输出（intermediate dtype，fp32），仅当输出设备就是 VAE 所在设备时（--gpu-only）计入
```

* **A_max 怎么得到。** 每个 VAE（按 latent 通道数和维数）第一次受管理解码时，先用 8×8 的零 latent 跑一次小解码，用 forward hook 记录所有模块输入输出里最大的张量，换算成「每个 latent 像素多少字节」并缓存；这几种 decoder 都是全卷积、尺度因子固定，激活大小与 latent 面积成正比。小解码期间保存并恢复全局 RNG 状态。小解码失败时退回静态上界：最宽的卷积通道数 × 完整输出分辨率。
* **为什么是 4 × A_max。** 第二层不改 forward，峰值时刻同时存活的大张量是：残差块的输入 x（留给 shortcut）、norm 的输出、正在写的卷积输出；上采样处是上采样前的输入、上采样后的张量、卷积输出。按 62b3c94 的代码逐块数下来，LDM 约 2.5 × A_max（全尺寸 256→128 的残差块：x 256 通道 + norm1 输出 256 + conv1 输出 128），Wan 约 2 × A_max（RMS_norm 的 `F.normalize(x) * scale * gamma` 会多出两个临时量）。取 4 倍给分配器碎片留余量。
* **为什么是 2 × workspace。** columns 本身 ≤ workspace；每块还有输入块的连续拷贝（约 columns / k²）、输出块（拷进预分配输出前）和 GEMM 内部工作区。注意力的预算已经包括分数矩阵和 softmax 结果两份（§9.6）。
* **数字**（用真实宽度、随机权重的模型跑小解码得到，bf16）：

| 输出 | SDXL / Flux：A_max | Monoload 估算 | 原生（AMD，×2.73） | Qwen：A_max | Monoload 估算 | 原生（AMD） |
|---|---|---|---|---|---|---|
| 1344×768 | 0.49 GiB | 3.98 GiB | 11.43 GiB | 0.37 GiB | 3.49 GiB | 4.23 GiB |
| 2688×1536 | 1.97 GiB | 9.92 GiB | 45.73 GiB | 1.48 GiB | 7.95 GiB | 16.92 GiB |
| 3840×2160 | 3.96 GiB | 17.91 GiB | 91.86 GiB | 2.97 GiB | 13.96 GiB | 33.99 GiB |

  A_max 与报告 E.2 的推导一致（SDXL 是全尺寸 256 通道那张图：上采样后、送进 up0 第一个残差块之前；Qwen 是全尺寸 192 通道）。这些都是上界，真实峰值要看 bench（§9.10）；bench 会把估算值和实测的 `max_memory_allocated` 增量并排列出。
* 第一次解码某个 VAE 时要先加载权重才能跑小解码，所以那一次会调用两次 `load_models_gpu`（第一次只报 8×8 的小估算，第二次报真实估算）；之后用缓存，只调用一次。

### 9.5 卷积分块：行区间推导

记 H 方向的 kernel、stride、dilation 为 k、s、d，上下 zero padding 为 p_lo、p_hi，输入行数 H，则输出行数 `Hout = ⌊(H + p_lo + p_hi − d(k−1) − 1) / s⌋ + 1`。输出第 o 行读 padding 后的第 `s·o … s·o + d(k−1)` 行，即输入的第 `s·o − p_lo … s·o − p_lo + d(k−1)` 行。对输出行块 `[o0, o1)`：

```
lo = s·o0 − p_lo
hi = s·(o1 − 1) + d·(k − 1) − p_lo + 1        # 不含
真实输入行 [max(lo, 0), min(hi, H))，上方补 max(0, −lo) 行零，下方补 max(0, hi − H) 行零
```

可以证明 `hi ≤ H + p_hi`，所以补零只会出现在真实的上下边缘：块与块之间用的都是真实的相邻行，内部块一行零都不补。每块交给原来的 `_conv_forward`，H 方向 padding 临时设为 0，其他维度：

* zero padding 对称的维度（W，以及 Conv3d 的 T）照常交给卷积自己处理；
* `padding="same"` 且 kernel 为偶数时两侧不对称（torch 把多出的一格放在高端）：H 方向按上式处理，其他维度对每块显式 `F.pad`；`padding="valid"` 就是 0；
* `padding_mode` 为 reflect / replicate / circular：与 torch 自己的做法相同，先对整个输入按该模式 `F.pad`，再对 pad 好的输入按 padding = 0 分块（这时整张 pad 出来的副本与原生一样存在）；
* stride、dilation、groups、任意 kernel 都按上式处理（groups 只影响 Cin/groups）。测试覆盖：3×3/5×5 dil2/4×4 s3/3×1/1×3/2×5 s(2,1) dil(1,2)、groups 2 和 depthwise、same / valid、三种非零 padding_mode、Conv3d（含 stride 和 replicate）。

**Conv3d autopad="causal_zero" 与 Wan CausalConv3d。** 62b3c94 里 `CausalConv3d.forward` 在 T=1、没有 cache 时直接 `super().forward(x, autopad="causal_zero")`：comfy Conv3d 把时间 kernel 截成最后 T 帧，padding 为 (0, p, p)，**不做任何整张拷贝**；T>1 时只在时间维 `torch.cat` 零帧（整张拷贝一次），空间 padding 仍交给卷积。这两种情况分块都只动 H，与时间维无关。（需求里写的「先对整个张量做 F.pad，再以 padding=0 调卷积」是 Wan 官方仓库的写法，不是这个版本；这个版本的单帧路径没有整张 pad 的副本。）

**预算。** 每块的 columns 估算为 `(Cin/groups) × ∏k × 块输出行数 × Wout（Conv3d 再乘 Tout）× batch × dtype 字节`（causal_zero 时 kT 取截断后的值）。整层的估算不超过预算就直接调用原 `_conv_forward`，与原生完全相同；超过时每块行数 = `⌊预算 / 每行 columns⌋`，至少 1 行。1×1、stride 1、无 padding 的 Conv2d 在 Slow2d 里不做 im2col（直接 GEMM），工作区按 0 计，不分块。

输出按完整形状预分配一次（dtype、设备取第一块的结果），逐块 `copy_` 进去。每块的输入是原张量在 H 上的一个视图，卷积内部会拷成连续的块（只有块那么大）。

### 9.6 注意力分块

每块 query 数 = `⌊预算 / (2 × B × N × 元素字节)⌋`（至少 1），即一块的分数矩阵和它的 softmax 结果两份合计不超过预算。4K、bf16、1 GiB 时 N = 129600，每块 2071 个 query，约 63 块。

* split（AMD 上实际用的）：复刻 `normal_attention` + `slice_attention`：`r1 = zeros_like(k)`；每块 `s1 = bmm(q[:, i:end], k) * scale`，`softmax(s1, dim=2)`（输入 dtype），`r1[:, :, i:end] = bmm(v, s2)`。与原生唯一的区别是块大小固定由预算决定（原生按空闲内存选 steps，且要求整除）。
* pytorch：与 `pytorch_attention` 相同的 reshape，SDPA 对 q 分块、K/V 完整；块数为 1 时与原生是同一次调用。
* xformers：`memory_efficient_attention` 对 q 分块；遇到 NotImplementedError 时改走分块的 split（原生在这里退回 `slice_attention`）。

每个 query 的 softmax 仍然覆盖全图，没有引入局部注意力。

### 9.7 OOM 策略

```
workspace = MONOLOAD_VAE_WORKSPACE
loop:
    try: 解码（逐张，OpChunking 生效）
    except e: raise_non_oom(e)（非 OOM 原样抛出）；标记 OOM
    离开 except 块后（异常和它引用的张量都已释放）：soft_empty_cache(True)
    workspace 已到下限 min(64 MiB, 设置值) → 抛 MonoloadVAEOOMError（写明下限、重试次数、latent 形状、估算值）
    否则 workspace 减半，打警告，重试
```

`decode_tiled_` 等 tiled 路径**从不调用**（测试里替换成计数器确认）。所有实例属性在 `OpChunking.__exit__` 里还原，异常时也一样；与 prompt 结束后的 LoRA 释放没有共享状态。

### 9.8 覆盖范围与兜底

| 情况 | 处理 |
|---|---|
| 4D latent（2D VAE：SDXL、Flux `ae` 等） | 第二层 |
| 5D latent 给 2D VAE | 与原生一样取第一帧，第二层 |
| 5D、T=1 给 3D VAE（Wan 2.1 / `qwen_image_vae` 等） | 第二层 |
| 多帧视频（5D、T>1） | **暂时**交给原生，打日志。这是第一阶段暂时不做，不是永远不做：多帧需要按时空分别规划（时间 cache、首帧特例），留到第一层之后 |
| 1D / 音频 latent、`comfy_has_chunked_io` 的 VAE（LTX、MiniMax：自己往预分配输出里写） | 原生，打日志 |
| 用户显式用 `VAEDecodeTiled` 节点或 `VAE.decode_tiled` | 不经过 `VAE.decode`，保持原生（用户自己选择了 tiled 的语义） |
| 直接调用 `first_stage_model.decode` 的第三方代码 | 不经过 `VAE.decode`，原生 |

第二层对结构没有要求：只要卷积是 `torch.nn.Conv2d/Conv3d`（含 comfy.ops），注意力是 ComfyUI 的三个 VAE 注意力函数之一，就会被分块；其他算子照原样运行（不会出错，只是峰值不一定降下来）。

### 9.9 开关与限制

| 设置 | 效果 |
|---|---|
| 默认 | 安装管理入口，预算 1 GiB |
| `MONOLOAD_VAE_WORKSPACE` | 预算：`1G`、`512M`、`768`（纯数字按 MiB）等 |
| `MONOLOAD_DISABLE_VAE=1` | 只关 VAE 部分，LoRA 部分照常 |
| `MONOLOAD_DISABLE_VAE_STRIPE=1` | 只关第一层（§9.13），所有受管理的解码走第二层 |
| `MONOLOAD_VAE_BUDGET` | 第一层的峰值预算；不设时用默认策略（128 行条带的峰值内最高的条带，§9.13.4） |
| `MONOLOAD_VAE_STRIPE_ROWS` | 强制第一层的条带核心高度（扫参、调试） |
| `MONOLOAD_EXACT=1` | VAE 走原生：分块改变 GEMM 形状，不保证逐位一致 |
| `MONOLOAD_DISABLE=1` | 什么都不装 |

命名：需求里举例的是 `MONOLOAD_VAE=0`；现有开关都是「设为 1 时改变默认行为」（`MONOLOAD_DISABLE`、`MONOLOAD_EXACT`、`MONOLOAD_KEEP_LORA`），所以关闭开关取名 `MONOLOAD_DISABLE_VAE=1`，与 `MONOLOAD_DISABLE=1` 对应。

限制：

* 不逐位一致：块的 GEMM 形状不同，累加顺序可能不同。CPU fp32 上整个 decoder 的差异在 1e-6 量级；CT 700（bf16）上与原生的像素 RMSE ≤ 9.6e-4，零星单点 max|Δ| 到 0.096，fp32 参照确认是 bf16 固有噪声（§9.12）。
* 第二层不减少激活本身：4K 时 SDXL 的全尺寸激活仍有约 4 GiB 一张，实测峰值 alloc 11.2 GiB（§9.12）。再往下要靠第一层。
* 速度：算量约 1 倍，块多了有额外的启动和拷贝开销；实测 1344×768 慢 1–4%，更高分辨率快 9–17%（§9.12）。
* 不认识的注意力实现、非 `torch.nn.ConvNd` 的卷积不分块。

### 9.10 测试与待真机确认的问题

**CPU 测试**（`tests/test_vae.py`，锁定镜像，`--cpu`，不需要模型文件）：

* 卷积 53 项（§9.5 列出的 27 种配置，多数在 1 字节预算——每块 1 行——和 64 KiB 预算下各一次）：与不分块的结果比，相对误差最大 4.6e-7（多数为 0；判定阈值 fp32 1e-5）；cast 路径（fp16 权重、fp32 输入）带 weight_function 时，17 块只调用一次 weight_function；1×1 / stride 1 不分块。
* 注意力：split / pytorch 在 fp32、bf16 下，块大小 1、5、N、超过 N，与原生相同（fp32 相对误差 ≤ 3.5e-7，bf16 为 0）；`AttnBlock` 的实例替换与还原；不认识的实现保持原样。
* 整个 decoder：用 ComfyUI 自己的类、小通道配置、随机权重构造 state dict，交给 `comfy.sd.VAE` 识别：SDXL 式（`AutoencoderKL`，z=4）、Flux 式（`AutoencodingEngine`，z=16）、Wan/Qwen 式（`WanVAE`，5D T=1）。16 KiB 和 256 KiB 预算（绝大多数卷积被分块，16 KiB 时每块 1 行），与原生 `VAE.decode` 比较：raw 输出 max|Δ| ≤ 7.2e-6，像素 ≤ 3.2e-6（fp32）；bf16 VAE 为 0；batch 2、5D 给 2D VAE、7×13 的奇数尺寸同样通过；输出形状、dtype、设备、NHWC 与原生相同；`load_models_gpu` 收到的是 Monoload 的估算；RNG 状态不变；预算足够大时不分块、与原生逐位一致。
* 兜底与 OOM：多帧 latent 交给原生且结果相同；模拟「预算高于 128 MiB 就 OOM」时从 512 MiB 两次减半到 128 MiB 后成功；始终 OOM 时在 64 MiB 抛 `MonoloadVAEOOMError`；两种情况都没有调用 `decode_tiled_`，异常后模块上没有残留的实例属性；非 OOM 异常原样抛出。
* 日志计时：受管理的解码在开始计时前和停止计时前各同步一次设备（`comfy.model_management.synchronize()`），测试确认每次解码恰好同步两次。
* 合计 131 项检查，0 失败。`tests/test_entry.py`：默认、`MONOLOAD_DISABLE=1`、`MONOLOAD_KEEP_LORA=1`、`MONOLOAD_EXACT=1`、`MONOLOAD_DISABLE_VAE=1`、`MONOLOAD_VAE_WORKSPACE=512M` 六种组合下的安装状态全部正确；LoRA 部分的 `test_dtype_paths.py` 两种模式（198 / 108 项）照旧全部通过。

**真机要回答的问题**（`tests/bench_vae.py`，用法见 README §9.7；结果见 §9.12）：

1. 原生峰值的主因是不是 im2col / Slow2d 的 columns（`--profile` 的算子表和峰值时刻的存活分配）。——是。
2. 第二层把峰值降到多少，与 §9.4 的估算差多少；GTT / cgroup 是否同步下降。——4K 降到 1/4～1/6，估算是有效上界，GTT 同步下降。
3. 精度：`monoload vs native` 与 `native2 vs native` 的量级对比，块边界附近有没有系统性误差。——PSNR ≥ 60 dB，没有接缝；max|Δ| 离群点经 fp32 参照确认是 bf16 固有噪声。
4. 速度：冷/热耗时与原生相比。——低分辨率慢 1–4%，高分辨率快 9–17%。

### 9.11 后两阶段计划（第二阶段已完成，见 §9.13）

* **共同部分。** 结构识别按真实模块（类型、通道、kernel、stride、padding、norm、注意力）而不是文件名；每种结构第一次使用时用小 latent 和原生对比自检，不通过就对这个 VAE 禁用第一层、醒目警告，退回第二层。适配接口是 `vae.STRIPE_ADAPTERS`（`match` / `self_test` / `plan` / `run`），第二阶段登记了 Wan 2.1 的适配器。
* **第二阶段：Qwen（Wan `Decoder3d` 单帧）。** 已实现，见 §9.13。
* **第三阶段：LDM（SDXL / Flux `ae`）。** 与第二阶段的区别是 30 个 GroupNorm 需要全图统计，GN 的输出依赖整张图，不能像 RMS 那样只看局部行。计划：
  * 前缀同样整图算到最低分辨率的残差块之后（H/8，512 通道），中间的全局注意力用第二层的 query 分块。
  * 之后按「统计调度」逐个残差块推进：对一个残差块，pass A 按条带计算 `GN1 → SiLU → conv1`，只累计 GN2 输入的统计 (n, mean, M2)（条带内 fp32 Welford，跨条带 Chan 合并，只统计核心行，不统计 halo），不保存 conv1 的完整输出；统计冻结后，pass B 按条带重算 conv1，再 `GN2 → SiLU → conv2`，加上正确坐标上的 shortcut，写出完整的块输出 Y，同时累计下一个块 GN1 的统计。旧的输入 X 等所有消费者完成后才释放。这样每个块的 conv1 算两遍、conv2 一遍，全 decoder 的卷积约 1.3 倍（报告 C.5），峰值是两份完整的块输入输出（4K 时 C256 约 4 GiB + C128 约 2 GiB）加条带工作集。
  * 上采样边界同样滚动存档；全尺寸的块输出若太大，可以只存到 H/2，最后一级做全条带。按预算选条带高度和存档位置。
  * 区间倒推、精确行检查、自检、OOM 和开关都沿用第二阶段的框架；统计另做逐 GN 的对照测试（条带累计的 mean / var 与整图 GroupNorm 的统计比较，fp32 下 1e-6 量级）。
* 两个阶段都要求：预算满足不了就报错；OOM 只缩小条带；误差验收照 §9.10 的指标。

### 9.12 真机结果（CT 700，2026-10，af9abc6）

设置：`--gpu-only --bf16-vae`，`cudnn.enabled = False`（Slow2d），VAE 注意力为 split；SDXL 用 WAI v17 内置 VAE，Flux 用 `ae.safetensors`，Qwen 用 `qwen_image_vae.safetensors`；随机 latent（种子 0）；每档 1 次冷启动 + 3 次热启动，`--profile`；先 `/free`，脚本进程里只有这一个 VAE；预算 1 GiB。

**峰值主因：im2col 展开缓冲（已确认）。** 原生解码峰值时刻的存活分配（分配器内存历史回放）：

| VAE | 解码期间峰值（1344×768 / 2688×1536 / 3840×2160） | 其中最大的一块：卷积的 columns | 位置 |
|---|---|---|---|
| SDXL / Flux | 5.68 / 22.70 / 45.61 GiB | 4.43 / 17.72 / 35.60 GiB（78%） | `ResnetBlock.forward` 的卷积（model.py:218）→ `torch/nn/modules/conv.py:_conv_forward` |
| Qwen | 3.98 / 15.92 / 31.99 GiB | 3.32 / 13.29 / 26.70 GiB（83%） | 全尺寸卷积 |

columns 的大小与 §9.1 / 调研 E.2 的推导吻合（`18·Cin·Hout·Wout` 字节）。峰值时刻其余的存活块是同一层的输入（norm 输出）、上一层的输出和卷积输出，各 2–4 GiB。原生的注意力也很大，但不是峰值：2688 时一张 7.75 GiB 的分数矩阵（`mul` 和 `_softmax` 各一份）；4K 时按空闲内存切成两片，每片 15.64 GiB。原生 4K 每次都先在 HIPCachingAllocator 报 OOM（申请 38.2 GB 的 columns 时 free 只有 11.8 GB），释放缓存重试后才成功，离真 OOM 只差一步。Qwen 原生 4K 的 GTT 峰值增量 59.1 GiB，而 GTT 上限是 62.5 GiB。

**显存（GiB；原生 → monoload；1344×768 / 2688×1536 / 3840×2160）：**

| VAE | `max_memory_allocated` 增量 | GTT 峰值增量（= reserved 增量） | monoload 估算 | 估算 / 实测 GTT |
|---|---|---|---|---|
| SDXL | 5.68→2.41 / 22.70→6.15 / 45.61→11.17 | 8.36→3.72 / 42.84→9.25 / 52.48→14.86 | 3.98 / 9.92 / 17.91 | 1.07 / 1.07 / 1.21 |
| Flux `ae` | 5.68→2.41 / 22.71→6.15 / 45.61→11.18 | 8.36→3.72 / 42.84→9.24 / 52.50→14.89 | 3.98 / 9.92 / 17.92 | 1.07 / 1.07 / 1.20 |
| `qwen_image_vae` | 3.98→1.82 / 15.92→3.80 / 31.99→6.46 | 6.10→2.12 / 29.37→5.05 / 59.14→9.59 | 3.49 / 7.95 / 13.96 | 1.65 / 1.57 / 1.46 |

* 4K 的 alloc 峰值降到原来的 1/4（SDXL / Flux）和 1/5（Qwen），GTT 峰值降到 1/3.5 和 1/6。
* **估算是有效上界**：9 个场景里估算都大于实测的 reserved / GTT 增量。最紧的是 SDXL / Flux 的 1344 和 2688（1.07 倍），这两档的 reserved 比 allocated 多出 1.3–3.1 GiB 的分配器缓存，估算仍然盖住了它。`load_models_gpu` 没有因此卸载任何模型（`unload` 全为 0）。原生的 91.86 GiB 估算在这个只有 VAE 的进程里恰好也没触发卸载，但服务里常驻 UNet / CLIP 时会。
* cgroup 采样增量 ≤ 0.1 GiB：GTT 不计入容器的 cgroup。这台机器上没有拿到按 fd 重置的 `memory.peak`（表里是 n/a），只有采样值。

**耗时（热启动中位数，秒；原生 → monoload）：**

| VAE | 1344×768 | 2688×1536 | 3840×2160 |
|---|---|---|---|
| SDXL | 0.878 → 0.908（+3.4%） | 4.720 → 4.106（−13%） | 12.157 → 10.184（−16%） |
| Flux `ae` | 0.872 → 0.906（+3.9%） | 4.741 → 4.102（−13%） | 12.142 → 10.108（−17%） |
| `qwen_image_vae` | 0.664 → 0.670（+0.9%） | 3.436 → 3.036（−12%） | 8.064 → 7.338（−9%） |

冷、热启动几乎相同（monoload 冷启动多一次 8×8 的形状探测，可以忽略）。高分辨率反而更快：原生每次都要现场分配几十 GiB 的 columns（Flux 4K 的 profile 里 `hipMalloc` 占 4.25 s CPU 时间；SDXL 的 `aten::empty` CPU 总时间从 1344 的 0.41 s 涨到 4K 的 7.4 s），4K 还要经历一次 OOM → 释放缓存 → 重试；分块后每块 ≤ 1 GiB，块之间复用分配器缓存。保留意见：bench 在每次运行前 `empty_cache`，服务里原生的分配开销取决于当时的缓存状态（缓存留着时，原生又会多占几十 GiB GTT）。低分辨率慢 1–4% 是块的启动和拷贝开销。

**精度（monoload vs native；clamp 后的 fp32 像素，范围 [0,1]）：** `native2 vs native` 全部逐位为 0，GPU 是确定的，所以下面的差异全部来自分块。

| VAE | PSNR (dB) | RMSE | p99\|Δ\| | max\|Δ\|：块边界附近 / 其余行 |
|---|---|---|---|---|
| SDXL | 68.3 / 62.5 / 61.2 | 3.8e-4 / 7.5e-4 / 8.7e-4 | 0.00098 / 0.0024 / 0.0025 | 0.0024 / 0.0386；0.0108 / 0.0959；0.0362 / 0.0528 |
| Flux `ae` | 67.5 / 64.9 / 61.5 | 4.2e-4 / 5.7e-4 / 8.4e-4 | 0.0020 / 0.0020 / 0.0024 | 0.0039 / 0.0083；0.0083 / 0.0208；0.0283 / 0.0464 |
| `qwen_image_vae` | 63.5 / 60.5 / 60.3 | 6.7e-4 / 9.4e-4 / 9.6e-4 | 0.0020 / 0.0024 / 0.0024 | 0.0059 / 0.0107；0.0217 / 0.0337；0.0176 / 0.0547 |

* RMSE 全部 ≤ 9.6e-4，即 PSNR ≥ 60 dB，达到 §9.10 的初步目标；通道均值偏移 ≤ 2e-4；没有 NaN/Inf。
* p99 是 0.001–0.0025。bf16 在 |x| ≈ 0.5 处的 ulp 约为 0.002（raw 范围 [−1,1]，换成像素再除以 2），所以 99% 的像素只差 1–2.5 个 bf16 ulp。
* **没有接缝**：块边界附近（每个卷积块边界映射到输出后上下各 4 行）的 RMSE 与其余行相同（SDXL 4K 8.6e-4 对 8.7e-4，Qwen 4K 两者都是 9.6e-4），9 个场景里最大误差都不在边界附近。这与 §9.5 的构造一致：每块读的是真实的相邻行，块边界只改变 GEMM 的形状，不改变参与运算的数据。
* **max|Δ| 的离群点**：像素 max|Δ| 是零星单点，0.008–0.096，超过调研 D.3 里 max ≤ 1e-2 的研发目标。下面的 fp32 参照确认它是 bf16 固有噪声。

**fp32 参照（`bench_vae_fp32_*`，76e56a1，`--modes native,monoload --warm 0 --fp32-ref`）：离群点是 bf16 固有噪声，不是分块引入的。** 同一个 VAE 用同一个加载方式再加载一份 fp32 的（`vae_dtype` 强制为 fp32），解码结果作为真值；原生 fp32 放得下的分辨率（1344、2688）同时跑整图和分块两种，4K 原生 fp32 OOM 后退回 tiled（bench 识别为 `OOM(tiled)` 并中止），改用分块 fp32 作参照。

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

* **分块在 fp32 下与整图等价**：fp32 分块 vs fp32 整图，像素 max|Δ| 2.8e-6～9.7e-6（PSNR 133～139 dB），也就是 fp32 的累加顺序误差。这直接验证了 §9.5 的构造：分块不改变参与运算的数据，只改变 GEMM 的形状。
* **对 fp32 真值，monoload 与原生同样准**：RMSE、PSNR、mean、p99 几乎相同，SDXL 2688 上 monoload 还略好（8.25e-4 对 8.91e-4）。原生自己对 fp32 的 max 误差就有 0.017–0.067，同样超过 1e-2；max ≤ 1e-2 对 bf16 解码本身就不现实，bf16 下应当看 RMSE / p99 和对 fp32 的相对表现。
* 在 |monoload − native| > 0.01 的像素上，monoload 和原生各有约一半更接近 fp32（表最后一列），两者在这些点上对 fp32 的平均误差相当；离群点离块边界的距离（`bdist`）不集中在 0。SDXL 1344 和 2688 最大的离群点都在图像最右边几列（W 方向不分块），而且在这些点上原生离 fp32 更远。所以离群点是两条 bf16 计算路径各自的舍入噪声落在了不同位置：GEMM 形状一变，1 ulp 的差异就会在 30 多层卷积和归一化之间逐层放大到少数像素上。
* fp32 的显存：Qwen 原生 fp32 在 4K 时 GTT 一度到 60.3 GiB（上限 62.5 GiB）后 OOM，monoload fp32 是 20.4 GiB；SDXL / Flux 4K 的 monoload fp32 是 30.9 GiB。monoload 的估算在 fp32 下同样是上界（例如 SDXL 4K 估 33.7、实测 30.9）。
* **原生估算会挤掉其他模型（实测例子）**：这一轮进程里同时加载了 bf16 和 fp32 两份 VAE。原生的估算（bf16 4K 91.86 GiB；fp32 2688 91.45 GiB、4K 183.72 GiB；Qwen fp32 4K 67.98 GiB）超过空闲显存，`load_models_gpu` → `free_memory` 就把另一份 VAE 卸载了：这些原生行的 `unload = 1`，日志里下一次运行前出现 `Requested to load … loaded completely`。monoload 的行全部是 0。服务里常驻 UNet / CLIP 时，被挤掉的就是它们，下一次采样要重新加载。这是 §9.1「`--gpu-only` 下也会走卸载流程」的实测例子。

**结论（2026-10）：第一阶段验收通过。** 峰值主因确认是 im2col 展开缓冲；4K 峰值降到原来的 1/4～1/6，原生 4K「先 OOM 再重试」的边缘状态消失；高分辨率速度不降反升；没有接缝；估算是有效上界，原生的过大估算会挤掉其他模型而 monoload 不会；精度与原生同为 bf16 的水平，max|Δ| 离群点是 bf16 固有噪声，分块在 fp32 下与整图等价。日志计时已改为同步后计时。第二、三阶段（第一层条带解码）的优先级待定：第二层之后 4K 峰值已在 15 GiB 以内，第一层主要的收益是进一步去掉全尺寸激活（SDXL 4K 约 4 GiB 一张）。

### 9.13 第一层：Wan 2.1 VAE 单帧的条带解码（第二阶段）

实现：`monoload/vae_stripe.py`（结构识别、区间倒推、执行计划和内存模型、自检），接入在 `monoload/vae.py` 的 `_decode_layer1`。测试：`tests/test_vae_stripe.py`。

#### 9.13.1 结构核对（62b3c94）

逐项核对了需求里写的结构，结论一致：

* `WanVAE.decode(z)`：`iter_ = 1 + T//2`，T=1 时为 1，`feat_map=None`；`x = conv2(z)`，`decoder(x[:, :, 0:1], feat_cache=None)`，返回 `torch.cat(out_chunks, 2)`，形状 `[b, C_out, 1, 8H8, 8W8]`。
* `CausalConv3d.forward`：`cache_x is None and x.shape[2] == 1` 时走 `super().forward(x, autopad="causal_zero")`，comfy Conv3d 把时间 kernel 截成最后一片，空间 padding 由模块自己的 `padding=(0, k//2, k//2)` 负责，没有整张 pad 的副本。
* `Decoder3d.forward` 在 `feat_cache=None` 时对 conv1、middle、head 都是 `layer(x)`；`run_up` 对 upsamples 也是 `layer(x)`，拆帧分支要求 `upsample3d` 且 `x.shape[2] > 2`，T=1 永远不走。`Resample` 在 `feat_cache=None` 时不执行 `time_conv`，`rearrange` 成 4D 后 `nn.Upsample(scale_factor=2, mode='nearest-exact')` + `Conv2d(3×3, padding 1)`。
* `nearest-exact` 在 scale 2 时，输出行 i 取输入行 `⌊(i + 0.5)/2⌋ = ⌊i/2⌋`，与 nearest 相同。
* `RMS_norm`：`F.normalize(x, dim=1) * sqrt(C) * gamma (+ bias)`，逐位置，沿通道。`ResidualBlock`：`residual(x) + shortcut(x)`，residual 为 RMS → SiLU → conv3×3 → RMS → SiLU → Dropout → conv3×3，shortcut 为 Identity 或 1×1 CausalConv3d。
* dim=96、dim_mult [1,2,4,4]：通道 384 → 384 → (×2 上采样到 192) 192→384 → 384 → (192) → 192 → (96) → 96，head 96→3，与需求一致。

所以单帧解码就是一串模块依次调用 `conv2 → decoder.conv1 → middle[*] → upsamples[*] → head[*]`，条带解码直接在这串模块上做，不需要改 Wan 的任何代码。

#### 9.13.2 拆点与存档

* **前缀（整图）：** `conv2`、`decoder.conv1`、`middle`（残差块、全局注意力、残差块）和 upsamples 里第一个 Resample 之前的所有模块（最低分辨率的 3 个残差块），都在 H/8 上整图计算；注意力继续用第二层的 query 分块。
* **存档：** 前缀的输出，即第一个 Resample 的输入（H/8，384 通道；4K 时 95 MiB，bf16）。
* **为什么放在这里：** 这是调研推荐的拆点，核对后仍然认为最合适。往后挪一个分辨率（H/4 末尾），存档变成 4 倍（380 MiB），而且 H/4 上 384 通道的残差块整图计算本身就要 2 GiB 左右，超出 2–4 GiB 的目标；往前挪（middle 之前）则全局注意力进入条带，不可能。H/8 存档的代价是 H/4 那一级的重算比例最大（halo 在 H/4 上每侧约 12 行），但它在总算量里只占约 1/4（各级卷积算量大致为 H : H/2 : H/4 ≈ 74 : 74 : 55），整体重算在常用条带高度下是 1.05–1.26 倍（§9.13.4）。

#### 9.13.3 区间倒推与「精确行」

条带部分拆成单元（unit），每个单元记下 halo（在它的输出分辨率上每侧需要多算几行）和 scale（输出行 / 输入行）：

| 单元 | halo | scale | 倒推 `need_in([a, b))` |
|---|---|---|---|
| ResidualBlock（两个 3×3 卷积；shortcut 是逐点的） | 2 | 1 | `[a−2, b+2)` ∩ 图像 |
| Resample（nearest×2 + 3×3 卷积） | 1（在输出分辨率上） | 2 | 先 `[a−1, b+1)` ∩ 输出图像，再 `[⌊c0/2⌋, ⌈c1/2⌉)` ∩ 输入图像 |
| head 的 3×3 CausalConv3d | 1 | 1 | `[a−1, b+1)` ∩ 图像 |
| RMS_norm、SiLU（head） | 0 | 1 | 不变 |

一条输出条带 `[o0, o1)` 从 head 往回逐个单元推出每个单元需要的输入行（`stripe_needs`），所有区间都是全局行号；4K 时推到存档上是每侧约 7 行。残差块两条支路的需求取并集，就是 residual 支路的需求（shortcut 是逐点的），所以整个残差块按一个单元处理。

**执行（和需求的写法不同之处）：** 每个单元都是原模型里的模块实例，**原样**调用在一段行的切片上，不改它的 padding，也不绕过模块的 forward（第二层在 `_conv_forward` 这一级把单帧 Conv3d 换成等价的 `F.conv2d`，见 §9.13.9，与条带无关）。模块会对切片的上下边照常补零；凡是受切片内部边界补零影响的输出行，恰好是每侧 halo 行。`valid_out` 按切片的全局起止行算出哪些输出行是精确的：靠近不是图像真实边缘的切片边的 halo 行不精确，真实边缘处的补零与整图相同。每个单元运行后都检查「下一步需要的行 ⊆ 精确的行」，不满足就是内部错误（`StripeError`），然后只保留需要的那些行（按全局坐标裁剪）。所以：

* 补零只在图像真实的上下边缘进入结果，条带内部边界上用的都是真实的相邻行（halo）；
* 模块、权重、cast / weight_function、第二层的卷积分块全都原样生效，条带里的大卷积仍受工作区约束；
* 代价是每个卷积在切片两侧多算几行随后丢掉的输出（残差块每侧最多 3 行、卷积 1 行），在常用条带高度下是 1–3% 的额外算量（已计入下面的重算比例）。

需求里设想的是「按块位置决定是否补零、内部边界不补零」，也就是复用第二层改 padding 的逻辑。这里改为「原样调用 + 丢掉受影响的行」：效果相同，不碰模块属性，而且对残差块这样的复合模块也能直接用（不必拆开它的 forward）。正确性由逐单元的精确行检查和测试保证（§9.13.7）。

每条带最后的核心行用 `copy_` 直接写进预先分配的输出缓冲（输出 dtype，相当于原生的 `.to(dtype)` + `copy_`），整张输出只有这一份，没有整张的中间结果。`process_output` 在一个样本写完后对它原地执行，和原生相同。

#### 9.13.4 内存模型、分配器、默认条带高度

（第二阶段真机验收（README §10.2）后按实测重写，725a010 复测后又重写了分配器部分；与 4e54d20、725a010 的对照写在各段里。）

**估算（`Plan`，全部按形状事先算出）：**

```
live     = 常驻 + max(前缀, 存档 + 条带)          张量本身同时存活的峰值
  常驻 = 输出缓冲（整个 batch，输出 dtype，输出设备就是 VAE 设备时） + 一个 latent 样本
  前缀 = max over 前缀模块 peak(整图 H/8)
  条带 = max over 条带、单元 peak(该单元的切片)
arena    = live + live/32 + 64 MiB（只有一条条带时 live/8），向上取整到 2 MiB      解码在其中运行，= reserved
estimate = arena + largest + 16 MiB            交给 load_models_gpu，reserved 的上界
  largest = 这次解码里最大的单个分配（im2col columns 块 ≤ 工作区、最大的激活平面、上采样结果、qkv、输出缓冲）
```

peak 按 62b3c94 的 forward 逐个数同时存活的张量。S 是模块输入的存储（调用期间由调用方持有；条带里是上一个单元的整份输出，模块拿到的是它的行切片），A / B 是模块所处理的那些行上一张输入 / 输出通道的平面，e 是元素字节数：

| 模块 | 依次出现的存活集合（取最大） |
|---|---|
| RMS_norm | `F.normalize(x) * scale * gamma + 0`：任何时刻两个临时量 → S + 2A |
| ResidualBlock | RMS 2A；SiLU 输出 + conv1 输出 + conv1 额外 → A + B + X₁；对 conv1 输出做 RMS 3B；conv2 输入 + 输出 + 额外 2B + X₂；`x + shortcut(old_x)` 2B（1×1 shortcut 卷积时 2B + 它的额外，相加时 3B）→ S + max(…) |
| Resample | 切片的连续化拷贝 A + 上采样结果 4A；上采样结果 + 卷积输出 4B + 额外 → S + max(5A, 4A + 4B + X) |
| head 的 3×3 卷积 | 输出 + 额外（输入是行切片，不连续：未分块时整份拷贝）→ S + B + X |
| AttentionBlock（前缀） | norm 输出 P + qkv 3P（q/k/v 局部变量一直持有到返回）+ 注意力输出 P + 分数块 → S + 5P + 分数；SDPA / xformers 的 q/k/v/输出连续化拷贝再加 3P |

卷积的「额外」X 按第二层的包装实际怎么跑来算：整层 columns ≤ 工作区时不分块，额外 = 整层 columns（+ 输入不连续时的整份拷贝）；否则按行分块，额外 = 一块的 columns（≤ 工作区）+ 这一块的输入拷贝 + 这一块的输出；再加截断后权重的连续化拷贝。分数块 = 2 × query 块行数 × N × e（≤ 工作区）。

**与实测对照（4K，bf16，工作区 384 MiB；实测是 bench 的 alloc 增量，含存档和输出缓冲）：** 432 行 1.70 / 1.70 GiB，240 行 1.24 / 1.24，128 行 0.97 / 0.97，前缀 958 MiB / 0.94 GiB（≤ 64 行时的实测峰值就是前缀）；1344×768 单条带 1131 MiB / 1.11 GiB；2688 三条 512 行 1.42 / 1.42。各项都在 bench 两位小数的精度（±5 MiB）以内。432 行时 profile 记下的峰值构成（上采样结果 630 MiB、卷积输出 315、columns 384、块拷贝 43 + 21、输入 160、存档 95、输出 95）与模型逐项对得上。第一版模型（RES 2Cin+4Cout、Resample 6Cin+4Cout、注意力 10C，外加 2 × 工作区）把这些量大约数了两遍：4K 估算 2.78 GiB（实测 alloc 1.70），短条带 1.77 GiB（实测 0.94）。

**分配器（reserved 才是 GTT 里真正占掉的）。** 张量的存活量（alloc）模型与真机一致，但 reserved 由 PyTorch 的缓存分配器决定：释放的块留在缓存里，只有尺寸放得下的请求才能复用；块只能和同一个 segment 里相邻的空闲块合并，每个新尺寸的请求在没有合适空闲块时就单独 hipMalloc 一个 segment。三个版本在 4K 上的实测：

| 版本 | 处理 | 32 行 | 64 行 | 128 行 | 155 行 | 240 行 | 432 行 |
|---|---|---|---|---|---|---|---|
| 4e54d20 | 不清缓存，自上而下 | 1.07 | 1.07 | 1.07 | — | 1.44 | 2.66 |
| 725a010 | 前缀后清缓存，最大条带先跑 | 1.07 | 1.14 | 1.36 | 1.44 | 1.46 | 2.09 |
| 本版 | arena（模拟） | 1.13 | 1.13 | 1.13 | 1.13（144 行） | 1.35 | 1.82 |

（reserved 增量，GiB；alloc 在三个版本里相同：0.94 / 0.94 / 0.97 / 1.04 / 1.24 / 1.70。）

用 `tests/alloc_sim.py`（§9.13.10）把这些读数逐个复现出来（24 个真机读数，误差 ≤ 0.02 GiB），再看 segment 的布局，原因是：

* **4e54d20：** 前缀在缓存里留下 2 个 384 MiB（整图卷积的 columns）和几个 96 MiB 的 segment。条带 ≤ 128 行时，条带要的块全都能从这些 segment 里切出来，reserved 停在前缀的 1.07；条带高了，上采样结果（432 行时 630 MiB）等更大的块放不进任何已有的 segment，只能另开，叠在前缀之上（2.66）。
* **725a010：** 前缀后清空缓存，条带阶段从零开始，每种尺寸的块各开 segment。分块卷积的输出缓冲是在第一块算完之后才分配的，恰好落进第一块的 columns 刚释放的洞里（例如 128 行时一个 356 MiB 的 segment 变成 158 空 | 62 输出 | 137 空），下一块的 384 MiB columns 放不进去，只能再开一个 segment（1.36）。高条带时则因为不再叠在前缀之上而受益（2.66 → 2.09）。
* 所以「清不清缓存」本身不是关键：两种做法都让条带阶段的块落在为别的尺寸开的 segment 里，碎片随条带高度、分辨率、dtype 不规则地变化（同一个模拟里，清 / 不清、两种条带顺序四种组合各有输赢，reserved 是 alloc 的 1.1–1.7 倍）。

**对策（本版）：**

* **arena：** 解码开始时先分配一个 `arena` 字节的块并立刻释放（`reserve_arena`），缓存里就有一个这么大的空闲 segment，之后前缀和条带的所有张量都从这一个 segment 里按 best fit 切出来，释放后在 segment 内合并回去，不再出现「每种尺寸一个 segment、彼此不能合并」。前缀和条带之间不再清缓存：条带直接复用前缀用过的空间（4e54d20 短条带的好处），又不会叠出新的 segment（725a010 高条带的好处）。只在默认的分配器配置下使用（CUDA/HIP、native 后端、没有 `max_split_size_mb`、没有 `expandable_segments`），否则不分 arena，行为与以前相同。arena 这个块在 `max_memory_allocated` 里也算一次，所以 bench 的 alloc 列现在约等于 arena；`bench_vae.py --no-arena` 可以看张量本身的峰值。
* **分块卷积先分配输出：** `_ConvChunker` 在第一块之前就分配好整层的输出（形状由输出尺寸算出，dtype 同输入；万一第一块返回的 dtype 不同就按它重新分配），之后每块的 columns 都能复用同一个洞。结果逐元素不变（只是分配顺序）。第二层也一样受益。
* **上采样单元的输入先做连续拷贝：** 条带里传给 Resample 的是上一个单元输出的行切片（非连续），上采样会在内部再拷一份；改为由我们先 `contiguous()`（上一个单元的整份输出随之释放），内部不再拷贝。存活量不变或更小，碎片少一些。
* 最大的条带先跑、自检后清空缓存（725a010 加的）保留；725a010 的「前缀后清缓存」去掉。

**arena 要多大：** 用模拟器对每个计划二分查找「不溢出（不另开 segment）的最小 arena」：输出缓冲前置之后，多条带的计划只需要 live 的 0.90–1.022 倍（4K 默认 0.996，1344 / 2688 默认 1.010–1.011，8K 1.001），只有一条条带（整图当一条）时要到 1.10–1.12 倍（整图尺寸的激活平面 1–2 GiB，夹在中间的洞放不下下一张）。所以 arena = live + live/32 + 64 MiB（一条条带时 live/8）。

**估算为什么还要加 largest：** 在 384 个计划的网格（512² … 8K，bf16 / fp32，工作区 384 / 192 / 128 MiB，默认策略和 32 … 整图的强制高度）里，默认计划从不溢出；但一部分高条带的计划会溢出，而且每次都恰好是**一个**请求找不到连续的洞，于是单独开一个 segment：要么是一块 columns（例如 2688、384 行、bf16：洞 383 MiB，请求 384 MiB），要么是一张全分辨率平面（8K、fp32、480 行：1389 MiB）。之后同尺寸的请求都复用这个 segment，不会再长。所以上界取 `arena + largest + 16 MiB`（16 MiB 给 ≤ 1 MiB 的小块池，实测 2–6 MiB）：384 个计划里 reserved 全部 ≤ 估算，最小余量 16 MiB（fp32 1344、2 条 384 行，溢出被 largest 盖住）。代价是估算比实际 reserved 高一个 largest（默认计划约 0.39 GiB），但这只影响交给 `load_models_gpu` 的数，不多占内存；实际占用就是 arena。

**默认条带高度：在不明显变慢的前提下尽量低峰值。** 第一版是「3 GiB 预算内取最高的条带」：4K 选出 5 条 432 行，估 2.78 GiB，实测 reserved 2.66。真机扫参（4K，热启动；8% 以内是顺序 / 温度噪声）：

| 核心行数 | 32 | 64 | 128 | 240 | 432 |
|---|---|---|---|---|---|
| 耗时 (s) | 11.7 | 9.8 | 8.8 | 9.2 | 8.1–8.9 |
| reserved (GiB) | 1.07 | 1.07 | 1.07 | 1.44 | 2.66 |
| 重算 | 2.09× | 1.54× | 1.26× | 1.13× | 1.07× |

128 行与 432 行一样快，峰值已到下限（由整图前缀决定）；64 行起明显变慢。725a010 复测（README §10.2）：r128 8.38 s、r155 8.26 s、r256 8.00 s、r512 7.78 s，r64 9.16 s、r32 10.65 s，结论不变。所以规则是：

* **目标 = `DEFAULT_POLICY_ROWS`（128）行条带的峰值（arena，也就是实际 reserved）；在 arena 不超过目标的前提下取最高的条带。**

  arena 随条带高度单调不减，等价于先算 128 行的 arena，再在 `[8, H]` 上二分查找 arena 不超过它的最高核心高度。含义：

  * 条带阶段的峰值低于前缀时（大图），128 行和更高的条带峰值一样（都是前缀），就用更高的条带，少重算，不多占内存。
  * 条带阶段的峰值高于前缀时（小图），就是 128 行，不再往下切（再矮就慢）。
  * 图像不足 128 行时就是一条整图。

  （725a010 用的是估算而不是 arena 来选；估算现在多了 largest 这一项，它随条带高度跳变，不适合用来比较峰值，所以改用 arena。）
* 工作区沿用 384 MiB（`min(MONOLOAD_VAE_WORKSPACE, 384 MiB)`）。模拟显示更小的工作区会明显压低默认计划的峰值（128 MiB 时 4K 默认 0.87 GiB，2688 0.56，1344 0.36），但卷积块数会变成约 3 倍，速度要真机量（`bench_vae.py --workspace 128M`），这一轮不改默认。
* 设了 `MONOLOAD_VAE_BUDGET` 时改用预算：取**估算**不超过预算的最高条带（预算是给 `load_models_gpu` 的上界，所以和估算比；工作区 = `min(MONOLOAD_VAE_WORKSPACE, max(64 MiB, 预算/8))`），8 行条带也放不下就抛 `MonoloadError`，写明需要多少。设了 `MONOLOAD_VAE_STRIPE_ROWS` 时强制那个高度（覆盖默认规则和预算，估算超过预算时日志注明）。

全尺寸 Qwen 结构（dim 96，输出在 GPU 上）按本版算出的默认计划（`vae.choose_plan`；「reserved」是模拟器的结果，= arena + 小块池 2 MiB）：

| 输出 | dtype | 默认计划 | 重算 | live | arena = reserved | 估算 | 725a010 实测 reserved / 估算 |
|---|---|---|---|---|---|---|---|
| 1344×768 | bf16 | 6 条 128 行 | 1.23× | 595 MiB | 678 MiB（0.66 GiB） | 1.05 GiB | 0.70 / 0.74 |
| 2688×1536 | bf16 | 12 条 128 行 | 1.25× | 793 MiB | 884 MiB（0.86 GiB） | 1.25 GiB | 1.02 / 0.95 |
| 3840×2160 | bf16 | 15 条 144 行 | 1.22× | 1053 MiB | 1152 MiB（1.13 GiB） | 1.52 GiB | 1.44 / 1.25（14 条 155 行） |
| 7680×4320 | bf16 | 13 条 333 行 | 1.10× | 3058 MiB | 3218 MiB（3.14 GiB） | 4.27 GiB | — |
| 1344×768 | fp32 | 6 条 128 行 | 1.23× | 734 MiB | 822 MiB | 1.19 GiB | 0.96 / 0.89 |
| 2688×1536 | fp32 | 12 条 128 行 | 1.25× | 1084 MiB | 1184 MiB | 1.55 GiB | 1.41 / 1.27 |
| 3840×2160 | fp32 | 14 条 155 行 | 1.21× | 1627 MiB | 1742 MiB | 2.27 GiB | 2.07 / 1.90（13 条 167 行） |

强制高度（bf16，4K）：r32 / r64 / r128 都是 1.13 GiB（前缀决定），r256（9 条 240 行）1.35，r512（5 条 432 行）1.82（725a010 实测 2.09，4e54d20 2.66）。

（「重算」是整个 decoder 的卷积算量相对整图解码，前缀只算一次。）

#### 9.13.5 OOM

条带高度和工作区一起减半，重新做计划、重跑整个解码（前缀在 H/8 上，重跑代价小）；高度到 8 行（或强制的更小值）且工作区到 64 MiB 仍 OOM，就抛 `MonoloadVAEOOMError`。不退回 tiled，也不退回第二层：第二层的峰值更高，退回去没有意义。arena 在 `run` 的开头分配，它 OOM 时和解码中途 OOM 一样处理（重试时 arena 随计划变小）。测试里替换 `run_stripes` 模拟 OOM，确认 `decode_tiled_` 和第二层的 `_run` 都没有被调用。

#### 9.13.6 识别、自检、开关、日志

* **识别（`wan_structure`，按真实结构，不看文件名）：**
  * `type(first_stage_model) is WanVAE`、`type(decoder) is Decoder3d`（精确类型，子类不算）；
  * `conv2` 是 1×1、`conv1` 是 3×3 的 CausalConv3d：stride 1、dilation 1、groups 1、zeros、padding `(0, k//2, k//2)`；
  * upsamples 只含 ResidualBlock 和 Resample（`upsample2d` / `upsample3d`、`nearest(-exact)` ×2、3×3 / stride 1 / padding 1 的 Conv2d），至少有一个 Resample，没有 AttentionBlock；
  * 每个 ResidualBlock 的 residual 正好是 `[RMS_norm, SiLU, CausalConv3d(3), RMS_norm, SiLU, Dropout, CausalConv3d(3)]`，shortcut 是 Identity（输入输出通道相同）或 1×1 CausalConv3d；
  * RMS_norm 都是 channel_first，Dropout 处于 eval 或 p=0；head 是 `[RMS_norm, SiLU, CausalConv3d(3)]`；middle 只含 ResidualBlock / AttentionBlock；
  * 模型的任何模块上都没有 forward hook 或实例级 `forward` 替换（例如 bypass LoRA 注入），全局 module hook 也没有；
  * latent 是 5D、T=1、通道数等于 conv2 的输入通道；没有 vae_options。
  * 任何一项不符就不匹配，走第二层；对每个模型实例打一次日志写明原因（例如 `upsamples[14].residual[5]: Dropout p=0.5 in training mode`、`first-stage model is AutoencoderKL, not comfy.ldm.wan.vae.WanVAE`）。Wan 2.2 的 48 通道 VAE（`comfy.ldm.wan.vae2_2.WanVAE`）类型就不同，不匹配。
* **自检（`self_test`）：** 每种结构（签名：z 通道、dim、dim_mult、残差块数、各单元的类型和通道）在本进程第一次使用时运行一次，结果缓存：
  * 用模型的配置重新构造 `conv2` 和 `Decoder3d`，以 fp32 加载模型的权重（不复制 encoder，测完释放；全尺寸 Qwen 约 0.35 GB）。重新识别这个副本，确认结构签名一致。
  * 24×24 latent（固定种子），参照是原生的 `WanVAE.decode(副本, z)` 整图解码；条带解码强制 40 行（192 行输出，5 条，最后一条 32 行），工作区 8 MiB，让条带内的大卷积也走第二层的分块。
  * 通过条件：输出全部有限，且 `max|条带 − 整图| / max(1, max|整图|) ≤ 1e-4`。fp32 下实测在 1e-6 量级或为 0（测试模型），而 halo 少算一行的错误在 1e-1 量级，阈值两边都有两个数量级以上的余量。
  * 在 fp32 下比较，不会被 bf16 噪声误判；参照用的是 Wan 自己的 `decode`，所以 Wan 的 forward 将来若有变化（例如单帧路径改了），自检也能发现。
  * 全局随机数状态不变（`fork_rng`）；需要的显存在启动自检前单独报给 `load_models_gpu`（副本权重 + 工作区 + 256 MiB）。自检的张量全在 `_self_test_run` 里，返回后全部失效，再 `gc.collect()` + `soft_empty_cache(True)`：第一次解码从和之后一样的缓存状态开始（第一版没有清，4K 冷启动 reserved 比热启动多 0.38 GiB，超过了估算）。
  * 不通过（数值不符、内部错误或任何异常）：这种结构在本进程内禁用第一层，改走第二层，日志里用醒目的警告写明原因。测试里故意把残差块的 halo 少算一行：单元检查会抓到（`StripeError`）；若连精确行的规则也一起写错，数值比较会抓到（误差 0.086）。两种情况都回退到第二层，结果与原生相同。
* **开关：** `MONOLOAD_DISABLE_VAE_STRIPE=1` 关掉第一层（所有受管理的解码走第二层）；`MONOLOAD_VAE_BUDGET`（设了就取预算内最高的条带，不设用默认策略）；`MONOLOAD_VAE_STRIPE_ROWS`（强制条带高度，优先于前两者）。`MONOLOAD_DISABLE_VAE=1`、`MONOLOAD_EXACT=1` 时 VAE 走原生，`MONOLOAD_DISABLE=1` 时什么都不装，都不变。
* **日志：** 启动时写明第一层开 / 关和策略（默认策略或预算）；每次解码写明用了哪一层，第一层写出条带数、核心高度、重算比例、存档大小、策略（`default: peak of 128-row stripes` / `MONOLOAD_VAE_BUDGET` / `forced N rows`）、工作区、估算和耗时，第二层写出没用第一层的原因。

#### 9.13.7 测试（`tests/test_vae_stripe.py`，CPU，fp32，不需要模型文件）

* 区间：`need_in` / `stripe_needs` 与暴力依赖展开逐条对照（300 条随机单元链、993 条条带，0 处不符）；`split_rows` 均分、覆盖全图。
* 精确行：残差块（有 / 无 1×1 shortcut）、两种 Resample、head 卷积，在 17 行输入的每一个切片上，`valid_out` 给出的行与整图结果一致（≤ 3.6e-7），紧邻的下一行在内部边界上不一致，也就是 halo 不多不少。
* 整个 decoder（ComfyUI 的 `WanVAE`，dim 16，随机权重，经 `comfy.sd.VAE`）：条带高度 1（96 条）、7（不整除）、40、整图、按预算自动选（4 条 24 行）、默认策略（96 行的图 1 条，320 行的图 3 条 107 行）；13×9 奇数尺寸、1×1 和 3×5 的很小 latent、batch 2、条带内再分块（工作区 16 KiB）、bf16。与原生整图解码的 raw 输出差 ≤ 2e-6（fp32；bf16 为 0），条带边界附近 ±4 行的误差不比其他区域大；像素的形状、dtype、设备、范围与原生一致。
* 识别、开关、自检、预算、OOM：见上文各节；SDXL 式 / Flux 式结构继续走第二层（日志原因是「不是 WanVAE」），与原生的差和第一阶段相同。
* 默认策略与内存模型：目标等于 128 行条带的 arena、选出的条带不超过目标且再高一档（160 行）就超过；`MONOLOAD_VAE_STRIPE_ROWS` 优先于默认策略和预算，只设预算时取预算内最高的条带；三种工作区下估算随条带高度单调不减，live = 常驻 + max(前缀, 存档 + 条带)、arena 按规则、估算 = arena + largest + 16 MiB，最大的条带排在第一个。
* 分配器：自检后清空一次缓存，解码过程中不再清；在 CPU 上强制打开 arena 时，batch 2 的解码只分配一次 arena（大小等于计划的 arena），结果不变；CPU 上、以及 `max_split_size_mb` / `expandable_segments` 配置下不用 arena。
* 缓存分配器（`tests/alloc_sim.py`，全尺寸 Wan decoder 放在 meta 设备上，§9.13.10）：模拟器复现 4e54d20 / 725a010 的 5 个真机读数（误差 ≤ 0.03 GiB）；本版 9 个计划（1344 默认和整图、1920×1088、2688、4K 默认 / 32 行 / 512 行、fp32 4K、8K）的模拟 reserved ≤ 估算。
* 单帧 Conv3d 改走 conv2d（在 CPU 上强制打开闸门测）：3×3（causal_zero）和 1×1 shortcut（输入是行切片）的 CausalConv3d，不分块和分块两种工作区，与模块原样输出一致（≤ 1e-5），且确实走了 conv2d；T=3 的输入、实例上已有别人的 `_conv_forward` 替换时不改走；整个 decoder 打开闸门后第一层仍与原生一致；CPU 张量上闸门关闭。
* `tests/test_vae.py`（第二层）在关掉第一层后照旧全部通过，`tests/test_entry.py` 在各开关组合下通过。

#### 9.13.8 限制与待真机确认

* 只覆盖 Wan 2.1 VAE 的单帧解码。多帧视频仍交给原生；LDM decoder（SDXL、Flux）在第三阶段之前继续走第二层。
* 前缀整图计算，4K 的峰值下限是 alloc 0.94 / reserved 1.07 GiB（实测），主要是全局注意力；8K 时前缀存活约 2.6 GiB。
* 存活量的模型与真机 alloc 吻合（§9.13.4）；arena 的大小和估算里的 largest 是用分配器模拟器在 384 个计划上验证的（0 处 reserved > 估算），不是数学证明；模拟器本身用 24 个真机读数校验过。bench 每行仍打印估算和实测 reserved，用来复核。
* 估算比实际 reserved 高一个 largest（默认计划约 0.39 GiB）：这是给「一个请求被碎片挤出 arena」留的余量，只影响交给 `load_models_gpu` 的数。
* 依赖 PyTorch 缓存分配器的默认行为（best fit、同 segment 内拆分合并）。配置了 `max_split_size_mb` / `expandable_segments` 或换了分配器后端时不用 arena，估算不再有模拟器的验证。
* 条带越矮重算越多；条带内的卷积在切片两侧多算几行随后丢掉（已计入重算比例）。
* 第一版（4e54d20）已在真机上验收（README §10.2）：4K GTT 59.2 → 2.7 GiB，速度与原生相同，精度与原生同为 bf16 水平，没有接缝，SDXL 仍走第二层。
* 725a010（默认策略、conv2d 改道）已复测（README §10.2）：计划、速度、精度、`aten::fill_` 都符合预期；reserved 超过估算、矮条带被清缓存反而变差，本版处理。
* 本版待真机确认（README 9.7「第二阶段第三轮复测」）：每一行（含冷启动）的 reserved ≤ 估算；默认计划的 reserved 与模拟一致（1344 约 0.66、2688 约 0.86、4K 约 1.13 GiB）；4K 扫参里 r32 / r64 / r128 都是约 1.13、r512 约 1.82；速度与 725a010 相同；精度不变（分块卷积先分配输出、上采样输入先连续化都不改变数值）。

#### 9.13.9 单帧 Conv3d 改走 conv2d（`aten::fill_`）

真机 profile 里一次 4K 解码有 159308 次 `aten::fill_`（GPU 时间可忽略，CPU 上是 15.9 万次 kernel 启动），和 1097 次 `aten::resize_`（173 GB）。来源是 PyTorch 的卷积后端选择：CUDA/HIP 上关掉 cuDNN（MIOpen 也由这个开关控制）后，4D 输入走 Slow2d，5D 输入一律走 SlowDilated3d（`_select_conv_backend`：`input.ndimension() == 5 && input.is_cuda()`）。SlowDilated3d 的前向（`DilatedConvolution.cu`）对每个样本先 `output_n.select(0, n).fill_(bias[n])` 逐个输出通道设 bias（bias[n] 是 GPU 上的 0 维张量，`fill_` 里再发一个 `copy_`），再 vol2col（`columns.resize_`）+ GEMM（beta = 1）。每次调用的 `fill_` 次数等于输出通道数：159308 / 1097 ≈ 145，正好是这个 decoder 各卷积输出通道数（384 / 192 / 96 / 3）的平均量级。条带越多，卷积调用越多，128 行条带时会到 50 万次左右。

改法（`vae_ops._ConvChunker._conv`，第一层和第二层共用）：Conv3d 的调用满足下面全部条件时，改成在第 0 帧上调 `F.conv2d(x[:, :, 0], weight[:, :, -1], bias, stride/padding/dilation 的 H/W 部分, groups)`，再 `unsqueeze(2)`：

* 输入在 CUDA/HIP 上、`torch.backends.cudnn.enabled` 为 False、没有开 comfy 的 `NVIDIA_MEMORY_CONV_BUG_WORKAROUND`，即 3D 调用本来就会走 SlowDilated3d；
* 输入是单帧（T=1），模块的时间 padding 是 0，且时间 kernel 实际为 1（`autopad="causal_zero"` 时 comfy 本来就把权重截成最后一片；或者 kT 本身就是 1）；padding_mode 是 zeros，padding 是整数元组；
* 这个模块的 `_conv_forward` 是 torch 的或 comfy.ops 的（类上的原方法，实例上没有别人的替换），这样跳过它不会丢掉任何别的逻辑（comfy 的版本只做 causal_zero 截断和 NVIDIA 的 workaround）。

为什么结果不变：Slow2d 的前向是 `output.copy_(bias)` 后对每个样本 im2col + `gemm('n','n', HW, Cout, Cin·kh·kw, 1, columns, weight, beta=1, output)`；SlowDilated3d 在 kT=1、T=1 时的 vol2col 产生同样的 columns，GEMM 的参数完全相同。所以两条路径在同一块 GPU 上应当逐位一致（1×1 卷积时 Slow2d 直接用输入当 columns，数值也一样）。区别只是 bias 用一次 `copy_` 设好，而不是每个通道一次 `fill_`；1×1 卷积还省掉了 columns。CPU 上闸门不打开（CPU 的 Conv3d 不走 SlowDilated3d），测试里强制打开后与模块原样输出一致到 1e-5（§9.13.7）。

真机复测（725a010，README §10.2）确认：4K 默认计划的 profile 里 `aten::fill_` 从 159308 次降到 9396 次（CUDA 合计约 57 ms）；第二层（单帧 Wan 解码同样改道）的精度数字与改道前逐位相同；第一层 fp32 与整图仍只差 7.3e-6 / 8.1e-6；速度普遍快了 5–12%（4K 扫参各高度、第二层 4K 7.14 → 6.91 s）。


#### 9.13.10 缓存分配器模拟（`tests/alloc_sim.py`）

reserved（GTT）由 PyTorch 的缓存分配器决定，不能只靠存活量模型推。`tests/alloc_sim.py` 在 CPU 上复现它：

* **分配器（`AllocatorSim`）：** 按 c10/cuda/CUDACachingAllocator.cpp 的规则和锁定镜像里 `c10/core/AllocatorConfig.h` 的常数：请求向上取整到 512 B；≤ 1 MiB 走小块池（2 MiB segment），1–10 MiB 开 20 MiB 的 segment，更大的按 2 MiB 取整单独开；从同一个池里 best fit（不小于请求的最小空闲块，同样大小取低地址）；大块池里拆分后剩余 > 1 MiB 才拆；释放时与同一 segment 里相邻的空闲块合并；`empty_cache` 释放整个空闲的 segment。分配器配置取默认（没有 max_split_size、expandable segments、垃圾回收阈值）。
* **轨迹（`Tracer`）：** 在 meta 设备上构造全尺寸的 Wan decoder（qwen_image_vae 的维度），在 `TorchDispatchMode` 下跑 Monoload 的第一层或第二层：每个 aten 算子新产生的存储算一次分配，一个存储的所有张量和视图都消失时算一次释放（`StorageWeakRef`）。GPU 内核内部的缓冲按内核代码补上：Slow2d（关掉 cuDNN 的 4D 卷积）和 SlowDilated3d（5D）对非连续的输入 / 权重各拷一份，分配输出，再分配 im2col / vol2col 的 columns（Slow2d 的 1×1、stride 1、无 padding 不需要）；上采样对非连续输入拷一份。权重和 CPU 上的 latent 不计（bench 测的是解码前后的增量）。
* **开关：** 可以复现以前的版本：`clear`（前缀后清缓存，725a010）、`order`（自上而下，4e54d20）、`conv2d`（单帧 Conv3d 走 SlowDilated3d，4e54d20）、`arena=0`（不分 arena）、`out_first=False`（分块卷积在第一块之后才分配输出，725a010 之前）。
* **校验：** 24 个真机读数（4e54d20 和 725a010 的 4K 扫参、1344 / 2688、fp32 三档、第二层三档），alloc 和 reserved 的误差都 ≤ 0.02 GiB；峰值时刻的存活块与 profile 的「峰值构成」逐项一致（例如 725a010 4K 155 行：384 / 242 / 121 / 95 / 95 / 63 / 43 / 21 MiB）。

用法：`python tests/alloc_sim.py`（校验表），`python tests/alloc_sim.py --res 3840x2160 --rows 32,128,512 [--version v1|v2] [--peak] [--segments]`（某个计划的 alloc / reserved、峰值时刻的存活块、reserved 峰值时的 segment 布局）。在锁定镜像里用 `tests/docker_run.sh` 跑，不需要模型文件；4K 的一次模拟约 5 秒。