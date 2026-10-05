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
* **可以关。** `MONOLOAD=0`（总开关，§10）：方法照装，但每个替换的方法先判断 `_active()`，不启用就原样调用原方法，行为与原生逐位一致。`MONOLOAD_DISABLE=1`：插件不调用 `install()`，什么都不装。`uninstall()` 恢复原方法（测试里用来在同一进程中对比原生和 Monoload；调用前要先卸载所有模型）。

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

每个模型有一个**绑定**（`_Binding`，放在 `patcher.model` 上，所有 clone 共用；review 01）：当前生效的 patcher 的 `patches` 和合并方式（exact）、当前生效的 `hook_patches`（key → hook patch 列表）、LoRA 张量的设备副本。模型上所有运行时 patch 每次调用都读它。`load` / `partially_load` / 装运行时 patch 时绑定指向正在加载的 patcher：同一个 `patches_uuid` 的 clone 加载时原生 `partially_load` 不卸载、权重全部已加载时也不调用 `load()`（只重新应用它的 forced hooks 就返回），所以要在调用原函数**之前**把绑定指向它。hook 状态由最后一次 `patch_hooks` / `unpatch_hooks` 的那个 clone 写，与原生一致（原生的 clone 共用权重和 `hook_backup`）。以前每个 patcher 一份状态、运行时 patch 绑着装它的那个 patcher，同 uuid 的 clone 会用前一个 clone 的 hook 强度、patches 和合并方式（CLIP 路径用 ComfyUI 自带的节点就能遇到，差 0.02；`tests/test_lora_clone_binding.py`）。没有运行时 patch 时绑定清空，不留住已经不用的 clone 的东西。

`patch_hooks(hooks)` 用原生的 `get_combined_hook_patches(hooks)` 算出组合（包括 keyframe 强度），写进绑定；只被 hook 改到、还没有运行时 patch 的层补挂一个，不再生效的 hook-only patch 摘掉。**不写权重、不备份、不缓存。** 采样时正/负条件可能挂着不同的 hook 组，每一步会来回切换；在这里只是换一个 dict。CLIP 的 `SetClipHooks`（`forced_hooks`）走同一条路。

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
* **部分加载 + hook（lora-lowvram-hook）。** 卸到 CPU 的层上，普通 LoRA 由原生的 `LowVramPatch` 在计算时加。这样的层同一个 key 上再有 hook 时，Monoload 的运行时 patch 排在 `LowVramPatch` 前面，以前它把普通 LoRA 也加了一遍（加了两次，差 0.025）。现在运行时 patch 看到同一个 key 后面有原生 `LowVramPatch`，就只加 hook；hook 撤掉之后什么都不加。顺序与原生相同：原生先把 hook 合并进存储的权重，计算时再由 `LowVramPatch` 加普通 LoRA。exact 下与原生逐位一致，fused 差 ≤ 1.4e-4（`tests/test_lora_lowvram_hook.py`）。
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
| 第一层：条带解码 | 认得的结构（LDM `Decoder`、Wan `Decoder3d` 单帧） | 低分辨率前缀（含 mid 全局注意力）整图算；只在每个分辨率阶段末尾存档；按输出条带倒推每层所需的输入行区间并重算；GroupNorm 的全局统计逐层空跑求得（条带内 fp32 Welford，跨条带 Chan 合并）；按峰值预算自动选条带高度和存档方案，满足不了就报错 | **第二阶段已实现 Wan `Decoder3d` 单帧（§9.13）**；第三阶段已实现 LDM（§9.14，待真机） |
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
| 2D latent 的音频 VAE（ACE-Step、LTX 2 音频、MiniMax H3 音频；`extra_1d_channel` 已设，或放大倍数大于 64，图像 VAE 是 1 / 4 / 8 / 16 / 32） | 原生，打日志（第四阶段 4b-0 起，§9.17） |
| decoder 在 batch 的各帧之间混合（SVD 的 `VideoDecoder`：batch 就是时间轴，逐样本解码会改变结果） | 原生，打日志（4b-0 起；以后做视频第二层时改成整批一次） |
| first-stage model 里没有卷积也没有 ComfyUI 的 VAE 注意力（像素空间「VAE」） | 原生，打日志（4b-0 起：第二层没有可做的） |
| 用户显式用 `VAEDecodeTiled` 节点或 `VAE.decode_tiled` | 不经过 `VAE.decode`，保持原生（用户自己选择了 tiled 的语义） |
| 直接调用 `first_stage_model.decode` 的第三方代码 | 不经过 `VAE.decode`，原生 |

第二层对结构没有要求：只要卷积是 `torch.nn.Conv2d/Conv3d`（含 comfy.ops），注意力是 ComfyUI 的三个 VAE 注意力函数之一，就会被分块；其他算子照原样运行（不会出错，只是峰值不一定降下来）。

### 9.9 开关与限制

| 设置 | 效果 |
|---|---|
| 默认 | 安装管理入口，预算 1 GiB |
| `MONOLOAD_VAE_WORKSPACE` | 工作区：`1G`、`512M`、`768`（纯数字按 MiB）等。第二层直接用它；第一层用 `min(它, 128 MiB)`（设了 `MONOLOAD_VAE_BUDGET` 时 `min(它, max(64 MiB, 预算/8))`），见 §9.13.11 |
| `MONOLOAD_DISABLE_VAE=1` | VAE 解码的全局默认改成原生（包装照装，§10），LoRA 部分照常；节点 `mode auto` 的 VAE 仍管理 |
| `MONOLOAD_DISABLE_VAE_STRIPE=1` | 只关第一层（§9.13），所有受管理的解码走第二层 |
| `MONOLOAD_VAE_BUDGET` | 第一层的峰值预算；不设时用默认策略（128 行条带的峰值内最高的条带，§9.13.4） |
| `MONOLOAD_VAE_STRIPE_ROWS` | 强制第一层的条带核心高度（扫参、调试） |
| `MONOLOAD_EXACT=1` | VAE 的全局默认是原生：分块改变 GEMM 形状，不保证逐位一致（节点 `mode auto` 的 VAE 仍管理） |
| `MONOLOAD=0` | VAE 的全局默认是原生（总开关，§10） |
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

### 9.11 后两阶段计划（第二阶段已完成，见 §9.13；第三阶段的实际做法见 §9.14）

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

实现：`monoload/vae_engine.py`（引擎：区间倒推、执行计划和估算、条带执行、arena、自检流程）和 `monoload/vae_wan.py`（Wan 适配器：结构识别、单元、内存模型、fp32 副本），接入在 `monoload/vae.py` 的 `_decode_layer1`。测试：`tests/test_vae_stripe.py`。（第二阶段时这些都在一个文件 `monoload/vae_stripe.py` 里，第三阶段 3a 拆开，行为和数字不变，见 §9.14.1。）

第二阶段结束时的状态、提交记录、规矩和第三阶段（LDM decoder）的入口分析见 docs/HANDOFF.md。

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
| 85a5c6f | arena（实测，与模拟一致） | 1.13 | 1.13 | 1.13 | 1.13（144 行） | 1.35 | 1.82 |

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

**估算为什么还要加 largest：** 在 384 个计划的网格（512² … 8K，bf16 / fp32，工作区 384 / 192 / 128 MiB，默认策略和 32 … 整图的强制高度）里，工作区 384 MiB 时默认计划从不溢出（128 MiB 时 8K 的默认计划溢出一次，见 §9.13.11）；但一部分高条带的计划会溢出，而且每次都恰好是**一个**请求找不到连续的洞，于是单独开一个 segment：要么是一块 columns（例如 2688、384 行、bf16：洞 383 MiB，请求 384 MiB），要么是一张全分辨率平面（8K、fp32、480 行：1389 MiB）。之后同尺寸的请求都复用这个 segment，不会再长。所以上界取 `arena + largest + 16 MiB`（16 MiB 给 ≤ 1 MiB 的小块池，实测 2–6 MiB）：384 个计划里 reserved 全部 ≤ 估算，最小余量 16 MiB（fp32 1344、2 条 384 行，溢出被 largest 盖住）。代价是估算比实际 reserved 高一个 largest（默认计划：工作区 384 MiB 时约 0.39 GiB，128 MiB 时 0.14–0.3 GiB），但这只影响交给 `load_models_gpu` 的数，不多占内存；实际占用就是 arena。

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
* 工作区：85a5c6f 时是 384 MiB（`min(MONOLOAD_VAE_WORKSPACE, 384 MiB)`），下面的表和扫参都是在它下面测的；workspace 实验之后默认改成 **128 MiB**（`LAYER1_WORKSPACE`，§9.13.11）：4K 默认 0.87 GiB / 8.44 s，2688 0.56 / 3.45 s，1344 0.36 / 0.71 s。
* 设了 `MONOLOAD_VAE_BUDGET` 时改用预算：取**估算**不超过预算的最高条带（预算是给 `load_models_gpu` 的上界，所以和估算比；工作区 = `min(MONOLOAD_VAE_WORKSPACE, max(64 MiB, 预算/8))`），8 行条带也放不下就抛 `MonoloadError`，写明需要多少。设了 `MONOLOAD_VAE_STRIPE_ROWS` 时强制那个高度（覆盖默认规则和预算，估算超过预算时日志注明）。第三阶段之后，设了预算时先比第二层：第二层放得下就用第二层，放不下才是这里的「预算内最高条带」（「预算内最快」，§9.14.10）。

85a5c6f（工作区 384 MiB）的默认计划（Qwen，输出在 GPU 上；实测 `bench_vae_l1c_*`，模拟器的预测在括号里；现在的默认见 §9.13.11）：

| 输出 | dtype | 默认计划 | 重算 | live | 实测 reserved（模拟） | 估算 | 热启动耗时（原生） | 725a010 实测 reserved / 估算 |
|---|---|---|---|---|---|---|---|---|
| 1344×768 | bf16 | 6 条 128 行 | 1.23× | 595 MiB | 0.70 GiB（0.66），GTT 0.71 | 1.05 GiB | 0.672 s（0.654） | 0.70 / 0.74 |
| 2688×1536 | bf16 | 12 条 128 行 | 1.25× | 793 MiB | 0.87 GiB（0.86） | 1.25 GiB | 3.216 s（3.396） | 1.02 / 0.95 |
| 3840×2160 | bf16 | 15 条 144 行 | 1.23× | 1053 MiB | 1.13 GiB（1.13） | 1.52 GiB | 8.241 s（7.889） | 1.44 / 1.25（14 条 155 行） |
| 7680×4320 | bf16 | 13 条 333 行 | 1.10× | 3058 MiB | （3.14 GiB，只有模拟） | 4.27 GiB | — | — |
| 1344×768 | fp32 | 6 条 128 行 | 1.23× | 734 MiB | 0.80 GiB（0.80） | 1.19 GiB | — | 0.96 / 0.89 |
| 2688×1536 | fp32 | 12 条 128 行 | 1.25× | 1084 MiB | 1.16 GiB（1.16） | 1.55 GiB | — | 1.41 / 1.27 |
| 3840×2160 | fp32 | 14 条 155 行 | 1.21× | 1627 MiB | 1.70 GiB（1.70） | 2.27 GiB | — | 2.07 / 1.90（13 条 167 行） |

所有行（含冷启动）的 reserved / GTT 都在估算以内。强制高度（bf16，4K，实测）：r32 / r64 / r128 / r144 都是 1.13 GiB，r256（9 条 240 行）1.35，r512（5 条 432 行）1.82（725a010 2.09，4e54d20 2.66）；耗时 10.60 / 9.12 / 8.35 / 8.23 / 7.99 / 7.81 s，与 725a010 持平。

**144 行以下峰值不再下降的原因：** live = 常驻 + max(前缀, 存档 + 条带)。4K 时前缀的存活峰值是 954 MiB（全局注意力：norm 输出 P + qkv 3P + 注意力输出 P + 分数块 384 MiB，P = 95 MiB），而 144 行条带的「存档 + 条带」是 938 MiB，已经低于前缀；128 / 64 / 32 行的条带部分是 898 / 741 / 662 MiB，都只是让条带部分更低，峰值仍是前缀的 954 MiB（加常驻 99 MiB = 1.03 GiB 存活，arena 1.12 GiB）。所以 4K 的瓶颈在前缀，要再降只能降前缀：工作区（分数块）或把注意力的 qkv 也分块。工作区改成 128 MiB 后前缀降到 698 MiB（6P + 128 MiB），4K 默认计划 14 条 155 行、峰值 0.87 GiB，仍由前缀决定（§9.13.11）。

（「重算」是整个 decoder 的卷积算量相对整图解码，前缀只算一次。）

#### 9.13.5 OOM

条带高度和工作区一起减半，重新做计划、重跑整个解码（前缀在 H/8 上，重跑代价小）。新计划的估算高于第一次的计划时跳过这一档、继续减半（日志写明；设了预算时第一次的计划本来就在预算内，强制高度超预算照跑的除外），交给 `load_models_gpu` 的量因此不用重算（review 06：45 个真实的重试序列里没有出现过估算变大，这是保险，`tests/test_vae_retry.py` 注入 OOM 和估算来测）；高度到 8 行（或强制的更小值）且工作区到 64 MiB 仍 OOM，就抛 `MonoloadVAEOOMError`。不退回 tiled，也不退回第二层：第二层的峰值更高，退回去没有意义。arena 在 `run` 的开头分配，它 OOM 时和解码中途 OOM 一样处理（重试时 arena 随计划变小）。测试里替换 `run_stripes` 模拟 OOM，确认 `decode_tiled_` 和第二层的 `_run` 都没有被调用。

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
* 缓存分配器（`tests/alloc_sim.py`，全尺寸 Wan decoder 放在 meta 设备上，§9.13.10）：模拟器复现 4e54d20 / 725a010 / 85a5c6f 和工作区 128 MiB 默认计划的 9 个真机读数（误差 ≤ 0.03 GiB）；本版 9 个计划（1344 默认和整图、1920×1088、2688、4K 默认 / 32 行 / 512 行、fp32 4K、8K）的模拟 reserved ≤ 估算。
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
* 85a5c6f 已复测（README §10.2），与模拟一致：默认 1344 / 2688 / 4K 的 GTT 0.71 / 0.87 / 1.13 GiB，fp32 0.80 / 1.16 / 1.70，所有行（含冷启动）都在估算以内；4K 扫参 r32–r144 1.13、r256 1.35、r512 1.82；速度与 725a010 持平，精度不变。**第二阶段验收通过。**
* workspace 实验已做（README §10.2，§9.13.11）：第一层默认工作区改成 128 MiB（4K 0.87 GiB），第二层保持 1 GiB。第二阶段到此结束，第三阶段见 docs/HANDOFF.md。

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
* **校验：** 33 个真机读数（4e54d20、725a010、85a5c6f 的 4K 扫参、1344 / 2688、fp32 三档、第二层三档，以及 workspace 128 MiB 的默认三档），alloc 和 reserved 的误差都 ≤ 0.02 GiB。重放以前的版本时工作区固定为 384 MiB（`layer1_ws`），4e54d20 / 725a010 也关掉上采样输入的连续化（`contiguous=False`）；峰值时刻的存活块与 profile 的「峰值构成」逐项一致（例如 725a010 4K 155 行：384 / 242 / 121 / 95 / 95 / 63 / 43 / 21 MiB）。

用法：`python tests/alloc_sim.py`（校验表），`python tests/alloc_sim.py --res 3840x2160 --rows 32,128,512 [--version v1|v2] [--peak] [--segments]`（某个计划的 alloc / reserved、峰值时刻的存活块、reserved 峰值时的 segment 布局）。在锁定镜像里用 `tests/docker_run.sh` 跑，不需要模型文件；4K 的一次模拟约 5 秒。

#### 9.13.11 工作区（workspace）对两层的作用，与待做的实验

**是什么：** 工作区是「一次最多给分块的临时缓冲多少字节」。它只管两类临时缓冲：

* 卷积的 im2col 展开缓冲（columns）：一层卷积整层展开超过工作区，就按输出行分块，每块的 columns ≤ 工作区（`_ConvChunker`）；
* 注意力的分数矩阵：按 query 分块，每块的分数 + softmax ≤ 工作区（`split_attention_chunked` 等）。

激活本身（每层的输入输出、上采样结果、存档、输出缓冲）不受它影响。

**两层怎么取值：**

* `MONOLOAD_VAE_WORKSPACE`（默认 1 GiB）是总设置。第二层（逐算子分块，SDXL / Flux，以及没走第一层的解码）直接用它。
* 第一层（条带）用 `layer1_workspace()`：默认是 `min(MONOLOAD_VAE_WORKSPACE, LAYER1_WORKSPACE)`，`LAYER1_WORKSPACE` 现在是 128 MiB（5d668b6 之前是 384），所以日志里写 `workspace 128 MiB`；设了 `MONOLOAD_VAE_BUDGET` 时是 `min(MONOLOAD_VAE_WORKSPACE, max(64 MiB, 预算/8))`。也就是说 `MONOLOAD_VAE_WORKSPACE` 只有设得比 128 MiB 小时才会改变第一层；设大了第一层仍是 128。

**调小会影响什么：**

* 第一层：前缀注意力的分数块（384 MiB 时 4K 前缀峰值 954 MiB 里的 384 MiB）和条带里卷积的 columns 都变小，所以 live、arena、估算、默认条带高度（默认规则按 arena 比）都会变；卷积块数变多（同一层卷积分成更多次 GEMM），注意力 query 块变多。数值只差 GEMM 形状带来的舍入。
* 第二层：columns 和分数块变小。Qwen 第二层的峰值主要是整图激活，模拟显示 reserved 几乎不变（4K 1 GiB / 384 / 128 MiB：9.59 / 9.69 / 9.63 GiB）；SDXL 第二层 4K 实测 14.86 GiB 里 columns 占多少要实测。
* 估算（`load_models_gpu`）和 OOM 重试的起点（重试时工作区减半，下限 64 MiB）。

**模拟器的预测（Qwen bf16，默认计划，reserved GiB；实验前算的）：**

| 工作区 | 1344 | 2688 | 4K | 第二层 1344 / 2688 / 4K |
|---|---|---|---|---|
| 1 GiB（第一层实际 384） | 0.66 | 0.87 | 1.13 | 2.10 / 5.05 / 9.59 |
| 384 MiB | 0.66 | 0.87 | 1.13 | 1.26 / 4.51 / 9.69 |
| 256 MiB | 0.51 | 0.71 | 1.00 | 1.25 / 5.10 / 9.78 |
| 128 MiB | 0.36 | 0.56 | 0.87 | 1.18 / 4.87 / 9.63 |

命令见 README 9.7 的 J（Qwen，第一层 384 / 256 / 128，第二层 1G / 384 / 128）和 K（SDXL 第二层 1G / 512 / 256 / 128）；bench 的模式名加 `-w<MiB>` 就是只对这个模式改工作区。

**实测（5d668b6，`bench_vae_ws_*`）：** 与模拟一致，所有行 reserved ≤ 估算。

| | 1344 | 2688 | 4K |
|---|---|---|---|
| 第一层 384 MiB：GTT / 热启动 | 0.66 GiB / 0.67 s | 0.87 / 3.21 | 1.13 / 8.24 |
| 第一层 128 MiB：GTT / 热启动 | 0.36 / 0.71 | 0.56 / 3.45 | 0.87 / 8.44（14 条 155 行） |

* 第二层，Qwen 4K：调小内存不降（reserved 9.59 → 9.63–9.69 GiB），反而变慢（6.94 → 7.83 / 9.07 s）。
* 第二层，SDXL 4K：调小后 alloc 降了，但 reserved 反而升高（14.99 → 15.32–15.86 GiB，碎片），也更慢（9.89 → 13.15 s）。中低分辨率调小有收益，但不是峰值瓶颈。

**决定（用户定）：**

* 第一层默认工作区 384 → **128 MiB**（`LAYER1_WORKSPACE`）。峰值优先、多算可以接受：峰值降 23–45%，耗时多 2–7%。不选 64 MiB，是为了给 OOM 重试留一档可以缩（重试下限仍是 `MIN_WORKSPACE` = 64 MiB）。预算路径 `max(64 MiB, 预算/8)` 不变。
* 第二层保持 1 GiB。SDXL / Flux 在第三阶段改走第一层。
* 新默认就是实测过的 `-w128` 配置。用 `alloc_sim` 复核：384 个计划的网格里工作区 128 MiB 的 124 个计划 reserved 全部 ≤ 估算（最小余量 140 MiB）；默认计划 1344 / 2688 / 4K 的模拟 reserved 0.36 / 0.56 / 0.87 GiB 与实测相同，估算 0.50 / 0.71 / 1.16 GiB。工作区小了以后，高条带（例如强制 512 行）和 8K 默认计划更常有一个请求被挤出 arena（8K：arena 2.88、reserved 3.36 GiB），都在估算以内。
### 9.14 第三阶段：LDM decoder（SDXL / SD1.5 / SD3 / Flux `ae`）走第一层

第三阶段的入口分析在 docs/HANDOFF.md 第 6 节；这一节记录实际的做法。

#### 9.14.1 3a：拆出引擎和适配器（不改行为）

`monoload/vae_stripe.py` 拆成两个文件：

* `monoload/vae_engine.py`：与 decoder 无关的部分。区间（`Unit`、`need_in`、`valid_out`、`stripe_needs`、`split_rows`）、`conv_extra`（第二层包装下一次卷积的额外内存）、`arena_bytes`、`Plan`（条带、倒推、顺序、存活量 → arena → 估算的公式；每个模块的峰值由适配器给）、`run_prefix` / `run_stripes`（行维 `hdim` 由 `Plan` 带着，不再是模块常量）、`arena_supported` / `reserve_arena`、`StripeAdapter`（`plan()`、`run()`、`output_bytes()`、`selftest_memory()` 的通用实现）、自检的流程（`self_test`：fp32 副本、固定种子的小 latent、强制 40 行条带、8 MiB 工作区、与整图比、按结构缓存、`fork_rng`、结束后清缓存）。
* `monoload/vae_wan.py`：Wan 2.1 单帧适配器 `WanStripe`：结构识别（`wan_structure`）、`build_units`、签名、按 forward 数的内存模型（`res_peak` / `up_peak` / `unit_peak` / `unit_largest` / `prefix_peak` / `prefix_largest` / `prefix_macs`）、`hdim = 3`、输出形状、`fp32_copy()`（返回绑定在 fp32 副本上的适配器）和 `reference_decode()`（`WanVAE.decode`）。

适配器接口写在 `vae_engine.py` 的模块注释里。`vae.py` 只经过接口：`_layer1_self_test` 调 `vae_engine.self_test`，自检的显存是 `bound.selftest_memory()`，输出缓冲是 `bound.output_bytes()`（原来的 `_out_bytes` / `_selftest_memory` 去掉了）。`tests/alloc_sim.py` 经过 `vae._select_layer1` 选适配器，不再直接调用某个适配器的 `match`。

执行顺序与原来逐行相同（前缀 → 释放 latent 副本 → 第一次分配输出缓冲 → 条带），所以分配顺序不变。验证：`tests/alloc_sim.py` 的 33 行校验表与拆分前逐字相同；`tests/test_vae_stripe.py` 73 项全过，每一行输出（含误差数字）与拆分前相同（只差计时）；`test_vae.py` 131 项、`test_entry.py` 7 种开关组合、`test_dtype_paths.py` 两种模式全过。

#### 9.14.2 结构核对（62b3c94）与识别

`comfy.sd.VAE` 对 SD1.5 / SDXL 建 `AutoencoderKL`（`AutoencodingEngineLegacy`，`post_quant_conv` 1×1 后接 decoder），对 Flux `ae` / SD3 建 `AutoencodingEngine`（直接 decoder），decoder 都是 `comfy.ldm.modules.diffusionmodules.model.Decoder`。逐行核对了 4D 输入、`conv3d=False` 时的执行路径：

* `Decoder.forward`：`conv_carry_causal_3d([z], conv_in)` 对非 `CarriedConv3d` 就是 `conv_in(z)`；`mid.block_1(h, temb=None)` → `mid.attn_1(h)` → `mid.block_2`；`carried=False` 时 `h = [h]`，循环只跑一次；每级 `block[i](h, None, None, None)`，`attn` 为空时不调用；`i_level != 0` 时 `upsample(h, None, None)`；最后 `norm_out` → `nonlinearity`（`F.silu`，不是原地）→ `conv_out`；`tanh_out=False` 时没有 tanh。
* `ResnetBlock.forward`（`temb=None`）：`norm1 → swish（SiLU(inplace=True)，作用在 norm1 的新输出上）→ conv1 → norm2 → swish → dropout（inplace，eval 时不做事）→ conv2`，`in != out` 时 `nin_shortcut(x)`（1×1），最后 `x + h`。`x` 不被原地修改。
* `Upsample.forward`：4D 时 `interpolate(x, scale_factor=(2.0, 2.0), mode="nearest")`，然后 3×3 conv。
* GroupNorm 是 `comfy.ops` 的 `GroupNorm`（`torch.nn.GroupNorm` 子类，32 组、eps 1e-6、affine）；有 cast / weight_function 时走 `forward_comfy_cast_weights`：`CastBiasWeightContext` 里 `F.group_norm(input, num_groups, weight, bias, eps)`。

拆点与 Wan 相同：**前缀**（整图，H/8）是 `[post_quant_conv,] conv_in, mid.block_1, mid.attn_1, mid.block_2, up[L-1].block[*]`，存档是它的输出（512 通道）；**条带部分**的单元依次是 `up[L-1].upsample`、各级的 ResnetBlock 和 Upsample、`norm_out`、`nonlinearity`、`conv_out`（SDXL：15 个单元）。halo：ResnetBlock 2、Upsample 1（输出分辨率上）、`conv_out` 1、norm / silu 0。

**识别**（`vae_ldm.ldm_structure`，任何一项不符就走第二层，日志写原因）：

* `first_stage_model` 的类型恰好是 `AutoencoderKL` / `AutoencodingEngineLegacy` / `AutoencodingEngine`；decoder 恰好是 `Decoder`；`post_quant_conv`（有的话）是 1×1、通道对得上；没有 `bn`（Flux 2 的 `batch_norm_latent`）。
* 非默认分支一律不认：`carried`（3D conv / 时间维）、`tanh_out`、`give_pre_end`（这个版本没有这个属性，别的版本有的话为 True 也不认）、up 级里有注意力（`attn_resolutions`）、Upsample 的 scale 不是 2.0 或没有 conv、`conv_shortcut`（3×3 shortcut）。VideoDecoder 一类类型不同，直接不认。
* 每个 ResnetBlock：norm1 / norm2 是 affine 的 `GroupNorm`、通道整除组数；conv1 / conv2 3×3、stride 1、padding 1、zeros；通道对得上；swish 是 SiLU；Dropout 处于 eval 或 p=0；通道变化时 `nin_shortcut` 1×1。mid 的 AttnBlock 类型恰好是 `AttnBlock`、q/k/v/proj 1×1。`norm_out`、`conv_out` 同样检查。
* 模型上没有 forward hook、实例级 forward；没有 vae_options；latent 4D、通道数对得上。

#### 9.14.3 GroupNorm 统计量跨条带调度：一个参数化的机制

条带部分有 19 个 GroupNorm（9 个 ResnetBlock 各两个 + `norm_out`），每个都要它的输入在**整张图**上每组的均值和方差，而它的输入又依赖前面所有 GroupNorm 的统计量。做法：

* **统计遍。** 按调用顺序，对每个 GroupNorm 跑一遍：从最近的存档出发，按条带算出这个 GroupNorm 的输入（前面的 GroupNorm 都已有统计量），只在每条带的「核心行」上累加统计量，算完冻结。norm1 / `norm_out` 的输入就是单元的输入；norm2 在 ResnetBlock 内部，统计遍的最后一步是一个「部分单元」`norm1 → swish → conv1`（halo 1，同样调用原模块实例）。
* **最后一遍**从最后一个存档出发，按输出条带解码，所有 GroupNorm 都用冻结的整图统计量。
* **方案 = 存档位置。** A / D / B / C 是同一个机制，只是「哪些单元的输入整张存下来」不同（`vae_ldm.scheme_positions`）：A 不存（每遍都从 H/8 存档出发），D 存 H/4 级的输出，B 存 H/4、H/2 级的输出，C 再加全分辨率每个残差块的输出。每遍从不超过目标位置的最近存档出发。
* **存档不需要额外的遍。** 某个位置的存档在「第一个经过它、且它之前的 GroupNorm 都已冻结」的统计遍里顺便写好（每条带把它在该位置的核心行拷进去），之后的遍从它出发，更早的存档随即释放。所以 HANDOFF 表里的「20 + 1 / 20 + 2 / 20 + 4 遍」都是 19 个统计遍 + 1 个最后一遍。
* **核心行**：统计遍的条带按目标位置的行均分；在更低分辨率的位置上，核心行是 `[⌊t0·h_p/h_t⌋, ⌊t1·h_p/h_t⌋)`。因为倒推的需求区间包含这个范围（逐级 floor 可以合并），存档拷贝和统计都只用精确行，每行恰好被一条带计入一次。
* **统计遍的条带高度**单独选：先算每遍用 1 行条带时也躲不开的峰值（它持有的存档 + 最小的条带），与前缀、最后一遍的峰值取最大，作为上限；每遍在这个上限内取最高的条带（二分）。所以统计遍不抬高峰值，低分辨率的遍用很高的条带、少算重复的 halo。遍的代价只依赖几何和它自己的条带，按结构缓存，规划一次 4K 解码 < 0.6 s（第一次），之后是查表。
* **存档放在哪：两种布局，取 arena 小的那个。** 存档是大块、长寿命的分配，处理不好会把 arena 切碎（越来越大的存档 H/4 → H/2 → H 放不进前面释放的洞：C 不处理时要 1.37 倍存活量的 arena）。
  * 分开：每个存档单独分配，没人再从它出发就释放；arena 另加「新存档分配时已经释放、但比它小的存档」的总大小作为余量（这些洞它用不上）。
  * 存档池：所有存档放进一次分配的池，按槽轮换（一个存档只和它的前一个同时存活，两个槽就够）；没有洞，但池从第一个存档起整块占到最后。
  * 每个计划两种都算（各自的统计遍条带高度也随之重选），用 arena 小的。SDXL 4K 默认计划：D 只有一个存档，两种一样；B 选「分开」（池会让最后一遍多占 H/4 的槽，2.57 → 2.17 GiB）；C 选「池」（分开要 1.37 倍，池 4.70 GiB，代价是早期统计遍的条带变矮，重算 4.0 → 5.7 倍）。

#### 9.14.4 统计量的数值，GroupNorm 的替换

* **统计量：fp32，shifted data + 块内 var_mean + Chan 合并。** 每组先取第一块的均值 K 作为平移量；每块（≤ 工作区的行数）拷成 fp32、原地减 K，用 `torch.var_mean`（PyTorch 的 Welford / 两遍归约）得到块的均值和 M2；块与块、条带与条带之间用 Chan 等人的成对合并（`d = m_b − m_a，mean += d·n_b/n，M2 += M2_b + d²·n_a·n_b/n`）。理由：`E[x²] − E[x]²` 在均值远大于标准差时灾难性抵消（测试里均值 1000、标准差 0.01 时相对误差 1.9e3）；Chan 合并本身不相减两个大和；平移让合并的量都很小，否则累计均值的 fp32 舍入（1000 附近 6e-5）会通过 d² 项放大（不平移时同一测试 1.2e-3，平移后 ≤ 1e-5 的检查通过）。方差是有偏的 `M2/n`，与 GroupNorm 一致。
* **应用：与 ATen 的 GroupNorm 内核相同的算法。** 有了每组的 mean / rstd（`rsqrt(var + eps)`），每通道 `a = rstd·weight，b = bias − mean·a`（fp32），`y = x·a + b` 用 fp32 的 `addcmul` 算、写回 x 的 dtype（ATen CUDA 的 `ComputeFusedParams` 就是这样，bf16 输入时内部用 float）。按行分块，每块的 fp32 临时量 ≤ 工作区。
* **有意偏离「模块原样调用」。** GroupNorm 在条带上原样调用会用条带自己的统计量（这正是 tiled 不等价的原因），所以在受管理的第一层解码期间（`vae_engine.GlobalNorms`，只在 `stripe_pass` 里），条带部分的 19 个 GroupNorm 实例各设一个实例属性 `forward`：先走模块自己的权重路径（`comfy.ops` 的 `run_every_op`、`CastBiasWeightContext`，即 cast / weight_function / bias_function，包括 Monoload 的运行时 LoRA；普通 `torch.nn.GroupNorm` 直接用参数），再按冻结的统计量做上面的计算。没有统计量就调用是内部错误。退出（含异常）时删掉实例属性。前缀里的 GroupNorm（整图）不替换，原生计算。其余模块（卷积、上采样、SiLU、残差块本身）仍原样调用在切片上。
* **兜底：fp32 自检。** 每种结构 + 方案在本进程第一次使用时，用 fp32 副本、24×24 latent、40 行条带（5 条）、8 MiB 工作区跑完整的 19 个统计遍 + 最后一遍，与原模型类自己的 `decode`（`AutoencoderKL.decode` / `AutoencodingEngine.decode`，在副本上）整图比，相对误差 ≤ 1e-4（实测 1e-6 量级）。测试里故意注入的三种错误都被抓到：统计量只用条带自己的（tiled 式）、丢掉一条带的统计、halo 少一行。

#### 9.14.5 内存模型与 arena

* **单元的存活量**按 62b3c94 的 forward 数（`vae_ldm.py` 注释）：ResnetBlock `S + max(A + G, A + B + 卷积额外, 2B + G, 2B + 卷积额外, 2B，有 nin_shortcut 时再加 2B + 1×1 权重、3B)`，G 是冻结统计量 GroupNorm 的 fp32 块（前缀里原生 GroupNorm 为 0）；Upsample 与 Wan 的 Resample 相同；norm_out `S + A + G`；silu `S + A`；conv_out 与 Wan 的 head 卷积相同；注意力与 Wan 相同。统计遍的末尾另加目标张量 + fp32 统计块。
* **两个修正（模拟器发现的）：** (1) `nin_shortcut` 是 1×1 卷积，第二层从不分块它，Slow2d 会把非连续的行切片整份拷一份——共用的 `conv_extra` 把它当成按行分块，低估了（fp32 1344、256 行时存活量低估 10.5%）。(2) 这份拷贝发生在残差块的最后，arena 已被块内的临时量切碎，它常常放不下（4K A 需要 1.15 倍存活量的 arena）。改成：有 `nin_shortcut` 的残差块在调用前把行切片拷成连续的（`Unit.contiguous`，与 Upsample 单元一样），Slow2d 不再拷；4K A 需要的 arena 降到 1.07 倍。修正后模型与模拟的张量峰值差 −0.1% … +0.7%。这两处只在 LDM 适配器里，Wan 的模型不变。
* **arena**：没有存档的计划（A，以及 Wan）照旧 `存活量 + 存活量/32 + 64 MiB`；有存档的计划（D / B / C）用 `/16`（存档把 arena 切成几段，模拟里最多要存活量 + 9.5%），「分开」布局再加上面的余量。估算仍是 `arena + largest + 16 MiB`，largest 里包括最大的存档（或整个池）。

#### 9.14.6 模拟结果（`tests/alloc_sim.py`）

**模拟器先对上第二层的真机读数。** SDXL / Flux 的 meta 构造（`--model sdxl|flux`）复现了 18 个读数，误差都 ≤ 0.02 GiB：README §10.1 的 SDXL / Flux 三档（af9abc6，分块卷积在第一块之后才分配输出），以及 workspace 实验 K 的 SDXL 十二个 GTT 读数（1G / 512M / 256M / 128M × 三档），包括 4K「工作区越小 reserved 反而越高」（14.97 / 15.84 / 15.40 / 15.30，实测 14.99 / 15.86 / 15.42 / 15.32）。§10.1 和实验 K 的 4K 相差 0.13 GiB，正是 85a5c6f 改了分块卷积输出的分配顺序：模拟器用两种顺序分别得到 14.84 和 14.97。

**第一层各方案（SDXL bf16，默认条带策略，GiB；reserved 就是 arena）：**

| 方案 | 1344×768：reserved / 估算 / 重算 | 2688×1536 | 3840×2160 |
|---|---|---|---|
| A（默认） | 0.47 / 0.61 / 11.9× | 0.79 / 0.99 / 12.1× | 1.10 / 1.48 / 12.1× |
| D | 0.54 / 0.68 / 8.0× | 1.01 / 1.27 / 8.1× | 1.52 / 2.03 / 8.1× |
| B | 0.62 / 0.76 / 5.1×（分开） | 1.33 / 1.84 / 5.1×（分开） | 2.17 / 3.18 / 5.3×（分开） |
| C | 0.79 / 1.30 / 5.9×（池） | 2.44 / 4.42 / 5.2×（池） | 4.70 / 8.67 / 5.7×（池） |

Flux 与 SDXL 只差 latent 通道，数字相同到 0.01。重算是整个 decoder 的卷积算量相对整图解码（前缀算一次）；HANDOFF 粗估的「条带部分」倍数与之量级一致（A 13×、D 9–10×、B 6×、C 4.5×）。

**4K 的方案 × 条带高度：** A 在 32 / 64 / 96 行时都是 1.06 GiB（13.1× / 12.5× / 12.3×），D 是 1.09 / 1.22 / 1.36 GiB（10.3× / 9.0× / 8.5×）。4K 的峰值下限约 1.05 GiB，由整图前缀（全局注意力）和输出缓冲决定；D 用 32 行条带也能到下限，算量比 A 少。默认规则（128 行条带的峰值为目标）是第二阶段按 Wan 的速度定的，对 LDM 是否合适要看真机（命令 N / O）。

**估算是 reserved 的上界：** 202 个计划（SDXL 1024² / 1344 / 1920×1088 / 2688 / 4K × bf16 / fp32 × 四种方案 × 默认 / 32 / 128 / 256 行，加工作区 64 MiB、Flux、8K），reserved 全部 ≤ 估算；只有 2 个计划有一个请求落在 arena 外（fp32 4K 强制 256 行，余量 13 MiB；8K 方案 D），都被估算里的 largest 盖住，与 Wan 8K 默认计划的情形相同。

**存活量模型与模拟的张量峰值：** 修正后（§9.14.5）差 −0.1% … +0.7%。

**规划耗时：** 每遍的代价只与几何和它自己的条带有关，按结构缓存；条带的代价按「每个单元跑多少行」的模式去重。4K 一次 `choose_plan` 第一次 < 0.6 s（修正前 10 s），之后是查表。

#### 9.14.7 默认值与开关

* **方案：默认 B**（`vae_ldm.DEFAULT_SCHEME`；7059bdd 之前是 A，CT 700 实测后改成 B，§9.14.10）。A 在每个分辨率上峰值都最低（§9.14.6），但 4K 要 75 s，B 2.17 GiB / 42 s。设了 `MONOLOAD_VAE_BUDGET` 而没有强制方案时，方案由预算策略选（§9.14.10）。`MONOLOAD_VAE_GN_SCHEME=A|B|C|D` 选别的方案（大小写都行，不认识的值打警告、用默认）；bench 的模式名加 `-g<S>`（`monoload-gD`、`monoload-r32-gA`），`--gn-schemes ADBC` 一条命令扫遍。
* **条带高度：沿用第二阶段的默认规则**（以 128 行条带的 arena 为目标，取不超过它的最高条带），强制高度 `MONOLOAD_VAE_STRIPE_ROWS` 照旧且优先；设了预算 `MONOLOAD_VAE_BUDGET` 时是「预算内最快」（§9.14.10）；统计遍的条带高度由引擎在这个峰值内自动取（§9.14.3）。
* **工作区**：第一层照旧 128 MiB（`LAYER1_WORKSPACE`）；GroupNorm 的 fp32 块也受它限制。
* **自检**按「结构 + 方案」缓存：换方案会重新自检一次。
* **OOM**：与 Wan 相同，条带高度和工作区一起减半重试（统计遍的条带随之变矮），到 8 行 / 64 MiB 抛 `MonoloadVAEOOMError`；不退回 tiled，不退回第二层。
* **日志**：启动时写明 LDM 也走第一层和当前方案（`B (default)` 或 `D (forced, MONOLOAD_VAE_GN_SCHEME)`）；每次解码 `-> layer 1 (LDM stripes, GroupNorm scheme B): 17 stripes of 128 rows (core), recompute 5.28x, checkpoint 127 MiB; 19 statistics passes, saves 506 MiB+1012 MiB; ...`；不认的结构 `VAE layer 1 (stripes) not used for AutoencoderKL: ...; decoder.up[1] has attention -> layer 2`（两个适配器的原因都列出）。

#### 9.14.8 测试（`tests/test_vae_ldm.py`，CPU，不需要模型文件）

见 README §8 的结果。要点：统计量（Moments 对 fp64，含均值远大于标准差的情形；冻结统计量的 GroupNorm 对 `F.group_norm`，fp32 / bf16、整图 / 行切片；`GlobalNorms` 走 weight_function 路径、退出复原、缺统计量报错）；识别（11 种不认的结构都走第二层且与原生一致）；整个 decoder 与原生 `VAE.decode`（SDXL 式 / Flux 式，四种方案，条带 1 / 7 / 40 / 默认，奇数、很小的 latent、batch 2、ch 64、16 KiB 工作区）；bf16 对 fp32 真值与原生 bf16 同一水平；自检抓住三种注入的错误；SDXL 4K 全尺寸的计划（19 遍、各方案的存档数、峰值与重算的顺序）；OOM；开关；分配器模拟（第二层复现真机、第一层 reserved ≤ 估算）。§9.14.10 之后加了预算策略：第二层放得下选第二层；否则预测最快的方案（排序与候选里的预测一致）；一条带不跑统计遍、取默认方案；选中的方案自检失败选下一个；强制高度 / 方案 / 第二层优先于预算；强制方案放不下报错（不退回第二层）；都放不下报错并列出各需要多少；全尺寸 SDXL 三档 × 20 / 3 / 1.5 GiB 选中预测表里的配置，4K 3 GiB 的计划模拟 reserved ≤ 估算。`tests/test_vae_stripe.py`（Wan）照旧全过，Wan 的模拟校验表与拆分前相同。

#### 9.14.9 限制与待真机确认

* **速度。** 方案 A 的卷积算量约是整图解码的 12 倍（D 约 8 倍、B 约 5 倍、C 约 4–6 倍）。耗时不与算量成正比；真机结果和按分辨率级拟合的耗时模型见 §9.14.10（4K：A 75 s、D 58 s、B 42 s、C 35.5 s，第二层 9.9 s）。
* **arena 和估算是用模拟器验证的**（§9.14.6 的网格），不是证明；模拟器对 LDM 第二层复现了 18 个真机读数，第一层的 LDM 读数在 7059bdd 上实测与模拟一致（≤ 0.01 GiB，§9.14.10）。
* **统计量与原生不逐位一致**：累加顺序不同（fp32），和原生 GroupNorm 内核的差在 1e-6 量级（fp32 测试），bf16 下由 bench 的 `--fp32-ref` 判断。
* 只认 2D 图像的 LDM decoder；3D（carried / VideoConv3d）、up 级带注意力、`tanh_out` 等非默认分支走第二层。Flux 2 的 `batch_norm_latent` 也走第二层（可以以后加，它只是 decoder 之前的逐点变换）。

#### 9.14.10 CT 700 实测（7059bdd）与决定：默认方案 B，设了预算时「预算内最快」

**实测（SDXL，bf16，GTT 增量 GiB / 热启动 s；Flux `ae` 与 SDXL 相同）：** 内存与模拟器一致（≤ 0.01 GiB），每一行热启动都 ≤ 估算。

| | 1344×768 | 2688×1536 | 3840×2160 |
|---|---|---|---|
| 原生 | 8.36 / 0.86 | 42.8 / 4.65 | 52.5 / 11.9 |
| 第二层 | 3.72 / 0.88 | 9.21 / 4.01 | 15.0 / 9.9 |
| 第一层 A（当时的默认） | 0.47 / 8.9 | 0.79 / 37.0 | 1.10 / 75.0 |

| 4K | A | D | B | C | A r32 / r64 / r96 | D r32 / r64 / r96 |
|---|---|---|---|---|---|---|
| GTT / s | 1.10 / 75.0 | 1.52 / 57.8 | 2.17 / 42.1 | 4.70 / 35.5 | 1.06 / 79.3 · 1.06 / 76.5 · 1.06 / 75.7 | 1.09 / 69.1 · 1.22 / 61.5 · 1.36 / 58.9 |

精度：4K 对 fp32 的 RMSE SDXL 0.00105（原生 0.00108）、Flux 0.00107（0.00107），各方案、各高度 PSNR 都约 61.2 dB，通道均值偏移与原生同一水平，条带边界附近不比其他行差。Qwen 回归：GTT 0.36 / 0.56 / 0.87、热启动 0.72 / 3.43 / 8.46 s，不变。

**决定（用户定）：** 1 GiB 和 2 GiB 的差别不重要，重要的是能自定义。

1. LDM 的默认 GroupNorm 方案改成 **B**（`vae_ldm.DEFAULT_SCHEME`）：4K 2.17 GiB / 42 s，对 A 1.10 GiB / 75 s。默认条带规则不变（128 行条带的 arena 为目标）。
2. 设了 `MONOLOAD_VAE_BUDGET` 时改成**「预算内最快」**（下面）。
3. `MONOLOAD_VAE_GN_SCHEME` / `MONOLOAD_VAE_STRIPE_ROWS` / `MONOLOAD_DISABLE_VAE_STRIPE` 仍可强制，且优先于预算。

**耗时模型（第一版；第二版加了工作区和前缀注意力，见 §9.14.11）。** 耗时与卷积算量不成正比：C 的算量 5.7× 却比 B（5.3×）快。原因是算量落在哪一级分辨率：全分辨率（128 通道、宽 3840）每 MAC 的耗时约是 H/4、H/2 的两倍（GEMM 窄、GroupNorm / SiLU 是访存密集的），C 的存档让重算挪到了低分辨率。所以按级拟合：

```
耗时 = Σ_级 c_级 × 该级的卷积 MAC（每 10¹² MAC 的秒数）
c = { 前缀（H/8，整图一次）: 1.35, H/4: 0.113, H/2: 0.129, H: 0.269 }
```

每个计划的各级 MAC 由引擎在规划时算出（`Plan.work_levels`：前缀、最后一遍的条带、每个统计遍，按输出宽度归级）。用 CT 700 上 12 个读数拟合（4K 的 A / D / B / C，A 的 1344 / 2688 / 4K，A 和 D 的 r32 / r64 / r96），最大误差 2.3%（例如 4K：A 75.0 → 74.4、D 57.8 → 56.8、B 42.1 → 42.4、C 35.5 → 35.7）。试过再加「每次调用」的固定开销项，拟合出负系数，不用。前缀的系数大，是因为它包含中间级的注意力（按 query 分块）。

这些秒数只对 CT 700 的 bf16 成立；预算策略只用它给**同一次解码**的几个第一层配置排序，排序只依赖各级之间的比例，换机器时比例大致不变（全分辨率级通道少、访存密集）。Wan 只有一种配置，不需要模型（`predict_seconds` 返回 None）。

**预算内最快（`vae.choose_budget`）：**

* **候选**：第二层（工作区 `MONOLOAD_VAE_WORKSPACE`，估算同 §9.4）；第一层的每个变体（LDM：A / B / C / D 各一个；Wan：一个），各取**估算不超过预算的最高条带**（Wan 第一层仍是「预算内最高的条带」）。
* **排序**：第二层能放下就选第二层——它每个卷积只算一次，实测总是最快（SDXL 4K 9.9 s，第一层最快的 C 35.5 s；Qwen 4K 6.9 s，第一层 8.4 s；1344 上第二层也不慢于单条带的第一层）。放不下时在放得下的第一层变体里选耗时模型预测最快的；预测相同（例如只有一条带）时取默认方案。没有用「各候选都用模型预测」的统一排序，因为第二层的耗时形态（整图大 GEMM、分块注意力）不在这个模型里，而第二层比第一层快的结论有真机数据直接支持。
* **第一层的工作区**：依次试 `min(MONOLOAD_VAE_WORKSPACE, max(64 MiB, 预算/8))`（第二阶段的预算工作区）、128 MiB、64 MiB；每个变体取预测最快的那档（同样快时取大的，Wan 取第一档放得下的）。原因：工作区越大估算越高，预算/8 会把本来放得下的方案挤出去（SDXL 4K、预算 3 GiB：方案 B 在 384 MiB 工作区下最少要 3.26 GiB，64 MiB 下 3.01）。耗时模型不含工作区的影响（Qwen 上 384 → 128 MiB 慢 2–7%），这是已知的近似。
* **强制设置优先**：`MONOLOAD_DISABLE_VAE_STRIPE=1` → 第二层（超出预算也跑，日志注明）；`MONOLOAD_VAE_STRIPE_ROWS` → 第一层、这个高度，方案仍按预算内最快选，一个都放不下时取估算最小的照跑（日志注明）；`MONOLOAD_VAE_GN_SCHEME` → LDM 第一层用这个方案，高度仍取预算内最高。强制了第一层的设置时不考虑第二层。强制的第一层配置因为自检未通过不能用时（这次解码里刚失败，或者这个进程里早先失败、已缓存，两种情况决定相同；强制意图看实际生效的设置，不看过滤后的候选），改走第二层，但第二层也要放得下预算，放不下就报错，写明强制的设置、自检失败的变体和第二层需要多少（review 02，`tests/test_vae_selftest_budget.py`；以前首次失败时第二层不受预算限制、缓存后却又受限）。
* **自检**：只对选中的变体做（按「结构 + 方案」缓存）；不通过就排除它，选下一个。
* **都放不下**：抛 `MonoloadError`，列出第二层和每个方案最少需要多少（最少的高度不一定是 8 行，见下），以及第一层不可用时的原因。不认识的 decoder（只有第二层可选）在预算放不下第二层时也报错——预算是显式设置，严格执行。
* **日志**：`[Monoload] VAE MONOLOAD_VAE_BUDGET 3.00 GiB -> layer 1 scheme D 309 rows (workspace 64 MiB) 2.94 GiB, ~53.6 s: the fastest predicted that fits; others: layer 2 17.91 GiB (over); layer 1 scheme B 32 rows (workspace 64 MiB) 3.01 GiB, ~58.1 s (over); ...`，之后照常是这一次解码的那一行。`last_decode()["candidates"]` 记录所有候选。

**顺带改的两处引擎行为：**

* **只有一条带时不跑统计遍。** 整张图一次通过每个 GroupNorm，原生的 GroupNorm 本身就用整图统计量，统计遍和存档都是多余的。以前一条带的计划也跑 19 遍（重算 3–4 倍）；现在算量 1 倍、各方案相同。这只在条带高度 ≥ 图高时发生（预算大但放不下第二层、或强制高度），默认规则在 SDXL / Flux 的三档上都是多条带，计划不变。
* **找「预算内最高条带」时，最小高度放不下不再直接放弃**：从 8 行开始翻倍找第一个放得下的高度，再往上二分。有统计遍时估算在最矮的高度不单调：统计遍被挤到 1–3 行、存档改用池布局，`largest` 变成整个池（SDXL 4K 方案 B：8 / 16 行 3.57 GiB，32–96 行 3.09 GiB）。只有最小高度放不下时才多试几档，原来能找到的计划不变。

**预测（SDXL bf16；估算 / 模拟 reserved GiB，耗时为模型预测、第二层为实测）：**

| 预算 | 1344×768 | 2688×1536 | 3840×2160 |
|---|---|---|---|
| 20G | 第二层 3.98 / 3.71，0.88 s | 第二层 9.92 / 9.21，4.0 s | 第二层 17.91 / 14.97，9.9 s |
| 3G | 第一层 1 条（不跑统计遍）2.48 / 1.97，约 1.3 s | B 4 条 384 行 2.73 / 2.20，约 20 s | D 7 条 309 行 2.94 / 2.33，约 54 s |
| 2G | C 2 条 384 行 1.23 / 0.97，约 3.2 s | B 8 条 192 行 1.98 / 1.48，约 20 s | D 16 条 135 行 1.99 / 1.48，约 57 s |
| 1.5G | C 2 条 384 行 1.16 / 0.90，约 3.3 s | D 8 条 192 行 1.45 / 1.15，约 27 s | A 15 条 144 行 1.48 / 1.10，约 73 s |

4K 的 1.5G 选 A 而不是 D：D 最少要 1.53 GiB（8 行），放不下。Qwen（Wan）：20G → 第二层（2.10 / 5.05 / 9.59 GiB）；3G → 第一层 1 条 768 行 / 2 条 768 行 / 4 条 540 行（1.31 / 1.96 / 2.09）；1.5G → 768 / 384 / 240 行（1.06 / 1.08 / 1.12）。所有选中的计划模拟 reserved ≤ 估算 ≤ 预算。

**限制：** 耗时模型只用 SDXL 4K（和 A 的三档）拟合，没有 B / C 在 1344 / 2688 的读数；Flux 的 decoder 与 SDXL 相同（只有 `conv_in` 的输入通道不同），沿用同一组系数。工作区对耗时的影响不在模型里。第二层「总是最快」是 CT 700 上的实测结论，换到第一层比第二层快的机器上会选慢的那个（不会超预算）。

#### 9.14.11 01377c4 的实测，收紧有存档时的估算，耗时模型加入工作区

**实测（01377c4，命令 Q / R / S）：** 选中的配置都与预测一致，GTT ≤ 估算 ≤ 预算，精度 60–62 dB。默认 B：SDXL 0.62 / 4.6 s、1.33 / 19.5 s、2.17 / 42.0 s；Flux 0.62 / 4.7、1.33 / 19.6、2.18 / 41.9。Qwen 按预算逐档与预测一致。耗时模型在 1344 / 2688 上误差 ±10% 以内。

**问题：** SDXL 4K、预算 3G 选中了 D（7 条 309 行，工作区 64 MiB），实测 2.33 GiB / 63.2 s（模型 53.6 s）；而默认的 B（17 条 128 行）只要 2.17 GiB / 42.0 s。B 被排除，是因为它的估算 3.17 GiB 超过了预算：`arena + largest + 16 MiB` 里的 largest 正好是 1 GiB 的 H/2 存档。D 的 309 行 / 64 MiB 又慢在工作区：卷积分块翻倍，耗时模型里没有这一项。

**收紧估算：存档的位置有保证时，不算进 largest。** largest 是给「碎片把一个请求挤出 arena」留的余量。存档是大块、长寿命的分配，但它们分配的时刻和位置是确定的，可以在规划时证明它一定放得进 arena：

* **检查点前置。** 有存档的计划，检查点不再是前缀输出的那个张量（它落在 arena 的哪里取决于前缀的分配历史），而是前缀之前就分配好的缓冲区（arena 刚分出来，只有它一块，所以在最前面），前缀的输出拷进去。代价：前缀期间多占一个检查点（4K 127 MiB，模型里前缀的存活量加上它；B / C / D 的峰值不在前缀，arena 不变）。
* **之后的长寿命分配是确定的序列：** 检查点 → 输出缓冲（前缀之后分配）→ 每个存档（或整个池）在建它的那一遍开始时分配、在不再有遍从它出发后释放（检查点也是）。统计遍在下一遍开始前释放了它的所有临时量，所以这些时刻 arena 里只有这几块。
* **`vae_engine.saves_fit`** 按缓存分配器的规则（best fit：够大的空闲块里最小的、同样大取地址低的、从块头切；释放后与相邻空闲块合并）在 arena 里重放这个序列。每个存档都找得到空闲块，就说明它不会被挤出 arena（它若被放进 arena 外已有的空闲块，arena 只会更空）。这时估算的 largest 只在其余分配里取；放不下时照旧把最大的存档算进去。
* SDXL 4K 默认计划：B 3.17 → **2.68 GiB**（largest 0.99 → 0.49，现在是统计遍里一条整高的 H/4 条带，不是存档）、C 8.67 → 5.21、D 2.03 不变（它的 largest 本来就是同样大小的统计遍条带）。arena（实际 reserved）不变。
* 检查点在 arena 最前面，与之后的存档之间就是输出缓冲和检查点死后留下的洞；`saves_fit` 不成立的计划（实际没有遇到）照旧把最大的存档算进 largest。

**验证时发现的老问题：很高的条带。** 把网格扩大到「方案 B / C / D × 条带 128 … 1080 行 × 工作区 64 / 128 / 384 MiB」之后，4K 方案 B 的 540 / 768 行条带 reserved 超过了估算（7.20 对 5.21、9.65 对 6.34 GiB），收紧之前也一样（去掉检查点前置，结果相同）：以前的网格没有这么高的 B。原因是一次「挤出」变成了好几次：最后一遍时 arena 里是 [检查点的洞 127 MiB][输出][H/4 存档的洞 506 MiB][H/2 存档 1012 MiB][全分辨率的临时量 1.3 GiB × 2][601 MiB 空闲]，一个 688 MiB 的请求哪个洞都放不下，另开 segment；之后不同尺寸的请求又各开一个，缓存住的 segment 越积越多。用模拟器的 `arena_need`（同样放置所需的最小 arena）量了 144 个这类计划，arena 规则补了两条（只对有存档的计划）：

* **死掉的存档留下的洞**（`Plan.front_arena`）：用同一个长寿命序列算出每一遍开始时「最高的存活块的位置」和它下面的洞；某一遍最大的临时量比每个洞都大、而洞的总量又超过规则给的余量时，这一遍的临时量只能放在存活块上面：arena ≥ 存活块的顶 + 临时量 × 9/8 + 64 MiB。
* **余量**（`arena_bytes`）：条带不超过 3 条时用存活量 / 8（与整图一条带相同，几块极大的平面），工作区大于 128 MiB 时余量至少是一个工作区（384 MiB 的 columns 块像一块同样大的平面一样切碎 arena）。

默认计划（128 行左右的条带、128 MiB 工作区）的 arena 不变（`arena_need` 都在原来的 arena 以内）；变大的是很高的条带、只有 2–3 条带的计划（例如 1344 方案 C 两条 384 行：0.90 → 1.05 GiB）和 384 MiB 工作区的计划。

**验证（`tests/alloc_sim.py`）：** 403 个计划（原来的 202 个、预算会选到的工作区 64 / 192 MiB 的 B / C / D、上面 144 个高条带的计划、8K 的 B / C）reserved 全部 ≤ 估算；只有 2 个计划有一个请求落在 arena 外（fp32 4K A 256 行，余量 13 MiB，与以前相同；4K B 1080 行 / 64 MiB，余量 1.0 GiB），都被 largest 盖住。Wan 的 33 个和 SDXL / Flux 第二层的 18 个真机读数照旧复现。

**耗时模型加入工作区。** 第一版只按分辨率级的 MAC 计时，看不到工作区：64 MiB 让卷积的行块数翻倍（每块一次 im2col、一次 GEMM、输入输出各拷一次），4K D 309 行 / 64 MiB 的调用数 38828，128 MiB 时约一半。第二版：

```
耗时 = 0.0922 × MAC(H/4) + 0.1294 × MAC(H/2) + 0.2464 × MAC(H)      （秒 / 10¹² MAC，条带和统计遍）
     + 0.237 ms × 卷积 GEMM 调用数                                    （vae_ldm.unit_calls，按工作区算行块）
     + 0.239 × 2 × 通道 × token² / 10¹²                               （前缀中间块的整图注意力，token = latent 像素数）
```

用 22 个 CT 700 读数拟合（7059bdd 的 12 个 + 01377c4 的默认 B 两档和 R 的 8 个，工作区 64 / 128 / 192 / 384 MiB），最大误差 5.3%，只有整图一条带的 1344（1.00 s，模型 0.83）差 −17%。4K D 309 行 / 64 MiB 从 −15%（53.6 对 63.2 s）变成 −5%（60.0 s）。注意力项让 1344 不再系统性偏高（第一版 +10%）：它与 token 数的平方成正比，第一版把它摊进了「前缀的 MAC」。前缀的卷积在拟合里系数为 0（与注意力项共线，且对同一张图的所有计划相同），不再单列。

**按预算选的结果（SDXL bf16；估算 / 模拟 reserved GiB，耗时为模型预测）：**

| 预算 | 1344×768 | 2688×1536 | 3840×2160 |
|---|---|---|---|
| 3G | 整图 1 条（384 MiB）2.48 / 1.97，0.8 s | C 4 × 384（128 MiB）2.99 / 2.73，12.0 s | **B 12 × 180（128 MiB）2.96 / 2.43，41.1 s** |
| 2G | C 2 × 384（256 MiB）1.44 / 1.18，2.8 s | B 11 × 140（256 MiB）1.97 / 1.70，18.9 s | D 18 × 120（128 MiB）1.87 / 1.49，58.8 s |
| 1.5G | C 2 × 384（192 MiB）1.25 / 1.05，2.8 s | B 16 × 96（128 MiB）1.48 / 1.22，20.6 s | A 17 × 128（128 MiB）1.48 / 1.10，74.1 s |
| 1G | C 2 × 384（128 MiB）0.99 / 0.86，4.2 s | D 23 × 67（128 MiB）1.00 / 0.80，30.7 s | 放不下（A 最少 1.38） |

4K 3G 现在选 B（以前 D 7 × 309 / 64 MiB，实测 63.2 s）。4K 1.5G 仍是 A，但工作区从 64 回到 128 MiB（模型现在知道 64 MiB 更慢）。验证命令是 README §9.7 的 T。

### 9.15 Monoload VAE Settings 节点：单独设置某个 VAE

插件的第一个节点（README §4.1）。环境变量对所有 VAE 生效；这个节点让工作流里的某一个 VAE 用自己的峰值预算、GroupNorm 方案、条带高度和模式。

**节点的注册结构（`monoload/nodes/`）：** 每个节点是一个模块里的一个类，用 ComfyUI 经典的节点接口（`INPUT_TYPES`、`RETURN_TYPES`、`FUNCTION`、`CATEGORY`）外加 `DISPLAY_NAME`；`monoload/nodes/__init__.py` 的 `NODES` 列出它们，生成 `NODE_CLASS_MAPPINGS`（键是类名）和 `NODE_DISPLAY_NAME_MAPPINGS`，插件入口 `__init__.py` 直接导出这两个表。以后加节点：写一个模块，把类加进 `NODES`。节点在任何开关下都注册（包括 `MONOLOAD_DISABLE=1`），保存过的工作流总能加载；开关让节点的功能失效时，由节点自己说明。

**副本，而不是修改输入：** 节点返回 `copy.copy(vae)`，在副本上设一个属性（`vae_overrides.ATTR`，一个只含设了的项的 dict）。浅拷贝共享 `first_stage_model` 和 `patcher`：权重不多占内存，`load_models_gpu`、卸载、ComfyUI 的缓存都看到同一个 `ModelPatcher`（测试确认模型管理里只有一个已加载模型）；`VAE` 对象自己在解码 / 编码时只写 `size` 这个缓存。输入的 VAE 不变，工作流里其他直接用它的分支不受影响；encode 不经过 Monoload，结果相同。节点串联时，下游节点没设的项沿用上游副本的值。`vae_overrides.py` 不导入 torch / ComfyUI，节点在 `MONOLOAD_DISABLE=1` 时也不导入解码的代码。

**每次解码怎么取设置（`vae.resolve_settings`，逐项）：** 副本自己的值 → 全局设置（环境变量 `MONOLOAD_VAE_BUDGET` / `MONOLOAD_VAE_GN_SCHEME` / `MONOLOAD_VAE_STRIPE_ROWS` / `MONOLOAD_DISABLE_VAE_STRIPE`，测试和 bench 用 `set_*` 改的也算在这一级）→ 默认值。节点上的「不设」是 `default` / `0`（预算 0 = 不设，条带高度 0 = 自动）。语义与环境变量完全相同：节点的方案、高度就是「强制」，节点的预算走同一个 `choose_budget`（放不下就报错、写明各需要多少），模式 `layer 2 only` 等于 `MONOLOAD_DISABLE_VAE_STRIPE=1`，`auto` 是第一层 + 第二层（压过全局的 `MONOLOAD_DISABLE_VAE_STRIPE=1`），`native` 是这个 VAE 走 ComfyUI 自己的解码。

**怎么生效：** `_decode`（`VAE.decode` 的包装）先取这次的设置，再在 `_Applied` 里把它们换进全局的 `_SETTINGS` 和 `vae_ldm` 的方案，解码完（包括报错）换回来。解码路径的其余代码不用改，任何调用 `vae.decode` 的节点都生效；`decode_tiled` 不经过包装，按现有规则保持原生。ComfyUI 一次执行一个 prompt，解码不会并发，这样换是安全的（测试确认三个副本交替解码各用各的设置，报错后全局设置复原）。

**全局开关（9b30154 时）最高，§10 之后不再是：** 起初 `MONOLOAD_DISABLE`、`MONOLOAD_DISABLE_VAE`、`MONOLOAD_EXACT` 时包装不装，节点不能把功能打开。总开关分支（§10）之后包装总是装上，这些变量只是全局默认值，节点上明确选的模式压过它们；只有 `MONOLOAD_DISABLE=1` 什么都不装，节点原样透传输入。

**日志：** 每次受管理的解码，那一行末尾是 `settings: budget 3.00 GiB (node), GroupNorm scheme chosen by the budget (default), stripe rows auto (default), mode auto (default)`（来源 `node` / `env` / `default`）；按预算选的那一行也带上；`mode native` 时是 `VAE decode left native: mode native (Monoload VAE Settings node); settings: ...`。`last_decode()` 里有 `settings` 和 `settings_source`。

**测试：** `tests/test_vae_node.py`（README §8）；ComfyUI 加载器的注册在 `tests/test_entry.py` 的 8 种开关组合里检查。真机验证：`tests/check_vae_node.py`（README §9.7 的 U）。

### 9.16 第四阶段 4a：全部 VAE 的盘点（vae-inventory）

目的：把 VAE 解码管理推广到全部 VAE 之前，先弄清楚锁定镜像（ComfyUI 0.31.0）的 `comfy/sd.py` 能构造哪些 VAE、各自的结构、现在 Monoload 怎么处理、原生和第二层的峰值，再由用户定先做哪几种。这一步不改插件行为。

**方法（三个脚本，都不需要模型文件）：**

* `tests/vae_inventory.py`：每种 VAE 按 sd.py 的配置在 meta 设备上建出全尺寸的 first-stage model，把它的 state dict（meta 张量）交给 `comfy.sd.VAE` 本身（bf16，`is_amd()` 为真），所以分支、`latent_dim`、比例、`memory_used_decode` 都是 ComfyUI 自己的。然后：结构统计（GroupNorm / RMS / Pixel / Layer / Batch norm、带 `optimized_attention` 的注意力、Conv2d / Conv3d / ConvTranspose、Linear）、Monoload 现在的路（`_native_reason`、各适配器的 `match`）、`tests/alloc_sim.py` 的追踪（CT 700 的后端：4D 卷积 Slow2d im2col、5D 卷积 SlowDilated3d vol2col、缓存分配器）：原生、第二层（1 GiB 工作区）、有适配器的第一层。追踪新加了**设备上限**：一次申请需要新段而超过 60 GiB（62.5 GiB GTT 减去权重和机器其余部分）时，先释放全部缓存的空闲段再试（CUDA / HIP 分配器 OOM 时就是这样做的），还超过就记一次真 OOM（原生这时会退回 tiled）。加上这条之后，SDXL 4K 原生的模拟值是 52.48 GiB，与 CT 700 实测（§9.12）完全一致；其余已测的读数（SDXL / Flux 1344 原生 8.36、第二层 3.71–3.72，Qwen 4K 原生 59.14、第二层 9.59，各第一层）也都在 0.02 GiB 以内。
* `tests/probe_vae_gaps.py`：CPU 上随机权重的真实小解码，确认下面的「现有缺口」，以及第二层在多帧视频 decoder 上是否精确。
* `tests/check_models.py`：给 CT 700 用，只读 safetensors 头（加上 64 KiB 以下的小张量），在 meta 上识别 `models/` 下每个文件：VAE 是哪一种、checkpoint 内置的 VAE、diffusion model 是什么模型、要哪种 latent（因此要哪个 VAE）、`models/vae` 里哪个文件对得上（README §9.11 的命令 W）。

**用户实际在用的（CT 700 上的文件）：** `models/vae/ae.safetensors`（Flux `ae`，Z-Image / Lumina 2 也用它）、`models/vae/qwen_image_vae.safetensors`（Wan 2.1 结构；Krea 2、Anima、Qwen-Image 都用它，ComfyUI 的 `Krea2` / `Anima` 配置的 latent 格式都是 `Wan21`）、`waiIllustriousSDXL_v170` 内置的 SDXL VAE（`wai_v17_fp8_test` 是 SDXL UNet，也用它）。这三种**图像解码现在都已经走第一层**。`novaAnimeAM_v5029B`、`luciddreamerZ_*` 从文件名看不出模型类型，命令 W 会给出答案（按命名推测分别是 Anima → `qwen_image_vae`、Z-Image → `ae`）。

**表一：结构与现在的路**（✔ = 有；「整图统计」= 需要整张图统计量的归一化）

| VAE（用在哪些模型） | first-stage model / decoder，latent | 整图统计的归一化 | 时间维因果缓存 | 注意力 | 特殊输出 | 现在走 | 为什么 |
|---|---|---|---|---|---|---|---|
| SD1.x / SD2.x / SDXL（Pony、Illustrious…）**用户在用** | `AutoencoderKL` / LDM `Decoder`，4D `[B,4,H/8,W/8]` | ✔ GroupNorm ×52（整图） | — | H/8 全局（mid） | — | 第一层 LDM | 已支持 |
| Flux.1 / Z-Image / Lumina 2 / Chroma / HiDream / SD3（`ae`）**用户在用** | `AutoencodingEngine` / LDM `Decoder`，4D z16 | ✔ 同上 | — | 同上 | — | 第一层 LDM | 已支持 |
| Flux 2 / Ideogram 4 / Lens / Ernie-Image | `AutoencoderKL`（`batch_norm_latent`）/ LDM `Decoder`，4D `[B,128,H/16,W/16]` → 反归一化 + 2×2 还原成 z32 `[B,32,H/8,W/8]` | ✔ 同上 | — | 同上 | — | 第二层 | `ldm_structure` 拒绝 `bn` |
| SD x4 upscaler | `AutoencoderKL` / LDM `Decoder`（ch_mult [1,2,4]，4x） | ✔ | — | H/4 全局 | — | 第一层 LDM | 已支持（3 级） |
| SVD img2vid | `AutoencodingEngine` / `VideoDecoder`，4D，**batch 当时间轴** | ✔ GroupNorm ×80 | 时间混合（Conv3d 核 [3,1,1]、时间注意力）跨整个 batch | H/8 | — | 第二层 | **现有缺口 1** |
| HunyuanImage 2.1 | `AutoencodingEngine` / `hunyuan_video.vae.Decoder`，4D z64，32x | ✔ GroupNorm ×72 | — | H/32 全局 | 上采样是先卷积再 2×2 depth-to-space（`PixelUnshuffle2D`）+ 重复通道的残差 | 第二层 | 不认识的结构 |
| HunyuanImage 2.1 Refiner | `AutoencodingEngine` / `vae_refiner.Decoder`（`refiner_vae=False`），5D T=1，16x | ✔ GroupNorm 在 C×T×H×W 上（一张图内部解成 4 帧，取最后一帧） | 非因果 Conv3d（时间补零） | 3D 全局 | — | 第二层 | 不认识 |
| HunyuanVideo 1.5 | `AutoencodingEngine` / `vae_refiner.Decoder`（RMS），5D z32，16x，时间 4x | — RMS（逐位置） | ✔ CarriedConv3d：每 2 个 latent 帧一段，带 2 帧 carry | 3D 全局（T×H/16×W/16 个 token） | 首帧特例（时间上采样） | T=1 第二层；多帧原生 | 多帧未放开 |
| HunyuanVideo 1.0 / Kandinsky 5 视频 | `AutoencoderKL` / LDM `Decoder`（conv3d，`CarriedConv3d`），5D z16，8x | ✔ GroupNorm：mid 在全视频上，up 级每个时间段各自统计 | ✔ 每 2 帧一段带 carry | 3D 全局（T×H/8×W/8） | — | T=1 第二层；多帧原生 | `post_quant_conv` 是 Conv3d；多帧未放开 |
| Wan 2.1 / Qwen-Image / Krea 2 / Anima / Cosmos Predict 2 / JoyImage（`qwen_image_vae`）**用户在用** | `WanVAE` / `Decoder3d`，5D z16，8x，时间 4x | — RMS | ✔ `feat_cache`（CACHE_T=2）；首帧单独，之后每段 2 个 latent 帧 | 每帧 2D（mid） | — | T=1 第一层 Wan；多帧原生 | 多帧未放开 |
| Wan 2.2 5B | `vae2_2.WanVAE` / `Decoder3d`（dec_dim 256），5D z48，16x（patchify 2） | — RMS | ✔ `feat_cache`，每个 latent 帧一段；输出逐帧 `torch.cat` | 每帧 2D | up 级的 `DupUp3D` 捷径、`unpatchify` | T=1 第二层；多帧原生 | 不认识（类名同为 WanVAE，但不是 2.1 的类） |
| Mochi | `VideoVAE` / genmo `Decoder`，5D z12，8x，时间 6x | ✔ GroupNorm **逐帧**（`GroupNormSpatial`） | 因果 Conv3d（`PConv3d`），**整段视频一次算** | 只有时间维 1D 注意力（ComfyUI 通用 `optimized_attention`，第二层不分块） | `DepthToSpaceTime` | 原生 | 多帧未放开 |
| LTX-Video 0.9.0 / 0.9.5+ / LTX 2 | `VideoVAE` / lightricks `Decoder`，5D z128，32x，时间 8x | — PixelNorm | ✔ 因果 Conv3d，按时间块解码 | — | `comfy_has_chunked_io`：写进预分配输出 | 原生 | chunked io |
| CogVideoX | `AutoencoderKLCogVideoX` / `Decoder3D`，5D z16，8x | ✔ GroupNorm（`SpatialNorm3D`，在每个时间块上统计） | ✔ `conv_cache`；低分辨率级整段算，高分辨率级按时间块滚动 | — | 解完的块先放到 CPU | 原生 | 多帧未放开 |
| Cosmos 1.0（CV8x8x8） | `CausalContinuousVideoTokenizer` / `DecoderFactorized`，5D z16 | ✔ GroupNorm（num_groups=1，整段） | 因果 Conv3d（复制补边），整段一次算 | 空间注意力（已知函数）+ 时间注意力 | 小波 `unpatcher3d` | 原生 | 多帧未放开 |
| SeedVR2 | `VideoAutoencoderKLWrapper` / `Decoder3D`，5D z16 | ✔ GroupNorm 逐帧 | ✔ 因果，自带 `memory_limit` 切片 | diffusers 式（第二层不分块） | `handles_tiling` | T=1 第二层 | 不认识 |
| MiniMax H3 视频 | `MiniMaxH3VideoVAE` / `ViT3DDecoder`，5D z24，16x | GroupNorm / RMS（transformer） | 内部按 17 帧 / 256 px 分块 | 内部 | chunked io + 自己分块 | 原生 | chunked io |
| Mage-VAE | `MageVAE`（一步扩散 codec），4D z128，16x | 少量 | — | 有 | — | 第二层 | 不认识 |
| TAESD / TAEF1 / TAEF2 | `TAESD`，4D | TAEF2 低分辨率级有 4 组 GroupNorm | — | — | — | 第二层 | 不认识 |
| TAEHV / TAEW2.2 / lighttae | `TAEHV`，5D | — | 帧间 memblock | — | 输出逐帧搬到 intermediate device | 多帧原生 | 多帧未放开 |
| Stable Cascade Stage A / Stage C previewer | `StageA`（ConvTranspose、depthwise、LayerNorm 逐像素）/ `Previewer`（BatchNorm） | — | — | — | — | 第二层 | 不认识 |
| 像素空间（Chroma Radiance、Z-Image pixel、PixelDiT、HiDream O1） | `PixelspaceConversionVAE`（恒等） | — | — | — | — | 第二层 | **现有缺口 3** |
| ACE-Step 音频 / LTX 2 音频 / MiniMax H3 音频 | `MusicDCAE` / `AudioVAE` / `MiniMaxH3AudioVAE`，**2D latent**（`latent_dim` 2） | — | — | — | 输出是波形 | 第二层 | **现有缺口 2** |
| Stable Audio 1 / 3、MMAudio、Hunyuan3D、TripoSplat | 1D latent | — | — | — | — | 原生 | `latent_dim` 1 |

**表二：峰值（alloc_sim，CT 700，bf16，GiB，GTT / reserved 增量；图像为「1344×768 / 3840×2160」）与第一层的预期**

| VAE | 原生 | 第二层 | 第一层 | ComfyUI 估算（AMD） | 难度 / 风险 |
|---|---|---|---|---|---|
| SDXL / Flux `ae` | 8.36 / 52.48（实测一致） | 3.71 / 14.97 | 0.62 / 2.17（已实现） | 11.4 / 91.9 | — |
| Flux 2 | 8.36 / 52.50 | 3.72 / 15.05 | **预计 0.62 / 2.18**：decoder 与 Flux `ae` 相同，只多了 latent 级的反归一化和 2×2 还原（放进前缀，H/16 级，几 MiB） | 11.4 / 91.9 | **低**：`vae_ldm` 接受 `bn`，前缀加一步；自检、方案、耗时模型照用 |
| SD x4 upscaler（输出 1344×768） | 16.25 | 3.30 | 已走第一层（未单独模拟） | 45.7 | — |
| HunyuanImage 2.1（输出 3840×2176） | 4.66 / 38.11 | 2.34 / 11.10 | 粗估 4K 2–3 GiB（全分辨率 128 通道、GroupNorm，与 SDXL 相近；未模拟） | 1.35 / 10.9 | 中：新单元「先卷积再 depth-to-space」，区间倒推要新写；GroupNorm 统计照用 |
| HunyuanImage 2.1 Refiner（1344×768） | **50.29** | 4.43 | 粗估 1–2 GiB（未模拟） | 5.38（低估约 10 倍，原生实际会 OOM → tiled） | 高：内部 4 帧、GroupNorm 跨帧、非因果时间补零 |
| Wan 2.1 / `qwen_image_vae` 单帧 | 6.09 / 59.14（实测一致） | 2.10 / 9.59 | 0.36 / 0.87（已实现） | 4.2 / 34.0 | — |
| Wan 2.2 单帧 | 5.21 / 34.23 | 4.01 / **19.78**（第二层不够：全分辨率级 256 通道，激活本身大） | 粗估 4K 0.5–1.5 GiB（未模拟） | 15.4 / **123.6**（`load_models_gpu` 会卸载一切） | 中：照 `vae_wan` 写，多 `DupUp3D` 捷径、patchify |
| HunyuanVideo 1.5 单帧 / HunyuanVideo 1.0 单帧 | 12.94 / 23.50（1344） | 3.31 / 5.38 | 未估 | 27.7 / 5.4 | 中 |
| SeedVR2 单帧（1920×1080） | 26.26 | 9.12 | 未估 | 0.31 | 高（自带切片机制） |
| Mage-VAE / Stage A | 0.62 / 1.44 | 相同（没有可分块的大卷积） | 不需要 | — | 不做 |
| TAESD（1344 / 4K） | 1.77 / 14.23 | **2.66** / 4.88（1344 时第二层反而高，**现有缺口 4**） | 不需要 | 11.4 / 91.9 | — |
| Stage C previewer（1024²） | 3.27 | 2.02 | 不需要 | 11.6 | — |
| SVD（1024×576，14 帧） | 25.71 | 2.75（**结果错**，缺口 1） | — | 6.5 | — |
| 视频（多帧） | | | | | |
| Wan 2.1，832×480×81 | 8.72 | 4.96 | 粗估 1–2 GiB（按时间段条带；未模拟） | 5.2 | 第二层：低；第一层：高 |
| Wan 2.2，1280×704×121 | 39.38 | 10.39 | 同上 | 13.4 | 第二层：低 |
| HunyuanVideo 1.0，848×480×73 | 62.2（**超上限 → 原生退回 tiled**） | 15.21 | 未估 | 17.0 | 第二层：低 |
| HunyuanVideo 1.5，1280×720×121 | 113（**OOM → tiled**） | 21.64 | 未估 | 24.7 | 第二层：低 |
| CogVideoX，720×480×49 | 35.02 | 11.66 | 未估 | **88.3** | 第二层：低（解完的块放 CPU，见注） |
| Cosmos 1.0，1280×704×121 | 30.18 | 12.99 | 未估 | 10.7 | 第二层：低 |
| Mochi，848×480×85 | 362（OOM → tiled） | **59.3（仍在上限，第二层不够）** | 要第一层（整段视频一次算，单个激活 8.5 GiB） | 68.2 | 高 |
| LTX 0.9.0 / LTX 2，768×512×97 | 30.6 / 59.3（**模拟里碎片严重**：alloc 只有 5.5 / 6.1，待真机确认） | 6.70 / 4.97 | 未估 | 5.7 | 中：要支持 `output_buffer`（chunked io） |
| TAEHV / TAEW2.2 / MiniMax H3 视频 | 1.31 / 3.04 / 0.99 | 相同 | 不需要 | — | 不做 |

注：模拟值都是「权重已加载、解码前清空缓存」之后的增量，输出缓冲（fp32）按 `--gpu-only` 算在设备上；CogVideoX 放到 CPU 的块不计。第二层的数字假设逐样本解码、1 GiB 工作区，估算（`estimate` / `_probe`）还没有为多帧做，这是 4b 的工作。

**第二层在多帧视频上可以直接用**（`probe_vae_gaps.py` 第 4 项）：小尺寸随机权重的 Wan 2.1（5 个 latent 帧）、Wan 2.2、HunyuanVideo 1.0、HunyuanVideo 1.5、CogVideoX（全尺寸），各自的 `decode`（带时间因果缓存）在 16 KiB 工作区下几乎所有卷积按行分块，与不分块相比相对误差 ≤ 2.3e-6。原因：分块只在 H 方向，时间维的缓存拼接发生在卷积调用之前，每次调用拿到的已经是拼好的输入。

**现有缺口（4a 时的代码；1–3 已在 4b-0 修掉，§9.17；Flux 2 第一层见 §9.18）：**

1. **SVD 的结果被改变**：`VideoDecoder` 的时间混合以整个 batch 为时间轴（`timesteps` 默认等于 batch 大小），第二层逐样本解码，等于每帧单独解码。4 帧的小解码：与原生 max|Δ| = 1、平均 0.13（像素值 [0,1]），与「原生逐帧单独解码」逐位相同。（原生自己也按空闲内存切 batch，`batch_number = free / memory_used_decode`，切了同样会变；但通常一次装得下。）
2. **2D latent 的音频 VAE 被管理**：ACE-Step（`[B,8,16,T]`）、LTX 2 音频、MiniMax H3 音频的 `latent_dim` 是 2，`_native_reason` 只看 `latent_dim`，于是走第二层。输出与原生相同，但 ACE 的形状探测（8×8 latent）失败，退回静态上界时用的放大倍数是 4096，交给 `load_models_gpu` 的估算约 **1 PiB**——每次解码都会把其他模型全部卸载。设计上（§9.8）音频应该原生。
3. **像素空间「VAE」被管理**：恒等变换，没有卷积，却报 2 GiB 估算（原生 24 KiB），可能白白腾出 2 GiB。
4. **小 decoder 第二层反而更高**：TAESD 1344×768 原生 1.77、第二层 2.66 GiB（第二层的 1 GiB 工作区块和预分配输出比原生最大的 columns 1.1 GiB 还占地方）。4K 时第二层仍然好得多（14.2 → 4.9）。影响小（TAESD 一般用于预览，不经过 `VAE.decode`）。

另外看到 ComfyUI 自己的估算对很多 VAE 偏差很大（Wan 2.2 4K 单帧 123.6 GiB、CogVideoX 88 GiB → 卸载一切；Refiner 5.4 GiB 而实际 50 GiB → 真 OOM 后退回 tiled），被 Monoload 管理之后用的是 Monoload 的估算，这本身也是推广的收益之一。

**建议的实施顺序**（等用户定）：

0. **修缺口 1–3**（一个分支，小，纯正确性）：SVD 这类跨 batch 耦合的 decoder 暂时走原生（或者第二层整批一次解码，峰值仍远低于原生，但估算要按 batch 算）；音频（`extra_1d_channel` 已设、或放大倍数是音频量级）走原生；没有卷积和注意力的 first-stage model 走原生。缺口 4 可选（例如原生最大的 columns 不超过工作区时直接原生）。
1. **Flux 2 第一层**：几乎就是改 `ldm_structure` 和前缀；4K 15.05 → 约 2.2 GiB。Flux 2 系列（Flux 2、Ideogram 4、Lens、Ernie-Image 都用这个 VAE）是现在的主流新模型。
2. **多帧视频第二层（通用）**：放开 `_native_reason` 的多帧限制（按 VAE 类型逐个放开，先 Wan 2.1 / 2.2、HunyuanVideo 1.0 / 1.5、CogVideoX、Cosmos），`_probe` / `estimate` 认识 5D 多帧（激活和 T 不成正比：Wan / HunyuanVideo / CogVideoX 都按时间段解码，探测要用能覆盖一段的帧数，输出缓冲按全长算）；LTX 需要走 `output_buffer`。HunyuanVideo 1.0 / 1.5 原生在 CT 700 上会 OOM 退回 tiled，这里变成整段精确解码。用户的 `qwen_image_vae` 就是 Wan 2.1 VAE，将来做 Wan 2.1 视频就用得上。
3. **Wan 2.2 单帧第一层**（照 `vae_wan` 写）：4K 19.8 → 约 1 GiB（粗估）。
4. **HunyuanImage 2.1 第一层**：4K 11.1 → 约 2–3 GiB（粗估）。
5. **视频第一层**（按时间段的条带，GroupNorm 按时间段统计）：工作量大，等真有需要再说；Mochi 只有这条路能降下来。

不建议做：TAE 系列、Stage A / C、Mage、MiniMax H3 视频（自己分块）、SeedVR2 的第一层（自带切片）、音频和 3D。

**待真机确认：** LTX 原生的碎片（模拟 reserved 30–59 GiB 而 alloc 只有 5–6 GiB）；命令 W 的输出（用户两个看不出类型的模型用哪个 VAE）。

### 9.17 第四阶段 4b-0：修盘点发现的缺口（vae-coverage-fixes）

用户定的顺序：① 修缺口 1–3 → ② Flux 2 第一层 → ③ 视频以后再定。SVD 先走原生，以后做视频第二层时再做 SVD 整批第二层。

**改动（`vae._native_reason`）：** 判断按真实结构和 VAE 对象的属性，结果按模型缓存（`_TRAITS`，弱引用）：

* **2D latent 的音频 VAE → 原生**（`_audio_ratio`）：ComfyUI 给 ACE-Step、LTX 2 音频设了 `extra_1d_channel`；MiniMax H3 音频没设，但它的 `upscale_ratio` 是 800（latent 帧 → 采样点），图像 VAE 只有 1 / 4 / 8 / 16 / 32，所以阈值取 64（`AUDIO_MIN_RATIO`）。
* **在 batch 的各帧之间混合的 decoder → 原生**（`_model_traits`）：模型里有 `comfy.ldm.modules.temporal_ae` 的 `VideoResBlock` / `AE3DConv` / `AttnVideoBlock`（SVD 的 `VideoDecoder` 用它们，`timesteps` 默认等于 batch 大小）。
* **没有可分块的算子 → 原生**：模型里既没有 `torch.nn.Conv2d` / `Conv3d`，也没有 ComfyUI 三个 VAE 注意力函数之一（像素空间 VAE 是恒等变换，以前白报 2 GiB 估算）。
* `_native_reason` 的所有理由都改走消息表（`vae.nr_*`，英文 / 中文）；以前是写死的英文。英文措辞不变（如 `multi-frame video latent (T=...)`）。

缺口 4（TAESD 小图第二层反而高）不在用户定的范围里，没动。

**验证：** `tests/test_vae.py` 第 5 部分：全尺寸随机权重的 SVD `VideoDecoder`，3 帧一个 batch：走原生、与原生逐位相同，而逐帧单独解码与原生差 0.82（像素）；ACE 式（`extra_1d_channel` + 4096）、MiniMax 式（800）走原生，比例 4 / 8 / 16 / 32 照常管理；真实的像素空间 VAE 走原生、结果相同、`load_models_gpu` 收到 ComfyUI 自己的估算；SDXL / Wan 单帧照常管理。`tests/probe_vae_gaps.py` 用真实的 ACE（MusicDCAE）和 MiniMax 音频重跑：都走原生，结果与原生相同，不再有 1 PiB 的估算。`tests/vae_inventory.py --no-trace`：图像 VAE 的路一个没变。

### 9.18 第四阶段 4b-1：Flux 2 VAE 走第一层（vae-flux2-layer1）

**结构核对（0.31.0）：** `comfy.sd.VAE` 看到 `bn.running_mean` 就给 ddconfig 加 `batch_norm_latent`，建 `AutoencoderKL`（z 32，embed 32，有 `post_quant_conv`），latent 128 通道、比例 16。`AutoencodingEngineLegacy.decode` 先 `z = z * sqrt(running_var + bn_eps) + running_mean`（BatchNorm 反归一化，`bn_eps` 1e-4，buffer 每次 `cast_to` 成 latent 的 dtype），再 `rearrange("... (c pi pj) i j -> ... c (i pi) (j pj)", pi=2, pj=2)`（128 × H/16 → 32 × H/8），然后是和 Flux `ae` 相同的 `post_quant_conv` + LDM `Decoder`（ch 128，ch_mult [1,2,4,4]）。真实文件 `flux2-vae.safetensors`（Comfy-Org/flux2-dev，sha256 d64f3a68…）用命令 W 核对过：`AutoencoderKL / Decoder | latent 128 ch, x16`。

**做法（`vae_ldm.py`）：**

* `ldm_structure` 接受这个 BatchNorm（`_check_bn_latent`）：`torch.nn.BatchNorm2d`、非 affine、有 running 统计量、`ps == [2, 2]`、有 `bn_eps`、特征数 = 4 × `post_quant_conv` 的输入通道；其他样子的 BatchNorm 仍走第二层并写明原因。
* latent 这一步是前缀的第一个模块 `LatentUnpatch`：用模型自己的 buffer、同样的运算和顺序，所以与原生给 `post_quant_conv` 的输入逐位相同（测试核对）。这一步在原生里也不是模块调用，是 `decode` 里的几行运算，所以这里照抄运算；fp32 自检把整条路（含这一步）与模型类自己的 `decode` 对比兜底。
* 引擎加一个钩子 `StripeAdapter.decoder_hw(samples)`：计划在 decoder 的分辨率（H/8）上做（`plan` / `smallest_plan` / 存档缓冲的形状），LDM 适配器在有 BatchNorm latent 时返回 latent 尺寸的 2 倍；其余适配器不变。`LatentUnpatch` 的内存模型：输出和一个临时量各一份 latent 大小（`prefix_peak`），没有卷积量。
* 结构签名多了 `bn` 一项，第一次使用单独做 fp32 自检；fp32 副本带上 BatchNorm 的 buffer、`bn_eps`、`ps`，参照解码就是 `AutoencoderKL.decode` 本身。自检的 latent 是 decoder 24 × 24 对应的 12 × 12（几何与其他 LDM 相同）。
* 名字 `LDM stripes (batch-norm latent), GroupNorm scheme B`；方案、条带、预算、耗时模型（decoder 相同，`TIME_COEF` 照用）、VAE 设置节点、Info 节点都不用改。启动日志、节点 tooltip（两份 nodeDefs.json）、README 的列表加上 Flux 2。

**模拟（`tests/alloc_sim.py --model flux2`，bf16，GiB）：** 默认 B：1344×768 reserved 0.62 / 估算 0.76，2688×1536 1.33 / 1.61，4K 2.18 / 2.69；4K 的 A 1.11 / 1.49、D 1.53 / 2.04、C 4.71 / 5.22；预算 3G 选 4K B 12 × 180（2.44 / 2.97），1.5G 选 A。都与 Flux `ae` 相差 ≤ 0.01 GiB，计划（条带数、高度、方案、按预算的选择、预测耗时）完全相同。第二层 3.72 / 9.25 / 15.05，原生 8.36 / 42.8（实测 Flux）/ 52.50。

**测试：** `tests/test_vae_flux2.py`（46 项）：识别（含 3 种不认的 BatchNorm → 第二层 == 原生）；`LatentUnpatch` 与原生逐位相同；四种方案 × 条带高度 7 / 40 / 默认、奇数和 1×1 latent、batch 2、16 KiB 工作区、bf16 对 fp32 真值（RMSE 0.00983，原生 0.00996），fp32 与原生差 ≤ 4.5e-6；自检通过并与 SDXL 式分开缓存，注入错误的 latent 步骤 / 条带内统计 → 自检失败、第二层、== 原生；模拟 OOM → 缩条带，到底 → `MonoloadVAEOOMError`，不走 tiled 也不走第二层；预算：等于第二层估算选第二层、稍低选第一层、1 KiB 报错列出各自需要；VAE 设置节点（副本强制方案 C、16 行；`layer 2 only`）和 Info 节点直接生效；alloc_sim 上 reserved ≤ 估算（6 种）。真实权重：下载的 `flux2-vae.safetensors` 在 CPU 上 768×512 解码，A / D / B / C 都走第一层，与原生 max|Δ| ≤ 1.6e-6（fp32）。`tests/make_synthetic_vaes.py` 多一个 `synthetic_flux2`，bench / 节点检查脚本 CPU 冒烟通过。真机命令与逐行预测：README §9.12（X / Y / Z）。

### 9.19 第一次使用时的自检峰值不在估算里（分析；用户选了 B + A，实现见 §9.20）

**现象（CT 700，Flux 2，4fbaa22）：** bench X 1344×768 第一次（含 B 的自检）reserved +0.78 / GTT +0.81，估算 0.76，第二次 0.62；`check_vae_node.py` 4K 节点预算 3G，副本第一次（含自检）reserved +2.68 / GTT +2.96，估算 2.97、预算 3.00，再次 2.44。

**自检做了什么（`vae_engine._self_test_run`）：** ① `fp32_copy`：decoder（+ `post_quant_conv` / BatchNorm）的 fp32 副本；② `reference_decode`：模型类自己的 `decode` 在 24 × 24 latent（输出 192 × 192）上整图 fp32 解码，**不在第二层分块下**；③ 同一 latent 按 40 行强制条带走第一层（8 MiB 工作区）；④ 释放、`gc.collect()`、`soft_empty_cache(True)`。之前 `load_models_gpu` 收到的是 `selftest_memory()` = 参数 × 4 + 16 MiB + 256 MiB（LDM 约 461 MiB）。

**alloc_sim 模拟（meta 设备，CT 700 后端）：**

| 结构 | fp32 副本 | 参照解码后的 reserved 峰值 | 整个自检 | 自检后仍占 |
|---|---|---|---|---|
| SDXL / Flux / Flux 2（LDM） | 189 MiB | 736 MiB | **738 MiB** | 0 |
| Wan 2.1（qwen_image_vae） | 280 MiB | 692 MiB | 692 MiB | 0 |
| 参照解码也在 8 MiB 分块下：LDM / Wan | 同上 | 320 / 342 MiB | 320 / 342 MiB | 0 |
| 再把自检 latent 降到 16：LDM / Wan | 同上 | 254 / 314 MiB | 254 / 314 MiB | 0 |

峰值的大头是参照解码里全分辨率 3×3 卷积的 Slow2d im2col columns（fp32，256 通道 × 9 × 192²，约 340 MiB）加 fp32 激活，其次是 fp32 权重副本；自检和解码是先后执行的，自检结束时全部释放、缓存清空。

**两个现象分别是什么：**

1. **1344×768：自检本身的峰值高过了这次解码。** 自检约 0.74 GiB（模拟；实测 0.78 含碎片），解码只要 0.62，所以第一次的峰值是自检的。与图的大小无关：4K A 第一次（含 A 的自检，bench Y）reserved 1.11 = 第二次，因为解码 1.10 高过自检。凡是这次解码的峰值低于约 0.75 GiB（小图、紧预算）都会出现。
2. **4K 节点副本：多出的 0.24 GiB reserved（GTT 多 0.52）不是自检的峰值**（0.74 < 2.44）。这个进程（`check_vae_node.py`）在副本解码之前没做过任何 GPU 计算，第一批 GPU 计算的一次性开销都落在这次测量里：BLAS 句柄和工作区（经 PyTorch 的缓存分配器分配、进程内一直活着，会钉住它所在的段，`empty_cache` 释放不了，arena 只好另开一段：0.24 + 2.44 = 2.68，「再次」从新起点量是 2.44）、第一次用到的 GPU 内核代码（只在 GTT 里，不在 reserved 里：GTT 比 reserved 多 0.28）。bench 里原生先跑过，所以没有这一项。ComfyUI 服务里采样早就付过这笔开销，VAE 解码时不会再有。这是推断，要在真机上区分（下面的命令 AA）。

**可选的改法和代价：**

| 改法 | 做法 | 好处 | 代价 |
|---|---|---|---|
| A 自检算进估算和预算 | 某结构第一次使用时，交给 `load_models_gpu` 和预算比较的都是 max(计划估算, 自检估算)；自检估算用 alloc_sim 校验过的公式（fp32 权重 + 参照解码上界） | 估算重新是上界；预算严格成立 | 不改自检时自检要 0.74 GiB：小于这个的预算第一次就放不下（换方案也没用），第二次又放得下，前后不一致；要多维护一个内存模型 |
| B 自检变小 | 参照解码也在第二层分块下跑（工作区与条带不同，如 32 MiB，块边界不同）；可再把自检 latent 24 → 16 | 0.74 → 0.32 GiB（latent 16：0.25；Wan 0.34 / 0.31）；剩下主要是 fp32 权重副本，省不掉（1e-4 的判据要 fp32） | 参照和条带共用第二层的卷积分块代码，分块本身的错误自检查不出（第二层另有 `test_vae.py` 的 53 项单测）；latent 16 时条带边界少一些；耗时几乎不变 |
| C1 自检放进 arena | 先按选好的计划预留 arena，再跑自检（自检的块从 arena 的空闲段里分），然后解码 | 自检 ≤ arena 时 reserved 不增加；与 B 合用时几乎所有解码（arena ≥ 0.35 GiB）都不会超出 | 执行顺序变复杂（预算选择、自检失败退回第二层都要在 arena 预留之后处理）；arena 小于自检时仍要 A 兜底 |
| C2 加载 VAE 时就自检 | 包装 `VAE.__init__` / VAELoader，加载后立刻自检 | 解码时不再有自检 | 加载时就要把 VAE 搬上 GPU（违背 ComfyUI 的惰性加载）、每次加载多 1–2 s 和约 0.74 GiB 瞬时峰值（哪怕这个 VAE 从不走第一层）；峰值只是挪到了加载时 |
| C3 自检结果缓存到磁盘 | 按结构 + torch / ComfyUI / Monoload 版本记住通过，下次进程不再自检 | 只有第一次运行付一次 | 违反「每个进程第一次使用前自检」的规矩（要你同意）；驱动或库升级后缓存可能过期，需要版本键 |
| 不改 | 文档写明第一次解码多一次约 0.74 GiB 的自检峰值 | 零改动 | 紧预算下第一次可能超预算 |

第 2 种现象（进程里第一批 GPU 计算）与改法无关：Monoload 不该为它预留，也测不准（每台机器、每个库版本不同）。

**建议：** B + A。先把自检降到约 0.3 GiB（参照用不同工作区分块，latent 保持 24），再把这个（小而稳定、模拟可校验的）自检峰值算进第一次的估算和预算比较；C1 以后需要时再加。

**真机区分（`tests/check_selftest_mem.py`，README §9.12 的 AA）：** 新进程里依次单独测「第一批 GPU 计算（`--warmup`）」「自检本身」「解码 1」「解码 2」，每一步之前清缓存，报告 reserved / GTT 峰值和清缓存后仍占的量。预测：自检 reserved 峰值约 0.74 GiB；不加 `--warmup` 时自检之后仍占一些 reserved 和 GTT（第一批 GPU 计算的一次性开销），加了以后这部分出现在 warm-up 一行、自检之后为 0；解码 1 = 解码 2 = 2.44（4K 预算 3G）。

### 9.20 自检缩小、并算进第一次的估算和预算（vae-selftest-budget，用户选 B + A）

**CT 700 的诊断（命令 AA，0a4e177）：** 自检 reserved 峰值不加 warm-up 0.85、加 warm-up 0.78 GiB（之后留下 0），报给 `load_models_gpu` 的是 0.45；进程里第一批 GPU 计算留下 0.07 GiB reserved（加 warm-up 后出现在 warm-up 一行），自检之后 GTT 仍剩 0.08（量小，先不管）。§9.19 的两点推断都成立。

**B：自检变小。** 参照解码（模型类自己的整图 `decode`）也在第二层分块下跑，工作区 `SELFTEST_REF_WORKSPACE` = 32 MiB，与条带那遍的 8 MiB 不同（块边界不同，保留一部分独立性）。自检 latent 仍是 24。alloc_sim：LDM（SDXL / Flux / Flux 2）0.74 → **0.37 GiB**（378–380 MiB），Wan 0.69 → 0.38（390 MiB），结束后都不留东西。剩下的主要是 fp32 权重副本（LDM 189 MiB，Wan 280 MiB）。

**新的自检上界 `StripeAdapter.selftest_memory()`：** fp32 权重副本 + max(参照解码的第二层上界 = 4 × 最大激活 + 2 × 32 MiB + 输出, 条带那遍的计划估算 + 输出) + 参照输出 + 32 MiB 余量；全部按自检的几何（24 × 24，fp32）算，与图的大小无关，按结构缓存。LDM / Flux 2 / Wan 都是 430 MiB，模拟峰值 378–390 MiB 在其内（以前给的 0.45 GiB 低于真实的 0.74）。

**A：算进第一次的估算和预算比较。**

* 预算（`choose_budget`）：某个第一层变体的结构还没自检过时，它的估算按 max(计划估算, 自检上界) 比较预算（自检和解码先后执行，第一次的峰值是两者较大的那个）。候选标签和「放不下」的报错里写明「首次使用：解码前的自检最多占 X；解码本身 Y」（`vae.cand_selftest`，中英文）。只有这次就要做的自检才算：自检过一次之后，同一个预算按计划估算比较。
* 记录和日志：这次解码里跑了自检时，`last_decode()` / Info 的估算 `total` = max(计划, 自检)，另记 `plan` 和 `selftest`；解码日志的估算后面加「解码前的首次自检最多 X」（`vae.est_selftest`）。交给 `load_models_gpu` 的不变：自检之前用自检上界，解码之前用计划估算（自检已经释放）。
* 默认策略（不设预算）不受影响：没有预算要比较，只是记录里的估算变成两者较大的那个。

**影响：** 自检上界 0.42 GiB，只有比它小的预算会在第一次解码时把第一层排除（以前的 1344×768 第一次 0.78 > 估算 0.76 的情况：现在自检约 0.40 < 解码 0.62，第一次峰值就是解码的）。

**测试：** `tests/test_vae_flux2.py`：参照解码确实在 32 MiB 的块下跑（还有条带的 8 MiB）；第一次解码的记录 `selftest` = 自检上界、`total` = max，第二次没有；把自检上界改成比预算大时第一次报错并写明自检，自检记录在案后同一预算走第一层、== 原生；alloc_sim 上 Flux 2 / SDXL 的自检峰值 ≤ 上界且 ≤ 0.45 GiB。`tests/test_vae_stripe.py`：Wan 的自检峰值 ≤ 上界。`alloc_sim.selftest_trace(model)` 按 `_self_test_run` 的步骤重放自检，含最后的比较（max|参照|、max|条带 − 参照|、isfinite；review 07 补上）：它分配 0.84 MiB（小块池），reserved 不增加，Wan / Flux 2 / SDXL 的峰值仍是 390 / 380 / 378 MiB ≤ 上界 430 MiB，上界不用改。

### 9.21 第一次解码多 0.13 GiB、解码后 arena 被输出钉住（分析；用户选了 ① a、② a + 解码结束时清缓存，实现见 §9.22）

**现象（命令 AA，Flux 2，4K，预算 3G）：** ① 解码 1 的 reserved 峰值 2.57、解码 2 是 2.44（两条命令都这样，自检和第一批 GPU 计算已经单独测过），多出的 0.13 解码完就释放了；② 两次解码之后、`empty_cache` 之后都还留着 2.44 GiB reserved 和 GTT。

**① 的原因：第二层的形状探测留下的缓存块。** 设了预算时 `choose_budget` 要比较第二层，第二层的估算要 `_probe`：每个模型（和 latent 布局）第一次用 8 × 8 的 latent 跑一次小解码（带 forward hook，不在第二层分块下），按模型缓存。它在解码 1 的测量窗口里、arena 预留之前运行；用完的块留在分配器的缓存里没有清，arena（一整块 2.44 GiB）放不进这些小段，只能另开一段，峰值 = 探测留下的缓存 + arena。解码 2 时探测已缓存，不再运行。alloc_sim 复现：先跑探测再按同一计划解码，reserved 峰值 2.55（实测 2.57），不跑探测 2.44；探测之后、预留 arena 之前清一次缓存就回到 2.44。这些块是缓存、不是活着的张量，所以解码完 `empty_cache` 就释放了。对照：没有预算的默认策略不跑探测（bench X 1344 第一次没有这项）；bench Z 的 4K 第一次没有这项是因为 1344 时已经探测过（缓存键与分辨率无关）；在 `check_vae_node.py` 里探测之后紧接着是自检，自检结束时清缓存，所以那里的多出量另有来源（第一批 GPU 计算，持久的工作区钉住了它落脚的那一段）。

**① 的可选改法：**

| 改法 | 代价 |
|---|---|
| a 预留 arena 之前清一次缓存（`_decode_layer1`，每次解码一次 `soft_empty_cache`） | 一次 `empty_cache`（毫秒级）；之前缓存的块（采样留下的）这时还给系统，下一次采样要重新 `hipMalloc`（`_MemProbe` 进入时本来就清一次，影响相同） |
| b 只在形状探测之后清缓存（每个模型一次） | 几乎没有；只针对这一个来源 |
| c 把探测留下的缓存算进估算 | 不建议：大小取决于分配器状态，估算会变松 |

建议 a（覆盖解码之前所有残留，和 `_MemProbe` 的做法一致）。

**② 的原因：输出缓冲分配在 arena 的段里。** 已从代码确认：`StripeAdapter.run` 先 `reserve_arena`（申请一整块再释放，缓存成一个空闲段），然后才分配 `pixel_samples`（整个 batch 的输出，fp32，`--gpu-only` 时在设备上；4K 约 95 MiB）。这时唯一放得下它的空闲块在 arena 段里，按最佳匹配就切在那里——这本来是设计里的一部分（输出算在 arena 的 persistent 里，§9.13.4）。解码完其他块都释放了，但输出还活着，缓存分配器只释放完全空闲的段，于是整段 2.44 GiB 一直 reserved。alloc_sim 复现：输出活着时 `empty_cache` 后仍留 2.44（默认计划 2.18）。对照：第二层 / 原生的输出落在别的小段里，只留约 0.1 GiB。

**② 的实际影响：** ComfyUI 开 `--gpu-only` 时 VAEDecode 的输出（IMAGE）留在 GPU 上，节点输出缓存一直保留到这个节点下次重新执行；`main.py` 在 prompt 之后（按 gc 间隔）调用 `soft_empty_cache()` 也释放不了这一段。PyTorch 进程内部这段的空闲部分可以复用（ComfyUI 的 `get_free_memory` 把 reserved − active 算作空闲，下一次采样的块可以放进去），损失的是进程外：在统一内存的 CT 700 上，这 2.4 GiB GTT 是被钉住的系统内存，CPU 那边（加载模型文件、页缓存）用不了。一个工作流里有两个 VAEDecode 时会钉住两段。

**② 的可选改法：**

| 改法 | 做法 | 代价 |
|---|---|---|
| a 输出在 arena 之外单独分配 | `run` 里先分配输出（独立的段，按 2 MiB 取整），再预留 arena（arena 里不再含输出） | alloc_sim：峰值不变（4K 预算计划 2.44，默认 2.18），输出活着时清缓存后只留 0.09 GiB；arena 的布局（persistent / long_lived、`saves_fit`）要改并重新验证 51 个读数和 reserved ≤ 估算；输出在 CPU 上时（不开 `--gpu-only`）不变 |
| a + 解码结束时清缓存 | 在 a 的基础上，受管理的解码结束时 `soft_empty_cache()` | arena 立刻还给系统，不用等 ComfyUI 的 gc 间隔；下一次大分配要重新 `hipMalloc`（毫秒级） |
| c 结束时把输出拷到 arena 外 | 解码完再拷一份 | 做不到：旧输出活着时 arena 段释放不了，新的一份按最佳匹配还会落进 arena 的空闲部分 |
| d 用私有内存池（`torch.cuda.MemPool`）放解码的临时块 | 输出在默认池，临时块在私有池，用完整个池释放 | ROCm 上这个接口是否可靠未知；alloc_sim 不模拟内存池，验证手段没有了；改动大 |
| 不改 | 文档写明 | 每次第一层解码之后，在输出被 ComfyUI 缓存期间多钉住一个 arena（2.2–2.4 GiB）的 GTT |

建议 a + 解码结束时清缓存；① 的 a 可以和它一起做（同一个函数里）。

### 9.22 解码前后清缓存、输出单独一段（vae-arena-output，§9.21 ① a、② a + 结束时清缓存）

**CT 700 验收通过（4f140ea，README §10.6）：** decode 1 = decode 2 = 2.43 GiB，解码后只留输出 0.09；方案 A 4K 1.10（检查点下移生效）；预算选择不变；自检 0.37（加 warm-up）≤ 上界 0.42，不加 warm-up 的 0.45 含进程第一批 GPU 计算的约 0.07（不在上界里，§9.19）。

**`StripeAdapter.run` 的顺序：** 清缓存（`soft_empty_cache`）→ 分配输出缓冲（整个 batch）→ 预留 arena → 逐个样本解码 → 清缓存。

* 先清缓存：之前的工作（设了预算时的形状探测，§9.21 ①）留在缓存里的块还给系统，arena 和解码的第一批请求不会落进这些小段里。
* 输出在 arena 之前分配：缓存刚清空，它只能单独开一段（`out_segment`：≥ 10 MiB 按 2 MiB 取整，1–10 MiB 一个 20 MiB 的段，≤ 1 MiB 在小块池的 2 MiB 段里）。arena 里不再有一直活着的块。
* 结束时清缓存：解码完 arena 整段空闲，立刻还给系统，只留输出（4K 95 MiB），不用等 ComfyUI 的 gc 间隔（§9.21 ②）。
* 输出在 CPU 上时（不开 `--gpu-only`）：输出不占设备，`out_segment` = 0，arena 同样在结束时还回去。
* 代价：每次解码两次 `soft_empty_cache`（同步 + `empty_cache`，毫秒级）；之前缓存的块（采样留下的）这时还给系统，下一次采样要重新 `hipMalloc`。

**计划（`Plan`）：** persistent 只剩 latent 的拷贝；存档的位置（`long_lived` / `saves_fit`）里不再有输出；largest 的候选去掉输出；估算 = `out_segment` + arena + largest + 16 MiB。`OUTPUT_IN_ARENA = True` 还原以前的布局（不清缓存、输出在第一个样本的前缀之后分配、在 arena 里），只给 alloc_sim 重放旧版本的读数用。

**布局变了带出的一个问题，以及处理（`move_low`）：** 新布局在 alloc_sim 里扫了 37 个配置，只有一类变高：没有存档的方案 A，4K，工作区 128 MiB（Flux 2 / SDXL；bench Z 的 4K 1.5G 选的就是它，bench Y 的 A 行也是），1.105 → 1.229 GiB（仍 ≤ 估算 1.49）。原因：前缀的最后一步（mid 块的残差相加）的输出（检查点，127 MiB）落在它的一个输入后面，那个输入释放后，检查点前面留下一个 127 MiB 的洞。以前输出缓冲（95 MiB）正好放进这个洞；现在洞空着，统计 pass 的卷积 columns 块是 128 MiB，比洞大 1 MiB，放不进，只好在 arena 外另开一段。方案 B / C / D 有存档，检查点是前缀之前在 arena 最前面预先分配的，没有这个洞。

处理：检查点不在 arena 最前面时（没有存档），前缀之后试着分配一块检查点大小的块。按最佳匹配，它落在能放下它的最小空闲块里：如果落在检查点下面（那个洞），就把检查点拷过去，原位置释放后与后面的空闲区连成一片；如果落在检查点上面（arena 的尾部）或者新开了一段，就立刻释放，块并回原处，布局不变（新开的段随即清掉）。代价：一次检查点大小的分配，移动时多一次拷贝（4K 127 MiB，毫秒级）。峰值不增加：拷贝时两份检查点同时活着，但新的一份放在已经空着的洞里。

试过两种更简单的做法，都不行：

* 无条件挪：洞比检查点小时，检查点反而往上挪，SDXL A 4K 64 MiB 工作区 1.02 → 1.14 GiB，Qwen fp32 4K 1.44 → 1.69。
* 没有存档时也在前缀之前预先分配检查点：前缀的峰值多一个检查点，A 4K 32–96 行 1.06 → 1.18。

**重新验证（alloc_sim）：**

* **CT 700 读数：** 原有的 51 个读数（v1 / v2 / v3 / w128 = 5d668b6..6324592，旧布局重放）都在 0.02 GiB 以内，与改动前相同。另把 Flux 2 的 12 个读数（README §10.5 的 X / Y / Z）加进了 `MEASURED`，同样用旧布局重放，都在 0.01 GiB 以内。
* **37 个配置的新布局（Qwen 9、Flux 2 19、SDXL 9；默认策略、各方案、bench Z 的六档）：**
  * reserved 全部 ≤ 旧布局：相等，或低 2–12 MiB（arena 的余量按不含输出的 live 算）。
  * reserved 全部 ≤ 估算；估算变化 ≤ 6 MiB。
  * 输出活着时清缓存，只剩输出的段：1344 0–12 MiB、2688 48 MiB、4K 96 MiB、8K 380 MiB。旧布局是整个 arena。
  * Qwen 4K 5 × 432 行和 8K 默认在两种布局下都有一块请求落在 arena 外（以前就有：need > arena），估算包住了它。

| 配置（Flux 2，GiB） | 旧布局 | 新布局 | 估算（旧 → 新） | 解码后留下（旧 → 新） |
|---|---|---|---|---|
| 4K 预算 3G，B 12 × 180（AA / Z / 节点副本） | 2.439（先跑形状探测 2.547） | 2.434（先跑探测也是 2.434） | 2.968 → 2.962 | 2.438 → 0.094 |
| 4K 默认 B 17 × 128 | 2.182 | 2.176 | 2.688 → 2.682 | 2.180 → 0.094 |
| 4K A 17 × 128（1.5G） | 1.105 | 1.104（不挪检查点 1.229） | 1.490 → 1.488 | 1.104 → 0.094 |
| 1344 默认 B 6 × 128 | 0.621 | 0.609 | 0.758 | 0 → 0 |

**预算选择不变：** bench Z 的六档、`check_vae_node.py` 的节点副本，选中的方案、条带高度和工作区都与以前相同。候选的估算只变了几 MiB，例如 2688 1.5G 的 A 220 行 1.50 → 1.49。

**测试：**

* `tests/test_vae_stripe.py`：
  * 解码的顺序：清缓存 → 输出 → arena → 清缓存（CPU 上没有 arena）。
  * `move_low` 的三种情况：洞在下面就挪；块在上面不挪；新开了段不挪，并清缓存。
  * 估算 = 输出段 + arena + largest + 16 MiB。
  * Qwen 三个尺寸新旧布局对比：峰值不升、解码后只留输出、清缓存次数多 2、条带不变。
  * 旧布局重放 w128 的两个读数。
* `tests/test_vae_flux2.py`：
  * AA 的两个现象：旧布局先跑探测 2.55、不跑 2.44，与实测 2.57 / 2.44 相差 ≤ 0.03，解码后留下整个 arena；新布局两者相等，只留输出。
  * 方案 A 4K 128 MiB：新布局 ≤ 旧布局，旧布局与实测 1.11 相差 ≤ 0.03；不挪检查点时高 0.1 GiB 以上。
* `alloc_sim.decode_trace` 新增：
  * `output_in_arena`：旧布局；
  * `probe`：先跑预算的形状探测；
  * `info["stays"]`：输出活着时清缓存后留下的量。

## 10. 总开关 `MONOLOAD` 与优先级（settings-master-switch）

**规则：** 环境变量是全局默认值；节点上明确选的值只对那一个模型 / VAE 生效，而且总是压过全局。逐项判断：节点上明确选的 > 高级环境变量 > 内置默认；节点上选「跟随全局」（`default`）的项继承全局。

**总开关（`monoload/settings.py`，不导入 torch / ComfyUI）：** `MONOLOAD` 不设或 `1`（`true` / `yes` / `on`）= 开启，所有模型和 VAE 用默认策略；`0`（`false` / `no` / `off`）= 全局原生；其他值警告后按开启处理。启动时读一次（`master()`；测试用 `set_master()`，切换前要先卸载模型）。`MONOLOAD_DISABLE=1` 仍是「什么都不装」。

**钩子总是装上，关闭时直通：**

* `hotpatch`：`_active(patcher)` 先看 `_enabled(patcher)`（这一版就是总开关；LoRA 节点分支会改成逐个 patcher 判断），不启用就返回 False，所有替换的方法调用原方法。`patch_weight_to_device` 和 `ModelPatcherDynamic.load` 以前不看 `_active`，现在也看。于是 `MONOLOAD=0` 时整体加载走原生的「原地合并 + 备份」，lowvram 走原生的 `LowVramPatch`，hook 走原生的 `patch_hook_weight_to_device`，撤掉时按备份还原——全是原生代码，结果逐位一致（`tests/test_master_switch.py`）。
* `unpatch_model` 以前在 `_active` 时去掉运行时 patch；现在改成看模型上的标记 `_monoload_runtime`（装运行时 patch 时设，全部去掉时清）：不论开关怎么变，模型上都不会残留运行时 patch，没有运行时 patch 的模型也不用遍历模块。
* `release`：总是包装 `execute_async`，每个 prompt 结束时判断 `enabled()` = 总开关开 且 不是 `MONOLOAD_KEEP_LORA`（全局默认「保留」，`keep()` / `set_keep()`）。
* `vae`：总是包装 `VAE.decode`（ComfyUI 接口不符时除外）。没有自己模式的 VAE 按 `global_mode()` 决定：`MONOLOAD=0` → 原生；`MONOLOAD_DISABLE_VAE=1` / `MONOLOAD_EXACT=1`（`set_native()`）→ 原生；`MONOLOAD_DISABLE_VAE_STRIPE=1` → 只用第二层；否则 `auto`。来源记成 `env`，日志写明是哪个变量（`mode native (env MONOLOAD=0)`）。全局原生的解码直接调用原 `decode`，只在 DEBUG 级别记一行，避免每次解码都刷日志。

**开销（`MONOLOAD=0`，CPU，把原方法换成空函数只量包装本身）：** `patch_weight_to_device` 每次 +0.2 µs（每次加载每个被 patch 的权重调用一次）；`VAE.decode` 包装每次约 3.4 µs（每次解码一次，`resolve_settings` + 记录）；其他方法多一次函数调用和一次判断。相对一次加载或解码可以忽略。

**VAE 节点的变化：**

* `mode` 的 `default` = 跟随全局；`auto` = 为这个 VAE 打开管理（全局关着、包括 `MONOLOAD=0` 也打开）。
* 预算改成下拉框 `budget`：`default`（跟随全局 `MONOLOAD_VAE_BUDGET`）/ `unlimited`（这个 VAE 不限预算，覆盖全局预算；存成 `budget: None`，来源 `node`，日志 `budget unlimited (node)`）/ `custom`（用 `budget_gib`，必须 > 0）。`budget_gib` 只在 `custom` 时生效，选别的时填了大于 0 的值会在日志里说明没用上。
* 存进工作流的取值都是英文、不变（`default` 等）。ComfyUI 的控件值按位置存（`widgets_values`），所以新下拉框排在最后：旧工作流（`budget_gib`, `gn_scheme`, `stripe_rows`, `mode`）照常加载、各控件值对得上；代价是界面上 `budget` 在 `budget_gib` 下面隔了三项，以及旧工作流里 `budget_gib > 0` 的预算不再生效（`budget` 默认是 `default`），要手动改成 `custom`（README §4.1）。
* `MONOLOAD_DISABLE=1`：节点原样返回输入的 VAE（不是副本），日志说明一次。

**高级变量的新含义：** `MONOLOAD_EXACT`（合并的全局默认是逐位一致；VAE 全局默认原生）、`MONOLOAD_KEEP_LORA`（全局默认保留）、`MONOLOAD_DISABLE_VAE`（VAE 全局默认原生）、`MONOLOAD_DISABLE_VAE_STRIPE`、`MONOLOAD_VAE_BUDGET`、`MONOLOAD_VAE_GN_SCHEME`、`MONOLOAD_VAE_STRIPE_ROWS`、`MONOLOAD_VAE_WORKSPACE` 都是全局默认值，节点上有对应项的可以逐项覆盖（`MONOLOAD_VAE_WORKSPACE` 节点上没有）。LoRA 的逐模型设置在 LoRA 节点分支里加。

**测试（`tests/test_master_switch.py`，18 项）：** 解析；`MONOLOAD=0` 且没有节点时，带 LoRA 的两层 `comfy.ops` 模型在整体加载、lowvram 加载（`lowvram_model_memory=1`）、Hook LoRA、撤掉 hook 四种情况下的输出与卸掉钩子的原版逐位相同，备份数相同，撤掉后权重逐位还原、没有残留的 weight function；VAE 解码逐位相同、没有 INFO 日志；释放只在开启且不保留时运行；包装开销；`MONOLOAD=0` / `MONOLOAD_DISABLE_VAE` / `MONOLOAD_EXACT` 下节点 `auto` 打开、`default` 原生；预算下拉框；`MONOLOAD_DISABLE` 透传。`tests/test_entry.py` 加了 `MONOLOAD=0` 组合，并改成检查每种组合下都装上、全局默认对得上。

## 11. Monoload LoRA Settings 节点：按模型决定 LoRA 怎么处理（lora-settings-node）

**设置存在哪里：** 节点输出 `model.clone()`（CLIP：`clip.clone()`，也就是克隆它的 patcher），在 clone 的 `model_options["monoload_lora"]` 里放一个只含明确选了的项的 dict（`monoload/lora_overrides.py`）。选 `model_options` 是因为 `ModelPatcher.clone()` 用 `deepcopy_list_dict` 复制它：`LoraLoader` / `LoraLoaderModelOnly`（`comfy.sd.load_lora_for_models` 里 `model.clone()` + `add_patches`）、`CLIP.clone()`、采样时的 clone 都会带上，所以节点放在 LoRA 加载器前后都行（测试两种位置都确认）。没有选 `attachments`：clone 时只是按引用复制，能用但语义上是「附加对象」；也没有用对象属性：clone 不复制。

**patches_uuid：** 设置和上游不同的 clone 换一个新的 `patches_uuid`。原因：同一个底模的两个 clone 共享 `self.model`，ComfyUI 切换时看 `model.current_weight_patches_uuid` 是否等于新 patcher 的 `patches_uuid` 决定要不要先还原权重（`partially_load` 里 `unpatch_weights`）。只改设置、不换 uuid 的话，切到另一个 clone 时不会重新加载，上一个 clone 的处理方式（运行时 patch 或烘焙进权重）会留下来。换了 uuid，切换时先 `unpatch_model`（原生还原备份——备份字典在 clone 之间共享——再由 `_unpatch_model` 按模型上的 `_monoload_runtime` 标记去掉运行时 patch），再按新 clone 的方式加载。节点全留 `default` 时 uuid 不变（不触发重新加载）。

**每项怎么取（`resolve`，逐项）：**

| 项 | 节点选了 | 否则 |
|---|---|---|
| `mode` | `enable` / `native` | 总开关：开 → `enable`（default），`MONOLOAD=0` → `native`（env） |
| `merge` | `fused` / `exact` | `MONOLOAD_EXACT=1` → `exact`（env），否则 `fused`（default） |
| `after_prompt` | `release` / `keep` | 模式是 `native` → `keep`（和原版一样，来源同 mode）；否则 `MONOLOAD_KEEP_LORA=1` → `keep`（env），否则 `release`（default） |

**运行时合并按 ModelPatcher 决定（核对过 hotpatch）：** 以前「是否接管」和「合并路径」都是全局的：`_active()` 只看类，`MonoloadRuntimePatch.__call__` 读全局 `_MODE["exact"]`。现在：

* `_enabled(patcher)` = `lora_overrides.enabled(patcher)`：patcher 的 `mode`，没有就看总开关。所有替换的方法都经过 `_active()`（它先看 `_enabled`），`patch_weight_to_device`、`ModelPatcherDynamic.load` 也看。没有节点时就是总开关，行为与以前相同。
* 合并路径：模型的绑定（§3.2）记着当前生效的 patcher 的 `exact = lora_overrides.merge_exact(patcher)`（节点选了就是 True / False，没选是 None = 每次调用时读全局 `settings.exact()`，`set_exact()` 照旧立刻生效）。`load` / `partially_load` 时绑定指向正在加载的 patcher，所以同一个 `patches_uuid`、不重新 `load()` 的 clone 也用自己的合并方式（review 01；以前写的「每次 load() 都重新装，所以总是对应当前加载的 patcher」在原生提前返回的路径上不成立）。
* 全局默认的 `exact` / `keep` 挪到 `settings.py`（`hotpatch.set_exact` / `is_exact`、`release.keep` / `set_keep` 转发过去），`lora_overrides` 不导入 torch / ComfyUI。

**prompt 结束后按 patcher 释放：** `release_after_prompt` 对每个已加载模型看 `wants_release(patcher)`；输出缓存里的 MODEL / CLIP 也按各自的 patcher 判断；没有 patcher 的 Hook LoRA 组按全局默认；`LoraLoader` 等的文件缓存（`loaded_lora`）在全局默认是释放、或者这次释放了任何东西时清掉（保留的模型的 LoRA 张量被它的 patches 引用着，清掉文件缓存不影响它，只是加载器重新执行时要重读文件）。全局默认是保留、而且这个进程里没用过 LoRA 节点时直接返回（`MONOLOAD=0` 不加节点时零开销）。原生模型的 `release`：`unpatch_model` 走原生，按备份逐位还原，然后照常指回底模。

**测试：** `tests/test_lora_node.py`（19 项）用 `tests/make_synthetic_checkpoint.py` 生成的随机权重 SD1.5（真实结构，fp16，约 2 GiB）和一个 rank 4 的普通 LoRA（改 UNet 的 184 个权重、文本编码器的 72 个），走 ComfyUI 自己的 `CheckpointLoaderSimple` / `LoraLoader` / `LoraLoaderModelOnly`，CPU 上 8×8 latent 采一步。`exact` 和 `native` 与卸掉钩子的原版比较 `torch.equal`（TE 输出和 latent），`native` 的备份数（256）与原版相同；`fused` 与原版的相对差异 3e-4。用同一个合成模型跑了 `tests/test_release.py`（默认 / `MONOLOAD_KEEP_LORA=1` / `MONOLOAD_EXACT=1`）：42 / 22 / 42 项全过，默认行为没变。真机：`tests/check_lora_node.py`（README §9.8 的 V）。

## 12. Monoload Info 节点（info-node）

**节点框里怎么显示文字（核对过锁定版本）：** ComfyUI 0.31.0 的后端只负责把节点返回的 `{"ui": {...}}` 通过 `executed` 消息发给前端，前端怎么画由前端扩展决定；经典节点接口本身没有「在节点里显示文字」的机制。前端 1.48.7（`comfyui_frontend_package`）里核心节点 `PreviewAny`（「Preview as Text」）用的是扩展 `Comfy.PreviewAny`：`onNodeCreated` 时调 `addTextPreviewWidgets(node)`（一个 `textPreview` 控件 + Markdown / 纯文本开关），`onExecuted` 时调 `updateTextPreviewWidgets(node, message)`（取 `message.text`，数组用空行连接）；这两个函数挂在 `window.comfyAPI.textPreviewWidgets` 上。所以插件导出 `WEB_DIRECTORY = "./web"`，`web/monoload_info.js` 对 `MonoloadInfo` 做同样的事；拿不到这两个函数（别的前端版本）时退回一个只读的多行文本控件。后端：`OUTPUT_NODE = True`（`text` 不接也执行），返回 `{"ui": {"text": [text]}, "result": (text,)}`，`IS_CHANGED` 返回 NaN（与自己不相等，每次都重新执行）。在云端容器里用锁定镜像起了 ComfyUI 服务，用 Chromium（Playwright）打开真实前端：节点框里显示了文字，连上 checkpoint → VAE Settings → VAE Decode → Info 和 LoRA Settings → Info 的图也显示了这次解码的记录和 LoRA 名字。

**内容（`monoload/info.py`，每次调用重新生成）：** 开头总是版本（`monoload.__version__`）和 commit（直接读插件目录的 `.git/HEAD` / refs / `packed-refs`，不调用 git）、总开关；不接 `vae` / `model` 时列出每一项全局默认值和来源；接了就分别写 VAE 段和 MODEL 段。来源：环境变量里设了就是 `env 名字=值`，否则值等于内置默认是 `built-in`，不等（测试 / bench 用 `set_*` 改过）是 `set at runtime`。

**解码记录按 VAE 对象：** `vae._RECORDS` 是 `WeakKeyDictionary`（VAE 对象 → 这个对象最后一次解码的记录），`_decode` 每次（受管理的、原生的、报错的）结束时写入，`decode_record(vae)` 读。节点做的副本是另一个对象，所以副本和原 VAE 各记各的；`last_decode()`（全局最后一次）照旧。受管理的解码外面套 `_MemProbe`：GPU 上记 `torch.cuda.max_memory_reserved` 相对解码前的增量（解码开始时 `reset_peak_memory_stats`），有 amdgpu 的 `mem_info_gtt_used` 时开一个线程每 20 ms 读一次取峰值增量；CPU 上都不测。原生解码只记策略、原因和耗时（不测内存，保持 `MONOLOAD=0` 下开销接近零）。

**LoRA 名字：** patch 里没有文件名。插件入口包装 `nodes.LoraLoader.load_lora`（`LoraLoaderModelOnly` 也调用它），在它返回的 clone 的 `model_options["monoload_lora_names"]` 末尾加一条 `{name, strength}`（强度为 0、返回原对象时不记）。只有元数据：patch、`patches_uuid`、数值都不变；`model_options` 随 clone 复制，所以串联的加载器按顺序累积。别的插件的加载器不经过它，Info 只显示被改动的权重数。Hook LoRA（`CreateHookLora`）挂在条件上，不在 MODEL 段里。

**`images` 输入：** 不读，只让 Info 依赖 VAE Decode 的输出，从而在解码之后执行。没接时 Info 可能在解码之前执行，显示的是上一次的记录（或 `not decoded yet`）。

**测试：** `tests/test_info_node.py`（16 项，不需要模型文件）；`tests/test_entry.py` 检查三个节点、web 目录、`LoraLoader` 的包装。

## 13. 多语言（i18n）

**界面（节点名、输入名、下拉显示、提示）：** 用 ComfyUI 官方的 locales 机制（https://docs.comfy.org/custom-nodes/i18n）：`locales/en/nodeDefs.json`、`locales/zh/nodeDefs.json`，结构 `{节点类名: {display_name, description, inputs: {输入名: {name, tooltip, options: {存储值: 显示文字}}}, outputs: {"0": {name}}}}`。服务端 `app/custom_node_manager.py` 的 `/i18n` 把各插件的 locales 合并后给前端。英文文件从节点的 `INPUT_TYPES` 生成（名字与真实输入名一致），中文手写；`tests/test_messages.py` 检查两份都覆盖每个节点、输入、下拉选项、输出。

**锁定版本上核对的结果（前端 1.48.7，Chromium 实测）：** 节点标题、输入名、输出名、提示都按 `Comfy.Locale` 翻译；**下拉选项的显示文字不翻译**——前端读了 nodeDefs 的 `options` 键（文档里有），但没有接到 combo 控件上，显示的仍是存储值。combo 控件本身有一个只影响显示的钩子 `widget.options.getOptionLabel`（`_displayValue`、下拉菜单、Vue 版的 WidgetSelect 都用它，值不变）。所以 `web/monoload_i18n.js` 在 `nodeCreated` / `loadedGraphNode` 时给三个节点的 combo 控件设 `getOptionLabel`，文字取自同一份 `/i18n` 数据（`[locale].nodeDefs.<节点>.inputs.<输入>.options.<值>`，locale 依次试 `Comfy.Locale`、去掉地区的部分、`en`），每次显示时现查，切换语言立即生效。实测：中文界面显示「跟随全局 / 自定义」，`app.graph.serialize()` 的 `widgets_values` 和 `graphToPrompt()` 的输入仍是 `default` / `custom`。将来前端自己支持 `options` 时，这个扩展设的是同一份文字，不冲突。英文界面的显示文字也带说明（`default (follow global)`、`exact (bit-identical)`），存储值不变。

**后端消息（`monoload/messages.py`）：** 一张表 `M = {key: (英文, 中文)}`，`msg(key, **字段)` 按当前语言取、用命名字段格式化。语言：`MONOLOAD_LANG` 不设 / `en` → 英文（默认），`zh` / `zh-CN` / `zh_CN` → 中文，其他值警告后用英文；启动时读一次，`set_lang()` 给测试用。覆盖所有用户能看到的日志和报错：启动日志、LoRA 的报错（DynamicVRAM、`force_patch_weights`、非 comfy.ops 参数、形状改变）、释放日志、VAE 的环境变量警告、每次解码的日志（包括按预算选的理由、候选列表、计划描述 `Plan.describe`）、预算放不下 / OOM 的报错（以前是中文，现在默认英文）、自检通过 / 失败、节点的日志和输入错误、Info 节点的全部文字；句子里的设置值和来源（`node` / `env` / `default`、`enable` / `native` …）也按语言显示（`label()`）。不在表里、保持英文的：「不应该发生」的内部诊断（`StripeError` 的 internal error、分块卷积的 internal error 等，给报 bug 用；自检失败时它们出现在已翻译的警告句子里），以及照抄 ComfyUI 原文的一条 hook 警告。`tests/test_messages.py` 检查每项都有中英文、字段一致，代码里消息表以外没有中文。

**文档语言不变**（README / DESIGN / HANDOFF 仍是中文）。

## 14. UI 实测之后的修正（polish-after-ui-test）

* **`budget_gib` 的精度**：前端 1.48.7 的 FLOAT 控件按 `step` 推保存精度（`precision = max(0, -floor(log10(step)))`，`onFloatValueChange` 用 `toFixed(precision)`）：step 0.25 → 1 位小数，0.25 存成 0.3。改成 `step` / `round` 0.01（2 位小数）；浏览器里核对过 0.25、1.37 原样保存。控件顺序改成 `budget` 紧挨 `budget_gib` 前面；dev 不做旧工作流兼容（用户定）。
* **Info**：总是在最后列全局默认值表；当前模式下不生效的设置标出来（VAE `native`：预算 / 方案 / 条带高度；`layer 2 only`：方案 / 条带高度；LoRA `native`：merge）；所用的 GroupNorm 方案附一句说明。
* **方案说明**（`vae_ldm.scheme_positions`）：A 不存（每遍统计都从 H/8 存档重算），D 存 H/4 级输出，B 存 H/4 和 H/2 级输出，C 再存全分辨率每个块的输入；写进 tooltip（en / zh）、下拉标签、Info、README §4.1。
* **预算等设置的来源**：`_Applied` 把逐项来源放进 `_SETTINGS["src"]`，`vae._from(item)` 给出「Monoload VAE 设置节点」或「环境变量 X」；预算行、预算报错、强制的条带高度 / 方案、只用第二层都写来源，预算放不下的建议按来源给（节点：改节点上的预算或改成跟随全局 / 不限；环境变量：改 `MONOLOAD_VAE_BUDGET`）。
* **「memory leak with model SDXLClipModel」警告**：在锁定镜像里用 `/prompt` API 复现（合成 SD1.5 + 只改 UNet 的 LoRA）。条件：LoRA 没有 text-encoder key（`LoraLoader` 的 CLIP clone 不带 patch）+ Monoload LoRA Settings 接了 CLIP（又一层不带 patch 的 clone）+ 解码报错。已加载的 CLIP 是第二层 clone；release 丢掉两个节点的缓存输出后，这两层 clone 被报错留下的引用环（执行器里的 traceback / 列表）留到 release 的 `gc.collect()` 才一起回收；ComfyUI 的 `LoadedModel._switch_parent` 只往上切一层，切的时候父节点也已经死了，于是 LoadedModel 没有 patcher、但模型（底模的 `cond_stage_model`）还活着，`cleanup_models_gc` 每次加载都报「memory leak」。不是 Monoload 持有引用（追查过 CLIP clone 的引用者：只有 ComfyUI 的输出缓存和执行器的列表；解码记录、LoRA 名字元数据、Info、报错对象都不引用 CLIP）。对照：正常运行、带 TE key 的 LoRA（clone 带 patch，release 直接指回底模）、不加 LoRA 设置节点（只有一层）都不出现。修复：release 前记下每个已加载 clone 的祖先链（弱引用），`gc.collect()` 之后把没有 patcher 的条目指回活着的最近祖先（同一个模型），再同步 uuid。`tests/test_release_chain.py` 构造同样的两层 clone + 引用环：去掉修复时出现两条警告、LoadedModel 没有 patcher，加上修复后指回底模、没有警告；服务端复现也确认消失。
* **第一次解码的实测峰值偏低**：一条带的解码 reserved 峰值不可能低于它自己的 arena，第一次只有 +1.25 GiB（arena 1.97）说明起点的 reserved 里有之前（采样）缓存下来的空闲块，被解码复用了；自检在测量窗口内只会让峰值变高。`_MemProbe` 改成先 `soft_empty_cache()` 再取起点；记录这次解码里有没有跑首次自检（`vae_engine._SELFTEST` 有没有变多），Info 注明「含首次自检」。

