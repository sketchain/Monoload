# GGUF / 量化格式上的运行时 LoRA（设计与进度）

> 这份是「GGUF / 量化 LoRA 支持」线程的工作文档，随做随写，换新会话从这里接上。合并到 `video` 时再把结论并进 HANDOFF / DESIGN。最后更新：2026-11-26。

## 0. 状态一览

| 项 | 分支 | 状态 |
|---|---|---|
| 调研 + 设计（本文件） | `gguf-quant-design`（基于 dev 339b754 的本地草稿） | 进行中 |
| 1 GGUF（ComfyUI-GGUF 的 Unet / CLIP 加载器） | `lora-gguf` | 设计完成，等 `video` 阶段 A 合并后实现 |
| 2 safetensors 各量化格式 | `lora-quant-formats` | 设计中 |
| 3 视频模型（Wan / HunyuanVideo / LTX，含 GGUF） | `lora-video-models` | 设计定了（§5 hook 方案），等阶段 A |
| 4 合成小模型测试 + 真机命令 | 各分支各自带 | 未开始 |

**基准**（协调者定）：ComfyUI **v0.46.3（cab5f4e）**，前端 1.59.6；GGUF 节点用 **kyuz0/ComfyUI-GGUF-H3 6d33de2**（city96 版加 H3 / Krea2 的 fork，city96 上游 1 月起没更新）。本文的源码引用都按这两份核过；版本再变要重核 §2。

**流程**（协调者定）：长期分支 `video`。阶段 A（适配新 ComfyUI）合进 `video` 之前只做调研和设计；之后从 `video` 开 feature 分支实现，`git merge --no-ff` 回 `video` 并 push。不碰 main / dev，不开 PR。真机由验证线程在用户 LXC（容器 `comfyui-latest`）上跑，这边只写命令。

## 1. 硬性要求（用户定，不能松）

* 模型加载后内存 + 显存任何时刻只能有一份权重：不接受整模型或一批层同时存在原权重和合并结果，不接受备份。允许：LoRA 文件常驻；临时合并时当前那一层短暂多一份。
* 量化层：在反量化出来的临时权重上直接合并，不量化回去（比原生精确、不逐位一致也行），不让常驻占用变大。GGUF 里没量化、又被 LoRA 改到的少数层可以留备份（本设计用不上：这些层也走运行时合并，没有备份）。
* 底模跨工作流常驻，LoRA 每个 prompt 结束释放（现有 release 机制）。默认融合 fp16 `addmm_`，留逐位一致开关（fused / exact）。
* 运行时任何东西都不许走 CPU。

## 2. 新版 ComfyUI / ComfyUI-GGUF 的相关事实

### 2.1 ComfyUI v0.46.3

* **量化张量**：`comfy.quant_ops.QuantizedTensor` 来自 `comfy_kitchen`（0.2.41）。格式（`QUANT_ALGOS`）：`float8_e4m3fn` / `float8_e5m2`（per-tensor scale，`TensorCoreFP8*Layout`）、`nvfp4`、`mxfp8`、`int8_tensorwise`（`TensorWiseINT8Layout`，只量化权重）、`convrot_w4a4`、`asym_w4a8_int8`、`w6a8_int8`。旧的 `scaled_fp8` 文件加载时被 `comfy.utils` 转成这套格式（`comfy/utils.py` ~1480）。层是 `mixed_precision_ops(...)` 的 `Linear`（另有 `MoEExperts`）。
* **有 weight_function 时**：`MixedPrecisionOps.Linear.forward` 的 `_use_quantized` 要求 `weight_function` 为空，所以挂了 LoRA 的层走 `CastBiasWeightContext` → `cast_bias_weight`：先 `.to(dtype)`，`QuantizedTensor` 再 `dequantize()`，然后依次调 weight function（`comfy/ops.py` ~385–483）。这条路和 0.31 一样，Monoload 现有的「在反量化临时权重上合并」直接适用。
* **原生打 LoRA**：`patch_weight_to_device` 对量化层用 `convert_func`（反量化）→ `calculate_weight` → `set_func`（`requantize_from_float`，随机舍入）写回，并备份原权重。
* **Dynamic VRAM（comfy-aimdo）**：不加 `--gpu-only` / `--highvram` / `--novram` / `--cpu`、ROCm ≥ 7.14 时默认开（`main.py` ~291），`CoreModelPatcher` 换成 `ModelPatcherDynamic`。Monoload 现在遇到它会报 `dynamic_vram`。CT 700 用 `--gpu-only`，不受影响；用户新 compose 的启动参数要验证线程确认（交给阶段 A）。
* **保存**：`model_state_dict_for_saving` 对没标 `comfy_patched_weights` 的层输出 `LazyCastingParam`（调 `patch_weight_to_device(return_weight=True)`），量化层输出 `LazyCastingQuantizedParam`（合并后量化回去，按量化格式存）。

### 2.2 ComfyUI-GGUF-H3（6d33de2，city96 6ea2651 的 fork）

* `GGMLTensor`（`torch.Tensor` 子类，`tensor_type` / `tensor_shape` / `patches`，H3 加了 `compute_dtype`）。H3 让 `.dtype` 对量化类型返回推断出的计算 dtype（fp16 / bf16 / fp32），**不是** uint8 存储 dtype——所以不能拿 `param.dtype` 判断是否量化，要看 `tensor_type`（`is_quantized()`）。F16 / F32 是「torch 兼容」类型，其余（Q4_K、Q8_0 …）是量化类型。BF16 张量加载时直接反量化成普通张量（≥2D 用 bf16，设备不支持 bf16 时 fp16；1D 用 fp32），**不是** GGMLTensor。
* **Q8_CR / int8_convrot 文件**：带 `comfy.gguf.quant.*` 元数据的张量加载成 ComfyUI 原生 `int8_tensorwise`（`TensorWiseINT8Layout`）的普通 `Parameter` + `comfy_quant`，模型用 `mixed_precision_ops` 建（`get_gguf_q8_ops`）。这类层就是 §4 的 `QuantizedTensor` 情况，不经 GGML 路径；同一文件里的普通 GGML 量化张量在加载时直接反量化。Q4_PT 已退役（加载即报错）。
* `GGMLOps`（继承 `comfy.ops.manual_cast`）：层的 `comfy_cast_weights = True`。权重或 bias 有一个是量化类型时，forward 走 `GGMLLayer.cast_bias_weight`（**方法**）→ `get_weight`：`dequantize_tensor(tensor, dtype, dequant_dtype)`（得到计算 dtype 的新张量）→ 有 `tensor.patches` 就 `comfy.lora.calculate_weight(patches, weight, key)`（中间 dtype fp32，或 `patch_dtype`）。**这条路不看 `weight_function`。** 两个都不是量化类型时走父类 → `comfy.ops.cast_bias_weight`（**函数**），会调 `weight_function`。
* **Dynamic VRAM 版**（`GGUFModelPatcherDynamic`、`*DynamicVRAM` 加载器、`quant_ops.GGMLLayout` 的 `QuantizedTensor`）只在 comfy-aimdo 开着时用；用户 `--gpu-only` 下用不上，Monoload 遇到 aimdo 照旧报 `dynamic_vram`，**不在本次范围**。
* `GGUFModelPatcher`（重写 `patch_weight_to_device`、`unpatch_model`、`load`、`clone`、`pin_weight_to_device`）：
  * 量化层：不改权重、不备份，把 patch 挂到 `GGMLTensor.patches`，每次 forward 在反量化结果上合并（patch 平时在 offload 设备，`patch_on_device` 才放计算设备）。**本来就是运行时合并**，但只有 fp32 中间量的 `calculate_weight`，没有融合 / 逐位一致开关，hook LoRA 不支持（原生 `patch_hook_weight_to_device` 会往量化张量里写）。
  * F16 / F32 层：原生方式合并进权重 + **备份**（违反要求）。
  * `load()` 总是 `force_patch_weights=True` 调父类（lowvram 层也走 `patch_weight_to_device`，因为量化层不认 `LowVramPatch`）。
  * `patch_weight_to_device` 签名没有 `return_weight`：保存时 `LazyCastingParam` 会 TypeError；而全量加载的层都标了 `comfy_patched_weights`，原生保存拿到的是 `ggml_save_to_state_dict` 的 meta 假数据。**原生 GGUF 本来就存不出可用的 checkpoint**（待 CPU 上确认）。
* Monoload 现在（dev）：`_active` 看到子类重写了 `patch_weight_to_device` 就整个走原生并打一条警告，所以 GGUF 模型的 F16 / F32 层被 LoRA 改到时有备份。
* 支持的架构（`IMG_ARCH_LIST`）：flux、sd1、sdxl、sd3、aura、hidream、cosmos、ltxv、hyvid、wan、lumina2、qwen_image，H3 另加 ideogram、krea2、minimax_h3；文本：t5、llama、qwen2vl、qwen3、qwen3vl、gemma3。
* `GGUFModelPatcher` 的五个重写（下面）H3 与 city96 相同。

## 3. 设计：GGUF（分支 `lora-gguf`）

**识别**：按结构，不看名字（city96 原版和 H3 fork 都要认）。`ModelPatcher` 子类的模块里有 `GGMLOps`，层类里有 `GGMLLayer`（带 `get_weight` / `cast_bias_weight` / `is_ggml_quantized`），patcher 类有 `patch_on_device` / `mmap_released`。在 `hotpatch.install()` 和每个 prompt 开始时（`release` 包装 `execute_async` 的入口）查一次 `sys.modules`，找到就装适配器（自定义节点的加载顺序不定，prompt 开始时一定都加载完了）。不认得的子类照旧走原生 + 警告。

**装什么**（只在适配器认得这个 GGUF 版本时；认不得照旧原生）：

1. `GGUFModelPatcher.patch_weight_to_device`：patcher 不受 Monoload 管（`_enabled` 为 False）时调 GGUF 原函数；否则和 `ModelPatcher` 的替换一样只挂 `MonoloadRuntimePatch`（量化层和 F16 / F32 层都一样），不碰 `GGMLTensor.patches`（顺手清掉这个 key 上残留的）、不备份。`return_weight=True`（保存）时自己算：反量化 → 与 exact 路径相同的合并 → 返回计算 dtype 的张量（见 4）。
2. `GGUFModelPatcher.load`：调 GGUF 原函数（保留它的 mmap 释放逻辑），但它传给父类的 `force_patch_weights=True` 在 Monoload 的 `_load` 里不报错（实例上打一个「GGUF 强制」标记，只在这次调用里有效）。效果：lowvram 层也挂运行时 patch，不挂 `LowVramPatch`。
3. `GGMLLayer.cast_bias_weight`（类上包一层）：调原函数（没有 `tensor.patches`，只反量化），再把这一层 `weight_function` / `bias_function` 里 **Monoload 的运行时 patch** 依次作用在结果上（别的 weight function，比如 `add_weight_wrapper` 的，原生 GGUF 在量化层上本来就不调，保持不调）。结果和参数共用存储时（量化权重配 F32 bias，`dequantize_tensor` 对 torch 兼容类型可能原样返回参数）先 clone 再合并，绝不原地改参数。
4. `MonoloadRuntimePatch` 认 GGML 量化参数（新增 `_is_ggml_quantized`）：
   * **默认（fused）**：和现在一样，在反量化出来的计算 dtype 临时权重上 `addmm_`（普通 LoRA / LoCon），其余类型 `calculate_weight(intermediate_dtype=计算 dtype)`。
   * 判断量化看 `tensor_type`，不看 `.dtype`（H3 的 `.dtype` 返回计算 dtype）。
   * **exact**：与原生 GGUF 量化层逐位一致：`calculate_weight(patches, 反量化临时权重, key, intermediate_dtype=fp32 或层的 patch_dtype)`，结果留在计算 dtype，**不**做 `stochastic_rounding` 到参数 dtype（GGML 参数 dtype 是 uint8 存储，没有意义；原生 GGUF 也不量化回去）。
   * hook LoRA：原生 GGUF 不支持；这里在 base 之后按 fp32 临时量 `calculate_weight`（`original_weights` 用反量化后的参数），回到计算 dtype。判定标准：与「这些层先反量化成计算 dtype 的普通参数」走同一路径逐位一致。
   * F16 / F32 层：就是普通参数（`GGMLTensor` 子类），走 `comfy.ops.cast_bias_weight` → 现有路径；exact 与原生（GGUF 的合并 + 备份）逐位一致。传进来的临时权重先 `as_subclass(torch.Tensor)`，避免 `GGMLTensor` 的 `shape` 重写和子类分发。
   * `_native_base`（同 key 后面有原生 `LowVramPatch` 就只加 hook）：GGML 量化层上 `LowVramPatch` 永远不会被调（3 只调 Monoload 的），所以这时视为没有。
5. **释放**：`release.py` 不用改（按 patcher 的 `patches`，GGUF 的 `unpatch_model` 先清 `GGMLTensor.patches` 再调父类，父类是 Monoload 的 `_unpatch_model`，摘运行时 patch）。
6. **保存**：CPU 上实测（v0.46.3 + H3）原生 GGUF 根本存不出来：量化层的 `state_dict` 是 meta 设备上的 uint8 假数据，没标 `comfy_patched_weights` 的层在 `LazyCastingParam` 里直接 `RuntimeError: only Tensors of floating point dtype can require gradients`。**决定（替用户定，可改）**：Monoload 管着的 GGUF 模型（含 GGML 量化层）进 `model_state_dict_for_saving` 时直接报 `MonoloadUnsupportedError`，写明「GGUF 模型不能存成 checkpoint，原生也不行」，不做半份文件、不静默丢 LoRA。Q8_CR 文件里的 int8 层是 `QuantizedTensor`，单独看可以存，但同一模型的其余部分仍是 GGML，所以整模型一样报错。
7. **LoRA Settings / Info**：设置在 `model_options` 里，GGUF clone 照样复制；Info 加一行「GGUF：N 个量化层 / M 个 F16·F32 层被 LoRA 改到」。

**内存**：常驻 = GGUF 量化权重 + LoRA 文件（CPU）+ LoRA 张量在计算设备上的一份缓存（和现在的 safetensors 路径相同；原生 GGUF 默认每次 forward 把 patch 从 CPU 搬一次，不常驻）。每层临时：反量化结果一份（原生 GGUF 也有），exact 再加 LoRA 乘积的 fp32 中间量。没有备份。

## 4. 设计：safetensors 各格式（分支 `lora-quant-formats`）

逐格式确认（CPU 合成层 + 真机）：

| 格式 | 层 / 张量 | 运行时 LoRA 走的路 | 精度判定 | 状态 |
|---|---|---|---|---|
| bf16 / fp16 / fp32 | `manual_cast` / `disable_weight_init` | 现有路径 | exact = 原生逐位一致 | 已有（`test_dtype_paths`） |
| 普通 fp8（无 scale，e4m3 / e5m2） | `manual_cast`，参数是 fp8 | 现有：exact 下按原生随机舍入回 fp8 后再转计算 dtype（与原生烘进权重逐位一致）；fused 不舍入 | exact = 原生 | 待在新版上复核 |
| scaled fp8（三种来源：文件里的 `comfy_quant` 张量 / 头部 `_quantization_metadata` / 旧 `scaled_fp8` 标记键 + `scale_weight`） | `convert_old_quants`（`comfy/utils.py` ~1474）统一转成 `comfy_quant` → `mixed_precision_ops`，`QuantizedTensor`（FP8 Layout） | 在反量化临时权重上合并，不量化回去 | = 「先反量化」参照 | 已有（0.31），三种来源都要用 CPU 测试锁住（协调者要求） |
| 混合精度 fp8（如 flux2_dev_fp8mixed） | 同上，部分层 bf16 | 同上 + 非量化层走普通路径 | 同上 | 待查文件结构 |
| int8（`int8_tensorwise`） | `QuantizedTensor`（`TensorWiseINT8Layout`） | 同 scaled fp8 | 同上 | LXC 上确认：noctQ…_int8 是 192 层 `int8_tensorwise`（`comfy_quant` 张量） |
| nvfp4 / mxfp8 / w4a8 / w6a8 / convrot | `QuantizedTensor` | 同上 | 同上 | CPU 上 8 种格式 `from_float` → `dequantize()` 都能跑，回到 bf16（convrot / w4a8 要求输入维是 256 的倍数），待接进测试 |
| `MoEExperts`（专家权重库） | 一个参数装 E 个专家 | LoRA key 对不上单个专家，先不支持（报错写明） | — | 待查有没有这种 LoRA |

## 5. 设计：视频模型（分支 `lora-video-models`）

LoRA key 映射是 ComfyUI 自己的（`comfy/lora.py` `model_lora_keys_unet` + `lora_convert`），对 GGUF 和 safetensors 一样，Monoload 拿到的已经是模型 key。Monoload 这边要保证的是：被改到的参数都能运行时合并。

已知缺口：不是 comfy.ops 层 weight / bias 的参数（Wan 的 `blocks.N.modulation`、`head.modulation`，LTX 的 `scale_shift_table` 之类）。`model_lora_keys_unet` 的通用规则把非 `.weight` 的 key 也映射（`diff` 形式），Monoload 现在对这种 key 报 `lora_non_comfy_ops_param`。**方案（已定，§7）**：对这种参数在所属模块上挂 forward pre-hook / always-call 的 forward hook，进模块时把参数换成合并后的临时张量、出来换回，合并结果只在这个模块的 forward 期间存在（这些参数都很小，比如 Wan 14B 的 modulation 6×5120），等于「当前那一层短暂多一份」，不备份。

## 6. 测试计划

* CPU（云端 venv，ComfyUI 目标版本 + ComfyUI-GGUF + `gguf`）：合成小模型——用 `gguf` 包写一个小 GGUF（Q8_0 / Q4_K / F16 / F32 混合）、用 `QuantizedTensor.from_float` 造各格式的 mixed ops 层；每种：fused 对独立实现逐位一致、exact 对参照逐位一致、无备份、参数字节不变、release 后无运行时 patch、保存行为。
* 真机命令：每项写 `docker exec comfyui-latest python /opt/ComfyUI/custom_nodes/monoload/tests/...`，报每步耗时、GTT 峰值、和原生对比的 max|Δ|、有没有多出常驻。

## 7. 已定的事（2026-11-26，协调者替用户定）

* GGUF + 保存：明确报错（§3.6）。
* 视频 LoRA 改到 modulation / scale_shift_table / `.diff` / `.diff_b` 这类非 comfy.ops 小参数：按 §5 的 hook 方案支持（模块 forward 期间临时换上合并好的参数，不留备份），目标所有视频模型都能用，不等 LXC 上的 LoRA key 扫描结果。
* Dynamic VRAM：新容器继续 `--gpu-only`；Monoload 在 Dynamic VRAM 下报清楚的错即可，暂不兼容。
* 分支：阶段 A 合进 `video` 后从 `video` 拉 feature 分支，`--no-ff` 合回 `video`；本文件一起带进 `video`。
* LXC 检查结果（2026-11-26，底座线程跑，原文在 LXC `/srv/comfy-latest/logs/gguf_checks.out`）：`--gpu-only` 在 compose 和运行中容器里都有。int8 = `int8_tensorwise`。fp8 那几份（flux2_dev_fp8mixed、wai_v17_fp8_test、mistral_3_small_flux2_fp8、oldt5 / umt5 fp8 scaled）文件里都没有 `comfy_quant` 张量，靠 scale 键，属于头部元数据或旧 `scaled_fp8` 标记那两种来源，具体哪种待第 3 条检查。LoRA 小参数扫描为空，是因为视频 LoRA 还没下完，下完让验证线程再查。

## 8. 进度日志

* 2026-11-26：读完 HANDOFF / DESIGN / hotpatch / release / lora_overrides；读 ComfyUI v0.46.3 的 quant_ops / ops / model_patcher / lora 和 ComfyUI-GGUF 全部源码；写本文件。云端 venv 在装（download.pytorch.org 被代理挡，改用 PyPI 的 torch）。
* 2026-11-26：基准定为 ComfyUI v0.46.3 + ComfyUI-GGUF-H3 6d33de2，§2.2 按 H3 重写（`.dtype` 返回计算 dtype、Q8_CR 走原生 int8、Dynamic VRAM 版不在范围）。venv 装好（torch 2.15.0，`/home/user/venv`）。CPU 实测：8 种 `QuantizedTensor` 格式都能反量化；原生 GGUF 前向在反量化结果上合并 LoRA 与「反量化 + BA」逐位一致；原生 GGUF 保存失败（§3.6）。实验脚本在会话临时目录，没进仓库。
* 2026-11-26：现有小测试在 v0.46.3 + CPU venv 上试跑：`test_dtype_paths` 198 项全过；`test_lora_save` 缺合成 checkpoint（要先跑 make_synthetic_checkpoint）；`test_release` 加载插件失败（`PromptServer.instance` 为空、`nodes_replacements` 导入失败），属于阶段 A 适配新版 ComfyUI 的范围，这里不改。
