"""Monoload's user-facing messages (logs, errors, the Info node's text).

Every message is looked up here by key and formatted with named fields:
    msg("vae.left_native", reason=..., note=...)
The language is English by default; MONOLOAD_LANG=zh (also zh-CN / zh_CN /
chinese) switches to Chinese. It is read at import; set_lang() for tests.
Internal "should never happen" diagnostics (StripeError and the like) stay
English: they are for bug reports, not for the user to act on. The ComfyUI
interface (node names, inputs, dropdowns, tooltips) is translated by the
frontend from locales/<lang>/nodeDefs.json and follows ComfyUI's language.

This module imports neither torch nor ComfyUI.
"""

import logging
import os

LANGS = ("en", "zh")

M = {
    # ---- settings / switches -------------------------------------------------------------------------------
    "settings.master_bad": (
        "[Monoload] MONOLOAD={raw!r} not understood (1 = on, 0 = native ComfyUI); Monoload stays on",
        "[Monoload] MONOLOAD={raw!r} 无法识别（1 = 开启，0 = 原版 ComfyUI）；Monoload 保持开启"),
    "settings.lang_bad": (
        "[Monoload] MONOLOAD_LANG={raw!r} not supported (en, zh); using English",
        "[Monoload] MONOLOAD_LANG={raw!r} 不支持（en、zh）；使用英文"),
    "settings.master_on": ("on", "开启"),
    "settings.master_off": (
        "off (MONOLOAD=0): native ComfyUI unless a Monoload node enables Monoload for its model / VAE",
        "关闭（MONOLOAD=0）：原版 ComfyUI，只有 Monoload 节点开启的模型 / VAE 走 Monoload"),

    # ---- plugin entry --------------------------------------------------------------------------------------
    "entry.disabled": (
        "[Monoload] MONOLOAD_DISABLE is set: nothing installed, ComfyUI stays native and the Monoload nodes pass their input through",
        "[Monoload] 设了 MONOLOAD_DISABLE：什么都没装，ComfyUI 保持原样，Monoload 节点原样输出输入"),
    "entry.master": ("[Monoload] master switch MONOLOAD: {state}", "[Monoload] 总开关 MONOLOAD：{state}"),
    "entry.lora": (
        "[Monoload] LoRA: runtime merge (no in-place LoRA, no weight backups), merge {merge}; {after}",
        "[Monoload] LoRA：运行时合并（不原地改权重、不留权重备份），合并方式 {merge}；{after}"),
    "entry.merge_exact": ("bit-exact (MONOLOAD_EXACT=1)", "逐位一致（MONOLOAD_EXACT=1）"),
    "entry.merge_fused": (
        "fused fp16 addmm / relaxed (MONOLOAD_EXACT=1 for bit-exact)",
        "融合 fp16 addmm / 放宽（设 MONOLOAD_EXACT=1 改为逐位一致）"),
    "entry.after_keep": ("kept between prompts (MONOLOAD_KEEP_LORA=1)", "prompt 之间保留（MONOLOAD_KEEP_LORA=1）"),
    "entry.after_release": ("released after every prompt (base models stay loaded)", "每个 prompt 结束后释放（底模保持加载）"),
    "entry.names_missing": (
        "[Monoload] LoRA names for the Monoload Info node not available (LoraLoader not found)",
        "[Monoload] Monoload 信息节点拿不到 LoRA 名字（找不到 LoraLoader）"),
    "entry.vae_native": (
        "[Monoload] VAE decode: native by default ({var}); a Monoload VAE Settings node with mode auto manages its VAE",
        "[Monoload] VAE 解码：默认原生（{var}）；Monoload VAE 设置节点的模式选「自动」时管理那个 VAE"),
    "entry.vae_managed": (
        "[Monoload] VAE decode managed: op-level chunking (conv row blocks, attention query blocks), workspace {workspace} "
        "(MONOLOAD_VAE_WORKSPACE), own memory estimate, OOM -> smaller blocks, never tiled; images only (4D / 5D T=1), "
        "set MONOLOAD_DISABLE_VAE=1 for native",
        "[Monoload] VAE 解码已接管：逐算子分块（卷积按行分块、注意力按 query 分块），工作区 {workspace}（MONOLOAD_VAE_WORKSPACE），"
        "自己估算内存，OOM 时缩小分块，绝不退回 tiled；只管图像（4D / T=1 的 5D），设 MONOLOAD_DISABLE_VAE=1 恢复原生"),
    "entry.vae_layer1": (
        "[Monoload] VAE layer 1 (stripe decoding) on for recognized decoders (Wan 2.1 / qwen_image_vae single frame; LDM Decoder of "
        "SD1.5 / SDXL / SD3 / Flux ae with whole-image GroupNorm statistics, scheme {scheme}; self-tested on first use): {policy}{rows}; "
        "other decoders use layer 2; set MONOLOAD_DISABLE_VAE_STRIPE=1 to use layer 2 everywhere",
        "[Monoload] VAE 第一层（条带解码）对认得的 decoder 开启（Wan 2.1 / qwen_image_vae 单帧；SD1.5 / SDXL / SD3 / Flux ae 的 LDM "
        "Decoder，GroupNorm 用整图统计量，方案 {scheme}；第一次使用时自检）：{policy}{rows}；其他 decoder 走第二层；"
        "设 MONOLOAD_DISABLE_VAE_STRIPE=1 全部走第二层"),
    "entry.scheme_forced": ("{scheme} (forced, MONOLOAD_VAE_GN_SCHEME)", "{scheme}（强制，MONOLOAD_VAE_GN_SCHEME）"),
    "entry.scheme_default": ("{scheme} (default)", "{scheme}（默认）"),
    "entry.policy_budget": (
        "peak budget {budget} (MONOLOAD_VAE_BUDGET): the fastest of layer 2 and the layer-1 configurations whose estimate fits it",
        "峰值预算 {budget}（MONOLOAD_VAE_BUDGET）：在第二层和第一层各配置中，选估算放得下的里面最快的"),
    "entry.policy_default": (
        "default stripe policy: the peak of {rows}-row stripes, tallest stripes within it (MONOLOAD_VAE_BUDGET to choose a budget)",
        "默认条带策略：以 {rows} 行条带的峰值为目标，取其中最高的条带（用 MONOLOAD_VAE_BUDGET 指定预算）"),
    "entry.rows_forced": (", stripe height forced to {rows} rows (MONOLOAD_VAE_STRIPE_ROWS)", "，条带高度强制为 {rows} 行（MONOLOAD_VAE_STRIPE_ROWS）"),
    "entry.vae_layer2_only": (
        "[Monoload] VAE layer 1 (stripe decoding) off (MONOLOAD_DISABLE_VAE_STRIPE): every managed decode uses layer 2",
        "[Monoload] VAE 第一层（条带解码）关闭（MONOLOAD_DISABLE_VAE_STRIPE）：所有接管的解码都走第二层"),

    # ---- LoRA: hotpatch / release / node -------------------------------------------------------------------
    "lora.unsupported_head": ("[Monoload] unsupported ({kind})", "[Monoload] 不支持（{kind}）"),
    "lora.subclass_native": (
        "[Monoload] {cls} overrides patch_weight_to_device; Monoload leaves it native",
        "[Monoload] {cls} 重写了 patch_weight_to_device；Monoload 不接管它，保持原生"),
    "lora.dynamic_vram": (
        "ComfyUI runs with DynamicVRAM (comfy-aimdo); Monoload does not support LoRA in that mode. Start with --gpu-only / --highvram / "
        "--disable-dynamic-vram, set the model's Monoload LoRA Settings mode to native, or set MONOLOAD=0 / MONOLOAD_DISABLE=1.",
        "ComfyUI 开启了 DynamicVRAM（comfy-aimdo），Monoload 不支持在这种模式下打 LoRA。请用 --gpu-only / --highvram / "
        "--disable-dynamic-vram 启动，或把这个模型的 Monoload LoRA 设置的模式设成「原生」，或设 MONOLOAD=0 / MONOLOAD_DISABLE=1。"),
    "lora.not_layer_param": ("the patch target is not a parameter of a layer", "patch 的目标不是某个层的参数"),
    "lora.non_comfy_ops": (
        "the module {module} whose parameter the LoRA / patch changes is not a comfy.ops layer and has no runtime merge path; "
        "Monoload does not fall back to changing the weights with a backup.",
        "被 LoRA/patch 修改的参数所在的模块 {module} 不是 comfy.ops 层，没有运行时合并路径；Monoload 不会退回到「改权重+备份」。"),
    "lora.shape_change": (
        "the patch would change the weight's shape from {old} to {new}; the runtime merge cannot do that",
        "patch 会把权重形状从 {old} 改成 {new}，运行时合并无法支持"),
    "lora.force_patch": (
        "a node asks for the LoRA / patches to be baked into the weights (force_patch_weights, e.g. saving or merging a model). Monoload "
        "only merges at run time; set this model's Monoload LoRA Settings mode to native, or MONOLOAD=0 / MONOLOAD_DISABLE=1.",
        "有节点要求把 LoRA/patch 直接烘焙进权重（force_patch_weights，常见于保存/合并模型）。Monoload 只做运行时临时合并，"
        "不改权重、不备份；请把这个模型的 Monoload LoRA 设置的模式设成「原生」，或设 MONOLOAD=0 / MONOLOAD_DISABLE=1。"),
    "lora.force_patch_unload": (
        "force_patch_weights is not supported for models Monoload drives",
        "Monoload 接管的模型不支持 force_patch_weights"),
    "lora.internal_backup": ("[Monoload] internal error: weight backups appeared {keys}", "[Monoload] 内部错误：出现了权重备份 {keys}"),
    "lora.internal_hook_write": (
        "[Monoload] internal error: hooks must not take the weight-writing path (key={key})",
        "[Monoload] 内部错误：hook 不应走写权重的路径（key={key}）"),
    "lora.internal_hook_cache": (
        "[Monoload] internal error: hooks must not take the cached-weights path (key={key})",
        "[Monoload] 内部错误：hook 不应走缓存权重的路径（key={key}）"),
    "release.done": (
        "[Monoload] released LoRA after prompt: {models} loaded model(s) back to base, {outputs} cached output(s), {objects} node LoRA "
        "cache(s), {synced} clean clone(s) re-synced, {repointed} orphaned loaded model(s) re-pointed ({seconds:.2f}s)",
        "[Monoload] prompt 结束后释放 LoRA：{models} 个已加载模型回到底模，{outputs} 个缓存输出，{objects} 个节点的 LoRA 缓存，"
        "{synced} 个干净 clone 重新同步，{repointed} 个失去 patcher 的已加载模型重新指向（{seconds:.2f}s）"),
    "release.failed": ("[Monoload] releasing LoRA after the prompt failed", "[Monoload] prompt 结束后释放 LoRA 失败"),
    "node.lora_disabled": (
        "[Monoload] Monoload LoRA Settings: MONOLOAD_DISABLE is set, so the node passes MODEL and CLIP through unchanged",
        "[Monoload] Monoload LoRA 设置：设了 MONOLOAD_DISABLE，节点原样输出 MODEL 和 CLIP"),
    "node.lora_settings": ("[Monoload] Monoload LoRA Settings: {note}", "[Monoload] Monoload LoRA 设置：{note}"),
    "node.bad_choice": ("{item} {value!r} is not one of {choices}", "{item} 的值 {value!r} 不是 {choices} 之一"),
    "lora.note": (
        "LoRA settings: mode {mode} ({mode_src}), merge {merge} ({merge_src}), after prompt {after} ({after_src})",
        "LoRA 设置：模式 {mode}（{mode_src}），合并方式 {merge}（{merge_src}），prompt 结束后 {after}（{after_src}）"),
    "lora.src_native": ("as mode native, {src}", "随模式「原生」，{src}"),

    # ---- VAE node / overrides ------------------------------------------------------------------------------
    "node.vae_disabled": (
        "[Monoload] Monoload VAE Settings: MONOLOAD_DISABLE is set, so the node passes the VAE through unchanged",
        "[Monoload] Monoload VAE 设置：设了 MONOLOAD_DISABLE，节点原样输出 VAE"),
    "node.budget_unused": (
        "[Monoload] Monoload VAE Settings: budget_gib {gib} not used, budget is {budget} (choose custom to use it)",
        "[Monoload] Monoload VAE 设置：预算 (GiB) {gib} 没有使用，预算选的是 {budget}（选「自定义」才使用）"),
    "node.custom_zero": ("budget custom needs budget_gib > 0 (got {gib})", "预算选「自定义」时预算 (GiB) 必须大于 0（现在是 {gib}）"),
    "node.negative": ("{item} {value} < 0", "{item} {value} 小于 0"),

    # ---- VAE: environment ----------------------------------------------------------------------------------
    "vae.env_workspace": (
        "[Monoload] MONOLOAD_VAE_WORKSPACE={raw!r} not understood (examples: 1G, 512M, 768); using {default}",
        "[Monoload] MONOLOAD_VAE_WORKSPACE={raw!r} 无法识别（例：1G、512M、768）；使用 {default}"),
    "vae.env_budget": (
        "[Monoload] MONOLOAD_VAE_BUDGET={raw!r} not understood (examples: 3G, 2560M); using the default stripe policy",
        "[Monoload] MONOLOAD_VAE_BUDGET={raw!r} 无法识别（例：3G、2560M）；使用默认条带策略"),
    "vae.env_rows": (
        "[Monoload] MONOLOAD_VAE_STRIPE_ROWS={raw!r} is not a positive integer; ignored",
        "[Monoload] MONOLOAD_VAE_STRIPE_ROWS={raw!r} 不是正整数；忽略"),
    "vae.env_scheme": (
        "[Monoload] MONOLOAD_VAE_GN_SCHEME={raw!r} is not one of {schemes}; using {default}",
        "[Monoload] MONOLOAD_VAE_GN_SCHEME={raw!r} 不是 {schemes} 之一；使用 {default}"),
    "vae.api_differs": (
        "[Monoload] VAE decode NOT managed: ComfyUI API differs from what Monoload was written for ({bad}); VAE stays native",
        "[Monoload] VAE 解码没有接管：ComfyUI 的接口与 Monoload 编写时不同（{bad}）；VAE 保持原生"),

    # ---- VAE: per decode -----------------------------------------------------------------------------------
    "vae.settings_note": (
        "settings: budget {budget} ({budget_src}), GroupNorm scheme {scheme} ({scheme_src}), stripe rows {rows} ({rows_src}), mode {mode} ({mode_src})",
        "设置：预算 {budget}（{budget_src}），GroupNorm 方案 {scheme}（{scheme_src}），条带高度 {rows}（{rows_src}），模式 {mode}（{mode_src}）"),
    "vae.unlimited": ("unlimited", "不限"),
    "vae.none": ("none", "无"),
    "vae.auto": ("auto", "自动"),
    "vae.by_budget": ("chosen by the budget", "按预算选"),
    "vae.layer2_only": ("layer 2 only", "只用第二层"),
    "vae.left_native": ("[Monoload] VAE decode left native: {reason}; {note}", "[Monoload] VAE 解码保持原生：{reason}；{note}"),
    "vae.native_node": ("mode native (Monoload VAE Settings node)", "模式原生（Monoload VAE 设置节点）"),
    "vae.native_global": ("mode native ({var})", "模式原生（{var}）"),
    "vae.l1_not_used": ("[Monoload] VAE layer 1 (stripes) not used for {model}: {why} -> layer 2",
                        "[Monoload] {model} 不走 VAE 第一层（条带）：{why} -> 第二层"),
    "vae.l1_disabled": (
        'layer 1 off (mode layer 2 only, from {src})',
        '第一层关闭（模式「只用第二层」，来源：{src}）'),
    "vae.selftest_ok": ("[Monoload] VAE layer 1 ({name}) self-test passed: {detail}", "[Monoload] VAE 第一层（{name}）自检通过：{detail}"),
    "vae.selftest_failed": (
        "[Monoload] !!!!!!!! VAE layer 1 ({name}) SELF-TEST FAILED: {detail} !!!!!!!! layer 1 is disabled for this decoder structure in "
        "this process; decoding with layer 2 (op-level chunking) instead. Please report this.",
        "[Monoload] !!!!!!!! VAE 第一层（{name}）自检失败：{detail} !!!!!!!! 本进程里这种 decoder 结构不再走第一层，"
        "改走第二层（逐算子分块）。请报告这个问题。"),
    "vae.selftest_failed_short": ("layer-1 self-test failed", "第一层自检未通过"),
    "vae.policy_forced_rows": (
        'forced {rows} rows (from {src}){over}',
        '强制 {rows} 行（来源：{src}）{over}'),
    "vae.over_budget_note": (
        '; estimate above the budget',
        '；估算超过预算'),
    "vae.policy_default": ("default: peak of {rows}-row stripes", "默认：{rows} 行条带的峰值"),
    "vae.err_l1_budget": (
        '[Monoload] VAE layer 1 (stripe decoding) does not fit the peak budget {budget} (from {src}): latent {shape} needs at least about {need} ({rows}-row stripes, prefix {prefix}, stripes {stripes}). {advice}',
        '[Monoload] VAE 第一层（条带解码）在峰值预算 {budget}（来源：{src}）内放不下：latent {shape} 最少也需要约 {need}（{rows} 行的条带，前缀 {prefix}、条带 {stripes}）。{advice}'),
    "vae.cand_layer2": ("layer 2 {est}", "第二层 {est}"),
    "vae.cand_layer1": ("layer 1{scheme} {rows} rows (workspace {ws}) {est}{secs}", "第一层{scheme} {rows} 行（工作区 {ws}）{est}{secs}"),
    "vae.cand_scheme": (" scheme {scheme}", " 方案 {scheme}"),
    "vae.cand_secs": (", ~{secs:.1f} s", "，约 {secs:.1f} s"),
    "vae.cand_over": (" (over)", "（超出）"),
    "vae.budget_head": (
        'budget {budget} (from {src})',
        '预算 {budget}（来源：{src}）'),
    "vae.policy_join": ("{head}: {policy}{over}", "{head}：{policy}{over}"),
    "vae.est_above": ("; estimate above the budget", "；估算超过预算"),
    "vae.why_layer2": ("{head} -> layer 2 (estimate {est}): {why}", "{head} -> 第二层（估算 {est}）：{why}"),
    "vae.l2_forced_policy": (
        'layer 2 forced (mode layer 2 only, from {src})',
        '强制第二层（模式「只用第二层」，来源：{src}）'),
    "vae.l2_forced_why": (
        'forced by mode layer 2 only (from {src})',
        '由模式「只用第二层」强制（来源：{src}）'),
    "vae.forced_rows": (
        '{rows} rows (from {src})',
        '{rows} 行（来源：{src}）'),
    "vae.forced_scheme": (
        'scheme {scheme} (from {src})',
        '方案 {scheme}（来源：{src}）'),
    "vae.l2_fits_policy": ("layer 2 fits, the fastest candidate", "第二层放得下，最快的候选"),
    "vae.l2_fits_why": ("fits, and layer 2 is the fastest (every conv once, no recompute){l1}",
                        "放得下，而且第二层最快（每个卷积只算一次，不重算）{l1}"),
    "vae.l1_unavailable_note": ("; layer 1 not available: {why}", "；第一层不可用：{why}"),
    "vae.forced_pre": ("forced {what}; ", "强制 {what}；"),
    "vae.reason_over": ("no variant fits the budget at that height, the smallest estimate is used",
                        "这个高度下没有放得下预算的方案，用估算最小的"),
    "vae.reason_fastest": ("the fastest predicted that fits", "放得下的里面预计最快的"),
    "vae.reason_fits": ("the candidate that fits", "放得下的候选"),
    "vae.reason_only": ("the only candidate", "唯一的候选"),
    "vae.policy_forced_over": ("forced rows, estimate above the budget", "强制条带高度，估算超过预算"),
    "vae.policy_fastest": ("fastest within it", "预算内最快"),
    "vae.why_layer1": ("{head} -> {cand}: {reason}; others: {others}", "{head} -> {cand}：{reason}；其他：{others}"),
    "vae.others_none": ("none", "无"),
    "vae.l2_after_selftest": ("layer 2 (layer-1 self-test failed)", "第二层（第一层自检未通过）"),
    "vae.need_layer2": ("layer 2 needs about {est}", "第二层需要约 {est}"),
    "vae.need_layer1": ("layer 1{scheme} with {rows}-row stripes needs about {est} (prefix {prefix}, stripes {stripes}){failed}",
                        "第一层{scheme}用 {rows} 行的条带需要约 {est}（前缀 {prefix}、条带 {stripes}）{failed}"),
    "vae.need_scheme": (" (GroupNorm scheme {scheme})", "（GroupNorm 方案 {scheme}）"),
    "vae.need_failed": (", but its self-test failed", "，但自检未通过"),
    "vae.need_l1_unavailable": ("layer 1 not available ({why})", "第一层不可用（{why}）"),
    "vae.need_sep": ("; ", "；"),
    "vae.err_budget": (
        '[Monoload] VAE decode does not fit the peak budget {budget} (from {src}; latent {shape}): {needs}. {advice}',
        '[Monoload] VAE 解码在峰值预算 {budget}（来源：{src}）内放不下（latent {shape}）：{needs}。{advice}'),
    "vae.src_node": ("the Monoload VAE Settings node", "Monoload VAE 设置节点"),
    "vae.src_env": ("environment variable {var}", "环境变量 {var}"),
    "vae.advice_node": (
        "Raise the budget on the Monoload VAE Settings node, or set its budget to default (follow global) or unlimited.",
        "请在 Monoload VAE 设置节点上调大预算，或把预算改成「跟随全局」或「不限」。"),
    "vae.advice_env": (
        "Raise MONOLOAD_VAE_BUDGET or remove it (the default policy), or give this VAE its own budget with a Monoload VAE Settings node.",
        "请调大 MONOLOAD_VAE_BUDGET 或去掉它（用默认策略），或用 Monoload VAE 设置节点给这个 VAE 单独设预算。"),
    "vae.advice_l2_node": (" Or set the node's mode to layer 2 only.", "或把节点的模式设成「只用第二层」。"),
    "vae.advice_l2_env": (" Or set MONOLOAD_DISABLE_VAE_STRIPE=1 to use layer 2.", "或设 MONOLOAD_DISABLE_VAE_STRIPE=1 改走第二层。"),
    "vae.budget_log": ("[Monoload] VAE {why}{note}", "[Monoload] VAE {why}{note}"),
    "vae.err_oom_l1": (
        "[Monoload] VAE decode out of memory: layer 1 (stripe decoding) still runs out of memory with {rows}-row stripes and workspace "
        "{ws} ({retries} retries). Monoload never falls back to the approximate tiled decode, nor to layer 2 (its peak is higher). Free "
        "other models (/free), lower the resolution, or set this VAE's mode to native (MONOLOAD_DISABLE_VAE=1 for all). Latent {shape}, "
        "estimate {est}.",
        "[Monoload] VAE 解码显存不足：第一层（条带解码）的条带已缩到 {rows} 行、工作区 {ws}（共重试 {retries} 次）仍然 OOM。"
        "Monoload 不会退回到 tiled 近似解码，也不会退回第二层（第二层峰值更高）。可以先释放其他模型（/free）、降低分辨率，"
        "或把这个 VAE 的模式设成原生（全部原生：MONOLOAD_DISABLE_VAE=1）。latent {shape}，估算需要 {est}。"),
    "vae.retry_l1": (
        "[Monoload] VAE decode ran out of memory; retrying layer 1 with {rows}-row stripes, workspace {ws} (retry {retries})",
        "[Monoload] VAE 解码显存不足；第一层改用 {rows} 行的条带、工作区 {ws} 重试（第 {retries} 次）"),
    "vae.log_layer1": (
        "[Monoload] VAE decode {shape} -> layer 1 ({name}): {plan}; {policy}, workspace {ws}{retries}; arena {arena}, memory estimate {est} "
        "(native {native}), {secs:.2f}s; {note}",
        "[Monoload] VAE 解码 {shape} -> 第一层（{name}）：{plan}；{policy}，工作区 {ws}{retries}；arena {arena}，内存估算 {est}"
        "（原生 {native}），{secs:.2f}s；{note}"),
    "vae.retries": (", {n} OOM retries", "，OOM 重试 {n} 次"),
    "vae.probe_failed": (
        "[Monoload] VAE shape probe failed ({err}); memory estimate falls back to the widest conv at full resolution",
        "[Monoload] VAE 形状探测失败（{err}）；内存估算退回按全分辨率下最宽的卷积计算"),
    "vae.err_oom_l2": (
        "[Monoload] VAE decode out of memory: still out of memory with the smallest workspace {ws} ({retries} retries). Monoload never "
        "falls back to the approximate tiled decode (decode_tiled_). Free other models (/free), lower the resolution, or set this VAE's "
        "mode to native (MONOLOAD_DISABLE_VAE=1 for all). Latent {shape}, estimate {est}.",
        "[Monoload] VAE 解码显存不足：工作区已缩到下限 {ws}（共重试 {retries} 次）仍然 OOM。Monoload 不会退回到 tiled 近似解码"
        "（decode_tiled_）。可以先释放其他模型（/free）、降低分辨率，或把这个 VAE 的模式设成原生（全部原生：MONOLOAD_DISABLE_VAE=1）。"
        "latent {shape}，估算需要 {est}。"),
    "vae.retry_l2": ("[Monoload] VAE decode ran out of memory; retrying with workspace {ws} (retry {retries})",
                     "[Monoload] VAE 解码显存不足；改用工作区 {ws} 重试（第 {retries} 次）"),
    "vae.log_layer2": (
        "[Monoload] VAE decode {shape} -> layer 2, op-level chunking (workspace {ws}{retries}): {chunked} of {calls} conv call(s) in row "
        "blocks{attn}; {policy}memory estimate {est} (native {native}), {secs:.2f}s; {note}",
        "[Monoload] VAE 解码 {shape} -> 第二层，逐算子分块（工作区 {ws}{retries}）：{calls} 次卷积调用中 {chunked} 次按行分块{attn}；"
        "{policy}内存估算 {est}（原生 {native}），{secs:.2f}s；{note}"),
    "vae.attn_blocks": (", attention {calls} call(s) in query blocks of {sizes} tokens", "，注意力 {calls} 次按 query 分块（每块 {sizes} 个 token）"),
    "vae.attn_native": (", attention left native: {what}", "，注意力保持原生：{what}"),
    "vae.plan": ("{n} stripes of {rows} rows (core), recompute {rec:.2f}x, checkpoint {ckpt}", "{n} 条 {rows} 行的条带（核心），重算 {rec:.2f} 倍，存档 {ckpt}"),
    "vae.plan_passes": ("; {n} statistics passes, saves {saves}", "；{n} 遍统计，中间结果 {saves}"),

    # ---- Info node -----------------------------------------------------------------------------------------
    "info.version": ("Monoload {version} ({where})", "Monoload {version}（{where}）"),
    "info.commit": ("commit {commit}", "commit {commit}"),
    "info.commit_unknown": ("commit unknown", "commit 未知"),
    "info.branch": (", branch {branch}", "，分支 {branch}"),
    "info.disabled": (
        "MONOLOAD_DISABLE=1: nothing installed, ComfyUI is native and the Monoload nodes pass their inputs through",
        "MONOLOAD_DISABLE=1：什么都没装，ComfyUI 是原版，Monoload 节点原样输出输入"),
    "info.master": ("master switch MONOLOAD: {state} [{src}]", "总开关 MONOLOAD：{state} [{src}]"),
    "info.master_on": ("on", "开启"),
    "info.master_off": ("off: native ComfyUI unless a Monoload node enables it", "关闭：原版 ComfyUI，除非 Monoload 节点开启"),
    "info.installed": ("installed: runtime LoRA merge {a}, per-prompt LoRA release {b}, VAE decode wrapper {c}",
                       "已安装：LoRA 运行时合并 {a}，每个 prompt 后释放 LoRA {b}，VAE 解码接管 {c}"),
    "info.yes": ("yes", "是"),
    "info.no": ("no", "否"),
    "info.globals": ("global defaults (a Monoload node's explicit choice overrides them for its model / VAE):",
                     "全局默认值（Monoload 节点上明确选的值对它那个模型 / VAE 优先）："),
    "info.src_env": ("env {name}={value}", "环境变量 {name}={value}"),
    "info.src_builtin": ("built-in", "内置默认"),
    "info.src_runtime": ("set at runtime", "运行时设置"),
    "info.src_node": ("node", "节点"),
    "info.src_envvar": ("env {var}", "环境变量 {var}"),
    "info.row_lora_mode": ("LoRA mode", "LoRA 模式"),
    "info.row_lora_merge": ("LoRA merge", "LoRA 合并方式"),
    "info.row_lora_after": ("LoRA after prompt", "LoRA prompt 结束后"),
    "info.row_vae_mode": ("VAE mode", "VAE 模式"),
    "info.row_vae_budget": ("VAE budget", "VAE 预算"),
    "info.row_vae_scheme": ("VAE GroupNorm scheme", "VAE GroupNorm 方案"),
    "info.row_vae_rows": ("VAE stripe rows", "VAE 条带高度"),
    "info.row_vae_ws": ("VAE workspace", "VAE 工作区"),
    "info.enable_monoload": ("enable (Monoload)", "启用（Monoload）"),
    "info.native": ("native", "原生"),
    "info.exact": ("exact (bit-identical)", "逐位一致"),
    "info.fused": ("fused", "融合"),
    "info.keep": ("keep", "保留"),
    "info.release": ("release", "释放"),
    "info.budget_none": ("none (default policy)", "无（默认策略）"),
    "info.forced": (" (forced)", "（强制）"),
    "info.not_decoded": (
        "last decode: not decoded yet (connect images from its VAE Decode to run this node after the decode)",
        "上一次解码：尚未解码（把它的 VAE Decode 的图像接到 images，让这个节点在解码之后运行）"),
    "info.decode_head": ("last decode ({ago:.0f} s ago): ", "上一次解码（{ago:.0f} 秒前）："),
    "info.decode_native": ("native ComfyUI decode ({reason}){t}", "原版 ComfyUI 解码（{reason}）{t}"),
    "info.decode_error": ("error: no decode fits the budget {budget}", "错误：没有放得下预算 {budget} 的解码方式"),
    "info.decode_l1": ("layer 1 ({adapter}), {n} stripes of {rows} rows", "第一层（{adapter}），{n} 条 {rows} 行的条带"),
    "info.decode_l2": ("layer 2 (op-level chunking)", "第二层（逐算子分块）"),
    "info.decode_line": ("{what}, workspace {ws}, estimate {est}, measured peak {measured}{t}, OOM retries {retries}",
                         "{what}，工作区 {ws}，估算 {est}，实测峰值 {measured}{t}，OOM 重试 {retries} 次"),
    "info.selftest_note": (" (includes the first-use self-test)", "（含首次自检）"),
    "info.secs": (", {secs:.2f} s", "，{secs:.2f} s"),
    "info.no_gpu": ("n/a (no GPU)", "无（没有 GPU）"),
    "info.vae_head": ("VAE ({model}{copy}):", "VAE（{model}{copy}）："),
    "info.vae_copy": (", a Monoload VAE Settings copy", "，Monoload VAE 设置做的副本"),
    "info.vae_not_installed": ("  the managed decode is not installed: ComfyUI's own decode", "  解码接管没有安装：ComfyUI 自己的解码"),
    "info.vae_settings": ("  settings: mode {mode} [{mode_src}], budget {budget} [{budget_src}], GroupNorm scheme {scheme} [{scheme_src}], "
                          "stripe rows {rows} [{rows_src}]",
                          "  设置：模式 {mode} [{mode_src}]，预算 {budget} [{budget_src}]，GroupNorm 方案 {scheme} [{scheme_src}]，"
                          "条带高度 {rows} [{rows_src}]"),
    "info.scheme_hint": ("  GroupNorm scheme {scheme}: {hint}", "  GroupNorm 方案 {scheme}：{hint}"),
    "scheme.A": ("keeps nothing, recomputes from the H/8 checkpoint - lowest memory, slowest (SDXL 4K ~1.1 GiB / 75 s)",
                 "什么都不存，从 H/8 存档重算——内存最低、最慢（SDXL 4K 约 1.1 GiB / 75 s）"),
    "scheme.D": ("keeps the H/4 level output - between A and B (SDXL 4K ~1.5 GiB / 58 s)",
                 "存 H/4 级的输出——介于 A 和 B 之间（SDXL 4K 约 1.5 GiB / 58 s）"),
    "scheme.B": ("keeps the H/4 and H/2 level outputs - the built-in default (SDXL 4K ~2.2 GiB / 42 s)",
                 "存 H/4 和 H/2 级的输出——内置默认（SDXL 4K 约 2.2 GiB / 42 s）"),
    "scheme.C": ("also keeps every full-resolution block's input - most memory, fastest (SDXL 4K ~4.7 GiB / 36 s)",
                 "再存全分辨率每个块的输入——内存最高、最快（SDXL 4K 约 4.7 GiB / 36 s）"),
    "info.unused_native": (" (not used in native mode)", "（原生模式下不使用）"),
    "info.unused_layer2": (" (not used: layer 2 only)", "（只用第二层时不使用）"),
    "info.model_head": ("MODEL ({model}):", "MODEL（{model}）："),
    "info.lora_list": ("  LoRA: {items}", "  LoRA：{items}"),
    "info.lora_other": ("  LoRA: none from the LoRA loader nodes (other patches present)", "  LoRA：没有来自 LoRA 加载节点的（有其他 patch）"),
    "info.lora_none": ("  LoRA: none", "  LoRA：无"),
    "info.patched": ("  patched weights: {n}{hooks}", "  被改动的权重：{n}{hooks}"),
    "info.hooks": (", hook-LoRA groups: {n}", "，hook LoRA 组：{n}"),
    "info.settings": ("  {note}", "  {note}"),
    "info.state": ("  now: {state}", "  当前：{state}"),
    "info.state_not_loaded": ("not loaded", "没有加载"),
    "info.state_runtime": ("loaded; Monoload runtime merge on {n} weights (no weight backups)",
                           "已加载；Monoload 运行时合并 {n} 个权重（没有权重备份）"),
    "info.state_baked": ("loaded; native: LoRA baked into the weights ({n} backups)", "已加载；原生：LoRA 合并进权重（{n} 份备份）"),
    "info.state_clean": ("loaded; no LoRA in effect", "已加载；没有生效的 LoRA"),
    "info.state_other": ("the shared model is loaded for another clone", "共享的模型正以另一个 clone 加载着"),

    # ---- setting values and sources inside sentences (English: the stored value itself) ----------------------
    "v.follow": ("default", "跟随全局"),
    "v.enable": ("enable", "启用"),
    "v.native": ("native", "原生"),
    "v.fused": ("fused", "融合"),
    "v.exact": ("exact", "逐位一致"),
    "v.release": ("release", "释放"),
    "v.keep": ("keep", "保留"),
    "v.auto": ("auto", "自动"),
    "src.node": ("node", "节点"),
    "src.env": ("env", "环境变量"),
    "src.default": ("default", "默认"),

    # ---- misc ----------------------------------------------------------------------------------------------
    "env.no_comfyui": ("ComfyUI directory not found; set COMFYUI_PATH", "找不到 ComfyUI 目录；请设置环境变量 COMFYUI_PATH"),
}


def _lang_from_env():
    raw = os.environ.get("MONOLOAD_LANG", "").strip().lower().replace("_", "-")
    if raw in ("", "en", "en-us", "en-gb", "english"):
        return "en", None
    if raw in ("zh", "zh-cn", "zh-hans", "chinese", "cn"):
        return "zh", None
    return "en", raw


_LANG = [_lang_from_env()[0]]
_BAD = _lang_from_env()[1]


def lang():
    return _LANG[0]


def set_lang(code):
    """"en" / "zh" (tests); MONOLOAD_LANG at import."""
    if code not in LANGS:
        raise ValueError(code)
    _LANG[0] = code


def msg(key, **kw):
    """The message `key` in the current language, formatted with `kw`."""
    text = M[key][LANGS.index(_LANG[0])]
    return text.format(**kw) if kw else text


def label(value):
    """A setting value or a source ("node" / "env" / "default") inside a sentence."""
    for key in ("v." + str(value), "src." + str(value)):
        if key in M:
            return msg(key)
    return str(value)


def warn_bad_lang():
    """Called once by the plugin entry: an unknown MONOLOAD_LANG is reported."""
    if _BAD:
        logging.warning(msg("settings.lang_bad", raw=_BAD))
