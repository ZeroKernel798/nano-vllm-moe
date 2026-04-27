---
name: moe-fp8-deepep-roadmap
overview: 先优化现有 Triton MoE 与 FP8 路径，再通过 LinearKernel + MoE expert-kernel 抽象为 DeepEP 与其他量化方案铺路，最后接入 DeepEP 与 FP8 MoE / INT8 MoE / W4A8 等量化。
todos:
  - id: stage0
    content: "Stage 0: 清理 qwen2_moe.py 历史注释、标 Eager debug-only、补 CUDA graph EP guard、用本计划覆盖 Todo.md"
    status: in_progress
  - id: stage1-perf
    content: "Stage 1.1–1.3: dispatch 去 host-sync + moe_align Triton 化 + combine 去 fp32 atomic_add"
    status: pending
  - id: stage1-struct
    content: "Stage 1.4–1.8: BaseSparseMoeBlock、DispatchState dataclass、backend hoist、MoE 测试"
    status: pending
  - id: stage2-fp8-perf
    content: "Stage 2.1–2.4: W8A16 cuBLASLt + W8A8 epilogue 融合 + qweight_nk padding-aware + w8a8_dynamic"
    status: pending
  - id: stage2-fp8-rest
    content: "Stage 2.5–2.8: pre-Hopper dequant 缓存 + KV cache FP8 + export per-channel 统一 + fp8 单测"
    status: pending
  - id: stage3-abstract
    content: "Stage 3: LinearKernel 启用 + MoEExpertKernel 抽象 + buffer naming 收口 + kernel 测试"
    status: pending
  - id: stage4-fp8-moe
    content: "Stage 4: FP8 MoE 端到端（export、stacked scale、fp8 group-gemm、ppl 回归）"
    status: pending
  - id: stage5-deepep
    content: "Stage 5: DeepEPBackend (normal + low_latency)、compute 适配、CUDA graph、默认选择、benchmark"
    status: pending
  - id: stage6-newquant
    content: "Stage 6: INT8 MoE / W4A8 / GPTQ INT4 / KV cache INT8 / SmoothQuant 改进"
    status: pending
isProject: false
---

# nano-vllm-moe Roadmap (replaces Todo.md)

> 目标：先优化现有 MoE → 再优化现有 FP8 → 同步建立 LinearKernel / MoE expert-kernel 抽象 → 落地 FP8 MoE 端到端 → DeepEP 接入 → 其他量化方案。  
> 决策：FP8 优化在 Hopper 与 Ampere 上等优先；AWQ INT4 保留作为 reference 实现并纳入新抽象。

## Stage 0 — 基线对齐 / 死代码清理（约半天）

- 删除 [nanovllm/models/qwen2_moe.py](nanovllm/models/qwen2_moe.py) 前 ~600 行注释代码（lines 1–611）。
- 把 `Qwen{2,3}MoeEagerSparseMoeBlock` 标记为 debug-only 并加 docstring；后续 Stage 1 会下沉到共享基类。
- 在 [nanovllm/engine/model_runner.py](nanovllm/engine/model_runner.py) `capture_cudagraph()` 入口加：`if self.ep_size > 1 or backend_uses_dynamic_alltoall: skip & log`，避免动态路由 + cuda graph 出错。
- 替换本文件 [Todo.md](Todo.md) 为本计划。

## Stage 1 — MoE 性能优化（先做完）

> 全部围绕 [nanovllm/executor/moe/backends/triton.py](nanovllm/executor/moe/backends/triton.py) 与 [nanovllm/utils/moe.py](nanovllm/utils/moe.py)。先打开 perf 头部，结构改造放 Stage 3。

- **1.1 dispatch 去 host-sync**：`recv_counts.tolist()` / `.sum().item()` 用 device-resident `cumsum` 替代；4 次 `dist.all_to_all_single` 合并为 1 次（把 `x` / `local_ids` / `weights` 在 hidden 维拼成一个张量发送，落地后 split）。
- **1.2 `moe_align_block_size` 替换为 Triton kernel**：把现在 [utils/moe.py](nanovllm/utils/moe.py) 中的 `argsort + bincount + cumsum + repeat_interleave` 整体写成一个 device kernel，输出 `sorted_token_ids / sorted_weight_idx / expert_ids / num_blocks`。decode 小 batch 收益最大。
- **1.3 combine kernel 去掉 fp32 atomic_add**：将 `fused_moe_w2_combine_kernel` 改为写到 per-task `[num_recv, H]` 中间 buffer（无重叠写入），外层用 `index_add_` 还原 token 顺序；阻断 fp32 中转，bf16 throughput 直接受益。
- **1.4 抽出 `BaseSparseMoeBlock`**：合并 [qwen2_moe.py](nanovllm/models/qwen2_moe.py) `Qwen2MoeSparseMoeBlock` 与 [qwen3_moe.py](nanovllm/models/qwen3_moe.py) `Qwen3MoeSparseMoeBlock`；提供 `route()` / `shared_expert(x)` / `topk_postprocess(weights)` 三个 hook，差异（topk 归一化、shared expert）作为子类覆写。Eager 版本同时合并。
- **1.5 dispatch_state 类型化**：用 `@dataclass DispatchState` 替换 `dict[str, Any]`，predefine 字段 `recv_x / recv_local_ids / recv_weights / permute_indices / s_list / r_list / num_recv / handle: Any | None`，为 DeepEP 携带 `combine_handle` 留位。
- **1.6 `TP all_reduce` 位置统一**：所有路径都通过 `backend.combine(reduce_results=...)` 出口，移除 [qwen2_moe.py](nanovllm/models/qwen2_moe.py) line 904–905 的 layer 内 all_reduce。
- **1.7 backend 提到 `Qwen{2,3}MoeModel`**：`self.backend = TritonMoEBackend(...)`，layer 共享同一 instance（无状态，节省每层构造）。
- **1.8 测试**：`tests/moe/test_dispatch_roundtrip.py` 验证 dispatch+combine 在 ep=1/ep=2 下与 reference Eager 实现等价；`tests/moe/test_align_block_kernel.py` 对 1.2 新 kernel 做 numerical 对齐。

## Stage 2 — FP8 当前路径优化（dense linear）

> 全部围绕 [nanovllm/layers/fp8/](nanovllm/layers/fp8/)，Hopper / Ampere 等优先。

- **2.1 W8A16 Hopper 走 cuBLASLt**：在 [fp8/parallel.py](nanovllm/layers/fp8/parallel.py) `_forward_w8a16` 的 `cc[0] >= 9` 分支改为复用 `_scaled_mm`（`s_x = 1`、bf16 激活先量化到 fp8 再走，或用 `tl.dot(fp8, bf16)` Hopper 路径）；和现有 Triton autotune 跑 A/B，按硬件保留更快路径。
- **2.2 W8A8 epilogue 融合**：`launch_w8a8_static_gemm` 中 `out[:, :N].float() * w_s` 用 Triton epilogue kernel 融合 `weight_scale * cast bf16`；消除 fp32 中间张量。
- **2.3 `qweight_nk` 缓存 padding-aware**：load 阶段就生成 padded `[N_pad, K_pad]` 缓存，让 W8A8 静态路径在 K/N 不是 16 的倍数时也能命中缓存（[fp8/kernels.py](nanovllm/layers/fp8/kernels.py) line 263–273）。
- **2.4 引入 `fp8_scheme="w8a8_dynamic"`**：per-token `input_scale` 由 Triton kernel 在 quantize 时计算（去掉离线 calibration 依赖）；DeepSeek-V2/V3 兼容。`_scaled_mm` 的 `scale_a` 改为 `[M, 1]`。
- **2.5 Pre-Hopper fallback 缓存 dequant 后 weight**：[fp8/parallel.py](nanovllm/layers/fp8/parallel.py) line 99–101 把 dequant 后的 bf16 weight 在 first forward 缓存为 buffer，避免每次 forward 重 alloc。
- **2.6 KV cache FP8 打通**：把 [nanovllm/utils/kv_cache.py](nanovllm/utils/kv_cache.py) 与 [nanovllm/layers/kv_cache_kernels.py](nanovllm/layers/kv_cache_kernels.py) 的 `kv_cache_dtype="fp8_e4m3"` 写成可用：scale 存到 KV cache buffer 旁边，写入 / 读取时 quant/dequant，attention kernel 内联。
- **2.7 export 脚本统一 per-channel `weight_scale`**：[quant_w8a16_fp8.py](nanovllm/quant/quant_w8a16_fp8.py) 由 scalar 升为 `[N]`；移除 [fp8/parallel.py](nanovllm/layers/fp8/parallel.py) `_load_weight_scale` 的 scalar broadcast 兼容分支（loader 测试同步更新）。
- **2.8 测试**：`tests/quant/test_fp8_kernel.py` 用随机 `(M, N, K)` 比对 `F.linear(bf16) vs launch_w8a16_gemm` 与 `vs launch_w8a8_static_gemm`，容忍域 1e-2。

## Stage 3 — Linear / MoE 量化抽象（DeepEP 与新 quant 的基础）

- **3.1 启用 `LinearKernel`**：实现 [nanovllm/layers/linear_kernel.py](nanovllm/layers/linear_kernel.py) 中 `FP8Kernel` / `W8A8Kernel` / `AWQKernel` 的 `forward()`，把 [fp8/parallel.py](nanovllm/layers/fp8/parallel.py) / [smooth_quant_linear.py](nanovllm/layers/smooth_quant_linear.py) / [quant_linear.py](nanovllm/layers/quant_linear.py) 的 forward 改为委托。Loader 仍然写到各自 buffer，kernel 只做 compute。
- **3.2 引入 `MoEExpertKernel` 接口**：与 `LinearKernel` 平行，定义 `forward(recv_x, w13_buffers, w2_buffers, ...) -> local_out`；现有 [executor/moe/backends/triton.py](nanovllm/executor/moe/backends/triton.py) `compute()` 拆成 `TritonDenseExpertKernel` 与（Stage 4 加）`TritonFP8ExpertKernel`，`backend.compute()` 只负责 schedule，把 dtype 分支从 backend 移走。
- **3.3 buffer naming 收口**：定 canonical 命名 `qweight / weight_scale / input_scale`，AWQ 额外允许 `qzeros / scales`、INT8 W8A8 重命名 `qweight_kn → qweight`（K-major 上面 base 类一致）。更新 [docs/quant_layer_audit.md](docs/quant_layer_audit.md)。Loader 的 `MOE_EXPERT_RE`（[utils/loader.py](nanovllm/utils/loader.py) line 25–27）补全 stacked MoE 后缀（`weight_scale_stacked` 等）。
- **3.4 测试**：每个 kernel 单测 vs reference `F.linear`；`tests/quant/` 增加 `test_kernel_dispatch.py` 验证 LinearBase 委托正确。

## Stage 4 — FP8 MoE 端到端（量化 MoE 的 first class）

- **4.1 export 脚本支持 MoE**：扩展 [scripts/quantize/quantize.py](scripts/quantize/quantize.py)，把 `mlp.experts.<i>.{gate_proj,up_proj,down_proj}.weight` 写为 `qweight + weight_scale`（per-channel）。
- **4.2 stacked weight_scale buffer**：在 [qwen{2,3}_moe.py](nanovllm/models/qwen3_moe.py) 中 `Qwen*MoeSparseMoeBlock` 增加 `w13_weight_scale [E, 2*I_tp]` / `w2_weight_scale [E, H]` 注册；扩展 `load_hybrid_moe_weight` 路由对应 expert + local_id + tp shard。
- **4.3 FP8 MoE Triton kernel**：在 [nanovllm/kernels/group_gemm.py](nanovllm/kernels/group_gemm.py) 增加 fp8 weight 变体（仿 [fp8/kernels.py](nanovllm/layers/fp8/kernels.py) `w8a16_gemm_kernel`：K loop 内 `b_fp8.to(bf16)`，输出按 expert per-channel scale 乘）；提供 `TritonFP8ExpertKernel`（Stage 3.2）。
- **4.4 端到端**：Qwen2-MoE / Qwen3-MoE 加载 FP8 W8A16 ckpt；在 [scripts/eval/eval_ppl_nano_fp8.py](scripts/eval/eval_ppl_nano_fp8.py) 上跑 ppl 回归（vs BF16 基线 < 0.5%）。
- **4.5 测试**：`tests/moe/test_fp8_moe_kernel.py` 与 BF16 MoE 输出对齐（容忍 1e-1 absolute）。

## Stage 5 — DeepEP 接入

- **5.1 依赖**：[pyproject.toml](pyproject.toml) 加 `[project.optional-dependencies] deepep = ["deep_ep>=..."]`，README 增加 CUDA / nvshmem 安装说明。
- **5.2 `DeepEPBackend(MoEBackend)`**：实现两种 mode
  - `normal`：高吞吐，`dispatch` 输出 `(recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert, handle)`，对应 Stage 1.5 的 `DispatchState.handle`。
  - `low_latency`：固定 buffer，可被 CUDA graph 录制；用于 decode + bs 较小场景。
- **5.3 适配 compute()**：DeepEP 输出已 permute by expert，传 `tokens_per_expert` 即可，绕过 Stage 1.2 的 align kernel；`MoEExpertKernel` 接口扩 `compute(recv_x, tokens_per_expert | recv_local_ids, ...)` 两态。
- **5.4 CUDA graph 路径**：Stage 0 的 `enforce_eager` guard 在 `low_latency` mode 下解除；`model_runner.capture_cudagraph()` 区分动态 backend 与 DeepEP low-latency。
- **5.5 默认选择策略**：`MoEBackend.select(ep_size, batch_size, prefill/decode)` —— `ep_size > 1` 默认 DeepEP normal（prefill）/ low_latency（decode）；ep=1 强制 Triton。
- **5.6 Benchmark**：扩展 [scripts/benchmarks/ep_bench.py](scripts/benchmarks/ep_bench.py) 加 `--moe-backend {triton,deepep,deepep_ll}`；记录到 [docs/](docs/) 新报告。

## Stage 6 — 其他量化方案

> Stage 3 的抽象到位后，每条都是"加一个 Kernel 子类 + loader 后缀 + 一个 export 配方"。

- **6.1 INT8 MoE (W8A8)**：复用 `W8A8Kernel`，仿 Stage 4 在 MoE 端生成 stacked int8 buffer + per-expert scale。
- **6.2 W4A8 (DeepSeek-V3 / vLLM marlin)**：新增 `W4A8Kernel`；要么接 marlin 现成 kernel，要么自研 Triton W4A8。
- **6.3 GPTQ INT4**：新增 `GPTQKernel`；与 AWQ 共用 buffer 命名约定。
- **6.4 KV cache INT8**：在 Stage 2.6 FP8 KV 基础上加 INT8 路径（per-head per-token scale）。
- **6.5 SmoothQuant 标定改进**：[scripts/quantize/quantize.py](scripts/quantize/quantize.py) `_calibrate_input_scales` 加 alpha 扫描、per-channel 标定、可选用户 calib 数据集。

---

## 测试与验收策略（贯穿全程）

- **写测试在改代码之前**——loader 与量化路径优先；
- **每个 Stage 都需要 ppl 回归（WikiText-2）+ 输出 token 一致性测试**；
- **MoE Stage 1 / Stage 4 / Stage 5** 的 perf 改动需附带 [scripts/benchmarks/ep_bench.py](scripts/benchmarks/ep_bench.py) 数据落到 [docs/](docs/) 报告里；
- **新加的 kernel** 在 `tests/quant/` / `tests/moe/` 有数值对齐用例；
- **端到端 smoke**：`tests/integration/test_engine_smoke.py` 加载小模型 + 跑几个 token，避免抽象重构破坏接口。

## 风险登记

- **DeepEP 与 CUDA graph 兼容**：仅 `low_latency` mode 可录；Stage 5.4 必须有 fallback。
- **FP8 cuBLASLt 在不同 cuda / driver 下可能 segfault**：Stage 2.1 必须保留 Triton 路径作为 fallback，按 cc + cuda 版本路由。
- **MoE 量化精度**：FP8 MoE expert per-channel scale 不一定够，必要时 Stage 4 加 group-wise scale。
- **抽象重构破坏现有 ckpt 兼容性**：Stage 3 buffer 重命名同时维护一段 legacy alias 期，loader 同时识别旧名。
- **DeepEP 安装门槛高**（nvshmem）：保留 Triton 作为默认，DeepEP 仅在 `pip install -e .[deepep]` 时启用。

## 时间线建议（粗）

- 第 1 周：Stage 0 + Stage 1.1–1.3（perf 关键三项）
- 第 2 周：Stage 1.4–1.8 + Stage 2.1–2.4
- 第 3 周：Stage 2.5–2.8 + Stage 3.1–3.3
- 第 4 周：Stage 3.4 测试 + Stage 4.1–4.3
- 第 5 周：Stage 4.4–4.5 收口 + Stage 5.1–5.3
- 第 6 周：Stage 5.4–5.6 + Stage 6 视精力顺势推
