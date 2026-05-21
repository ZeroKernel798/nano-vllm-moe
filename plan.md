# nano-vllm-moe Optimization Plan

本计划按三条主线组织后续优化，避免实验继续发散：

1. 量化类：FP8 W8A16/W8A8、8-bit KV cache、量化 attention 算子、精度/效率评测和自研算子。
2. 推理框架级别：chunked prefill、prefix cache、HiRadix、HiCache。
3. MoE 内核：保留 `optimized` 作为唯一非 eager 后端（fused 已移除），MoE CUDA Graph 已在单卡默认开启，后续聚焦 EP graph 化与 FP8 KV graph 化。

本地仓库仍是事实源；远端 GPU 机器只作为验证环境。所有性能和精度结论都必须有远端日志。

---

## 1. Quantization Track

目标：在 RTX 4090 / SM89 上形成一条可解释的量化栈：BF16 baseline -> W8A16 -> mixed 8-bit KV -> quantized attention -> W8A8 shape-aware。量化优化必须同时回答三个问题：精度是否稳、端到端是否更快或更省、底层算子是否有可控实现。

### 1.1 W8A16

当前状态：

- BF16/FP16 原型是 W8A16 性能、显存和质量对照 baseline。
- Triton 可用：已绕开 SM90-only FP8->BF16 指令，改为 FP8->FP32->FP16 on-the-fly dequant。
- FP8 per-forward dequant matmul 只保留为 correctness fallback / debug path，不再作为性能主线。
- CUDA/PTX 未实现，是下一阶段主线。

路线：

1. 保留 BF16/FP16 原型作为真实 baseline。
2. Triton 作为当前 W8A16 主实现，先优化 `gate/up/down` 关键 MLP shape。
3. per-forward dequant matmul 仅用于 fallback/debug，不进入 README 性能主表。
4. 新增自研 CUDA/PTX W8A16 kernel：
   - FP8 weight resident。
   - BF16/FP16 activation。
   - per-output-channel weight scale。
   - output BF16。
   - 不采用 load-time full BF16 dequant cache 作为默认优化方向。

成功标准：

- 3B/7B smoke 通过。
- logits cosine、top-k overlap、token match 相对 BF16/FP16 baseline 无明显回退。
- 至少在 decode 或关键 MLP prefill shape 上超过 BF16/FP16 reference 或给出明确显存/速度 tradeoff。

### 1.2 W8A8

当前状态：

- W8A8 static checkpoint contract 已有：`qweight`、`weight_scale`、`input_scale`。
- Triton full path 是当前稳定 runtime baseline：Triton activation quant + `torch._scaled_mm`。
- `_scaled_mm` only 只作为 GEMM 拆解参考，不是 runtime baseline，因为 runtime activation quant 必须计入 full path。
- 自研 CUDA/PTX JIT 已有可用 single-warp `m16n8k32` primitive：对齐 CUTLASS SM89 `BLayout` 后，synthetic fragment gate exact，3B `M=16` q_proj/gate_proj full path 已快于 Triton full。
- 当前 PTX MMA 仍是 single-warp `16x8` tile，不是最终 GEMM kernel；下一阶段是 multi-warp CTA / staged shared-memory pipeline。

路线：

1. Triton full path 继续作为默认稳定实现和 regression baseline。
2. CUDA/PTX path 作为性能主线推进，保留显式 opt-in：
   - `NANOVLLM_FP8_W8A8_RUNTIME_BACKEND=cuda_ptx`。
   - `NANOVLLM_W8A8_JIT_KERNEL=ptx_mma`。
3. 下一步参考 `gemm-fp8` / CUTLASS / vLLM 的 SM89 FP8 tiling，把 single-warp primitive 扩成 multi-warp CTA：
   - 先做 `M=16` small-M 专用 CTA，覆盖 q_proj/gate_proj。
   - 再扩展 `M=32/64/128/512` 分档。
   - 目标 tile 从 `64x128x64` 或 `128x64x64` 开始，按 q_proj 与 gate/up/down shape 分别调。
4. 继续拆分并记录：
   - activation quant time。
   - GEMM time。
   - fused PTX full path。
   - output cast/bias time。
   - end-to-end layer time。
5. 不重新引入 dequant fallback、shape-aware `_scaled_mm` policy 或 hidden BF16 weight cache 作为默认 W8A8 runtime。

成功标准：

- WikiText PPL、MMLU/CEval choice-logit gate、logits/token gate 通过。
- Full path 对 Triton full 有稳定收益，而不是只赢 `scaled_mm` 拆解项。
- `M=16` 之外的 `M=32/64/128/512` 真实 Qwen projection shape 有覆盖。
- 端到端 7B 不出现不可接受质量回退。

下一步参考资料：

- CUTLASS operand layout：`/Users/a1-6/project/llm_infer/cutlass/include/cute/atom/mma_traits_sm89.hpp`。
- CUTLASS PTX wrapper：`/Users/a1-6/project/llm_infer/cutlass/include/cute/arch/mma_sm89.hpp`。
- `gemm-fp8` SM89 baseline：`/Users/a1-6/project/llm_infer/gemm-fp8/gemm_fp8/kernels/gemm.cu`，重点看 `64x128x64`、`128x64x128`、fastAcc `128x128x64` 配置。
- vLLM SM89 FP8 dispatch：`/Users/a1-6/project/llm_infer/vllm/csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x_sm89_fp8_dispatch.cuh`。
- FlashInfer MMA wrapper 仅作 operand/ldmatrix 参考，不作为 runtime 依赖。

### 1.3 8-bit / Mixed KV Cache

当前状态：

- 纯 FP8 K/V 对 7B 不稳，主要问题来自 K quantization。
- V-only FP8 已证明更稳。
- K-int8/V-FP8 已通过 512/8K/16K/32K token gate，并显著降低 KV bytes/block。

路线：

1. 保留 BF16 KV 作为质量 baseline。
2. 把 K-int8/V-FP8 作为主 KV 压缩方向。
3. 补齐短上下文 cutoff：短上下文若 native mixed KV 慢于 BF16，应自动回退。
4. 继续优化 mixed KV store/decode kernel：
   - fused store。
   - native mixed paged decode。
   - token-block split/reduction。
5. KV cache 与 prefix/HiCache 结合时，优先支持 BF16 和 K-int8/V-FP8 两种存储格式。

成功标准：

- 多 seed、多 prompt、多长度保持 exact token 或明确 bounded drift。
- 8K/16K/32K 长上下文有明确 memory 和 decode TPS 收益。
- README 只记录通过 gate 的 KV 模式，不宣传失败的纯 FP8 K/V。

### 1.4 Quantized Attention Kernel

目标：在 KV cache 压缩之外，探索 attention 计算本身的量化实现。当前路线只保留本仓库可控的独立 kernel / benchmark，不引入第三方 attention 或 KV 压缩实现作为 runtime 依赖。

当前状态：

- BF16 paged attention 是基础 reference。
- mixed KV 已经压缩存储，但 attention compute 仍需要进一步拆分 profile。
- attention compute 需要用本仓库可维护的 microbench 拆解 Q/K/V、score、P/V matmul、online softmax 等环节是否能用低精度路径获益。

路线：

1. 先建立 attention microbench：
   - prefill attention。
   - decode paged attention。
   - short/medium/long context。
   - BF16 KV 与 K-int8/V-FP8 KV 两种输入。
2. 拆分 attention 耗时：
   - Q/K load 与 dequant。
   - QK score。
   - softmax。
   - PV。
   - output writeback。
3. 评估自研低精度 attention 路径：
   - per-token / per-block scale。
   - int8 或 fp8 Q/K/V compute。
   - online dequant 与 fused attention。
   - 和现有 paged KV layout 的兼容成本。
4. 先做独立 kernel / benchmark，不急着进入默认 runtime。
5. 如果质量 gate 稳定，再接入 shape/context-aware backend selection。

成功标准：

- 相对 BF16 attention logits/output 误差可控。
- 在长上下文 decode 或 prefill attention 上有明确 TPS/latency 收益。
- 与 K-int8/V-FP8 KV cache 能组合，不重复引入额外不可控量化误差。

### 1.5 Quantization Bench Gates

每个量化模式至少覆盖：

- Kernel microbench：真实 Qwen shape，decode-like `M=1`，prefill `M=128/512/2048`。
- Model smoke：3B 快速 gate，7B 主 gate。
- Quality：PPL、logits cosine、top-k overlap、token match、MMLU/CEval。
- Memory：checkpoint size、peak allocated/reserved、KV bytes/block、max context/capacity。

---

## 2. Inference Framework Track

目标：在不改变模型数学的情况下减少重复计算、控制长上下文调度开销，并让结果可测。

### 2.1 Chunked Prefill

当前状态：

- 已有 `prefill_first` 和 `decode_first` 两种 policy。
- 作为长上下文调度基础保留。

路线：

1. 保持 correctness 和 benchmark 覆盖。
2. 明确 policy 适用场景：
   - `prefill_first`：简单稳定。
   - `decode_first`：降低 decode 等待。
3. 与 KV compression、prefix cache、HiCache 组合测试。

成功标准：

- 长 prompt 不因 BF16 reference 或大 activation 先 OOM 而干扰容量结论。
- 输出 TTFT、decode TPS、prefill TPS、peak memory。

### 2.2 Prefix Cache L0/L1: Current Hash Prefix Cache

当前状态：

- `BlockManager` 已有 block hash prefix cache。
- 以完整 KV block 为单位复用 GPU KV。

路线：

1. 增加 stats：
   - cache hit blocks/tokens。
   - scheduled prefill tokens。
   - logical prompt tokens。
   - prefix cache lookup time。
2. 增加 benchmark：
   - no reuse。
   - shared prefix。
   - partial shared prefix。
   - block-aligned vs unaligned。
3. 作为 HiRadix/HiCache 对比 baseline。

成功标准：

- 多请求共享长前缀时，后续请求实际 prefill tokens 明显下降。
- TTFT 相比 no-cache baseline 明显下降。

### 2.3 HiRadix L1

目标：用 radix tree 替代单纯 hash map metadata，仍只管理 GPU KV blocks。

路线：

1. 实现 mini HiRadix metadata：
   - `match_prefix(tokens)`。
   - `insert(tokens, block_table)`。
   - node split。
   - LRU/LFU 或最小 LRU eviction。
2. 与当前 hash prefix cache 做 A/B。
3. 先不做 CPU offload，不接 L3 storage。

对比维度：

- lookup time。
- matched prefix length。
- hit tokens。
- branching/shared-prefix workload 下的复用率。
- eviction 后的命中率。

成功标准：

- 在分叉 prefix workload 中比 hash prefix cache 更好解释、更高命中或更稳定。
- 不明显拉高普通请求调度开销。

### 2.4 HiCache L1/L2

目标：在 HiRadix metadata 上增加 CPU host KV cache，让 GPU 放热 KV，CPU 放冷 KV。

路线：

1. L1：GPU KV blocks。
2. L2：CPU pinned KV blocks。
3. radix node 记录 KV 所在层级：
   - GPU resident。
   - CPU resident。
   - missing。
4. 先做同步 host->gpu load-back，再考虑 async prefetch。
5. 暂不实现 L3 storage、Mooncake、3FS、NIXL。

成功标准：

- 在 GPU KV block 受限时，L1/L2 比纯 GPU prefix cache 保留更多可复用前缀。
- host->gpu copy overhead 小于重算 prefill 的收益。
- 支持 BF16 KV，后续扩展到 K-int8/V-FP8 KV。

---

## 3. Fused MoE Kernel Track

目标：保留当前 MoE runtime 的简洁结构，同时让单卡 fused MoE 有更强性能故事。EP 仍只保留 torch all-to-all baseline，不作为当前优化主线。

### 3.1 Current MoE Backends

当前结构：

```text
router -> prepare/finalize -> expert backend -> finalize output
```

backend：

- `eager`：正确性参考。
- `optimized`：当前 practical optimized path。
- `fused`：Triton grouped-GEMM experiment。

路线：

1. 保持三个 backend 的 correctness 对齐。
2. `fused` 继续作为主要 MoE kernel 优化对象。
3. `optimized` 保留作为 fallback / 对照。
4. 增加 MoE profiler 输出：
   - router。
   - prepare。
   - w13/gate-up。
   - activation。
   - w2/down-combine。
   - finalize。

### 3.2 MoE CUDA Graph Optimization

当前判断：

- 单卡 `NoEPPrepareFinalize.supports_cuda_graph=True`。
- EP `TorchAllToAllPrepareFinalize.supports_cuda_graph=False`。
- 因此先做单卡 decode MoE CUDA Graph，不碰动态 all-to-all。

路线：

1. 确认 `enforce_eager=False` 下 MoE decode 是否真的被 CUDA Graph 覆盖。
2. 做 graph-safe fused MoE path：
   - 按 batch size bucket capture。
   - 预分配 `activated_out`、`local_out_fp32`、sorted buffers。
   - 减少 forward 内 `torch.empty/zeros`。
   - 固定或上界化 `num_blocks`，kernel 内 mask 无效块。
3. 动态路由相关操作先保持 bucket 化，不追求 EP graph。

成功标准：

- MoE decode TPS 相比 eager/no-graph 有明确提升。
- capture/replay 不破坏路由正确性。
- graph path 和 eager/fused path 输出误差可控。

### 3.3 Future CUDA/PTX MoE

如果 Triton fused MoE 和 CUDA Graph 路线收敛，再考虑 CUDA/PTX：

- fused gate/up + activation。
- down/combine。
- FP8/INT8 expert weight support。
- 与 W8A16/W8A8 quantization track 共享 scale/layout 经验。

---

## Immediate Priorities

### Phase 1: Prefix Cache & HiCache Track ✅ 完成

1. **Prefix Cache L0/L1: Stats + Benchmark** ✅
   - Token hit rate 75% (shared-prefix 场景)
   - 支持 hash/radix 两种 backend

2. **HiRadix L1: Radix Tree** ✅
   - 实现 `RadixTreePrefixCache` 类
   - 支持 LRU eviction
   - 环境变量: `NANOVLLM_PREFIX_CACHE_BACKEND=radix`

3. **HiCache L1/L2: CPU Host KV Cache** ✅ (框架)
   - `HiCacheManager` 类实现 GPU/CPU 两层缓存

### Phase 2: CUTLASS C++ Kernel ✅ 框架完成

4. **CUTLASS C++ FP8 GEMM Kernel** ✅ (框架)
   - `csrc/fp8_gemm.cu`
   - SM89 tensor core 实现

### Phase 3: Documentation & Testing (进行中)

5. **Update Documentation** - 进行中

6. **Comprehensive Benchmark** - 待完成

## Completed

### 推理框架优化
- Prefix Cache L0/L1: Stats + Benchmark (token hit rate 75%)
- HiRadix L1: Radix tree 实现 (支持 hash/radix 切换)
- HiCache L1/L2: CPU host KV cache 框架

### 量化集成
- CUTLASS C++ FP8 GEMM 框架

### W8A8
- CUTLASS 集成：通过 `torch._scaled_mm` 实现，cosine 0.9993
- Triton fused kernel：作为默认 runtime

## Deferred

- 完整 HiCache L3 storage
- EP CUDA Graph
- PTX MMA W8A8 (数值问题未解决)
- W8A16 load-time BF16 weight cache
