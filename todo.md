# nano-vllm-moe TODO

> 当前目标：围绕 MoE 推理路径持续优化。已完成第一轮 MoE/EP 解耦和三种 MoE expert backend 对比；下一步重点是 EP 通信层优化、MoE kernel 继续收敛、FP8/INT8/INT4 等量化路径接入。

## 0. 当前状态

### 0.1 已完成：MoE 与 EP 初步解耦

当前 MoE runtime 已从原先 `TritonMoEBackend.dispatch -> compute -> combine` 的单体结构，拆成更接近 vLLM modular MoE 的轻量结构：

```text
router -> prepare/finalize(EP/NoEP) -> experts kernel -> MoEKernel -> BaseSparseMoeBlock
```

已落地的核心文件：

- `nanovllm/executor/moe/config.py`：`MoEParallelConfig`，集中管理 TP/EP rank、size、local experts、local intermediate size。
- `nanovllm/executor/moe/router.py`：`SoftmaxTopKRouter`，统一 Qwen2/Qwen3 router top-k 逻辑。
- `nanovllm/executor/moe/prepare_finalize/`：`NoEPPrepareFinalize` 与 `TorchAllToAllPrepareFinalize`。
- `nanovllm/executor/moe/experts/`：三种 expert compute backend。
- `nanovllm/executor/moe/kernel.py`：串联 prepare、experts、finalize。
- `nanovllm/executor/moe/blocks/base.py`：`BaseSparseMoeBlock`，Qwen2/Qwen3 共享 MoE 主流程。
- `nanovllm/models/qwen2_moe.py` / `nanovllm/models/qwen3_moe.py`：模型文件不再直接写 EP dispatch/compute/combine 细节。

### 0.2 已完成：三种 MoE expert backend

当前支持通过 `moe_backend` 选择 expert backend：

| backend | 作用 | 当前状态 |
| --- | --- | --- |
| `transformers` | native/eager expert loop，作为语义基线 | 已跑通 |
| `mini_sglang` | mini-sglang 风格 Triton fused expert GEMM + reduce | 已跑通 |
| `fused` | nano-vllm 当前高融合 group GEMM 路径 | 已跑通 |

关键补充：`mini_sglang` backend 已不是 Python loop 近似实现，而是移植了 mini-sglang 风格 Triton GEMM/reduce kernel 到 `nanovllm/kernels/sglang_moe.py`，并在 `nanovllm/executor/moe/experts/sglang.py` 中调用。

### 0.3 已完成：脚本与远端验证

新增/更新脚本：

- `scripts/benchmarks/ep_bench.py`：支持小负载参数和资源参数，便于 EP smoke。
- `scripts/benchmarks/ep_tp_bench.py`：支持 `--configs` 指定如 `1,2`，避免默认扫超出当前 GPU 数量的配置。
- `scripts/benchmarks/moe_backend_bench.py`：对比 `transformers,mini_sglang,fused` 三种 MoE backend。
- `scripts/examples/example.py`：已用 EP=2 小配置跑通。

远端验证摘要：

- 远端模型路径：`/home/ubuntu/project/models/qwen/Qwen1.5-MoE-A2.7B-Chat`
- 远端 Python：`/home/ubuntu/miniconda3/envs/nano-vllm/bin/python`
- GPU：2 张 RTX 4090。
- `pytest tests/moe tests/loader/test_loader_mapping.py`：通过。
- `example.py`：EP=2 通过。
- `ep_bench.py`：EP=2 小负载通过。
- `ep_tp_bench.py --configs 1,2`：通过。
- `moe_backend_bench.py --backends transformers,mini_sglang,fused`：通过。

当前小负载对比结果：

```text
backend=transformers, wall=0.212s, prefill_tps=41.31, decode_tps=11.05
backend=mini_sglang, wall=0.140s, prefill_tps=67.23, decode_tps=15.28
backend=fused,       wall=0.131s, prefill_tps=73.92, decode_tps=15.69
```

说明：这是小负载 smoke 结果，用于确认 backend 路径和趋势；稳定性能结论需要扩大 batch/seq/output 并多轮取均值。

### 0.4 已知运行约束

- 当前分布式初始化仍使用固定端口 `2335`；远端测试前必须清理旧监听进程，否则会触发 `EADDRINUSE`。
- `Qwen1.5-MoE-A2.7B-Chat` 在单张 4090 上可能 OOM；EP=1 失败不能作为 EP=2 失败依据。
- 多进程 spawn 脚本不要通过 stdin heredoc 运行，应使用仓库脚本或远端 `/tmp/nano_*.py` 文件。
- `.remote-logs/` 已加入 `.gitignore`，远端验证日志不应提交。

---

## 1. MoE 当前优化方向

### 1.1 完善 MoE backend 选择与实验能力

- 为 `moe_backend` 增加更完整的 README/脚本文档，明确 `transformers` / `mini_sglang` / `fused` 的用途和限制。
- 在 benchmark 中加入多轮运行与均值/方差统计。
- 增加结果输出为 JSON/CSV，便于长期性能回归。
- 增加更大负载配置：prefill-heavy、decode-heavy、mixed workload。

### 1.2 数值一致性测试

需要补充自动化测试，避免后续 kernel 优化破坏语义：

- `transformers` vs `mini_sglang` 输出对齐。
- `transformers` vs `fused` 输出对齐。
- EP=1 / EP=2 dispatch+combine roundtrip 对齐。
- Qwen2 shared expert 路径对齐。
- Qwen3 top-k renormalize 路径对齐。

建议新增：

```text
tests/moe/test_backend_consistency.py
tests/moe/test_ep_prepare_finalize.py
tests/moe/test_qwen2_shared_expert.py
```

### 1.3 kernel 层继续优化

当前 `fused` 路径仍是性能主线，后续重点：

- 减少 dispatch/combine 中的 host sync，如 `.tolist()` / `.item()`。
- 优化 `moe_align_block_size`：从 torch 张量操作逐步下沉到 device kernel。
- 评估当前 `fused_moe_w2_combine_kernel` 的 fp32 中间/atomic 路径，探索无 atomic 或更轻量 combine。
- 对 decode 小 batch 做专门 config/autotune。
- 对 prefill 大 batch 做更稳定的 block size 配置。

---

## 2. EP 后续优化方向

### 2.1 抽象 EP 通信层

当前已经有 `NoEPPrepareFinalize` 和 `TorchAllToAllPrepareFinalize`，下一步建议继续拆细为更明确的通信抽象：

```text
prepare_finalize/
  no_ep.py
  torch_alltoall.py
  deepep.py
```

后续目标：保持 expert kernel 不变，仅替换通信方案。

### 2.2 Torch all-to-all 路径优化

当前 torch all-to-all 版本可用，但仍有优化空间：

- 合并多次 all-to-all：当前 hidden、expert id、routing weight 分开发送，可尝试打包通信。
- 减少 CPU 同步：`send_counts.tolist()`、`recv_counts.sum().item()` 会同步 host。
- 使用 device-side prefix sum/cumsum 管理 offsets。
- 将 dispatch metadata 显式类型化，避免 dict contract 漂移。

### 2.3 DeepEP 接入

DeepEP 应作为 prepare/finalize 的实现接入，而不是替换整个 MoE backend：

```text
router -> DeepEPPrepareFinalize -> experts backend(transformers/mini_sglang/fused) -> finalize
```

接入步骤：

1. 新增 `DeepEPPrepareFinalize` skeleton。
2. 先支持 normal dispatch/combine。
3. 再支持 low-latency decode 路径。
4. 明确 DeepEP handle 生命周期与 cleanup。
5. 与 `fused` expert kernel 组合测试。
6. 再与 FP8/INT8 expert kernel 组合测试。

### 2.4 CUDA graph 与动态 EP

动态 all-to-all 与 CUDA graph 可能冲突。后续需要 backend 暴露能力：

```python
supports_cuda_graph: bool
uses_dynamic_alltoall: bool
```

短期策略：

- EP>1 或 dynamic all-to-all backend 时强制 eager。
- 给出明确日志，避免 silent fallback。

---

## 3. 量化接入路线

### 3.1 dense linear 量化继续优化

现有 FP8/INT8 dense linear 路径后续继续推进：

- W8A16：评估 Hopper cuBLASLt / Triton 路径取舍。
- W8A8 static：融合 scale epilogue，减少 fp32 中间张量。
- W8A8 dynamic：引入 per-token activation scale。
- KV cache FP8：先保留 runtime capability guard，再逐步落 kernel。
- INT8 SmoothQuant：继续补齐 W8A8/W8A16 的性能和一致性测试。

### 3.2 FP8 MoE expert kernel

MoE 量化应在新的 expert kernel 抽象下接入：

```text
experts/
  triton_grouped_gemm.py
  sglang.py
  fp8_grouped_gemm.py
  int8_grouped_gemm.py
```

FP8 MoE 需要解决：

- checkpoint/export 中 expert weight scale 的布局。
- `w13_stacked` / `w2_stacked` 对应 scale 的 stacked loader。
- FP8 dequant 是否在 kernel 内完成。
- per-tensor / per-channel / block scale 的统一接口。
- 与 EP dispatch 后 local expert id 的 scale 索引对齐。

### 3.3 INT8 / W4A8 / GPTQ INT4 MoE

后续量化优先级建议：

1. INT8 MoE W8A16：实现简单，先作为量化 MoE 基线。
2. INT8 MoE W8A8 static/dynamic：需要 activation scale 和 epilogue 支持。
3. W4A8 / GPTQ INT4：先保留 reference path，再做性能 kernel。
4. AWQ INT4：保持现有实现作为 reference，后续纳入统一 `MoEExpertsKernel`。

### 3.4 量化测试指标

每个量化路径都需要：

- 小模型/小 batch correctness。
- 与 bf16/eager 的 token-level 差异。
- perplexity 或固定 prompt 一致性。
- prefill/decode 吞吐。
- 显存峰值。
- EP=1/EP=2 对比。

---

## 4. Benchmark 与验证规范

### 4.1 默认远端 smoke 参数

在当前 2 张 RTX 4090 环境下，优先使用：

```bash
--tp-size 1 \
--ep-size 2 \
--enforce-eager \
--max-model-len 64 \
--max-num-batched-tokens 64 \
--max-num-seqs 1 \
--gpu-memory-utilization 0.95 \
--num-seqs 1 \
--min-input-len 4 \
--max-input-len 4 \
--min-output-len 2 \
--max-output-len 2
```

### 4.2 后端对比命令

```bash
PYTHONPATH=/home/ubuntu/project/nano-vllm-moe \
/home/ubuntu/miniconda3/envs/nano-vllm/bin/python \
  scripts/benchmarks/moe_backend_bench.py \
  --model-path /home/ubuntu/project/models/qwen/Qwen1.5-MoE-A2.7B-Chat \
  --backends transformers,mini_sglang,fused \
  --tp-size 1 \
  --ep-size 2 \
  --enforce-eager \
  --max-model-len 64 \
  --max-num-batched-tokens 64 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.95 \
  --num-seqs 1 \
  --min-input-len 4 \
  --max-input-len 4 \
  --min-output-len 2 \
  --max-output-len 2
```

### 4.3 每次远端测试前

- 清理固定端口 `2335` 监听进程。
- 检查 `nvidia-smi --query-compute-apps` 无残留测试进程。
- 日志写入 `.remote-logs/`，但不要提交。
- 多进程测试使用脚本文件，不使用 stdin heredoc。

---

## 5. 近期 TODO

### P0 — 稳定性与测试

- [ ] 给 `moe_backend` 增加单元测试和配置错误提示。
- [ ] 增加三 backend 数值一致性测试。
- [ ] 增加 EP prepare/finalize roundtrip 测试。
- [ ] benchmark 输出 JSON/CSV。
- [ ] 远端脚本统一在启动前执行 cleanup helper。

### P1 — EP 优化

- [ ] 将 `TorchAllToAllPrepareFinalize` 的 host sync 降到最低。
- [ ] 合并 all-to-all payload，减少通信次数。
- [ ] 明确 `PrepareResult` / EP metadata contract。
- [ ] 新增 `DeepEPPrepareFinalize` skeleton。
- [ ] 接入 DeepEP normal path。
- [ ] 接入 DeepEP low-latency decode path。

### P2 — MoE kernel 优化

- [ ] Triton 化或优化 `moe_align_block_size`。
- [ ] 优化 fused W2 combine，减少 fp32/atomic 开销。
- [ ] 为 decode/prefill 分别 autotune kernel config。
- [ ] 对比 `mini_sglang` 与 `fused` 在更大 batch/seq 下的稳定收益。

### P3 — 量化 MoE

- [ ] FP8 MoE scale loader 与 stacked scale 布局。
- [ ] FP8 grouped GEMM expert kernel。
- [ ] INT8 MoE W8A16 baseline。
- [ ] INT8 MoE W8A8 static/dynamic。
- [ ] W4A8 / GPTQ INT4 MoE reference path。
- [ ] 量化 MoE 的 ppl/固定 prompt/吞吐/显存回归。
