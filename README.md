# Nano-vLLM-MOE

本项目是以Nano-vLLM项目为基底，添加了对Qwen2、Qwen3系列的Moe模型支持，并提供了EP、TP以及EP、TP混合的调度实现。用于在现有的Nano-vLLM的基础上，学习和分析LLM的分布式并行策略。原项目地址为：https://github.com/GeeeekExplorer/nano-vllm.git

## 主要特点

* Tensor Parallelism — Column/Row 并行分片策略，显著提升计算密集型任务的硬件利用率
* Expert Parallelism — 专为大规模 MoE 设计的专家分片架构，支持超大规模专家库的高效驻留与并行调度
* Hybrid TP+EP — 灵活的嵌套并行模式，允许动态调整并行粒度，实现通信与计算的最优平衡
* Triton Group-GEMM — 使用 Triton 实现了 Group-Gemm 等核心算子，显著降低 MoE 层通信延迟
* Expert Overlap Execution — 支持 Shared Expert 与 Sparse Expert 并行执行，利用计算掩盖 all-to-all 通信延迟
* Benchmarking Suite — 完善的性能评测套件，提供 TTFT、TPOT 等详细指标输出
* Preserved Optimizations — 完整保留了原项目的 Prefix caching, Torch compilation 以及 CUDA graph 等生产级优化
* MoE-Aware CUDA Graph — 计划绕过动态路由的同步限制，实现 MoE 全路径的 CUDA Graph 录制
* Quantization — 计划支持 FP8 与 INT4 量化，用于缓解 Decode 阶段的带宽瓶颈，进一步推高生成吞吐

## 安装

git clone https://github.com/ZeroKernel798/nano-vllm-moe.git

cd nano-vllm-moe

pip install -e .

## 模型下载

推荐使用内置的下载脚本从 ModelScope 极速获取模型

python scripts/tools/downmodel.py --model-id qwen/Qwen1.5-MoE-A2.7B-Chat --cache-dir ./my_models


## Benchmark

使用 `scripts/benchmarks/ep_bench.py` 模拟生产环境下随机输入序列，测试Group-Gemm相关实现的性能

**测试配置**
- Hardware: NVIDIA A100-SXM4-80GB (HBM2e / NVLink)
- Model: Qwen1.5-MoE-A2.7B-Chat
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 128–1024 tokens
- Output Length: Randomly sampled between 128–1024 tokens
- Parallelism Strategy: TP = 1, EP = 1

**测试结果:**
|          实现方案          | input  tokens | output tokens | 解码吞吐（tokens/s）| 总吞吐（tokens/s）|
|---------------------------|---------------|-------------- |--------------------|------------------|
|        Python Loop        |     155216    |     153174    |      156.67        |      315.42      |      
|     Triton Group-GEMM     |     155216    |     153174    |      1286.5        |      2590.1      |        
|Triton Group-GEMM w Overlap|     155216    |     153174    |      1328.5        |      2674.7      |   



可使用 `scripts/benchmarks/ep_tp_bench.py` 模拟生产环境下随机输入序列多卡推理测试，横向对比各种并行策略，一键生成测试结果

**测试配置**
- Hardware: NVIDIA A100-SXM4-80GB (HBM2e / NVLink)
- Model: Qwen1.5-MoE-A2.7B-Chat
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 128–512 tokens
- Output Length: Randomly sampled between 128–512 tokens
- Expert Execution Logic: Group-Gemm + Overlap

**测试结果:**
| 并行配置  | 显卡数 | 解码吞吐（tokens/s） | 总吞吐（tokens/s）|
|----------|-------|--------------------|--------------------|
|TP=1, EP=1|   1   |      2240.64       |     4609.59        |
|TP=1, EP=2|   2   |      1937.79       |     3986.54        |
|TP=2, EP=1|   2   |      2455.39       |     5051.39        |
|TP=2, EP=2|   4   |      1821.96       |     3748.26        |
|TP=4, EP=1|   4   |      2430.07       |     4999.32        |


可使用 `scripts/benchmarks/pd_bench.py` 输入给定长度和批次的序列，分析 prefill 和 decode 阶段，不同分布式策略的性能

**测试配置**
- Hardware: NVIDIA A100-SXM4-80GB (HBM2e / NVLink)
- Model: Qwen1.5-MoE-A2.7B-Chat
- Expert Execution Logic: Group-Gemm + Overlap

**测试结果:**
#### 🚀 Prefill 阶段性能 (Prefill Stage)
*关注点：TP 对计算密集型任务的加速比，以及长文本下的延迟表现。*
| 阶段    |   测试配置   | 吞吐量 (Throughput) | TTFT (Latency) | 加速比 |  性能损耗/增益说明  |
|---------|-------------|---------------------|---------------|--------|-------------------|
| Prefill | TP=1,EP=1,BS=128,L=1024 | 27340.26 tok/s | 2730.31 ms | 1.00x |    -----    |
| Prefill | TP=2,EP=1,BS=128,L=1024 | 40771.19 tok/s | 1797.09 ms | 1.49x | 计算/通信比平衡，收益显著 |
| Prefill | TP=1,EP=2,BS=128,L=1024 | 23027.90 tok/s | 3254.64 ms | 0.84x | all_to_all 通信开销过大 |
| Prefill | TP=2,EP=2,BS=128,L=1024 | 32439.96 tok/s | 2332.75 ms | 1.19x | 混合并行抵消了部分 TP 收益 |
| Prefill | TP=4,EP=1,BS=128,L=1024 | 54041.02 tok/s | 1384.21 ms | 1.98x | 4 卡并行接近 2 倍收益，受限于小模型计算量 |
| Prefill | TP=1,EP=1,BS=64,L=4096 | 26460.86 tok/s | 5285.74 ms | 1.00x | ----- |
| Prefill | TP=2,EP=1,BS=64,L=4096 | 40035.93 tok/s | 3501.27 ms | 1.51x | 序列变长，计算密度增加，TP 效率提升 |
| Prefill | TP=1,EP=2,BS=64,L=4096 | 22754.84 tok/s | 6129.04 ms | 0.86x | 模型太小，EP 依旧亏损 |
| Prefill | TP=2,EP=2,BS=64,L=4096 | 32328.42 tok/s | 4356.89 ms | 1.22x | 较上一批测试 有轻微提升 |
| Prefill | TP=4,EP=1,BS=64,L=4096 | 52433.40 tok/s | 2693.42 ms | 1.98x | 达到目前框架在 2.7B 模型下的性能瓶颈 |

#### ⚡ Decode 阶段性能 (Decode Stage)
*关注点：通信延迟对小模型生成速度的影响，寻找 Batch Size 的性能拐点。*
| 阶段    |   测试配置   | 吞吐量 (Throughput) | TPOT (Latency) | 加速比 |  性能损耗/增益说明  |
|---------|-------------|---------------------|---------------|--------|-------------------|
| Decode | TP=1,EP=1,BS=1,L=4 | 9.28 tok/s | 107.56 ms | 1.00x | ----- |
| Decode | TP=2,EP=2,BS=1,L=4 | 7.65 tok/s | 130.72 ms | 0.82x | 混合并行带来的同步延迟导致响应变慢 |
| Decode | TP=4,EP=1,BS=1,L=4 | 9.34 tok/s | 107.17 ms | 1.01x | 计算量极小，4 卡同步开销抵消了计算收益 |
| Decode | TP=1,EP=1,BS=256,L=4 | 4864.58 tok/s | 53.00 ms | 1.00x | ----- | 
| Decode | TP=2,EP=2,BS=256,L=4 | 2945.15 tok/s | 87.48 ms | 0.61x | 复杂的分布式链路导致性能大幅跌落 |
| Decode | TP=4,EP=1,BS=256,L=4 | 4032.50 tok/s | 64.20 ms | 0.83x | 卡间 All-Reduce 耗时超过计算耗时 |
> **Bottleneck analysis**: <small>
经过对 Qwen-1.5-MoE-2.7B 的深度压测，我们发现多卡并行（TP/EP）在 decode 阶段表现不佳，这并非框架缺陷，而是由以下两个核心技术瓶颈决定的：
1、计算密度不足。对于 2.7B 这样的小规模模型，Decode 阶段单次算子的执行时间（us级）已经接近甚至小于 NVLink 的同步延迟。在 TP 或 EP 模式下，多卡之间频繁的 all-reduce 或 all-to-all 通信开销完全盖过了并行带来的计算收益。在这种级别的计算任务面前，“力大砖飞”的分布式策略反而成了性能累赘。2、MoE 动态特性对 CUDA Graph 的局限。目前的 MoE 实现采用了动态的 Token 分发逻辑，这需要 CPU 实时介入来决定下一阶段的计算规模。这种数据依赖的动态调度与 CUDA Graph 要求的“全静态计算图”存在天然冲突，导致目前无法通过录制 Graph 来消除 Python 调度和 Kernel Launch 的 Overhead。这也是 BS=1 时 TPOT 难以进一步下探的技术原因。后续将考虑引入量化以及 MoE 逻辑重构，利用 CUDA Graph 进一步优化，全部测试结果见docs/pd_separate_report.md</small>

## Llama 3.1 8B 量化与基线对比（RTX 4090，单卡）

以下数据来自本仓库中的 `scripts/benchmarks/bench.py`、`scripts/benchmarks/pd_bench.py` 与 WikiText 困惑度脚本，详细原始表格见 `docs/fp8_bf16_bench.md`、`docs/pd_separate_report_bf16.md`、`docs/pd_separate_report_fp8_w8a8.md`、`docs/pd_separate_report_fp8_w8a16.md` 与 `docs/wikitext.md`。硬件为 **NVIDIA GeForce RTX 4090**，模型为 **Meta Llama 3.1 8B** 系列；P/D 测试为 **TP=1, EP=1**。

### 端到端混合负载（scripts/benchmarks/bench.py）

**测试配置（与 `docs/fp8_bf16_bench.md` 一致）**

- Hardware: NVIDIA GeForce RTX 4090
- Model: Llama 3.1 8B（BF16 基座 / W8A8 ModelOpt 导出 / W8A16 纯权重量化导出）
- Parallelism: TP = 1, EP = 1
- Sequences: 256
- Input / output length: 随机长度（`--random-lens`），上界为 1024 tokens，下界为 100 tokens
- max_model_len: 4096

**测试结果**

| 方案 | wall_time (s) | prefill_tps | decode_tps | total_gen_tokens |
|------|---------------|-------------|------------|------------------|
| BF16 基线 | 74.900 | 11070.90 | 2454.68 | 133966 |
| FP8 W8A8 | 69.216 | 12908.28 | 2620.51 | 133966 |
| FP8 W8A16 | 99.906 | 4182.87 | 2858.90 | 133966 |

*分析要点：* 在相同随机负载与总生成 token 数下，**W8A8** 总墙钟时间最短，prefill / decode 吞吐均高于 BF16，说明静态 FP8 激活 + FP8 权重路径在本机与 cuBLASLt 组合下对混合负载更友好。**W8A16** 的 prefill_tps 明显偏低而 decode_tps 高于 BF16，与「prefill 偏算力、W8A16 Triton 路径与 BF16 主路径效率差异；decode 偏权重带宽、FP8 权重量更省」的典型现象一致。

### Prefill / Decode 阶段分离（scripts/benchmarks/pd_bench.py）

**测试配置**

- Hardware: NVIDIA GeForce RTX 4090
- Model: 同上，单卡 TP=1, EP=1
- Prefill：多组 (batch, seq_len)；
- Decode：短提示 + 长生成阶段下的 batch 扫描

**代表性对比（摘自数据报告，单位 tok/s）**

| 阶段 | 配置 | BF16 | W8A8 | W8A16 |
|------|------|------|------|-------|
| Prefill | BS=32, L=512 | 10994.44 | 11209.68 | 9712.15 |
| Prefill | BS=32, L=4096 | 10081.92 | 10452.48 | 9027.82 |
| Decode | BS=1, L=4 | 60.40 | 79.53 | 80.59 |
| Decode | BS=256, L=4 | 7686.51 | 9404.59 | 7010.33 |

*分析要点：* **Prefill** 上 BF16 在多数配置下略逊于或接近 W8A8，W8A16 普遍低于 BF16（大块 GEMM 上 BF16 Tensor Core 主路径仍很成熟）。**Decode** 上 W8A8 在中高 batch 下相对 BF16 提升明显（权重更省带宽）；W8A16 在 BS=1 略优，在 BS=256 时低于 BF16，与实现路径和 batch 形态有关。完整矩阵见上述 `docs/pd_separate_report_*.md`。

### WikiText-2 困惑度（scripts/eval/eval_ppl_wikitext.py / scripts/eval/eval_ppl_nano_fp8.py）

**测试配置**

- Dataset: wikitext-2-raw-v1，split=test
- 滑动窗口：max_length=2048，stride=512（与 `docs/wikitext.md` 中记录一致）
- FP8 侧：鉴于 HuggingFace 原生框架不支持 FP8 算子，采用反量化代理评估（De-quantization Proxy），即将 FP8 权重还原为 BF16 后进行推理。该指标反映了 **权重量化（Weight Quantization）** 带来的精度损耗上限

**测试结果**

| 模型 | WikiText-2 (test) perplexity |
|------|------------------------------|
| BF16 基线 | ≈ 6.4008 |
| FP8 W8A8（反量化评估） | ≈ 6.4207 |
| FP8 W8A16（反量化评估） | ≈ 6.4208 |

*分析要点：* 三者困惑度差距在 **约 0.02（相对变化不足 0.4%）** 量级，说明当前 FP8 导出在语言建模指标上与 BF16 非常接近；W8A8 与 W8A16 的反量化 PPL 几乎相同，主要反映**权重量化**带来的差异，与「W8A8 激活量化未在 HF 评估中完全复现」的设定一致。
