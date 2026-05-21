# Nano-vLLM-MoE

本项目基于 Nano-vLLM，专注在单卡（RTX 4090 24GB 级别）环境下验证三条推理优化路线：FP8 权重/激活量化、KV cache 压缩与前缀复用、MoE 调度与 CUDA Graph。代码刻意保持小而可读，每条优化都配 microbench 和质量 gate，结论可以追溯到具体远端日志。原项目地址：https://github.com/GeeeekExplorer/nano-vllm.git

## 主要特点

* FP8 W8A16 量化 — 基于 Triton 的 on-the-fly dequant GEMM，权重体积 `-44.8%`，decode TPS `1.21x`
* FP8 W8A8 静态量化 — Torch `_scaled_mm` / 自定义 PTX / in-repo CUTLASS 三后端，checkpoint `-42.8%`，WikiText PPL 漂移仅 `+0.0267`
* K-int8 / V-FP8 混合 KV — 长上下文显存压力路径，bytes/block 约 BF16 的 `0.52x`，teacher-forced PPL 全长度 `< +1.55%`
* 模块化 MoE Runtime — `router → prepare/finalize → expert → finalize` 四段拆分，`eager` 与 `optimized` 双后端可切换
* MoE CUDA Graph — 单卡 (`ep_size=1`) 默认开启，decode `4.55x` vs eager 后端，`128/128` greedy-token 一致
* Hash + Radix Prefix Cache — block-level metadata 双后端，shared-prefix 命中率 `87.5%`
* Chunked Prefill — `decode_first` 调度策略，32K 长 prefill 干扰下短 decode 卡顿从 `4197 ms` 压到 `57 ms`
* Quantization Suite — 完整 PPL / MMLU logit-rank / linear backend microbench 质量门

## 安装

```bash
git clone https://github.com/ZeroKernel798/nano-vllm-moe.git
cd nano-vllm-moe
pip install -e .
```

## Quick Start

```bash
python scripts/moe/moe_local_compute_bench.py --device cuda --backends eager,optimized
python scripts/prefix_cache/prefix_cache_bench.py --model /path/to/model --scenario shared-prefix
python scripts/generation/chunked_prefill_bench.py --model-path /path/to/model --phases 1,3
```

## Benchmark

### FP8 W8A16 权重量化

**测试配置**

- Hardware: NVIDIA GeForce RTX 4090 24GB
- Model: Qwen2.5-7B-Instruct（BF16 基线 / W8A16 静态导出）
- Parallelism: TP=1, EP=1
- 量化导出: `scripts/quantization/quantize.py --scheme fp8_w8a16`，量化 `252/434` 个 tensor
- Decode 场景: 输入 512，输出 128 / 16，warmup 1，repeat 3

**测试结果**

| 方案 | Model size bytes | Decode TPS (out=16) | Decode TPS (out=128) | 相对 BF16 |
|------|------:|------:|------:|------:|
| BF16 基线 | `6,183,464,346` | `130.69 tok/s` | — | `1.00x` |
| FP8 W8A16 | `3,413,055,132` | `158.36 tok/s` | `158.73 tok/s` | `1.21x` |

**分析要点**：W8A16 是最简单的权重量化路径，权重体积从 `6.18 GB` 降到 `3.41 GB`（`-44.8%`）；Triton 的 on-the-fly dequant 在 decode 阶段把权重带宽压力换成计算，512 prompt 上 decode `1.21x`。CUDA reserved/allocated 端到端差异小于 checkpoint 差异，所以只把它定位为 "weight-only 省占用 + decode 改善"，不延伸到整卡显存减半。

---

### FP8 W8A8 静态量化

**测试配置**

- Hardware: NVIDIA GeForce RTX 4090 24GB
- Model: Qwen2.5 系列（BF16 基线 / W8A8 静态导出，激活 scale 离线校准）
- Parallelism: TP=1, EP=1
- 量化导出: `scripts/quantization/quantize.py --scheme fp8_w8a8_static`
- Linear backend: Torch `_scaled_mm` / 自定义 PTX / in-repo CUTLASS（自定义 epilogue）
- 质量评测: WikiText-2 PPL（1024 tokens）+ MMLU 120 题 logit-rank
- E2E throughput 跑在 3B-scale config（`hidden_size=2048`, `intermediate_size=11008`, `36` layers），真 7B microbench 单独提供

**质量 gate（7B）**

| Gate | BF16 | W8A8 | Delta |
|------|------:|------:|------:|
| Checkpoint size | `15.24 GB` | `8.72 GB` | `-42.8%` |
| WikiText-2 PPL, 1024 tokens | `8.3971` | `8.4238` | `+0.0267` |
| MMLU logit-rank, 120 题 | `80.00%` | `79.17%` | `-0.83 pp` |
| MMLU prediction agreement | reference | `96.67%` | `4 / 120` flips |

**Linear backend microbench（3B-scale）**

| Layer | Shape `(M,K,N)` | BF16 | Torch | PTX | CUTLASS | Winner |
|------|---|------:|------:|------:|------:|------|
| `self_attn.q_proj` | `1,2048,2048` | `0.0095 ms` | `0.0495 ms` | `0.0284 ms` | `0.0121 ms` | CUTLASS |
| `mlp.up_proj` | `16,2048,11008` | `0.0301 ms` | `0.0486 ms` | `0.0304 ms` | `0.0112 ms` | CUTLASS |
| `mlp.down_proj` | `256,11008,2048` | `0.0927 ms` | `0.0793 ms` | `0.1983 ms` | `0.0834 ms` | Torch |
| `mlp.up_proj` | `1024,2048,11008` | `0.3252 ms` | `0.1864 ms` | `0.7310 ms` | `0.1853 ms` | CUTLASS |

**3B-scale E2E throughput（同 W8A8 CUTLASS vs native BF16，普通 serving 路径）**

| 阶段 | Input | Output | BF16 total tok/s | W8A8 total tok/s | W8A8/BF16 |
|------|------:|------:|------:|------:|------:|
| mixed | 1024 | 128 | `1154.21` | `1785.43` | `1.55x` |
| mixed | 4096 | 128 | `3674.52` | `5608.63` | `1.53x` |
| mixed | 16384 | 128 | `8541.24` | `11494.63` | `1.35x` |
| mixed | 32640 | 128 | `9776.19` | `11788.54` | `1.21x` |

**分析要点**：W8A8 在显著缩小 checkpoint 的同时保持质量门通过——PPL 漂移仅 `+0.0267`，MMLU 仅降 `0.83 pp`。Linear backend 测试显示 CUTLASS 在测试覆盖的 M 上整体更稳，原先 `M<=16` 自动走 PTX 的策略已暂停校准；Torch `_scaled_mm` 仍是必须保留的中等 M 参考。3B-scale 上 W8A8 CUTLASS 在 1K..16K 普通 serving 场景中 total throughput 提升 `1.35x..1.55x`。

> **当前边界**：W8A8 端到端 Triton 在 SM89 仍触发 compiler abort；fused Triton W8A8 路径暂未启用。真 7B BF16 与配对 W8A8 export 已完成但 E2E 仍待补齐。

---

### K-int8 / V-FP8 混合 KV Cache

**测试配置**

- Hardware: NVIDIA GeForce RTX 4090 24GB
- Model: Qwen2.5-7B-Instruct BF16
- 量化口径: K int8 + V FP8，K group=32，scale dtype float16
- 质量 gate: WikiText teacher-forced PPL/NLL（不再以 token exact alignment 为主 gate）
- 显存测试: 固定 512 KV blocks 同容量对照 + 8K/64 generation peak

**Teacher-forced PPL（K group=32）**

| Context | BF16 PPL | Mixed KV PPL | Delta | bytes/block |
|------:|------:|------:|------:|------:|
| 4K | reference | — | `+1.20%` | — |
| 8K | reference | — | `+1.20%` | — |
| 16K | reference | — | `+1.55%` | — |
| 32K | reference | — | `-0.16%` | `0.5195x` BF16 |

**显存收益（7B，固定 512 KV blocks）**

| 指标 | BF16 KV | Mixed KV | Delta |
|------|------:|------:|------:|
| KV arena | `7.00 GiB` | `3.64 GiB` | `-3.36 GiB` |
| 8K/64 generation peak allocated | `22.41 GiB` | `19.04 GiB` | `-3.37 GiB` |

**分析要点**：K-int8/V-FP8 在 PPL 主 gate 上全长度通过（最差 `+1.55%`，32K 甚至略低于 BF16），同时 bytes/block 减半。同容量 KV blocks 下 KV arena 从 `7.00 GiB` 降到 `3.64 GiB`。Full FP8 KV（K 也用 FP8）在 4K/8K PPL smoke 上发散到 `1509.77`，所以主线只保留 K-int8/V-FP8 混合路径。Token exact alignment 在多 seed 检查中暴露 K group-size 敏感（32K seed1 只在 group-1 exact），因此 BF16 KV 仍是质量 reference。

---

### MoE Runtime（router → prepare/finalize → expert → finalize）

**测试配置**

- Hardware: NVIDIA GeForce RTX 4080 SUPER 16GB，driver `580.142`，Python 3.12.3
- Model: Qwen1.5-MoE-A2.7B-Chat
- Parallelism: TP=1, EP=1
- Workload: 8 固定 prompts，input 16 / output 16，max model length 128，repeat 3

**Local Compute（synthetic，8 expert / top-1 / 256 tokens）**

| 后端 | Mean latency | Tokens/s | Correctness vs eager |
|------|------:|------:|------|
| `eager` | `1.974 ms` | `129.7K` | reference |
| `optimized` | `0.945 ms` | `270.8K` | cosine `0.9999999` |

**End-to-end 三模式对比**

| 模式 | Wall mean | Prefill TPS | Decode TPS | Decode 加速 | Greedy match |
|------|------:|------:|------:|------:|------|
| `eager` 后端（Python per-token expert loop） | `3.268 s` | `376.14` | `40.98` | `1.00x` | reference |
| `optimized` 后端，`--enforce-eager`（Triton fused MoE） | `1.142 s` | `1213.93` | `115.81` | `2.83x` | exact |
| `optimized` 后端，CUDA Graph | `0.747 s` | `1242.94` | `186.45` | `4.55x` | `128/128` tokens |

**分析要点**：`optimized` 后端通过 Triton fused MoE kernel 把 expert 计算从 Python per-token 循环搬到 GPU，已经带来 `2.83x` 的 decode 提升。在此之上启用 CUDA Graph 又叠加 `1.61x`，把 `torch.bincount` 的 host-sync、side-stream overlap、动态 workspace 分配等 capture 不安全的点逐个修掉之后，单卡 `ep_size=1` 上可以稳定 capture/replay 并且与 eager 输出完全一致（`128/128`）。

> **当前边界**：EP all-to-all 因动态 collective 仍强制 eager；fused 后端已移除，`optimized` 是唯一非 eager 后端。

---

### Prefix Cache（hash + radix）

**测试配置**

- Hardware: NVIDIA GeForce RTX 4090 24GB
- Model: Qwen2.5-3B-Instruct
- Workload: 8 请求，shared prefix `1024` tokens，unique `128` tokens，output `8`，block size `256`
- Backend: `hash`（默认）/ `radix`（block-level radix metadata）

**测试结果**

| Backend | Scenario | Token Hit Rate | Total Time | Prefill TPS | Decode TPS |
|---------|----------|------:|------:|------:|------:|
| hash | no-reuse | `0.0%` | `2.367 s` | `4288.8` | `259.0` |
| hash | shared-prefix | `87.5%` | `1.540 s` | `1551.4` | `257.1` |
| hash | partial-shared | `43.8%` | `1.637 s` | `3981.1` | `253.6` |
| radix | no-reuse | `0.0%` | `1.750 s` | `6013.4` | `259.1` |
| radix | shared-prefix | `87.5%` | `1.484 s` | `1621.4` | `256.1` |
| radix | partial-shared | `43.8%` | `1.614 s` | `4030.2` | `261.1` |

**分析要点**：shared-prefix 命中率按预期达到 `7/8 = 87.5%`，partial-shared 达到 `43.8%`。hash 和 radix 在直接 shared-prefix 测试上命中率一致；hash 是默认高效实现，radix 作为已接入的 block-level metadata backend 保留，主要为后续 branching-prefix workload 服务。

---

### Chunked Prefill（decode_first 调度）

**测试配置**

- Hardware: NVIDIA GeForce RTX 4090 24GB
- Model: Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct
- Workload: 请求 A 短 prompt `32` tokens / 输出 `128` tokens；请求 B 长 prompt `2K..32K` / 输出 `1`，在 A 生成 8 个 token 后插入
- 关注指标: A 的最大 inter-token latency（越低越好）和 B 的 TTFT

**7B 结果（max inter-token latency on request A）**

| B prompt | no chunk | prefill_first 128 | decode_first 128 | decode_first 1024 (A max / B TTFT) |
|------:|------:|------:|------:|---:|
| 2K | `2255.0 ms` | `473.2 ms` | `52.7 ms` | `120.4 / 238.7 ms` |
| 8K | `778.2 ms` | `1774.7 ms` | `52.0 ms` | `140.3 / 1032.5 ms` |
| 16K | `1739.6 ms` | `3868.8 ms` | `57.1 ms` | `165.9 / 2268.1 ms` |
| 32K | `4197.3 ms` | `9221.8 ms` | `57.2 ms` | `219.2 / 5375.0 ms` |

**3B 结果（max inter-token latency on request A）**

| B prompt | no chunk | prefill_first 128 | decode_first 128 | decode_first 1024 (A max / B TTFT) |
|------:|------:|------:|------:|---:|
| 2K | `2640.9 ms` | `572.2 ms` | `61.8 ms` | `77.6 / 153.6 ms` |
| 16K | `888.0 ms` | `4419.1 ms` | `65.4 ms` | `103.3 / 1418.1 ms` |
| 32K | `2273.4 ms` | `9069.3 ms` | `63.6 ms` | `136.6 / 3344.0 ms` |

**分析要点**：`decode_first` 是降低短请求 decode 卡顿的关键。chunk `128` 在全部 B 长度上把 A 的最大 inter-token latency 压到约 `60 ms`，32K 长 prefill 干扰下 7B 从 `4197 ms` 降到 `57 ms`。`prefill_first` 即使拆 chunk 也会连续服务 B 的 prefill，长 B 下反而把 A 卡顿拉到秒级。chunk 越大 B 的 TTFT 越低、A 的卡顿越高，`512/1024` 适合更偏吞吐的场景，`128` 适合 decode 响应性优先。

---

## 当前边界

- W8A8 端到端 Triton generation 在 SM89 当前路径上仍触发 compiler abort，因此 README 数据使用 HF 质量 gate 和明确的 CUDA/PTX microbench。
- 7B W8A8 linear backend 中 in-repo CUTLASS 是稳定的全 M 对照路线，旧的 PTX small-M auto threshold 已暂停，等补完被中断的 `scalar_m32` 和 `auto_7b` 检查后再校准。
- MoE CUDA Graph 在单卡 `ep_size=1` 默认开启；EP all-to-all 因动态 collective 不可 capture 仍强制 eager。
- Chunked prefill 的 `prefill_first` 不是短 decode 流的延迟优化——32K 插入 prefill + chunk `128` 时，最大 A pause 在 3B/7B 分别约 `9069 / 9222 ms`。
- K-int8/V-FP8 KV 主质量 gate 已切换为 teacher-forced NLL/PPL；strict token exact alignment 在 16K/32K 多 seed 检查中暴露 K group-size 敏感，BF16 KV 仍是质量 reference。
- 远端验证环境没有可用 `vllm` 包，README 不展示 vLLM baseline 对比。

## 仓库结构

| Path | 用途 |
|------|------|
| `nanovllm/engine/` | scheduler、sequence、model runner、block manager、radix tree |
| `nanovllm/executor/moe/` | 模块化 MoE runtime（router / prepare-finalize / experts / blocks） |
| `nanovllm/quantization/` | FP8 runtime、CUDA/PTX JIT、CUTLASS extension、quantization registry |
| `nanovllm/models/` | Llama / Qwen2 / Qwen2-MoE / Qwen3 / Qwen3-MoE |
| `scripts/generation/` | generation 与 chunked prefill benchmark |
| `scripts/kv_cache/` | FP8 / 混合 KV 验证脚本 |
| `scripts/moe/` | MoE local compute 与 backend benchmark |
| `scripts/prefix_cache/` | hash / radix prefix cache benchmark |
| `scripts/quantization/` | FP8 export、质量 gate 与 microbench |
| `opt/` | 各优化方向的设计与实验记录 |
| `docs/benchmarks.md` | 提取后的完整 benchmark 结果 |
