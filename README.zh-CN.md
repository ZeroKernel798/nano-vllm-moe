# Nano-vLLM-MoE

Nano-vLLM-MoE 是一个轻量 vLLM-style 推理实验项目。当前只保留三条主线：

1. **MoE runtime**：模块化 router、prepare/finalize 和 expert backend，用于本地与基础 EP 实验。
2. **Chunked prefill**：scheduler 和 sequence 支持 partial prefill，并保留 `prefill_first` / `decode_first` 两种策略。
3. **量化重构**：当前按 RTX 4090 / 7B 主线推进：BF16 baseline -> W8A16 -> FP8 KV cache -> W8A8。

EP 只保留基础 `torch` all-to-all 路径，不再携带早期实验原型。历史 README 仅保留在 `docs/README_legacy.md`，其它旧实验文档和过期 benchmark 计划已清理。

## 当前范围

| 方向 | 状态 | 主要文件 |
| --- | --- | --- |
| MoE runtime | 阶段性稳定 | `nanovllm/executor/moe/`, `scripts/moe/` |
| Chunked prefill | 当前保留功能 | `nanovllm/engine/scheduler.py`, `nanovllm/engine/sequence.py`, `scripts/generation/chunked_prefill_bench.py` |
| Quantization | 当前重构主线 | `nanovllm/quantization/`, `scripts/quantization/`, `scripts/kv_cache/` |
| EP | 只保留基础语义路径 | `nanovllm/executor/moe/prepare_finalize/torch_alltoall.py` |

## MoE Runtime

当前 MoE 路径：

```text
router -> prepare/finalize -> expert backend -> finalize output
```

Expert backends：

| Backend | 角色 | 文件 |
| --- | --- | --- |
| `eager` | correctness/reference | `nanovllm/executor/moe/experts/eager_experts.py` |
| `optimized` | 当前实际优化路径 | `nanovllm/executor/moe/experts/optimized.py` |
| `fused` | Triton grouped-GEMM 实验路径 | `nanovllm/executor/moe/experts/fused.py` |

EP 暂不作为性能收益声明，只保留 `torch` all-to-all baseline，保证分布式 MoE 语义可测。

## Chunked Prefill

Chunked prefill 是 scheduler-level 功能，用于把长 prefill 请求拆成多个小块调度。

| Policy | 行为 | 用途 |
| --- | --- | --- |
| `prefill_first` | 优先继续 prefill chunk | 简单 correctness/default 行为 |
| `decode_first` | prefill chunk 之间优先 decode | latency-control 实验 |

运行：

```bash
python scripts/generation/chunked_prefill_bench.py \
  --model-path /path/to/model \
  --max-model-len 0 \
  --phases 1,2,3
```

## RTX 4090 7B 量化栈

量化重构按 7B 优先推进。小模型仍可做 smoke，但 README 和主线结论以 Qwen2.5-7B on RTX 4090 为准。

| 阶段 | Mode | 目标 |
| --- | --- | --- |
| 0 | BF16 | 质量、延迟、显存 baseline |
| 1 | W8A16 | 稳定 weight-only FP8 checkpoint/runtime |
| 2 | W8A16 + FP8 KV | 长上下文显存压力和 KV 精度 |
| 3 | W8A8 | W8A16/KV 稳定后再做 aggressive activation quant |

### 最新 7B Baseline

测试设置：2026-05-13，1x RTX 4090 24GB，Qwen2.5-7B-Instruct，固定 synthetic prompt `input=512, output=64`，`max_model_len=2048`，`gpu_memory_utilization=0.9`，一次 warmup、一次计时。远端证据根目录：`.remote-logs/quantization/4090_7b_stack_20260513/`。

| Mode | Checkpoint size | PPL proxy | Bench prefill TPS | Bench decode TPS | Memory-run prefill TPS | Memory-run decode TPS | Peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 15.24 GB | 7.4367 | 10634.5 | 62.50 | 489.6 | 61.81 | 19.98 GB |
| W8A16 | 8.72 GB | 7.4772 | 5258.7 | 15.22 | 668.7 | 15.26 | 20.92 GB |

W8A16 checkpoint contract 健康（`196` 个 qweight tensor 和 `196` 个 weight-scale tensor）。当前 SM89 runtime 使用 per-forward BF16 dequant matmul，因此 W8A16 目前是显存/质量里程碑，还不是速度收益点。

### 最新 7B FP8 KV Probe

同一个 7B W8A16 checkpoint，`output=16`，native FP8 paged decode；脚本内同时跑 BF16 KV reference 和 FP8 KV。

| Prompt | KV mode | KV storage | Prefill TPS | Decode TPS | Wall time | Peak reserved | Token match |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | BF16 KV | 11.09 GB | 4224.7 | 8.48 | 3.71 s | 21.16 GB | reference |
| 8192 | FP8 KV | 2.90 GB | 7811.0 | 8.63 | 2.79 s | 21.17 GB | 1/16 |
| 16384 | BF16 KV | 10.01 GB | 5673.4 | 9.27 | 4.51 s | 21.31 GB | reference |
| 16384 | FP8 KV | 1.81 GB | 9623.6 | 10.35 | 3.15 s | 21.31 GB | 2/16 |

FP8 KV 显著降低 KV storage，但 token match 还不合格。下一步先做 logits divergence debug，不能直接把 FP8 KV 作为默认路径。

2026-05-14 backend isolation 更新：7B W8A16 在 `input=512, output=8` 下，`gather_dequant` 和 `full_dequant` 完全一致，但二者相对 BF16 的生成 token match 仍只有 `0.25`；KV scale cache 从 float16 改成 float32 也不能修复。因此下一步应重点查 FP8 KV storage quantization 本身，而不是只查 native paged attention kernel。

2026-05-14 K/V sensitivity 更新：7B 发散来自 K cache FP8 量化。新增实验性 K-BF16/V-FP8 模式，在 512-token 和 8K-token prompt 上都保持 exact token match，同时 KV bytes/block 降到 BF16 的 `75.39%`，因此当前安全方向是 V-only FP8。

K quantization probe：FP8 K 仍不稳定；fake symmetric int8 K 在 vector 或 16/32/64 group 下都能在 7B 512-token probe 上保持 exact tokens。下一步压缩方向应是 K-int8/V-FP8，而不是 K-FP8/V-FP8。

K-int8/V-FP8 mixed KV 已在 7B W8A16 的 512/8K/16K token-match gate 上通过，per-block KV bytes 降到 BF16 的 `51.95%`（含 scale）。当前优化路径使用 Triton fused mixed-KV store 和 native mixed paged decode；512 短上下文 decode 仍慢于 BF16，但长上下文 decode 已有明确收益。

| Prompt | Backend | Exact tokens | Logits cosine | Model TPS | 相对 BF16 加速 | KV bytes/block vs BF16 | KV blocks vs BF16 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | BF16 KV | reference | reference | 4.34 | 1.00x | 1.0000 | 1.00x |
| 8192 | K-int8/V-FP8 native | yes | 0.99989 | 14.64 | 3.37x | 0.5195 | 1.56x |
| 16384 | BF16 KV | reference | reference | 4.11 | 1.00x | 1.0000 | 1.00x |
| 16384 | K-int8/V-FP8 native | yes | 0.99987 | 10.77 | 2.62x | 0.5195 | 1.57x |
| 32512 | BF16 KV | reference | reference | 4.03 | 1.00x | 1.0000 | 1.00x |
| 32512 | K-int8/V-FP8 native | yes | 0.99986 | 6.43 | 1.59x | 0.5195 | 1.53x |

## 仓库结构

| Path | 用途 |
| --- | --- |
| `nanovllm/engine/` | scheduler、sequence、model runner、block manager |
| `nanovllm/executor/moe/` | 模块化 MoE runtime |
| `nanovllm/quantization/` | quantization method registry 和 FP8 runtime |
| `scripts/moe/` | MoE local compute、backend、基础 EP 脚本 |
| `scripts/generation/` | generation 和 chunked prefill benchmark |
| `scripts/quantization/` | FP8 export/eval/runtime benchmark suite |
| `scripts/kv_cache/` | FP8 KV 验证和 microbench |
| `opt/` | 仅保留当前重构记录 |
| `docs/README_legacy.md` | legacy README 归档 |

## Quick Start

```bash
pip install -e .
```

运行 MoE local compute：

```bash
python scripts/moe/moe_local_compute_bench.py --device cuda --backends eager,optimized,fused
```

运行 chunked prefill：

```bash
python scripts/generation/chunked_prefill_bench.py --model-path /path/to/model --max-model-len 0
```

运行 7B 量化栈：

```bash
python scripts/quantization/run_4090_7b_stack.py \
  --bf16-model-path /path/to/Qwen2.5-7B-Instruct \
  --w8a16-model-path /path/to/Qwen2.5-7B-Instruct-FP8-W8A16 \
  --stages bf16,w8a16,kv
```

## 下一步

1. 扩大 native K-int8/V-FP8 到 16K/32K 和更多 seeds，并增加短上下文 backend cutoff。
2. 优化或替换当前 SM89 W8A16 dequant-matmul runtime。
3. W8A16 和 FP8 KV gate 清楚后，再重新进入 W8A8。
