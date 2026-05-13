# nano-vllm-moe Roadmap

本轮目标是让项目从早期探索状态收敛为清爽的重构主线。当前只保留：MoE、chunked prefill、量化；EP 只保留基础 torch all-to-all 语义路径。

## Active Tracks

| Track | Status | Next Gate |
| --- | --- | --- |
| MoE runtime | 阶段性稳定 | 保持 `eager/optimized/fused` 三 backend，补最小 correctness/profiler 即可 |
| Chunked prefill | 保留主线功能 | 维护 `prefill_first/decode_first` 两种 policy，避免继续堆旧实验表 |
| 4090 7B quantization | 当前主线 | BF16 -> W8A16 -> FP8 KV -> W8A8 |
| EP | 基础保留 | 只保留 `torch` all-to-all baseline，不再携带旧原型 |

## Cleanup Decisions

- 保留 `docs/README_legacy.md` 作为历史归档。
- 删除旧 benchmark docs、旧 opt 长日志、旧 remote logs、SageAttention 探索、trace replay/xpu-perf 计划、旧 EP 原型脚本。
- `opt/` 只保留当前重构说明和 4090 量化主线。
- README 只写重构后的主线和当前可信结果，不再堆历史矩阵。

## Current Priority

1. Debug 7B FP8 KV logits/token divergence。
2. 优化 7B W8A16 SM89 runtime，避免 per-forward BF16 dequant matmul 拖慢 decode。
3. 在 W8A16 和 FP8 KV 稳定后，再进入 W8A8。
4. MoE 只做维护性 correctness/profiler，不再阻塞量化主线。
5. EP 只保留 baseline，直到量化主线完成后再决定是否重启。

## Latest Validation Snapshot

- 7B BF16/W8A16 远端验证：`.remote-logs/quantization/4090_7b_stack_20260513/`。
- W8A16 checkpoint size `8.72 GB` vs BF16 `15.24 GB`；PPL proxy `7.4772` vs `7.4367`。
- 当前 W8A16 decode 慢于 BF16，原因是 SM89 路径仍走 per-forward BF16 dequant matmul。
- W8A16 + FP8 KV 在 8K/16K 有明确 KV storage 收益，但 token match 只有 `1/16` 和 `2/16`，质量未过 gate。
