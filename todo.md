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

1. Debug 7B FP8 KV logits/token divergence：已定位为 K cache FP8 量化导致；V-only FP8 在 512/8K prompt 上 token exact match，当前安全方向改为 K-BF16/V-FP8，完整 K+V FP8 继续实验。
2. 补 7B FP8-only 32K capacity 或 chunked-prefill capacity 脚本，避免 BF16 reference 先 OOM 干扰容量结论。
3. 优化 7B W8A16 SM89 runtime，避免 per-forward BF16 dequant matmul 拖慢 decode；候选是 SM89 Triton weight-only kernel 或一次性 load-time dequant speed baseline。
4. 在 W8A16 + FP8 KV 有稳定质量 gate 后，再进入 W8A8 static 全链路；先跑 representative linear shapes 的 activation-quant + scaled-mm breakdown。
5. MoE 只做维护性 correctness/profiler，不再阻塞量化主线。
6. EP 只保留 baseline，直到量化主线完成后再决定是否重启。

## Latest Validation Snapshot

- 7B BF16/W8A16 远端验证：`.remote-logs/quantization/4090_7b_stack_20260513/`。
- W8A16 checkpoint size `8.72 GB` vs BF16 `15.24 GB`；PPL proxy `7.4772` vs `7.4367`。
- 当前 W8A16 decode 慢于 BF16，原因是 SM89 路径仍走 per-forward BF16 dequant matmul。
- W8A16 + FP8 KV 在 8K/16K 有明确 KV storage 收益，但 token match 只有 `1/16` 和 `2/16`，质量未过 gate。
- 2026-05-14 路线复盘已同步到 `opt/4090_quant_stack_opt.md`：当前不要直接跳 W8A8；先修 FP8 KV 质量，再处理 SM89 W8A16 速度，最后做 W8A8 profile-driven promotion。
- 2026-05-14 FP8 KV backend isolation：0.5B smoke 通过；7B W8A16 `input_len=512/output_len=8` 上 float16 scale 的 native/gather/full vs BF16 token match 都只有 `0.25`，gather/full 彼此 `max_abs=0.0`；float32 scale 不改善。证据在 `.remote-logs/kv_debug_20260514/`。

- 2026-05-14 K/V sensitivity：fake K-only FP8 复现 7B 发散（token match `0.25`），fake V-only FP8 保持 exact match；真实 `fp8_v_only` 在 512 和 8K prompt 上 exact match，KV bytes/block 为 BF16 的 `75.39%`，KV block count 约 `1.33x`。证据在 `.remote-logs/kv_debug_20260514/7b_v_fp8_*.json`。

- 2026-05-14 fp8_v_only suite 接入：`run_4090_7b_stack.py --stages kv` 默认使用 `--kv-cache-dtype fp8_v_only`；8K/16K token exact match，per-block KV bytes 为 BF16 的 `75.39%`。
- 2026-05-14 K quant probe：FP8 K 仍不稳，E4M3 group scale 最好也只有 token match `0.625`，E5M2 vector 为 `0.75`；int8 K vector/group 16/32/64 全部 token exact match，其中 group 16/32 logits 最稳。
- 2026-05-14 K-int8/V-FP8 hybrid：真实 `kv_cache_dtype="k_int8_v_fp8"` 已实现，K 用 int8 group32 scale、V 用 FP8；512/8K/16K prompt 全部 token exact match，per-block KV bytes 为 BF16 的 `51.95%`。
- 2026-05-14 mixed KV 优化：修正容量估算和 accuracy suite；新增 Triton fused store 与 native mixed paged decode。7B W8A16 8K gate 保持 exact match，native mixed decode `14.64` model TPS vs BF16 `4.34`、gather mixed `4.43`，per-block KV bytes `51.95%`，block ratio `1.56x`。512 短上下文 native 仍慢，下一步做 16K/32K gate 与短上下文阈值/融合优化。证据在 `.remote-logs/kv_mixed_opt_20260514/`。
- 2026-05-14 native mixed KV 16K/32K：8K/16K/32K 均 exact match；native mixed model TPS 分别为 `14.64/10.77/6.43`，对 BF16 为 `3.37x/2.62x/1.59x`；bytes/block 保持 `51.95%`，block ratio 约 `1.53x-1.57x`。完整数据保存到 `.remote-logs/kv_mixed_opt_20260514/native_8k16k32k_summary.{json,csv}`，README 已写入展示表。下一步是更多 seeds 与 native kernel 按 token-block split/reduction 优化。
