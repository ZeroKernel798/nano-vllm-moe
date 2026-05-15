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

1. 继续 W8A8 shape-aware runtime：3B/7B shape profile、短 smoke、同源 WikiText PPL、MMLU 300 logit-rank 和 GSM8K 50 numeric probe 已完成；下一步扩大 MMLU/CEval 覆盖，再决定是否作为最终 7B quality stage 推广。
2. Broaden native K-int8/V-FP8 gates：8K/16K/32K 已 exact match，下一步加更多 seeds/prompts，并做短上下文 backend cutoff。
3. 优化 7B W8A16 SM89 runtime：已修复 Triton kernel 的 SM90-only FP8->BF16 cast 问题，当前改为 FP8->FP32->FP16 on-the-fly dequant，不使用 load-time cache；下一步优化 tile/转换开销以改善 prefill。
4. 补 7B FP8-only 32K capacity 或 chunked-prefill capacity 脚本，避免 BF16 reference 先 OOM 干扰容量结论。
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
- 2026-05-14 W8A8 shape-aware restart：3B W8A8 microbench 显示 `mlp.gate/up_proj` full W8A8 约为 BF16 `0.49x-0.51x`、W8A16 dequant `0.28x-0.29x`，`down_proj` 约为 BF16 `0.93x`；2K `q_proj/o_proj` 仍需回退。默认 W8A8 activation quant 改为 Triton，`NANOVLLM_FP8_W8A8_SCALED_MM_MIN_DIM=3072` 控制 shape cutoff；3B smoke 通过，shape-aware bench prefill TPS `10047.6` vs all-scaled-mm `9277.6`，3B logits gate avg cosine `0.998857` 且 generation exact match `1.0`。7B profile 显示 `gate/up/down/q/o` W8A8-vs-BF16 分别约 `0.49x/0.49x/0.78x/0.88x/0.87x`，7B 512/16 smoke 通过，prefill TPS `8775.6`、decode TPS `61.48`、model size `8.72 GB`；7B HF proxy compare 因同时加载 BF16+W8A8 在 24GB 上 OOM。证据在 `.remote-logs/w8a8_opt_20260514/`。
- 2026-05-15 W8A8 memory-safe quality gate：确认 7B W8A8 checkpoint calibration 使用 WikiText-2 validation、cache_dir `/home/ubuntu/project/datasets`、samples `32`、max_length `256`。新增 `compare_logits.py --sequential-load` 和 `run_quant_suite.py --sequential-compare`，避免 BF16+W8A8 双模型同驻。远端同源 WikiText PPL：BF16 `8.3971`、W8A8 HF proxy `8.4238`（1024 tokens）；sequential logits/token gate：avg cosine `0.999505`、top1 match `0.667`、top10 overlap `0.967`、generation exact match `0.667`、token match ratio `0.75`。证据在 `.remote-logs/w8a8_opt_20260515/`。
- 2026-05-15 MMLU/GSM8K gate：新增 `eval_choice_logits.py` 做 memory-safe 多选题 logit-rank gate，MMLU 300 题（6 科目各 50）BF16 accuracy `0.6033`、W8A8 `0.6200`、prediction agreement `0.9567`、flips `13/300`；MMLU 100 题 smoke 中两者 accuracy 均 `0.68`、agreement `0.96`。新增 `eval_gsm8k_generate.py` 做 GSM8K greedy numeric probe，50 题 BF16 `0.32`、W8A8 `0.36`，但 same-number agreement `0.28`、flips `36/50`，说明 GSM8K 对生成路径蝴蝶效应更敏感，适合作补充探针。证据在 `.remote-logs/w8a8_opt_20260515/7b_w8a8_mmlu_*.json` 和 `.remote-logs/w8a8_opt_20260515/7b_w8a8_gsm8k_50.json`。
- 2026-05-15 W8A16/W8A8 性能瓶颈复查：远端 A/B 在 `.remote-logs/w8_perf_opt_20260515/`。3B W8A16 load-time dequant 将 512/16 wall time 从 `0.4687s` 降到 `0.1679s`，但 7B full cache 在 24GB OOM；按新要求不采用提前解压作为优化方向。W8A8 fallback-cache 在 3B 从 `0.2350s` 降到 `0.2062s`，但仍需 7B memory gate 后才能考虑默认化。
- 2026-05-15 SM89 W8A16 Triton 修复：把 kernel 内权重反量化从 direct FP8->BF16 改为 FP8->FP32->FP16，并把 activation 也转 FP16 后做 `tl.dot`，避免 SM90-only 指令。3B Triton 512/16 在 `.remote-logs/w8a16_triton_sm89_20260515/` 编译通过，wall time `0.8247s` vs 同设置 per-forward dequant `1.1368s`，decode TPS `143.90` vs `37.76`；7B Triton smoke 也通过，wall time `0.9960s`、decode TPS `71.79`。当前 prefill 仍弱，下一步优化 tile 和转换开销。
