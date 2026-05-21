# Quantization Scripts

FP8 W8A16/W8A8 权重量化和评测脚本。该目录关注 dense/linear 量化，不包含 KV cache 专项。

## Script Map

| 脚本 | 作用 | 输出重点 |
| --- | --- | --- |
| `quantize.py` | 离线导出 FP8 W8A16 / W8A8 static checkpoint | qweight、weight_scale、input_scale contract |
| `inspect_checkpoint.py` | 检查量化 checkpoint 健康度 | tensor 数量、scale 匹配、config 字段 |
| `bench_quant.py` | nano runtime 量化吞吐测试 | prefill TPS、decode TPS、e2e TPS |
| `memory_quant.py` | 显存阶段记录 | load/generate 前后 allocated/reserved |
| `eval_ppl_quant.py` | 量化 PPL proxy | WikiText/text PPL，质量损失 |
| `compare_logits.py` | BF16 vs quant logits/token 对齐 | cosine、max abs、top-k overlap、token match |
| `w8a16_linear_microbench.py` | W8A16 linear kernel microbench | dequant matmul、cached BF16 matmul、Triton on-the-fly kernel |
| `fp8_linear_microbench.py` | W8A8 linear 分桶 microbench | torch baseline、PTX small-M、CUTLASS large-M、auto policy |
| `probe_scaled_mm.py` | PyTorch `_scaled_mm` capability probe | 仅作对照路径的 SM / dtype / scale contract 可用性 |
| `run_quant_suite.py` | 一键运行 inspect/bench/memory/PPL/compare | suite 状态码和统一输出目录 |
| `run_4090_7b_stack.py` | RTX 4090 上的 7B BF16/W8A16/FP8-KV/W8A8 分阶段 suite | 7B 主线结果目录、manifest、各阶段状态码 |
| `summarize_quant_runs.py` | 汇总 JSON/JSONL 结果 | 表格/CSV 汇总 |
| `common.py` / `workloads.py` / `ppl_*.py` | 公共工具 | 元信息、负载构造、PPL adapter |

## Typical Commands

```bash
python scripts/quantization/quantize.py \
  --model-path /path/to/bf16-model \
  --output-path /path/to/fp8-model \
  --scheme fp8_w8a8_static

python scripts/quantization/run_quant_suite.py \
  --model-path /path/to/fp8-model \
  --baseline-model-path /path/to/bf16-model \
  --output-dir .remote-logs/quantization/run

python scripts/quantization/run_4090_7b_stack.py \
  --bf16-model-path /path/to/Qwen2.5-7B-Instruct \
  --w8a16-model-path /path/to/Qwen2.5-7B-Instruct-FP8-W8A16 \
  --stages bf16,w8a16,kv \
  --output-dir .remote-logs/quantization/4090_7b_stack

NANOVLLM_W8A8_JIT_KERNEL=ptx_mma \
NANOVLLM_W8A8_JIT_B_LAYOUT=cutlass \
python scripts/quantization/fp8_linear_microbench.py \
  --w8a8-model-path /path/to/Qwen2.5-7B-Instruct-FP8-W8A8-Static \
  --weight-name model.layers.0.self_attn.q_proj \
  --weight-name model.layers.0.mlp.gate_proj \
  --m 16 \
  --backend torch \
  --backend ptx \
  --backend cutlass \
  --backend auto \
  --output-json .remote-logs/w8a8_buckets/run.json
```

## Notes

- `bench_quant.py` / `memory_quant.py` 走 nano runtime，反映真实推理路径。
- `eval_ppl_quant.py` 对部分 FP8 格式使用临时 BF16 dequantized HF proxy，只反映权重量化质量，不代表 nano runtime 性能。
- W8A8 当前最重要的专项证据来自 `fp8_linear_microbench.py`。
- W8A8 runtime baseline 必须比较 full path：`torch` 使用 torch activation quant + `torch._scaled_mm`，`ptx` 使用 inline PTX MMA，`cutlass` 使用 CUTLASS FP8 GEMM，`auto` 按 M bucket 在 PTX/CUTLASS 间选择。
- 当前 PTX inline MMA 使用 CUTLASS SM89 `BLayout` 对齐单 warp `m16n8k32` operand；bucket benchmark 应覆盖 `M=1/2/4/8/16/32/64/128/256/512/1024` 和 q/gate/up/down 真实 7B shape。
