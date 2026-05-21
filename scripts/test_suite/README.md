# nano-vllm-moe Test Suite

本目录包含针对 nano-vllm-moe 各项优化的综合测试脚本。

## 测试矩阵

| 优化类型 | 精度测试 | 效率测试 | 显存测试 |
|---------|---------|---------|---------|
| BF16 baseline | PPL, logits cosine | TPS, latency | peak memory |
| W8A16 | PPL vs BF16, cosine | TPS vs BF16 | model size reduction |
| W8A8 | PPL, MMLU, cosine | TPS vs BF16 | model size |
| FP8 KV cache | token match | decode TPS | KV bytes/block |
| K-int8/V-FP8 KV | token match (8K/16K/32K) | decode TPS | KV compression ratio |
| Quantized Attention | output cosine | attention latency | N/A |
| MoE CUDA Graph | output match | decode TPS | N/A |

## 测试脚本

### comprehensive_test.py
综合测试入口，支持配置不同测试类型。

```bash
# BF16 baseline
python comprehensive_test.py --model-path /path/to/bf16_model --quantization-type bf16

# W8A16
python comprehensive_test.py --model-path /path/to/w8a16_model --quantization-type w8a16

# K-int8/V-FP8 KV cache
python comprehensive_test.py --model-path /path/to/w8a16_model --quantization-type w8a16 --kv-cache-dtype k_int8_v_fp8
```

### 测试类型

1. **精度测试 (accuracy)**
   - PPL (WikiText-2 validation)
   - logits cosine similarity vs BF16
   - token match rate
   - MMLU/CEval choice-logit accuracy

2. **效率测试 (efficiency)**
   - prefill TPS
   - decode TPS
   - wall time
   - latency (TTFT, TPOT)

3. **显存测试 (memory)**
   - model size
   - peak allocated memory
   - KV bytes/block
   - KV compression ratio

## 通过标准

根据 plan.md 定义，各优化需要满足以下条件：

### W8A16
- [ ] 3B/7B smoke 通过
- [ ] PPL delta < 0.1 vs BF16
- [ ] logits cosine > 0.99
- [ ] decode TPS >= BF16 或有明确显存/速度 tradeoff

### W8A8
- [ ] WikiText PPL vs BF16 delta < 0.5
- [ ] MMLU accuracy > 0.95 * BF16
- [ ] 大 Linear shape TPS > BF16
- [ ] generation token match > 0.95

### K-int8/V-FP8 KV
- [ ] 8K/16K/32K prompt token exact match
- [ ] KV bytes/block < 55% of BF16
- [ ] decode TPS > BF16

### MoE CUDA Graph
- [ ] decode TPS 提升
- [ ] output match with eager mode
