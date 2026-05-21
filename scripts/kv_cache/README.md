# KV Cache Scripts

当前脚本只维护 K-int8/V-FP8 混合 KV cache 路线。全 FP8、V-only FP8 和 fake quant 已经从代码主线移除，原因和历史路线记录在 `opt/kv_cache_quant.md`。

| Script | Purpose |
| --- | --- |
| `kv_cache_fp8_smoke.py` | BF16 KV vs K-int8/V-FP8 smoke，记录 token match 和 storage ratio |
| `kv_cache_fp8_logits.py` | 单 case logits tracing，对比 native/gather_dequant |
| `kv_cache_logits_dump.py` | 单 backend、单进程生成 logits trace，并用 `torch.save` 保存 `.pt` |
| `compare_logits_pt.py` | 离线 `torch.load` 多 backend `.pt`，做 allclose/cosine/top-k/token 对比 |
| `kv_cache_group_gate.py` | 多长度/seed/backend 的分进程 group-size quality gate |
| `kv_cache_fp8_accuracy_suite.py` | 多 seed/prompt accuracy suite |
| `kv_cache_concurrency_sweep.py` | BF16 KV vs K-int8/V-FP8 group-16 的多长度极限并发扫描 |

## 并发容量扫描计划

`kv_cache_concurrency_sweep.py` 默认按 `2K/4K/8K/16K/32K` 多档输入长度扫并发 `N`，分别比较 BF16 KV 和 `K-int8/V-FP8, K group=16`。每个 `(input_len, mode, N)` 独立进程运行并保存 JSON；summary CSV/JSON 记录每个长度下的 BF16 最大并发、mixed KV 最大并发、提升比例和失败原因。

并发扫描前必须先完成 `K group=16` 的 16K/32K 精度 gate。32K 精度和并发都按 total context budget 记录 requested/effective input length，确保容量提升结论不是通过牺牲模型最大长文本精度换来的。

32K 档按 total context budget 处理：如果模型 `max_position_embeddings=32768` 且需要生成 token，脚本应把实际 prompt length 调整为 `32768 - output_len`，并同时记录 requested/effective input length。

## Logits 对比方案

7B/8192 这类长上下文 case 不适合把 logits 序列化进 JSON：

- JSON 浮点文本会丢失不必要的细节，也很慢。
- logits tensor 很大，JSON 文件庞大且解析成本高。
- `kv_cache_fp8_logits.py` 在同一个 Python 进程内顺序加载 BF16/native/gather，7B/8192 下会受到 CUDA cache、编译缓存和碎片影响，容易在第二次或第三次建模时 OOM。

后续端到端 logits 排查改用分进程 `.pt` trace：

1. 每个 backend 单独运行 `kv_cache_logits_dump.py`，只负责一个 backend，并用 `torch.save` 保存 logits。
2. JSON 只写 metadata、tokens、storage ratio 和 `.pt` 路径，不保存 logits 本体。
3. 离线运行 `compare_logits_pt.py`，用 `torch.load` 读取 `.pt` 后做严格 `allclose`、cosine、max/mean abs、top-k overlap 和 token match。

示例：

```bash
python scripts/kv_cache/kv_cache_logits_dump.py \
  --model-path /path/to/model \
  --backend bf16 \
  --input-len 8192 \
  --output-len 8 \
  --max-model-len 9216 \
  --max-num-batched-tokens 9216 \
  --output-pt .remote-logs/kv_cache/bf16.pt \
  --output-json .remote-logs/kv_cache/bf16.json

python scripts/kv_cache/kv_cache_logits_dump.py \
  --model-path /path/to/model \
  --backend native \
  --input-len 8192 \
  --output-len 8 \
  --max-model-len 9216 \
  --max-num-batched-tokens 9216 \
  --gpu-memory-utilization 0.90 \
  --output-pt .remote-logs/kv_cache/native.pt \
  --output-json .remote-logs/kv_cache/native.json

python scripts/kv_cache/compare_logits_pt.py \
  --left-pt .remote-logs/kv_cache/bf16.pt \
  --right-pt .remote-logs/kv_cache/native.pt \
  --output-json .remote-logs/kv_cache/bf16_vs_native.json
```
