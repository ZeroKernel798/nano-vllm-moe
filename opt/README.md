# Optimization Notes

这里仅保留当前主线文档。历史实验可以从提交历史或远端日志追溯，但不要作为 active route。

| File | Purpose |
| --- | --- |
| `w8a16.md` | W8A16：简单 Triton 路径，证明省显存和 decode |
| `w8a8.md` | W8A8：Torch baseline -> PTX 小 M -> CUTLASS 大 M -> benchmark 分桶 |
| `kv_cache_quant.md` | KV cache 量化：K-int8/V-FP8 长上下文显存路线 |
| `chunked_prefill_opt.md` | Chunked prefill 记录 |
| `prefix_cache_opt.md` | Prefix cache 记录 |
| `moe_cuda_graph.md` | MoE CUDA Graph gate：当前 capture 失败，保留负结果和下一步 |

当前量化方向：

- W8A16：一个简单 Triton 方案。
- W8A8：Torch baseline、PTX small-M、CUTLASS large-M、benchmark auto bucket。
- KV cache：K-int8/V-FP8，先证明 8K 稳定和 KV 存储收益，再推进 16K/32K。
- MoE CUDA Graph：当前不是主线优化结果；单卡 MoE capture 失败，EP all-to-all 仍强制 eager。
