# KV Cache Scripts

FP8 KV cache is part of the quantization refactor. It is not a default quality path yet; current work focuses on memory savings and logits/token divergence debugging.

| Script | Purpose |
| --- | --- |
| `kv_cache_fp8_smoke.py` | BF16 KV vs FP8 KV smoke with token match and storage ratios |
| `kv_cache_fp8_logits.py` | single-case logits tracing |
| `kv_cache_fp8_accuracy_suite.py` | multi-seed/prompt accuracy suite |
| `fp8_paged_attention_microbench.py` | isolated native FP8 paged attention microbench |
