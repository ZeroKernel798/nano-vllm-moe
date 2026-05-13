# Generation Scripts

| Script | Purpose |
| --- | --- |
| `bench.py` | mixed prefill/decode generation throughput |
| `pd_bench.py` | prefill/decode separated benchmark |
| `chunked_prefill_bench.py` | three-phase chunked prefill validation |

Chunked prefill keeps two policies: `prefill_first` and `decode_first`.
