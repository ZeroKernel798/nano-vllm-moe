# MoE Scripts

MoE is stage-stable and kept focused on the current runtime split: router, prepare/finalize, expert backend, finalize.

| Script | Purpose |
| --- | --- |
| `moe_local_compute_bench.py` | synthetic single-device MoE local compute benchmark for `eager/optimized/fused` |
| `moe_backend_bench.py` | end-to-end MoE backend comparison on a model path |
| `ep_prepare_finalize_microbench.py` | baseline `torch` all-to-all prepare/finalize microbench |
| `ep_bench.py` | small MoE generation benchmark |
| `ep_tp_bench.py` | TP/EP smoke benchmark |

Old EP prototype backends were removed. The retained EP backend is the baseline `torch` all-to-all path.
