# MoE Scripts

MoE is stage-stable and kept focused on the current runtime split: router, prepare/finalize, expert backend, finalize.

| Script | Purpose |
| --- | --- |
| `moe_local_compute_bench.py` | synthetic single-device MoE local compute benchmark for `eager/optimized` |
| `moe_backend_bench.py` | end-to-end MoE backend comparison on a model path |
| `ep_prepare_finalize_microbench.py` | baseline `torch` all-to-all prepare/finalize microbench |
| `ep_bench.py` | small MoE generation benchmark |
| `ep_tp_bench.py` | TP/EP smoke benchmark |

Old EP prototype backends were removed. The retained EP backend is the baseline `torch` all-to-all path.

On single GPU (`ep_size=1`) CUDA Graph decode is now the default for the `optimized` backend. Remote check on Qwen1.5-MoE-A2.7B-Chat measured decode at `186.45 tok/s` vs `40.98 tok/s` eager (`4.55x`), with `128/128` greedy-token match. See `opt/moe_cuda_graph.md` for the capture/replay design.
