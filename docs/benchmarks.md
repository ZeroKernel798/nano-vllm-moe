# Benchmarks

This document stores extracted benchmark results, not pointers to temporary logs. All results below were run on 1x NVIDIA GeForce RTX 4090 24GB, driver `570.153.02`, Python `3.11.15`, PyTorch `2.5.1+cu121`.

## 7B W8A8 Quality

Model pair:

| Role | Model |
| --- | --- |
| BF16 baseline | Qwen2.5-7B-Instruct |
| W8A8 checkpoint | Qwen2.5-7B-Instruct FP8 W8A8 static |

Checkpoint contract:

| qweight tensors | weight-scale tensors | input-scale tensors | Contract |
| ---: | ---: | ---: | --- |
| `196` | `196` | `196` | healthy |

Quality gates:

| Gate | BF16 | W8A8 | Delta / agreement |
| --- | ---: | ---: | ---: |
| WikiText-2 PPL, 1024 tokens | `8.397079` | `8.423828` | `+0.026749` |
| MMLU logit-rank, 120 questions | `0.8000` | `0.7917` | `-0.0083` |
| MMLU prediction agreement | reference | `0.9667` | `4 / 120` flips |

The W8A8 HF proxy quality path is close to BF16 on both PPL and MMLU. The nano runtime Triton W8A8 generation path currently hits a Triton SM89 compiler failure, so end-to-end W8A8 serving is not promoted as a README claim.

## 7B W8A8 CUDA/PTX Microbench

`NANOVLLM_W8A8_JIT_KERNEL=auto_7b`, backend `cuda_ptx`, 80 repeats, 20 warmups.

| Layer | Shape `(M,K,N)` | BF16 ms | W8A8 JIT ms | W8A8 / BF16 | Speedup | Cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `self_attn.q_proj` | `1,3584,3584` | `0.030242` | `0.029746` | `0.984x` | `1.02x` | `0.999636` |
| `mlp.gate_proj` | `1,3584,18944` | `0.149500` | `0.039600` | `0.265x` | `3.78x` | `0.999602` |
| `mlp.down_proj` | `1,18944,3584` | `0.148241` | `0.048511` | `0.327x` | `3.06x` | `0.999663` |
| `self_attn.q_proj` | `16,3584,3584` | `0.049297` | `0.048624` | `0.986x` | `1.01x` | `0.999275` |
| `mlp.gate_proj` | `16,3584,18944` | `0.149232` | `0.053407` | `0.358x` | `2.79x` | `0.999303` |
| `mlp.down_proj` | `16,18944,3584` | `0.230999` | `0.230578` | `0.998x` | `1.00x` | `0.999305` |

The current JIT is most valuable for decode MLP shapes, especially gate projection. It is not yet a full GEMM replacement.

## 7B K-int8/V-FP8 KV Cache

Model: Qwen2.5-7B-Instruct FP8 W8A16. Output length: 8 decode tokens. Backend: native mixed paged decode.

| Prompt length | Exact match | Token match | First mismatch | Logits cosine | Pre-divergence cosine | BF16 model TPS | Mixed-KV model TPS | Speedup | Bytes/block vs BF16 |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `8192` | `1.000` | `1.000` | none | `0.999800` | `0.999800` | `0.555` | `3.432` | `6.18x` | `0.5195x` |
| `16384` | `0.000` | `0.625` | token `5` | `0.972212` | `0.999819` | `3.185` | `18.329` | `5.75x` | `0.5195x` |

The 8K gate proves strong mixed-KV accuracy with roughly half the KV bytes per block. The 16K run shows a quality boundary: logits stay very close before divergence, but generation is not exact. A 32K attempt hit a BF16-reference attention boundary before a clean comparison could be recorded.

## Prefix Cache

Model: Qwen2.5-3B-Instruct. Workload: 8 requests, average prompt length 1152, max output 16, block size 256.

| Backend | Scenario | Token hit rate | Block hit rate | Hit blocks | Prefill TPS | Decode TPS | Total time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hash | no-reuse | `0.000` | `0.000` | `0` | `6154.6` | `274.3` | `1.938 s` |
| hash | shared-prefix | `0.875` | `0.875` | `28` | `1583.6` | `266.8` | `1.748 s` |
| hash | partial-shared | `0.438` | `0.438` | `14` | `3956.0` | `261.2` | `1.890 s` |
| radix | no-reuse | `0.000` | `0.000` | `0` | `5932.5` | `268.8` | `2.005 s` |
| radix | shared-prefix | `0.875` | `0.875` | `28` | `1395.9` | `223.3` | `2.010 s` |
| radix | partial-shared | `0.438` | `0.438` | `14` | `3910.0` | `248.4` | `1.931 s` |

Hash and radix reach the same reuse rate on direct scenarios. Radix keeps extra structure for branching prefixes and LRU eviction.

## Chunked Prefill

Model: Qwen2.5-3B-Instruct. Scenario: request A decodes 64 tokens after a 512-token prompt; request B with a 4096-token prompt is injected after 16 A tokens. Metric focus is B TTFT and A decode interruption.

Correctness smoke:

| Tokens | Chunk | Output match | Logits cosine |
| ---: | ---: | --- | ---: |
| `512` | `512` | yes | `0.999980` |

Interference result:

| Chunk | Policy | B TTFT | A avg ITL | A p95 ITL | A max ITL | Prefill steps |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `4096` | prefill_first | `181.4 ms` | `10.95 ms` | `8.83 ms` | `191.1 ms` | `2` |
| `128` | prefill_first | `1009.8 ms` | `24.06 ms` | `8.82 ms` | `1019.0 ms` | `36` |
| `128` | decode_first | `1277.4 ms` | `24.30 ms` | `40.75 ms` | `42.4 ms` | `36` |
| `256` | prefill_first | `517.9 ms` | `16.25 ms` | `8.80 ms` | `527.1 ms` | `18` |
| `256` | decode_first | `635.3 ms` | `16.09 ms` | `39.77 ms` | `41.0 ms` | `18` |
| `512` | prefill_first | `256.6 ms` | `12.10 ms` | `8.82 ms` | `265.7 ms` | `9` |
| `512` | decode_first | `326.4 ms` | `12.26 ms` | `40.54 ms` | `42.5 ms` | `9` |
| `1024` | prefill_first | `192.4 ms` | `11.06 ms` | `8.83 ms` | `201.5 ms` | `5` |
| `1024` | decode_first | `233.5 ms` | `11.31 ms` | `11.65 ms` | `60.4 ms` | `5` |

`decode_first` trades B TTFT for much lower worst-case decode interruption. At chunk 512, max A decode pause drops from `191.1 ms` no-chunk to `42.5 ms`, a `4.5x` reduction, while B TTFT remains within `1.8x` of the no-chunk baseline.

## MoE Local Compute

Synthetic single-GPU MoE local compute. Shape: hidden `2048`, intermediate `1408`, 8 experts, top-1, FP16, 10 repeats.

| Tokens | Backend | Mean latency | Tokens/s | Speedup vs eager | Cosine vs eager | Max abs |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `16` | eager | `2.752 ms` | `5.8K` | `1.00x` | `1.0000002` | `0` |
| `16` | optimized | `0.966 ms` | `16.6K` | `2.85x` | `1.0000000` | `0.00098` |
| `256` | eager | `1.974 ms` | `129.7K` | `1.00x` | `0.9999999` | `0` |
| `256` | optimized | `0.945 ms` | `270.8K` | `2.09x` | `0.9999999` | `0` |

## MoE CUDA Graph Gate

Model: Qwen1.5-MoE-A2.7B-Chat. Workload: TP=1, EP=1, 8 fixed prompts, input 16, output 16, max model length 128. Backend: `optimized` experts with `torch` EP backend. Environment: NVIDIA GeForce RTX 4080 SUPER, driver `580.142`, Python `3.12.3`.

Eager backend vs eager `optimized` vs CUDA Graph `optimized`, repeat 3 (no discard, each repeat has its own warmup):

| Mode | Wall mean | Prefill TPS mean | Decode TPS mean | Speedup (decode) | Greedy match vs eager |
| --- | ---: | ---: | ---: | ---: | --- |
| `eager` backend, `--enforce-eager` | `3.268 s` | `376.14` | `40.98` | `1.00x` | reference |
| `optimized` backend, `--enforce-eager` | `1.142 s` | `1213.93` | `115.81` | `2.83x` | exact vs eager backend |
| CUDA Graph `optimized` | `0.747 s` | `1242.94` | `186.45` | `4.55x` | `128 / 128` tokens, `8 / 8` rows |

CUDA Graph alone (on top of `optimized` eager): `1.61x` decode, `1.53x` end-to-end wall.

Per-repeat decode TPS:

| Repeat | eager backend | optimized eager | optimized graph |
| ---: | ---: | ---: | ---: |
| `0` | `40.69` | `115.80` | `186.28` |
| `1` | `41.08` | `115.56` | `186.50` |
| `2` | `41.17` | `116.06` | `186.57` |

Capture fix summary (`opt/moe_cuda_graph.md` for full detail):

- Q1 — `Qwen2MoeSparseMoeBlock.forward` falls back to single-stream when `torch.cuda.is_current_stream_capturing()` is true, so the shared-expert side stream no longer trips "operation failed during capture".
- Q2 — `OptimizedExperts` keeps a per-`num_tasks` `_GraphWorkspace` (sorted-token-ids, expert-ids, gate-up, activated, expert-out, output, num-tokens-post-padded) so no allocation happens inside `torch.cuda.graph(...)`.
- Q3 — `capture_cudagraph` warms up each `bs` twice with `torch.cuda.synchronize()` before capture, so Triton JIT and autotune are done host-side.
- Q4 — `moe_align_block_size_fixed` replaces `torch.bincount` (host-syncs to size its output) with a static-shape `scatter_add_`; the capture path now unconditionally uses the fixed-capacity align.

The experimental `NANOVLLM_MOE_GRAPH_TOKEN_KERNEL=1` token kernel remains in the tree as a debug toggle but is no longer the production CUDA Graph path because the `optimized` backend now passes exact greedy match and is fast enough on this workload.

Evidence: `.remote-logs/moe_cuda_graph_capture_fix_20260520/{device.txt,eager_repeat3.json,graph_repeat3.json,greedy_eager.json,greedy_graph.json}`.

## Missing Framework Baseline

The validation environment did not have an importable `vllm` package, and `python -m vllm.benchmarks.benchmark_throughput` was unavailable. No vLLM baseline is reported here. This avoids presenting a framework comparison without a valid measurement.
