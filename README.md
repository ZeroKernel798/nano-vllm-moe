# Nano-vLLM-MoE

Nano-vLLM-MoE is a compact vLLM-style inference playground focused on three current tracks:

1. **MoE runtime**: modular router, prepare/finalize, and expert backends for local and baseline EP experiments.
2. **Chunked prefill**: scheduler and sequence state support partial prefill, with `prefill_first` and `decode_first` policies.
3. **Quantization refactor**: the active line is RTX 4090 / 7B first: BF16 baseline -> W8A16 -> FP8 KV cache -> W8A8.

EP is intentionally kept minimal: only the baseline `torch` all-to-all prepare/finalize path remains. Historical reports are archived only in `docs/README_legacy.md`; old experiment notes and stale benchmark plans were removed to keep the project clean.

## Current Scope

| Track | Status | Main Files |
| --- | --- | --- |
| MoE runtime | Stage-stable refactor | `nanovllm/executor/moe/`, `scripts/moe/` |
| Chunked prefill | Active scheduler feature | `nanovllm/engine/scheduler.py`, `nanovllm/engine/sequence.py`, `scripts/generation/chunked_prefill_bench.py` |
| Quantization | Active refactor line | `nanovllm/quantization/`, `scripts/quantization/`, `scripts/kv_cache/` |
| EP | Baseline only | `nanovllm/executor/moe/prepare_finalize/torch_alltoall.py` |

## MoE Runtime

The MoE path is now structured as:

```text
router -> prepare/finalize -> expert backend -> finalize output
```

Expert backends:

| Backend | Role | File |
| --- | --- | --- |
| `eager` | correctness/reference path | `nanovllm/executor/moe/experts/eager_experts.py` |
| `optimized` | current practical optimized path | `nanovllm/executor/moe/experts/optimized.py` |
| `fused` | Triton grouped-GEMM experiment | `nanovllm/executor/moe/experts/fused.py` |

EP is not a performance claim right now. The only retained EP backend is the baseline `torch` all-to-all path so distributed MoE semantics remain testable without carrying old prototype clutter.

## Chunked Prefill

Chunked prefill is a scheduler-level feature for splitting long prefill requests into smaller scheduled chunks. Two policies are kept:

| Policy | Behavior | Use |
| --- | --- | --- |
| `prefill_first` | continue prefill chunks before decode | simple correctness/default behavior |
| `decode_first` | prioritize decode between prefill chunks | latency-control experiments |

Main benchmark:

```bash
python scripts/generation/chunked_prefill_bench.py \
  --model-path /path/to/model \
  --max-model-len 0 \
  --phases 1,2,3
```

## RTX 4090 7B Quantization Stack

The quantization refactor is intentionally 7B-first. Smaller models can still be used for smoke tests, but README results and project direction should be driven by Qwen2.5-7B on RTX 4090.

Current stage order:

| Stage | Mode | Purpose |
| --- | --- | --- |
| 0 | BF16 | baseline quality, latency, and memory |
| 1 | W8A16 | stable weight-only FP8 checkpoint/runtime path |
| 2 | W8A16 + FP8 KV | long-context memory pressure and KV accuracy |
| 3 | W8A8 | aggressive activation-quantized mode after W8A16/KV are stable |

### Latest 7B Baseline

Benchmark setup: 2026-05-13, 1x RTX 4090 24GB, Qwen2.5-7B-Instruct, fixed synthetic prompt `input=512, output=64`, `max_model_len=2048`, `gpu_memory_utilization=0.9`, one warmup and one measured run. Evidence root: `.remote-logs/quantization/4090_7b_stack_20260513/` on the remote validation machine.

| Mode | Checkpoint size | PPL proxy | Bench prefill TPS | Bench decode TPS | Memory-run prefill TPS | Memory-run decode TPS | Peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 15.24 GB | 7.4367 | 10634.5 | 62.50 | 489.6 | 61.81 | 19.98 GB |
| W8A16 | 8.72 GB | 7.4772 | 5258.7 | 15.22 | 668.7 | 15.26 | 20.92 GB |

W8A16 checkpoint contract is healthy (`196` qweight tensors and `196` weight-scale tensors). The original SM89 Triton path failed because direct FP8-to-BF16 conversion emitted SM90-only instructions; the current W8A16 Triton kernel avoids that by doing on-the-fly FP8->FP32->FP16 dequantization inside the GEMM loop, with no load-time decompressed weight cache. On 2026-05-15, the 3B 512/16 smoke improved from `1.1368 s` per-forward dequant to `0.8247 s` with the SM89-safe Triton path, and decode TPS rose from `37.76` to `143.90`; a 7B 512/16 Triton smoke also compiles and runs (`0.9960 s`, decode TPS `71.79`). Prefill remains weak, so tile/conversion optimization is still the next W8A16 runtime task.

### Latest 7B FP8 KV Probe

Same 7B W8A16 checkpoint, `output=16`, native FP8 paged decode, BF16 KV reference and FP8 KV in the same script.

| Prompt | KV mode | KV storage | Prefill TPS | Decode TPS | Wall time | Peak reserved | Token match |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | BF16 KV | 11.09 GB | 4224.7 | 8.48 | 3.71 s | 21.16 GB | reference |
| 8192 | FP8 KV | 2.90 GB | 7811.0 | 8.63 | 2.79 s | 21.17 GB | 1/16 |
| 16384 | BF16 KV | 10.01 GB | 5673.4 | 9.27 | 4.51 s | 21.31 GB | reference |
| 16384 | FP8 KV | 1.81 GB | 9623.6 | 10.35 | 3.15 s | 21.31 GB | 2/16 |

FP8 KV clearly reduces KV storage, but token match is not acceptable yet. The next task is logits-divergence debugging before FP8 KV can become a default mode.

2026-05-14 backend isolation update: on 7B W8A16 with `input=512, output=8`, `gather_dequant` and `full_dequant` are identical, but both still match only `0.25` of BF16 generated tokens; changing KV scale cache from float16 to float32 does not fix it. This points the next investigation at FP8 KV storage quantization itself, not just the native paged attention kernel.

2026-05-14 K/V sensitivity update: 7B divergence is caused by FP8 K cache quantization. A new experimental K-BF16/V-FP8 mode keeps exact token match on 512-token and 8K-token prompts while reducing KV bytes per block to `75.39%` of BF16, so V-only FP8 is the current safe KV-memory direction.

K quantization probe: FP8 K remains unstable, while fake symmetric int8 K with vector or 16/32/64-wide groups keeps exact tokens on the 7B 512-token probe. The next compression target is K-int8/V-FP8 rather than K-FP8/V-FP8.

K-int8/V-FP8 mixed KV now passes the 512/8K/16K token-match gate on 7B W8A16, with per-block KV bytes reduced to `51.95%` of BF16 including scales. The current optimized path uses a Triton fused mixed-KV store and native mixed paged decode; short 512-token decode is still slower than BF16, but long-context decode has clear wins.

| Prompt | Backend | Exact tokens | Logits cosine | Model TPS | Speedup vs BF16 | KV bytes/block vs BF16 | KV blocks vs BF16 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | BF16 KV | reference | reference | 4.34 | 1.00x | 1.0000 | 1.00x |
| 8192 | K-int8/V-FP8 native | yes | 0.99989 | 14.64 | 3.37x | 0.5195 | 1.56x |
| 16384 | BF16 KV | reference | reference | 4.11 | 1.00x | 1.0000 | 1.00x |
| 16384 | K-int8/V-FP8 native | yes | 0.99987 | 10.77 | 2.62x | 0.5195 | 1.57x |
| 32512 | BF16 KV | reference | reference | 4.03 | 1.00x | 1.0000 | 1.00x |
| 32512 | K-int8/V-FP8 native | yes | 0.99986 | 6.43 | 1.59x | 0.5195 | 1.53x |

### Latest 7B W8A8 Quality Probe

W8A8 has restarted as a profiled, shape-aware backend rather than a blanket mode. On 3B W8A8 static at `M=512`, large MLP projections benefit from `_scaled_mm`, while 2K attention projections lose to activation-quant overhead and now fall back to FP8-weight dequant matmul by default. The runtime cutoff is controlled by `NANOVLLM_FP8_W8A8_SCALED_MM_MIN_DIM` and defaults to `3072`; W8A8 activation quant defaults to the Triton path.

| Gate | BF16 | W8A8 shape-aware | Takeaway |
| --- | ---: | ---: | --- |
| MMLU logit-rank, 300 questions | `0.6033` | `0.6200` | no accuracy regression; prediction agreement `0.9567` |
| WikiText-2 validation PPL, 1024 tokens | `8.3971` | `8.4238` | small absolute drift `+0.0267` |
| GSM8K numeric, 50 questions | `0.32` | `0.36` | no accuracy regression, but generation agreement is low |
| 7B model size | `15.24 GB` | `8.72 GB` | checkpoint is `42.8%` smaller |

The 3B smoke passes with shape-aware prefill TPS `10047.6` versus all-scaled-mm `9277.6`; the BF16-vs-W8A8 correctness probe has avg logits cosine `0.998857` and exact generated tokens on all three prompts. The same cutoff has now started on 7B: checkpoint contract is healthy (`196` qweight/weight-scale/input-scale tensors), shape profile shows `0.49x/0.49x/0.78x` W8A8-vs-BF16 for `gate/up/down` and `0.87x-0.88x` for attention projections, and the 512/16 smoke reports prefill TPS `8775.6`, decode TPS `61.48`, model size `8.72 GB`. On 2026-05-15, W8A8 fallback-layer dequant caching improved the 3B 512/16 wall time from `0.2350 s` to `0.2062 s`, but it remains opt-in until a 7B memory gate passes. On the same WikiText-2 validation data used for W8A8 calibration, the 7B HF proxy PPL is `8.4238` versus BF16 `8.3971` over 1024 tokens. A memory-safe MMLU logit-rank gate over 300 questions reports BF16 accuracy `0.6033`, W8A8 accuracy `0.6200`, and prediction agreement `0.9567`; GSM8K 50-question greedy numeric probe reports BF16 `0.32`, W8A8 `0.36`, but only `0.28` same-number agreement, so GSM8K remains a generation-sensitivity probe rather than the primary quantization gate.

## Repository Map

| Path | Purpose |
| --- | --- |
| `nanovllm/engine/` | scheduler, sequence state, model runner, block manager |
| `nanovllm/executor/moe/` | modular MoE runtime |
| `nanovllm/quantization/` | quantization method registry and FP8 runtime |
| `scripts/moe/` | MoE local compute, backend, and baseline EP scripts |
| `scripts/generation/` | generation and chunked prefill benchmarks |
| `scripts/quantization/` | FP8 export/eval/runtime benchmark suite |
| `scripts/kv_cache/` | FP8 KV validation and microbenchmarks |
| `opt/` | current refactor notes only |
| `docs/README_legacy.md` | archived legacy README |

## Quick Start

```bash
pip install -e .
```

Run MoE local compute:

```bash
python scripts/moe/moe_local_compute_bench.py --device cuda --backends eager,optimized,fused
```

Run chunked prefill:

```bash
python scripts/generation/chunked_prefill_bench.py --model-path /path/to/model --max-model-len 0
```

Run the 7B quantization stack:

```bash
python scripts/quantization/run_4090_7b_stack.py \
  --bf16-model-path /path/to/Qwen2.5-7B-Instruct \
  --w8a16-model-path /path/to/Qwen2.5-7B-Instruct-FP8-W8A16 \
  --stages bf16,w8a16,kv
```

## Next Work

1. Broaden native K-int8/V-FP8 gates to 16K/32K and more seeds, then add a short-context backend cutoff.
2. Replace or optimize the current SM89 W8A16 dequant-matmul runtime.
3. Broaden MMLU/CEval coverage before final W8A8 promotion.
