# Nano-vLLM-MoE

Nano-vLLM-MoE is a compact vLLM-style inference playground focused on three current tracks:

1. **MoE runtime refactor**: router, prepare/finalize, expert compute, and MoE block are separated so each part can be tested independently.
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

W8A16 checkpoint contract is healthy (`196` qweight tensors and `196` weight-scale tensors). Current SM89 runtime uses per-forward BF16 dequant matmul, so W8A16 is a memory/quality milestone, not a speed win yet.

### Latest 7B FP8 KV Probe

Same 7B W8A16 checkpoint, `output=16`, native FP8 paged decode, BF16 KV reference and FP8 KV in the same script.

| Prompt | KV mode | KV storage | Prefill TPS | Decode TPS | Wall time | Peak reserved | Token match |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | BF16 KV | 11.09 GB | 4224.7 | 8.48 | 3.71 s | 21.16 GB | reference |
| 8192 | FP8 KV | 2.90 GB | 7811.0 | 8.63 | 2.79 s | 21.17 GB | 1/16 |
| 16384 | BF16 KV | 10.01 GB | 5673.4 | 9.27 | 4.51 s | 21.31 GB | reference |
| 16384 | FP8 KV | 1.81 GB | 9623.6 | 10.35 | 3.15 s | 21.31 GB | 2/16 |

FP8 KV clearly reduces KV storage, but token match is not acceptable yet. The next task is logits-divergence debugging before FP8 KV can become a default mode.

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

1. Debug FP8 KV logits/token divergence on 7B.
2. Replace or optimize the current SM89 W8A16 dequant-matmul runtime.
3. Re-enter W8A8 only after W8A16 and FP8 KV have clean 7B gates.
