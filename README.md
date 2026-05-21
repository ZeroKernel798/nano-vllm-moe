# Nano-vLLM-MoE

A vLLM-style inference playground built on top of Nano-vLLM, focused on single-GPU (RTX 4090 24GB-class) serving experiments along three tracks: FP8 weight/activation quantization, KV cache compression with prefix reuse, and MoE scheduling with CUDA Graph. The runtime is intentionally small and readable, with every optimization backed by microbenchmarks and a quality gate that traces to remote logs. Upstream: https://github.com/GeeeekExplorer/nano-vllm.git

## Highlights

* **FP8 W8A16 weight-only** — Triton on-the-fly dequant GEMM; weight footprint `-44.8%`, decode TPS `1.21x`
* **FP8 W8A8 static** — Three measured linear backends (Torch `_scaled_mm`, custom PTX, in-repo CUTLASS); checkpoint `-42.8%`, WikiText PPL drift only `+0.0267`
* **Mixed K-int8 / V-FP8 KV** — Long-context memory-pressure path; bytes/block `0.52x` of BF16, teacher-forced PPL within `+1.55%` end-to-end
* **Modular MoE runtime** — `router → prepare/finalize → expert → finalize` decomposition with switchable `eager` / `optimized` backends
* **MoE CUDA Graph** — Default-on for single GPU (`ep_size=1`); decode `4.55x` vs eager backend with `128/128` greedy-token match
* **Hash + Radix prefix cache** — Two block-level backends with matching hit rate; `87.5%` shared-prefix hit rate
* **Chunked prefill** — `decode_first` scheduling cuts short-decode inter-token pause from `4197 ms` to `57 ms` under 32K prefill interference
* **Quantization suite** — PPL / MMLU logit-rank / linear-backend microbench gates

## Install

```bash
git clone https://github.com/ZeroKernel798/nano-vllm-moe.git
cd nano-vllm-moe
pip install -e .
```

## Quick Start

```bash
python scripts/moe/moe_local_compute_bench.py --device cuda --backends eager,optimized
python scripts/prefix_cache/prefix_cache_bench.py --model /path/to/model --scenario shared-prefix
python scripts/generation/chunked_prefill_bench.py --model-path /path/to/model --phases 1,3
```

## Benchmark

### FP8 W8A16 weight-only quantization

**Setup**

- Hardware: NVIDIA GeForce RTX 4090 24GB
- Model: Qwen2.5-7B-Instruct (BF16 baseline / W8A16 export)
- Parallelism: TP=1, EP=1
- Export: `scripts/quantization/quantize.py --scheme fp8_w8a16`, quantized `252/434` tensors
- Decode workload: input 512, output 128 / 16, warmup 1, repeat 3

**Results**

| Mode | Model size bytes | Decode TPS (out=16) | Decode TPS (out=128) | vs BF16 |
|------|------:|------:|------:|------:|
| BF16 baseline | `6,183,464,346` | `130.69 tok/s` | — | `1.00x` |
| FP8 W8A16 | `3,413,055,132` | `158.36 tok/s` | `158.73 tok/s` | `1.21x` |

**Analysis**: W8A16 is the simplest weight-only path. Weight footprint drops from `6.18 GB` to `3.41 GB` (`-44.8%`); Triton on-the-fly dequant trades weight bandwidth for compute, giving `1.21x` decode on the 512-prompt gate. CUDA reserved/allocated deltas are smaller than the checkpoint delta, so the README scope stays "weight-only memory saving + decode improvement" rather than overall VRAM halving.

---

### FP8 W8A8 static quantization

**Setup**

- Hardware: NVIDIA GeForce RTX 4090 24GB
- Model: Qwen2.5 family (BF16 baseline / W8A8 static export with offline-calibrated activation scales)
- Parallelism: TP=1, EP=1
- Export: `scripts/quantization/quantize.py --scheme fp8_w8a8_static`
- Linear backends: Torch `_scaled_mm`, custom PTX, in-repo CUTLASS (custom epilogue)
- Quality eval: WikiText-2 PPL (1024 tokens) + MMLU logit-rank (120 questions)
- E2E throughput on a 3B-scale config (`hidden_size=2048`, `intermediate_size=11008`, `36` layers); true 7B microbench provided separately

**Quality gates (7B)**

| Gate | BF16 | W8A8 | Delta |
|------|------:|------:|------:|
| Checkpoint size | `15.24 GB` | `8.72 GB` | `-42.8%` |
| WikiText-2 PPL, 1024 tokens | `8.3971` | `8.4238` | `+0.0267` |
| MMLU logit-rank, 120 questions | `80.00%` | `79.17%` | `-0.83 pp` |
| MMLU prediction agreement | reference | `96.67%` | `4 / 120` flips |

**Linear-backend microbench (3B-scale)**

| Layer | Shape `(M,K,N)` | BF16 | Torch | PTX | CUTLASS | Winner |
|------|---|------:|------:|------:|------:|------|
| `self_attn.q_proj` | `1,2048,2048` | `0.0095 ms` | `0.0495 ms` | `0.0284 ms` | `0.0121 ms` | CUTLASS |
| `mlp.up_proj` | `16,2048,11008` | `0.0301 ms` | `0.0486 ms` | `0.0304 ms` | `0.0112 ms` | CUTLASS |
| `mlp.down_proj` | `256,11008,2048` | `0.0927 ms` | `0.0793 ms` | `0.1983 ms` | `0.0834 ms` | Torch |
| `mlp.up_proj` | `1024,2048,11008` | `0.3252 ms` | `0.1864 ms` | `0.7310 ms` | `0.1853 ms` | CUTLASS |

**3B-scale E2E throughput (W8A8 CUTLASS vs native BF16, plain serving path)**

| Phase | Input | Output | BF16 total tok/s | W8A8 total tok/s | W8A8 / BF16 |
|------|------:|------:|------:|------:|------:|
| mixed | 1024 | 128 | `1154.21` | `1785.43` | `1.55x` |
| mixed | 4096 | 128 | `3674.52` | `5608.63` | `1.53x` |
| mixed | 16384 | 128 | `8541.24` | `11494.63` | `1.35x` |
| mixed | 32640 | 128 | `9776.19` | `11788.54` | `1.21x` |

**Analysis**: W8A8 shrinks the checkpoint meaningfully while passing all quality gates — PPL drift only `+0.0267`, MMLU only `-0.83 pp`. The linear-backend bucket table shows CUTLASS is the stable all-M default candidate; the previous `M<=16: PTX` auto rule is paused for recalibration. Torch `_scaled_mm` remains the necessary mid-M reference. On the 3B-scale config, W8A8 CUTLASS improves mixed total throughput by `1.35x..1.55x` for 1K..16K prompts.

> **Boundary**: End-to-end Triton W8A8 still compiler-aborts on SM89 in the current path; the fused W8A8 path is not enabled. A true 7B BF16 + paired W8A8 export are available but the full E2E matrix is still being filled in.

---

### Mixed K-int8 / V-FP8 KV cache

**Setup**

- Hardware: NVIDIA GeForce RTX 4090 24GB
- Model: Qwen2.5-7B-Instruct BF16
- Quantization: K int8 + V FP8, K group=32, scale dtype float16
- Quality gate: WikiText teacher-forced PPL/NLL (token exact alignment is debug evidence only)
- Memory test: 512 fixed KV blocks same-capacity check + 8K/64 generation peak

**Teacher-forced PPL (K group=32)**

| Context | BF16 PPL | Mixed KV PPL | Delta | bytes/block |
|------:|------:|------:|------:|------:|
| 4K | reference | — | `+1.20%` | — |
| 8K | reference | — | `+1.20%` | — |
| 16K | reference | — | `+1.55%` | — |
| 32K | reference | — | `-0.16%` | `0.5195x` BF16 |

**Memory (7B, 512 fixed KV blocks)**

| Metric | BF16 KV | Mixed KV | Delta |
|------|------:|------:|------:|
| KV arena | `7.00 GiB` | `3.64 GiB` | `-3.36 GiB` |
| 8K/64 generation peak allocated | `22.41 GiB` | `19.04 GiB` | `-3.37 GiB` |

**Analysis**: Mixed K-int8/V-FP8 passes the PPL gate across all measured lengths (worst `+1.55%`, 32K actually slightly under BF16) while halving bytes/block. Same-capacity allocation drops the KV arena from `7.00 GiB` to `3.64 GiB`. Full FP8 KV (FP8 for K too) explodes to `1509.77` PPL on the 4K/8K smoke, so the mainline retains the mixed path only. Strict token exact alignment is seed-sensitive at 16K/32K (32K seed1 only becomes exact at K group-1, which loses the capacity gain), so BF16 KV remains the quality reference.

---

### MoE runtime (router → prepare/finalize → expert → finalize)

**Setup**

- Hardware: NVIDIA GeForce RTX 4080 SUPER 16GB, driver `580.142`, Python 3.12.3
- Model: Qwen1.5-MoE-A2.7B-Chat
- Parallelism: TP=1, EP=1
- Workload: 8 fixed prompts, input 16 / output 16, max model length 128, repeat 3

**Local compute (synthetic, 8-expert top-1, 256 tokens)**

| Backend | Mean latency | Tokens/s | Correctness vs eager |
|------|------:|------:|------|
| `eager` | `1.974 ms` | `129.7K` | reference |
| `optimized` | `0.945 ms` | `270.8K` | cosine `0.9999999` |

**End-to-end three-mode comparison**

| Mode | Wall mean | Prefill TPS | Decode TPS | Decode speedup | Greedy match |
|------|------:|------:|------:|------:|------|
| `eager` backend (Python per-token expert loop) | `3.268 s` | `376.14` | `40.98` | `1.00x` | reference |
| `optimized` backend, `--enforce-eager` (Triton fused MoE) | `1.142 s` | `1213.93` | `115.81` | `2.83x` | exact |
| `optimized` backend, CUDA Graph | `0.747 s` | `1242.94` | `186.45` | `4.55x` | `128/128` tokens |

**Analysis**: The `optimized` backend moves expert compute off the Python per-token loop via Triton fused MoE, already netting `2.83x` decode. Layering CUDA Graph on top adds another `1.61x` after the capture-unsafe spots are fixed one by one — `torch.bincount` host-sync replaced by `scatter_add_`, shared-expert side-stream collapsed during capture, expert workspace pre-allocated during warmup. Single-GPU `ep_size=1` captures and replays cleanly with exact `128/128` output match vs eager.

> **Boundary**: EP all-to-all still forces eager because of dynamic collectives. The legacy `fused` backend was removed; `optimized` is the only non-eager backend.

---

### Prefix cache (hash + radix)

**Setup**

- Hardware: NVIDIA GeForce RTX 4090 24GB
- Model: Qwen2.5-3B-Instruct
- Workload: 8 requests, shared prefix `1024` tokens, unique `128` tokens, output `8`, block size `256`
- Backends: `hash` (default) / `radix` (block-level radix metadata)

**Results**

| Backend | Scenario | Token hit rate | Total time | Prefill TPS | Decode TPS |
|---------|----------|------:|------:|------:|------:|
| hash | no-reuse | `0.0%` | `2.367 s` | `4288.8` | `259.0` |
| hash | shared-prefix | `87.5%` | `1.540 s` | `1551.4` | `257.1` |
| hash | partial-shared | `43.8%` | `1.637 s` | `3981.1` | `253.6` |
| radix | no-reuse | `0.0%` | `1.750 s` | `6013.4` | `259.1` |
| radix | shared-prefix | `87.5%` | `1.484 s` | `1621.4` | `256.1` |
| radix | partial-shared | `43.8%` | `1.614 s` | `4030.2` | `261.1` |

**Analysis**: Shared-prefix hit rate matches the expected `7/8 = 87.5%`; partial-shared lands at `43.8%`. Hash and radix backends produce the same hit rate on direct shared-prefix tests. Hash is the default efficient implementation; radix is kept as the block-level metadata backend for future branching-prefix experiments.

---

### Chunked prefill (decode_first scheduling)

**Setup**

- Hardware: NVIDIA GeForce RTX 4090 24GB
- Model: Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct
- Workload: request A is a short prompt (`32` tokens / `128` output); request B is a long prompt (`2K..32K` / `1` output) inserted after A generates 8 tokens
- Tracked metric: A's max inter-token latency (lower is better) and B's TTFT

**7B max inter-token latency on request A**

| B prompt | no chunk | prefill_first 128 | decode_first 128 | decode_first 1024 (A max / B TTFT) |
|------:|------:|------:|------:|---:|
| 2K | `2255.0 ms` | `473.2 ms` | `52.7 ms` | `120.4 / 238.7 ms` |
| 8K | `778.2 ms` | `1774.7 ms` | `52.0 ms` | `140.3 / 1032.5 ms` |
| 16K | `1739.6 ms` | `3868.8 ms` | `57.1 ms` | `165.9 / 2268.1 ms` |
| 32K | `4197.3 ms` | `9221.8 ms` | `57.2 ms` | `219.2 / 5375.0 ms` |

**3B max inter-token latency on request A**

| B prompt | no chunk | prefill_first 128 | decode_first 128 | decode_first 1024 (A max / B TTFT) |
|------:|------:|------:|------:|---:|
| 2K | `2640.9 ms` | `572.2 ms` | `61.8 ms` | `77.6 / 153.6 ms` |
| 16K | `888.0 ms` | `4419.1 ms` | `65.4 ms` | `103.3 / 1418.1 ms` |
| 32K | `2273.4 ms` | `9069.3 ms` | `63.6 ms` | `136.6 / 3344.0 ms` |

**Analysis**: `decode_first` is the key to keeping short-request decode responsive. Chunk `128` caps the max A inter-token latency around `60 ms` across all B lengths, dropping the 7B 32K pause from `4197 ms` to `57 ms`. `prefill_first` keeps serving B's prefill back-to-back even when chunked, so long B drags A's pauses into the seconds range — it is **not** a latency fix for an active short-decode stream. Larger chunks lower B's TTFT at the cost of higher A pauses; `512/1024` favors throughput, `128` favors decode responsiveness.

---

## Current Boundaries

- W8A8 end-to-end Triton generation still triggers a compiler abort on SM89, so README claims rely on HF quality gates and explicit CUDA/PTX microbenchmarks.
- For 7B W8A8 linear backends, in-repo CUTLASS is the stable all-M comparison route. The old `M<=16: PTX` auto threshold is paused until the interrupted `scalar_m32` and `auto_7b` checks complete.
- MoE CUDA Graph decode is default-on for single GPU (`ep_size=1`). EP all-to-all still forces eager because dynamic collectives are not capture-safe in this runtime.
- Chunked prefill `prefill_first` is not a latency fix for active short-decode streams: under 32K inserted prefill and chunk `128`, A's max pause is about `9069 ms` on 3B and `9222 ms` on 7B.
- The mixed KV quality gate is now teacher-forced PPL/NLL. Strict token exact alignment is K group-size sensitive at 16K/32K across seeds, so BF16 KV remains the quality reference.
- The remote validation environment did not have a usable `vllm` package, so no vLLM baseline is shown in the README.

## Repository Map

| Path | Purpose |
|------|------|
| `nanovllm/engine/` | scheduler, sequence, model runner, block manager, radix tree |
| `nanovllm/executor/moe/` | modular MoE runtime (router / prepare-finalize / experts / blocks) |
| `nanovllm/quantization/` | FP8 runtime, CUDA/PTX JIT, CUTLASS extension, quantization registry |
| `nanovllm/models/` | Llama / Qwen2 / Qwen2-MoE / Qwen3 / Qwen3-MoE |
| `scripts/generation/` | generation and chunked prefill benchmarks |
| `scripts/kv_cache/` | FP8 / mixed KV validation scripts |
| `scripts/moe/` | MoE local compute and backend benchmarks |
| `scripts/prefix_cache/` | hash / radix prefix cache benchmarks |
| `scripts/quantization/` | FP8 export, quality gates, microbenchmarks |
| `opt/` | per-track design notes and experimental records |
| `docs/benchmarks.md` | extracted full benchmark results |
