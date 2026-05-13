# RTX 4090 FP8 Quantization Stack

## Direction Summary

This track turns the next phase into a 24GB RTX 4090 / SM89 quantized inference stack. The goal is not to blindly enable the most aggressive quantization everywhere. The goal is to build a staged, measurable path from a stable BF16 baseline to W8A16, then FP8 KV cache, and finally W8A8 where the kernel profile shows a real win.

The target project story is:

> A memory-aware FP8 inference stack for single-card RTX 4090 serving, combining weight-only FP8, FP8 KV cache, chunked prefill, and W8A8 profiling to improve long-context capacity while keeping quality gates explicit.

## Stage Order

| Stage | Mode | Weight | Activation | KV Cache | Goal | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | BF16 baseline | BF16 | BF16 | BF16 | Establish quality, latency, and memory baseline | Repeatable generation, PPL/logits, peak memory |
| 1 | W8A16 balanced | FP8 | BF16 | BF16 | Stable weight-only memory reduction | Logits/PPL close to BF16 and no severe latency regression |
| 2 | W8A16 + FP8 KV | FP8 | BF16 | FP8 | Long-context and concurrency headroom on 24GB | KV memory reduction with bounded logits divergence |
| 3 | W8A8 aggressive | FP8 | FP8 | FP8 optional | Explore activation quantization and SM89 FP8 GEMM limits | Layer/shape profile proves where W8A8 wins |

## Models And Hardware

Primary remote hardware:

- GPU: RTX 4090 24GB, SM89.
- Main environment: `REMOTE_PYTHON=/home/ubuntu/miniconda3/envs/nano-vllm/bin/python`.
- Primary dense models:
  - Fast smoke: Qwen1.5-0.5B-Chat.
  - Main validation: Qwen2.5-3B-Instruct.
  - Resume-grade target: Qwen2.5-7B-Instruct.

The 4090 constraint is part of the point: 24GB is tight enough that weight memory, KV memory, prefill policy, and quality loss must be traded off together.

## Stage 0: BF16 Baseline

### Purpose

Create a stable reference before making any quantization claim. Every quantized mode must compare against the same prompt set, model revision, scheduler settings, and memory reporting path.

### Benchmark Matrix

| Model | Workload | Suggested Shape | Metrics |
| --- | --- | --- | --- |
| 0.5B | smoke | short prompt + short decode | correctness, script health |
| 3B | prefill-heavy | fixed long prompt, short output | TTFT, prefill TPS, peak memory |
| 3B | decode-heavy | short prompt, long output | TPOT/ITL, decode TPS, peak memory |
| 7B | balanced | medium prompt, medium output | quality, memory, latency |
| 7B | long-context | 8K/16K/32K prompt if supported | OOM boundary, KV pressure |

### Pass Gate

- BF16 generation is repeatable with fixed prompts and deterministic sampling settings.
- Peak allocated/reserved memory is captured.
- PPL/logits scripts run for at least the 0.5B and 3B targets before 7B claims are made.

## Stage 1: W8A16 Balanced Mode

### Purpose

Make weight-only FP8 the stable quantization baseline. This is the most useful first resume milestone because it demonstrates checkpoint export, quantized runtime loading, quality validation, and memory reduction without activation quantization noise.

### Implementation Scope

- Reuse the existing FP8 checkpoint contract and W8A16 runtime path.
- Keep activation dtype BF16/FP16.
- Report model weight memory, peak memory, prefill TPS, decode TPS, logits cosine, top-k overlap, and PPL delta.
- Treat W8A16 as the default stable quantized mode until W8A8 proves a shape-specific win.

### Pass Gate

- 3B W8A16 passes logits/PPL checks against BF16.
- 3B W8A16 shows lower memory than BF16 with acceptable prefill/decode regression or improvement.
- 7B W8A16 runs without OOM under the chosen benchmark shapes and has a documented quality result.

## Stage 2: W8A16 + FP8 KV Cache

### Purpose

Combine weight memory reduction with KV memory reduction. This is the main 4090 long-context story: weight quantization frees model memory, while FP8 KV increases context/concurrency headroom.

### Implementation Scope

- Compare BF16 KV, FP8 gather-dequant, and native FP8 paged decode where available.
- Reuse the existing chunked prefill benchmark as the interference-control layer.
- Measure KV bytes/token, peak memory, max context before OOM, decode latency, token exact match, and logits divergence before the first token mismatch.

### Pass Gate

- 3B W8A16 + FP8 KV passes long-context correctness checks at 8K/16K and attempts 32K.
- 7B W8A16 + FP8 KV reports a clear capacity or memory improvement over W8A16 + BF16 KV.
- Any speed regression is attributed to gather/dequant/native attention overhead rather than hidden in an end-to-end average.

## Stage 3: W8A8 Aggressive Mode

### Purpose

Explore activation quantization only after W8A16 and FP8 KV have a stable baseline. W8A8 on SM89 should be treated as a profiled optimization, not a blanket default.

### Implementation Scope

- Split W8A8 Linear time into activation quantization, scaled GEMM, output cast, and framework overhead.
- Compare torch `_scaled_mm`, Triton activation quant, and any future CUTLASS/Marlin-style backend in the same shape matrix.
- Produce a layer/shape rule: which Linear shapes use W8A8, which stay W8A16, and which stay BF16.

### Pass Gate

- W8A8 quality passes 0.5B and 3B logits/PPL checks.
- W8A8 wins at least one important prefill-heavy shape after activation quantization overhead is included.
- If W8A8 loses, document the bottleneck and keep it as an experimental backend instead of default.

## Final README Story

The final public-facing result should not be a raw collection of quantization experiments. It should be a small decision table:

| Serving Mode | Weights | Activations | KV | Best For | Expected Claim |
| --- | --- | --- | --- | --- | --- |
| Quality | BF16 | BF16 | BF16 | baseline correctness | reference quality and latency |
| Balanced | W8A16 | BF16 | BF16 | stable 3B/7B serving | lower memory with bounded quality delta |
| Long Context | W8A16 | BF16 | FP8 | 24GB long-context pressure | larger context/concurrency envelope |
| Aggressive | W8A8 | FP8 | FP8 optional | profiled prefill-heavy shapes | only enabled where profiling proves benefit |

## Resume Bullets To Earn

- Built a 4090/SM89-focused FP8 inference stack with W8A16 weights, FP8 KV cache, and W8A8 profiling for long-context serving under a 24GB memory budget.
- Implemented staged quality gates using logits cosine, top-k overlap, PPL delta, token match, and memory/latency benchmarks before enabling aggressive quantization paths.
- Profiled W8A8 runtime into activation quantization, scaled GEMM, cast, and framework overhead to drive shape-aware backend selection instead of blind full-model quantization.
- Combined quantized weights, FP8 KV cache, and chunked prefill to study the tradeoff between context length, decode latency, and quality on consumer GPUs.

## Next Concrete Tasks

1. Debug native FP8 KV logits divergence on 7B 8K/16K before treating FP8 KV as a default mode.
2. Fix or improve the SM89 W8A16 runtime path; current per-forward BF16 dequant matmul is memory/quality-positive but speed-negative.
3. Add a safer 32K capacity path: either FP8-only reference mode, smaller temporary activations, chunked prefill integration, or a script mode that does not require BF16 reference in the same run.
4. After W8A16 and FP8 KV quality are understood, restart W8A8 profiling with activation quantization and scaled GEMM breakdown.
5. Keep README focused on 7B results and only add W8A8 after it has a defensible 7B gate.

## 2026-05-13 Track Reset

Decision: treat MoE as stage-stable for now and switch the active optimization focus to the RTX 4090 quantization stack. The agreed order is BF16 baseline, W8A16, FP8 KV cache, and finally W8A8. Existing W8A8 and FP8 KV notes remain valid historical evidence, but new claims should be produced through this staged matrix so README and resume bullets are easier to defend.

## 2026-05-13 7B BF16/W8A16 And FP8 KV Baseline

### Scope

Per user direction, skip 0.5B and 3B for now and make Qwen2.5-7B-Instruct the active benchmark target. Added `scripts/quantization/run_4090_7b_stack.py` as the orchestration entrypoint so 7B quantization runs stay in the quantization script group instead of becoming ad-hoc shell commands.

### Remote Evidence

- Root: `.remote-logs/quantization/4090_7b_stack_20260513/`
- BF16/W8A16 suite: `.remote-logs/quantization/4090_7b_stack_20260513/bf16_w8a16/`
- FP8 KV 8K/16K/32K attempt: `.remote-logs/quantization/4090_7b_stack_20260513/kv/`
- FP8 KV 32K retry at `gpu_memory_utilization=0.95`: `.remote-logs/quantization/4090_7b_stack_20260513/kv_32k_retry_gmu095/`
- Cleanup logs: `.remote-logs/quantization/4090_7b_stack_20260513/cleanup.log`, `kv_cleanup.log`, `kv_32k_retry_cleanup.log`

### 7B BF16 vs W8A16 Result

Config: `input_len=512`, `output_len=64`, `max_model_len=2048`, `gpu_memory_utilization=0.9`, `bench_warmup=1`, `bench_repeat=1`, PPL proxy `max_tokens=512`; compare_logits skipped because previous 7B compare was known to OOM on 24GB.

| Mode | Exit | Checkpoint size | PPL proxy | Bench prefill TPS | Bench decode TPS | Memory-run prefill TPS | Memory-run decode TPS | Peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0 | 15.24 GB | 7.4367 | 10634.5 | 62.50 | 489.6 | 61.81 | 19.98 GB |
| W8A16 | 0 | 8.72 GB | 7.4772 | 5258.7 | 15.22 | 668.7 | 15.26 | 20.92 GB |

W8A16 checkpoint inspect passed: `196` qweight tensors, `196` weight-scale tensors, no missing scales, `quantization_type=fp8_w8a16`. Quality proxy is close to BF16 (`+0.0405` PPL on 512 tokens), but current runtime logs `FP8 W8A16 using per-forward BF16 dequant matmul`, so decode is substantially slower. Treat W8A16 as a stable memory/quality milestone, not a speed claim yet.

### 7B W8A16 + FP8 KV Result

Config: same W8A16 checkpoint, `output_len=16`, native FP8 paged decode, BF16 KV reference and FP8 KV in the same smoke script.

| Prompt | Exit | KV mode | KV storage | Prefill TPS | Decode TPS | Wall time | Peak reserved | Token match |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 0 | BF16 KV | 11.09 GB | 4224.7 | 8.48 | 3.71 s | 21.16 GB | reference |
| 8192 | 0 | FP8 KV | 2.90 GB | 7811.0 | 8.63 | 2.79 s | 21.17 GB | 1/16 |
| 16384 | 0 | BF16 KV | 10.01 GB | 5673.4 | 9.27 | 4.51 s | 21.31 GB | reference |
| 16384 | 0 | FP8 KV | 1.81 GB | 9623.6 | 10.35 | 3.15 s | 21.31 GB | 2/16 |
| 32512 | 1 | FP8 KV attempt | n/a | n/a | n/a | n/a | n/a | allocation/OOM boundary |

Storage conclusion: FP8 KV total storage ratio is `26.1%` of BF16 at 8K and `18.0%` at 16K in this allocation regime; per-block total bytes are `50.8%` of BF16 including scale overhead. Quality conclusion: current native FP8 KV is not quality-stable because token match is only `1/16` at 8K and `2/16` at 16K. The next FP8 KV task is logits-divergence debugging, not README promotion as a default serving mode.

32K boundary: with `gpu_memory_utilization=0.9`, FP8 KV allocation ended with zero available KV blocks; with `0.95`, the BF16 reference path OOMed during 32K prefill MLP activation before FP8 comparison. Need a dedicated FP8-only 32K capacity script or chunked prefill integration to separate capacity from BF16-reference memory pressure.

### Documentation Sync

Updated `README.md` and `README.zh-CN.md` with the 7B result tables and current conclusion. Updated `todo.md` to make 7B the active quantization target and to record the FP8 KV quality/32K blockers.
