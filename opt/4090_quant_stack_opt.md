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

## 2026-05-14 Route Review

### Current State

The staged 4090 route is still the right order: BF16 baseline -> W8A16 -> W8A16 + FP8 KV -> W8A8. BF16 and W8A16 already have 7B evidence under `.remote-logs/quantization/4090_7b_stack_20260513/`. W8A16 is quality-stable enough to keep as the balanced checkpoint target, but it is not a speed win on RTX 4090 because SM89 falls back to per-forward BF16 dequant matmul. FP8 KV has strong storage evidence at 8K/16K but fails the quality gate because generated token match is only `1/16` and `2/16`.

### Implementation Hooks

- W8A16/W8A8 runtime is centered in `nanovllm/quantization/fp8.py` and `nanovllm/quantization/kernels/fp8.py`.
- W8A16 on SM89 currently selects `w8a16_dequant_matmul`; the Triton weight-only path is only selected for capability >= 9.0.
- W8A8 static already has a checkpoint contract with `input_scale`, `qweight_scaled_mm`, `weight_scale_scaled_mm`, activation quant backends, and `torch._scaled_mm` path.
- FP8 KV cache is centered in `nanovllm/layers/attention.py`, `nanovllm/layers/kv_cache_kernels.py`, and `nanovllm/layers/fp8_paged_attention.py`.
- The 7B orchestration entrypoint remains `scripts/quantization/run_4090_7b_stack.py`; targeted KV debug uses `scripts/kv_cache/kv_cache_fp8_logits.py`, `kv_cache_fp8_accuracy_suite.py`, and `fp8_paged_attention_microbench.py`.

### Next Execution Order

1. Reproduce FP8 KV divergence with a small deterministic logits trace and compare native paged decode against gather-dequant/full-dequant, so the bug is isolated to store scales, native attention math, or FP8 precision itself.
2. If gather/full dequant matches BF16 better than native, fix `fp8_paged_attention.py`; if all FP8 paths diverge similarly, inspect scale granularity and store quantization in `kv_cache_kernels.py`.
3. Add a FP8-only 32K capacity mode or chunked-prefill capacity script so long-context capacity can be measured without the BF16 reference path OOMing first.
4. In parallel after KV correctness is understood, evaluate an SM89 W8A16 speed improvement: either enable/retune a Triton weight-only kernel for SM89 or cache/dequant weights once at load time for a speed baseline, while preserving the memory-mode checkpoint result separately.
5. Only after W8A16 + FP8 KV has a bounded quality story, run W8A8 static export and `fp8_linear_microbench.py` across representative Qwen2.5-7B linear shapes; promote W8A8 only for shapes where full activation-quant + scaled-mm time beats BF16/W8A16.

### Decision

Do not jump directly to full W8A8 serving. The immediate blocker for the full route is FP8 KV quality, followed by SM89 W8A16 runtime speed. W8A8 is implemented enough for profiling, but should remain experimental until the balanced and long-context stages have defensible gates.

## 2026-05-14 FP8 KV Backend Isolation

### Scope

Extended `scripts/kv_cache/kv_cache_fp8_logits.py` so one run can compare FP8 KV decode backends against BF16 and against each other. The backends are:

- `native`: custom FP8 paged attention reads FP8 KV directly.
- `gather_dequant`: gather only decode-visible FP8 KV tokens, dequantize them, then call FlashAttention.
- `full_dequant`: dequantize the full FP8 cache, then call FlashAttention.

Added `--isolated-runs` because Qwen2.5-7B W8A16 OOMs when BF16 and all FP8 variants are loaded sequentially in one process.

### Remote Evidence

- Environment: `.remote-logs/kv_debug_20260514/env.log`
- 0.5B script smoke: `.remote-logs/kv_debug_20260514/0p5b_len256_all_backends_retry.json`
- 7B W8A16 float16 scale: `.remote-logs/kv_debug_20260514/7b_w8a16_len512_all_backends_isolated.json`
- 7B W8A16 float32 scale: `.remote-logs/kv_debug_20260514/7b_w8a16_len512_float32scale.json`
- Failed non-isolated 7B attempt: `.remote-logs/kv_debug_20260514/7b_w8a16_len512_all_backends.log` failed with CUDA OOM during repeated model loads.

### Result

0.5B smoke passed: all three FP8 backends matched BF16 tokens for `input_len=256`, `output_len=4`; `gather_dequant` and `full_dequant` were bitwise identical in logits (`max_abs=0.0`).

7B W8A16 at `input_len=512`, `output_len=8`, `max_model_len=1024`, `gpu_memory_utilization=0.9` reproduced the divergence at small scale:

| Scale dtype | Backend | Token match vs BF16 | First mismatch | Argmax match | Cosine mean | Top-k overlap | Max abs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float16 | native | 0.25 | 2 | 0.25 | 0.6990 | 0.4500 | 25.1875 |
| float16 | gather_dequant | 0.25 | 2 | 0.25 | 0.6960 | 0.4625 | 25.4375 |
| float16 | full_dequant | 0.25 | 2 | 0.25 | 0.6960 | 0.4625 | 25.4375 |
| float32 | gather_dequant | 0.25 | 2 | 0.25 | 0.6901 | 0.4938 | 25.0625 |
| float32 | full_dequant | 0.25 | 2 | 0.25 | 0.6901 | 0.4938 | 25.0625 |

Pairwise result: `gather_dequant` and `full_dequant` are identical on 7B (`token_match=1.0`, `argmax_match=1.0`, `max_abs=0.0`) for both float16 and float32 scale caches. `native` differs from gather/full, but the major BF16-vs-FP8 divergence is already present in gather/full.

### Conclusion

The primary 7B quality problem is not the gather workspace, full-cache dequant path, or scale-cache dtype. It is also not solely the native paged attention kernel, although native has extra drift versus gather/full. The next debugging target should be FP8 KV storage quantization itself: scale granularity, E4M3 saturation/underflow, K vs V sensitivity, and whether long-context 7B needs a less aggressive scheme such as per-vector/per-channel scale variants or selective BF16 K/V retention.

## 2026-05-14 K/V Sensitivity And V-only FP8 Fix

### Scope

Added a BF16-cache fake FP8 store mode to isolate whether K or V quantization causes the 7B divergence, then added an experimental real `kv_cache_dtype="fp8_v_only"` path that stores K in BF16 and V in FP8 E4M3 with per-token/per-head V scale.

### Remote Evidence

- K/V fake FP8 ablation: `.remote-logs/kv_debug_20260514/7b_fake_fp8_kv_ablation_retry.json`
- Real V-only FP8, 512 prompt: `.remote-logs/kv_debug_20260514/7b_v_fp8_real.json`
- Real V-only FP8, 8K prompt: `.remote-logs/kv_debug_20260514/7b_v_fp8_len8192.json`

### Result

| Mode | Prompt | Token match | Argmax match | Cosine mean | Top-k overlap | Max abs | KV bytes/block vs BF16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fake K-only FP8 | 512 | 0.25 | 0.25 | 0.6914 | 0.4688 | 25.0 | n/a |
| fake V-only FP8 | 512 | 1.00 | 1.00 | 0.99994 | 0.9938 | 0.25 | n/a |
| fake K+V FP8 | 512 | 0.25 | 0.25 | 0.6912 | 0.4625 | 25.0625 | n/a |
| real V-only FP8 | 512 | 1.00 | 1.00 | 0.99992 | 0.9938 | 0.25 | 0.7539 |
| real V-only FP8 | 8192 | 1.00 | 1.00 | 0.99996 | 0.9813 | 0.1875 | 0.7539 |

### Conclusion

The 7B divergence is driven by FP8 K cache quantization. V cache is much more tolerant: real K-BF16/V-FP8 keeps exact generated-token match on the tested 512 and 8K prompts while reducing per-block KV storage to `75.39%` of BF16 including scale overhead, and increasing available KV block count by about `1.33x`. Treat V-only FP8 as the current safe long-context direction; keep full K+V FP8 experimental until a better K quantization scheme is found.

## 2026-05-14 V-only Suite And K Quantization Probe

### V-only Suite Integration

`kv_cache_fp8_smoke.py` now accepts `--kv-cache-dtype fp8_v_only`, and `run_4090_7b_stack.py` defaults the KV stage to `fp8_v_only`. This makes the validated K-BF16/V-FP8 path the suite default while full K+V FP8 remains selectable with `--kv-cache-dtype fp8_e4m3`.

Remote evidence: `.remote-logs/kv_debug_20260514/v_only_suite_8k16k/`.

| Prompt | Token match | Exact | V-only prefill TPS | V-only decode TPS | Total bytes/block vs BF16 | Block ratio vs BF16 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 8192 | 1.00 | yes | 10008.1 | 11.11 | 0.7539 | 0.3465 |
| 16384 | 1.00 | yes | 9249.1 | 12.14 | 0.7539 | 0.2391 |

The storage ratios in these smoke outputs are affected by different allocated block counts between BF16 reference and quantized runs; the stable per-block conclusion is that K-BF16/V-FP8 uses `75.39%` of BF16 KV bytes per block including V scales.

### K Quantization Probe

Added fake K quantization knobs to `kv_cache_fp8_logits.py`: `--fake-fp8-format {e4m3,e5m2,int8}` and `--fake-fp8-group-size`. This keeps the BF16 cache path but round-trips K through candidate quantizers before storage, so it isolates K quantization quality without implementing a full K quantized cache yet.

Remote evidence: `.remote-logs/kv_debug_20260514/k_ablation/`.

| K quant | Group | Token match | Exact | Argmax match | Cosine mean | Top-k overlap | Max abs |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| E4M3 | vector | 0.25 | no | 0.25 | 0.6914 | 0.4688 | 25.0 |
| E4M3 | 16 | 0.625 | no | 0.625 | 0.9295 | 0.7688 | 22.1719 |
| E4M3 | 32 | 0.625 | no | 0.625 | 0.9331 | 0.7625 | 22.2188 |
| E4M3 | 64 | 0.625 | no | 0.625 | 0.9324 | 0.7625 | 22.1875 |
| E5M2 | vector | 0.75 | no | 0.75 | 0.9164 | 0.7000 | 21.8906 |
| E5M2 | 16 | 0.375 | no | 0.375 | 0.6760 | 0.5438 | 29.5156 |
| E5M2 | 32 | 0.375 | no | 0.375 | 0.7917 | 0.5625 | 23.8750 |
| E5M2 | 64 | 0.375 | no | 0.375 | 0.6623 | 0.5312 | 29.2656 |
| int8 | vector | 1.00 | yes | 1.00 | 0.9987 | 0.9875 | 1.9375 |
| int8 | 16 | 1.00 | yes | 1.00 | 0.9998 | 1.0000 | 0.5000 |
| int8 | 32 | 1.00 | yes | 1.00 | 0.9998 | 1.0000 | 0.5000 |
| int8 | 64 | 1.00 | yes | 1.00 | 0.9996 | 0.9875 | 0.8438 |

### Decision

K should not use the current FP8 E4M3 storage. E4M3 group scaling improves logits but still misses tokens, and E5M2 is not consistently better. Symmetric int8 K with group size 16 or 32 is the first promising K compression direction. The next implementation target is a hybrid K-int8/V-FP8 cache, likely with per-token/per-head/per-group K scale and the existing V-FP8 path.

## 2026-05-14 K-int8/V-FP8 Mixed KV Gate

### Scope

Implemented experimental `kv_cache_dtype="k_int8_v_fp8"`: K is stored as symmetric int8 with per-token/per-head/group-32 scale, V is stored as FP8 E4M3 with per-token/per-head scale. Decode currently dequantizes the full mixed cache back to BF16 before FlashAttention, so this is a precision/capacity gate rather than the final optimized performance path.

### Remote Evidence

- Root: `.remote-logs/kv_debug_20260514/k_int8_v_fp8/`
- 512 prompt: `.remote-logs/kv_debug_20260514/k_int8_v_fp8/512.json`
- 8K prompt: `.remote-logs/kv_debug_20260514/k_int8_v_fp8/8192.json`
- 16K prompt: `.remote-logs/kv_debug_20260514/k_int8_v_fp8/16384.json`

### Result

| Prompt | Token match | Exact | Mixed prefill TPS | Mixed decode TPS | Wall time | Total bytes/block vs BF16 | Block ratio vs BF16 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 1.00 | yes | 4527.3 | 6.66 | 2.37 s | 0.5195 | 0.6096 |
| 8192 | 1.00 | yes | 9931.0 | 7.49 | 2.83 s | 0.5195 | 0.5105 |
| 16384 | 1.00 | yes | 9241.9 | 9.14 | 3.41 s | 0.5195 | 0.3525 |

### Conclusion

The mixed KV precision gate passes on the tested 512/8K/16K prompts. Compared with BF16 KV, K-int8/V-FP8 reduces per-block KV bytes to `51.95%` including K/V scales while keeping exact generated tokens in these probes. This should become the main KV compression track. Performance is not final because the current decode path still full-dequantizes mixed KV before FlashAttention; optimized mixed-KV attention can be addressed after precision gates are broader.

## 2026-05-14 Mixed KV Capacity And Native Decode Optimization

### Scope

Fixed mixed KV capacity accounting, parameterized the logits accuracy suite with `--kv-cache-dtype k_int8_v_fp8`, added a Triton fused store kernel for K-int8/V-FP8, and added a decode-only native paged attention kernel that reads int8 K + FP8 V directly with scales. Baselines are preserved with `NANOVLLM_K_INT8_V_FP8_STORE=torch` and `--fp8-decode-backend gather_dequant/full_dequant`.

### Remote Evidence

- Root: `.remote-logs/kv_mixed_opt_20260514/`
- 7B full-dequant baseline attempt: `.remote-logs/kv_mixed_opt_20260514/7b_full_dequant_len512.log` failed with CUDA OOM at `gpu_memory_utilization=0.92` because full-cache BF16 dequant needs an additional full KV buffer.
- 512 gather baseline: `.remote-logs/kv_mixed_opt_20260514/7b_gather_dequant_len512/summary.json`
- 512 Triton store + gather: `.remote-logs/kv_mixed_opt_20260514/7b_gather_store_triton_len512/summary.json`
- 512 Triton store + native: `.remote-logs/kv_mixed_opt_20260514/7b_native_store_triton_len512/summary.json`
- 8K Triton store + gather: `.remote-logs/kv_mixed_opt_20260514/7b_gather_store_triton_len8192/summary.json`
- 8K Triton store + native: `.remote-logs/kv_mixed_opt_20260514/7b_native_store_triton_len8192/summary.json`

### Result

| Prompt | Backend | Exact | Cosine mean | Quant/BF16 model TPS | Native/gather profile | Store profile | Bytes/block vs BF16 | Block ratio |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | gather + torch store | yes | 0.99928 | 0.668x | gather `5.00 ms` | torch store `0.49 ms` | 0.5195 | 2.06x |
| 512 | gather + Triton store | yes | 0.99848 | 0.870x | gather `4.90 ms` | Triton store `0.43 ms` | 0.5195 | 1.40x |
| 512 | native + Triton store | yes | 0.99839 | 0.796x | native `5.54 ms` | Triton store `0.12 ms` | 0.5195 | 1.40x |
| 8192 | gather + Triton store | yes | 0.99990 | 1.114x | gather `4.99 ms` | Triton store `0.15 ms` | 0.5195 | 1.57x |
| 8192 | native + Triton store | yes | 0.99989 | 3.372x | native `0.81 ms` | Triton store `0.13 ms` | 0.5195 | 1.56x |

### Conclusion

The optimized path keeps exact generated tokens on the tested 512 and 8K 7B W8A16 gates. The capacity fix reports the expected per-block storage ratio `0.51953125`; mixed KV exposes about `1.56x` more blocks than BF16 on the 8K run. Full-cache dequant is not a viable high-GMU baseline because it OOMs; gather-dequant is the practical correctness baseline, and native mixed paged attention is the performance path. At 8K, native mixed decode raises model TPS from BF16 `4.34` to `14.64` (`3.37x`) and from gather mixed `4.43` to `14.64` (`3.30x`). At 512, native is still slower than BF16/gather because per-head Triton launch and short-context overhead dominate.

### Next

Broaden the 7B native mixed-KV gate to 16K/32K and more seeds, then optimize short-context native decode by increasing work per program or adding a threshold that uses gather/FlashAttention below a context-length cutoff.

## 2026-05-14 Native Mixed KV 16K/32K Extension

### Remote Evidence

- 16K native: `.remote-logs/kv_mixed_opt_20260514/7b_native_store_triton_len16384/summary.json`
- 32K native: `.remote-logs/kv_mixed_opt_20260514/7b_native_store_triton_len32512/summary.json`
- Combined summary: `.remote-logs/kv_mixed_opt_20260514/native_8k16k32k_summary.json` and `.remote-logs/kv_mixed_opt_20260514/native_8k16k32k_summary.csv`

### Result

| Prompt | Exact | Cosine mean | BF16 model TPS | Native mixed model TPS | Speedup | Bytes/block vs BF16 | Block ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | yes | 0.99989 | 4.34 | 14.64 | 3.37x | 0.5195 | 1.56x |
| 16384 | yes | 0.99987 | 4.11 | 10.77 | 2.62x | 0.5195 | 1.57x |
| 32512 | yes | 0.99986 | 4.03 | 6.43 | 1.59x | 0.5195 | 1.53x |

### Conclusion

Native K-int8/V-FP8 keeps exact generated tokens and stable logits on the tested 8K/16K/32K 7B W8A16 prompts. The long-context headline is now README-worthy: `51.95%` per-block KV bytes, about `1.5x` more allocated KV blocks, and `1.59x` to `3.37x` model-TPS speedup over BF16 KV on these gates. The speedup declines at 32K because the current one-program-per-query-head Triton decode scales linearly with context; the next kernel step is to split long contexts across token blocks and reduce partials, while keeping a short-context fallback for 512-token prompts.

## 2026-05-14 W8A8 Shape-aware Runtime Start

### Scope

Restarted W8A8 after the K-int8/V-FP8 gate passed. The first optimization is profile-driven rather than blanket W8A8: default activation quantization now uses the Triton elementwise path, `fused_triton` microbench is guarded because it aborts the SM89 Triton compiler, and runtime uses `_scaled_mm` only for large linear shapes while small 2K attention projections fall back to FP8-weight dequant matmul.

### Remote Evidence

- Root: `.remote-logs/w8a8_opt_20260514/`
- Initial 3B shape profile: `.remote-logs/w8a8_opt_20260514/3b_w8a8_microbench_m512_torch_triton.json`
- Post-patch guarded microbench: `.remote-logs/w8a8_opt_20260514/3b_w8a8_microbench_m512_after_patch.json`
- Additional shapes: `.remote-logs/w8a8_opt_20260514/3b_w8a8_more_shapes_m512_triton.json`
- Shape-aware 3B smoke: `.remote-logs/w8a8_opt_20260514/3b_w8a8_shape_aware_smoke/`
- All-scaled-mm comparison: `.remote-logs/w8a8_opt_20260514/3b_w8a8_all_scaled_mm_smoke/`
- 3B logits/token gate: `.remote-logs/w8a8_opt_20260514/3b_w8a8_shape_aware_compare/`
- 7B shape profile: `.remote-logs/w8a8_opt_20260514/7b_w8a8_shapes_m512_triton.json`
- 7B cutoff-3072 smoke: `.remote-logs/w8a8_opt_20260514/7b_w8a8_shape_aware_cut3072_smoke/`
- Failed 7B HF proxy compare: `.remote-logs/w8a8_opt_20260514/7b_w8a8_shape_aware_compare/` failed with CUDA OOM while loading both BF16 and W8A8 proxy models on 24GB.

### Result

Representative 3B W8A8 linear profile at `M=512`:

| Weight | Shape `(M,K,N)` | W8A8 full vs BF16 | W8A8 full vs W8A16 dequant | Triton act-quant speedup vs torch | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `mlp.gate_proj` | `512,2048,11008` | `0.49x-0.51x` | `0.28x-0.29x` | `1.78x-2.22x` | use W8A8 `_scaled_mm` |
| `mlp.up_proj` | `512,2048,11008` | `0.49x` | `0.28x` | `2.09x` | use W8A8 `_scaled_mm` |
| `mlp.down_proj` | `512,11008,2048` | `0.93x` | `0.46x` | `2.13x` | use W8A8 `_scaled_mm`, marginal vs BF16 |
| `self_attn.q_proj/o_proj` | `512,2048,2048` | `2.61x-2.66x` | `1.49x-1.52x` | `1.73x-1.80x` | fallback from W8A8 `_scaled_mm` |

Model-level 3B smoke with shape-aware runtime passes: checkpoint contract has `252` qweight/weight_scale/input_scale tensors, benchmark prefill TPS is `10047.6`, decode TPS is `87.62`, and memory run reports model size `3.41 GB`. The all-scaled-mm comparison has lower benchmark prefill TPS (`9277.6`) on the same 512/16 smoke, supporting the shape cutoff. The 3B BF16-vs-W8A8 correctness gate passes on three prompts with avg logits cosine `0.998857`, top1 match rate `1.0`, top10 overlap `0.9667`, and exact generated tokens for all prompts.

The 7B W8A8 checkpoint also has a healthy contract (`196` qweight/weight_scale/input_scale tensors). At `M=512`, the 7B shape profile shows W8A8 full vs BF16 ratios of `0.49x` for `gate_proj`, `0.49x` for `up_proj`, `0.78x` for `down_proj`, `0.88x` for `q_proj`, and `0.87x` for `o_proj`, so the default cutoff was lowered from `4096` to `3072`. The cutoff-3072 7B smoke passes with benchmark prefill TPS `8775.6`, decode TPS `61.48`, and memory-run model size `8.72 GB`. The existing HF proxy logits/token compare is not memory-safe for 7B on 24GB because it loads BF16 and W8A8 proxy models together and OOMs.

### Conclusion

W8A8 is now worth continuing, but only as a shape-aware backend. Large MLP projections win clearly because `_scaled_mm` speed offsets activation quantization; 2K attention projections lose because activation quantization dominates, while 7B's 3584-wide attention projections already benefit. The current default cutoff is `NANOVLLM_FP8_W8A8_SCALED_MM_MIN_DIM=3072`. The next gate is a memory-safe 7B PPL/logits path plus longer generation before advertising W8A8 as a complete 7B quality stage.

## 2026-05-15 W8A8 Memory-safe 7B Quality Gate

### Scope

Confirmed the 7B W8A8 checkpoint was calibrated on WikiText-2 validation with `cache_dir=/home/ubuntu/project/datasets`, `samples=32`, `max_length=256`, and `batch_size=2`. Added a memory-safe comparison mode: `compare_logits.py --sequential-load` loads BF16 and W8A8 proxy models one at a time, and `run_quant_suite.py --sequential-compare` passes it through the suite. This avoids the previous 24GB OOM from loading both 7B models concurrently.

### Remote Evidence

- Sequential logits/token gate: `.remote-logs/w8a8_opt_20260515/7b_w8a8_seq_compare/`
- BF16 same-data PPL: `.remote-logs/w8a8_opt_20260515/7b_bf16_wikitext_ppl/`
- W8A8 same-data PPL: `.remote-logs/w8a8_opt_20260515/7b_w8a8_wikitext_ppl/`

### Result

| Gate | BF16 | W8A8 | Notes |
| --- | ---: | ---: | --- |
| WikiText-2 validation PPL, 1024 tokens | `8.3971` | `8.4238` | same dataset/cache as calibration |
| Sequential logits avg cosine | reference | `0.999505` | 3 fixed prompts |
| Sequential top1 match | reference | `0.667` | 2/3 prompts |
| Sequential top10 overlap | reference | `0.967` | 3 fixed prompts |
| Sequential generation exact match | reference | `0.667` | avg token match ratio `0.75` |

### Conclusion

The memory-safe quality gate is now usable on a 24GB RTX 4090. Same-data PPL drift is small (`+0.0267` absolute), and logits cosine stays high, but greedy generation is not exact on all prompts. W8A8 should remain shape-aware and promising, not final-default, until a broader prompt/token gate confirms acceptable generation stability.

## 2026-05-15 MMLU Logit-rank And GSM8K Numeric Probes

### Scope

Added two quality gates beyond fixed prompts. `eval_choice_logits.py` evaluates MMLU/CEval-style multiple-choice tasks by ranking the next-token logits for `A/B/C/D`, loading BF16 and W8A8 sequentially to stay within 24GB. `eval_gsm8k_generate.py` runs a greedy numeric GSM8K probe and extracts the final number from generated text. MMLU/CEval is the primary quantization gate because it avoids long-generation butterfly effects; GSM8K is a generation-sensitivity supplement.

### Remote Evidence

- MMLU smoke, 100 questions: `.remote-logs/w8a8_opt_20260515/7b_w8a8_mmlu_smoke.json`
- MMLU 300 questions: `.remote-logs/w8a8_opt_20260515/7b_w8a8_mmlu_300.json`
- GSM8K 50 questions: `.remote-logs/w8a8_opt_20260515/7b_w8a8_gsm8k_50.json`

### Result

| Gate | Samples | BF16 | W8A8 | Agreement | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| MMLU smoke | 100 | `0.68` | `0.68` | `0.96` | abstract algebra + elementary math |
| MMLU mixed | 300 | `0.6033` | `0.6200` | `0.9567` | 6 subjects, 50 each |
| GSM8K numeric | 50 | `0.32` | `0.36` | `0.28` | greedy generation final-number probe |

### Conclusion

The MMLU logit-rank gate supports W8A8: accuracy does not regress on this 300-question slice, and prediction agreement stays high (`95.67%`). GSM8K accuracy also does not regress on the 50-question slice, but same-number agreement is low because generation diverges frequently; use it as a supplementary generation-stability signal, not as the primary quantization correctness criterion. The next quality step is to broaden MMLU and add CEval subsets before promoting W8A8 as a final 7B quality stage.

## 2026-05-15 W8A16/W8A8 Performance Bottleneck Pass

### Scope

Rechecked the current W8A16 and W8A8 bottlenecks with explicit baseline/candidate A/B runs on the remote RTX 4090. The goal was to keep stable baselines while testing conservative runtime switches, not to promote an unbounded memory-heavy mode as default.

### Remote Evidence

- Root: `.remote-logs/w8_perf_opt_20260515/`
- Earlier W8A8 layer microbench baseline: `.remote-logs/w8_perf_opt_20260515_baseline/` on the remote machine.
- 3B model-level A/B: `3b_w8a16_dequant_512x16.json`, `3b_w8a16_load_dequant_512x16.json`, `3b_w8a8_shape_default_512x16.json`, `3b_w8a8_shape_cache_fallback_512x16.json`, `3b_w8a8_all_load_dequant_512x16.json`.
- 7B model-level A/B: `7b_w8a16_dequant_512x16.json`, `7b_w8a16_load_dequant_attention_512x16.json`; full 7B load-dequant was also tried and failed with CUDA OOM during warmup.

### Runtime Changes

- Added `NANOVLLM_FP8_W8A16_BACKEND` with `auto`, `dequant_matmul`, `load_dequant`, `load_dequant_attention`, and `triton` choices.
- Kept the existing SM89 default safe: `auto` still falls back to per-forward dequant on RTX 4090 because the Triton W8A16 kernel emits SM90-only BF16 conversion instructions and fails ptxas on SM89.
- Added optional load-time BF16 dequant caching for W8A16. Full-model caching is fast on 3B but OOMs on 7B/24GB, so it remains opt-in only.
- Added `load_dequant_attention` as a 7B-safe compromise that caches only attention projections and leaves MLP per-forward dequantized.
- Added optional W8A8 dequant-cache support for fallback layers via `NANOVLLM_FP8_W8A8_CACHE_DEQUANT=1`, while preserving the `NANOVLLM_FP8_W8A8_SCALED_MM_MIN_DIM=3072` shape-aware default.

### 3B Results (`input=512`, `output=16`, warmup 1, repeat 2)

| Mode | Wall time | Prefill TPS | Decode TPS | E2E TPS | Takeaway |
| --- | ---: | ---: | ---: | ---: | --- |
| W8A16 per-forward dequant baseline | `0.4687 s` | `5509.8` | `38.15` | `1127.3` | stable but dequant dominates |
| W8A16 load-time dequant | `0.1679 s` | `9805.8` | `116.75` | `3144.3` | `2.79x` lower wall time, memory-heavy |
| W8A8 shape-aware default | `0.2350 s` | `6544.5` | `86.86` | `2249.8` | still pays fallback dequant overhead |
| W8A8 shape-aware + cached fallback | `0.2062 s` | `7173.7` | `100.61` | `2563.8` | `1.14x` lower wall time vs default |
| W8A8 all cached dequant | `0.1823 s` | `7450.2` | `115.45` | `2897.2` | fastest W8A8 3B mode but loses W8A8 scaled-mm benefit |

### 7B Results (`input=512`, `output=16`, warmup 1, repeat 1)

| Mode | Wall time | Prefill TPS | Decode TPS | E2E TPS | Takeaway |
| --- | ---: | ---: | ---: | ---: | --- |
| W8A16 per-forward dequant baseline | `1.0892 s` | `5108.0` | `15.17` | `484.7` | current safe baseline |
| W8A16 attention-only load-dequant | `1.0273 s` | `5380.2` | `16.10` | `514.0` | small but real `1.06x` wall-time win |
| W8A16 full load-dequant | OOM | n/a | n/a | n/a | exceeds 24GB during warmup |
| W8A16 Triton kernel | compile fail | n/a | n/a | n/a | ptxas rejects SM90-only BF16 conversions on SM89 |

### Conclusion

The immediate W8A16 bottleneck is not checkpoint quality or model loading; it is per-forward FP8-to-BF16 weight dequantization. Load-time BF16 dequant proves the speed ceiling on 3B, but full 7B caching is not viable on a 24GB RTX 4090. The safe 7B direction is selective caching, starting with attention projections for a small win, or a true SM89-compatible weight-only kernel. For W8A8, activation quantization and scaled-mm overhead remain bad for small attention shapes, and fallback layers also suffer from per-forward dequant. Caching fallback dequant improves 3B but should remain opt-in until a 7B memory gate is run.

### Next

1. Add memory reporting for load-time dequant/cache modes to quantify the model-size tradeoff instead of only throughput.
2. Probe `load_dequant_attention` at 7B with `output=64` and mixed KV enabled to see whether decode-heavy workloads amplify the small 512/16 gain.
3. Build or adapt an SM89-safe W8A16 kernel that avoids the current Triton BF16 conversion codegen failure.
4. Run W8A8 fallback-cache on 7B with a conservative memory shape before considering any default change.

## 2026-05-15 SM89 W8A16 On-the-fly Triton Kernel Fix

### Scope

Reworked the W8A16 Triton kernel to avoid the RTX 4090 / SM89 ptxas failure caused by direct FP8-to-BF16 conversion. The kernel now performs on-the-fly dequantization inside the GEMM loop by casting FP8 weights through FP32 and then FP16 before `tl.dot`; activations are also cast through FP32 to FP16 so both dot operands have the same dtype. No load-time BF16 weight cache is used.

### Remote Evidence

- Root: `.remote-logs/w8a16_triton_sm89_20260515/`
- Initial failed log with mixed BF16/FP16 dot: `3b_w8a16_triton_512x16.log`
- Passing 3B Triton run: `3b_w8a16_triton_512x16_fp16dot.json`
- Same-settings 3B per-forward dequant baseline: `3b_w8a16_dequant_same_settings.json`
- Passing 7B Triton smoke: `7b_w8a16_triton_512x16_fp16dot.json`

### Result

| Model | Backend | Wall time | Prefill TPS | Decode TPS | E2E TPS | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3B | per-forward dequant baseline | `1.1368 s` | `692.4` | `37.76` | `464.4` | same command shape, no warmup |
| 3B | Triton FP8->FP32->FP16 on-the-fly | `0.8247 s` | `710.7` | `143.90` | `640.2` | compiles on SM89; no cache |
| 7B | Triton FP8->FP32->FP16 on-the-fly | `0.9960 s` | `650.6` | `71.79` | `530.1` | compiles and runs on SM89; no cache |

### Conclusion

The SM90-only instruction issue is fixed for the W8A16 Triton path: the passing logs show no ptxas BF16 conversion failure on RTX 4090. The current kernel is a real on-the-fly dequantization path and does not allocate BF16 decompressed weights. It is decode-friendly but prefill is still weak, likely because the simple FP16 dot path loses BF16/FP8 tensor-core efficiency and still pays conversion overhead inside the K loop. Keep it as an opt-in/runtime candidate for now while optimizing tile choices and conversion placement.
