# nano-vllm-moe Roadmap

The active goal is to keep the project readable and keep each optimization track tied to one clear story.

## Active Tracks

| Track | Status | Next Gate |
| --- | --- | --- |
| MoE runtime | Stage-stable; CUDA Graph default-on for single GPU | Hold `optimized` as the only non-eager backend; revisit EP graph + FP8 KV graph next |
| Chunked prefill | Mainline feature | Keep `prefill_first` and `decode_first` policy evidence current |
| Prefix cache | Mainline feature | Keep hash/radix benchmark evidence current |
| Quantization | Active | Follow the clean W8A16/W8A8 split below |
| KV cache quantization | Active | Full FP8 KV fails the 4K/8K PPL smoke; use mixed `group=32`. Same-capacity 512-block memory check now shows `0.5195x` KV arena and up to `3.36 GiB` lower PyTorch peak on 7B. |
| EP | Baseline only | Keep torch all-to-all semantics as the reference path |

## Quantization Mainline

The active quantization route is now intentionally narrow:

- W8A16: one simple Triton scheme to prove FP8-weight memory saving and decode behavior.
- W8A8: Torch baseline -> PTX small-M -> CUTLASS large-M -> benchmark-selected M buckets.
- KV cache: K-int8/V-FP8 for long-context memory pressure; strict token exact alignment exposed seed-sensitive drift, but teacher-forced WikiText PPL smoke shows K group-8 native is close to BF16 through 32K. `group=32` now also passes 4K/8K (`+1.20%` PPL), 16K (`+1.55%`), and 32K (`-0.16%`) smoke at `0.5195x` BF16 bytes/block. Full FP8 KV fails the 4K/8K PPL smoke (`8.47` -> `1509.77` PPL), so the active memory path remains mixed `group=32`. With fixed 512 KV blocks, mixed KV reduces KV arena from `7.00 GiB` to `3.64 GiB`; 8K/64 generation peak allocated drops by `3.36 GiB`.
- Other historical routes are not mainline. Do not promote them in README, roadmap, or default runtime policy.

## MoE CUDA Graph

Source of truth: `opt/moe_cuda_graph.md`.

Latest remote evidence, capture fix (Q1–Q4) on `optimized`:

- Environment: RTX 4080 SUPER, driver `580.142`, Python `3.12.3`.
- Workload: Qwen1.5-MoE-A2.7B-Chat, TP=1, EP=1, 8 fixed prompts, input 16 / output 16, `optimized` expert backend.
- `fused` backend removed; `optimized` is now the sole non-eager MoE backend.
- `eager` backend, `--enforce-eager` (Python per-token loop): wall `3.268 s`, prefill `376.14`, decode `40.98` tok/s — `1.00x` baseline.
- `optimized` backend, `--enforce-eager` (Triton fused MoE): wall `1.142 s`, prefill `1213.93`, decode `115.81` tok/s — `2.83x` decode vs eager backend.
- `optimized` backend, CUDA Graph (this work): wall `0.747 s`, prefill `1242.94`, decode `186.45` tok/s — `4.55x` decode vs eager backend, `1.61x` over `optimized` eager.
- Greedy fixed-prompt match: `128 / 128` tokens, `8 / 8` rows — exact equality vs eager.
- Root cause was `torch.bincount` host-syncing inside `moe_align_block_size_fixed`; fixed via `scatter_add_`. Side-stream / dynamic-tensor / JIT issues also addressed via Q1/Q2/Q3.
- Evidence: `.remote-logs/moe_cuda_graph_capture_fix_20260520/{device.txt,eager_repeat3,graph_repeat3,greedy_eager,greedy_graph}.json`.

Keep:

- `optimized` backend as the default CUDA Graph path on single GPU (`ep_size=1`).
- Existing `ep_size>1` eager fallback for dynamic all-to-all.

Next gate:

- EP > 1: replace `torch_alltoall.uses_dynamic_alltoall=True` with a padded-capacity all-to-all that lets `ep_size>1` capture.
- FP8 KV + CUDA graph: remove `.item()` from FP8 attention / kv-cache decode paths before promoting FP8 KV under graph.
- Larger M: validate the same capture-fix on larger decode shapes (16, 32, 64) and on the 7B Qwen2.5-MoE / Qwen3-MoE models when checkpoints are available.

## W8A16

Source of truth: `opt/w8a16.md`.

Latest remote evidence:

- 7B W8A16 checkpoint exported from BF16 7B, quantized `252/434` tensors.
- Model size bytes: BF16 `6,183,464,346`, W8A16 `3,413,055,132` (`44.8%` smaller).
- Memory workload, input 512 / output 16: BF16 decode `130.69 tok/s`, W8A16 decode `158.36 tok/s`.
- W8A16 decode workload, input 512 / output 128 / warmup 1 / repeat 3: decode `158.73 tok/s`.
- Evidence: `.remote-logs/quantization/7b_mainline/{bf16_memory,w8a16_memory,w8a16_decode}.json`.

Keep:

- FP8 weights.
- BF16 activations.
- Triton on-the-fly dequant GEMM.
- Claims around checkpoint size, quality, and decode.

Drop from the active story:

- BF16 weight cache.
- Hybrid cache variants.
- W8A16 CUDA/PTX.
- Down-projection special cases.
- Large-prefill optimization claims.

## W8A8

Source of truth: `opt/w8a8.md`.

Current state:

- Current remote "7B" paths were rechecked and are actually 3B-scale configs: `hidden_size=2048`, `intermediate_size=11008`, `num_hidden_layers=36`, about `3.09B` BF16 safetensors parameters. Do not report them as true 7B results.
- 3B-scale W8A8 static checkpoint exported with WikiText-2 parquet calibration from `/root/autodl-tmp/datasets/_raw/wikitext-2-raw-v1`, quantized `252/434` tensors.
- First bucket table completed on the custom-epilogue CUTLASS path for `M=1..1024` and q/up/down 3B-scale shapes.
- Evidence: `.remote-logs/w8a8_bucket_20260519/m{1,2,4,8,16,32,64,128,256,512,1024}.{json,csv,log}` and `.remote-logs/w8a8_bucket_20260519/all.csv`.
- Native BF16 q/up/down microbench was added and merged with existing FP8 Torch/CUTLASS data; evidence: `.remote-logs/w8a8_3b_bf16_native_20260519/` and `.remote-logs/w8a8_3b_three_path_20260519/summary.{json,csv}`.
- 3B-scale end-to-end BF16 vs W8A8 CUTLASS throughput matrix completed for normal serving, input `1K/2K/4K/8K/16K/32K`, phases `prefill/decode/mixed`. 32K uses total-context budget fitting and is no longer skipped. Evidence: `.remote-logs/w8a8_e2e_matrix_3b_20260519_trace/results.{json,csv}` and `.remote-logs/w8a8_e2e_matrix_3b_20260519_32k_fit/results.{json,csv}`.
- E2E result: W8A8 CUTLASS improves mixed total throughput by `1.35x..1.55x` for 1K..16K prompts and decode throughput by `1.43x..1.55x`; TTFT is lower by about `21%..34%` on the same range.
- True 7B BF16 checkpoint is now present at `/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct`: `hidden_size=3584`, `intermediate_size=18944`, `num_hidden_layers=28`, BF16 safetensors about `14.19 GB`.
- True 7B W8A8 static checkpoint exported at `/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct-FP8-W8A8-Static`, quantized `196/339` tensors. Evidence: `.remote-logs/w8a8_7b_export_20260519/export.log`.
- True 7B q/up/down microbench completed for BF16 Torch, FP8 Torch, and FP8 CUTLASS. Evidence: `.remote-logs/w8a8_7b_bf16_native_20260519/` and `.remote-logs/w8a8_7b_three_path_20260519/summary.{json,csv}`.
- True 7B end-to-end BF16 vs W8A8 CUTLASS throughput matrix completed for normal serving, input `1K/2K/4K/8K/16K/32K`, phases `prefill/decode/mixed`; all 36 rows are `ok`. Evidence: `.remote-logs/w8a8_e2e_matrix_7b_20260519/results.{json,csv}`.
- 32K E2E policy: 32K means total context budget. For generation tests with `max_position_embeddings=32768`, the benchmark should reduce actual prompt length to `32768 - output_len` and record both requested and effective lengths. It should not skip solely because `requested_input_len=32768` plus generated tokens exceeds the context limit.
- Result: current `M<=16: PTX, M>16: CUTLASS` auto policy is not supported by the new table. CUTLASS beats the tested PTX configs across the completed small-M table, and Torch `_scaled_mm` remains competitive for several medium-M down/up cases.
- Partial scalar PTX check reached `M=1..16`; `M=32` and `auto_7b` were intentionally interrupted and are not a final conclusion.

Keep:

- Torch `_scaled_mm` as baseline.
- PTX as an experimental decode / small-M candidate until the interrupted `scalar_m32` and `auto_7b` checks are completed.
- CUTLASS for prefill / large M.
- One benchmark table that chooses the auto bucket boundary.

Drop from the active story:

- Triton fused W8A8.
- Shape-aware BF16/W8A16 fallback.
- Dequant-matmul fallback.
- Many backend environment combinations.
- Projection-specific default policy unless benchmark data justifies a simple bucket rule.

## Next Documentation Gate

Before more implementation work, make README match the same story:

| Mode | Mainline Claim |
| --- | --- |
| W8A16 | Simple Triton path for FP8-weight memory saving and decode |
| W8A8 Torch | Reference baseline |
| W8A8 PTX | Small-M / decode optimization |
| W8A8 CUTLASS | Large-M / prefill optimization |
| W8A8 auto | Benchmark-selected M bucket policy |
| K-int8/V-FP8 KV | Long-context KV storage reduction; exact drift is debug evidence, broader PPL/NLL quality is now the active gate |
