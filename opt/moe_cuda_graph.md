# MoE CUDA Graph

This note tracks the current CUDA Graph gate for the MoE runtime.

## Remote Gate

Environment:

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| Driver | 580.142 |
| Memory | 32760 MiB |
| Python | `/root/miniconda3/bin/python`, Python 3.12.3 |
| Model | `/root/autodl-tmp/models/qwen/Qwen1.5-MoE-A2.7B-Chat` |
| Workload | 8 fixed prompts, input 16, output 16 |
| Parallelism | TP=1, EP=1 |
| Backend under main check | `optimized` experts, `torch` EP backend |

Command shape:

```bash
python scripts/moe/moe_backend_bench.py \
  --model-path /root/autodl-tmp/models/qwen/Qwen1.5-MoE-A2.7B-Chat \
  --backends optimized \
  --ep-size 1 \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 8 \
  --num-seqs 8 \
  --min-input-len 16 \
  --max-input-len 16 \
  --min-output-len 16 \
  --max-output-len 16 \
  --fixed-prompts \
  --moe-profile
```

### Eager Baseline

`--enforce-eager`, repeat 3, discard first repeat:

| Metric | Value |
| --- | ---: |
| Successful measured runs | `2 / 2` |
| Wall time mean | `1.1452 s` |
| Wall time stdev | `0.0382 s` |
| Prefill TPS mean | `1254.29 tok/s` |
| Prefill TPS stdev | `15.34 tok/s` |
| Decode TPS mean | `115.12 tok/s` |
| Decode TPS stdev | `4.07 tok/s` |

Measured repeats after discarding warmup:

| Repeat | Wall | Prefill TPS | Decode TPS |
| ---: | ---: | ---: | ---: |
| 1 | `1.1182 s` | `1265.14` | `118.01` |
| 2 | `1.1722 s` | `1243.44` | `112.24` |

MoE profile for the two measured repeats:

| Repeat | Expert avg | Prepare avg | Finalize avg |
| ---: | ---: | ---: | ---: |
| 1 | `1.5097 ms` | `0.0568 ms` | `0.0497 ms` |
| 2 | `1.5669 ms` | `0.0607 ms` | `0.0540 ms` |

### Initial CUDA Graph Result

The non-eager CUDA Graph path did not reach decode timing. It failed during graph capture:

| Backend | Result |
| --- | --- |
| `optimized` | `CUDA error: operation failed due to a previous error during capture` |
| `eager` | `CUDA error: operation failed due to a previous error during capture` |
| `fused` | not measured cleanly in the same process after the capture failure; subsequent runs hit CUDA OOM from retained graph/private-pool state |

This is a negative gate, not a throughput result. The current MoE CUDA Graph state is:

- Single-GPU MoE with `ep_size=1` still fails capture before replay.
- Multi-GPU EP remains outside CUDA Graph because dynamic all-to-all forces eager mode in `Config.__post_init__`.
- The active runtime claim remains MoE local compute and eager generation stability, not MoE CUDA Graph decode acceleration.

Evidence:

- `.remote-logs/moe_cuda_graph_20260520/device.txt`
- `.remote-logs/moe_cuda_graph_20260520/eager_repeat3.json`
- `.remote-logs/moe_cuda_graph_20260520/graph_smoke.json`
- `.remote-logs/moe_cuda_graph_20260520/graph_backends_smoke.json`

## Next Steps

1. Isolate the capture failure with `CUDA_LAUNCH_BLOCKING=1` on a smaller single-forward MoE decode case.
2. Check whether router top-k, token sorting, `moe_align_block_size`, or Triton expert kernels allocate or launch capture-unsafe work during graph capture.
3. After capture succeeds, rerun the same fixed-prompt eager-vs-graph benchmark and require exact output match plus decode TPS improvement before promoting the feature.

## Experimental Token Kernel

An experimental graph-only token-level Triton MoE kernel was added behind:

```bash
NANOVLLM_MOE_GRAPH_TOKEN_KERNEL=1
```

The path avoids the dynamic route sort/bincount alignment that breaks CUDA Graph capture. It is not enabled by default because the correctness gate is not clean yet.

Implementation changes:

- Disable `NANOVLLM_MOE_PROFILE` timing synchronization while the current CUDA stream is capturing.
- Use conservative MoE model warmup before graph capture to avoid graph allocator instability.
- Add fixed-capacity alignment for graph capture experiments.
- Add a token-level Triton graph kernel that computes the routed expert directly from `topk_ids`.

Performance gate, same workload as above, repeat 3, discard first:

| Mode | Wall mean | Prefill TPS mean | Decode TPS mean | Result |
| --- | ---: | ---: | ---: | --- |
| Eager optimized | `1.1237 s` | `1233.06 tok/s` | `117.70 tok/s` | baseline ok |
| Experimental graph token kernel | `0.3962 s` | `1258.84 tok/s` | `407.69 tok/s` | `3.46x` decode speedup, correctness not clean |

Correctness gate:

| Check | Result |
| --- | ---: |
| Greedy rows exact | `6 / 8` rows |
| Token match | `112 / 128` tokens (`0.875`) |

Conclusion:

- CUDA Graph decode can now be made to capture and replay for single-GPU MoE.
- The experimental token kernel is materially faster on the measured decode workload.
- It is not a default optimization because generated tokens are not exact vs eager on the fixed greedy check.

Evidence:

- `.remote-logs/moe_cuda_graph_fix_20260520/eager_token_repeat3.json`
- `.remote-logs/moe_cuda_graph_fix_20260520/graph_token_repeat3.json`
- `.remote-logs/moe_cuda_graph_fix_20260520/output_match_token_kernel_ptact.json`

Next gate:

1. Make token-level graph kernel numerically match the eager/optimized path, or compare logits and prove the drift is acceptable for a broader quality gate.
2. Only enable `NANOVLLM_MOE_GRAPH_TOKEN_KERNEL` by default after exact greedy output match or an explicit quality acceptance criterion passes.
3. Keep EP (`ep_size>1`) eager-only until dynamic all-to-all has a capture-safe implementation.

## Capture Fix (Q1–Q4) — production path

After the experimental token-kernel notes above, the project removed the `fused`
backend and made the default `optimized` backend CUDA Graph safe end-to-end. The
production graph decode path is now the `optimized` backend, no env flag
required.

Root cause of the earlier capture failure, isolated under `CUDA_LAUNCH_BLOCKING=1`:

```
torch.bincount(flat_topk_ids, minlength=num_experts)
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
```

`torch.bincount` host-syncs to size its output even when `minlength` is given,
so any code path that called `moe_align_block_size_fixed` during capture would
abort the graph. Prior eager / optimized / fused failures and the OOM cascade
that followed all chained off this single sync.

Fixes applied (all in `nanovllm/`):

- **Q1**: `models/qwen2_moe.py` `Qwen2MoeSparseMoeBlock.forward` — detect
  `torch.cuda.is_current_stream_capturing()` and force a single-stream path
  (`shared_expert_stream → None`). The side stream is not in the captured set
  and would trigger "operation failed during capture".
- **Q2**: `executor/moe/experts/optimized.py` — `_GraphWorkspace` keyed by
  `num_tasks` provides persistent `num_tokens_post_padded`, `sorted_token_ids`,
  `expert_ids`, `gate_up`, `activated`, `expert_out`, `output`. Workspace is
  populated during the pre-capture warmup forward (regular allocator → stable
  addresses) and replayed by capture, so no `torch.tensor(...)` / `torch.empty`
  happens inside `torch.cuda.graph(...)`.
- **Q3**: `engine/model_runner.py` `capture_cudagraph` — runs the warmup
  forward twice per `bs` and calls `torch.cuda.synchronize()` before
  `with torch.cuda.graph(...)`, so Triton JIT and autotune are complete before
  capture starts.
- **Q4**: `utils/moe.py` `moe_align_block_size_fixed` — `torch.bincount`
  replaced by a static-shape `scatter_add_`. The `optimized.apply` capture
  branch now unconditionally selects the fixed-capacity align (no env flag
  needed at runtime).

`fused` backend removed. `optimized` is the only non-eager MoE backend.

Result on the same workload as above (Qwen1.5-MoE-A2.7B-Chat, TP=1, EP=1,
8 fixed prompts, input 16 / output 16, RTX 4080 SUPER, repeat 3):

| Mode | Wall mean | Prefill TPS | Decode TPS | Expert avg | Greedy match vs eager |
| --- | ---: | ---: | ---: | ---: | --- |
| `eager` backend, `--enforce-eager` (Python per-token expert loop) | `3.268 s` | `376.14` | `40.98` | `7.13 ms` | (reference for eager backend) |
| `optimized` backend, `--enforce-eager` | `1.142 s` | `1213.93` | `115.81` | `1.55 ms` | reference for production |
| `optimized` backend, CUDA Graph | `0.747 s` | `1242.94` | `186.45` | `2.75 ms` (24 calls vs 384) | `128 / 128` tokens, `8 / 8` rows |

Per-repeat decode TPS:

| Repeat | eager backend | optimized eager | optimized graph |
| ---: | ---: | ---: | ---: |
| `0` | `40.69` | `115.80` | `186.28` |
| `1` | `41.08` | `115.56` | `186.50` |
| `2` | `41.17` | `116.06` | `186.57` |

Speedups vs `eager` backend baseline:

| Step | Decode TPS | Speedup |
| --- | ---: | ---: |
| `eager` backend | `40.98` | `1.00x` |
| `optimized` backend, `--enforce-eager` (Triton fused MoE) | `115.81` | `2.83x` |
| `optimized` backend, CUDA Graph (this work) | `186.45` | `4.55x` |

End-to-end wall: `3.268 s` → `1.142 s` → `0.747 s` (`2.86x` then `1.53x`,
cumulative `4.38x`). No correctness regression at any step. The CUDA Graph
contribution alone is `1.61x` decode TPS on top of the already-fast
`optimized` kernel.

The CUDA Graph "Expert avg" of `2.75 ms` is misleadingly larger than the
optimized eager `1.55 ms` because the timing hook only fires 24 times instead
of 384 — each captured layer's expert work is merged into one block of replay
time. End-to-end decode TPS is the correct lens.

Evidence:

- `.remote-logs/moe_cuda_graph_capture_fix_20260520/device.txt`
- `.remote-logs/moe_cuda_graph_capture_fix_20260520/eager_backend_repeat3.json`
- `.remote-logs/moe_cuda_graph_capture_fix_20260520/eager_repeat3.json`
- `.remote-logs/moe_cuda_graph_capture_fix_20260520/graph_repeat3.json`
- `.remote-logs/moe_cuda_graph_capture_fix_20260520/graph_blocking_repeat1.log`
- `.remote-logs/moe_cuda_graph_capture_fix_20260520/greedy_eager.json`
- `.remote-logs/moe_cuda_graph_capture_fix_20260520/greedy_graph.json`

### Investigation process and gotchas

Recording what actually happened during the capture-fix work, in case the same
class of issue resurfaces. Suspicion order at the start of the session was
Q1 → Q2 → Q3 → Q4; the actual root cause turned out to be Q4, so the order
below is the order things became known.

1. **The bench script swallowed the real traceback.**
   `scripts/moe/moe_backend_bench.py:104-159` wraps the LLM construction in
   `try/except Exception as exc:` and only stores `repr(exc)` in the result
   dict. The reported error from the failing run was a single line:
   `operation failed due to a previous error during capture`, with no
   stacktrace pointing at the offending kernel. Lesson: when CUDA Graph
   capture fails, drop into a `/tmp/<name>.py` standalone repro that calls
   `LLM(...)` with no exception handler, run it under
   `CUDA_LAUNCH_BLOCKING=1`, and read the *first* traceback in the chain.
   That first frame is the real culprit; everything after it is just
   `capture_end()` propagating the failure.

2. **OOM cascade hid the failure on repeats > 0.**
   Repeat 0 of the failing benchmark reported
   `operation failed during capture`. Repeats 1 and 2 reported
   `OutOfMemoryError: Tried to allocate 660.00 MiB. GPU 0 has a total
   capacity of 31.47 GiB of which 566.62 MiB is free`. Both OOMs were
   consequences of the first failed capture retaining ~30 GiB of
   graph / private-pool state that wasn't released until the process exited.
   Lesson: when debugging capture, use `--repeat 1` so you only see the
   first (real) error.

3. **Wrong root-cause hypothesis: side stream (Q1) alone was not enough.**
   Initial reading of the code path identified `Qwen2MoeSparseMoeBlock`'s
   `shared_expert_stream` overlap as the most suspicious capture-time
   work, since side streams aren't part of `torch.cuda.graph(...)`'s capture
   set. After implementing Q1 the capture still failed in exactly the same
   way. Q1 is still required (it would fail later on any workload that
   actually exercised the side stream), but it wasn't *the* cause on this
   workload. Lesson: don't conclude a fix is needed from code reading
   alone; verify with a fresh capture attempt before moving to the next
   hypothesis.

4. **Q4 was the real cause: `torch.bincount` host-syncs during capture.**
   Once the standalone repro printed a real traceback, it pointed at
   `nanovllm/utils/moe.py:77`:
   ```python
   tokens_per_expert = torch.bincount(flat_topk_ids, minlength=num_experts)
   ```
   with `CUDA error: operation not permitted when stream is capturing`.
   Even with `minlength` set, PyTorch's CUDA `bincount` computes
   `max(input)` to size its output and synchronizes to CPU to read it. The
   fix is a static-shape `scatter_add_` into a `torch.zeros(num_experts)`,
   which has no host sync.

5. **Workspace was nearly a single instance; it had to be a `num_tasks` dict.**
   The first draft of `_GraphWorkspace` stored exactly one workspace on
   each `OptimizedExperts` instance and replaced it whenever `num_tasks`
   changed. That looked fine until you remember that `capture_cudagraph`
   loops over `graph_bs = [1, 2, 4, 8, 16, ..., max_bs]`, each with its own
   `num_tasks`. With a single workspace, every iteration after the first
   would invalidate the current workspace and allocate a new one — exactly
   inside `with torch.cuda.graph(...)`. The final design keys the
   workspace dict by `num_tasks`, so each captured `bs` lazily allocates
   its workspace during *warmup* (regular allocator, stable address), and
   the capture call finds it and reuses it.

6. **Workspace must be created during warmup, not inside capture.**
   This is what makes `NANOVLLM_MOE_GRAPH_ALIGN=1` the right gating
   condition instead of `torch.cuda.is_current_stream_capturing()` alone.
   `capture_cudagraph` already sets that env var around the whole
   `for bs in graph_bs:` body, including the warmup forward(s). Because
   warmup runs first and triggers the same code path, the workspace
   tensors are allocated by the regular caching allocator with stable
   device addresses; capture then sees those addresses and bakes them
   into the graph. If we had gated only on `is_current_stream_capturing()`,
   the first capture call would have allocated through the graph's
   private pool, which is technically supported but couples graph life
   cycle to workspace life cycle and was the kind of behavior we set out
   to avoid for Q2.

7. **Q3 (double warmup + sync) is defensive but not visibly load-bearing here.**
   On this workload, capture also passed with a single warmup. The double
   warmup + `torch.cuda.synchronize()` was kept because Triton kernel JIT
   and autotune happen lazily on first call, and a single warmup is not
   guaranteed to drain all of it before the capture's first kernel
   launch. The cost is a few extra ms per captured `bs` at startup,
   spent once.

8. **CUDA Graph "Expert avg" metric is misleading and needs explanation.**
   `NANOVLLM_MOE_PROFILE=1` wraps each `apply()` call in a CUDA event
   timer. Under eager, this fires 384 times per repeat (16 decode steps
   × 24 layers). Under graph, the timer wraps the *Python* call that
   issues `graph.replay()` once per decode step, so it fires 24 times per
   repeat and each tick is much fatter. The decode TPS, not the
   per-call expert avg, is the correct comparison lens.

Out of scope (deferred):

- FP8 KV cache + CUDA graph: `nanovllm/layers/attention.py:245, 292` and
  `nanovllm/layers/kv_cache_kernels.py:204, 319` still call
  `context.context_lens.max().item()` on FP8 KV paths; needs replacement
  before FP8 KV runs under graph.
- EP > 1: `torch_alltoall.py` keeps `uses_dynamic_alltoall=True` and is
  force-eager. A padded-capacity all-to-all is required for graph compatibility.
- Larger batch / 7B MoE: only validated on `bs=8` decode and Qwen1.5-MoE-A2.7B.
  Bigger decode shapes and the 7B / Qwen3-MoE families still need a sweep.
