# KV Cache 量化主线

## 定位

KV cache 量化只服务一个目标：降低长上下文 KV 显存压力。它不替代 W8A16/W8A8 的权重和激活量化故事。

当前代码只保留两条路径：

```text
BF16 KV reference
K-int8 / V-FP8 mixed KV mainline
```

其他实验分支已经从代码主线移除，只在本文记录为什么最后选择混合量化。

## 为什么不是全 FP8 KV

最直接的方案是把 K 和 V 都存成 FP8。这个方案显存收益明显，但实验暴露出核心问题：

- K 参与 attention score：`q @ k`。
- K 的量化误差会直接改变 softmax 前的 score 分布。
- 生成式 decode 对早期 token 的 argmax 很敏感，K 的 FP8 误差会导致 token drift。

因此，全 FP8 KV 适合做 ablation，不适合作为可维护主线。

## 为什么不是 V-only FP8

V-only FP8 的诊断价值很高：K 保持 BF16，只压 V，可以证明 V 的 FP8 量化相对安全。

但它不是最终路线：

- K 仍占 BF16 存储，显存收益不够彻底。
- 它更像“定位问题在 K”的实验，而不是最终产品形态。

结论是：V 可以大胆压到 FP8，但 K 需要更稳的格式。

## 为什么选择 K-int8 / V-FP8

最终选择混合量化：

```text
K: int8 group quant
V: FP8 E4M3
```

原因：

- K 用 group-wise int8，保留更稳定的 attention-score 路径。
- V 用 FP8 E4M3，降低 value cache 存储。
- K scale 使用 per-token / per-head / group-32。
- V scale 使用 per-token / per-head。
- 总 KV bytes/block 目标约为 BF16 的 `0.52x`，同时保持 8K gate 稳定。

这条路线的能力展示点不是“支持很多 dtype”，而是：

- 通过 ablation 定位 FP8 K 的精度问题。
- 用混合格式平衡 score 稳定性和显存收益。
- 实现 quantized store、scale layout、gather reference 和 native paged decode。
- 用 8K/16K/32K gate 限制 claim。

## 保留的代码路径

### BF16 reference

默认路径：

```text
kv_cache_dtype="bf16"
```

作用：

- 质量 reference。
- FlashAttention reference。
- 对比 KV bytes/block、token match、logits cosine 和 decode latency。

### K-int8 / V-FP8 mainline

实验路径：

```text
kv_cache_dtype="k_int8_v_fp8"
experimental_kv_cache_fp8=True
kv_cache_scale_dtype="float16"
```

代码入口：

| 部分 | 文件 |
| --- | --- |
| dtype 归一化和 guard | `nanovllm/utils/kv_cache.py` |
| block 预算和 cache 分配 | `nanovllm/engine/model_runner.py` |
| Attention 分发 | `nanovllm/layers/attention.py` |
| mixed KV store | `nanovllm/layers/kv_cache_kernels.py` |
| gather-dequant reference | `nanovllm/layers/kv_cache_kernels.py` |
| native paged decode | `nanovllm/layers/fp8_paged_attention.py` |

## Runtime Policy

Runtime policy 保持最小：

```text
bf16:
    标准 FlashAttention 路径

k_int8_v_fp8 + native:
    单序列 decode 使用 native paged attention

k_int8_v_fp8 + gather_dequant:
    reference/debug 路径；gather 可见 token 后反量化，再走 FlashAttention
```

不再维护：

- 全 FP8 K/V runtime。
- V-only FP8 runtime。
- fake quant env knobs。
- full-dequant decode backend。
- 每个长度、每个 layer、每个 backend 的特例 policy。

## 操作入口

### Smoke

```bash
python scripts/kv_cache/kv_cache_fp8_smoke.py \
  --model-path /path/to/model \
  --kv-cache-dtype k_int8_v_fp8 \
  --fp8-decode-backend native \
  --input-len 8192 \
  --output-len 8 \
  --max-model-len 9216 \
  --output-json .remote-logs/kv_cache/smoke.json
```

看这些字段：

- `comparison.exact_sequence_match`
- `comparison.token_match_rate`
- `total_bytes_per_block_ratio_quant_over_bf16`
- `block_ratio_quant_over_bf16`

### Accuracy Suite

```bash
python scripts/kv_cache/kv_cache_fp8_accuracy_suite.py \
  --model-path /path/to/model \
  --kv-cache-dtype k_int8_v_fp8 \
  --fp8-decode-backend native \
  --input-len 8192 \
  --output-len 8 \
  --seeds 0,1,2 \
  --prompt-ids 0,1,2 \
  --output-dir .remote-logs/kv_cache/accuracy
```

看这些字段：

- `exact_match_rate`
- `token_match_rate_mean`
- `cosine_mean`
- `top_k_overlap_mean`
- `model_tps_ratio_quant_over_bf16`
- `total_bytes_per_block_ratio_quant_over_bf16`

### Logits Trace

7B/8192 之后的端到端 logits 排查不要再把 logits 本体写进 JSON，也不要依赖同一个 Python 进程顺序加载 BF16/native/gather。改用分进程 `.pt` trace：

```bash
python scripts/kv_cache/kv_cache_logits_dump.py \
  --model-path /path/to/model \
  --backend bf16 \
  --input-len 8192 \
  --output-len 8 \
  --max-model-len 9216 \
  --max-num-batched-tokens 9216 \
  --output-pt .remote-logs/kv_cache/bf16.pt \
  --output-json .remote-logs/kv_cache/bf16.json

python scripts/kv_cache/kv_cache_logits_dump.py \
  --model-path /path/to/model \
  --backend native \
  --input-len 8192 \
  --output-len 8 \
  --max-model-len 9216 \
  --max-num-batched-tokens 9216 \
  --gpu-memory-utilization 0.90 \
  --output-pt .remote-logs/kv_cache/native.pt \
  --output-json .remote-logs/kv_cache/native.json

python scripts/kv_cache/compare_logits_pt.py \
  --left-pt .remote-logs/kv_cache/bf16.pt \
  --right-pt .remote-logs/kv_cache/native.pt \
  --output-json .remote-logs/kv_cache/bf16_vs_native.json
```

原因：

- logits tensor 用 `torch.save` 保留原始 tensor 精度，避免 JSON 文本浮点和庞大文件开销。
- 每个 backend 单独进程运行，避免 7B/8192 下同进程多次建模带来的 CUDA cache、Inductor 编译缓存和碎片 OOM。
- JSON 只保存 metadata、token、storage ratio、summary 和 `.pt` 路径。

旧的同进程 trace 入口仍可用于短上下文 smoke：

```bash
python scripts/kv_cache/kv_cache_fp8_logits.py \
  --model-path /path/to/model \
  --fp8-decode-backends native,gather_dequant \
  --input-len 8192 \
  --output-len 8 \
  --output-json .remote-logs/kv_cache/logits.json
```

判断逻辑：

- native drift、gather 不 drift：优先查 native paged attention。
- native 和 gather 都 drift：优先查 store quantization、scale granularity 或混合格式本身。

## Benchmark Gate

只维护这张 gate 表：

| Gate | 目标 |
| --- | --- |
| 8K prompt | exact match 或 token match 接近 1.0 |
| 16K prompt | 记录首次 drift token；稳定后再提升为 claim |
| 32K prompt | 先证明 capacity，不强行和 BF16 reference 同驻 |
| bytes/block | 明确低于 BF16，目标约 `0.52x` |
| block ratio | 明确高于 BF16，展示可用 KV block 增益 |
| decode backend | native 是主线；gather_dequant 只作 reference |

## 7B BF16 权重 KV-only 崩点扫描

本轮只测试 KV 量化，不启用 W8A16/W8A8 权重量化：

```text
model=/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct
weights=BF16
kv_cache_dtype=k_int8_v_fp8
kv_cache_scale_dtype=float16
decode_backends=native,gather_dequant
gpu_memory_utilization=0.99
```

远端证据：

- 粗扫：`.remote-logs/kv_cache_collapse_20260519_kint8/`
- 8K 附近细扫：`.remote-logs/kv_cache_collapse_20260519_kint8/fine_8k/`
- 64-token decode 压力：`.remote-logs/kv_cache_collapse_20260519_kint8/out64/`

### 结论

K-int8/V-FP8 KV 在 7B 上不是固定到 8K 必崩，但从 7K 到 9K 开始出现 prompt/seed 敏感的 token drift。当前不能继续把 8K 描述成宽泛稳定 gate；更准确的状态是：部分 8K/short-decode case 稳定，但真实长文本或不同 seed 会触发 drift。

同时发现两类问题：

1. native paged attention 的特定 case 不稳定。
   - `input_len=8192, seed=0, output_len=8`
   - native：第 `3` 个 decode token drift，token match `0.375`，logits cosine min `0.690786`
   - gather_dequant：exact match，logits cosine min `0.999582`
   - 这个 case 更像 native paged attention 的 block/tiling 问题。
2. KV 量化格式本身也会触发 drift。
   - `input_len=7168, seed=0, output_len=8`
   - native 和 gather_dequant 都第 `5` 个 decode token drift，token match `0.625`
   - `input_len=9216, seed=0, output_len=8`
   - native 和 gather_dequant 都第 `1` 个 decode token drift，token match `0.125`
   - 这些 case 指向 K-int8/V-FP8 store/scale granularity 或混合格式本身的误差边界。

### 粗扫结果

`output_len=8, seed=0`：

| input_len | native | gather_dequant | 备注 |
| ---: | --- | --- | --- |
| 1024 | exact, cosine min `0.999172` | exact, cosine min `0.999382` | 稳定 |
| 2048 | exact, cosine min `0.999711` | drift@5, match `0.625` | gather debug path 不稳定 |
| 4096 | exact, cosine min `0.997827` | exact, cosine min `0.997226` | 稳定 |
| 6144 | exact, cosine min `0.998071` | drift@5, match `0.625` | gather debug path 不稳定 |
| 8192 | drift@3, match `0.375` | exact, cosine min `0.999582` | native 最小复现 |
| 10240 | exact, cosine min `0.999696` | exact, cosine min `0.999779` | 稳定 |
| 12288 | exact, cosine min `0.997511` | exact, cosine min `0.997118` | 稳定 |
| 16384 | exact, cosine min `0.999134` | exact, cosine min `0.999115` | 稳定 |
| 24576 | OOM | OOM | BF16/reference 路径在 48GB 4090 上 OOM，不是精度结论 |

### 8K 附近细扫

`output_len=8, seeds=0,1,2`：

| input_len | native 结果 | gather_dequant 结果 | 备注 |
| ---: | --- | --- | --- |
| 7168 | seed0 drift@5；seed1/2 exact | seed0 drift@5；seed1 drift@1；seed2 exact | 已有格式/后端敏感性 |
| 7680 | seeds 0/1/2 exact | seeds 0/1/2 exact | 稳定 |
| 7936 | seeds 0/1/2 exact | seeds 0/1/2 exact | 稳定，seed2 cosine min 约 `0.981` |
| 8192 | seed0 drift@3；seed1/2 exact | seeds 0/1/2 exact | native kernel 复现点 |
| 8448 | seeds 0/1/2 exact | seeds 0/1/2 exact | 稳定 |
| 8704 | seeds 0/1/2 exact | seeds 0/1/2 exact | 稳定 |
| 9216 | seed0 drift@1；seed1/2 exact | seed0 drift@1；seed1/2 exact | 格式误差复现点 |

### 64-token decode 压力

| input_len | seed | native | gather_dequant | 备注 |
| ---: | ---: | --- | --- | --- |
| 4096 | 0 | exact, cosine min `0.884649` | exact, cosine min `0.881550` | token 未漂，但 logits margin 已明显扰动 |
| 4096 | 1 | exact, cosine min `0.999260` | exact, cosine min `0.998390` | 稳定 |
| 6144 | 0 | exact, cosine min `0.998071` | drift@5, match `0.4375` | gather debug path 不稳定 |
| 6144 | 1 | exact, cosine min `0.999736` | exact, cosine min `0.999593` | 稳定 |
| 8192 | 0 | drift@3, match `0.1406`, cosine min `0.083460` | exact, cosine min `0.895436` | native 长输出崩溃明显 |
| 8192 | 1 | exact, cosine min `0.999112` | exact, cosine min `0.999292` | 稳定 |

### 额外工程问题

`gpu_memory_utilization=0.90` 时，K-int8/V-FP8 初始化曾在 7B 上触发 `config.num_kvcache_blocks == 0` 断言。`allocate_kv_cache()` 已改为清理 CUDA cache 后按当前 free memory 分配 KV blocks，不再把 warmup/model-load peak 作为硬扣减。远端 smoke 通过：`.remote-logs/kv_cache_budget_20260520/smoke_u090.json`，K-int8/V-FP8 `num_kvcache_blocks=325`，`total_bytes_per_block_ratio_quant_over_bf16=0.51953125`。

### 下一步

- `input_len=8192, seed=0, output_len=8` 的 native drift 不再直接归因到 Triton native kernel。新增 `scripts/kv_cache/native_paged_attention_check.py`，在 8192 context、block table 乱序、`block_tokens=16/32/64` 下分别验证了 native kernel 直接读 cache、以及 Triton store 后再 native decode，两者都和 PyTorch reference 对齐，`max_abs=0.0..6.1035e-05`，`cosine_min=0.99999988`。证据：`.remote-logs/kv_cache_native_debug_20260520/native_kernel_check_seed0*.json` 和 `native_store_kernel_check_seed0_shuffle.json`。
- 分进程 `.pt` trace 重新验证后，8192/seed0 不再复现 token drift：BF16 与 native exact match，logits cosine min `0.9991775`，argmax match `1.0`。证据：`.remote-logs/kv_cache_pt_debug_20260520/bf16_vs_native_8192_seed0.json`。
- `input_len=9216, seed=0, output_len=8` 仍是 K-int8/V-FP8 格式误差最小复现：native 和 gather_dequant 都生成 `[6, 220, 16, 15, 15, 4, 220, 16]`，而 BF16 为 `[6, 82, 198, 220, 16, 15, 15, 15]`；两条 quantized 路径的 first mismatch 都是 `1`。证据：`.remote-logs/kv_cache_pt_debug_20260520/bf16_vs_{native,gather}_9216_seed0.json`。
- 当前优先方向是 K scale granularity。K group-16 已经修复 9216/seed0 的 token drift，group-8/4/1 继续改善 logits；recent K BF16 window 128/256/512 没有修复，说明这个复现点不是只保最近 K 就能解决的近端敏感问题。
- 在把 group-16 或更细粒度变成主线前，需要补 8192 safe case 回归、更多 seeds/lengths，以及 native backend 下同样的 group-size 验证；BF16 KV 仍是 quality path。

## K granularity debug

本轮只围绕 K 做排查，不重新验证 V-only。固定 case：

```text
model=/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct
input_len=9216
seed=0
output_len=8
backend=gather_dequant
scale_dtype=float16
BF16 reference tokens=[6, 82, 198, 220, 16, 15, 15, 15]
evidence=.remote-logs/kv_cache_k_debug_20260520/
```

### K group size

`NANOVLLM_K_INT8_GROUP_SIZE` 只作为本轮 debug knob 引入，默认仍为 `32`。group 越小，K scale 越细，scale bytes/block 越高。

| K group | Tokens | Exact | First mismatch | Cosine min | Argmax match | Top-k overlap | Total bytes/block | Blocks |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | `[6, 220, 16, 15, 15, 4, 220, 16]` | no | 1 | `0.800783` | `0.125` | `0.5625` | `7,626,752` | `1994` |
| 16 | `[6, 82, 198, 220, 16, 15, 15, 15]` | yes | - | `0.998027` | `1.0` | `0.91875` | `7,856,128` | `1936` |
| 8 | `[6, 82, 198, 220, 16, 15, 15, 15]` | yes | - | `0.999321` | `1.0` | `0.94375` | `8,314,880` | `1829` |
| 4 | `[6, 82, 198, 220, 16, 15, 15, 15]` | yes | - | `0.999819` | `1.0` | `0.98125` | `9,232,384` | `1647` |
| 1 | `[6, 82, 198, 220, 16, 15, 15, 15]` | yes | - | `0.999850` | `1.0` | `0.98125` | `14,737,408` | `1032` |

结论：9216/seed0 的漂移主要由 K group-32 scale 粒度不足触发。group-16 是当前最小修复点，bytes/block 只从 `7,626,752` 增到 `7,856,128`，约 `+3.0%`，但 logits 仍比 group-8/4 粗；group-1 接近 BF16 bytes/block，不适合作为内存主线。

### Recent K BF16 window

`NANOVLLM_K_INT8_RECENT_BF16_WINDOW` 是 debug-only 验证：量化 K/V 正常写入，同时保存一份 BF16 K shadow，`gather_dequant` decode 时把最近 N 个 token 的 K 覆盖回 BF16。本轮实现保存完整 K shadow，因此 block 数只用于诊断，不代表最终 residual cache 的真实内存策略。

| Recent K BF16 window | Tokens | Exact | First mismatch | Cosine min | Argmax match | Top-k overlap | Notes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 128 | `[6, 220, 16, 15, 15, 4, 220, 16]` | no | 1 | `0.799640` | `0.125` | `0.5625` | no improvement |
| 256 | `[6, 220, 16, 15, 15, 4, 220, 16]` | no | 1 | `0.799011` | `0.125` | `0.56875` | no improvement |
| 512 | `[6, 220, 16, 15, 15, 4, 220, 16]` | no | 1 | `0.799742` | `0.125` | `0.5625` | no improvement |

结论：这个 prompt 的漂移不是 recent-token K 量化误差单独造成的。与 group-size 扫描合看，下一步应优先评估 `K group=16` 或 `8` 的全套 gates，而不是 recent BF16 window。

## Accepted Next Plan

用户确认后的执行目标是把 KV cache 量化从“单 case 精度排查”推进到“保留最大长文本精度前提下的并发容量证明”。顺序不能反过来：先证明 `K group=16` 在模型最大上下文附近不崩，再跑并发上限。当前优先级如下。

### 1. 先稳定 K group-16

把 `K group=16` 作为主线候选。原因：

- group-32 在 `input_len=9216, seed=0, output_len=8` 早期 drift。
- group-16 是最小修复点，token exact，bytes/block 只比 group-32 增加约 `3.0%`。
- group-8/4/1 作为 debug reference 保留；除非多 seed 或长 decode 证明 group-16 仍不稳，否则不升级默认粒度。

待补 gate：

| Gate | Backend | Lengths | Seeds | Output | 目的 |
| --- | --- | --- | --- | ---: | --- |
| short decode quality | gather_dequant | 8K, 12K, 16K, 32K | 0,1,2 | 8 | 验证格式本身在模型上限附近稳定 |
| short decode quality | native | 8K, 12K, 16K, 32K | 0,1,2 | 8 | 验证 native 8bit decode attention 在模型上限附近稳定 |
| long decode stress | native | 4K, 8K, 16K, 32K | 0,1 | 64 | 观察累计误差和首次 drift |

32K 精度 gate 按 total context budget 执行：如果 `max_position_embeddings=32768`，则 `output_len=8` 使用 `effective_input_len=32760`，`output_len=64` 使用 `effective_input_len=32704`，结果中同时记录 requested/effective length。32K 不再作为“只证明 capacity”的单独口径；它必须作为 group-16 精度硬门槛。

记录字段：

- exact token match
- first mismatch index
- logits cosine min/mean
- argmax match rate
- top-k overlap
- bytes/block
- block ratio vs BF16

### 2. 修 mixed KV 的 chunked prefill 最小路径

之前 32K chunked prefill benchmark 使用的是默认 BF16 KV，证明的是调度策略，不是 mixed KV。mixed KV 的 blocker 在 chunked prefill 的后续 chunk：第 2 个 chunk 开始需要读历史 KV，而历史 KV 已经是 K-int8/V-FP8，不能直接交给 FlashAttention 的 BF16 KV path。

短期策略先保持简单：

```text
第 2 个及之后 prefill chunk:
  只按当前有效上下文长度 gather 历史 KV
  将这部分 K-int8/V-FP8 反量化到临时 BF16 workspace
  与当前 chunk 的 BF16 K/V 一起走现有 FlashAttention prefill
```

这个方案不是最终高性能 kernel，但能先证明 mixed KV 可以接上 24K/32K chunked prefill，并避免 full allocated cache dequant 带来的 OOM/虚假瓶颈。

长期方案再考虑真正的 mixed-KV prefill attention：

```text
Q: 当前 chunk BF16
K/V: 历史 paged K-int8/V-FP8 + 当前 chunk BF16
kernel: 直接读取量化历史 KV，不做大规模 dequant
```

### 3. 核心价值 gate：多长度 x N 极限并发

KV cache 量化的主 claim 应该落在并发容量，而不是只证明单条 32K。并发测试不只看 8K，而是做一条 context length capacity curve。新增脚本：

```text
scripts/kv_cache/kv_cache_concurrency_sweep.py
```

目标 workload：

```text
model=/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct
input_lens=2048,4096,8192,16384,32768
output_len=1  # 先测 capacity；通过后再补 output_len=8 decode smoke
N=1..max_n
mode A: BF16 KV
mode B: K-int8/V-FP8, K group=16
gpu_memory_utilization=0.90
```

32K 的语义按 total context budget 处理：如果 `max_position_embeddings=32768` 且 `output_len > 0`，实际 prompt length 应调整为 `32768 - output_len`，结果里同时记录 `requested_input_len=32768` 和 `effective_input_len`。

进入并发 sweep 的前置条件：`K group=16` 必须先通过 16K/32K 的 short decode quality，且 32K native 至少完成 `output_len=8` exact/token gate。否则并发数字只能作为容量 debug，不能作为“保留最大长文本精度后的并发提升”结论。

脚本行为：

- 每个 `(input_len, mode, N)` 独立进程运行，避免 CUDA 碎片和上一轮失败污染。
- 线性增加 N，直到 OOM、block 不够、scheduler 无法分配或其它运行失败。
- pass/fail 都保存 JSON，不吞掉 OOM；失败原因是结果的一部分。
- 输出总汇总 JSON/CSV，给出每个长度下的最大可通过并发。

每个 case 记录：

- mode
- N
- requested_input_len / effective_input_len / output_len
- pass/fail
- failure reason / traceback 摘要
- num_kvcache_blocks
- required prompt blocks
- block_size
- peak memory
- TTFT / wall time
- generated token 数量

最终要拿到的核心数字：

```text
input_len=2K:  BF16 KV supports N1, K-int8/V-FP8 group-16 supports N2, ratio=N2/N1
input_len=4K:  BF16 KV supports N1, K-int8/V-FP8 group-16 supports N2, ratio=N2/N1
input_len=8K:  BF16 KV supports N1, K-int8/V-FP8 group-16 supports N2, ratio=N2/N1
input_len=16K: BF16 KV supports N1, K-int8/V-FP8 group-16 supports N2, ratio=N2/N1
input_len=32K: BF16 KV supports N1, K-int8/V-FP8 group-16 supports N2, ratio=N2/N1
```

当前 block 粗略容量换算：

```text
BF16 blocks ~= 1036
K group-16 blocks ~= 1936
block_size = 256
theoretical block ratio ~= 1.87x
```

因此期望对外表达为：

> 在同一张卡上，K-int8/V-FP8 group-16 在 2K/4K/8K/16K/32K 多档长文本下系统性提高并发上限；其中 8K 档从 `N1` 提升到 `N2`，并发容量约提升 `N2/N1`，接近 block-level 理论提升 `1.87x`。

### 4. 文档更新规则

完成上述验证后必须更新：

- `opt/kv_cache_quant.md`：完整表、失败原因、证据路径和结论。
- `todo.md`：下一步 gate 状态。
- `README.md` / `README.zh-CN.md`：只有当 2K/4K/8K/16K/32K 多档并发结果稳定且数字可读时，加入摘要行；不要把 debug 表搬进 README。

## README 表述

对外可以这样写：

> KV cache 量化最终采用 K-int8/V-FP8 混合格式：全 FP8 KV 的主要问题是 K 量化会破坏 attention score 并引发 token drift；V-only FP8 证明 V 量化相对安全但显存收益不足。因此主线使用 K group-wise int8 保持 score 稳定，V 使用 FP8 E4M3 降低存储，并通过 native paged decode 验证 8K/16K/32K 长上下文 gate。

## K group-size gate update

本轮目标是先回答 32K 精度 gate，而不是直接跑并发 claim。代码侧做了两个小调整：

- 新增 `scripts/kv_cache/kv_cache_group_gate.py`，每个 backend 单独进程 dump `.pt` logits，再用 `compare_logits_pt.py` 离线比较，避免旧同进程 accuracy suite 在 7B 长上下文下因为 CUDA/Inductor cache 和碎片 OOM。
- mixed KV native decode 的默认 `NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS` 从 `64` 改为 `32`。8K/seed2 在 group-16、block_tokens=64 下 native drift；block_tokens=16/32 都 exact，32 的 logits 略好且迭代次数少于 16。

远端证据：

- 8K seeds 0/1/2，native 默认 block_tokens=32：`.remote-logs/kv_cache_g16_gate_20260520/group_gate_8k_native_default32_seeds012/`
- 8K seed2 block_tokens sweep：`.remote-logs/kv_cache_g16_gate_20260520/native_block{16,32}_8k_seed2/`
- 16K seed0 native/gather：`.remote-logs/kv_cache_g16_gate_20260520/group_gate_16k_seed0_default32/`
- 32K seed0 full-prefill BF16 OOM：`.remote-logs/kv_cache_g16_gate_20260520/group_gate_32k_seed0_default32/`
- 32K seed0 chunked prefill u0.90 mixed OOM：`.remote-logs/kv_cache_g16_gate_20260520/group_gate_32k_seed0_chunk8192_default32/`
- 32K seed0 chunked prefill u0.80 pass：`.remote-logs/kv_cache_g16_gate_20260520/group_gate_32k_seed0_chunk8192_u080_default32/`
- 16K seeds 1/2 group-16：`.remote-logs/kv_cache_g16_gate_20260520/group_gate_16k_seeds12_default32/`
- 16K seed1 group-8：`.remote-logs/kv_cache_g16_gate_20260520/group_gate_16k_seed1_g8_default32/`
- 32K seeds 1/2 group-8 fixed trace：`.remote-logs/kv_cache_g16_gate_20260520/group_gate_32k_seeds12_chunk8192_u080_g8_fixedtrace_default32/`
- 32K seed1 group-4/group-2/group-1 fixed trace：`.remote-logs/kv_cache_g16_gate_20260520/group_gate_32k_seed1_chunk8192_u080_g{4,2,1}_fixedtrace_default32/`

### 8K native tile fix

`input_len=8192, output_len=8, K group=16, backend=native`：

| Seed | block_tokens | Exact | First mismatch | Cosine min | Argmax match | Top-k overlap |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 32 | yes | - | `0.999649` | `1.0` | `0.98125` |
| 1 | 32 | yes | - | `0.999662` | `1.0` | `1.0` |
| 2 | 64 | no | 3 | `0.748076` | `0.625` | `0.8125` |
| 2 | 32 | yes | - | `0.995296` | `1.0` | `0.9875` |
| 2 | 16 | yes | - | `0.994171` | `1.0` | `0.98125` |

结论：8K/seed2 的 drift 是 native paged decode 的 tile-size 数值敏感问题，不是 K group-16 格式本身；`block_tokens=32` 是当前更稳的默认值。gather_dequant 在同一 8K/seed2 case exact，cosine min `0.996457`。

### 16K and 32K short decode

| Gate | Backend | Config | Exact | Cosine min | Argmax match | Notes |
| --- | --- | --- | --- | ---: | ---: | --- |
| 16K seed0 | native | group-16, block_tokens=32, full prefill | yes | `0.999569` | `1.0` | pass |
| 16K seed0 | gather_dequant | group-16, full prefill | yes | `0.998384` | `1.0` | pass |
| 16K seed1 | native/gather | group-16, full prefill | no | `0.841466` | `0.25` | format drift |
| 16K seed1 | native/gather | group-8, full prefill | yes | `0.998506` / `0.998834` | `1.0` | group-8 fixes seed1 |
| 16K seed2 | native/gather | group-16, full prefill | yes | `0.999483` / `0.999676` | `1.0` | pass |
| 32K seed0 | native | group-16, block_tokens=32, chunk 8192, u0.80 | yes | `0.969565` | `1.0` | pass |
| 32K seed0 | gather_dequant | group-16, chunk 8192, u0.80 | yes | `0.969565` | `1.0` | pass |
| 32K seed1 | native/gather | group-8, chunk 8192, u0.80 | no | `0.998886` / `0.999034` | `0.875` | last decode token drift |
| 32K seed1 | native/gather | group-4, chunk 8192, u0.80 | no | `0.999491` / `0.999151` | `0.875` | still drifts |
| 32K seed1 | native/gather | group-2, chunk 8192, u0.80 | no | `0.999861` / `0.999772` | `0.875` | still drifts |
| 32K seed1 | native/gather | group-1, chunk 8192, u0.80 | yes | `0.999485` / `0.999179` | `1.0` | exact but poor capacity |
| 32K seed2 | native/gather | group-8, chunk 8192, u0.80 | yes | `0.999727` / `0.998897` | `1.0` | pass after trace fix |

32K 精度 gate 的实际命令语义是 total context budget：`input_len=32760, output_len=8, max_model_len=32768`。全量 BF16 prefill 会在 gate/up projection 处 OOM（需要额外 `2.31 GiB`，只剩 `2.03 GiB`），因此 32K reference 必须走 chunked prefill。`max_num_batched_tokens=8192, gpu_memory_utilization=0.90` 时 BF16 reference 可跑，但 mixed KV 因预分配更多 KV blocks 后只剩约 `28 MiB` workspace 而 OOM；把精度 gate 的 utilization 降到 `0.80` 后 mixed native/gather 都能完成，`num_kvcache_blocks=1705`，bytes/block 仍为 `7,856,128`。

Trace tooling note: the first 32K seeds 1/2 run over-counted chunked prefill intermediate chunks as generated tokens, producing 11 rows for `output_len=8`. `kv_cache_logits_dump.py` now records logits/tokens only for chunks that actually append a token. Fixed-trace evidence is the source of truth for 32K seeds 1/2.

当前结论：

- K group-16 + native block_tokens=32 不足以通过 16K 多 seed；16K seed1 需要 group-8 才 exact。
- 32K seed1 是当前 blocker。group-8/4/2 都只差最后一个 decode token或一个早期 token，logits cosine 很高但 token exact 不过；group-1 才 exact。
- group-1 的 bytes/block `14,737,408`，blocks `909`，已经接近或低于 BF16 的可用 block 数，不适合作为容量提升主线。
- 因此并发 capacity sweep 暂停，不应把 32K group-16/group-8 写成“保留最大长文本精度后的并发提升”。
- 下一步优先做 32K seed1 的格式 ablation：V-FP8 误差、residual BF16 tail、或 selective K precision，而不是直接跑 2K/4K/8K/16K/32K x N 主结论。

## 32K-first recovery plan

严格 token exact alignment 已降级为 debug/归因指标，不再作为唯一质量 gate。KV cache 量化主质量 gate 改为 teacher-forced NLL/PPL，再用 exact、first mismatch、logits cosine、top-k/rank/margin 解释边界样本。容量和并发 claim 仍必须等质量 gate 通过后再单独跑独立进程 sweep。

原则：必须先解决 32K 质量问题，再考虑容量和并发。任何 32K PPL/NLL 或任务质量未通过的配置都只能作为 debug 数据，不能进入 README claim，也不能作为并发 sweep 的主线。

### Fixed target

固定复现点：

```text
model=/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct
input_len=32760
output_len=8
max_model_len=32768
max_num_batched_tokens=8192
gpu_memory_utilization=0.80
seeds=0,1,2
decode_backends=native,gather_dequant
native_block_tokens=32
quality_reference=BF16 KV
```

当前 blocker 是 `seed=1`。`K group=8/4/2` 都在 fixed trace 下出现 1-token drift，`K group=1` exact 但容量收益不足。因此下一轮只围绕 `32K seed1` 做定位，找到可保留容量收益的修复后，再回归 seeds `0/1/2`。

### Phase A: V-FP8 attribution

先判断 32K seed1 是否主要受 V-FP8 影响。新增 debug-only KV 格式：

```text
K-int8 / V-BF16
```

最小实验矩阵：

| Case | K format | V format | Expected meaning |
| --- | --- | --- | --- |
| A1 | group-8 int8 | BF16 | 如果 exact，V-FP8 是主要 blocker 或至少强相关 |
| A2 | group-4 int8 | BF16 | 如果 A1 不过但 A2 过，K 和 V 都在边界上 |
| A3 | group-2 int8 | BF16 | 继续确认 K 粒度是否仍影响 |
| A4 | group-1 int8 | FP8 | 已知 exact，用作 K 近似上限 reference |

判定：

- `group-8 + V-BF16` exact：优先设计 V residual/BF16-tail，而不是继续收紧全局 K。
- `group-8 + V-BF16` 仍 drift，但 `group-1 + V-FP8` exact：K 精度仍是主因。
- 两边都能改善但单独都不充分：进入 selective K + V tail 联合方案。

实现要求：

- 只作为 debug path，不加入默认 runtime policy。
- 记录 bytes/block 和 blocks，即使 debug 格式不作为主线。
- native/gather 都跑；若两者同漂，优先认为是格式误差；若 native-only 漂，再回到 kernel/tile 排查。

### Phase B: Residual BF16 KV tail

如果 Phase A 指向 V 或远端历史累积误差，尝试 residual BF16 tail。不同于此前只保 recent K BF16，这次保完整 recent KV：

```text
old tokens: K-int8 / V-FP8
recent tail: K-BF16 / V-BF16
```

窗口扫描：

| Tail window | Purpose |
| ---: | --- |
| 256 | 最小增量，验证最后 token 是否近端敏感 |
| 512 | 常见 residual cache 尺度 |
| 1024 | 仍有较小内存增量，优先候选 |
| 2048 | 如果 1024 不够，确认需要更长尾部 |
| 4096 | 上限诊断；若才修复，需要重新评估容量价值 |

判定：

- 若 `group-8 + KV tail <=1024` 修复 32K seed1，则把它作为最优候选进入 multi-seed 回归。
- 若需要 `tail >=4096`，先算容量收益再决定是否值得继续。
- 如果任何 KV tail 都不能修复，再转向 selective K precision。

内存记录必须包含：

- base mixed KV bytes/block
- tail shadow bytes
- effective bytes/block
- blocks at `gpu_memory_utilization=0.80` and `0.90`

### Phase C: Selective K precision

全局 group-1 exact 但容量收益差，因此只把 group-1/BF16 用在局部范围：

```text
old K: group-8 or group-16 int8
recent K tail: group-1 int8 or BF16
V: FP8 first, BF16 tail only if Phase A says V matters
```

优先矩阵：

| Case | Base K | Tail K | V | Tail window |
| --- | --- | --- | --- | ---: |
| C1 | group-8 | group-1 | FP8 | 512 |
| C2 | group-8 | group-1 | FP8 | 1024 |
| C3 | group-8 | BF16 | FP8 | 512 |
| C4 | group-8 | BF16 | FP8 | 1024 |
| C5 | group-8 | BF16 | BF16 tail | 512/1024 |

判定：

- 优先选择 exact 且 effective bytes/block 最低的配置。
- 若 `group-8 + recent K group-1 tail` 可修复，优于全局 group-4/2/1。
- 若必须 `recent K BF16 + V BF16 tail` 才修复，重新评估是否仍满足 KV 容量主线。

### Phase D: Margin analysis, not pass criteria

32K seed1 的 group-8/4/2 漂移只有 1 token，且 logits cosine 很高。需要补充 margin 诊断，但不能放宽 gate：

- BF16 argmax logit
- quant argmax logit
- BF16 top-2 margin
- quant top-2 margin
- BF16 token rank under quant logits
- quant token rank under BF16 logits

判定：

- margin 很小：说明是边界样本，residual tail 或 selective precision 可能足够。
- margin 明显：说明格式误差仍大，需要更强格式变化。

注意：margin 只用于定位。32K 质量 gate 的主标准已经改为 teacher-forced NLL/PPL；exact token match 只能作为强一致信号，不能继续作为唯一 pass/fail 标准。

### Promotion gates

一个候选配置只有同时满足以下条件，才能进入并发 sweep：

| Gate | Requirement |
| --- | --- |
| 8K/16K/32K teacher-forced PPL | BF16 KV vs mixed KV relative PPL regression 在小样本上接近 0，扩大样本后目标不超过 `3%-5%` |
| Logits/debug alignment | exact、first mismatch、cosine、top-k/rank/margin 保留为解释数据，不单独否决高质量 PPL case |
| 32K trace correctness | fixed trace only，不允许 chunked prefill 中间 chunk 计入 generated tokens |
| Capacity sanity | effective bytes/block 明显低于 BF16；blocks 明显高于 BF16 |
| Runtime policy | 默认路径保持简单；debug-only knobs 不直接升主线 |

只有这些 gate 通过后，才恢复 `kv_cache_concurrency_sweep.py`：

```text
input_lens=2048,4096,8192,16384,32768
output_len=1 first
then output_len=8 smoke
N=1..max_n
modes=BF16 KV vs chosen mixed KV candidate
```

## Teacher-forced KV PPL/NLL

本轮新增 `scripts/kv_cache/kv_cache_ppl_eval.py`，把 KV cache 量化质量从“生成 token exact 对齐”改成 teacher-forced 语言建模指标：

```text
prefill context tokens
for each eval token:
  use current logits to score the ground-truth next token
  force that ground-truth token through scheduler.postprocess()
  continue decode so KV cache grows through the normal runtime path
```

这样每一步都会真实读写 decode-time KV cache，同时避免 greedy 第一个 token drift 后把后续全部放大成不同文本。

测试配置：

```text
model=/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct
dataset=/root/autodl-tmp/datasets/_raw/wikitext-2-raw-v1
backend=BF16 KV vs K-int8/V-FP8 native
K group=8
native_block_tokens=32
eval_tokens=64
windows=1 per length, except 4K/8K combined summary uses 2 windows total
```

远端证据：

- Smoke：`.remote-logs/kv_cache_ppl_20260520/smoke/summary.json`
- 4K/8K：`.remote-logs/kv_cache_ppl_20260520/7b_g8_4k8k_u080/summary.{json,csv}`
- 16K：`.remote-logs/kv_cache_ppl_20260520/7b_g8_16k_u060_chunk4096/summary.{json,csv}`
- 32K total-context：`.remote-logs/kv_cache_ppl_20260520/7b_g8_32k_u050_chunk4096/summary.{json,csv}`

### PPL Summary

| Context | BF16 PPL | K-int8/V-FP8 group-8 PPL | Relative PPL regression | Notes |
| ---: | ---: | ---: | ---: | --- |
| 4K+8K aggregate | `5.38628` | `5.38691` | `+0.0118%` | 2 windows, 128 eval tokens total |
| 16K | `3.75960` | `3.73692` | `-0.6033%` | u0.60, chunk 4096 to avoid prefill workspace OOM |
| 32K | `4.84199` | `4.83085` | `-0.2301%` | context `32704`, output/eval `64`, u0.50, chunk 4096 |

结论：

- 在这组 WikiText teacher-forced 小样本上，`K group=8` 的 mixed KV 没有表现出 PPL 崩溃；4K/8K/16K/32K 的 relative PPL regression 都接近 0。
- 这说明此前 32K seed1 的 1-token exact drift 更像生成式边界样本/argmax 敏感性，不足以单独否定 mixed KV 的语言建模质量。
- 本轮 PPL 仍是小样本，不足以写成最终 README 质量 claim；下一步应扩大 windows、补 `group=16` 对照、补 `gather_dequant` reference，并加入 PG19/LongBench 或 RULER 类长文本任务。
- PPL 脚本输出的 `num_kvcache_blocks` 只用于记录当次进程状态，不作为容量结论。4K/8K 和 16K 的顺序加载会受 CUDA 释放/碎片影响，容量 claim 仍必须用 `kv_cache_concurrency_sweep.py` 的独立进程结果。

### K group-size PPL smoke

用户提出可放宽 exact alignment 后，补测 `K group=16/32` 的 4K/8K teacher-forced PPL。目标是确认能否把 K scale 粒度放粗一点，以换取更低 scale overhead。

证据：

- `group=16`：`.remote-logs/kv_cache_ppl_20260520/7b_g16_4k8k_u080/summary.{json,csv}`
- `group=32`：`.remote-logs/kv_cache_ppl_20260520/7b_g32_4k8k_u080/summary.{json,csv}`
- `group=32 16K`：`.remote-logs/kv_cache_ppl_20260520/7b_g32_16k_u060_chunk4096/summary.{json,csv}`
- `group=32 32K`：`.remote-logs/kv_cache_ppl_20260520/7b_g32_32k_u050_chunk4096/summary.{json,csv}`

| K group | BF16 PPL | Mixed KV PPL | Relative PPL regression | Bytes/block ratio vs BF16 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | `5.38628` | `5.38691` | `+0.0118%` | `0.5664x` |
| 16 | `5.38628` | `5.45723` | `+1.3172%` | `0.5352x` |
| 32 | `5.38628` | `5.45073` | `+1.1965%` | `0.5195x` |

`group=32` 长上下文补测：

| Context | BF16 PPL | Mixed KV group-32 PPL | Relative PPL regression | Notes |
| ---: | ---: | ---: | ---: | --- |
| 16K | `3.75960` | `3.81790` | `+1.5507%` | 64 eval tokens, u0.60, chunk 4096 |
| 32K | `4.84199` | `4.83413` | `-0.1625%` | context `32704`, 64 eval tokens, u0.50, chunk 4096 |

结论：

- 在 4K/8K PPL 小样本上，`group=16` 和 `group=32` 都没有 PPL 崩溃；退化约 `1.2%-1.3%`，低于临时可接受线 `3%-5%`。
- `group=32` 的 16K/32K 也没有 PPL 崩溃；16K 回退约 `+1.55%`，32K 近似持平。它的 exact alignment 曾在 9216/seed0 早期 drift，但当前 PPL/NLL 结果显示语言建模损失不大。
- 当前可把 `group=32` 作为 mixed KV 容量优先 fallback。下一步可以先尝试 full KV FP8 的 PPL smoke；如果 full FP8 不崩且回退可接受，再进入 FP8 sweep；如果失败，则回退到 mixed `group=32` 做并发容量 sweep。

## Same-capacity KV memory check

本轮先尝试 full FP8 KV PPL smoke，再改用更直接的同容量显存口径验证 mixed KV。full FP8 KV 仍只作为 ablation，不进入主线。

### Full FP8 KV PPL smoke

配置：

```text
model=/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct
dataset=/root/autodl-tmp/datasets/_raw/wikitext-2-raw-v1
backends=BF16 KV vs full FP8 KV gather
contexts=4K,8K
eval_tokens=32 per context
```

证据：`.remote-logs/kv_cache_fp8_ppl_20260520/full_fp8_4k8k_smoke/summary.json`

| Backend | PPL | Mean NLL | Argmax match | KV bytes/block | Blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 KV | `8.47187` | `2.13675` | `0.5781` | `14,680,064` | `912` |
| Full FP8 KV gather | `1509.77137` | `7.31971` | `0.0938` | `7,454,720` | `156` |

结论：full FP8 KV 的 storage ratio 是 `0.5078x`，但 PPL 相对 BF16 回退约 `177x`，质量明显失败。继续使用 K-int8/V-FP8 mixed KV。

### First concurrency sweep diagnosis

第一轮并发 sweep 试图直接比较 BF16 与 mixed 的最大并发，但失败点主要发生在 prefill MLP activation：

```text
8K: BF16 N=2, mixed N=2
16K: BF16 N=1, mixed N=1
32K: both fail at N=1
failure site: qwen2.py gate_up_proj / act_fn OOM
evidence=.remote-logs/kv_cache_concurrency_20260520/group32_len8k16k32_n1_8/
```

这不是 KV block 容量结论。BF16 和 mixed 没有同时驻留、不会互相抢显存；脚本每个 `(length, mode, N)` 都新起独立 Python 进程。问题是该 workload 的 prefill activation 先打满显存，掩盖了 KV storage 差异。

### Same fixed KV blocks memory result

为直接回答“相同输入时显存占用是否更小”，新增 `scripts/kv_cache/kv_cache_memory_case.py`：一个进程只跑一种 KV mode，固定 workload，并记录 PyTorch memory、`nvidia-smi`、KV arena 和生成统计。

同时调整 `ModelRunner.allocate_kv_cache()`：当 `Config.num_kvcache_blocks > 0` 时尊重显式 block 数；否则仍按 `gpu_memory_utilization` 自动预算。这样可以固定 BF16 与 mixed 拥有相同 KV 容量，直接比较显存。

配置：

```text
model=/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct
KV blocks=512
K group=32
decode backend=native
max_num_batched_tokens=8192
chunked_prefill_policy=prefill_first
```

证据：

- `.remote-logs/kv_cache_memory_20260520/same_blocks512_8k_out64/{bf16,k_int8_v_fp8_g32}.json`
- `.remote-logs/kv_cache_memory_20260520/same_blocks512_16k_out1/{bf16,k_int8_v_fp8_g32}.json`

| Workload | Mode | KV arena | KV bytes/block | PyTorch allocated after generate | PyTorch peak allocated | `nvidia-smi` after generate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8K input / 64 output / 1 seq | BF16 KV | `7.0000 GiB` | `14,680,064` | `21.2644 GiB` | `22.4052 GiB` | `23,672 MiB` |
| 8K input / 64 output / 1 seq | K-int8/V-FP8 group-32 | `3.6367 GiB` | `7,626,752` | `17.9011 GiB` | `19.0419 GiB` | `20,228 MiB` |
| 16K input / 1 output / 1 seq | BF16 KV | `7.0000 GiB` | `14,680,064` | `21.2644 GiB` | `22.4052 GiB` | `23,672 MiB` |
| 16K input / 1 output / 1 seq | K-int8/V-FP8 group-32 | `3.6367 GiB` | `7,626,752` | `18.7761 GiB` | `19.9169 GiB` | `21,116 MiB` |

结论：

- 在相同 `512` KV blocks 容量下，K-int8/V-FP8 group-32 的 KV arena 是 BF16 的 `0.5195x`，从 `7.00 GiB` 降到 `3.64 GiB`。
- 8K/64 workload 的端到端 PyTorch peak allocated 少 `3.36 GiB`，`nvidia-smi` 进程显存少 `3444 MiB`。
- 16K/1 workload 的端到端 PyTorch peak allocated 少 `2.49 GiB`，`nvidia-smi` 进程显存少 `2556 MiB`；长 prefill 的临时 workspace/activation 会吃掉一部分端到端收益，但 KV arena 节省保持稳定。
- 如果不固定 block 数，当前 runtime 会按 `gpu_memory_utilization` 尽量把可用显存转换成更多 KV blocks；此时 mixed KV 表现为 block 数更多，而不是总显存更少。
