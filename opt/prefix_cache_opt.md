# Prefix Cache Optimization

## 目标

优化 KV cache prefix 复用，减少重复 prefill 计算。

## Backend 对比

| Backend | 环境变量 | 特点 |
|---------|---------|------|
| hash | `NANOVLLM_PREFIX_CACHE_BACKEND=hash` (默认) | 简单高效，使用 xxhash |
| radix | `NANOVLLM_PREFIX_CACHE_BACKEND=radix` | block-level radix metadata，命中率与 hash 对齐，便于扩展 branching prefix |

## L0/L1: Hash Prefix Cache

- `BlockManager` 使用 `hash_to_block_id` 字典存储 block hash -> block_id 映射
- 使用 xxhash 计算 block hash，支持 prefix chaining

## L1: HiRadix (Radix Tree)

- `RadixTreePrefixCache` 类实现 radix tree 数据结构
- 当前只保留已经接入 `BlockManager` 的 block-level prefix match/insert/remove
- 用于后续 branching-prefix workload 的 metadata 验证
- 文件: `nanovllm/engine/radix_tree.py`

## Benchmark 结果

远端证据：`.remote-logs/prefix_chunked_20260518/`。

模型: Qwen2.5-3B-Instruct, 8 请求, prefix=1024, unique=128, output=8, block size=256。

| Backend | Scenario | Token Hit Rate | Total Time | Prefill TPS | Decode TPS |
|---------|----------|---------------:|-----------:|------------:|-----------:|
| hash | no-reuse | `0.0%` | `2.367 s` | `4288.8` | `259.0` |
| hash | shared-prefix | `87.5%` | `1.540 s` | `1551.4` | `257.1` |
| hash | partial-shared | `43.8%` | `1.637 s` | `3981.1` | `253.6` |
| radix | no-reuse | `0.0%` | `1.750 s` | `6013.4` | `259.1` |
| radix | shared-prefix | `87.5%` | `1.484 s` | `1621.4` | `256.1` |
| radix | partial-shared | `43.8%` | `1.614 s` | `4030.2` | `261.1` |

结论：shared-prefix 命中率按预期达到 `7/8 = 87.5%`，partial-shared 达到 `43.8%`。hash 与 radix 的命中率一致；radix 在这组运行中 wall time 略低，但不把单次 wall time 当作主要 claim。当前 claim 收敛为：hash 是默认高效实现，radix 是已接入且结果对齐的 block-level metadata backend。

## 当前保留范围

- 保留 `hash` prefix cache：默认路径，已有 shared/partial/no-reuse benchmark。
- 保留 `radix` prefix cache：已有同一组 benchmark，命中率与 hash 对齐。
- 删除未接入运行时、没有效果证据的 HiCache L1/L2 骨架代码。
- 删除 radix 中未被 `BlockManager` 调用的 LRU/ref-count API，避免把未验证能力暴露成已完成实现。

## 后续优化路线

1. branching-prefix benchmark：
   - 扩展 `scripts/prefix_cache/prefix_cache_bench.py`，构造共享前缀后分 2 到 4 个分支的请求。
   - 新增 radix metadata 指标：node count、leaf count、max depth、matched blocks。
   - 只有当 radix 在可解释性或性能上给出稳定收益，再扩大 README claim。

2. eviction-pressure：
   - 先设计 BlockManager 级容量压力测试，再实现 eviction。
   - 要求记录 hit rate、recompute tokens、OOM/容量边界。
   - 在有测试前，不恢复 LRU runtime API。

3. tiered/HiCache：
   - 先写清楚 GPU block 到 CPU pinned block 的所有权和异步 copy 时序。
   - 必须与 `BlockManager`、attention block table、KV storage layout 一起实现。
   - 只有跑通端到端 load-back 正确性和容量收益后，再把代码放回 runtime。
