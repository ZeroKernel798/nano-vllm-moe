# nano-vllm-moe Refactor Todo

> Generated: 2026-04-21  
> Based on full codebase audit of current state.

---

## Stage 0 — Bug Fixes & Dead Code（零风险，~2h）

不改任何行为，全部是"本该早就删掉/修掉"的东西。

- [ ] **删除 `loader.py` 前 80 行注释代码** — 文件头部 lines 1–80 是完整的旧实现注释，直接删除
- [ ] **确认 `model_runner.py` 死 import** — 检查 `from nanovllm.models.qwen3 import Qwen3ForCausalLM`（如存在）是否被 `model_dict` 路由绕过成死代码，是则删除
- [ ] **迁移 `nanovllm/quant/test_awqint4.py`** — 测试脚本混在生产模块里，移到 `tests/` 或 `scripts/` 目录
- [ ] **补全 MoE expert key 的 unmatched 警告** — `loader.py` lines 129–135，`get_param_or_buffer` 返回 `None` 时静默跳过，没有追加到 `unmatched_keys`，补上
- [ ] **评估 CUDA graph 对 EP all-to-all 的安全性** — `model_runner.py` 的 `capture_cudagraph()` 路径对 `Qwen3MoeSparseMoeBlock` 中的 all-to-all 无任何 guard，至少加一个断言或 `enforce_eager` 检查

---

## Stage 1 — 结构清理（不改行为）

- [ ] **统一 TP group API（Stage 2 的先决条件）** — `qwen3.py` 用 `dist.get_world_size()`（全局 group），`qwen3_moe.py` 用 `dist.get_world_size(tp_group)`（显式 group）；在任何 block 合并之前，统一为显式 `tp_group` 参数传递
- [ ] **建目录骨架** — 新建 `nanovllm/executor/`、`nanovllm/executor/moe/`、`nanovllm/executor/moe/backends/`，各加空 `__init__.py`
- [ ] **审计现有量化层结构** — 在设计新接口前，先完整盘点 `layers/quant_linear.py`、`layers/smooth_quant_linear.py`、`layers/fp8/`、`layers/fp8_linear.py` 四处实现的接口差异，形成一份对比表

---

## Stage 2 — MoE Backend 抽象

> 先决条件：Stage 1 TP group 统一必须完成。

- [ ] **提取 Triton kernel 调用** — 将 `qwen3_moe.py` 中的 `fused_moe_w13_kernel` / `fused_moe_w2_combine_kernel` 调用下沉到 `nanovllm/executor/moe/backends/triton.py`
- [ ] **定义 `MoEBackend` 抽象接口** — 抽象基类包含 `dispatch()`、`compute()`、`combine()` 三个方法；triton backend 实现它
- [ ] **将 `Qwen3MoeSparseMoeBlock`** 改为通过 `MoEBackend` 调用，而不是直接调 kernel
- [ ] **合并 dense attention block** — TP group 统一后，将 `qwen3.py` 的 `Qwen3Attention` 和 `qwen3_moe.py` 的 `Qwen3MoeAttention` 合并为 `layers/attention_block.py`；注意两者输出命名不一致，合并时对齐

---

## Stage 3 — Loader 规范化（与 Stage 2 并行推进）

> 这条线与 Stage 2 几乎无依赖，可以并行。

- [ ] **先写 `tests/loader/` 测试** — 在改任何 loader 代码之前，用已知 checkpoint 的 diff 建立正确性基准
- [ ] **泛化 MoE expert key 解析** — `loader.py` lines 115–118 的 `parts[2]`、`parts[5]`、`parts[6]` 硬编码索引易碎；替换为 regex 或具名捕获解析
- [ ] **补全 FP8 后缀支持** — 验证 `weight_scale` / `input_scale` 后缀能覆盖 `layers/fp8/` 和 `layers/fp8_linear.py` 所有变体（两处命名约定不同）
- [ ] **加 loader smoke test 进 CI** — 加载一个小模型 shard，断言 unmatched keys 为零

---

## Stage 4 — 量化后端抽象

- [ ] **定义 `LinearKernel` 接口** — `forward(x, weight_data) -> Tensor`；具体实现：`DenseKernel`、`AWQKernel`、`FP8Kernel`、`W8A8Kernel`
- [ ] **迁移 AWQ 三个类** — `AWQRowParallelLinear`、`AWQMergedColumnParallelLinear`、`AWQQKVParallelLinear` 改用 `AWQKernel`
- [ ] **迁移 `fp8_linear.py` 和 `smooth_quant_linear.py`** 各自的 kernel
- [ ] **写量化数值测试** — 每个 kernel 对比 `F.linear` 参考输出，确认误差在容忍范围内
- [ ] **`QuantizedTensor` 暂缓** — 这个抽象在本阶段过于 ambitious，推迟到 Stage 6

---

## Stage 5 — DeepEP 接入

- [ ] **评估 `moe_align_block_size`** — `utils/moe.py` 是纯 Python 实现，是 token dispatch 关键路径；接入 DeepEP 前决策：(a) 替换为 CUDA kernel，(b) 用 DeepEP 自带 dispatch 原语，(c) 接受 Python 版本的开销
- [ ] **实现 `DeepEPBackend`** — 实现 Stage 2 定义的 `MoEBackend` 接口
- [ ] **加 EP routing 的 CUDA graph guard** — 确保 `ep_size > 1` 且 DeepEP 激活时 CUDA graph capture 被 bypass
- [ ] **Benchmark** — DeepEP vs Triton backend 在目标 batch size 下的对比

---

## Stage 6 — DeepSeek / vLLM 特性对齐

- [ ] Shared Expert 支持
- [ ] MLA Attention
- [ ] Chunked Prefill
- [ ] 更接近 vLLM runtime 的接口设计
- [ ] `QuantizedTensor` 抽象（从 Stage 4 延期至此）

---

## 测试写作优先级

**在对应代码改动开始之前**就写测试，不等到阶段末尾：

| 优先级 | 测试目录 | 原因 |
|--------|----------|------|
| 1 | `tests/loader/` | 改动范围最大，最容易用 diff 验证，没有它重构就要靠全量 bench 回归 |
| 2 | `tests/quant/` | 量化正确性与 loader 耦合，loader 改了就得同步验证 |
| 3 | `tests/moe/` | dispatch / combine round-trip |
| 4 | `tests/models/` | 端到端输出回归（成本最高，放最后） |

---

## 执行时间线

### 第 1 周
| 事项 | 工时 |
|------|------|
| Stage 0 全部 | ~2h |
| Stage 1 TP group 统一 | ~1 天 |
| Stage 2 Triton 提取 + `MoEBackend` 接口草稿（不迁实现） | ~2 天 |
| Stage 3 loader 测试 + MoE key 解析修复 | ~2 天 |

### 第 2 周
| 事项 | 工时 |
|------|------|
| Stage 2 attention block 合并 | ~1 天 |
| Stage 3 loader 规范化完成 + smoke test | ~1 天 |
| Stage 4 `LinearKernel` 接口 + AWQ 迁移 | ~2 天 |

### 第 3 周
| 事项 | 工时 |
|------|------|
| Stage 4 FP8/W8A8 kernel 迁移 + 数值测试 | ~2 天 |
| Stage 5 评估 + DeepEP backend stub | ~2 天 |

---

## 风险登记表

| 风险 | 位置 | 缓解措施 |
|------|------|----------|
| CUDA graph 与 EP all-to-all 不兼容 | `model_runner.py capture_cudagraph()` | Stage 0 加 `enforce_eager` guard，Stage 5 之前必须解决 |
| TP group API 不一致导致 block 合并出错 | `qwen3.py` vs `qwen3_moe.py` | Stage 1 统一后才允许进入 Stage 2 合并 |
| FP8 后缀命名不一致导致 loader 静默失败 | `layers/fp8/` vs `layers/fp8_linear.py` | Stage 3 loader 审计中统一 |
| 纯 Python `moe_align_block_size` 成为 perf 噪声 | `utils/moe.py` | Stage 5 中评估替换方案 |
| 量化路径无数值回归测试 | `layers/quant_linear.py` 等 | Stage 4 开始前必须先建测试 |
