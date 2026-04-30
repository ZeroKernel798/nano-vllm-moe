# nano-vllm-moe 向 vLLM MoE / EP 对齐建议

本文记录 `nano-vllm-moe` 当前 MoE / EP 实现与 vLLM 的接口分层差异，并给出一套适合 nano 项目的轻量化对齐方案。目标不是完整复制 vLLM 的工程复杂度，而是学习其接口边界，把 EP 通信、MoE 本地专家计算、Router、Shared Expert 等逻辑解耦，方便后续独立优化和横向对比。

## 1. 总体结论

`nano-vllm-moe` 当前已经有一个不错的雏形：

- `nanovllm/executor/moe/backends/base.py` 定义了 `MoEBackend.dispatch / compute / combine`。
- `nanovllm/executor/moe/backends/triton.py` 实现了当前 Triton Group-GEMM + torch all-to-all EP 路径。
- `nanovllm/models/qwen2_moe.py` 和 `nanovllm/models/qwen3_moe.py` 已经把 MoE 权重堆叠成 `w13_stacked / w2_stacked`，具备向统一 MoE block 抽象演进的基础。

但当前主要问题是：

- EP 通信、专家计算、TP reduce 都耦合在 `TritonMoEBackend` 中。
- Qwen2 / Qwen3 的 Router 差异散落在模型文件里。
- 模型层直接创建 backend，并直接传 `tp_group / ep_group / local_num_experts / local_inter_size`。
- 未来接入 DeepEP、FP8 MoE、INT8 MoE 时，很容易产生组合爆炸。

建议对齐 vLLM 的核心思想：

```text
Router -> Prepare/Finalize(通信/量化/dispatch/combine) -> Experts(本地专家计算) -> MoEKernel/Runner
```

对 nano 来说，建议轻量实现为：

```text
MoERouter
MoEPrepareFinalize
MoEExpertsKernel
MoEKernel
BaseSparseMoeBlock
```

这样后续可以独立比较：

```text
NoEP + TritonExperts
TorchAllToAll + TritonExperts
DeepEP + TritonExperts
DeepEP + FP8Experts
NoEP + FP8Experts
```

## 2. vLLM 中值得参考的关键分层

### 2.1 FusedMoEPrepareAndFinalizeModular

位置：

- `vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:251`

职责：

- 对输入 activation 做量化。
- 根据 `topk_ids / topk_weights` 做 dispatch。
- 返回本 rank 需要处理的 token、scale、expert metadata、可能被通信重排后的 `topk_ids / topk_weights`。
- 在专家计算完成后做 finalize / combine。

它是 vLLM 中 EP 通信和专家计算解耦的核心。DeepEP、naive DP/EP、no DP/EP 等都可以作为不同 Prepare/Finalize 实现挂进去。

### 2.2 FusedMoEExpertsModular

位置：

- `vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:763`

职责：

- 只负责本地专家计算。
- 输入已经 prepare 好的 activation、`topk_ids / topk_weights`、专家权重。
- 内部完成 permute、expert GEMM、activation、第二次 GEMM、unpermute 或等价 fused kernel。
- 不直接关心 DeepEP / all-to-all / DP / SP 通信细节。

### 2.3 FusedMoEKernel

位置：

- `vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:1482`

职责：

- 组合 `prepare_finalize` 和 `fused_experts`。
- 对外提供统一 `apply()`。
- 内部执行：

```text
prepare -> fused_experts -> finalize
```

这正是 nano 目前 `TritonMoEBackend.dispatch -> compute -> combine` 可以演进成的结构。

### 2.4 BaseRouter / FusedTopKRouter

位置：

- `vllm/vllm/model_executor/layers/fused_moe/router/base_router.py:270`
- `vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:116`

职责：

- 将 routing 从模型层中抽出来。
- 统一处理：
  - softmax / sigmoid scoring
  - top-k
  - renormalize
  - EPLB logical-to-physical expert id mapping
  - topk ids dtype 转换

nano 当前 Qwen2 / Qwen3 routing 的主要差异就是是否 renormalize top-k 权重：

- Qwen2 MoE：通常 `renormalize=False`。
- Qwen3 MoE：需要 `renormalize=True`。

这部分适合抽成 `SoftmaxTopKRouter`。

### 2.5 MoERunner

位置：

- `vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:531`

职责：

- 调度 MoE forward。
- 处理 shared expert overlap。
- 处理 router 和 quant method 调用。
- 处理 sequence parallel、naive dispatch/combine、final reduce 等外围逻辑。

nano 暂时不需要完整引入 Runner 复杂度，但可以保留 `BaseSparseMoeBlock` 作为轻量 Runner：模型层只负责组装权重和调用统一 MoE kernel。

### 2.6 EP Group dispatch/combine

位置：

- `vllm/vllm/distributed/parallel_state.py:1093`

vLLM 在 `GroupCoordinator` 上暴露：

```python
dispatch(hidden_states, topk_weights, topk_ids, ...)
combine(hidden_states, ...)
```

Prepare/Finalize 不直接到处写 `dist.all_to_all_single`，而是通过通信抽象层调用。nano 可以先不用完整实现 GroupCoordinator，但至少应该把 `torch.distributed.all_to_all_single` 包到单独 `PrepareFinalize` 或 `EPCommunicator` 中。

## 3. nano 当前实现的主要耦合点

### 3.1 TritonMoEBackend 责任过重

位置：

- `nanovllm/executor/moe/backends/triton.py:14`

当前 `TritonMoEBackend` 同时负责：

- 根据 `topk_ids` 计算目标 EP rank。
- 做 `send_counts / recv_counts`。
- 调用多次 `dist.all_to_all_single`。
- 调用 `moe_align_block_size`。
- 调用 `fused_moe_w13_kernel` 和 `fused_moe_w2_combine_kernel`。
- combine 时再做 all-to-all 回传和 TP all-reduce。

这会导致后续 DeepEP 接入时要么改这个类，要么新建另一个大 backend，最终形成：

```text
TritonTorchA2A
TritonDeepEP
FP8TorchA2A
FP8DeepEP
INT8TorchA2A
INT8DeepEP
```

组合爆炸。

### 3.2 模型层知道太多 EP / TP 细节

位置：

- `nanovllm/models/qwen3_moe.py:109`
- `nanovllm/models/qwen2_moe.py:169`

当前模型层持有：

- `tp_group`
- `ep_group`
- `tp_size / tp_rank`
- `ep_size / ep_rank`
- `local_num_experts`
- `local_inter_size`
- `backend = TritonMoEBackend(...)`

这些信息可以收口到一个 `MoEParallelConfig` 或 `MoEContext` 中，减少 Qwen2 / Qwen3 重复代码。

### 3.3 Router 分散在模型文件中

位置：

- `nanovllm/models/qwen3_moe.py:167`
- `nanovllm/models/qwen2_moe.py:260`

Qwen3 需要：

```python
topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
```

Qwen2 当前没有统一抽象。建议抽成：

```python
SoftmaxTopKRouter(top_k, renormalize=True/False)
```

模型配置只决定 `renormalize`，forward 不再手写 routing 细节。

### 3.4 CUDA Graph 与动态 EP 路径冲突

位置：

- `nanovllm/engine/model_runner.py:335`

当前 `capture_cudagraph()` 没有区分 EP / dynamic all-to-all 后端。对于 torch all-to-all 或 DeepEP high-throughput 这类动态 shape / 非 graph-compatible 路径，应该跳过 capture。

建议短期加 guard：

```python
if self.ep_size > 1:
    self.enforce_eager = True
```

更好的长期做法是让 backend 暴露：

```python
backend.supports_cuda_graph: bool
backend.uses_dynamic_alltoall: bool
```

## 4. 建议的 nano 轻量接口设计

### 4.1 MoEParallelConfig

建议新增：

```python
@dataclass
class MoEParallelConfig:
    tp_group: dist.ProcessGroup | None
    ep_group: dist.ProcessGroup | None
    tp_size: int
    tp_rank: int
    ep_size: int
    ep_rank: int
    global_num_experts: int
    local_num_experts: int
    intermediate_size: int
    local_inter_size: int
```

由 `BaseSparseMoeBlock` 或工厂函数创建，避免每个模型重复计算。

### 4.2 MoERouter

建议新增：

```python
class MoERouter(ABC):
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
```

基础实现：

```python
class SoftmaxTopKRouter(MoERouter):
    def __init__(self, top_k: int, renormalize: bool):
        self.top_k = top_k
        self.renormalize = renormalize

    def select_experts(self, hidden_states, router_logits):
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights.to(torch.float32), topk_ids
```

Qwen2 / Qwen3 差异只体现在构造参数：

```python
Qwen2: SoftmaxTopKRouter(top_k, renormalize=False)
Qwen3: SoftmaxTopKRouter(top_k, renormalize=True)
```

### 4.3 PrepareResult

建议新增：

```python
@dataclass
class PrepareResult:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    a1_scale: torch.Tensor | None = None
    expert_tokens_meta: object | None = None
    ctx: object | None = None
```

`ctx` 用于保存 combine 需要的上下文，例如：

- `permute_indices`
- `send_counts`
- `recv_counts`
- `m_tokens`
- `hidden_size`
- `top_k`

### 4.4 MoEPrepareFinalize

建议新增：

```python
class MoEPrepareFinalize(ABC):
    supports_cuda_graph: bool = False
    uses_dynamic_alltoall: bool = True

    def prepare(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        apply_router_weight_on_input: bool = False,
    ) -> PrepareResult:
        raise NotImplementedError

    def finalize(
        self,
        expert_out: torch.Tensor,
        prepare: PrepareResult,
        *,
        output_shape: tuple[int, int],
        reduce_tp: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError
```

基础实现：

- `NoEPPrepareFinalize`
  - `ep_size == 1`。
  - 只做本地 `repeat_interleave / flatten`。
  - 不调用 all-to-all。
  - `supports_cuda_graph=True`。

- `TorchAllToAllPrepareFinalize`
  - 对应当前 `TritonMoEBackend.dispatch/combine` 里的 all-to-all 逻辑。
  - 初期可以保留 `tolist()` / `.item()`，后续再优化。
  - `supports_cuda_graph=False`。

- `DeepEPPrepareFinalize`
  - 后续接 DeepEP high-throughput / low-latency。
  - 不影响专家 kernel。

### 4.5 MoEExpertsKernel

建议新增：

```python
class MoEExpertsKernel(ABC):
    def apply(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        *,
        local_num_experts: int,
        local_inter_size: int,
        hidden_size: int,
        model_dtype: torch.dtype,
        w13_weight_scale: torch.Tensor | None = None,
        w2_weight_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
```

基础实现：

- `TritonGroupedGemmExperts`
  - 对应当前 `TritonMoEBackend.compute`。
  - 调用 `moe_align_block_size`、`fused_moe_w13_kernel`、`fused_moe_w2_combine_kernel`。

后续扩展：

- `FP8GroupedGemmExperts`
- `INT8GroupedGemmExperts`
- `DeepGemmExperts`
- `CutlassExperts`

### 4.6 MoEKernel

建议新增：

```python
class MoEKernel:
    def __init__(
        self,
        prepare_finalize: MoEPrepareFinalize,
        experts: MoEExpertsKernel,
        parallel_config: MoEParallelConfig,
    ):
        self.prepare_finalize = prepare_finalize
        self.experts = experts
        self.parallel_config = parallel_config

    def __call__(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        *,
        model_dtype: torch.dtype,
        reduce_tp: bool = True,
        w13_weight_scale: torch.Tensor | None = None,
        w2_weight_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prepared = self.prepare_finalize.prepare(x, topk_weights, topk_ids)
        local_out = self.experts.apply(
            prepared.hidden_states,
            prepared.topk_ids,
            prepared.topk_weights,
            w13,
            w2,
            local_num_experts=self.parallel_config.local_num_experts,
            local_inter_size=self.parallel_config.local_inter_size,
            hidden_size=x.shape[-1],
            model_dtype=model_dtype,
            w13_weight_scale=w13_weight_scale,
            w2_weight_scale=w2_weight_scale,
        )
        return self.prepare_finalize.finalize(
            local_out,
            prepared,
            output_shape=x.shape,
            reduce_tp=reduce_tp,
        )
```

这就是 nano 版本的轻量 `FusedMoEKernel`。

## 5. BaseSparseMoeBlock 建议

建议抽出：

```python
class BaseSparseMoeBlock(nn.Module):
    def __init__(self, config, tp_group=None, ep_group=None, *, router, has_shared_expert=False):
        ...
```

统一管理：

- `hidden_size`
- `global_num_experts`
- `top_k`
- `moe_intermediate_size`
- `MoEParallelConfig`
- `w13_stacked`
- `w2_stacked`
- `load_hybrid_moe_weight`
- `load_replicated_weight`
- `router`
- `moe_kernel`

Qwen2 / Qwen3 子类只需要覆盖：

- 是否有 shared expert。
- router 是否 renormalize。
- shared expert forward 和 overlap 策略。

建议结构：

```text
nanovllm/executor/moe/
  router.py
  config.py
  kernel.py
  prepare_finalize/
    base.py
    no_ep.py
    torch_all2all.py
    deepep.py
  experts/
    base.py
    triton_grouped_gemm.py
  blocks/
    base_sparse_moe.py
```

也可以先保持文件较少：

```text
nanovllm/executor/moe/router.py
nanovllm/executor/moe/kernel.py
nanovllm/executor/moe/prepare_finalize.py
nanovllm/executor/moe/experts.py
```

等稳定后再拆目录。

## 6. 推荐落地路线

### Stage 1：机械拆分，不改行为

目标：保持结果一致，只把职责拆开。

任务：

1. 新增 `MoEPrepareFinalize`、`PrepareResult`。
2. 把 `TritonMoEBackend.dispatch/combine` 移到 `TorchAllToAllPrepareFinalize`。
3. 把 `TritonMoEBackend.compute` 移到 `TritonGroupedGemmExperts`。
4. 新增 `MoEKernel` 串联：`prepare -> experts -> finalize`。
5. 保留旧 `TritonMoEBackend` 作为兼容 wrapper，内部调用新组件。

收益：

- 行为不变。
- 后续 DeepEP 只替换 prepare/finalize。
- 后续 FP8 只替换 experts kernel。

### Stage 2：抽 Router

目标：减少 Qwen2 / Qwen3 重复。

任务：

1. 新增 `MoERouter` 和 `SoftmaxTopKRouter`。
2. Qwen2 使用 `renormalize=False`。
3. Qwen3 使用 `renormalize=True`。
4. 模型层 forward 中只保留：

```python
router_logits = self.gate(x)
topk_weights, topk_ids = self.router.select_experts(x, router_logits)
sparse_out = self.moe_kernel(...)
```

### Stage 3：抽 BaseSparseMoeBlock

目标：统一权重布局与加载逻辑。

任务：

1. 合并 Qwen2 / Qwen3 中重复的：
   - `w13_stacked / w2_stacked` 创建。
   - `load_hybrid_moe_weight`。
   - `load_replicated_weight`。
   - `local_num_experts / local_inter_size` 计算。
2. Qwen2 只保留 shared expert 逻辑。
3. Qwen3 只保留无 shared expert 的配置差异。

### Stage 4：加 NoEPPrepareFinalize

目标：明确单卡 / `ep_size=1` 路径。

任务：

1. `ep_size == 1` 时使用 `NoEPPrepareFinalize`。
2. 不走 `dist.all_to_all_single`。
3. 保持 `supports_cuda_graph=True`。
4. 用它做单卡 MoE kernel benchmark baseline。

### Stage 5：CUDA Graph guard

目标：避免动态 EP 通信误进 CUDA Graph。

任务：

1. `MoEPrepareFinalize` 暴露：

```python
supports_cuda_graph: bool
uses_dynamic_alltoall: bool
```

2. `model_runner.capture_cudagraph()` 判断：

```python
if any_moe_backend_not_graph_safe:
    skip capture
```

短期可以先粗暴处理：

```python
if self.ep_size > 1:
    skip capture
```

### Stage 6：接 DeepEPPrepareFinalize

目标：只替换 EP 通信，不动专家计算。

任务：

1. 实现 `DeepEPPrepareFinalize`。
2. high-throughput / low-latency 可以先作为两个类或一个类的 mode。
3. 保持输出 `PrepareResult` 与 `TorchAllToAllPrepareFinalize` 一致。
4. 与 `TritonGroupedGemmExperts` 组合验证。

组合：

```text
DeepEPPrepareFinalize + TritonGroupedGemmExperts
```

后续再做：

```text
DeepEPPrepareFinalize + FP8GroupedGemmExperts
```

### Stage 7：专家 kernel 量化扩展

目标：专家计算独立优化。

任务：

1. `TritonGroupedGemmExperts` 支持 bf16/fp16。
2. `FP8GroupedGemmExperts` 支持 FP8 w8a8 / w8a16。
3. `INT8GroupedGemmExperts` 支持 int8 w8a16。
4. scale、zero point 等量化参数收口到 experts kernel 或 quant config，不放进 EP 通信层。

## 7. 与 vLLM 的对应关系

| nano 建议模块 | vLLM 对应模块 | 说明 |
|---|---|---|
| `MoERouter` | `BaseRouter / FusedTopKRouter` | routing、top-k、renormalize |
| `MoEPrepareFinalize` | `FusedMoEPrepareAndFinalizeModular` | quantize、dispatch、combine、finalize |
| `MoEExpertsKernel` | `FusedMoEExpertsModular` | 本地专家 GEMM / activation |
| `MoEKernel` | `FusedMoEKernel` | 串联 prepare -> experts -> finalize |
| `BaseSparseMoeBlock` | `FusedMoE + MoERunner` 的轻量版本 | 模型层 MoE orchestration |
| `TorchAllToAllPrepareFinalize` | `MoEPrepareAndFinalizeNaiveDPEPModular` / EP group | torch 通信 baseline |
| `NoEPPrepareFinalize` | `MoEPrepareAndFinalizeNoDPEPModular` | 单卡 / 非 EP baseline |
| `DeepEPPrepareFinalize` | `DeepEPHTPrepareAndFinalize` / `DeepEPLLPrepareAndFinalize` | DeepEP 通信后端 |

## 8. 最小可执行改造示意

改造前：

```python
dispatch_state = self.backend.dispatch(...)
local_out = self.backend.compute(...)
output = self.backend.combine(...)
```

改造后：

```python
router_logits = self.gate(x)
topk_weights, topk_ids = self.router.select_experts(x, router_logits)

output = self.moe_kernel(
    x,
    topk_weights,
    topk_ids,
    self.w13_stacked,
    self.w2_stacked,
    model_dtype=x.dtype,
    reduce_tp=True,
)
```

`MoEKernel` 内部：

```python
prepared = self.prepare_finalize.prepare(x, topk_weights, topk_ids)
local_out = self.experts.apply(
    prepared.hidden_states,
    prepared.topk_ids,
    prepared.topk_weights,
    w13,
    w2,
    ...,
)
output = self.prepare_finalize.finalize(local_out, prepared, output_shape=x.shape)
```

## 9. 优先不要做的事情

短期不建议：

- 直接复制 vLLM 的完整 `FusedMoE` / `MoERunner` / `CustomOp` 体系。
- 一开始就实现 EPLB、redundant experts、expert placement strategy。
- 一开始就把 DeepEP、FP8、INT8 同时塞进同一个 backend。
- 让 `DeepEPPrepareFinalize` 直接依赖某个专家 kernel。
- 把量化 scale 逻辑和 all-to-all 通信强耦合。

应该先保证接口边界稳定，再逐步替换组件。

## 10. 最终目标

完成上述改造后，nano 的 MoE 实验应该能用配置组合表达：

```python
prepare_finalize = NoEPPrepareFinalize(...)
experts = TritonGroupedGemmExperts(...)
```

或：

```python
prepare_finalize = TorchAllToAllPrepareFinalize(...)
experts = TritonGroupedGemmExperts(...)
```

或：

```python
prepare_finalize = DeepEPPrepareFinalize(mode="low_latency")
experts = FP8GroupedGemmExperts(...)
```

这样每次优化都可以明确回答：

- 是 EP 通信变快了？
- 是 expert GEMM 变快了？
- 是 router / align / combine 减少了开销？
- 是量化减少了通信量还是提升了 GEMM throughput？

这就是向 vLLM 对齐最值得借鉴的部分：接口边界清晰，后端可插拔，方便独立优化和对比。
