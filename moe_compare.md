# MoE 实现过程对比：mini-sglang / nano-vllm-moe / transformers Qwen2Moe

本文对比当前目录下三套 MoE 实现：

- `mini-sglang`
- `nano-vllm-moe`
- `transformers` 中的 `qwen2_moe`

重点从 forward 执行过程、router、expert 权重布局、dispatch/compute/combine、shared expert、TP/EP 支持等角度展开。

## 1. 总体结论

| 项目 | 定位 | 实现特点 |
| --- | --- | --- |
| `transformers` Qwen2Moe | 标准 PyTorch/eager 语义实现 | 清晰、可训练、易读；按 expert 循环，推理性能不是主要目标 |
| `mini-sglang` | 推理 runtime 风格实现 | `MoELayer + backend` 分层简洁；使用 `sgl_kernel` topk/align + Triton fused GEMM；当前路径没有 Qwen2Moe shared expert |
| `nano-vllm-moe` | 推理优化 + MoE runtime 实验实现 | 支持 Qwen2Moe shared expert、stacked expert weight、TP/EP、all-to-all、Triton group GEMM；功能最完整但耦合较重 |

一句话概括：

- `transformers` 是最标准的参考语义实现。
- `mini-sglang` 是更干净的推理 backend 抽象。
- `nano-vllm-moe` 是功能更完整的 Qwen2Moe 推理 runtime 实现。

## 2. Transformers Qwen2Moe

相关文件：

- `transformers/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py`
- `transformers/src/transformers/models/qwen2_moe/configuration_qwen2_moe.py`

### 2.1 模型结构

`Qwen2MoeSparseMoeBlock` 由四部分组成：

- `Qwen2MoeTopKRouter`
- `Qwen2MoeExperts`
- `shared_expert`
- `shared_expert_gate`

核心结构在：

```text
Qwen2MoeSparseMoeBlock
  gate: Qwen2MoeTopKRouter
  experts: Qwen2MoeExperts
  shared_expert: Qwen2MoeMLP
  shared_expert_gate: Linear(hidden_size, 1)
```

### 2.2 Router 流程

`Qwen2MoeTopKRouter.forward()` 的流程：

```text
hidden_states
  -> F.linear(hidden_states, router_weight)
  -> softmax(router_logits)
  -> topk(router_probs)
  -> optional top-k renormalize
  -> routing_weights, selected_experts
```

对应逻辑：

```python
router_logits = F.linear(hidden_states, self.weight)
router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
if self.norm_topk_prob:
    router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
```

### 2.3 Expert 权重布局

Transformers 没有使用 `ModuleList` 保存每个 expert，而是用 stacked 3D 参数：

```python
self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim))
self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, intermediate_dim))
```

其中：

- `gate_up_proj[e]` 包含 expert `e` 的 `gate_proj` 和 `up_proj`
- `down_proj[e]` 是 expert `e` 的 `down_proj`

### 2.4 Expert 计算流程

`Qwen2MoeExperts.forward()` 是 eager/PyTorch loop 实现：

```text
top_k_index
  -> one_hot expert mask
  -> 找到命中的 experts
  -> for each expert:
       取出属于该 expert 的 tokens
       gate, up = linear(token, gate_up_proj[expert]).chunk(2)
       hidden = act(gate) * up
       hidden = linear(hidden, down_proj[expert])
       hidden *= routing_weight
       index_add_ 回 final_hidden_states
```

关键点：

- 按 expert 循环。
- 每个 expert 只处理路由到自己的 token。
- top-k 的多个 expert 输出通过 `index_add_` 累加回原 token 位置。
- 语义清晰，但没有 fused kernel，也没有 EP all-to-all。

### 2.5 Shared Expert

Qwen2Moe 还有 dense shared expert：

```text
shared_expert_output = shared_expert(hidden_states)
shared_weight = sigmoid(shared_expert_gate(hidden_states))
shared_expert_output *= shared_weight
```

最后：

```text
output = sparse_expert_output + shared_expert_output
```

### 2.6 Decoder Layer 中的 MoE 选择

`Qwen2MoeDecoderLayer` 根据配置决定当前层使用 sparse MoE 还是普通 MLP：

```python
if (layer_idx not in config.mlp_only_layers) and (
    config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
):
    self.mlp = Qwen2MoeSparseMoeBlock(config)
else:
    self.mlp = Qwen2MoeMLP(config, intermediate_size=config.intermediate_size)
```

### 2.7 Transformers 实现特点

优点：

- 最接近模型论文/权重语义。
- 易读、易调试、适合训练和验证正确性。
- 支持 router logits、aux loss 等训练相关逻辑。

不足：

- expert loop 是 Python 级别循环。
- 没有 fused dispatch/compute/combine。
- 没有 expert parallel all-to-all。
- 推理性能不是核心目标。

## 3. mini-sglang

相关文件：

- `mini-sglang/python/minisgl/models/qwen3_moe.py`
- `mini-sglang/python/minisgl/models/utils.py`
- `mini-sglang/python/minisgl/layers/moe.py`
- `mini-sglang/python/minisgl/moe/fused.py`
- `mini-sglang/python/minisgl/kernel/moe_impl.py`
- `mini-sglang/python/minisgl/kernel/triton/fused_moe.py`

### 3.1 模型入口

`qwen3_moe.py` 中的 decoder layer 使用：

```python
from .utils import MoEMLP as Qwen3MLP
```

也就是说，Qwen3Moe 的 MLP 被替换为 `MoEMLP`。

Decoder layer 中：

```text
input_layernorm
  -> self_attn
  -> post_attention_layernorm
  -> MoEMLP
```

### 3.2 MoEMLP：Router 和 MoELayer 分离

`MoEMLP` 位于 `models/utils.py`：

```python
self.experts = MoELayer(
    num_experts=config.num_experts,
    top_k=config.num_experts_per_tok,
    hidden_size=config.hidden_size,
    intermediate_size=config.moe_intermediate_size,
    renormalize=config.norm_topk_prob,
)
self.gate = LinearReplicated(
    config.hidden_size,
    config.num_experts,
    has_bias=False,
)
```

forward 流程：

```text
hidden_states
  -> gate.forward(hidden_states)
  -> router_logits
  -> experts.forward(hidden_states, router_logits)
```

即：

- `MoEMLP` 负责 router linear。
- `MoELayer` 负责 expert 权重和 backend 调用。
- 真正的 top-k、dispatch、expert compute 都交给 MoE backend。

### 3.3 MoELayer 权重布局

`MoELayer` 中保存 stacked expert weight：

```python
self.gate_up_proj = torch.empty(
    num_experts,
    2 * intermediate_size_per_partition,
    hidden_size,
)
self.down_proj = torch.empty(
    num_experts,
    hidden_size,
    intermediate_size_per_partition,
)
```

其中：

- `intermediate_size_per_partition = intermediate_size / tp_size`
- `gate_up_proj` 是 gate/up 合并权重
- `down_proj` 是 down 权重
- TP 下每个 rank 只保留 intermediate 维度的一片

### 3.4 Backend 调用

`MoELayer.forward()` 不直接实现 MoE 逻辑，而是调用全局 backend：

```python
final_hidden_states = ctx.moe_backend.forward(
    hidden_states=hidden_states,
    w1=self.gate_up_proj,
    w2=self.down_proj,
    gating_output=router_logits,
    topk=self.top_k,
    renormalize=self.renormalize,
    activation=self.activation,
    apply_router_weight_on_input=self.apply_router_weight_on_input,
)
```

TP 场景下：

```python
if self.tp_size > 1:
    final_hidden_states = self._comm.all_reduce(final_hidden_states)
```

### 3.5 FusedMoe TopK

`FusedMoe.forward()` 先调用 `fused_topk()`：

```python
topk_weights, topk_ids = fused_topk(
    hidden_states=hidden_states,
    gating_output=gating_output,
    topk=topk,
    renormalize=renormalize,
)
```

`fused_topk()` 使用 `sgl_kernel.topk_softmax`：

```python
from sgl_kernel import topk_softmax

topk_softmax(topk_weights, topk_ids, gating_output.float(), renormalize)
```

也就是把：

```text
softmax + topk
```

放进外部 kernel 中完成。

### 3.6 moe_align_block_size

`moe_align_block_size()` 调用 `sgl_kernel.moe_align_block_size`，生成 fused GEMM 所需的调度信息：

- `sorted_token_ids`
- `expert_ids`
- `num_tokens_post_pad`

作用：

```text
topk_ids
  -> 按 expert 排序 token
  -> 每个 expert 的 token 数按 BLOCK_SIZE_M padding
  -> 生成每个 block 对应的 expert id
```

这是 fused MoE kernel 的上游 schedule 阶段。

### 3.7 Expert Compute

`fused_experts_impl()` 是核心：

```text
hidden_states, w1, w2, topk_weights, topk_ids
  -> moe_align_block_size
  -> fused_moe_kernel_triton(hidden_states, w1)   # gate/up
  -> silu_and_mul / gelu_and_mul
  -> fused_moe_kernel_triton(intermediate, w2)    # down
  -> moe_sum_reduce_triton                        # top-k sum
```

第一阶段：

```python
fused_moe_kernel_triton(
    curr_hidden_states,
    w1,
    intermediate_cache1,
    curr_topk_weights,
    curr_topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    apply_router_weight_on_input,
    topk_ids.shape[1],
    config,
)
```

激活阶段：

```python
FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
```

第二阶段：

```python
fused_moe_kernel_triton(
    intermediate_cache2,
    w2,
    intermediate_cache3,
    curr_topk_weights,
    curr_topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    not apply_router_weight_on_input,
    1,
    config,
)
```

最后 reduce：

```python
moe_sum_reduce_triton(intermediate_cache3, out_hidden_states)
```

### 3.8 mini-sglang 实现特点

优点：

- 模型层和 MoE backend 分层较清楚。
- `MoELayer` 不关心具体 top-k、align、kernel 实现。
- 使用 fused topk、fused expert GEMM、Triton reduce，适合推理。
- TP 逻辑相对简单：按 intermediate 切分，最后 all-reduce。

不足：

- 当前看到的是 Qwen3Moe 路径，不是 Qwen2Moe 的完整 shared expert 语义。
- 没有 nano-vllm-moe 那种显式 EP dispatch/combine。
- `moe_align_block_size` 依赖 `sgl_kernel` 外部实现。

## 4. nano-vllm-moe

相关文件：

- `nano-vllm-moe/nanovllm/models/qwen2_moe.py`
- `nano-vllm-moe/nanovllm/executor/moe/backends/triton.py`
- `nano-vllm-moe/nanovllm/utils/moe.py`
- `nano-vllm-moe/nanovllm/kernels/group_gemm.py`
- `nano-vllm-moe/nanovllm/utils/loader.py`

### 4.1 两套 SparseMoeBlock

`qwen2_moe.py` 中有两个 MoE block：

```text
Qwen2MoeEagerSparseMoeBlock
Qwen2MoeSparseMoeBlock
```

其中：

- `Qwen2MoeEagerSparseMoeBlock` 是 PyTorch 对照版。
- `Qwen2MoeSparseMoeBlock` 是优化版，接入 Triton backend、TP、EP、stacked expert weight。

### 4.2 Eager SparseMoeBlock

Eager 版流程接近 Transformers：

```text
hidden_states
  -> gate
  -> softmax
  -> topk
  -> shared_expert + sigmoid(shared_expert_gate)
  -> one_hot(selected_experts)
  -> for each expert:
       expert(hidden_states[top_x])
       multiply routing weight
       index_add_ 回 sparse_hidden_states
  -> shared_output + sparse_hidden_states
```

这个实现适合：

- 校验 optimized 版正确性。
- 作为功能参考。
- 对比 fused MoE 的数值结果。

### 4.3 Optimized SparseMoeBlock 权重布局

优化版保存 local expert 权重：

```python
self.local_num_experts = self.global_num_experts // self.ep_size
self.local_inter_size = self.moe_intermediate_size // self.tp_size

self.w13_stacked = nn.Parameter(torch.zeros(
    self.local_num_experts, 2 * self.local_inter_size, self.hidden_size
))
self.w2_stacked = nn.Parameter(torch.zeros(
    self.local_num_experts, self.hidden_size, self.local_inter_size
))
```

含义：

- EP 维度：每个 rank 只持有一部分 experts。
- TP 维度：每个 rank 只持有 intermediate 维度的一片。
- `w13_stacked` 合并 `gate_proj` 和 `up_proj`。
- `w2_stacked` 保存 `down_proj`。

### 4.4 MoE 权重加载

`load_hybrid_moe_weight()` 负责把 checkpoint 中的 expert 权重切到对应 rank：

```python
if not (self.ep_rank * self.local_num_experts <= global_expert_id < (self.ep_rank + 1) * self.local_num_experts):
    return

local_id = global_expert_id % self.local_num_experts
start = self.tp_rank * self.local_inter_size
size = self.local_inter_size
```

加载规则：

```text
gate_proj -> w13_stacked[local_id, 0:size]
up_proj   -> w13_stacked[local_id, size:2*size]
down_proj -> w2_stacked[local_id]
```

`utils/loader.py` 中会拦截 checkpoint 名称：

```text
model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight
model.layers.{layer}.mlp.experts.{expert}.up_proj.weight
model.layers.{layer}.mlp.experts.{expert}.down_proj.weight
```

并映射到：

```text
w13_stacked
w2_stacked
```

### 4.5 Optimized Forward 主流程

`Qwen2MoeSparseMoeBlock.forward()`：

```text
hidden_states
  -> flatten to [M, H]
  -> optionally launch shared expert on separate CUDA stream
  -> router gate
  -> softmax
  -> topk
  -> backend.dispatch
  -> backend.compute
  -> backend.combine
  -> wait shared expert stream
  -> shared_out + sparse_out
  -> TP all_reduce
  -> reshape back
```

伪代码：

```python
x = hidden_states.view(-1, hidden_size).contiguous()

shared_out = shared_expert(x) * sigmoid(shared_expert_gate(x))

router_logits = self.gate(x)
routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)

dispatch_state = self.backend.dispatch(...)
local_out_fp32 = self.backend.compute(...)
sparse_out = self.backend.combine(...)

output = shared_out + sparse_out
if tp_size > 1:
    dist.all_reduce(output, group=tp_group)
```

### 4.6 Backend Dispatch

`TritonMoEBackend.dispatch()` 做 token 到 expert rank 的分发。

#### EP size = 1

如果没有 EP：

```text
x repeat_interleave top_k
recv_local_ids = topk_ids.flatten()
recv_weights = topk_weights.flatten()
```

也就是说，每个 token 按 top-k 展开成本地任务。

#### EP size > 1

如果有 EP：

```text
target_ep_ranks = topk_ids // local_num_experts
flat_target_ep_ranks = target_ep_ranks.flatten()
permute_indices = argsort(flat_target_ep_ranks)
expanded_x = x.repeat_interleave(top_k)
dispatched_x = expanded_x[permute_indices]
```

然后通过 `all_to_all_single` 交换：

- token hidden states
- local expert ids
- routing weights

核心逻辑：

```python
dist.all_to_all_single(recv_x, dispatched_x, r_list, s_list, group=self.ep_group)
dist.all_to_all_single(recv_local_ids, local_ids, r_list, s_list, group=self.ep_group)
dist.all_to_all_single(recv_weights, weights, r_list, s_list, group=self.ep_group)
```

### 4.7 moe_align_block_size

`nanovllm/utils/moe.py` 中的 `moe_align_block_size()` 用纯 torch 实现调度信息生成：

输入：

```text
recv_local_ids: [num_recv, 1]
num_experts
block_size
```

输出：

```text
sorted_token_ids
sorted_weight_idx
expert_ids
num_blocks
```

其中：

- `sorted_token_ids`：按 expert 排序并 padding 后的 token id。
- `sorted_weight_idx`：对应 `recv_weights` 的索引，用于 W2 combine kernel 找 routing weight。
- `expert_ids`：每个 block 对应哪个 expert。
- `num_blocks`：总 block 数。

相比 mini-sglang，这里多了 `sorted_weight_idx`，因为 nano 的 W2 kernel 直接做 routing weight multiply 和 atomic add。

### 4.8 Backend Compute

`TritonMoEBackend.compute()` 流程：

```text
recv_x, recv_local_ids, recv_weights
  -> optional FP8 dequant
  -> moe_align_block_size
  -> fused_moe_w13_kernel
  -> fused_moe_w2_combine_kernel
  -> local_out_fp32
```

如果权重是 FP8 存储，会先做：

```python
w13_stacked = w13_stacked.view(torch.float8_e4m3fn).to(model_dtype)
w13_stacked = w13_stacked * w13_weight_scale.to(model_dtype).unsqueeze(-1)
```

然后执行两个 Triton kernel。

### 4.9 fused_moe_w13_kernel

`fused_moe_w13_kernel` 位于 `kernels/group_gemm.py`。

它做：

```text
recv_x
  -> 按 sorted_token_ids 取 token
  -> 按 expert_ids 取 expert weight
  -> 同时计算 gate 和 up
  -> activated = silu(gate) * up
  -> 写入 activated_out
```

其中 SwiGLU 在 kernel 内完成：

```python
activated = (acc_gate * tl.sigmoid(acc_gate)) * acc_up
```

### 4.10 fused_moe_w2_combine_kernel

`fused_moe_w2_combine_kernel` 做：

```text
activated_out
  -> down projection
  -> multiply routing weight
  -> atomic_add 回 token 输出位置
```

核心逻辑：

```python
route_weight = tl.load(routing_weights_ptr + offs_weight)
accumulator = tl.dot(a, b)
accumulator = accumulator * route_weight[:, None]
tl.atomic_add(c_ptrs, accumulator)
```

这里和 mini-sglang 不同：

- nano 在 W2 kernel 内完成 routing weight multiply。
- nano 在 W2 kernel 内 atomic add 回 token 维度。
- mini-sglang 是 W2 后得到 `[M, top_k, H]`，再用 `moe_sum_reduce_triton` 做 top-k reduce。

### 4.11 Backend Combine

`combine()` 负责把 EP 计算结果发回原 rank，并恢复 token 顺序。

EP size > 1：

```python
dist.all_to_all_single(combined_x, local_out_fp32.to(model_dtype), s_list, r_list, group=self.ep_group)
```

然后：

```python
sparse_out_flat[permute_indices] = combined_x
output = sparse_out_flat.view(m_tokens, top_k, hidden_size).sum(dim=1)
```

EP size = 1：

```python
output = combined_x.view(m_tokens, top_k, hidden_size).sum(dim=1)
```

如果 `reduce_results=True` 且 TP size > 1，还会做 TP all-reduce。

在 `Qwen2MoeSparseMoeBlock.forward()` 里，`combine(..., reduce_results=False)`，最后统一对 `shared_out + sparse_out` 做 TP all-reduce。

### 4.12 nano-vllm-moe 实现特点

优点：

- 完整保留 Qwen2Moe shared expert 语义。
- 支持 TP 和 EP。
- 支持 expert all-to-all dispatch/combine。
- 使用 stacked expert weight，便于 fused kernel。
- `w13` 和 `w2_combine` 两个 Triton kernel 更贴合 Qwen2Moe 的推理计算。
- shared expert 可以和 sparse expert 路径 overlap。

不足：

- `Qwen2MoeSparseMoeBlock` 同时关心 router、shared expert、backend、TP/EP、weight loading，模型逻辑和 runtime 逻辑耦合较重。
- `TritonMoEBackend` 同时负责 dispatch、compute、combine，不是纯 expert kernel backend。
- `moe_align_block_size()` 是 execution schedule 的一部分，但目前放在 utils 中。
- EP 通信和 expert compute 尚未完全分层，后续接 DeepEP 或替换 kernel 时需要进一步拆分。

## 5. 三者 forward 流程对齐

### 5.1 Transformers

```text
hidden_states
  -> router linear
  -> softmax
  -> topk
  -> one_hot expert mask
  -> Python for each expert
  -> gate/up linear
  -> activation * up
  -> down linear
  -> multiply routing weight
  -> index_add_ sparse output
  -> shared expert + sigmoid gate
  -> sparse + shared
```

### 5.2 mini-sglang

```text
hidden_states
  -> LinearReplicated router
  -> MoELayer
  -> backend.fused_topk via sgl_kernel.topk_softmax
  -> sgl_kernel.moe_align_block_size
  -> Triton fused_moe_kernel for w1/gate_up
  -> silu_and_mul or gelu_and_mul
  -> Triton fused_moe_kernel for w2/down
  -> Triton moe_sum_reduce
  -> TP all_reduce
```

### 5.3 nano-vllm-moe

```text
hidden_states
  -> flatten [M, H]
  -> shared expert on optional CUDA stream
  -> router linear
  -> softmax
  -> topk
  -> EP dispatch all-to-all if needed
  -> moe_align_block_size
  -> Triton fused_moe_w13_kernel
  -> Triton fused_moe_w2_combine_kernel
  -> EP combine all-to-all if needed
  -> restore token order and sum top-k
  -> shared + sparse
  -> TP all_reduce
```

## 6. 关键差异总结

| 维度 | Transformers | mini-sglang | nano-vllm-moe |
| --- | --- | --- | --- |
| Router | PyTorch `linear + softmax + topk` | `sgl_kernel.topk_softmax` | PyTorch `linear + softmax + topk` |
| Expert 权重 | stacked 3D 参数，全量 experts | stacked 3D 参数，按 TP 切 intermediate | stacked 3D 参数，按 EP 切 experts、按 TP 切 intermediate |
| Expert 计算 | Python expert loop | Triton fused MoE GEMM + reduce | Triton `w13` + `w2_combine` group GEMM |
| Shared expert | 有 | 当前 Qwen3Moe 路径无 | 有，并支持 overlap |
| Top-k 合并 | `index_add_` | `moe_sum_reduce_triton` | W2 kernel 内 `atomic_add` + combine sum |
| TP | 标准 HF，不是自定义 TP runtime | intermediate shard + all-reduce | intermediate shard + all-reduce |
| EP | 无 | 当前路径无显式 EP all-to-all | 有 all-to-all dispatch/combine |
| 代码边界 | 模型语义清楚 | backend 抽象清楚 | 功能完整但 runtime 耦合较重 |
| 适用场景 | 正确性参考、训练、模型语义 | 简洁推理 runtime | MoE 推理优化、TP/EP 实验 |

## 7. 如果继续演进 nano-vllm-moe

从当前结构看，`nano-vllm-moe` 可以继续拆成更清晰的四层：

```text
MoERouter
MoEDispatcher
MoEExpertKernel
MoECombiner / MoERuntime
```

建议方向：

1. 把 router 逻辑从 `Qwen2MoeSparseMoeBlock` 中抽出。
2. 把 shared expert 作为 Qwen2Moe-specific hook 保留在模型层。
3. 把 EP all-to-all dispatch/combine 从 `TritonMoEBackend` 中拆成 dispatcher。
4. 把 `fused_moe_w13_kernel` / `fused_moe_w2_combine_kernel` 包成纯 expert kernel backend。
5. 让 `moe_align_block_size()` 成为 expert kernel schedule 的一部分，而不是普通 utils。
6. 后续接 DeepEP 时，尽量让 DeepEP 输出能直接对接 expert kernel 所需的 sorted/schedule 信息，避免重复 align。

目标是让：

- Qwen2/Qwen3 模型差异留在模型层。
- EP 通信差异留在 dispatcher 层。
- Triton/FP8/naive expert 差异留在 expert kernel 层。
- combine/reduce 逻辑统一收敛到 runtime 层。
