# MoE / EP 分析与演进路线

## 1. 目标

这份文档的目标是把当前项目里的 MoE 与 EP 实现拆清楚，并给出一条适合继续演进的封装路线。核心诉求有两个：

1. 先单独优化 MoE。
2. 能够低成本替换和比较不同 MoE 与 EP 实现。

建议的总路线是：

- 先把 MoE expert compute 和 EP communication 解耦。
- 先做 `naive MoE vs fused MoE` 的统一抽象和对比。
- 再做 `torch EP vs DeepEP` 的统一抽象和对比。

---

## 2. 当前实现概览

当前项目中的 MoE 主体实现集中在以下几个文件：

- `nanovllm/models/qwen2_moe.py`
- `nanovllm/models/qwen3_moe.py`
- `nanovllm/executor/moe/backends/base.py`
- `nanovllm/executor/moe/backends/triton.py`
- `nanovllm/utils/moe.py`
- `nanovllm/kernels/group_gemm.py`
- `nanovllm/utils/loader.py`

### 2.1 当前执行链

当前 sparse MoE 的主流程基本是：

1. `gate(x)` 计算 router logits
2. `softmax + topk`
3. 按 expert / ep rank 分发 token
4. 执行 expert compute
5. 回收结果并做 top-k combine
6. 如有 TP，做 all-reduce

Qwen2 和 Qwen3 都遵循这条主线，但有模型差异：

- Qwen2 有 shared expert
- Qwen3 没有 shared expert
- Qwen3 对 top-k weights 还要做归一化

### 2.2 当前 Eager 与 Triton 两条路径

当前实际存在两种 MoE 形式：

- Eager 参考实现
- Triton Group-GEMM fused 实现

它们分别散落在：

- `Qwen2MoeEagerSparseMoeBlock`
- `Qwen2MoeSparseMoeBlock`
- `Qwen3MoeEagerSparseMoeBlock`
- `Qwen3MoeSparseMoeBlock`

也就是说，目前“算法差异”和“模型差异”是混在一起的。

### 2.3 当前 backend 的职责过重

`nanovllm/executor/moe/backends/triton.py` 里的 `TritonMoEBackend` 当前同时承担：

- EP=1 与 EP>1 的 token dispatch
- 本地 expert 排序与 block 对齐
- Triton fused expert compute
- combine / inverse permute
- TP all-reduce
- 一部分 FP8 权重处理

这会带来一个直接问题：

当前的 backend 更像是一整套 MoE runtime，而不是单纯的一个可替换 backend。

这会让后续做下面这些事情都比较别扭：

- 比较 naive MoE 和 fused MoE
- 比较不同 dispatch 路线
- 接入 DeepEP
- 接入不同量化 expert kernel

---

## 3. 当前代码结构分析

## 3.1 模型层承担了过多调度逻辑

在 `qwen2_moe.py` 和 `qwen3_moe.py` 里，`SparseMoeBlock.forward()` 不只是“定义模型结构”，它还直接串联了：

- routing
- dispatch
- backend compute
- combine
- shared expert overlap

这意味着模型层和 runtime 层耦合较深。

比较明显的重复包括：

- Qwen2/Qwen3 的 stacked expert weight 管理
- `load_hybrid_moe_weight()`
- gate/topk/dispatch/compute/combine 主干流程

目前两者真正的结构差异其实不多，主要是：

- `topk_weights` 的后处理策略不同
- 是否有 shared expert

所以现状适合抽公共基类。

## 3.2 `moe_align_block_size` 是当前关键调度热点

`nanovllm/utils/moe.py` 中的 `moe_align_block_size()` 当前负责：

- 统计每个 expert 的 token 数
- 按 expert 排序
- 生成 block 对齐后的 `sorted_token_ids`
- 生成对应 `sorted_weight_idx`
- 构造 `expert_ids`

这一步是 fused kernel 的上游 schedule 阶段，本质上已经不是“工具函数”，而是 MoE execution schedule 的一部分。

它的特征是：

- 对 decode 小 batch 场景可能很敏感
- 当前是纯 torch 张量操作
- 后续很适合被替换成 device kernel 或 dispatcher 原生输出

所以不建议继续把它放成一个普通 util 函数并隐藏在 backend 内部。

## 3.3 当前 EP 与 expert compute 没有分层

现在 EP 路径直接写在 `TritonMoEBackend.dispatch()` 和 `combine()` 里，核心模式是：

- 先按 `target_ep_rank` 排序
- `all_to_all_single` 发 `x`
- `all_to_all_single` 发 `local expert ids`
- `all_to_all_single` 发 `routing weights`
- 本地算完后再 `all_to_all_single` 发回
- inverse permute 恢复顺序

这条路径能工作，但它的问题不是“功能不对”，而是“抽象位置不对”：

- EP 是 communication concern
- expert kernel 是 compute concern

现在两者绑在一个 backend 里，后续如果你想：

- 保持 Triton compute 不变，只换通信方案
- 保持通信方案不变，只换 fused/naive kernel

都会显得很笨重。

## 3.4 运行时切换能力不足

当前运行时能切换的只有很有限的开关，例如：

- `group_gemm_enable`
- `enforce_eager`
- `tp_size`
- `ep_size`

但没有一个一等的配置项来表达：

- 我要用哪个 MoE expert kernel
- 我要用哪个 dispatcher
- 我要用哪个 EP backend

这会让对比实验只能靠改代码，而不是改配置。

---

## 4. 当前实现的优点与问题

## 4.1 优点

当前实现已经具备几个很有价值的基础：

- 已有 Eager 参考路径，适合作为数值对齐基线
- 已有 Triton fused expert kernel，性能优化方向明确
- 已有 TP + EP 混合并行基础框架
- 已有 stacked expert weight 布局，便于做 fused expert compute
- 已有 benchmark 脚本和部分测试

## 4.2 问题

### 问题 1：模型差异和实现差异耦合

Qwen2 / Qwen3 的“模型逻辑差异”和“MoE 实现差异”耦合在同一个类层级里，导致后续替换实现会碰到模型文件。

### 问题 2：backend 太胖

当前 `TritonMoEBackend` 既不是纯 dispatch backend，也不是纯 compute backend，而是整条 runtime 链路。

### 问题 3：没有统一的 dispatch state

当前 dispatch 返回 dict，字段本身虽然够用，但不稳定、不显式，后续很难扩展不同 backend 的 contract。

### 问题 4：MoE compare 与 EP compare 混在一起

如果不先解耦，后续任何性能差异都可能来自：

- router
- dispatch
- expert kernel
- combine
- TP reduction
- 通信实现

这样实验结论会不干净。

### 问题 5：未来 DeepEP 接入点不够清晰

DeepEP 最自然的接入层应该是 dispatcher 层，而不是直接替换整套 Triton backend。

---

## 5. 推荐的目标架构

建议把当前 MoE runtime 拆成 4 层。

## 5.1 `BaseSparseMoeBlock`

职责：

- 定义 MoE forward 主流程
- 调 router
- 调 dispatcher
- 调 expert kernel
- 调 combine
- 可选 shared expert merge

这个类应该是模型层和执行层之间的桥。

它不关心具体用的是 naive expert kernel 还是 Triton fused expert kernel，也不关心 EP 是 torch all-to-all 还是 DeepEP。

它只关心流程。

### 需要保留的模型 hook

建议只保留少量模型差异 hook：

- `route_logits(x)`
- `postprocess_topk_weights(weights)`
- `compute_shared_expert(x)`
- `merge_shared_and_sparse(shared_out, sparse_out)`

这样：

- Qwen2 负责 shared expert
- Qwen3 负责 topk 权重归一化

其余逻辑共用。

## 5.2 `DispatchState`

建议引入显式 dataclass，而不是继续返回裸 dict。

建议字段至少包括：

- `recv_x`
- `recv_local_ids`
- `recv_weights`
- `permute_indices`
- `send_counts`
- `recv_counts`
- `num_recv`
- `tokens_per_expert`
- `schedule_meta`

说明：

- `tokens_per_expert` 便于后续 DeepEP 直传
- `schedule_meta` 可留给 block-aligned schedule、expert map 等扩展

## 5.3 `TokenDispatcher`

职责：

- 接收 `x`, `topk_ids`, `topk_weights`
- 按目标执行本地重排或跨卡通信
- 输出 `DispatchState`

建议实现三个版本：

- `LocalTokenDispatcher`
- `TorchAllToAllDispatcher`
- `DeepEPDispatcher`

这样 EP 实现会变成一个单独的可替换维度。

## 5.4 `MoEExpertKernel`

职责：

- 只做 expert compute
- 不做跨卡通信
- 不负责 router
- 不负责 TP all-reduce

建议实现三个版本：

- `NaiveExpertKernel`
- `TritonFusedExpertKernel`
- 后续 `TritonFP8ExpertKernel`

这样你就能单独比较：

- `naive MoE`
- `fused MoE`
- 后续 `fp8 fused MoE`

## 5.5 `MoERuntime` 或 `Combiner`

职责：

- 把 dispatcher 和 kernel 串起来
- 处理 combine
- 处理 EP 回收
- 处理 TP all-reduce

这里可以做成：

- 一个独立 `MoERuntime`
- 或者 `dispatcher + kernel + combine` 的轻 glue 层

重点不是类名，而是职责边界清晰。

---

## 6. 推荐目录结构

建议往下面这个结构演进：

```text
nanovllm/executor/moe/
  block.py
  router.py
  runtime.py
  state.py
  dispatchers/
    base.py
    local.py
    torch_ep.py
    deepep.py
  kernels/
    base.py
    naive.py
    triton_fused.py
    triton_fp8.py
```

模型文件中：

- `qwen2_moe.py`
- `qwen3_moe.py`

只保留：

- 模型层结构
- shared expert 定义
- router 差异策略
- weight layout 注册

不要继续在模型文件里堆完整的 runtime 调度逻辑。

---

## 7. 推荐演进路线

下面这条路线的核心原则是：

- 先稳定 MoE 本地算子层
- 再抽离 EP 通信层
- 最后接 DeepEP 和量化

## Stage 1：抽公共 `BaseSparseMoeBlock`

目标：

- 不改行为
- 先统一结构

需要做的事：

1. 合并 Qwen2/Qwen3 sparse block 的公共 forward 主流程
2. 抽出 shared expert hook
3. 抽出 top-k 后处理 hook
4. 合并 `load_hybrid_moe_weight()`
5. 统一 stacked expert weight 布局接口

产出：

- Qwen2/Qwen3 MoE 主干逻辑统一
- 后续切 kernel 时不用同时改两个模型文件

这是第一优先级。

## Stage 2：拆分 `dispatcher` 与 `expert kernel`

目标：

- 让 naive MoE 和 fused MoE 可替换、可比较

建议先固定不做 EP，对齐在单卡或 `ep_size=1` 场景完成。

### 2.1 先实现 `LocalTokenDispatcher`

行为：

- `ep_size=1` 时只做本地展开与重排
- 输出统一 `DispatchState`

### 2.2 实现 `NaiveExpertKernel`

建议逻辑：

- 基于当前 eager 实现思路
- 按 expert 分桶
- gather token
- 调本地 MLP 或等价 stacked weight matmul
- scatter / index_add 回输出

这个实现的价值不是性能，而是：

- 清晰
- 易 debug
- 易做 reference

### 2.3 实现 `TritonFusedExpertKernel`

初期直接把当前 `TritonMoEBackend.compute()` 中的逻辑迁过去：

- `moe_align_block_size()`
- `fused_moe_w13_kernel`
- `fused_moe_w2_combine_kernel`

这样第二阶段结束后，你就可以直接比较：

- `LocalTokenDispatcher + NaiveExpertKernel`
- `LocalTokenDispatcher + TritonFusedExpertKernel`

这正是你想先做的 MoE compare。

## Stage 3：把 EP 从当前 backend 中单独拆出来

目标：

- 在不动 expert kernel 的前提下比较不同 EP 方案

### 3.1 实现 `TorchAllToAllDispatcher`

把当前 `TritonMoEBackend.dispatch()` 和 `combine()` 中与 EP 相关的逻辑迁移出来：

- rank mapping
- sort / permute
- `all_to_all_single`
- inverse permute
- 回收与还原

### 3.2 保持 expert kernel 完全不感知通信细节

kernel 层只看到：

- `recv_x`
- `recv_local_ids`
- `recv_weights`
- `tokens_per_expert` 或 block schedule

不要让 kernel 层知道：

- send counts 来自哪里
- EP 用了什么库
- 回收是怎么做的

这样第三阶段结束后，你就能直接比较：

- `torch_ep + naive kernel`
- `torch_ep + triton fused kernel`

## Stage 4：接入 DeepEP

目标：

- 让 DeepEP 成为 dispatcher 维度，而不是全链路重写

建议做法：

1. 单独实现 `DeepEPDispatcher`
2. 输出尽量复用 `DispatchState`
3. 如果 DeepEP 本身已按 expert 排好 token，则直接输出：
   - `recv_x`
   - `recv_local_ids` 或 `tokens_per_expert`
   - `recv_weights`
4. 尽量绕过额外的 `moe_align_block_size()`

关键原则：

不要为了接 DeepEP 去污染 Triton expert kernel 的职责边界。

## Stage 5：量化 MoE 与 FP8 MoE

目标：

- 在已有 `MoEExpertKernel` 抽象上自然扩展量化方案

建议新增：

- `TritonFP8ExpertKernel`
- 后续 INT8 / W4A8 对应 expert kernel

这一步不应该再改 dispatcher 抽象。

---

## 8. 推荐的实验矩阵

建议分两轮做，不要一开始把所有维度混在一起。

## 8.1 第一轮：只比较 MoE expert kernel

固定：

- `ep_size=1`
- 相同 router
- 相同输入
- 相同 weight layout

比较：

1. `local + naive`
2. `local + triton_fused`

关注指标：

- 数值一致性
- prefill 吞吐
- decode 吞吐
- `moe_align_block_size` 占比
- 小 batch 与大 batch 的收益差异

## 8.2 第二轮：比较 EP 路线

固定：

- 相同 expert kernel
- 相同 router

比较：

1. `torch_ep + naive`
2. `torch_ep + triton_fused`
3. `deepep + triton_fused`

关注指标：

- all-to-all 延迟
- token imbalance 下的性能
- 小 batch decode 下的退化程度
- 是否能减少额外 permute / regroup 开销

---

## 9. 为什么这个路线更适合当前项目

原因有四个。

## 9.1 它符合当前代码的真实边界

你现在已经有：

- eager reference
- fused Triton kernel
- torch distributed EP

所以不是从零设计，而是把已经存在的东西按职责重新分层。

## 9.2 它能保证实验结论干净

如果不拆层，你很难回答性能提升到底来自：

- dispatch
- expert kernel
- combine
- 通信
- weight layout

拆成 `dispatcher x expert_kernel` 二维后，结论会清楚很多。

## 9.3 它最利于接 DeepEP

DeepEP 最自然是 dispatcher 层增强，不是整套 MoE runtime 重写。

## 9.4 它最利于后续量化扩展

FP8 / INT8 / W4A8 更像 expert compute 维度的变化，不应该和 EP dispatch 耦合。

---

## 10. 当前最值得优先处理的点

按优先级排序，我建议这样做。

### P0：抽 `BaseSparseMoeBlock`

这是所有后续演进的入口。

### P1：把 `TritonMoEBackend` 拆成

- `TorchAllToAllDispatcher`
- `TritonFusedExpertKernel`

这是最关键的职责切分。

### P2：把 `moe_align_block_size` 升级成显式 schedule 阶段

不要继续把它当普通 util。

### P3：加显式 runtime config

建议在 config 中增加：

- `moe_kernel`
- `moe_dispatch`
- `ep_backend`
- `moe_use_shared_expert_overlap`

### P4：补测试

当前测试还不够覆盖结构重构后的稳定性。

建议最少补：

- `tests/moe/test_dispatch_roundtrip.py`
- `tests/moe/test_eager_vs_fused.py`
- `tests/moe/test_router_policy.py`
- `tests/moe/test_weight_loader_stacked.py`
- 后续 `tests/moe/test_deepep_dispatch.py`

---

## 11. 一个推荐的最终组合方式

最终希望达到的调用方式应接近：

```python
moe_block = SparseMoeBlock(
    router=Qwen3RouterPolicy(...),
    dispatcher=TorchAllToAllDispatcher(...),
    expert_kernel=TritonFusedExpertKernel(...),
    shared_expert=None,
)
```

或者单卡对比：

```python
moe_block = SparseMoeBlock(
    router=Qwen3RouterPolicy(...),
    dispatcher=LocalTokenDispatcher(...),
    expert_kernel=NaiveExpertKernel(...),
    shared_expert=None,
)
```

再或者未来 DeepEP：

```python
moe_block = SparseMoeBlock(
    router=Qwen2RouterPolicy(...),
    dispatcher=DeepEPDispatcher(...),
    expert_kernel=TritonFusedExpertKernel(...),
    shared_expert=Qwen2SharedExpert(...),
)
```

这种组合方式的最大好处是：

- 比较 MoE，不动 EP
- 比较 EP，不动 MoE kernel
- 加量化，只动 expert kernel

---

## 12. 结论

当前项目的 MoE 实现已经具备不错的基础，但现状更像“功能堆叠完成”，还不是“适合长期演进的抽象完成”。

最合适的路线不是继续在 `TritonMoEBackend` 上叠功能，而是尽快把当前实现拆成两个独立维度：

- `TokenDispatcher`
- `MoEExpertKernel`

然后在模型层之上用统一的 `BaseSparseMoeBlock` 串起来。

这样你就可以按下面顺序稳步推进：

1. 统一 Qwen2/Qwen3 的 sparse block 结构
2. 单独比较 `naive MoE vs fused MoE`
3. 单独比较 `torch EP vs DeepEP`
4. 在稳定抽象上继续做 FP8 / INT8 / W4A8

一句话总结：

先把“MoE 计算”和“EP 通信”拆开，再做优化和对比；否则后面的每一步都会继续互相污染。
