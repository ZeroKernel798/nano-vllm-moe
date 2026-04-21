# 量化层审计报告

> 生成时间：2026-04-21  
> 审计范围：`layers/quant_linear.py`、`layers/smooth_quant_linear.py`、`layers/fp8/parallel.py`（via `layers/fp8_linear.py`）

---

## 1. 类层次结构

| 特性 | AWQ INT4 | INT8 W8A8 | FP8 W8A16 / W8A8 |
|------|----------|-----------|-------------------|
| 基类 | `AWQLinearBase(LinearBase)` | `Int8LinearBase(nn.Module)` | `FP8LinearBase(nn.Module)` |
| 继承 `LinearBase`? | ✓（获得 `tp_rank`, `tp_group`, `tp_size`） | **✗**（直接 `nn.Module`） | **✗**（直接 `nn.Module`） |
| QKV parallel | `AWQQKVParallelLinear` | `Int8QKVParallelLinear` | `FP8QKVParallelLinear` |
| Merged column | `AWQMergedColumnParallelLinear` | `Int8MergedColumnParallelLinear` | `FP8MergedColumnParallelLinear` |
| Row parallel | `AWQRowParallelLinear` | `Int8RowParallelLinear` | `FP8RowParallelLinear` |
| `forward` 实现 | 调用 `F.linear(unpack_awq_int4(...))` | 调用 Triton `_w8a8_linear_kernel` | 调用 `launch_w8a16_gemm` 或 `launch_w8a8_static_gemm` |

---

## 2. Buffer 命名对比

| Buffer 语义 | AWQ INT4 | INT8 W8A8 | FP8 |
|-------------|----------|-----------|-----|
| 量化权重 | `qweight` `[K, N/8]` `int32`（packed） | `qweight_kn` `[K, N]` `int8` | `qweight` `[K, N]` `uint8` |
| 权重 scale（per-channel） | `scales` `[K//gs, N]` `float16`（2D，含 group 维度） | `weight_scales` `[N]` `float16` | `weight_scale` `[N]` `float32` |
| 零点 | `qzeros` `[K//gs, N/8]` `int32` | — | — |
| 输入 scale | — | — | `input_scale` `[1]` `float32`（仅 w8a8_static） |
| 转置权重缓存 | — | — | `qweight_nk` `[N, K]` `uint8`（仅 w8a8_static，预计算 qweight.T） |

---

## 3. weight_loader 签名对比

### QKV Parallel Linear

| 类 | 方法名 | 签名 |
|----|--------|------|
| `AWQQKVParallelLinear` | `qweight_loader` | `(param, loaded_weight, shard_id: str)` |
| `AWQQKVParallelLinear` | `qzeros_loader` | `(param, loaded_weight, shard_id: str)` |
| `AWQQKVParallelLinear` | `scales_loader` | `(param, loaded_weight, shard_id: str)` |
| `Int8QKVParallelLinear` | `qweight_loader` | `(param, loaded_weight, shard_id: str)` |
| `Int8QKVParallelLinear` | `weight_scales_loader` | `(param, loaded_weight, shard_id: str)` |
| `FP8QKVParallelLinear` | `qweight_loader` | `(param, loaded_weight, shard_id: str)` |
| `FP8QKVParallelLinear` | `weight_scale_loader` | `(param, loaded_weight, shard_id: str)` |
| `FP8QKVParallelLinear` | `input_scale_loader` | `(param, loaded_weight, shard_id: str)` |

### Merged Column Parallel Linear

| 类 | 方法名 | 签名 |
|----|--------|------|
| `AWQMergedColumnParallelLinear` | `qweight_loader` | `(param, loaded_weight, shard_id: int)` |
| `AWQMergedColumnParallelLinear` | `qzeros_loader` | `(param, loaded_weight, shard_id: int)` |
| `AWQMergedColumnParallelLinear` | `scales_loader` | `(param, loaded_weight, shard_id: int)` |
| `Int8MergedColumnParallelLinear` | `qweight_loader` | `(param, loaded_weight, shard_id: int)` |
| `Int8MergedColumnParallelLinear` | `weight_scales_loader` | `(param, loaded_weight, shard_id: int)` |
| `FP8MergedColumnParallelLinear` | `qweight_loader` | `(param, loaded_weight, shard_id: int)` |
| `FP8MergedColumnParallelLinear` | `weight_scale_loader` | `(param, loaded_weight, shard_id: int)` |
| `FP8MergedColumnParallelLinear` | `input_scale_loader` | `(param, loaded_weight, shard_id: int)` |

### Row Parallel Linear（无 shard_id）

| 类 | 绑定方式 | 签名 |
|----|----------|------|
| `AWQRowParallelLinear` | 方法 `qweight_loader` 等 | `(param, loaded_weight)` |
| `Int8RowParallelLinear` | `lambda p, w, *args: ...` | 捕获 `*args`，忽略 shard_id |
| `FP8RowParallelLinear` | `lambda p, w, *args: ...` | 捕获 `*args`，忽略 shard_id |

---

## 4. 发现的 Bug / 差异

### BUG-1：`Int8LinearBase` 不存 `self.tp_group`（严重）

`Int8LinearBase.__init__` 只存 `self.tp_size`，没有存 `self.tp_group`。  
但 `Int8RowParallelLinear.forward` 里调用了：
```python
dist.all_reduce(out, group=self.tp_group)  # AttributeError!
```
当 `tp_size > 1` 时会立即 crash。修复方法：在 `Int8LinearBase.__init__` 中加 `self.tp_group = tp_group`。

### BUG-2（已修）：FP8 / Int8 QKV 的 `tp_size` 使用全局 group

`FP8QKVParallelLinear.__init__` 和 `Int8QKVParallelLinear.__init__` 里，
`tp_size` 原先用 `dist.get_world_size()`（全局），而不是 `dist.get_world_size(tp_group)`。
已在 2026-04-21 修复。

### DIFF-1：scale buffer 名称与维度不统一

| 实现 | buffer 名 | dtype | 维度 |
|------|-----------|-------|------|
| AWQ | `scales` | `float16` | `[K//gs, N]`（二维，含 group 轴） |
| INT8 | `weight_scales` | `float16` | `[N]`（一维，per-channel） |
| FP8 | `weight_scale` | `float32` | `[N]`（一维，per-channel） |

### DIFF-2：量化权重 buffer 名称不统一

- AWQ: `qweight`（int32 packed）
- INT8: `qweight_kn`（int8，后缀 `_kn` 暗示 K×N 布局）
- FP8: `qweight`（uint8，与 AWQ 同名但语义不同）

### DIFF-3：`Int8LinearBase` 没有继承 `LinearBase`

AWQ 通过 `LinearBase` 获得了 `tp_rank`、`tp_size`、`tp_group` 的统一计算和存储。
INT8 和 FP8 均直接继承 `nn.Module`，各自重复实现了 `tp_size` 计算，逻辑有微差异。

### DIFF-4：AWQ 有 group 维度，INT8/FP8 无 group 概念

AWQ INT4 的 scales/zeros 是 `[K//group_size, N]` 二维（per-group scales）。  
INT8 和 FP8 只有 per-channel scale `[N]`，没有 group 概念。  
`LinearKernel` 接口设计时需要将 `scales` 语义抽象为可选 `group_size`。

---

## 5. LinearKernel 接口设计建议

基于以上审计，最小公分母接口应：
- 接收 `x: Tensor`（激活）
- 接收 `weight_data: dict[str, Tensor]`（由各 kernel 自行解析所需 buffer）
- 不假设 buffer 命名，由各实现类负责 pack/unpack
- `tp_size > 1` 的 all_reduce 不归 kernel 负责，由上层 linear 层处理

详见 `layers/linear_kernel.py`。

---

## 6. Stage 4 迁移优先级

1. 先修 `Int8LinearBase` 不存 `tp_group` 的 bug（BUG-1）
2. 统一 scale buffer 命名（`weight_scale: Tensor [N] float32`，AWQ 额外允许 `[groups, N]`）
3. 再定义 `LinearKernel` 接口并迁移 AWQ → `AWQKernel`
4. 最后迁移 FP8 / INT8
