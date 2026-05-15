# FP8 量化优化路线图

---

## 一、概念澄清

### 1.1 W8A16 vs W8A8

| 方案 | 权重精度 | 激活精度 | Tensor Core | 主要价值 |
|------|----------|----------|-------------|----------|
| **W8A16** | FP8 | BF16/FP16 | FP16 Tensor Core | 节省 50% 权重显存 |
| **W8A8** | FP8 | FP8 | FP8 Tensor Core | 节省显存 + 计算加速 |

**关键理解**：
- W8A16 本身就是用 FP16/BF16 做矩阵运算，这是正确的设计
- W8A8 才是真正利用 FP8 Tensor Core 的方案
- 两者的优化方向不同，不应混淆

### 1.2 当前状态

| 模块 | 状态 | 问题 |
|------|------|------|
| W8A16 Triton | ✅ 可用 | FP8→FP16 转换效率可优化 |
| W8A16 PyTorch | ✅ 可用 | 每次 forward 都转换权重 |
| W8A8 非融合 | ✅ 可用 | 3 次 kernel launch |
| W8A8 融合 | ❌ SM89 崩溃 | Triton 编译器问题 |
| KV K-int8/V-FP8 | ✅ 生产可用 | 51.95% 显存 |

---

## 二、W8A16 优化路线

### 模块目标
**目标**: 节省权重显存，优化 FP8→FP16 转换效率

### Level 0: PyTorch 原生 (Baseline)

**实现**:
```python
w_bf16 = w_fp8.to(torch.bfloat16) * weight_scale
out = torch.mm(x, w_bf16)
```

**特点**:
- 最简单，最容易理解
- 每次 forward 都做类型转换
- 性能参考基准

**文件**: `nanovllm/quantization/fp8.py:197-202`

---

### Level 1: Triton Kernel (当前实现)

**实现**:
```python
# 在 kernel 内逐块转换
for k_block in range(K_blocks):
    b_fp8 = load(weight_block)
    b_fp16 = b_fp8.to(fp32).to(fp16)  # 转换
    acc += dot(a_bf16, b_fp16)         # FP16 Tensor Core
```

**特点**:
- 单 kernel，减少 launch 开销
- 逐块转换，内存局部性好
- 使用 FP16 Tensor Core

**文件**: `nanovllm/quantization/kernels/fp8.py:w8a16_gemm_kernel`

**优化点**:
- [ ] FP8→FP16 是否需要经过 FP32？能否直接转换？
- [ ] Block size 调优 (prefill vs decode)
- [ ] 权重预转置优化内存访问

---

### Level 2: 优化 Triton Kernel

**改进方向**:

1. **直接转换路径**
   ```python
   # 当前: FP8 → FP32 → FP16
   b_fp16 = b_fp8.to(tl.float32).to(tl.float16)
   
   # 尝试: FP8 → FP16 (如果 Triton 支持)
   b_fp16 = b_fp8.to(tl.float16)
   ```

2. **权重预处理**
   - 加载时预处理为 FP16 格式
   - 或预处理为更利于转换的布局

3. **分离 prefill/decode 配置**
   - Prefill: 大 M，优化吞吐
   - Decode: M=1，优化延迟

**预期提升**: 1.2-1.5x

---

### Level 3: CUDA/cuBLASLt Kernel

**实现**:
```cpp
// 使用 cuBLASLt 融合操作
cublasLtMatmul(
    ...,
    CUBLASLT_EPILOGUE_BIAS,  // 融合 bias
    ...
);
```

**特点**:
- 融合 bias add
- 更好的内存布局控制
- 可能利用硬件加速的类型转换

**新增文件**:
```
nanovllm/quantization/kernels/
└── fp8_cuda/
    ├── __init__.py
    ├── w8a16_gemm.cu
    └── bindings.cpp
```

---

### Level 4: CUTLASS/PTX 深度优化

**实现**:
- CUTLASS GEMM 模板
- 自定义 PTX 内联汇编
- 完全掌控内存访问和计算流水线

---

## 三、W8A8 优化路线

### 模块目标
**目标**: 利用 FP8 Tensor Core 实现计算加速

### Level 0: PyTorch 分离实现 (Baseline)

**流程**:
```
BF16 激活 → FP8 量化 → FP8 GEMM → 加 bias → BF16 输出
   (kernel1)     (kernel2)   (kernel3)
```

**问题**: 3 次 kernel launch，小 batch 时开销大

---

### Level 1: torch._scaled_mm (当前实现)

**流程**:
```python
x_fp8 = quantize_activation(x, scale)  # kernel 1
out = torch._scaled_mm(x_fp8, w_fp8)   # kernel 2 (FP8 Tensor Core)
out = out + bias                        # kernel 3
```

**特点**:
- FP8 Tensor Core GEMM (真正的加速)
- 但量化开销在小 shape 时占比 15-30%

**文件**: `nanovllm/quantization/kernels/fp8.py:launch_scaled_mm_w8a8`

---

### Level 2: 修复融合 Triton Kernel

**问题**: 当前 `w8a8_fused_gemm_kernel` 在 SM89 上崩溃

**原因分析**:
```python
# 可能的问题代码
a = a.to(tl.float8e4nv)  # SM89 上 Triton FP8 支持不完整
acc = tl.dot(a, b)        # FP8 × FP8 dot 可能有问题
```

**修复方案**:
1. 使用更保守的类型转换路径
2. 或回退到 FP16 中间计算

**文件**: `nanovllm/quantization/kernels/fp8.py:w8a8_fused_gemm_kernel`

---

### Level 3: CUDA 融合 Kernel

**目标**: 单 kernel 完成 量化 + GEMM + bias

**实现**:
```cpp
// 伪代码
__global__ void w8a8_fused_kernel(...) {
    // 1. 加载 BF16 激活
    // 2. 在线量化为 FP8
    // 3. FP8 Tensor Core GEMM
    // 4. 加 bias
    // 5. 输出 BF16
}
```

**新增文件**:
```
nanovllm/quantization/kernels/
└── fp8_w8a8_cuda/
    ├── __init__.py
    ├── w8a8_fused_gemm.cu
    └── bindings.cpp
```

**参考**: vLLM `csrc/quantization/w8a8/fp8/common.cu`

---

### Level 4: Shape-Aware 后端选择

**目标**: 根据矩阵形状自动选择最优后端

```python
def select_backend(M, K, N):
    if M == 1:                    # Decode
        return "w8a8_decode_optimized"
    if K >= 2048 and N >= 11008:  # 大 MLP
        return "w8a8_fused"
    return "w8a16"                # 小 shape 避免 quant 开销
```

---

### Level 5: Block-wise 量化

**目标**: 提升精度

**改进**:
- 权重: 128x128 block scale
- 激活: per-token-group scale

---

## 四、KV Cache 优化路线

### 当前状态

| 方案 | 显存 | 精度 | 状态 |
|------|------|------|------|
| BF16 | 100% | 参考 | 参考 |
| V-FP8 | 75% | 好 | 可用 |
| K-int8/V-FP8 | 52% | 精确 | ✅ 生产可用 |

### 优化方向

1. **Native Attention 优化**: Split-K, Batch decode
2. **Full FP8 KV**: 研究 K 的 FP8 量化方案

---

## 五、模块化拆分

```
nano-vllm-moe/
├── nanovllm/
│   └── quantization/
│       ├── fp8.py                    # 量化方法注册
│       └── kernels/
│           ├── fp8.py                # Triton kernels (W8A16, W8A8)
│           ├── fp8_cuda/             # CUDA kernels (新增)
│           │   ├── __init__.py
│           │   ├── w8a16_gemm.cu     # W8A16 CUDA
│           │   ├── w8a8_fused.cu     # W8A8 融合
│           │   └── bindings.cpp
│           └── fp8_ptx/              # PTX 优化 (未来)
│
├── tests/
│   └── quantization/
│       ├── test_w8a16_kernels.py
│       └── test_w8a8_kernels.py
│
└── scripts/
    └── quantization/
        ├── bench_w8a16.py
        └── bench_w8a8.py
```

---

## 六、实施阶段

### Phase 1: 基础优化 (Week 1-2)

| 任务 | 模块 | 优先级 |
|------|------|--------|
| 验证 FP8→FP16 直接转换 | W8A16 | 高 |
| 修复 W8A8 融合 kernel SM89 崩溃 | W8A8 | 高 |
| 建立 benchmark 框架 | 基础设施 | 中 |

### Phase 2: CUDA 开发 (Week 3-6)

| 任务 | 模块 | 优先级 |
|------|------|--------|
| 搭建 CUDA 编译环境 | 基础设施 | 高 |
| W8A16 cuBLASLt kernel | W8A16 | 中 |
| W8A8 融合 CUDA kernel | W8A8 | 高 |
| Shape-aware 后端选择 | W8A8 | 中 |

### Phase 3: 深度优化 (Week 7-12)

| 任务 | 模块 | 优先级 |
|------|------|--------|
| CUTLASS 集成 | W8A16/W8A8 | 中 |
| PTX 内联汇编学习 | W8A16/W8A8 | 低 |
| Block-wise 量化 | W8A8 | 低 |

---

## 七、关键文件

| 文件 | 用途 | 修改频率 |
|------|------|----------|
| `nanovllm/quantization/kernels/fp8.py` | Triton kernels | 高 |
| `nanovllm/quantization/fp8.py` | 方法注册, backend 选择 | 中 |
| `nanovllm/layers/fp8_paged_attention.py` | FP8 KV attention | 低 |
| `nanovllm/layers/kv_cache_kernels.py` | KV cache 操作 | 低 |

---

## 八、性能目标 (7B on RTX 4090)

### W8A16 目标

| 指标 | 当前 | Level 2 目标 | Level 3 目标 |
|------|------|--------------|--------------|
| Prefill TPS | 650 | 900 | 1200 |
| Decode TPS | 71 | 90 | 110 |

### W8A8 目标

| 指标 | 当前 | Level 3 目标 | Level 4 目标 |
|------|------|--------------|--------------|
| Prefill TPS | 8775 | 12000 | 14000 |
| Decode TPS | 61 | 100 | 120 |

---

## 九、验证策略

### 精度验证
- PPL delta < 0.1 (WikiText-2)
- Token match > 99%

### 性能验证
```bash
# W8A16 benchmark
python scripts/quantization/fp8_linear_microbench.py \
    --w8a8-model-path <model> \
    --weight-name model.layers.0.mlp.gate_proj

# W8A8 benchmark  
python scripts/quantization/fp8_linear_microbench.py \
    --w8a8-model-path <model> \
    --weight-name model.layers.0.mlp.gate_proj \
    --act-quant-backend torch triton
```

---

## 十、学习资源

| 资源 | 路径 | 用途 |
|------|------|------|
| vLLM FP8 CUDA | `../vllm/csrc/quantization/w8a8/fp8/` | CUDA 实现参考 |
| ktransformers CUDA | `../ktransformers/kt-kernel/cuda/` | MoE kernel 参考 |
| CUTLASS | GitHub | GEMM 模板 |
| NVIDIA PTX ISA | 官方文档 | 底层优化 |
