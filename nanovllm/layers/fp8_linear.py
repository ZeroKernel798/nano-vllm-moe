# import torch
# import torch.nn.functional as F
# from torch import nn
# import torch.distributed as dist
# import triton
# import triton.language as tl
# from typing import Optional

# def divide(numerator, denominator):
#     assert numerator % denominator == 0
#     return numerator // denominator

# # ==============================================================================
# # 1. 动态量化 FP8 Triton Kernel
# # ==============================================================================
# @triton.jit
# def _fp8_linear_kernel(
#     a_ptr, b_ptr, c_ptr, x_scale_ptr, w_scale_ptr, bias_ptr,
#     M, N, K,
#     stride_am, stride_ak,
#     stride_bk, stride_bn,
#     stride_cm, stride_cn,
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     pid = tl.program_id(0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + (pid % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m

#     offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
    
#     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

#     w_scale = tl.load(w_scale_ptr + offs_bn)
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         k_mask = (k * BLOCK_SIZE_K + offs_k) < K
#         a_fp8 = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
#         b_fp8 = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
#         accumulator += tl.dot(a_fp8, b_fp8, out_dtype=tl.float32)
#         a_ptrs += BLOCK_SIZE_K * stride_ak
#         b_ptrs += BLOCK_SIZE_K * stride_bk

#     # 动态量化：按行（M的维度）加载每个 Token 独立算出来的 Scale
#     x_scales = tl.load(x_scale_ptr + offs_am)[:, None]
    
#     c = accumulator * x_scales * w_scale[None, :]
    
#     if bias_ptr is not None:
#         c += tl.load(bias_ptr + offs_bn)[None, :]

#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     tl.store(c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :], c.to(tl.bfloat16), 
#              mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


# # ==============================================================================
# # 2. FP8 基础动态量化层
# # ==============================================================================
# class FP8LinearBase(nn.Module):
#     def __init__(self, input_size, output_size, tp_dim=None, tp_group=None, **kwargs):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.tp_size = dist.get_world_size(tp_group) if tp_group else (dist.get_world_size() if dist.is_initialized() else 1)

#     def _init_quant_buffers(self, in_features, out_features):
#         self.register_buffer("qweight", torch.zeros((in_features, out_features), dtype=torch.uint8))
#         self.register_buffer("weight_scale", torch.zeros((out_features,), dtype=torch.float32))

#     def forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
#         target_dtype = x.dtype
#         x_2d = x.view(-1, x.shape[-1])
#         M, K = x_2d.shape
#         N = self.qweight.shape[1]
        
#         # --- 核心优化点：极速动态量化 ---
#         # 使用 amax 替代 abs().max()，性能在底层 C++ 层面有针对性优化
#         x_max = torch.amax(x_2d.abs(), dim=-1, keepdim=True).to(torch.float32) + 1e-5
#         x_scales = x_max / 448.0 
        
#         x_fp8 = (x_2d / x_scales).to(torch.float8_e4m3fn)
#         w_fp8 = self.qweight.view(torch.float8_e4m3fn)
        
#         output = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
        
#         grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        
#         _fp8_linear_kernel[grid](
#             x_fp8, w_fp8, output, 
#             x_scales,           # 传入形状为 (M, 1) 的动态 Scale 张量
#             self.weight_scale, 
#             getattr(self, "bias", None),
#             M, N, K,
#             x_fp8.stride(0), x_fp8.stride(1),
#             w_fp8.stride(0), w_fp8.stride(1),
#             output.stride(0), output.stride(1),
#             BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=8
#         )
#         return output.view(*x.shape[:-1], -1).to(target_dtype)

# # ==============================================================================
# # 3. 并行类实现
# # ==============================================================================
# class FP8MergedColumnParallelLinear(FP8LinearBase):
#     def __init__(self, input_size, output_sizes, bias=False, tp_group=None, **kwargs):
#         self.output_sizes = output_sizes
#         super().__init__(input_size, sum(output_sizes), tp_dim=1, tp_group=tp_group, **kwargs)
#         self._init_quant_buffers(input_size, sum(output_sizes) // self.tp_size)
#         self.bias = nn.Parameter(torch.empty(sum(output_sizes) // self.tp_size)) if bias else None
        
#         self.qweight.weight_loader = self.qweight_loader
#         self.weight_scale.weight_loader = self.weight_scale_loader

#     def qweight_loader(self, param, loaded_weight, shard_id):
#         offset = sum(self.output_sizes[:shard_id]) // self.tp_size
#         size = self.output_sizes[shard_id] // self.tp_size
#         param.data.narrow(1, offset, size).copy_(loaded_weight.t().contiguous())

#     def weight_scale_loader(self, param, loaded_weight, shard_id):
#         offset = sum(self.output_sizes[:shard_id]) // self.tp_size
#         size = self.output_sizes[shard_id] // self.tp_size
#         param.data.narrow(0, offset, size).fill_(loaded_weight.item())

#     def forward(self, x): return self.forward_fp8(x)

# class FP8QKVParallelLinear(FP8LinearBase):
#     def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads, bias=False, tp_group=None, **kwargs):
#         self.head_size = head_size
#         tp_size = dist.get_world_size() if dist.is_initialized() else 1
#         self.num_heads = total_num_heads // tp_size
#         self.num_kv_heads = total_num_kv_heads // tp_size
#         out_f = (self.num_heads + 2 * self.num_kv_heads) * head_size
#         super().__init__(hidden_size, out_f, tp_dim=1, tp_group=tp_group, **kwargs)
#         self._init_quant_buffers(hidden_size, out_f)
        
#         self.qweight.weight_loader = self.qweight_loader
#         self.weight_scale.weight_loader = self.weight_scale_loader

#     def qweight_loader(self, param, loaded_weight, shard_id):
#         if shard_id == "q": size, offset = self.num_heads * self.head_size, 0
#         elif shard_id == "k": size, offset = self.num_kv_heads * self.head_size, self.num_heads * self.head_size
#         else: size, offset = self.num_kv_heads * self.head_size, (self.num_heads + self.num_kv_heads) * self.head_size
#         param.data.narrow(1, offset, size).copy_(loaded_weight.t().contiguous())

#     def weight_scale_loader(self, param, loaded_weight, shard_id):
#         if shard_id == "q": size, offset = self.num_heads * self.head_size, 0
#         elif shard_id == "k": size, offset = self.num_kv_heads * self.head_size, self.num_heads * self.head_size
#         else: size, offset = self.num_kv_heads * self.head_size, (self.num_heads + self.num_kv_heads) * self.head_size
#         param.data.narrow(0, offset, size).fill_(loaded_weight.item())

#     def forward(self, x): return self.forward_fp8(x)

# class FP8RowParallelLinear(FP8LinearBase):
#     def __init__(self, input_size, output_size, bias=False, reduce_results=True, tp_group=None, **kwargs):
#         super().__init__(input_size, output_size, tp_dim=0, tp_group=tp_group, **kwargs)
#         self.reduce_results = reduce_results
#         self._init_quant_buffers(divide(input_size, self.tp_size), output_size)
#         self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        
#         self.qweight.weight_loader = lambda p, w, *args: p.data.copy_(w.t().contiguous())
#         self.weight_scale.weight_loader = lambda p, w, *args: p.data.fill_(w.item())

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.forward_fp8(x)
#         if self.tp_size > 1 and self.reduce_results:
#             dist.all_reduce(out, group=self.tp_group)
#         return out + self.bias if self.bias is not None else out
# 上面是可用的动态量化方案 w8a8

# 下面是w8a16
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import triton
import triton.language as tl
from typing import Optional

def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

@triton.jit
def _w8a16_linear_kernel(
    a_ptr, b_ptr, c_ptr, w_scale_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    w_scale = tl.load(w_scale_ptr + offs_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = (k * BLOCK_SIZE_K + offs_k) < K
        
        a_bf16 = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        
        b_fp8 = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        b_bf16 = b_fp8.to(tl.bfloat16)
        
        accumulator += tl.dot(a_bf16, b_bf16, out_dtype=tl.float32)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator * w_scale[None, :]
    
    if bias_ptr is not None:
        c += tl.load(bias_ptr + offs_bn)[None, :]

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :], c.to(tl.bfloat16), 
             mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

class FP8LinearBase(nn.Module):
    def __init__(self, input_size, output_size, tp_dim=None, tp_group=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = dist.get_world_size(tp_group) if tp_group else (dist.get_world_size() if dist.is_initialized() else 1)

    def _init_quant_buffers(self, in_features, out_features):
        self.register_buffer("qweight", torch.zeros((in_features, out_features), dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.zeros((out_features,), dtype=torch.float32))

    def forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        x_2d = x.view(-1, x.shape[-1])
        M, K = x_2d.shape
        N = self.qweight.shape[1]
        
        w_fp8 = self.qweight.view(torch.float8_e4m3fn)
        output = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
        
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        
        _w8a16_linear_kernel[grid](
            x_2d, w_fp8, output, 
            self.weight_scale, 
            getattr(self, "bias", None),
            M, N, K,
            x_2d.stride(0), x_2d.stride(1),
            w_fp8.stride(0), w_fp8.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=8
        )
        return output.view(*x.shape[:-1], -1).to(target_dtype)

class FP8MergedColumnParallelLinear(FP8LinearBase):
    def __init__(self, input_size, output_sizes, bias=False, tp_group=None, **kwargs):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), tp_dim=1, tp_group=tp_group, **kwargs)
        self._init_quant_buffers(input_size, sum(output_sizes) // self.tp_size)
        self.bias = nn.Parameter(torch.empty(sum(output_sizes) // self.tp_size)) if bias else None
        
        self.qweight.weight_loader = self.qweight_loader
        self.weight_scale.weight_loader = self.weight_scale_loader

    def qweight_loader(self, param, loaded_weight, shard_id):
        offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        size = self.output_sizes[shard_id] // self.tp_size
        param.data.narrow(1, offset, size).copy_(loaded_weight.t().contiguous())

    def weight_scale_loader(self, param, loaded_weight, shard_id):
        offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        size = self.output_sizes[shard_id] // self.tp_size
        param.data.narrow(0, offset, size).fill_(loaded_weight.item())

    def forward(self, x): return self.forward_fp8(x)

class FP8QKVParallelLinear(FP8LinearBase):
    def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads, bias=False, tp_group=None, **kwargs):
        self.head_size = head_size
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = total_num_kv_heads // tp_size
        out_f = (self.num_heads + 2 * self.num_kv_heads) * head_size
        super().__init__(hidden_size, out_f, tp_dim=1, tp_group=tp_group, **kwargs)
        self._init_quant_buffers(hidden_size, out_f)
        
        self.qweight.weight_loader = self.qweight_loader
        self.weight_scale.weight_loader = self.weight_scale_loader

    def qweight_loader(self, param, loaded_weight, shard_id):
        if shard_id == "q": size, offset = self.num_heads * self.head_size, 0
        elif shard_id == "k": size, offset = self.num_kv_heads * self.head_size, self.num_heads * self.head_size
        else: size, offset = self.num_kv_heads * self.head_size, (self.num_heads + self.num_kv_heads) * self.head_size
        param.data.narrow(1, offset, size).copy_(loaded_weight.t().contiguous())

    def weight_scale_loader(self, param, loaded_weight, shard_id):
        if shard_id == "q": size, offset = self.num_heads * self.head_size, 0
        elif shard_id == "k": size, offset = self.num_kv_heads * self.head_size, self.num_heads * self.head_size
        else: size, offset = self.num_kv_heads * self.head_size, (self.num_heads + self.num_kv_heads) * self.head_size
        param.data.narrow(0, offset, size).fill_(loaded_weight.item())

    def forward(self, x): return self.forward_fp8(x)

class FP8RowParallelLinear(FP8LinearBase):
    def __init__(self, input_size, output_size, bias=False, reduce_results=True, tp_group=None, **kwargs):
        super().__init__(input_size, output_size, tp_dim=0, tp_group=tp_group, **kwargs)
        self.reduce_results = reduce_results
        self._init_quant_buffers(divide(input_size, self.tp_size), output_size)
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        
        self.qweight.weight_loader = lambda p, w, *args: p.data.copy_(w.t().contiguous())
        self.weight_scale.weight_loader = lambda p, w, *args: p.data.fill_(w.item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward_fp8(x)
        if self.tp_size > 1 and self.reduce_results:
            dist.all_reduce(out, group=self.tp_group)
        return out + self.bias if self.bias is not None else out

# # 静态量化a8w8
# import torch
# import torch.nn.functional as F
# from torch import nn
# import torch.distributed as dist
# import triton
# import triton.language as tl
# from typing import Optional

# def divide(numerator, denominator):
#     assert numerator % denominator == 0
#     return numerator // denominator

# # ---------------------------------------------------------
# # 1. 核心 Kernel：静态 W8A8 引擎
# # ---------------------------------------------------------
# @triton.jit
# def _w8a8_static_linear_kernel(
#     a_ptr, b_ptr, c_ptr, 
#     a_scale_ptr, w_scale_ptr, bias_ptr,
#     M, N, K,
#     stride_am, stride_ak,
#     stride_bk, stride_bn,
#     stride_cm, stride_cn,
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     pid = tl.program_id(0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + (pid % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m

#     offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
    
#     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

#     # 加载 Scale (ModelOpt 导出通常是 float32)
#     a_scale = tl.load(a_scale_ptr)
#     w_scale = tl.load(w_scale_ptr + offs_bn)
    
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         k_mask = (k * BLOCK_SIZE_K + offs_k) < K
#         a_fp8 = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
#         b_fp8 = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        
#         accumulator += tl.dot(a_fp8, b_fp8, out_dtype=tl.float32)
        
#         a_ptrs += BLOCK_SIZE_K * stride_ak
#         b_ptrs += BLOCK_SIZE_K * stride_bk

#     # 反量化: 根据 ModelOpt 的逻辑进行缩放
#     c = accumulator * (a_scale * w_scale[None, :])
    
#     if bias_ptr is not None:
#         c += tl.load(bias_ptr + offs_bn)[None, :]

#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     tl.store(c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :], c.to(tl.bfloat16), 
#              mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

# # ---------------------------------------------------------
# # 2. 基础类
# # ---------------------------------------------------------
# class FP8LinearBase(nn.Module):
#     def __init__(self, input_size, output_size, tp_dim=None, tp_group=None, **kwargs):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.tp_size = dist.get_world_size(tp_group) if tp_group else (dist.get_world_size() if dist.is_initialized() else 1)

#     def _init_quant_buffers(self, in_features, out_features):
#         self.register_buffer("qweight", torch.zeros((in_features, out_features), dtype=torch.uint8))
#         self.register_buffer("weight_scale", torch.zeros((out_features,), dtype=torch.float32))
#         self.register_buffer("input_scale", torch.tensor([1.0], dtype=torch.float32))

#     def forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
#         target_dtype = x.dtype
#         x_2d = x.view(-1, x.shape[-1])
#         M, K = x_2d.shape
#         N = self.qweight.shape[1]
        
#         # 【关键改动】ModelOpt 导出的 input_scale 通常是除数 (max/448)
#         # 如果还是乱码，尝试将 / 改为 *
#         x_fp8 = (x_2d / self.input_scale).to(torch.float8_e4m3fn)
#         w_fp8 = self.qweight.view(torch.float8_e4m3fn)
        
#         output = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
#         grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        
#         _w8a8_static_linear_kernel[grid](
#             x_fp8, w_fp8, output, 
#             self.input_scale, self.weight_scale, 
#             getattr(self, "bias", None),
#             M, N, K,
#             x_fp8.stride(0), x_fp8.stride(1),
#             w_fp8.stride(0), w_fp8.stride(1),
#             output.stride(0), output.stride(1),
#             BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=8
#         )
#         return output.view(*x.shape[:-1], -1).to(target_dtype)

# # ---------------------------------------------------------
# # 3. 并行层逻辑 (修复了 weight_scale 加载)
# # ---------------------------------------------------------
# class FP8QKVParallelLinear(FP8LinearBase):
#     def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads, bias=False, tp_group=None, **kwargs):
#         self.head_size = head_size
#         tp_size = dist.get_world_size() if dist.is_initialized() else 1
#         self.num_heads = total_num_heads // tp_size
#         self.num_kv_heads = total_num_kv_heads // tp_size
#         out_f = (self.num_heads + 2 * self.num_kv_heads) * head_size
#         super().__init__(hidden_size, out_f, tp_dim=1, tp_group=tp_group, **kwargs)
#         self._init_quant_buffers(hidden_size, out_f)
        
#         self.qweight.weight_loader = self.qweight_loader
#         self.weight_scale.weight_loader = self.weight_scale_loader
#         self.input_scale.weight_loader = self.generic_loader

#     def generic_loader(self, param, loaded_weight, shard_id=None):
#         param.data.copy_(loaded_weight)

#     def qweight_loader(self, param, loaded_weight, shard_id):
#         if shard_id == "q": size, offset = self.num_heads * self.head_size, 0
#         elif shard_id == "k": size, offset = self.num_kv_heads * self.head_size, self.num_heads * self.head_size
#         else: size, offset = self.num_kv_heads * self.head_size, (self.num_heads + self.num_kv_heads) * self.head_size
#         param.data.narrow(1, offset, size).copy_(loaded_weight.t().contiguous())

#     def weight_scale_loader(self, param, loaded_weight, shard_id):
#         # 【FIX】切分 Per-channel Scale
#         if shard_id == "q": size, offset = self.num_heads * self.head_size, 0
#         elif shard_id == "k": size, offset = self.num_kv_heads * self.head_size, self.num_heads * self.head_size
#         else: size, offset = self.num_kv_heads * self.head_size, (self.num_heads + self.num_kv_heads) * self.head_size
        
#         w_s = loaded_weight.view(-1)
#         param.data.narrow(0, offset, size).copy_(w_s)

#     def forward(self, x): return self.forward_fp8(x)

# class FP8MergedColumnParallelLinear(FP8LinearBase):
#     def __init__(self, input_size, output_sizes, bias=False, tp_group=None, **kwargs):
#         self.output_sizes = output_sizes
#         super().__init__(input_size, sum(output_sizes), tp_dim=1, tp_group=tp_group, **kwargs)
#         self._init_quant_buffers(input_size, sum(output_sizes) // self.tp_size)
#         self.bias = nn.Parameter(torch.empty(sum(output_sizes) // self.tp_size)) if bias else None
        
#         self.qweight.weight_loader = self.qweight_loader
#         self.weight_scale.weight_loader = self.weight_scale_loader
#         self.input_scale.weight_loader = self.generic_loader

#     def generic_loader(self, param, loaded_weight, shard_id=None):
#         param.data.copy_(loaded_weight)

#     def qweight_loader(self, param, loaded_weight, shard_id):
#         offset = sum(self.output_sizes[:shard_id]) // self.tp_size
#         size = self.output_sizes[shard_id] // self.tp_size
#         param.data.narrow(1, offset, size).copy_(loaded_weight.t().contiguous())

#     def weight_scale_loader(self, param, loaded_weight, shard_id):
#         offset = sum(self.output_sizes[:shard_id]) // self.tp_size
#         size = self.output_sizes[shard_id] // self.tp_size
#         w_s = loaded_weight.view(-1)
#         param.data.narrow(0, offset, size).copy_(w_s)

#     def forward(self, x): return self.forward_fp8(x)

# class FP8RowParallelLinear(FP8LinearBase):
#     def __init__(self, input_size, output_size, bias=False, reduce_results=True, tp_group=None, **kwargs):
#         super().__init__(input_size, output_size, tp_dim=0, tp_group=tp_group, **kwargs)
#         self.reduce_results = reduce_results
#         self._init_quant_buffers(divide(input_size, self.tp_size), output_size)
#         self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        
#         self.qweight.weight_loader = lambda p, w, *args: p.data.copy_(w.t().contiguous())
#         # RowParallel 的输出维度没切，直接 copy 整个 weight_scale Tensor
#         self.weight_scale.weight_loader = lambda p, w, *args: p.data.copy_(w.view(-1))
#         self.input_scale.weight_loader = lambda p, w, *args: p.data.copy_(w)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.forward_fp8(x)
#         if self.tp_size > 1 and self.reduce_results:
#             dist.all_reduce(out, group=self.tp_group)
#         return out + self.bias if self.bias is not None else out