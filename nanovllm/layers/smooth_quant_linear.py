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

# ==============================================================================
# 1. 工业级 INT8 W8A8 融合内核 (加入 Autotune 防止 SM 饥饿)
# ==============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _w8a8_linear_kernel(
    a_ptr, b_ptr, c_ptr, x_scales_ptr, w_scales_ptr, bias_ptr,
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

    w_scales = tl.load(w_scales_ptr + offs_bn)

    # 累加器必须是 INT32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = (k * BLOCK_SIZE_K + offs_k) < K
        a_int8 = tl.load(a_ptrs, mask=k_mask[None, :], other=0)
        b_int8 = tl.load(b_ptrs, mask=k_mask[:, None], other=0)
        
        # 调用硬件底层的 DP4A 或 IMMA 整数乘加指令
        accumulator += tl.dot(a_int8, b_int8, out_dtype=tl.int32)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    x_scales = tl.load(x_scales_ptr + offs_am)[:, None]
    
    # 反量化：INT32 -> FP32 * scale * scale
    c = accumulator.to(tl.float32) * x_scales * w_scales[None, :]
    
    if bias_ptr is not None:
        c += tl.load(bias_ptr + offs_bn)[None, :]

    # 存储回全局内存
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :], c.to(tl.bfloat16), 
             mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

# ==============================================================================
# 2. 基础 INT8 量化层
# ==============================================================================
class Int8LinearBase(nn.Module):
    def __init__(self, input_size, output_size, tp_dim=None, tp_group=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = dist.get_world_size(tp_group) if tp_group else (dist.get_world_size() if dist.is_initialized() else 1)

    def _init_quant_buffers(self, in_features, out_features):
        self.register_buffer("qweight_kn", torch.zeros((in_features, out_features), dtype=torch.int8))
        self.register_buffer("weight_scales", torch.zeros((out_features,), dtype=torch.float16))

    def forward_w8a8(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        x_2d = x.view(-1, x.shape[-1])
        
        # 极速动态激活量化
        x_max = torch.amax(x_2d.abs(), dim=-1, keepdim=True).to(torch.float32) + 1e-5
        x_scales = x_max / 127.0
        x_int8 = torch.clamp(torch.round(x_2d / x_scales), -128, 127).to(torch.int8)
        
        M, K = x_int8.shape
        N = self.qweight_kn.shape[1]
        output = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
        
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        _w8a8_linear_kernel[grid](
            x_int8, self.qweight_kn, output, x_scales.to(torch.float32), self.weight_scales.to(torch.float32), getattr(self, "bias", None),
            M, N, K,
            x_int8.stride(0), x_int8.stride(1),
            self.qweight_kn.stride(0), self.qweight_kn.stride(1),
            output.stride(0), output.stride(1),
            GROUP_SIZE_M=8
        )
        return output.view(*x.shape[:-1], -1).to(target_dtype)

# ==============================================================================
# 3. 算子并行类
# ==============================================================================
class Int8MergedColumnParallelLinear(Int8LinearBase):
    def __init__(self, input_size, output_sizes, bias=False, tp_group=None, **kwargs):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), tp_dim=1, tp_group=tp_group, **kwargs)
        self._init_quant_buffers(input_size, sum(output_sizes) // self.tp_size)
        self.bias = nn.Parameter(torch.empty(sum(output_sizes) // self.tp_size)) if bias else None
        self.qweight_kn.weight_loader = self.qweight_loader
        self.weight_scales.weight_loader = self.weight_scales_loader

    def qweight_loader(self, param, loaded_weight, shard_id):
        offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        size = self.output_sizes[shard_id] // self.tp_size
        param.data.narrow(1, offset, size).copy_(loaded_weight.t().contiguous())

    def weight_scales_loader(self, param, loaded_weight, shard_id):
        offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        size = self.output_sizes[shard_id] // self.tp_size
        param.data.narrow(0, offset, size).copy_(loaded_weight.view(-1))

    def forward(self, x): return self.forward_w8a8(x)

class Int8QKVParallelLinear(Int8LinearBase):
    def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads, bias=False, tp_group=None, **kwargs):
        self.head_size = head_size
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = total_num_kv_heads // tp_size
        out_f = (self.num_heads + 2 * self.num_kv_heads) * head_size
        super().__init__(hidden_size, out_f, tp_dim=1, tp_group=tp_group, **kwargs)
        self._init_quant_buffers(hidden_size, out_f)
        self.qweight_kn.weight_loader = self.qweight_loader
        self.weight_scales.weight_loader = self.weight_scales_loader

    def qweight_loader(self, param, loaded_weight, shard_id):
        if shard_id == "q": size, offset = self.num_heads * self.head_size, 0
        elif shard_id == "k": size, offset = self.num_kv_heads * self.head_size, self.num_heads * self.head_size
        else: size, offset = self.num_kv_heads * self.head_size, (self.num_heads + self.num_kv_heads) * self.head_size
        param.data.narrow(1, offset, size).copy_(loaded_weight.t().contiguous())

    def weight_scales_loader(self, param, loaded_weight, shard_id):
        if shard_id == "q": size, offset = self.num_heads * self.head_size, 0
        elif shard_id == "k": size, offset = self.num_kv_heads * self.head_size, self.num_heads * self.head_size
        else: size, offset = self.num_kv_heads * self.head_size, (self.num_heads + self.num_kv_heads) * self.head_size
        param.data.narrow(0, offset, size).copy_(loaded_weight.view(-1))

    def forward(self, x): return self.forward_w8a8(x)

class Int8RowParallelLinear(Int8LinearBase):
    def __init__(self, input_size, output_size, bias=False, reduce_results=True, tp_group=None, **kwargs):
        super().__init__(input_size, output_size, tp_dim=0, tp_group=tp_group, **kwargs)
        self.reduce_results = reduce_results
        self._init_quant_buffers(divide(input_size, self.tp_size), output_size)
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        self.qweight_kn.weight_loader = lambda p, w, *args: p.data.copy_(w.t().contiguous())
        self.weight_scales.weight_loader = lambda p, w, *args: p.data.copy_(w.view(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward_w8a8(x)
        if self.tp_size > 1 and self.reduce_results:
            dist.all_reduce(out, group=self.tp_group)
        return out + self.bias if self.bias is not None else out