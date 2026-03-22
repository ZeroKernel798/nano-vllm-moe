# import triton
# import triton.language as tl

# @triton.jit
# def fused_moe_w2_combine_kernel(
#     a_ptr, b_ptr, c_ptr,
#     sorted_token_ids_ptr,
#     expert_ids_ptr,
#     N, K,
#     stride_am, stride_ak,
#     stride_be, stride_bk, stride_bn,
#     stride_cm, stride_cn,
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
# ):
#     pid = tl.program_id(axis=0)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#     pid_m = pid // num_pid_n
#     pid_n = pid % num_pid_n

#     expert_id = tl.load(expert_ids_ptr + pid_m)
#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_token = tl.load(sorted_token_ids_ptr + offs_m)
    
#     # 如果是 Padding (-1)，强制指针指向 0 以防段错误
#     mask_m = offs_token >= 0
#     safe_offs_token = tl.where(mask_m, offs_token, 0)

#     current_b_ptr = b_ptr + expert_id * stride_be
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

#     offs_k = tl.arange(0, BLOCK_SIZE_K)
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     mask_n = offs_n < N

#     # 使用安全的 Token 偏移量计算 A 的指针
#     a_ptrs = a_ptr + (safe_offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     b_ptrs = current_b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         mask_k = (k * BLOCK_SIZE_K + offs_k) < K
#         a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
#         b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
#         accumulator = tl.dot(a, b, acc=accumulator)
#         a_ptrs += BLOCK_SIZE_K * stride_ak
#         b_ptrs += BLOCK_SIZE_K * stride_bk

#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_n[None, :] * stride_cn)
#     tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])

# @triton.jit
# def fused_moe_w13_kernel(
#     a_ptr, b_ptr, c_ptr,
#     sorted_token_ids_ptr, expert_ids_ptr,
#     N, K, # 这里的 N 是 2 * inter_size
#     stride_am, stride_ak,
#     stride_be, stride_bk, stride_bn,
#     stride_cm, stride_cn,
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
# ):
#     pid = tl.program_id(axis=0)
#     # 因为融合了 Gate 和 Up，输出维度减半
#     inter_size = N // 2
#     num_pid_n = tl.cdiv(inter_size, BLOCK_SIZE_N) 
#     pid_m = pid // num_pid_n
#     pid_n = pid % num_pid_n

#     expert_id = tl.load(expert_ids_ptr + pid_m)
#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_token = tl.load(sorted_token_ids_ptr + offs_m)
    
#     # 护栏
#     mask_m = offs_token >= 0
#     safe_offs_token = tl.where(mask_m, offs_token, 0)

#     # 精准定位 Gate 和 Up 的指针
#     current_b_gate_ptr = b_ptr + expert_id * stride_be
#     current_b_up_ptr = b_ptr + expert_id * stride_be + inter_size * stride_bn

#     acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#     acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

#     offs_k = tl.arange(0, BLOCK_SIZE_K)
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
#     a_ptrs = a_ptr + (safe_offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     gate_ptrs = current_b_gate_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
#     up_ptrs = current_b_up_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         mask_k = (k * BLOCK_SIZE_K + offs_k) < K
#         a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
#         b_gate = tl.load(gate_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < inter_size), other=0.0)
#         b_up = tl.load(up_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < inter_size), other=0.0)
        
#         acc_gate = tl.dot(a, b_gate, acc=acc_gate)
#         acc_up = tl.dot(a, b_up, acc=acc_up)
        
#         a_ptrs += BLOCK_SIZE_K * stride_ak
#         gate_ptrs += BLOCK_SIZE_K * stride_bk
#         up_ptrs += BLOCK_SIZE_K * stride_bk

#     # 寄存器内直接完成 SiLU 激活与合并！
#     activated = (acc_gate * tl.sigmoid(acc_gate.to(tl.float32))) * acc_up
    
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_n[None, :] * stride_cn)
#     tl.store(c_ptrs, activated.to(c_ptr.dtype.element_ty), mask=mask_m[:, None] & (offs_n[None, :] < inter_size))



import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional

import triton
import triton.language as tl

@triton.jit
def fused_moe_w13_kernel(
    a_ptr, b_ptr, c_ptr,
    sorted_token_ids_ptr, sorted_weight_idx_ptr, expert_ids_ptr,
    num_blocks, num_valid_tokens, N, K,
    stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # -----------------------------------------------------------
    # 1. 块分组 (Tile Grouping) 提升 L2 命中率
    # -----------------------------------------------------------
    pid = tl.program_id(axis=0)
    inter_size = N // 2 
    num_pid_n = tl.cdiv(inter_size, BLOCK_SIZE_N) 
    num_pid_m = num_blocks 

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # 2. 内存偏移初始化 (提取到循环外，强制 int64 防溢出)
    # -----------------------------------------------------------
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_m).to(tl.int64)
    offs_weight = tl.load(sorted_weight_idx_ptr + offs_m).to(tl.int64)
    
    mask_m = offs_token < num_valid_tokens
    mask_bn = offs_n < inter_size

    # 【优化】循环外的基准指针初始化
    a_ptrs = a_ptr + (offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    current_b_gate_ptr = b_ptr + expert_id * stride_be
    current_b_up_ptr = b_ptr + expert_id * stride_be + inter_size * stride_bn

    gate_ptrs = current_b_gate_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    up_ptrs   = current_b_up_ptr   + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up   = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # -----------------------------------------------------------
    # 3. 极速 GEMM 主循环 (内层指针直接步进)
    # -----------------------------------------------------------
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = (k * BLOCK_SIZE_K + offs_k) < K
        
        # 直接加载，避免在循环里做复杂的乘法运算
        a      = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b_gate = tl.load(gate_ptrs, mask=mask_k[:, None] & mask_bn[None, :], other=0.0)
        b_up   = tl.load(up_ptrs, mask=mask_k[:, None] & mask_bn[None, :], other=0.0)
        
        acc_gate = tl.dot(a, b_gate, acc=acc_gate)
        acc_up   = tl.dot(a, b_up,   acc=acc_up)
        
        # 【修复】指针安全步进
        a_ptrs    += BLOCK_SIZE_K * stride_ak
        gate_ptrs += BLOCK_SIZE_K * stride_bk
        up_ptrs   += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # 4. SwiGLU 激活与隔离写回
    # -----------------------------------------------------------
    activated = (acc_gate * tl.sigmoid(acc_gate)) * acc_up
    
    c_ptrs = c_ptr + (offs_weight[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, activated.to(c_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_bn[None, :])

@triton.jit
def fused_moe_w2_combine_kernel(
    a_ptr, b_ptr, c_ptr, routing_weights_ptr,
    sorted_token_ids_ptr, sorted_weight_idx_ptr, expert_ids_ptr,
    num_blocks, num_valid_tokens, N, K,
    stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # -----------------------------------------------------------
    # 1. 块分组映射
    # -----------------------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = num_blocks

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # 2. 内存偏移初始化 (提取到循环外)
    # -----------------------------------------------------------
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_m).to(tl.int64)
    offs_weight = tl.load(sorted_weight_idx_ptr + offs_m).to(tl.int64)
    
    mask_m = offs_token < num_valid_tokens
    mask_n = offs_n < N
    
    # 获取路由权重，一次性准备好
    route_weight = tl.load(routing_weights_ptr + offs_weight, mask=mask_m, other=0.0).to(tl.float32)

    # 【优化】指针初始化移至循环外
    a_ptrs = a_ptr + (offs_weight[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + expert_id * stride_be + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # -----------------------------------------------------------
    # 3. 极速 GEMM 主循环
    # -----------------------------------------------------------
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = (k * BLOCK_SIZE_K + offs_k) < K
        
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        accumulator = tl.dot(a, b, acc=accumulator)
        
        # 【修复】指针安全步进
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # 4. 加权并原子写回 (还原位置)
    # -----------------------------------------------------------
    accumulator = accumulator * route_weight[:, None]
    
    c_ptrs = c_ptr + (offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])