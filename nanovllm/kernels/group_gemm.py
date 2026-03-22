import triton
import triton.language as tl

@triton.jit
def moe_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    N, K,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    expert_id = tl.load(expert_ids_ptr + pid_m)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_m)
    
    # 如果是 Padding (-1)，强制指针指向 0 以防段错误
    mask_m = offs_token >= 0
    safe_offs_token = tl.where(mask_m, offs_token, 0)

    current_b_ptr = b_ptr + expert_id * stride_be
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    # 使用安全的 Token 偏移量计算 A 的指针
    a_ptrs = a_ptr + (safe_offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = current_b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = (k * BLOCK_SIZE_K + offs_k) < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def fused_moe_w13_kernel(
    a_ptr, b_ptr, c_ptr,
    sorted_token_ids_ptr, expert_ids_ptr,
    N, K, # 这里的 N 是 2 * inter_size
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # 因为融合了 Gate 和 Up，输出维度减半
    inter_size = N // 2
    num_pid_n = tl.cdiv(inter_size, BLOCK_SIZE_N) 
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    expert_id = tl.load(expert_ids_ptr + pid_m)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_m)
    
    # 护栏
    mask_m = offs_token >= 0
    safe_offs_token = tl.where(mask_m, offs_token, 0)

    # 精准定位 Gate 和 Up 的指针
    current_b_gate_ptr = b_ptr + expert_id * stride_be
    current_b_up_ptr = b_ptr + expert_id * stride_be + inter_size * stride_bn

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    a_ptrs = a_ptr + (safe_offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
    gate_ptrs = current_b_gate_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    up_ptrs = current_b_up_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = (k * BLOCK_SIZE_K + offs_k) < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b_gate = tl.load(gate_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < inter_size), other=0.0)
        b_up = tl.load(up_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < inter_size), other=0.0)
        
        acc_gate = tl.dot(a, b_gate, acc=acc_gate)
        acc_up = tl.dot(a, b_up, acc=acc_up)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        gate_ptrs += BLOCK_SIZE_K * stride_bk
        up_ptrs += BLOCK_SIZE_K * stride_bk

    # 寄存器内直接完成 SiLU 激活与合并！
    activated = (acc_gate * tl.sigmoid(acc_gate.to(tl.float32))) * acc_up
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, activated.to(c_ptr.dtype.element_ty), mask=mask_m[:, None] & (offs_n[None, :] < inter_size))

