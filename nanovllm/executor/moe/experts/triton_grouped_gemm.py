from __future__ import annotations

import torch
import triton

from nanovllm.executor.moe.experts.base import MoEExpertsKernel
from nanovllm.kernels.group_gemm import fused_moe_w13_kernel, fused_moe_w2_combine_kernel
from nanovllm.utils.moe import moe_align_block_size


class TritonGroupedGemmExperts(MoEExpertsKernel):
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
        num_recv = x.shape[0]
        if w13.dtype == torch.uint8:
            w13 = w13.view(torch.float8_e4m3fn).to(model_dtype)
            if w13_weight_scale is None:
                raise ValueError("FP8 MoE w13_stacked requires w13_weight_scale")
            w13 = w13 * w13_weight_scale.to(model_dtype).unsqueeze(-1)
        if w2.dtype == torch.uint8:
            w2 = w2.view(torch.float8_e4m3fn).to(model_dtype)
            if w2_weight_scale is None:
                raise ValueError("FP8 MoE w2_stacked requires w2_weight_scale")
            w2 = w2 * w2_weight_scale.to(model_dtype).unsqueeze(-1)

        block_size_m, group_size_m = 32, 8
        sorted_token_ids, sorted_weight_idx, expert_ids, num_blocks = moe_align_block_size(
            topk_ids.view(-1, 1), local_num_experts, block_size_m
        )

        activated_out = torch.empty((num_recv, local_inter_size), device=x.device, dtype=model_dtype)
        grid_w13 = lambda meta: (num_blocks * triton.cdiv(local_inter_size, meta["BLOCK_SIZE_N"]),)
        fused_moe_w13_kernel[grid_w13](
            x,
            w13,
            activated_out,
            sorted_token_ids,
            sorted_weight_idx,
            expert_ids,
            num_blocks,
            num_recv,
            local_inter_size * 2,
            hidden_size,
            x.stride(0),
            x.stride(1),
            w13.stride(0),
            w13.stride(2),
            w13.stride(1),
            activated_out.stride(0),
            activated_out.stride(1),
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=group_size_m,
        )

        local_out_fp32 = torch.zeros((num_recv, hidden_size), device=x.device, dtype=torch.float32)
        grid_w2 = lambda meta: (num_blocks * triton.cdiv(hidden_size, meta["BLOCK_SIZE_N"]),)
        fused_moe_w2_combine_kernel[grid_w2](
            activated_out,
            w2,
            local_out_fp32,
            topk_weights,
            sorted_token_ids,
            sorted_weight_idx,
            expert_ids,
            num_blocks,
            num_recv,
            hidden_size,
            local_inter_size,
            activated_out.stride(0),
            activated_out.stride(1),
            w2.stride(0),
            w2.stride(2),
            w2.stride(1),
            local_out_fp32.stride(0),
            local_out_fp32.stride(1),
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=group_size_m,
        )
        return local_out_fp32
