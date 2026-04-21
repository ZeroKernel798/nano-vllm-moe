from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import triton

from nanovllm.executor.moe.backends.base import MoEBackend
from nanovllm.kernels.group_gemm import fused_moe_w13_kernel, fused_moe_w2_combine_kernel
from nanovllm.utils.moe import moe_align_block_size


class TritonMoEBackend(MoEBackend):
    """Current Triton Group-GEMM backend for sparse MoE."""

    def __init__(self, tp_group: Optional[dist.ProcessGroup], ep_group: Optional[dist.ProcessGroup]):
        self.tp_group = tp_group
        self.ep_group = ep_group
        self.tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1

    def dispatch(
        self,
        *,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_num_experts: int,
        top_k: int,
    ) -> dict[str, torch.Tensor | list[int] | int]:
        if self.ep_size <= 1:
            recv_x = x.repeat_interleave(top_k, dim=0)
            recv_local_ids = topk_ids.flatten()
            recv_weights = topk_weights.flatten()
            num_recv = recv_x.shape[0]
            permute_indices = torch.arange(num_recv, device=x.device)
            s_list = [num_recv]
            r_list = [num_recv]
        else:
            target_ep_ranks = torch.div(topk_ids, local_num_experts, rounding_mode="floor").clamp(0, self.ep_size - 1)
            flat_target_ep_ranks = target_ep_ranks.flatten()
            permute_indices = torch.argsort(flat_target_ep_ranks)

            expanded_x = x.repeat_interleave(top_k, dim=0)
            dispatched_x = expanded_x[permute_indices]

            send_counts = torch.bincount(flat_target_ep_ranks, minlength=self.ep_size).to(torch.long).to(x.device)
            recv_counts = torch.empty_like(send_counts)
            dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

            s_list, r_list = send_counts.tolist(), recv_counts.tolist()
            num_recv = recv_counts.sum().item()

            recv_x = torch.empty(num_recv, x.shape[1], device=x.device, dtype=x.dtype)
            recv_local_ids = torch.empty(num_recv, dtype=torch.long, device=x.device)
            recv_weights = torch.empty(num_recv, dtype=torch.float32, device=x.device)

            dist.all_to_all_single(recv_x, dispatched_x, r_list, s_list, group=self.ep_group)
            dist.all_to_all_single(
                recv_local_ids,
                (topk_ids % local_num_experts).flatten()[permute_indices],
                r_list,
                s_list,
                group=self.ep_group,
            )
            dist.all_to_all_single(recv_weights, topk_weights.flatten()[permute_indices], r_list, s_list, group=self.ep_group)

        return {
            "recv_x": recv_x,
            "recv_local_ids": recv_local_ids,
            "recv_weights": recv_weights,
            "permute_indices": permute_indices,
            "s_list": s_list,
            "r_list": r_list,
            "num_recv": num_recv,
        }

    def compute(
        self,
        *,
        recv_x: torch.Tensor,
        recv_local_ids: torch.Tensor,
        recv_weights: torch.Tensor,
        w13_stacked: torch.Tensor,
        w2_stacked: torch.Tensor,
        local_num_experts: int,
        local_inter_size: int,
        hidden_size: int,
        model_dtype: torch.dtype,
    ) -> torch.Tensor:
        num_recv = recv_x.shape[0]
        block_size_m, group_size_m = 32, 8
        sorted_token_ids, sorted_weight_idx, expert_ids, num_blocks = moe_align_block_size(
            recv_local_ids.view(-1, 1), local_num_experts, block_size_m
        )

        activated_out = torch.empty((num_recv, local_inter_size), device=recv_x.device, dtype=model_dtype)
        grid_w13 = lambda meta: (num_blocks * triton.cdiv(local_inter_size, meta["BLOCK_SIZE_N"]),)
        fused_moe_w13_kernel[grid_w13](
            recv_x,
            w13_stacked,
            activated_out,
            sorted_token_ids,
            sorted_weight_idx,
            expert_ids,
            num_blocks,
            num_recv,
            local_inter_size * 2,
            hidden_size,
            recv_x.stride(0),
            recv_x.stride(1),
            w13_stacked.stride(0),
            w13_stacked.stride(2),
            w13_stacked.stride(1),
            activated_out.stride(0),
            activated_out.stride(1),
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=group_size_m,
        )

        local_out_fp32 = torch.zeros((num_recv, hidden_size), device=recv_x.device, dtype=torch.float32)
        grid_w2 = lambda meta: (num_blocks * triton.cdiv(hidden_size, meta["BLOCK_SIZE_N"]),)
        fused_moe_w2_combine_kernel[grid_w2](
            activated_out,
            w2_stacked,
            local_out_fp32,
            recv_weights,
            sorted_token_ids,
            sorted_weight_idx,
            expert_ids,
            num_blocks,
            num_recv,
            hidden_size,
            local_inter_size,
            activated_out.stride(0),
            activated_out.stride(1),
            w2_stacked.stride(0),
            w2_stacked.stride(2),
            w2_stacked.stride(1),
            local_out_fp32.stride(0),
            local_out_fp32.stride(1),
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=group_size_m,
        )
        return local_out_fp32

    def combine(
        self,
        *,
        local_out_fp32: torch.Tensor,
        model_dtype: torch.dtype,
        m_tokens: int,
        hidden_size: int,
        top_k: int,
        permute_indices: torch.Tensor,
        s_list: list[int],
        r_list: list[int],
    ) -> torch.Tensor:
        if self.ep_size > 1:
            combined_x = torch.empty(m_tokens * top_k, hidden_size, device=local_out_fp32.device, dtype=model_dtype)
            dist.all_to_all_single(combined_x, local_out_fp32.to(model_dtype), s_list, r_list, group=self.ep_group)
        else:
            combined_x = local_out_fp32.to(model_dtype)

        sparse_out_flat = torch.zeros((m_tokens * top_k, hidden_size), device=local_out_fp32.device, dtype=model_dtype)
        sparse_out_flat[permute_indices] = combined_x
        output = sparse_out_flat.view(m_tokens, top_k, hidden_size).sum(dim=1)

        if self.tp_size > 1:
            dist.all_reduce(output, group=self.tp_group)
        return output
