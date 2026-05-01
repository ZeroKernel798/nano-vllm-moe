from __future__ import annotations

import torch
import torch.distributed as dist

from nanovllm.executor.moe.config import MoEParallelConfig
from nanovllm.executor.moe.prepare_finalize.base import MoEPrepareFinalize, PrepareResult


class TorchAllToAllPrepareFinalize(MoEPrepareFinalize):
    supports_cuda_graph = False
    uses_dynamic_alltoall = True

    def __init__(self, parallel_config: MoEParallelConfig) -> None:
        if parallel_config.ep_size <= 1:
            raise ValueError("TorchAllToAllPrepareFinalize requires ep_size > 1")
        self.parallel_config = parallel_config

    def prepare(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> PrepareResult:
        ep_size = self.parallel_config.ep_size
        local_num_experts = self.parallel_config.local_num_experts
        top_k = topk_ids.shape[-1]
        target_ep_ranks = torch.div(topk_ids, local_num_experts, rounding_mode="floor").clamp(0, ep_size - 1)
        flat_target_ep_ranks = target_ep_ranks.flatten()
        permute_indices = torch.argsort(flat_target_ep_ranks)

        expanded_x = x.repeat_interleave(top_k, dim=0)
        dispatched_x = expanded_x[permute_indices]
        dispatched_ids = (topk_ids % local_num_experts).flatten()[permute_indices]
        dispatched_weights = topk_weights.flatten()[permute_indices]

        send_counts = torch.bincount(flat_target_ep_ranks, minlength=ep_size).to(torch.long).to(x.device)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.parallel_config.ep_group)

        s_list = send_counts.tolist()
        r_list = recv_counts.tolist()
        num_recv = int(recv_counts.sum().item())

        recv_x = torch.empty(num_recv, x.shape[1], device=x.device, dtype=x.dtype)
        recv_ids = torch.empty(num_recv, dtype=torch.long, device=x.device)
        recv_weights = torch.empty(num_recv, dtype=torch.float32, device=x.device)

        dist.all_to_all_single(recv_x, dispatched_x, r_list, s_list, group=self.parallel_config.ep_group)
        dist.all_to_all_single(recv_ids, dispatched_ids, r_list, s_list, group=self.parallel_config.ep_group)
        dist.all_to_all_single(recv_weights, dispatched_weights, r_list, s_list, group=self.parallel_config.ep_group)

        return PrepareResult(
            hidden_states=recv_x,
            topk_ids=recv_ids,
            topk_weights=recv_weights,
            ctx={
                "top_k": top_k,
                "permute_indices": permute_indices,
                "s_list": s_list,
                "r_list": r_list,
            },
        )

    def finalize(
        self,
        expert_out: torch.Tensor,
        prepared: PrepareResult,
        *,
        output_shape: tuple[int, int],
        model_dtype: torch.dtype,
        reduce_tp: bool = True,
    ) -> torch.Tensor:
        m_tokens, hidden_size = output_shape
        top_k = prepared.ctx["top_k"]
        s_list = prepared.ctx["s_list"]
        r_list = prepared.ctx["r_list"]
        permute_indices = prepared.ctx["permute_indices"]

        combined_x = torch.empty(m_tokens * top_k, hidden_size, device=expert_out.device, dtype=model_dtype)
        dist.all_to_all_single(
            combined_x,
            expert_out.to(model_dtype),
            s_list,
            r_list,
            group=self.parallel_config.ep_group,
        )
        sparse_out_flat = torch.empty((m_tokens * top_k, hidden_size), device=expert_out.device, dtype=model_dtype)
        sparse_out_flat[permute_indices] = combined_x
        output = sparse_out_flat.view(m_tokens, top_k, hidden_size).sum(dim=1)
        if reduce_tp and self.parallel_config.tp_size > 1:
            dist.all_reduce(output, group=self.parallel_config.tp_group)
        return output
