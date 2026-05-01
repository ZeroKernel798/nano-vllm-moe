from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from nanovllm.executor.moe.experts.base import MoEExpertsKernel
from nanovllm.kernels.sglang_moe import launch_fused_moe_kernel, launch_moe_sum_reduce
from nanovllm.utils.moe import moe_align_block_size


def _default_config(m_tokens: int, num_experts: int) -> dict[str, int]:
    if m_tokens <= num_experts:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }


class MiniSglangExperts(MoEExpertsKernel):
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
        del hidden_size, w13_weight_scale, w2_weight_scale
        if x.device.type != "cuda":
            return self._fallback_eager(x, topk_ids, topk_weights, w13, w2, local_num_experts)

        num_tasks = x.shape[0]
        top_k = 1
        topk_ids_2d = topk_ids.view(num_tasks, top_k).to(torch.int64)
        sorted_token_ids, _sorted_weight_idx, expert_ids, num_blocks = moe_align_block_size(
            topk_ids_2d, local_num_experts, _default_config(num_tasks, local_num_experts)["BLOCK_SIZE_M"]
        )
        config = _default_config(num_tasks, local_num_experts)
        num_tokens_post_padded = torch.tensor(
            [num_blocks * config["BLOCK_SIZE_M"]], device=x.device, dtype=torch.int32
        )
        sorted_token_ids = sorted_token_ids.to(torch.int32)
        expert_ids = expert_ids.to(torch.int32)

        gate_up = torch.empty((num_tasks, top_k, 2 * local_inter_size), device=x.device, dtype=model_dtype)
        launch_fused_moe_kernel(
            x,
            w13,
            gate_up,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight=False,
            top_k=top_k,
            config=config,
            compute_type=model_dtype,
        )

        activated = F.silu(gate_up.view(num_tasks * top_k, 2 * local_inter_size)[..., :local_inter_size])
        activated = activated * gate_up.view(num_tasks * top_k, 2 * local_inter_size)[..., local_inter_size:]
        activated = activated.contiguous()

        expert_out = torch.empty((num_tasks, top_k, w2.shape[1]), device=x.device, dtype=model_dtype)
        launch_fused_moe_kernel(
            activated,
            w2,
            expert_out,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight=True,
            top_k=top_k,
            config=config,
            compute_type=model_dtype,
        )

        output = torch.empty((num_tasks, w2.shape[1]), device=x.device, dtype=model_dtype)
        launch_moe_sum_reduce(expert_out, output)
        return output

    @staticmethod
    def _fallback_eager(
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        local_num_experts: int,
    ) -> torch.Tensor:
        outputs = []
        for token, expert_id, weight in zip(x, topk_ids, topk_weights, strict=True):
            expert = int(expert_id.item())
            if expert < 0 or expert >= local_num_experts:
                raise IndexError(f"local expert id {expert} out of range [0, {local_num_experts})")
            gate_up = F.linear(token.unsqueeze(0), w13[expert])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_out = F.linear(F.silu(gate) * up, w2[expert]).squeeze(0)
            outputs.append(expert_out * weight.to(expert_out.dtype))
        if not outputs:
            return torch.empty(x.shape[0], w2.shape[1], device=x.device, dtype=x.dtype)
        return torch.stack(outputs, dim=0)
