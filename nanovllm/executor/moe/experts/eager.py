from __future__ import annotations

import torch
import torch.nn.functional as F

from nanovllm.executor.moe.experts.base import MoEExpertsKernel


class TransformersEagerExperts(MoEExpertsKernel):
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
        del local_inter_size, hidden_size, model_dtype, w13_weight_scale, w2_weight_scale
        output = torch.zeros(x.shape[0], w2.shape[1], device=x.device, dtype=x.dtype)
        for expert_id in range(local_num_experts):
            token_indices = torch.nonzero(topk_ids == expert_id, as_tuple=False).flatten()
            if token_indices.numel() == 0:
                continue
            gate_up = F.linear(x[token_indices], w13[expert_id])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_out = F.linear(F.silu(gate) * up, w2[expert_id])
            output[token_indices] = expert_out * topk_weights[token_indices, None].to(expert_out.dtype)
        return output
