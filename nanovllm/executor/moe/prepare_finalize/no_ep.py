from __future__ import annotations

import torch
import torch.distributed as dist

from nanovllm.executor.moe.config import MoEParallelConfig
from nanovllm.executor.moe.prepare_finalize.base import MoEPrepareFinalize, PrepareResult


class NoEPPrepareFinalize(MoEPrepareFinalize):
    supports_cuda_graph = True
    uses_dynamic_alltoall = False

    def __init__(self, parallel_config: MoEParallelConfig) -> None:
        if parallel_config.ep_size != 1:
            raise ValueError("NoEPPrepareFinalize requires ep_size == 1")
        self.parallel_config = parallel_config

    def prepare(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> PrepareResult:
        top_k = topk_ids.shape[-1]
        expanded_x = x if top_k == 1 else x.repeat_interleave(top_k, dim=0)
        return PrepareResult(
            hidden_states=expanded_x,
            topk_ids=topk_ids.flatten(),
            topk_weights=topk_weights.flatten(),
            ctx={"top_k": top_k},
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
        output = expert_out.to(model_dtype).view(m_tokens, top_k, hidden_size).sum(dim=1)
        if reduce_tp and self.parallel_config.tp_size > 1:
            dist.all_reduce(output, group=self.parallel_config.tp_group)
        return output
