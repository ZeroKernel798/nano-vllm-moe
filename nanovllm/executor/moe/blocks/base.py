from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.distributed as dist
from torch import nn

from nanovllm.executor.moe.config import MoEParallelConfig, make_moe_parallel_config
from nanovllm.executor.moe.experts import (
    MiniSglangExperts,
    MoEExpertsKernel,
    TransformersEagerExperts,
    TritonGroupedGemmExperts,
)
from nanovllm.executor.moe.kernel import MoEKernel
from nanovllm.executor.moe.prepare_finalize import NoEPPrepareFinalize, TorchAllToAllPrepareFinalize
from nanovllm.executor.moe.router import MoERouter, SoftmaxTopKRouter

MoEBackendName = Literal["transformers", "mini_sglang", "fused"]


class BaseSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config,
        *,
        tp_group: Optional[dist.ProcessGroup] = None,
        ep_group: Optional[dist.ProcessGroup] = None,
        renormalize_router_weights: bool,
        experts_backend: MoEBackendName = "fused",
        router: MoERouter | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.global_num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.parallel_config = make_moe_parallel_config(
            tp_group=tp_group,
            ep_group=ep_group,
            global_num_experts=self.global_num_experts,
            intermediate_size=self.moe_intermediate_size,
        )
        self.tp_group = self.parallel_config.tp_group
        self.ep_group = self.parallel_config.ep_group
        self.tp_size = self.parallel_config.tp_size
        self.tp_rank = self.parallel_config.tp_rank
        self.ep_size = self.parallel_config.ep_size
        self.ep_rank = self.parallel_config.ep_rank
        self.local_num_experts = self.parallel_config.local_num_experts
        self.local_inter_size = self.parallel_config.local_inter_size

        self.w13_stacked = nn.Parameter(
            torch.zeros(self.local_num_experts, 2 * self.local_inter_size, self.hidden_size)
        )
        self.w2_stacked = nn.Parameter(
            torch.zeros(self.local_num_experts, self.hidden_size, self.local_inter_size)
        )
        self.w13_stacked.weight_loader = self.load_hybrid_moe_weight
        self.w2_stacked.weight_loader = self.load_hybrid_moe_weight

        self.gate = nn.Linear(self.hidden_size, self.global_num_experts, bias=False)
        self.gate.weight.weight_loader = self.load_replicated_weight
        self.router = router or SoftmaxTopKRouter(self.top_k, renormalize=renormalize_router_weights)
        self.experts_backend_name = experts_backend
        self.moe_kernel = MoEKernel(
            prepare_finalize=self._make_prepare_finalize(self.parallel_config),
            experts=self._make_experts(experts_backend),
            parallel_config=self.parallel_config,
        )

    @staticmethod
    def _make_prepare_finalize(parallel_config: MoEParallelConfig):
        if parallel_config.ep_size <= 1:
            return NoEPPrepareFinalize(parallel_config)
        return TorchAllToAllPrepareFinalize(parallel_config)

    @staticmethod
    def _make_experts(experts_backend: MoEBackendName) -> MoEExpertsKernel:
        if experts_backend == "transformers":
            return TransformersEagerExperts()
        if experts_backend == "mini_sglang":
            return MiniSglangExperts()
        if experts_backend == "fused":
            return TritonGroupedGemmExperts()
        raise ValueError(f"Unsupported MoE experts backend: {experts_backend}")

    def load_replicated_weight(self, param, loaded_weight, **kwargs):
        del kwargs
        with torch.no_grad():
            param.copy_(loaded_weight)

    def load_hybrid_moe_weight(self, param, loaded_weight, global_expert_id, shard_id=None, **kwargs):
        del kwargs
        with torch.no_grad():
            expert_start = self.ep_rank * self.local_num_experts
            expert_end = (self.ep_rank + 1) * self.local_num_experts
            if not (expert_start <= global_expert_id < expert_end):
                return
            local_id = global_expert_id % self.local_num_experts
            start = self.tp_rank * self.local_inter_size
            size = self.local_inter_size
            if shard_id in [0, "w1", "gate_proj"]:
                param.data[local_id].narrow(0, 0, size).copy_(loaded_weight.narrow(0, start, size))
            elif shard_id in [1, "w3", "up_proj"]:
                param.data[local_id].narrow(0, size, size).copy_(loaded_weight.narrow(0, start, size))
            elif shard_id in [None, "w2", "down_proj"]:
                param.data[local_id].copy_(loaded_weight.narrow(1, start, size))
            else:
                raise ValueError(f"Unsupported MoE shard_id={shard_id}")

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.gate(x)
        return self.router.select_experts(x, router_logits)

    def apply_sparse_experts(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        reduce_tp: bool = True,
    ) -> torch.Tensor:
        return self.moe_kernel(
            x,
            topk_weights,
            topk_ids,
            self.w13_stacked,
            self.w2_stacked,
            model_dtype=x.dtype,
            reduce_tp=reduce_tp,
        )
