from __future__ import annotations

import torch

from nanovllm.executor.moe.config import MoEParallelConfig
from nanovllm.executor.moe.experts import MoEExpertsKernel
from nanovllm.executor.moe.prepare_finalize import MoEPrepareFinalize


class MoEKernel:
    def __init__(
        self,
        *,
        prepare_finalize: MoEPrepareFinalize,
        experts: MoEExpertsKernel,
        parallel_config: MoEParallelConfig,
    ) -> None:
        self.prepare_finalize = prepare_finalize
        self.experts = experts
        self.parallel_config = parallel_config

    def __call__(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        *,
        model_dtype: torch.dtype,
        reduce_tp: bool = True,
        w13_weight_scale: torch.Tensor | None = None,
        w2_weight_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prepared = self.prepare_finalize.prepare(x, topk_weights, topk_ids)
        local_out = self.experts.apply(
            prepared.hidden_states,
            prepared.topk_ids,
            prepared.topk_weights,
            w13,
            w2,
            local_num_experts=self.parallel_config.local_num_experts,
            local_inter_size=self.parallel_config.local_inter_size,
            hidden_size=x.shape[-1],
            model_dtype=model_dtype,
            w13_weight_scale=w13_weight_scale,
            w2_weight_scale=w2_weight_scale,
        )
        return self.prepare_finalize.finalize(
            local_out,
            prepared,
            output_shape=(x.shape[0], x.shape[1]),
            model_dtype=model_dtype,
            reduce_tp=reduce_tp,
        )
