from __future__ import annotations

import os
import time

import torch

from nanovllm.executor.moe.config import MoEParallelConfig
from nanovllm.executor.moe.experts import MoEExpertsKernel
from nanovllm.executor.moe.prepare_finalize import MoEPrepareFinalize
from nanovllm.executor.moe.profile import record_moe_profile


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
        profile = os.environ.get("NANOVLLM_MOE_PROFILE", "0") == "1"
        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter() if profile else 0.0
        prepared = self.prepare_finalize.prepare(x, topk_weights, topk_ids)
        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            record_moe_profile("prepare", time.perf_counter() - start)
            start = time.perf_counter()
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
        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            record_moe_profile("experts", time.perf_counter() - start)
            start = time.perf_counter()
        output = self.prepare_finalize.finalize(
            local_out,
            prepared,
            output_shape=(x.shape[0], x.shape[1]),
            model_dtype=model_dtype,
            reduce_tp=reduce_tp,
        )
        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            record_moe_profile("finalize", time.perf_counter() - start)
        return output
