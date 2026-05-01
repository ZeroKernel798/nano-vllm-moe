from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class MoEExpertsKernel(ABC):
    @abstractmethod
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
        raise NotImplementedError
