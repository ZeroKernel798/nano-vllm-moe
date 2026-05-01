from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PrepareResult:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    ctx: dict[str, Any]


class MoEPrepareFinalize(ABC):
    supports_cuda_graph: bool = False
    uses_dynamic_alltoall: bool = True

    @abstractmethod
    def prepare(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> PrepareResult:
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        expert_out: torch.Tensor,
        prepared: PrepareResult,
        *,
        output_shape: tuple[int, int],
        model_dtype: torch.dtype,
        reduce_tp: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError
