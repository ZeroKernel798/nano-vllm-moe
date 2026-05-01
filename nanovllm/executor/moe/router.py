from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class MoERouter(ABC):
    @abstractmethod
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class SoftmaxTopKRouter(MoERouter):
    def __init__(self, top_k: int, renormalize: bool) -> None:
        self.top_k = top_k
        self.renormalize = renormalize

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del hidden_states
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights.to(torch.float32), topk_ids
