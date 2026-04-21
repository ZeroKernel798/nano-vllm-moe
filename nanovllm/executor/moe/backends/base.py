from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class MoEBackend(ABC):
    """Backend interface for sparse MoE dispatch/compute/combine."""

    @abstractmethod
    def dispatch(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def compute(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def combine(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError
