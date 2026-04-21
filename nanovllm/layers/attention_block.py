"""Unified Qwen3 attention block shared by dense and MoE model variants.

``Qwen3Attention`` (qwen3.py) and ``Qwen3MoeAttention`` (qwen3_moe.py) were
structurally identical: same constructor signature, same forward logic, same
QK-norm application.  This module provides a single canonical implementation.

Both model files now import ``Qwen3AttentionBlock`` from here and alias it
locally under their original names for backwards compatibility.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope


class Qwen3AttentionBlock(nn.Module):
    """Multi-head attention with per-head QK-RMSNorm (Qwen3 style).

    Parameters
    ----------
    hidden_size:
        Model hidden dimension.
    num_heads:
        Total number of query heads (before TP sharding).
    num_kv_heads:
        Total number of key/value heads (before TP sharding).
    max_position:
        Maximum sequence length for rotary embedding.
    head_dim:
        Per-head dimension.  Defaults to ``hidden_size // num_heads``.
    rms_norm_eps:
        Epsilon for QK RMSNorm layers.
    qkv_bias:
        Whether to add bias to the QKV projection.
    rope_theta:
        Base frequency for RoPE.
    rope_scaling:
        Optional RoPE scaling config (passed through to ``get_rope``).
    tp_group:
        Tensor-parallel process group.  ``None`` means no tensor parallelism.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        tp_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__()

        tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            tp_group=tp_group,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            tp_group=tp_group,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o)
