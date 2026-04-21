"""Qwen2 FP8 W8A16 quantized model.

Weights are stored as float8_e4m3fn (uint8 view) with per-output-channel
float32 scale.  Activations remain in bfloat16.  Bias tensors are kept in
their original precision.

Load from a quantized checkpoint produced by scripts/quantize/quantize.py.
The checkpoint keeps the same HF key structure but replaces *.weight with
*.qweight (uint8, [N,K]) + *.weight_scale (float32, [N]).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from transformers import Qwen2Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.fp8 import (
    FP8MergedColumnParallelLinear,
    FP8QKVParallelLinear,
    FP8RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope


class Qwen2FP8Attention(nn.Module):
    def __init__(self, config: Qwen2Config, tp_group=None) -> None:
        super().__init__()
        tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = FP8QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", True),
            fp8_scheme="w8a16",
            tp_group=tp_group,
        )
        self.o_proj = FP8RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=False,
            fp8_scheme="w8a16",
            tp_group=tp_group,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 1_000_000),
            rope_scaling=None,
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o)


class Qwen2FP8MLP(nn.Module):
    def __init__(self, config: Qwen2Config, tp_group=None) -> None:
        super().__init__()
        self.gate_up_proj = FP8MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            fp8_scheme="w8a16",
            tp_group=tp_group,
        )
        self.down_proj = FP8RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            fp8_scheme="w8a16",
            tp_group=tp_group,
        )
        assert config.hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen2FP8DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, tp_group=None) -> None:
        super().__init__()
        self.self_attn = Qwen2FP8Attention(config, tp_group=tp_group)
        self.mlp = Qwen2FP8MLP(config, tp_group=tp_group)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2FP8Model(nn.Module):
    def __init__(self, config: Qwen2Config, tp_group=None) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, tp_group=tp_group)
        self.layers = nn.ModuleList(
            [Qwen2FP8DecoderLayer(config, tp_group=tp_group) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLMFP8(nn.Module):
    """Qwen2 with FP8 W8A16 quantized linear layers.

    Expected checkpoint format: *.qweight (uint8) + *.weight_scale (float32).
    Produced by ``scripts/quantize/quantize.py --scheme fp8_w8a16``.
    """
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen2Config, tp_group=None, **kwargs) -> None:
        super().__init__()
        self.model = Qwen2FP8Model(config, tp_group=tp_group)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, tp_group=tp_group)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
