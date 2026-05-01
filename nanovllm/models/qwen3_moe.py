import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from typing import Optional

from transformers import Qwen3MoeConfig
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention_block import Qwen3AttentionBlock
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from nanovllm.executor.moe.blocks import BaseSparseMoeBlock

# Alias for local backward compatibility and readability within this module.
Qwen3MoeAttention = Qwen3AttentionBlock


class Qwen3MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        reduce_results=True,
        tp_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            tp_group=tp_group
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            tp_group=tp_group
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x

# Eager 模式的 MoE (支持 TP，无 GroupGEMM，主要用于对齐/Debug)
class Qwen3MoeEagerSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config,
        tp_group: Optional[dist.ProcessGroup] = None,
        ep_group: Optional[dist.ProcessGroup] = None, 
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.tp_group = tp_group

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act=config.hidden_act,
                    tp_group=tp_group
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor):
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Qwen3 专属逻辑：归一化 routing weights
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for i in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[i])
            if top_x.shape[0] == 0:
                continue
            expert_out = self.experts[i](hidden_states[top_x])
            final_hidden_states.index_add_(0, top_x, expert_out * routing_weights[top_x, idx, None])

        return final_hidden_states.view(orig_shape)


# Triton Group-GEMM + TP + EP 版本的 MoE
class Qwen3MoeSparseMoeBlock(BaseSparseMoeBlock):
    def __init__(
        self,
        config,
        tp_group: Optional[dist.ProcessGroup] = None,
        ep_group: Optional[dist.ProcessGroup] = None,
        experts_backend: str = "fused",
    ) -> None:
        super().__init__(
            config,
            tp_group=tp_group,
            ep_group=ep_group,
            renormalize_router_weights=True,
            experts_backend=experts_backend,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_size).contiguous()
        topk_weights, topk_ids = self.route(x)
        output = self.apply_sparse_experts(x, topk_weights, topk_ids)
        return output.view(orig_shape)

class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int = -1,
        tp_group: Optional[dist.ProcessGroup] = None,
        ep_group: Optional[dist.ProcessGroup] = None,
        group_gemm_enable: bool = True,
        moe_backend: str = "fused",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            # rope_scaling=getattr(config, "rope_scaling", None),
            rope_scaling=None,
            tp_group=tp_group
        )
        
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            if group_gemm_enable:
                self.mlp = Qwen3MoeSparseMoeBlock(
                    config=config,
                    tp_group=tp_group,
                    ep_group=ep_group,
                    experts_backend=moe_backend,
                )
            else:
                self.mlp = Qwen3MoeEagerSparseMoeBlock(config=config, tp_group=tp_group, ep_group=ep_group)
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                tp_group=tp_group
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        mlp_out = self.mlp(hidden_states)
        return mlp_out, residual


class Qwen3MoeModel(nn.Module):
    def __init__(self, config, tp_group=None, ep_group=None, moe_backend: str = "fused") -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, tp_group=tp_group
        )
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, layer_idx, tp_group, ep_group, moe_backend=moe_backend)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self, 
        config, 
        tp_group: Optional[dist.ProcessGroup] = None, 
        ep_group: Optional[dist.ProcessGroup] = None,
        moe_backend: str = "fused",
        **kwargs 
    ) -> None:
        super().__init__()
        self.tp_group = tp_group
        self.ep_group = ep_group

        self.model = Qwen3MoeModel(
            config,
            tp_group=self.tp_group,
            ep_group=self.ep_group,
            moe_backend=moe_backend,
        )
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, tp_group=self.tp_group)
        
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
