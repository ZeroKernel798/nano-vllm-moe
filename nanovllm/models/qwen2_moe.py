import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2MoeConfig 

from nanovllm.executor.moe.blocks import BaseSparseMoeBlock
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from typing import Optional

class Qwen2MoeAttention(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 1000000,
        rope_scaling: tuple | None = None,
        tp_group: Optional[dist.ProcessGroup] = None, 
    ) -> None:
        super().__init__()
        # 使用传入的 tp_group 大小，而不是全局大小
        tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // self.total_num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # 将 tp_group 传给底层算子
        self.qkv_proj = QKVParallelLinear(
            hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, 
            bias=True, tp_group=tp_group
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, hidden_size, 
            bias=False, tp_group=tp_group
        )

        self.rotary_emb = get_rope(
            self.head_dim, rotary_dim=self.head_dim, max_position=max_position,
            base=rope_theta, rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(self, positions, hidden_states) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output = self.o_proj(attn_output)
        return output
    
class Qwen2MoeMLP(nn.Module):
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
            hidden_size, [intermediate_size] * 2, bias=False, tp_group=tp_group
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, 
            reduce_results=reduce_results, tp_group=tp_group
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen2MoeEagerSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        tp_group: Optional[dist.ProcessGroup] = None,
        ep_group: Optional[dist.ProcessGroup] = None, # 传进来但不用，保持接口一致
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)

        # 同样需要传入 tp_group 保证内部线性层并行正确
        self.shared_expert = Qwen2MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            tp_group=tp_group 
        )

        self.experts = nn.ModuleList(
            [
                Qwen2MoeMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act=config.hidden_act,
                    tp_group=tp_group
                ) for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(hidden_states.dtype)

        shared_output = self.shared_expert(hidden_states)
        shared_weight = torch.sigmoid(self.shared_expert_gate(hidden_states))
        shared_output = shared_output * shared_weight

        sparse_hidden_states = torch.zeros(
            hidden_states.shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for i in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[i])
            if top_x.shape[0] == 0:
                continue
            
            expert_out = self.experts[i](hidden_states[top_x])
            sparse_hidden_states.index_add_(0, top_x, expert_out * routing_weights[top_x, idx, None])

        return (shared_output + sparse_hidden_states).view(orig_shape)
    

class Qwen2MoeSparseMoeBlock(BaseSparseMoeBlock):
    def __init__(
        self,
        config,
        tp_group: Optional[dist.ProcessGroup] = None,
        ep_group: Optional[dist.ProcessGroup] = None,
        use_overlap: bool = True,
        experts_backend: str = "fused",
    ) -> None:
        super().__init__(
            config,
            tp_group=tp_group,
            ep_group=ep_group,
            renormalize_router_weights=False,
            experts_backend=experts_backend,
        )
        self.use_overlap = use_overlap
        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)
        self.shared_expert_gate.weight.weight_loader = self.load_replicated_weight
        self.shared_expert = Qwen2MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            reduce_results=False,
            tp_group=tp_group,
        )
        self.shared_expert_stream = (
            torch.cuda.Stream() if use_overlap and torch.cuda.is_available() else None
        )

    def _compute_shared_expert(self, x):
        out = self.shared_expert(x)
        out *= torch.sigmoid(self.shared_expert_gate(x))
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_size).contiguous()
        topk_weights, topk_ids = self.route(x)

        if self.shared_expert_stream is not None:
            self.shared_expert_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.shared_expert_stream):
                shared_out = self._compute_shared_expert(x)
        else:
            shared_out = self._compute_shared_expert(x)

        sparse_out = self.apply_sparse_experts(x, topk_weights, topk_ids, reduce_tp=False)

        if self.shared_expert_stream is not None:
            torch.cuda.current_stream().wait_stream(self.shared_expert_stream)

        output = shared_out + sparse_out
        if self.tp_size > 1:
            dist.all_reduce(output, group=self.tp_group)
        return output.view(orig_shape)

class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(
        self, 
        config: Qwen2MoeConfig, 
        layer_idx: int,
        tp_group: Optional[dist.ProcessGroup] = None,
        ep_group: Optional[dist.ProcessGroup] = None,
        group_gemm_enable : bool = True,
        moe_backend: str = "fused",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen2MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 1000000),
            tp_group=tp_group 
        )

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            if group_gemm_enable:
                self.mlp = Qwen2MoeSparseMoeBlock(
                    config=config,
                    tp_group=tp_group,
                    ep_group=ep_group,
                    experts_backend=moe_backend,
                )
            else:
                self.mlp = Qwen2MoeEagerSparseMoeBlock(config=config, tp_group=tp_group, ep_group=ep_group)
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                tp_group=tp_group
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual: torch.Tensor | None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        mlp_out = self.mlp(hidden_states)
        return mlp_out, residual


class Qwen2MoeModel(nn.Module):
    def __init__(self, config: Qwen2MoeConfig, tp_group=None, ep_group=None, moe_backend: str = "fused") -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, tp_group=tp_group
        )
        self.layers = nn.ModuleList(
            [
                Qwen2MoeDecoderLayer(config, layer_idx, tp_group, ep_group, moe_backend=moe_backend)
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


class Qwen2MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self, 
        config: Qwen2MoeConfig, 
        tp_group: Optional[dist.ProcessGroup] = None, 
        ep_group: Optional[dist.ProcessGroup] = None,
        moe_backend: str = "fused",
        **kwargs 
    ) -> None:
        super().__init__()
        
        self.tp_group = tp_group
        self.ep_group = ep_group

        # 直接把 Runner 传进来的 Group 往下传递
        self.model = Qwen2MoeModel(
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
