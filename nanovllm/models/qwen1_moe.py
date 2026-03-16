import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2MoeConfig # Qwen1.5 MoE 通常也使用这个 Config 类
from typing import Optional, Tuple

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

# --- 1. MLP 模块 (共享专家和稀疏专家共用) ---
class Qwen15MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        return self.down_proj(x)

# --- 2. Qwen1.5 MoE 核心 Block ---
class Qwen15MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen2MoeConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        # 路由层
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # 稀疏专家列表
        self.experts = nn.ModuleList([
            Qwen15MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
            ) for _ in range(self.num_experts)
        ])

        # 共享专家
        self.shared_expert = Qwen15MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
        )
        
        # ✅ 添加这一行，匹配权重文件里的 shared_expert_gate
        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        
        # 1. 路由计算
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 2. 共享专家分支 (带上门控权重)
        shared_output = self.shared_expert(hidden_states)
        # ✅ 这里的计算逻辑也要带上 sigmoid
        shared_weight = torch.sigmoid(self.shared_expert_gate(hidden_states))
        shared_output = shared_output * shared_weight

        # 3. 稀疏专家分支 (index_add)
        sparse_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for i in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[i])
            if top_x.shape[0] == 0:
                continue
            
            expert_out = self.experts[i](hidden_states[top_x])
            sparse_hidden_states.index_add_(0, top_x, expert_out * routing_weights[top_x, idx, None])

        # 4. 合并结果
        return (shared_output + sparse_hidden_states).view(orig_shape)
    
# --- 3. Attention 模块 (保持 Qwen2 风格) ---
class Qwen15MoeAttention(nn.Module):
    def __init__(self, config: Qwen2MoeConfig) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, self.head_dim, 
            config.num_attention_heads, config.num_key_value_heads, bias=True
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.rotary_emb = get_rope(
            self.head_dim, 
            rotary_dim=self.head_dim, 
            max_position=config.max_position_embeddings, 
            base=getattr(config, "rope_theta", 1000000.0)
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(self, positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v))

# --- 4. 解码层与模型封装 ---
class Qwen15MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2MoeConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen15MoeAttention(config)
        
        # Qwen1.5 MoE 的逻辑：所有层通常都是 MoE 结构
        self.mlp = Qwen15MoeSparseMoeBlock(config)

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

class Qwen15MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen2MoeConfig):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.model.layers = nn.ModuleList([Qwen15MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.model.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(self, input_ids, positions):
        hidden_states = self.model.embed_tokens(input_ids)
        residual = None
        for layer in self.model.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.model.norm(hidden_states, residual)
        return hidden_states

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)