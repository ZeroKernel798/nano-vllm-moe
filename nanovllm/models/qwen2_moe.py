import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
import triton
from transformers import Qwen2MoeConfig 

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

# from nanovllm.kernels.group_gemm import moe_gemm_kernel
from nanovllm.kernels.group_gemm import fused_moe_w13_kernel
from nanovllm.kernels.group_gemm import fused_moe_w2_combine_kernel
from nanovllm.utils.moe import moe_align_block_size

class Qwen2MoeAttention(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 1000000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // self.total_num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim, 
            self.total_num_heads,
            self.total_num_kv_heads, 
            bias=True
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, 
            hidden_size,
            bias=False
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
            self.num_kv_heads)

    def forward(
        self, 
        positions,
        hidden_states
    ) -> torch.Tensor:
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
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x

# class Qwen2MoeSparseMoeBlock(nn.Module):
#     def __init__(
#         self,
#         config: Qwen2MoeConfig
#     ) -> None:
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         self.hidden_act = config.hidden_act

#         self.num_experts = config.num_experts
#         self.top_k = config.num_experts_per_tok

#         self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

#         self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)

#         self.shared_expert = Qwen2MoeMLP(
#             hidden_size=config.hidden_size,
#             intermediate_size=config.shared_expert_intermediate_size,
#             hidden_act=config.hidden_act,
#         )

#         self.experts = nn.ModuleList(
#         [
#             Qwen2MoeMLP(
#                 hidden_size=config.hidden_size,
#                 intermediate_size=config.moe_intermediate_size,
#                 hidden_act=config.hidden_act,
#             ) for _ in range(self.num_experts)
#         ]
#     )

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         orig_shape = hidden_states.shape
#         hidden_states = hidden_states.view(-1, self.hidden_size)

#         # 计算路由的logits
#         router_logits = self.gate(hidden_states)
#         routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
#         routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
#         # routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
#         routing_weights = routing_weights.to(hidden_states.dtype)

#         # 共享专家
#         shared_output = self.shared_expert(hidden_states)
#         shared_weight = torch.sigmoid(self.shared_expert_gate(hidden_states))
#         shared_output = shared_output * shared_weight

#         # 稀疏专家
#         sparse_hidden_states = torch.zeros(
#             hidden_states.shape,
#             dtype=hidden_states.dtype,
#             device=hidden_states.device,
#         )
#         expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

#         for i in range(self.num_experts):
#             idx, top_x = torch.where(expert_mask[i])
#             if top_x.shape[0] == 0:
#                 continue
            
#             expert_out = self.experts[i](hidden_states[top_x])
#             sparse_hidden_states.index_add_(0, top_x, expert_out * routing_weights[top_x, idx, None])

#         return (shared_output + sparse_hidden_states).view(orig_shape)

class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size

        self.w13_stacked = nn.Parameter(torch.empty(self.num_experts, 2 * self.moe_intermediate_size, self.hidden_size))
        self.w2_stacked = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.moe_intermediate_size))
        self.w13_stacked.weight_loader = self.load_moe_weight
        self.w2_stacked.weight_loader = self.load_moe_weight

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)
        self.shared_expert = Qwen2MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
        )

    def load_moe_weight(self, param, loaded_weight, expert_id, shard_id=None):
        with torch.no_grad():
            if shard_id is not None:
                offset = shard_id * self.moe_intermediate_size
                param.data[expert_id].narrow(0, offset, self.moe_intermediate_size).copy_(loaded_weight)
            else:
                param.data[expert_id].copy_(loaded_weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_size).contiguous()
        model_dtype = x.dtype
        M = x.shape[0]

        shared_output = self.shared_expert(x)
        shared_weight = torch.sigmoid(self.shared_expert_gate(x))
        shared_output = shared_output * shared_weight

        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        flat_routing_weights = routing_weights.view(-1).contiguous()

        BLOCK_SIZE_M = 32
        GROUP_SIZE_M = 8 
        
        # 这里的 num_blocks 是对齐后的 Block 总数
        sorted_token_ids, sorted_weight_idx, expert_ids, num_blocks = moe_align_block_size(
            selected_experts, self.num_experts, BLOCK_SIZE_M
        )
        
        # 对齐后的总任务数 (包含 Padding)
        EM = num_blocks * BLOCK_SIZE_M 

        inter_size = self.moe_intermediate_size
        num_total_tasks = M * self.top_k
        activated_out = torch.zeros((num_total_tasks, inter_size), device=x.device, dtype=model_dtype)

        # 这里的 Grid 计算逻辑要考虑到 N 维的分块
        grid_w13 = lambda META: (num_blocks * triton.cdiv(inter_size, META['BLOCK_SIZE_N']),)
        
        fused_moe_w13_kernel[grid_w13](
            x, self.w13_stacked, activated_out,
            sorted_token_ids, sorted_weight_idx, expert_ids,
            num_blocks,      # <--- 对应 Kernel 的 num_blocks
            M,               # <--- 对应 Kernel 的 num_valid_tokens
            inter_size * 2,  # N
            self.hidden_size,# K
            x.stride(0), x.stride(1),
            self.w13_stacked.stride(0), self.w13_stacked.stride(2), self.w13_stacked.stride(1),
            activated_out.stride(0), activated_out.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M, 
            BLOCK_SIZE_N=64, 
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=GROUP_SIZE_M, 
        )

        combined_sparse_fp32 = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        
        grid_w2 = lambda META: (num_blocks * triton.cdiv(self.hidden_size, META['BLOCK_SIZE_N']),)
        
        fused_moe_w2_combine_kernel[grid_w2](
            activated_out, self.w2_stacked, combined_sparse_fp32, flat_routing_weights,
            sorted_token_ids, sorted_weight_idx, expert_ids,
            num_blocks,      # <--- 对应 Kernel 的 num_blocks
            M,               # <--- 对应 Kernel 的 num_valid_tokens
            self.hidden_size,# N
            inter_size,      # K
            activated_out.stride(0), activated_out.stride(1),
            self.w2_stacked.stride(0), self.w2_stacked.stride(2), self.w2_stacked.stride(1),
            combined_sparse_fp32.stride(0), combined_sparse_fp32.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M, 
            BLOCK_SIZE_N=64, 
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=GROUP_SIZE_M,
        )

        return (shared_output + combined_sparse_fp32.to(model_dtype)).view(orig_shape)

class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(
        self, 
        config: Qwen2MoeConfig, 
        layer_idx: int
    ) -> None:
        super().__init__()
        self.self_attn = Qwen2MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling = None,
        )
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen2MoeSparseMoeBlock(config=config)
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )

        self.input_layernorm = RMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps
        )
        
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps
        )

    def forward(
        self, 
        positions, 
        hidden_states, 
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen2MoeModel(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Qwen2MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
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

    def __init__(self, config: Qwen2MoeConfig) -> None:
        super().__init__()
        self.model = Qwen2MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
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