import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from typing import Optional

from transformers import Qwen3MoeConfig 
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
from nanovllm.executor.moe.backends import TritonMoEBackend

class Qwen3MoeAttention(nn.Module):
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
        # 使用传入的 tp_group 大小
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
            tp_group=tp_group
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            tp_group=tp_group
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
        
        # Qwen3 特有的 QK Norm
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
        output = self.o_proj(o)
        return output


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
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config,
        tp_group: Optional[dist.ProcessGroup] = None,
        ep_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.global_num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        
        self.tp_group = tp_group
        self.ep_group = ep_group
        
        self.tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        self.ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
        self.ep_rank = dist.get_rank(ep_group) if ep_group is not None else 0
        self.backend = TritonMoEBackend(tp_group=tp_group, ep_group=ep_group)
        
        self.local_num_experts = self.global_num_experts // self.ep_size
        self.local_inter_size = self.moe_intermediate_size // self.tp_size

        self.w13_stacked = nn.Parameter(torch.zeros(
            self.local_num_experts, 2 * self.local_inter_size, self.hidden_size
        ))
        self.w2_stacked = nn.Parameter(torch.zeros(
            self.local_num_experts, self.hidden_size, self.local_inter_size
        ))
        self.w13_stacked.weight_loader = self.load_hybrid_moe_weight
        self.w2_stacked.weight_loader = self.load_hybrid_moe_weight

        # Gate 层
        self.gate = nn.Linear(self.hidden_size, self.global_num_experts, bias=False)
        self.gate.weight.weight_loader = self.load_replicated_weight

        # 注意：Qwen3 不含 Shared Expert，无需创建 Shared Expert 和对应的 CUDA Stream

    def load_replicated_weight(self, param, loaded_weight, **kwargs):
        with torch.no_grad():
            param.copy_(loaded_weight)

    def load_hybrid_moe_weight(self, param, loaded_weight, global_expert_id, shard_id=None, **kwargs):
        with torch.no_grad():
            if not (self.ep_rank * self.local_num_experts <= global_expert_id < (self.ep_rank + 1) * self.local_num_experts):
                return 
            local_id = global_expert_id % self.local_num_experts
            start = self.tp_rank * self.local_inter_size
            size = self.local_inter_size
            if shard_id in [0, "w1", "gate_proj"]: 
                param.data[local_id].narrow(0, 0, size).copy_(loaded_weight.narrow(0, start, size))
            elif shard_id in [1, "w3", "up_proj"]: 
                param.data[local_id].narrow(0, size, size).copy_(loaded_weight.narrow(0, start, size))
            elif shard_id in [None, "w2", "down_proj"]: 
                param.data[local_id].copy_(loaded_weight.narrow(1, start, size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_size).contiguous()
        M, H = x.shape
        model_dtype = x.dtype

        # 1. 路由计算
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # 【核心差异】：Qwen3 必须对 topk 的权重进行归一化
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        # 强制转换为 float32 传入 Triton 算子，防止精度丢失
        topk_weights = topk_weights.to(torch.float32)

        # 2. EP Token 分发 (All-To-All)
        dispatch_state = self.backend.dispatch(
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            local_num_experts=self.local_num_experts,
            top_k=self.top_k,
        )
        recv_x = dispatch_state["recv_x"]
        recv_local_ids = dispatch_state["recv_local_ids"]
        recv_weights = dispatch_state["recv_weights"]
        permute_indices = dispatch_state["permute_indices"]
        s_list = dispatch_state["s_list"]
        r_list = dispatch_state["r_list"]

        # 3. 执行 Triton Group-GEMM
        local_out_fp32 = self.backend.compute(
            recv_x=recv_x,
            recv_local_ids=recv_local_ids,
            recv_weights=recv_weights,
            w13_stacked=self.w13_stacked,
            w2_stacked=self.w2_stacked,
            local_num_experts=self.local_num_experts,
            local_inter_size=self.local_inter_size,
            hidden_size=self.hidden_size,
            model_dtype=model_dtype,
        )

        # 4/5/6. 回收 + 规约
        output = self.backend.combine(
            local_out_fp32=local_out_fp32,
            model_dtype=model_dtype,
            m_tokens=M,
            hidden_size=H,
            top_k=self.top_k,
            permute_indices=permute_indices,
            s_list=s_list,
            r_list=r_list,
        )
        return output.view(orig_shape)


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int = -1,
        tp_group: Optional[dist.ProcessGroup] = None,
        ep_group: Optional[dist.ProcessGroup] = None,
        group_gemm_enable: bool = True
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
                self.mlp = Qwen3MoeSparseMoeBlock(config=config, tp_group=tp_group, ep_group=ep_group)
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
    def __init__(self, config, tp_group=None, ep_group=None) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, tp_group=tp_group
        )
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, layer_idx, tp_group, ep_group)
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
        **kwargs 
    ) -> None:
        super().__init__()
        self.tp_group = tp_group
        self.ep_group = ep_group

        self.model = Qwen3MoeModel(config, tp_group=self.tp_group, ep_group=self.ep_group)
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