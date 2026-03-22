# import torch
# import torch.distributed as dist
# import torch.nn.functional as F
# from torch import nn
# import triton
# from transformers import Qwen2MoeConfig 

# from nanovllm.layers.activation import SiluAndMul
# from nanovllm.layers.attention import Attention
# from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
# from nanovllm.layers.layernorm import RMSNorm
# from nanovllm.layers.linear import (
#     MergedColumnParallelLinear,
#     QKVParallelLinear,
#     RowParallelLinear,
# )
# from nanovllm.layers.rotary_embedding import get_rope

# # from nanovllm.kernels.group_gemm import moe_gemm_kernel
# from nanovllm.kernels.group_gemm import fused_moe_w13_kernel
# from nanovllm.kernels.group_gemm import fused_moe_w2_combine_kernel
# from nanovllm.utils.moe import moe_align_block_size

# class Qwen2MoeAttention(nn.Module):
#     def __init__(
#         self, 
#         hidden_size: int,
#         num_heads: int,
#         num_kv_heads: int,
#         max_position: int = 4096 * 32,
#         rope_theta: float = 1000000,
#         rope_scaling: tuple | None = None,
#     ) -> None:
#         super().__init__()
#         tp_size = dist.get_world_size()
#         self.hidden_size = hidden_size
#         self.total_num_heads = num_heads
#         self.total_num_kv_heads = num_kv_heads
#         self.head_dim = hidden_size // self.total_num_heads
#         self.num_heads = self.total_num_heads // tp_size
#         self.num_kv_heads = self.total_num_kv_heads // tp_size
        
#         self.q_size = self.num_heads * self.head_dim
#         self.kv_size = self.num_kv_heads * self.head_dim
#         self.scaling = self.head_dim**-0.5

#         self.qkv_proj = QKVParallelLinear(
#             hidden_size,
#             self.head_dim, 
#             self.total_num_heads,
#             self.total_num_kv_heads, 
#             bias=True
#         )

#         self.o_proj = RowParallelLinear(
#             self.total_num_heads * self.head_dim, 
#             hidden_size,
#             bias=False
#         )

#         self.rotary_emb = get_rope(
#             self.head_dim,
#             rotary_dim=self.head_dim,
#             max_position=max_position,
#             base=rope_theta,
#             rope_scaling=rope_scaling,
#         )

#         self.attn = Attention(
#             self.num_heads, 
#             self.head_dim, 
#             self.scaling, 
#             self.num_kv_heads)

#     def forward(
#         self, 
#         positions,
#         hidden_states
#     ) -> torch.Tensor:
#         qkv = self.qkv_proj(hidden_states)
#         q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
#         q, k = self.rotary_emb(positions, q, k)
#         attn_output = self.attn(q, k, v)
#         output = self.o_proj(attn_output)
#         return output
    

# class Qwen2MoeMLP(nn.Module):
#     def __init__(
#         self,
#         hidden_size: int,
#         intermediate_size: int,
#         hidden_act: str,
#         reduce_results=True,
#     ) -> None:
#         super().__init__()
#         self.gate_up_proj = MergedColumnParallelLinear(
#             hidden_size,
#             [intermediate_size] * 2,
#             bias=False,
#         )
#         self.down_proj = RowParallelLinear(
#             intermediate_size,
#             hidden_size,
#             bias=False,
#             reduce_results=reduce_results,
#         )
#         assert hidden_act == "silu"
#         self.act_fn = SiluAndMul()

#     def forward(self, x):
#         gate_up = self.gate_up_proj(x)
#         x = self.act_fn(gate_up)
#         x = self.down_proj(x)
#         return x

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

# # 可切换overlap以及串行逻辑
# # class Qwen2MoeSparseMoeBlock(nn.Module):
# #     def __init__(self, config, use_overlap: bool = True) -> None:
# #         super().__init__()
# #         self.hidden_size = config.hidden_size
# #         self.num_experts = config.num_experts
# #         self.top_k = config.num_experts_per_tok
# #         self.moe_intermediate_size = config.moe_intermediate_size
        
# #         # 方案切换开关
# #         self.use_overlap = use_overlap

# #         self.w13_stacked = nn.Parameter(torch.empty(self.num_experts, 2 * self.moe_intermediate_size, self.hidden_size))
# #         self.w2_stacked = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.moe_intermediate_size))
# #         self.w13_stacked.weight_loader = self.load_moe_weight
# #         self.w2_stacked.weight_loader = self.load_moe_weight
        
# #         self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
# #         self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)
# #         self.shared_expert = Qwen2MoeMLP(
# #             hidden_size=config.hidden_size,
# #             intermediate_size=config.shared_expert_intermediate_size,
# #             hidden_act=config.hidden_act,
# #         )

# #         if self.use_overlap:
# #             self.shared_expert_stream = torch.cuda.Stream()
# #         else:
# #             self.shared_expert_stream = None

# #     def load_moe_weight(self, param, loaded_weight, expert_id, shard_id=None):
# #         with torch.no_grad():
# #             if shard_id is not None:
# #                 offset = shard_id * self.moe_intermediate_size
# #                 param.data[expert_id].narrow(0, offset, self.moe_intermediate_size).copy_(loaded_weight)
# #             else:
# #                 param.data[expert_id].copy_(loaded_weight)

# #     def _compute_shared_expert(self, x):
# #         out = self.shared_expert(x)
# #         out *= torch.sigmoid(self.shared_expert_gate(x))
# #         return out

# #     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
# #         orig_shape = hidden_states.shape
# #         x = hidden_states.view(-1, self.hidden_size).contiguous()
# #         M = x.shape[0]
# #         model_dtype = x.dtype

# #         if self.use_overlap:
# #             # overlap
# #             with torch.cuda.stream(self.shared_expert_stream):
# #                 shared_out = self._compute_shared_expert(x)
# #         else:
# #             shared_out = self._compute_shared_expert(x)

       
# #         router_logits = self.gate(x)
# #         routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
# #         routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
# #         flat_routing_weights = routing_weights.view(-1).contiguous()

# #         BLOCK_SIZE_M = 32
# #         GROUP_SIZE_M = 8 
# #         sorted_token_ids, sorted_weight_idx, expert_ids, num_blocks = moe_align_block_size(
# #             selected_experts, self.num_experts, BLOCK_SIZE_M
# #         )
        
# #         inter_size = self.moe_intermediate_size
# #         num_total_tasks = M * self.top_k
# #         activated_out = torch.zeros((num_total_tasks, inter_size), device=x.device, dtype=model_dtype)

# #         grid_w13 = lambda META: (num_blocks * triton.cdiv(inter_size, META['BLOCK_SIZE_N']),)
# #         fused_moe_w13_kernel[grid_w13](
# #             x, self.w13_stacked, activated_out,
# #             sorted_token_ids, sorted_weight_idx, expert_ids,
# #             num_blocks, M, inter_size * 2, self.hidden_size, 
# #             x.stride(0), x.stride(1),
# #             self.w13_stacked.stride(0), self.w13_stacked.stride(2), self.w13_stacked.stride(1),
# #             activated_out.stride(0), activated_out.stride(1),
# #             BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, 
# #             GROUP_SIZE_M=GROUP_SIZE_M, 
# #         )

# #         combined_sparse_fp32 = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
# #         grid_w2 = lambda META: (num_blocks * triton.cdiv(self.hidden_size, META['BLOCK_SIZE_N']),)
        
# #         fused_moe_w2_combine_kernel[grid_w2](
# #             activated_out, self.w2_stacked, combined_sparse_fp32, flat_routing_weights,
# #             sorted_token_ids, sorted_weight_idx, expert_ids,
# #             num_blocks, M, self.hidden_size, inter_size,
# #             activated_out.stride(0), activated_out.stride(1),
# #             self.w2_stacked.stride(0), self.w2_stacked.stride(2), self.w2_stacked.stride(1),
# #             combined_sparse_fp32.stride(0), combined_sparse_fp32.stride(1),
# #             BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, 
# #             GROUP_SIZE_M=GROUP_SIZE_M,
# #         )

# #         if self.use_overlap:
# #             torch.cuda.current_stream().wait_stream(self.shared_expert_stream)
        
# #         return (shared_out + combined_sparse_fp32.to(model_dtype)).view(orig_shape)

# # tp ep
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist
# from typing import Optional, List, Tuple
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist
# from typing import Optional
# import triton

# # 假设这些已在你的环境下正确导入
# # from nanovllm.kernels.group_gemm import fused_moe_w13_kernel, fused_moe_w2_combine_kernel
# # from nanovllm.utils.moe import moe_align_block_size

# class Qwen2MoeSparseMoeBlock(nn.Module):
#     def __init__(
#         self, 
#         config, 
#         tp_group: Optional[dist.ProcessGroup] = None,
#         ep_group: Optional[dist.ProcessGroup] = None,
#         use_overlap: bool = True,
#     ) -> None:
#         super().__init__()
        
#         # --- 1. 基础属性赋值 (必须在 Parameter 定义之前) ---
#         self.hidden_size = config.hidden_size
#         self.global_num_experts = config.num_experts
#         self.top_k = config.num_experts_per_tok
#         self.moe_intermediate_size = config.moe_intermediate_size
        
#         # --- 2. 硬编码并行参数 (2卡测试: TP=1, EP=2) ---
#         self.tp_size = 1
#         self.ep_size = 2
#         global_rank = dist.get_rank() if dist.is_initialized() else 0
#         self.tp_rank = global_rank % self.tp_size
#         self.ep_rank = global_rank // self.tp_size
        
#         self.tp_group = tp_group
#         self.ep_group = ep_group
        
#         # 每一卡负责的规模
#         self.local_num_experts = self.global_num_experts // self.ep_size
#         self.local_inter_size = self.moe_intermediate_size // self.tp_size

#         # --- 3. 权重初始化 (Stacked 格式) ---
#         self.w13_stacked = nn.Parameter(torch.zeros(
#             self.local_num_experts, 2 * self.local_inter_size, self.hidden_size
#         ))
#         self.w2_stacked = nn.Parameter(torch.zeros(
#             self.local_num_experts, self.hidden_size, self.local_inter_size
#         ))
        
#         # 绑定 Loader
#         self.w13_stacked.weight_loader = self.load_hybrid_moe_weight
#         self.w2_stacked.weight_loader = self.load_hybrid_moe_weight

#         # Gate 层
#         self.gate = nn.Linear(self.hidden_size, self.global_num_experts, bias=False)
#         self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)
#         self.gate.weight.weight_loader = self.load_replicated_weight
#         self.shared_expert_gate.weight.weight_loader = self.load_replicated_weight

#         # 共享专家
#         self.shared_expert = Qwen2MoeMLP(
#             hidden_size=config.hidden_size,
#             intermediate_size=config.shared_expert_intermediate_size,
#             hidden_act=config.hidden_act,
#             reduce_results=False 
#         )

#         self.use_overlap = use_overlap
#         self.shared_expert_stream = torch.cuda.Stream() if use_overlap else None

#     # --- 辅助函数：权重加载 ---
#     def load_replicated_weight(self, param, loaded_weight, **kwargs):
#         with torch.no_grad():
#             param.copy_(loaded_weight)

#     def load_hybrid_moe_weight(self, param, loaded_weight, global_expert_id, shard_id=None, **kwargs):
#         with torch.no_grad():
#             if not (self.ep_rank * self.local_num_experts <= global_expert_id < (self.ep_rank + 1) * self.local_num_experts):
#                 return 

#             local_id = global_expert_id % self.local_num_experts
#             start = self.tp_rank * self.local_inter_size
#             size = self.local_inter_size

#             if shard_id in [0, "w1", "gate_proj"]: 
#                 param.data[local_id].narrow(0, 0, size).copy_(loaded_weight.narrow(0, start, size))
#             elif shard_id in [1, "w3", "up_proj"]: 
#                 param.data[local_id].narrow(0, size, size).copy_(loaded_weight.narrow(0, start, size))
#             elif shard_id in [None, "w2", "down_proj"]: 
#                 param.data[local_id].copy_(loaded_weight.narrow(1, start, size))

#     def _compute_shared_expert(self, x):
#         out = self.shared_expert(x)
#         out *= torch.sigmoid(self.shared_expert_gate(x))
#         return out

#     # --- 核心：前向传播 ---
#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         orig_shape = hidden_states.shape
#         x = hidden_states.view(-1, self.hidden_size).contiguous()
#         M, H = x.shape
#         model_dtype = x.dtype

#         # 1. 共享专家 (保持不变)
#         shared_out = self._compute_shared_expert(x)

#         # 2. Gating
#         router_logits = self.gate(x)
#         routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
#         topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)

#         # ---------------------------------------------------------
#         # 核心逻辑：如果只有 1 个 EP Rank，跳过通信，直接本地算
#         # ---------------------------------------------------------
#         if self.ep_size <= 1:
#             # 本地直接准备 Triton 参数
#             recv_x = x.repeat_interleave(self.top_k, dim=0)
#             recv_local_ids = topk_ids.flatten()
#             recv_weights = topk_weights.flatten()
#             num_recv = recv_x.shape[0]
            
#             # 为了让代码复用，我们伪造一个全 0 的排序索引
#             permute_indices = torch.arange(num_recv, device=x.device)
#             s_list = [num_recv]
#             r_list = [num_recv]
#         else:
#             # 3. 之前的 EP Dispatch 准备 (EP > 1 时才执行)
#             target_ep_ranks = torch.div(topk_ids, self.local_num_experts, rounding_mode='floor').clamp(0, self.ep_size - 1)
#             flat_target_ep_ranks = target_ep_ranks.flatten()
#             permute_indices = torch.argsort(flat_target_ep_ranks)
            
#             expanded_x = x.repeat_interleave(self.top_k, dim=0) 
#             dispatched_x = expanded_x[permute_indices]
            
#             send_counts = torch.bincount(flat_target_ep_ranks, minlength=self.ep_size)
#             send_counts = send_counts.to(torch.long).to(x.device)
#             recv_counts = torch.empty_like(send_counts)
#             dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

#             s_list, r_list = send_counts.tolist(), recv_counts.tolist()
#             num_recv = recv_counts.sum().item()
            
#             recv_x = torch.empty(num_recv, H, device=x.device, dtype=x.dtype)
#             recv_local_ids = torch.empty(num_recv, dtype=torch.long, device=x.device)
#             recv_weights = torch.empty(num_recv, dtype=torch.float32, device=x.device)

#             dist.all_to_all_single(recv_x, dispatched_x, r_list, s_list, group=self.ep_group)
#             dist.all_to_all_single(recv_local_ids, (topk_ids % self.local_num_experts).flatten()[permute_indices], r_list, s_list, group=self.ep_group)
#             dist.all_to_all_single(recv_weights, topk_weights.flatten()[permute_indices], r_list, s_list, group=self.ep_group)

#         # 5. 本地 Triton 计算 (无论 EP 是多少，这部分逻辑通用)
#         BLOCK_SIZE_M, GROUP_SIZE_M = 32, 8
#         sorted_token_ids, sorted_weight_idx, expert_ids, num_blocks = moe_align_block_size(
#             recv_local_ids.view(-1, 1), self.local_num_experts, BLOCK_SIZE_M
#         )
        
#         # W13
#         activated_out = torch.empty((num_recv, self.local_inter_size), device=x.device, dtype=model_dtype)
#         grid_w13 = lambda META: (num_blocks * triton.cdiv(self.local_inter_size, META['BLOCK_SIZE_N']),)
#         fused_moe_w13_kernel[grid_w13](
#             recv_x, self.w13_stacked, activated_out,
#             sorted_token_ids, sorted_weight_idx, expert_ids,
#             num_blocks, num_recv, self.local_inter_size * 2, self.hidden_size, 
#             recv_x.stride(0), recv_x.stride(1),
#             self.w13_stacked.stride(0), self.w13_stacked.stride(2), self.w13_stacked.stride(1),
#             activated_out.stride(0), activated_out.stride(1),
#             BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=GROUP_SIZE_M, 
#         )

#         # W2
#         local_out_fp32 = torch.zeros((num_recv, self.hidden_size), device=x.device, dtype=torch.float32)
#         grid_w2 = lambda META: (num_blocks * triton.cdiv(self.hidden_size, META['BLOCK_SIZE_N']),)
#         fused_moe_w2_combine_kernel[grid_w2](
#             activated_out, self.w2_stacked, local_out_fp32, recv_weights,
#             sorted_token_ids, sorted_weight_idx, expert_ids,
#             num_blocks, num_recv, self.hidden_size, self.local_inter_size,
#             activated_out.stride(0), activated_out.stride(1),
#             self.w2_stacked.stride(0), self.w2_stacked.stride(2), self.w2_stacked.stride(1),
#             local_out_fp32.stride(0), local_out_fp32.stride(1),
#             BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=GROUP_SIZE_M,
#         )

#         # 6. Combine (EP > 1 时才发回)
#         if self.ep_size > 1:
#             combined_x = torch.empty(M * self.top_k, H, device=x.device, dtype=model_dtype)
#             dist.all_to_all_single(combined_x, local_out_fp32.to(model_dtype), s_list, r_list, group=self.ep_group)
#         else:
#             combined_x = local_out_fp32.to(model_dtype)
        
#         # 7. 还原顺序并求和
#         sparse_out_flat = torch.zeros((M * self.top_k, H), device=x.device, dtype=model_dtype)
#         sparse_out_flat[permute_indices] = combined_x
#         sparse_out = sparse_out_flat.view(M, self.top_k, H).sum(dim=1)

#         # 8. 合并与 TP Reduce
#         output = shared_out + sparse_out
#         if self.tp_size > 1:
#             dist.all_reduce(output, group=self.tp_group)

#         return output.view(orig_shape)

# class Qwen2MoeDecoderLayer(nn.Module):
#     def __init__(
#         self, 
#         config: Qwen2MoeConfig, 
#         layer_idx: int
#     ) -> None:
#         super().__init__()
#         self.self_attn = Qwen2MoeAttention(
#             hidden_size=config.hidden_size,
#             num_heads=config.num_attention_heads,
#             num_kv_heads=config.num_key_value_heads,
#             max_position=config.max_position_embeddings,
#             rope_theta=getattr(config, "rope_theta", 1000000),
#             rope_scaling = None,
#         )
#         mlp_only_layers = getattr(config, "mlp_only_layers", [])
#         if (layer_idx not in mlp_only_layers) and (
#             config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
#         ):
#             self.mlp = Qwen2MoeSparseMoeBlock(config=config)
#         else:
#             self.mlp = Qwen2MoeMLP(
#                 hidden_size=config.hidden_size,
#                 intermediate_size=config.intermediate_size,
#                 hidden_act=config.hidden_act,
#             )

#         self.input_layernorm = RMSNorm(
#             config.hidden_size, 
#             eps=config.rms_norm_eps
#         )
        
#         self.post_attention_layernorm = RMSNorm(
#             config.hidden_size, 
#             eps=config.rms_norm_eps
#         )

#     def forward(
#         self, 
#         positions, 
#         hidden_states, 
#         residual: torch.Tensor | None,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         if residual is None:
#             residual = hidden_states
#             hidden_states = self.input_layernorm(hidden_states)
#         else:
#             hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
#         hidden_states = self.self_attn(positions, hidden_states)
#         hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
#         hidden_states = self.mlp(hidden_states)

#         return hidden_states, residual


# class Qwen2MoeModel(nn.Module):
#     def __init__(
#         self,
#         config: Qwen2MoeConfig,
#     ) -> None:
#         super().__init__()
#         self.embed_tokens = VocabParallelEmbedding(
#             config.vocab_size, config.hidden_size
#         )
#         self.layers = nn.ModuleList(
#             [
#                 Qwen2MoeDecoderLayer(config, layer_idx)
#                 for layer_idx in range(config.num_hidden_layers)
#             ]
#         )
#         self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#     ) -> torch.Tensor:
#         hidden_states = self.embed_tokens(input_ids)
#         residual = None
#         for layer in self.layers:
#             hidden_states, residual = layer(positions, hidden_states, residual)
#         hidden_states, _ = self.norm(hidden_states, residual)
#         return hidden_states


# class Qwen2MoeForCausalLM(nn.Module):
#     packed_modules_mapping = {
#         "q_proj": ("qkv_proj", "q"),
#         "k_proj": ("qkv_proj", "k"),
#         "v_proj": ("qkv_proj", "v"),
#         "gate_proj": ("gate_up_proj", 0),
#         "up_proj": ("gate_up_proj", 1),
#     }

#     def __init__(self, config: Qwen2MoeConfig) -> None:
#         super().__init__()
#         self.model = Qwen2MoeModel(config)
#         self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
#         if config.tie_word_embeddings:
#             self.lm_head.weight.data = self.model.embed_tokens.weight.data

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#     ) -> torch.Tensor:
#         hidden_states = self.model(input_ids, positions)
#         return hidden_states

#     def compute_logits(
#         self,
#         hidden_states: torch.Tensor,
#     ) -> torch.Tensor:
#         logits = self.lm_head(hidden_states)
#         return logits


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
from nanovllm.kernels.group_gemm import fused_moe_w13_kernel, fused_moe_w2_combine_kernel
from nanovllm.utils.moe import moe_align_block_size
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
    

class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(
        self, 
        config, 
        tp_group: Optional[dist.ProcessGroup] = None,
        ep_group: Optional[dist.ProcessGroup] = None,
        use_overlap: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.global_num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.use_overlap = use_overlap
        
        self.tp_group = tp_group
        self.ep_group = ep_group
        
        self.tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        self.ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
        self.ep_rank = dist.get_rank(ep_group) if ep_group is not None else 0
        
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

        self.gate = nn.Linear(self.hidden_size, self.global_num_experts, bias=False)
        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)
        self.gate.weight.weight_loader = self.load_replicated_weight
        self.shared_expert_gate.weight.weight_loader = self.load_replicated_weight

        self.shared_expert = Qwen2MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            reduce_results=False,
            tp_group=tp_group
        )

        self.shared_expert_stream = torch.cuda.Stream() if use_overlap else None

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

    def _compute_shared_expert(self, x):
        out = self.shared_expert(x)
        out *= torch.sigmoid(self.shared_expert_gate(x))
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_size).contiguous()
        M, H = x.shape
        model_dtype = x.dtype

        if self.use_overlap:
            self.shared_expert_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.shared_expert_stream):
                shared_out = self._compute_shared_expert(x)
        else:
            shared_out = self._compute_shared_expert(x)

        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.ep_size <= 1:
            recv_x = x.repeat_interleave(self.top_k, dim=0)
            recv_local_ids = topk_ids.flatten()
            recv_weights = topk_weights.flatten()
            num_recv = recv_x.shape[0]
            permute_indices = torch.arange(num_recv, device=x.device)
            s_list = [num_recv]
            r_list = [num_recv]
        else:
            target_ep_ranks = torch.div(topk_ids, self.local_num_experts, rounding_mode='floor').clamp(0, self.ep_size - 1)
            flat_target_ep_ranks = target_ep_ranks.flatten()
            permute_indices = torch.argsort(flat_target_ep_ranks)
            
            expanded_x = x.repeat_interleave(self.top_k, dim=0) 
            dispatched_x = expanded_x[permute_indices]
            
            send_counts = torch.bincount(flat_target_ep_ranks, minlength=self.ep_size).to(torch.long).to(x.device)
            recv_counts = torch.empty_like(send_counts)
            dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

            s_list, r_list = send_counts.tolist(), recv_counts.tolist()
            num_recv = recv_counts.sum().item()
            
            recv_x = torch.empty(num_recv, H, device=x.device, dtype=x.dtype)
            recv_local_ids = torch.empty(num_recv, dtype=torch.long, device=x.device)
            recv_weights = torch.empty(num_recv, dtype=torch.float32, device=x.device)

            dist.all_to_all_single(recv_x, dispatched_x, r_list, s_list, group=self.ep_group)
            dist.all_to_all_single(recv_local_ids, (topk_ids % self.local_num_experts).flatten()[permute_indices], r_list, s_list, group=self.ep_group)
            dist.all_to_all_single(recv_weights, topk_weights.flatten()[permute_indices], r_list, s_list, group=self.ep_group)

        BLOCK_SIZE_M, GROUP_SIZE_M = 32, 8 
        sorted_token_ids, sorted_weight_idx, expert_ids, num_blocks = moe_align_block_size(
            recv_local_ids.view(-1, 1), self.local_num_experts, BLOCK_SIZE_M
        )
        
        activated_out = torch.empty((num_recv, self.local_inter_size), device=x.device, dtype=model_dtype)
        grid_w13 = lambda META: (num_blocks * triton.cdiv(self.local_inter_size, META['BLOCK_SIZE_N']),)
        fused_moe_w13_kernel[grid_w13](
            recv_x, self.w13_stacked, activated_out,
            sorted_token_ids, sorted_weight_idx, expert_ids,
            num_blocks, num_recv, self.local_inter_size * 2, self.hidden_size, 
            recv_x.stride(0), recv_x.stride(1),
            self.w13_stacked.stride(0), self.w13_stacked.stride(2), self.w13_stacked.stride(1),
            activated_out.stride(0), activated_out.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=GROUP_SIZE_M, 
        )

        local_out_fp32 = torch.zeros((num_recv, self.hidden_size), device=x.device, dtype=torch.float32)
        grid_w2 = lambda META: (num_blocks * triton.cdiv(self.hidden_size, META['BLOCK_SIZE_N']),)
        fused_moe_w2_combine_kernel[grid_w2](
            activated_out, self.w2_stacked, local_out_fp32, recv_weights,
            sorted_token_ids, sorted_weight_idx, expert_ids,
            num_blocks, num_recv, self.hidden_size, self.local_inter_size,
            activated_out.stride(0), activated_out.stride(1),
            self.w2_stacked.stride(0), self.w2_stacked.stride(2), self.w2_stacked.stride(1),
            local_out_fp32.stride(0), local_out_fp32.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=GROUP_SIZE_M,
        )

        if self.ep_size > 1:
            combined_x = torch.empty(M * self.top_k, H, device=x.device, dtype=model_dtype)
            dist.all_to_all_single(combined_x, local_out_fp32.to(model_dtype), s_list, r_list, group=self.ep_group)
        else:
            combined_x = local_out_fp32.to(model_dtype)
        
        sparse_out_flat = torch.zeros((M * self.top_k, H), device=x.device, dtype=model_dtype)
        sparse_out_flat[permute_indices] = combined_x
        sparse_out = sparse_out_flat.view(M, self.top_k, H).sum(dim=1)

        if self.use_overlap:
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
        group_gemm_enable : bool = False
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
                self.mlp = Qwen2MoeSparseMoeBlock(config=config, tp_group=tp_group, ep_group=ep_group)
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
    def __init__(self, config: Qwen2MoeConfig, tp_group=None, ep_group=None) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, tp_group=tp_group
        )
        self.layers = nn.ModuleList(
            [
                Qwen2MoeDecoderLayer(config, layer_idx, tp_group, ep_group)
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
        **kwargs 
    ) -> None:
        super().__init__()
        
        self.tp_group = tp_group
        self.ep_group = ep_group

        # 直接把 Runner 传进来的 Group 往下传递
        self.model = Qwen2MoeModel(config, tp_group=self.tp_group, ep_group=self.ep_group)
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