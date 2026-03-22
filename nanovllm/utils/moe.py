import torch
def moe_align_block_size(topk_ids, num_experts, block_size, expert_map=None):
    """
    模拟 vLLM 的 C++ moe_align_block_size 逻辑 (完全体：返回 4 个值)
    """
    device = topk_ids.device
    M, top_k = topk_ids.shape
    num_total_tasks = M * top_k
    
    flat_topk_ids = topk_ids.view(-1)
    flat_token_ids = torch.arange(M, device=device).repeat_interleave(top_k)
    # 【核心】为了 W2 Combine Kernel 准备的全局任务索引
    flat_weight_idx = torch.arange(num_total_tasks, device=device)

    tokens_per_expert = torch.bincount(flat_topk_ids, minlength=num_experts)
    num_blocks_per_expert = (tokens_per_expert + block_size - 1) // block_size
    total_blocks = num_blocks_per_expert.sum().item()
    
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=device), 
        num_blocks_per_expert
    ).to(torch.int32)
    
    if expert_map is not None:
        expert_ids = expert_map[expert_ids.long()].to(torch.int32)

    sort_idx = torch.argsort(flat_topk_ids)
    sorted_tokens_raw = flat_token_ids[sort_idx]
    sorted_weight_idx_raw = flat_weight_idx[sort_idx] # 【新增】
    
    token_ranks = torch.arange(num_total_tasks, device=device)
    expert_offsets = torch.cumsum(tokens_per_expert, dim=0) - tokens_per_expert
    rank_within_expert = token_ranks - expert_offsets[flat_topk_ids[sort_idx]]
    
    expert_block_offsets = torch.cumsum(num_blocks_per_expert, dim=0) - num_blocks_per_expert
    padded_target_pos = expert_block_offsets[flat_topk_ids[sort_idx]] * block_size + rank_within_expert
    
    padded_size = total_blocks * block_size
    
    # 记录 Token 映射
    sorted_token_ids = torch.full((padded_size,), M, dtype=torch.int32, device=device)
    sorted_token_ids[padded_target_pos.long()] = sorted_tokens_raw.to(torch.int32)
    
    # 【新增】记录权重映射（对应 flat_routing_weights 的位置）
    sorted_weight_idx = torch.full((padded_size,), num_total_tasks, dtype=torch.int32, device=device)
    sorted_weight_idx[padded_target_pos.long()] = sorted_weight_idx_raw.to(torch.int32)
    
    # 记得返回这 4 个值！
    return sorted_token_ids, sorted_weight_idx, expert_ids, total_blocks