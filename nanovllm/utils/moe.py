import torch

# 辅助函数：严格的每专家隔离 Padding
def moe_align_helper(selected_experts, BLOCK_SIZE_M, num_experts):
    device = selected_experts.device
    flat_expert_ids = selected_experts.reshape(-1)
    
    sorted_indices = torch.argsort(flat_expert_ids)
    sorted_experts = flat_expert_ids[sorted_indices]
    
    tokens_per_expert = torch.bincount(flat_expert_ids, minlength=num_experts)
    expert_blocks = (tokens_per_expert + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    #  计算每个 Token 在 padding 后数组中的绝对安全位置
    expert_starts = torch.cumsum(tokens_per_expert, dim=0) - tokens_per_expert
    token_positions = torch.arange(len(sorted_indices), device=device) - expert_starts[sorted_experts]
    
    block_in_expert = token_positions // BLOCK_SIZE_M
    pos_in_block = token_positions % BLOCK_SIZE_M
    expert_block_starts = torch.cumsum(expert_blocks, dim=0) - expert_blocks
    global_block_id = expert_block_starts[sorted_experts] + block_in_expert
    
    # padded_positions 就是有效 Token 应该放入的坑位，中间会留出 -1 的空隙给 Padding
    padded_positions = global_block_id * BLOCK_SIZE_M + pos_in_block
    
    total_blocks = expert_blocks.sum().item()
    padded_size = total_blocks * BLOCK_SIZE_M
    
    aligned_token_ids = torch.full((padded_size,), -1, dtype=torch.int32, device=device)
    original_token_ids = (sorted_indices // selected_experts.shape[1]).to(torch.int32)
    aligned_token_ids[padded_positions] = original_token_ids
    
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=device), expert_blocks
    ).to(torch.int32)
    
    return aligned_token_ids, expert_ids, total_blocks, padded_positions, original_token_ids, sorted_indices