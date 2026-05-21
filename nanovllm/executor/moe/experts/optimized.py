from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton

from nanovllm.executor.moe.experts.base import MoEExpertsKernel
from nanovllm.kernels.optimized_moe import (
    launch_fused_moe_kernel,
    launch_moe_sum_reduce,
    launch_token_moe_graph_kernel,
)
from nanovllm.utils.moe import moe_align_block_size, moe_align_block_size_fixed


def _default_config(m_tokens: int, num_experts: int) -> dict[str, int]:
    if m_tokens <= num_experts:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }


@dataclass
class _GraphWorkspace:
    """Pre-allocated persistent buffers for CUDA-graph-safe MoE forward.

    All shapes are derived from the largest captured ``num_tasks`` and never
    grow at replay time, so no allocator activity happens on the hot path.
    """

    num_tasks: int
    num_experts: int
    local_inter_size: int
    hidden_size: int
    block_size_m: int

    num_tokens_post_padded: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    gate_up: torch.Tensor
    activated: torch.Tensor
    expert_out: torch.Tensor
    output: torch.Tensor

    @classmethod
    def create(
        cls,
        num_tasks: int,
        num_experts: int,
        local_inter_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "_GraphWorkspace":
        config = _default_config(num_tasks, num_experts)
        block_size_m = config["BLOCK_SIZE_M"]
        # Fixed-capacity alignment reserves the same number of blocks for every
        # expert, so total_blocks is purely a function of (num_tasks, num_experts,
        # block_size_m) and never depends on the routing decision.
        max_blocks_per_expert = (num_tasks + block_size_m - 1) // block_size_m
        total_blocks = num_experts * max_blocks_per_expert
        padded_size = total_blocks * block_size_m
        return cls(
            num_tasks=num_tasks,
            num_experts=num_experts,
            local_inter_size=local_inter_size,
            hidden_size=hidden_size,
            block_size_m=block_size_m,
            num_tokens_post_padded=torch.full(
                (1,), padded_size, dtype=torch.int32, device=device
            ),
            sorted_token_ids=torch.empty(padded_size, dtype=torch.int32, device=device),
            expert_ids=torch.empty(total_blocks, dtype=torch.int32, device=device),
            gate_up=torch.empty(
                (num_tasks, 1, 2 * local_inter_size), dtype=dtype, device=device
            ),
            activated=torch.empty(
                (num_tasks, local_inter_size), dtype=dtype, device=device
            ),
            expert_out=torch.empty(
                (num_tasks, 1, hidden_size), dtype=dtype, device=device
            ),
            output=torch.empty((num_tasks, hidden_size), dtype=dtype, device=device),
        )

    def matches(
        self,
        num_tasks: int,
        num_experts: int,
        local_inter_size: int,
        hidden_size: int,
    ) -> bool:
        return (
            self.num_tasks == num_tasks
            and self.num_experts == num_experts
            and self.local_inter_size == local_inter_size
            and self.hidden_size == hidden_size
        )


class OptimizedExperts(MoEExpertsKernel):
    def __init__(self) -> None:
        # Keyed by num_tasks so multi-bs CUDA Graph capture (graphs for
        # bs=1,2,4,8,16,...) each have their own persistent buffers and never
        # allocate inside ``torch.cuda.graph(...)``.
        self._graph_workspaces: dict[int, _GraphWorkspace] = {}

    def _get_workspace(
        self,
        num_tasks: int,
        num_experts: int,
        local_inter_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _GraphWorkspace:
        ws = self._graph_workspaces.get(num_tasks)
        if ws is None or not ws.matches(num_tasks, num_experts, local_inter_size, hidden_size):
            ws = _GraphWorkspace.create(
                num_tasks=num_tasks,
                num_experts=num_experts,
                local_inter_size=local_inter_size,
                hidden_size=hidden_size,
                dtype=dtype,
                device=device,
            )
            self._graph_workspaces[num_tasks] = ws
        return ws

    def apply(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        *,
        local_num_experts: int,
        local_inter_size: int,
        hidden_size: int,
        model_dtype: torch.dtype,
        w13_weight_scale: torch.Tensor | None = None,
        w2_weight_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hidden_size, w13_weight_scale, w2_weight_scale
        if x.device.type != "cuda":
            return self._fallback_eager(x, topk_ids, topk_weights, w13, w2, local_num_experts)
        is_capturing = torch.cuda.is_current_stream_capturing()
        if (
            is_capturing
            and os.environ.get("NANOVLLM_MOE_GRAPH_TOKEN_KERNEL", "0") == "1"
        ):
            output = torch.empty((x.shape[0], w2.shape[1]), device=x.device, dtype=model_dtype)
            launch_token_moe_graph_kernel(
                x,
                topk_ids,
                topk_weights,
                w13,
                w2,
                output,
                intermediate_size=local_inter_size,
            )
            return output

        num_tasks = x.shape[0]
        top_k = 1
        topk_ids_2d = topk_ids.view(num_tasks, top_k).to(torch.int64)

        # The workspace + fixed-align path is the only one that survives CUDA
        # Graph capture. We use it whenever we are either actively capturing OR
        # the env flag is set; ``capture_cudagraph`` sets the env flag during
        # both warmup and capture so the workspace is allocated by the warmup
        # call (regular allocator, stable address) and reused by the capture
        # replay — no allocation ever happens inside ``torch.cuda.graph(...)``.
        use_workspace = (
            is_capturing
            or os.environ.get("NANOVLLM_MOE_GRAPH_ALIGN", "0") == "1"
        )
        align = moe_align_block_size_fixed if use_workspace else moe_align_block_size
        sorted_token_ids, _sorted_weight_idx, expert_ids, num_blocks = align(
            topk_ids_2d, local_num_experts, _default_config(num_tasks, local_num_experts)["BLOCK_SIZE_M"]
        )
        config = _default_config(num_tasks, local_num_experts)
        block_size_m = config["BLOCK_SIZE_M"]
        sorted_token_ids = sorted_token_ids.to(torch.int32)
        expert_ids = expert_ids.to(torch.int32)

        if use_workspace:
            workspace = self._get_workspace(
                num_tasks=num_tasks,
                num_experts=local_num_experts,
                local_inter_size=local_inter_size,
                hidden_size=w2.shape[1],
                dtype=model_dtype,
                device=x.device,
            )
            # Workspace was sized with the same fixed-capacity formula; the
            # alignment outputs above must have identical shapes.
            workspace.sorted_token_ids.copy_(sorted_token_ids)
            workspace.expert_ids.copy_(expert_ids)
            sorted_token_ids = workspace.sorted_token_ids
            expert_ids = workspace.expert_ids
            num_tokens_post_padded = workspace.num_tokens_post_padded
            gate_up = workspace.gate_up
            expert_out = workspace.expert_out
            output = workspace.output
        else:
            num_tokens_post_padded = torch.tensor(
                [num_blocks * block_size_m], device=x.device, dtype=torch.int32
            )
            gate_up = torch.empty(
                (num_tasks, top_k, 2 * local_inter_size), device=x.device, dtype=model_dtype
            )
            expert_out = torch.empty(
                (num_tasks, top_k, w2.shape[1]), device=x.device, dtype=model_dtype
            )
            output = torch.empty(
                (num_tasks, w2.shape[1]), device=x.device, dtype=model_dtype
            )

        launch_fused_moe_kernel(
            x,
            w13,
            gate_up,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight=False,
            top_k=top_k,
            config=config,
            compute_type=model_dtype,
        )

        gate_view = gate_up.view(num_tasks * top_k, 2 * local_inter_size)
        if use_workspace:
            activated = workspace.activated
            torch.mul(F.silu(gate_view[..., :local_inter_size]), gate_view[..., local_inter_size:], out=activated)
        else:
            activated = F.silu(gate_view[..., :local_inter_size]) * gate_view[..., local_inter_size:]
            activated = activated.contiguous()

        launch_fused_moe_kernel(
            activated,
            w2,
            expert_out,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight=True,
            top_k=top_k,
            config=config,
            compute_type=model_dtype,
        )

        launch_moe_sum_reduce(expert_out, output)
        return output

    @staticmethod
    def _fallback_eager(
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        local_num_experts: int,
    ) -> torch.Tensor:
        outputs = []
        for token, expert_id, weight in zip(x, topk_ids, topk_weights, strict=True):
            expert = int(expert_id.item())
            if expert < 0 or expert >= local_num_experts:
                raise IndexError(f"local expert id {expert} out of range [0, {local_num_experts})")
            gate_up = F.linear(token.unsqueeze(0), w13[expert])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_out = F.linear(F.silu(gate) * up, w2[expert]).squeeze(0)
            outputs.append(expert_out * weight.to(expert_out.dtype))
        if not outputs:
            return torch.empty(x.shape[0], w2.shape[1], device=x.device, dtype=x.dtype)
        return torch.stack(outputs, dim=0)
