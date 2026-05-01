from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.distributed as dist


@dataclass(frozen=True)
class MoEParallelConfig:
    tp_group: Optional[dist.ProcessGroup]
    ep_group: Optional[dist.ProcessGroup]
    tp_size: int
    tp_rank: int
    ep_size: int
    ep_rank: int
    global_num_experts: int
    local_num_experts: int
    intermediate_size: int
    local_inter_size: int


def make_moe_parallel_config(
    *,
    tp_group: Optional[dist.ProcessGroup],
    ep_group: Optional[dist.ProcessGroup],
    global_num_experts: int,
    intermediate_size: int,
) -> MoEParallelConfig:
    tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
    ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
    ep_rank = dist.get_rank(ep_group) if ep_group is not None else 0
    if global_num_experts % ep_size != 0:
        raise ValueError(
            f"global_num_experts={global_num_experts} must be divisible by ep_size={ep_size}"
        )
    if intermediate_size % tp_size != 0:
        raise ValueError(
            f"intermediate_size={intermediate_size} must be divisible by tp_size={tp_size}"
        )
    return MoEParallelConfig(
        tp_group=tp_group,
        ep_group=ep_group,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        global_num_experts=global_num_experts,
        local_num_experts=global_num_experts // ep_size,
        intermediate_size=intermediate_size,
        local_inter_size=intermediate_size // tp_size,
    )
