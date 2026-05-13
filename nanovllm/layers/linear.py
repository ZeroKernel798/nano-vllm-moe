import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from typing import Optional

from nanovllm.quantization.base_config import QuantizeMethodBase


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
        tp_group: Optional[dist.ProcessGroup] = None, 
        quant_method: QuantizeMethodBase | None = None,
        **kwargs, 
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_group = tp_group
        self.quant_method = quant_method
        
        # 所有的 rank 和 size 都必须从传入的 tp_group 获取
        if tp_group is not None:
            self.tp_size = dist.get_world_size(tp_group)
            self.tp_rank = dist.get_rank(tp_group)
        else:
            self.tp_size = dist.get_world_size() if dist.is_initialized() else 1
            self.tp_rank = dist.get_rank() if dist.is_initialized() else 0

    def is_quantized(self) -> bool:
        return self.quant_method is not None

    def apply_linear(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(self, x, bias)
        return F.linear(x, self.weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_group: Optional[dist.ProcessGroup] = None,
        **kwargs,
    ):
        super().__init__(input_size, output_size, tp_group=tp_group, **kwargs)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_group: Optional[dist.ProcessGroup] = None,
        quant_method: QuantizeMethodBase | None = None,
        **kwargs,
    ):
        super().__init__(input_size, output_size, 0, tp_group=tp_group, quant_method=quant_method, **kwargs)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        if self.quant_method is None:
            self.weight = nn.Parameter(
                torch.empty(self.output_size_per_partition, self.input_size)
            )
            self.weight.weight_loader = self.weight_loader
        else:
            self.quant_method.create_weights(
                self,
                input_size_per_partition=self.input_size,
                output_partition_sizes=[self.output_size_per_partition],
                input_size=self.input_size,
                output_size=self.output_size,
                params_dtype=torch.get_default_dtype(),
            )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_linear(x, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        tp_group: Optional[dist.ProcessGroup] = None,
        quant_method: QuantizeMethodBase | None = None,
        **kwargs,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias=bias, tp_group=tp_group, quant_method=quant_method, **kwargs)
        if self.quant_method is not None:
            self.output_partition_sizes = [size // self.tp_size for size in output_sizes]

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def get_quant_output_offset_size(self, loaded_shard_id: int) -> tuple[int, int]:
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        return shard_offset, shard_size


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        tp_group: Optional[dist.ProcessGroup] = None,
        quant_method: QuantizeMethodBase | None = None,
        **kwargs,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        
        # 从组里取
        tp_size = dist.get_world_size(tp_group) if tp_group is not None else (dist.get_world_size() if dist.is_initialized() else 1)
        
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        output_size = (
            self.total_num_heads + 2 * self.total_num_kv_heads
        ) * self.head_size
        
        super().__init__(input_size, output_size, bias=bias, tp_group=tp_group, quant_method=quant_method, **kwargs)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def get_quant_output_offset_size(self, loaded_shard_id: str) -> tuple[int, int]:
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        return shard_offset, shard_size


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        reduce_results=True,
        tp_group: Optional[dist.ProcessGroup] = None,
        quant_method: QuantizeMethodBase | None = None,
        **kwargs,
    ):
        super().__init__(input_size, output_size, 1, tp_group=tp_group, quant_method=quant_method, **kwargs)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.reduce_results = reduce_results

        if self.quant_method is None:
            self.weight = nn.Parameter(
                torch.empty(self.output_size, self.input_size_per_partition)
            )
            self.weight.weight_loader = self.weight_loader
        else:
            self.quant_method.create_weights(
                self,
                input_size_per_partition=self.input_size_per_partition,
                output_partition_sizes=[self.output_size],
                input_size=self.input_size,
                output_size=self.output_size,
                params_dtype=torch.get_default_dtype(),
            )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.apply_linear(x, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1 and self.reduce_results:
            dist.all_reduce(y, group=self.tp_group)
        return y
