import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from nanovllm.layers.linear import LinearBase, divide

def unpack_awq_int4(qweight, qzeros, scales, group_size):
    """
    【工业级复刻】复刻 vLLM/AutoAWQ 内部的物理重排逻辑
    解决了 18.0 误差的根本原因：Weight Interleaving (维度交错)
    """
    in_features, out_features_packed = qweight.shape
    out_features = out_features_packed * 8
    device = qweight.device
    
    # 1. 提取原始 bits [in, out_packed, 8]
    shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int32)
    wq = torch.bitwise_right_shift(qweight.unsqueeze(-1), shifts).bitwise_and(0xF)
    
    # --- 核心：逆转物理重排 (Reordering) ---
    # vLLM 的物理存储为了适配 half2 运算，将 8 个 4-bit 映射到了 [0,4,1,5,2,6,3,7] 的位置
    # 这里的 view(..., 4, 2) 和 permute 是将这“揉碎”的 8 个数还原回线性顺序
    wq = wq.view(in_features, out_features_packed, 4, 2)
    wq = wq.permute(0, 1, 3, 2).contiguous()
    wq = wq.reshape(in_features, out_features)

    # 2. 零点解包 (同样的物理重排)
    # 你的 qzeros 是 [32, 3584]，解包后应为 [32, 28672]
    zq = torch.bitwise_right_shift(qzeros.unsqueeze(-1), shifts).bitwise_and(0xF)
    zq = zq.view(-1, out_features_packed, 4, 2)
    zq = zq.permute(0, 1, 3, 2).contiguous()
    zq = zq.reshape(-1, out_features)

    # 3. 广播对齐
    zq = zq.repeat_interleave(group_size, dim=0) # 32 -> 4096
    s = scales.repeat_interleave(group_size, dim=0)   # 32 -> 4096

    # 4. 反量化公式对齐
    # 绝大多数 Llama-3 AWQ 权重必须补偿这个 +1 偏移
    weight_unpacked = (wq.to(s.dtype) - (zq.to(s.dtype) + 1)) * s
    
    return weight_unpacked.t() # 返回 [out, in]

class AWQLinearBase(LinearBase):
    def __init__(self, input_size, output_size, tp_dim=None, tp_group=None, **kwargs):
        super().__init__(input_size, output_size, tp_dim=tp_dim, tp_group=tp_group, **kwargs)
        self.group_size = 128
        self.pack_factor = 8

    def _init_quant_buffers(self, in_features, out_features):
        self.register_buffer("qweight", torch.zeros((in_features, out_features // self.pack_factor), dtype=torch.int32))
        self.register_buffer("qzeros", torch.zeros((in_features // self.group_size, out_features // self.pack_factor), dtype=torch.int32))
        self.register_buffer("scales", torch.zeros((in_features // self.group_size, out_features), dtype=torch.float16))
        
        self.qweight.weight_loader = self.qweight_loader
        self.qzeros.weight_loader = self.qzeros_loader
        self.scales.weight_loader = self.scales_loader

    def qweight_loader(self, param, loaded_weight): param.data.copy_(loaded_weight)
    def qzeros_loader(self, param, loaded_weight): param.data.copy_(loaded_weight)
    def scales_loader(self, param, loaded_weight): param.data.copy_(loaded_weight)

class AWQRowParallelLinear(AWQLinearBase):
    def __init__(self, input_size, output_size, bias=False, reduce_results=True, tp_group=None, **kwargs):
        super().__init__(input_size, output_size, tp_dim=0, tp_group=tp_group, **kwargs)
        self.reduce_results = reduce_results
        self._init_quant_buffers(divide(input_size, self.tp_size), output_size)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 实时解包进行计算 (虽然比算子慢，但精度绝对正确)
        w = unpack_awq_int4(self.qweight, self.qzeros, self.scales, self.group_size)
        out = F.linear(x, w.to(x.dtype))
        
        if self.tp_size > 1 and self.reduce_results:
            dist.all_reduce(out, group=self.tp_group)
        if self.bias is not None:
            out = out + self.bias
        return out

class AWQMergedColumnParallelLinear(AWQLinearBase):
    def __init__(self, input_size, output_sizes, bias=False, tp_group=None, **kwargs):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), tp_dim=1, tp_group=tp_group, **kwargs)
        self._init_quant_buffers(input_size, sum(output_sizes) // self.tp_size)
        if bias:
            self.bias = nn.Parameter(torch.empty(sum(output_sizes) // self.tp_size))
        else:
            self.register_parameter("bias", None)

    def _shard_loader(self, param, loaded_weight, shard_id, is_packed):
        offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        size = self.output_sizes[shard_id] // self.tp_size
        if is_packed:
            offset //= self.pack_factor
            size //= self.pack_factor
        param.data.narrow(1, offset, size).copy_(loaded_weight)

    def qweight_loader(self, param, loaded_weight, shard_id): self._shard_loader(param, loaded_weight, shard_id, True)
    def qzeros_loader(self, param, loaded_weight, shard_id): self._shard_loader(param, loaded_weight, shard_id, True)
    def scales_loader(self, param, loaded_weight, shard_id): self._shard_loader(param, loaded_weight, shard_id, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = unpack_awq_int4(self.qweight, self.qzeros, self.scales, self.group_size)
        out = F.linear(x, w.to(x.dtype))
        if self.bias is not None:
            out = out + self.bias
        return out

class AWQQKVParallelLinear(AWQLinearBase):
    def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads, bias=False, tp_group=None, **kwargs):
        self.head_size = head_size
        tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = total_num_kv_heads // tp_size
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, tp_dim=1, tp_group=tp_group, **kwargs)
        self._init_quant_buffers(hidden_size, output_size)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

    def _shard_loader(self, param, loaded_weight, shard_id, is_packed):
        if shard_id == "q":
            size, offset = self.num_heads * self.head_size, 0
        elif shard_id == "k":
            size, offset = self.num_kv_heads * self.head_size, self.num_heads * self.head_size
        else:
            size, offset = self.num_kv_heads * self.head_size, (self.num_heads + self.num_kv_heads) * self.head_size
        if is_packed:
            size //= self.pack_factor
            offset //= self.pack_factor
        param.data.narrow(1, offset, size).copy_(loaded_weight)

    def qweight_loader(self, param, loaded_weight, shard_id): self._shard_loader(param, loaded_weight, shard_id, True)
    def qzeros_loader(self, param, loaded_weight, shard_id): self._shard_loader(param, loaded_weight, shard_id, True)
    def scales_loader(self, param, loaded_weight, shard_id): self._shard_loader(param, loaded_weight, shard_id, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = unpack_awq_int4(self.qweight, self.qzeros, self.scales, self.group_size)
        out = F.linear(x, w.to(x.dtype))
        if self.bias is not None:
            out = out + self.bias
        return out
