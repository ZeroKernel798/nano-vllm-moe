# import torch
# import torch.distributed as dist
# from torch import nn
# from transformers import LlamaConfig
# from typing import Optional

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



# class LlamaAttention(nn.Module):

#     def __init__(
#         self,
#         config: LlamaConfig,
#         hidden_size: int,
#         num_heads: int,
#         num_kv_heads: int,
#         rope_theta: float = 10000,
#         rope_scaling: tuple | None = None,
#         max_position_embeddings: int = 8192,
#         bias: bool = False,
#         bias_o_proj: bool = False,
#     ) -> None:
#         super().__init__()
#         self.hidden_size = hidden_size
#         tp_size = dist.get_world_size()
#         self.total_num_heads = num_heads
#         assert self.total_num_heads % tp_size == 0
#         self.num_heads = self.total_num_heads // tp_size
#         self.total_num_kv_heads = num_kv_heads
#         assert self.total_num_kv_heads % tp_size == 0
#         self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
#         # MistralConfig has an optional head_dim introduced by Mistral-Nemo
#         head_dim = getattr(config, "head_dim", None)
#         if head_dim is None:
#             head_dim = self.hidden_size // self.total_num_heads
#         self.head_dim = head_dim
#         self.q_size = self.num_heads * self.head_dim
#         self.kv_size = self.num_kv_heads * self.head_dim
#         self.scaling = self.head_dim**-0.5
#         self.rope_theta = rope_theta
#         self.max_position_embeddings = max_position_embeddings

#         self.qkv_proj = QKVParallelLinear(
#             hidden_size=hidden_size,
#             head_size=self.head_dim,
#             total_num_heads=self.total_num_heads,
#             total_num_kv_heads=self.total_num_kv_heads,
#             bias=bias,
#         )

#         self.o_proj = RowParallelLinear(
#             input_size=self.total_num_heads * self.head_dim,
#             output_size=hidden_size,
#             bias=bias_o_proj,
#         )

#         self.rotary_emb = get_rope(
#             self.head_dim,
#             rotary_dim=self.head_dim,
#             max_position=self.max_position_embeddings,
#             base=self.rope_theta,
#             rope_scaling=rope_scaling,
#         )

#         self.attn = Attention(
#             self.num_heads,
#             self.head_dim,
#             self.scaling,
#             self.num_kv_heads,
#         )

#     def forward(
#         self,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#     ) -> torch.Tensor:
#         qkv = self.qkv_proj(hidden_states)
#         q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
#         q, k = self.rotary_emb(positions, q, k)
#         o = self.attn(q, k, v)
#         output = self.o_proj(o)
#         return output


# class LlamaMLP(nn.Module):

#     def __init__(
#         self,
#         hidden_size: int,
#         intermediate_size: int,
#         hidden_act: str,
#         bias: bool = False,
#     ) -> None:
#         super().__init__()
#         self.gate_up_proj = MergedColumnParallelLinear(
#             input_size=hidden_size,
#             output_sizes=[intermediate_size] * 2,
#             bias=bias,
#         )
#         self.down_proj = RowParallelLinear(
#             input_size=intermediate_size,
#             output_size=hidden_size,
#             bias=bias,
#         )
#         assert hidden_act == "silu"
#         self.act_fn = SiluAndMul()

#     def forward(self, x):
#         x = self.gate_up_proj(x)
#         x = self.act_fn(x)
#         x = self.down_proj(x)
#         return x


# class LlamaDecoderLayer(nn.Module):

#     def __init__(
#         self,
#         config: LlamaConfig,
#     ) -> None:
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         rope_theta = getattr(config, "rope_theta", 10000)
#         rope_scaling = getattr(config, "rope_scaling", None)
#         max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
#         # Support abacusai/Smaug-72B-v0.1 with attention_bias
#         # Support internlm/internlm-7b with bias
#         attention_bias = getattr(config, "attention_bias", False) or getattr(
#             config, "bias", False
#         )
#         bias_o_proj = attention_bias
#         # support internlm/internlm3-8b with qkv_bias
#         if hasattr(config, "qkv_bias"):
#             attention_bias = config.qkv_bias

#         self.self_attn = LlamaAttention(
#             config=config,
#             hidden_size=self.hidden_size,
#             num_heads=config.num_attention_heads,
#             num_kv_heads=getattr(
#                 config, "num_key_value_heads", config.num_attention_heads
#             ),
#             rope_theta=rope_theta,
#             # rope_scaling=rope_scaling,
#             rope_scaling=None,
#             max_position_embeddings=max_position_embeddings,
#             bias=attention_bias,
#             bias_o_proj=bias_o_proj,
#         )
#         self.mlp = LlamaMLP(
#             hidden_size=self.hidden_size,
#             intermediate_size=config.intermediate_size,
#             hidden_act=config.hidden_act,
#             bias=getattr(config, "mlp_bias", False),
#         )
#         self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = RMSNorm(
#             config.hidden_size, eps=config.rms_norm_eps
#         )

#     def forward(
#         self,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#         residual: torch.Tensor | None,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         # Self Attention
#         if residual is None:
#             residual = hidden_states
#             hidden_states = self.input_layernorm(hidden_states)
#         else:
#             hidden_states, residual = self.input_layernorm(hidden_states, residual)
#         hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

#         # Fully Connected
#         hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
#         hidden_states = self.mlp(hidden_states)
#         return hidden_states, residual


# class LlamaModel(nn.Module):

#     def __init__(
#         self,
#         config: LlamaConfig,
#     ) -> None:
#         super().__init__()
#         self.embed_tokens = VocabParallelEmbedding(
#             config.vocab_size, config.hidden_size
#         )
#         self.layers = nn.ModuleList(
#             [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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


# class LlamaForCausalLM(nn.Module):
#     packed_modules_mapping = {
#         "q_proj": ("qkv_proj", "q"),
#         "k_proj": ("qkv_proj", "k"),
#         "v_proj": ("qkv_proj", "v"),
#         "gate_proj": ("gate_up_proj", 0),
#         "up_proj": ("gate_up_proj", 1),
#     }

#     def __init__(
#         self, 
#         config: LlamaConfig,
#         tp_group: Optional[dist.ProcessGroup] = None, 
#         ep_group: Optional[dist.ProcessGroup] = None,
#     ) -> None:
#         super().__init__()
#         self.model = LlamaModel(config)
#         self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
#         if config.tie_word_embeddings:
#             self.lm_head.weight.data = self.model.embed_tokens.weight.data

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#     ) -> torch.Tensor:
#         model_output = self.model(input_ids, positions)
#         return model_output

#     def compute_logits(
#         self,
#         hidden_states: torch.Tensor,
#     ) -> torch.Tensor:
#         logits = self.lm_head(hidden_states)
#         return logits
#   print(f"🔥 DEBUG ATTENTION: quant_config = {getattr(config, 'quantization_config', 'NONE')}")

import torch
import torch.distributed as dist
from torch import nn
from transformers import LlamaConfig
from typing import Optional

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

# 引入我们刚刚写好的高性能 FP8 算子
from nanovllm.layers.fp8_linear import (
    FP8QKVParallelLinear,
    FP8MergedColumnParallelLinear,
    FP8RowParallelLinear,
)

from nanovllm.layers.smooth_quant_linear import (
    Int8QKVParallelLinear,
    Int8MergedColumnParallelLinear,
    Int8RowParallelLinear,
)

class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, hidden_size: int, num_heads: int, num_kv_heads: int, rope_theta: float = 10000, rope_scaling: tuple | None = None, max_position_embeddings: int = 8192, bias: bool = False, bias_o_proj: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        # 核心改动：极简配置路由，只判断是否为 FP8
        # is_fp8 = getattr(config, "quantization_config", {}).get("quant_method") == "fp8"

        is_fp8 = getattr(config, "quantization_config", {}).get("quant_method") == "fp8"
        is_smoothquant = getattr(config, "quantization_config", {}).get("quant_method") == "smoothquant"
        print(f"🔥 DEBUG ATTENTION: quant_config = {getattr(config, 'quantization_config', 'NONE')}, is_smoothquant = {is_smoothquant}")

        if is_fp8:
            self.qkv_proj = FP8QKVParallelLinear(hidden_size=hidden_size, head_size=self.head_dim, total_num_heads=self.total_num_heads, total_num_kv_heads=self.total_num_kv_heads, bias=bias)
            self.o_proj = FP8RowParallelLinear(input_size=self.total_num_heads * self.head_dim, output_size=hidden_size, bias=bias_o_proj)
        elif is_smoothquant: # 👈 添加 INT8 分支
            self.qkv_proj = Int8QKVParallelLinear(hidden_size=hidden_size, head_size=self.head_dim, total_num_heads=self.total_num_heads, total_num_kv_heads=self.total_num_kv_heads, bias=bias)
            self.o_proj = Int8RowParallelLinear(input_size=self.total_num_heads * self.head_dim, output_size=hidden_size, bias=bias_o_proj)
        else:
            self.qkv_proj = QKVParallelLinear(hidden_size=hidden_size, head_size=self.head_dim, total_num_heads=self.total_num_heads, total_num_kv_heads=self.total_num_kv_heads, bias=bias)
            self.o_proj = RowParallelLinear(input_size=self.total_num_heads * self.head_dim, output_size=hidden_size, bias=bias_o_proj)

        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=self.rope_theta, rope_scaling=rope_scaling)
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, bias: bool = False, config=None) -> None:
        super().__init__()

        # 同样极简的 FP8 路由
        is_fp8 = getattr(config, "quantization_config", {}).get("quant_method") == "fp8"
        is_smoothquant = getattr(config, "quantization_config", {}).get("quant_method") == "smoothquant"
        
        if is_fp8:
            self.gate_up_proj = FP8MergedColumnParallelLinear(input_size=hidden_size, output_sizes=[intermediate_size] * 2, bias=bias)
            self.down_proj = FP8RowParallelLinear(input_size=intermediate_size, output_size=hidden_size, bias=bias)
        elif is_smoothquant: # 👈 添加 INT8 分支
            self.gate_up_proj = Int8MergedColumnParallelLinear(input_size=hidden_size, output_sizes=[intermediate_size] * 2, bias=bias)
            self.down_proj = Int8RowParallelLinear(input_size=intermediate_size, output_size=hidden_size, bias=bias)
        else:
            self.gate_up_proj = MergedColumnParallelLinear(input_size=hidden_size, output_sizes=[intermediate_size] * 2, bias=bias)
            self.down_proj = RowParallelLinear(input_size=intermediate_size, output_size=hidden_size, bias=bias)
            
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)
        
        self.self_attn = LlamaAttention(config=config, hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads), rope_theta=getattr(config, "rope_theta", 10000), max_position_embeddings=getattr(config, "max_position_embeddings", 8192), bias=attention_bias, bias_o_proj=attention_bias)
        self.mlp = LlamaMLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, bias=getattr(config, "mlp_bias", False), config=config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    # 打包映射表保持不变，用于正确路由权重到各个层
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: LlamaConfig, tp_group=None, ep_group=None) -> None:
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

# import torch
# import torch.distributed as dist
# from torch import nn
# from transformers import LlamaConfig
# from typing import Optional

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

# # 引入封装好的 W8A8 静态量化算子
# from nanovllm.layers.fp8_linear import (
#     FP8QKVParallelLinear,
#     FP8MergedColumnParallelLinear,
#     FP8RowParallelLinear,
# )

# class LlamaAttention(nn.Module):
#     def __init__(self, config: LlamaConfig, hidden_size: int, num_heads: int, num_kv_heads: int, rope_theta: float = 10000, rope_scaling: tuple | None = None, max_position_embeddings: int = 8192, bias: bool = False, bias_o_proj: bool = False) -> None:
#         super().__init__()
#         self.hidden_size = hidden_size
#         tp_size = dist.get_world_size() if dist.is_initialized() else 1
#         self.total_num_heads = num_heads
#         self.num_heads = self.total_num_heads // tp_size
#         self.total_num_kv_heads = num_kv_heads
#         self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
#         self.head_dim = getattr(config, "head_dim", self.hidden_size // self.total_num_heads)
#         self.q_size = self.num_heads * self.head_dim
#         self.kv_size = self.num_kv_heads * self.head_dim
#         self.scaling = self.head_dim**-0.5
#         self.rope_theta = rope_theta

#         # 核心改动：精准抓取 FP8 配置。兼容 "fp8" 或 NVIDIA 的 "modelopt" 标记
#         quant_config = getattr(config, "quantization_config", {})
#         quant_method = quant_config.get("quant_method", "")
#         is_fp8 = quant_method in ["fp8", "modelopt", "fp8_static"]
        
#         print(f"🔥 DEBUG ATTENTION: quant_method = {quant_method}, is_fp8 = {is_fp8}")

#         if is_fp8:
#             # W8A8 算子实例化（内部包含 weight_scale 和 input_scale 的处理）
#             self.qkv_proj = FP8QKVParallelLinear(hidden_size=hidden_size, head_size=self.head_dim, total_num_heads=self.total_num_heads, total_num_kv_heads=self.total_num_kv_heads, bias=bias)
#             self.o_proj = FP8RowParallelLinear(input_size=self.total_num_heads * self.head_dim, output_size=hidden_size, bias=bias_o_proj)
#         else:
#             self.qkv_proj = QKVParallelLinear(hidden_size=hidden_size, head_size=self.head_dim, total_num_heads=self.total_num_heads, total_num_kv_heads=self.total_num_kv_heads, bias=bias)
#             self.o_proj = RowParallelLinear(input_size=self.total_num_heads * self.head_dim, output_size=hidden_size, bias=bias_o_proj)

#         self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=self.rope_theta, rope_scaling=rope_scaling)
#         self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

#     def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
#         qkv = self.qkv_proj(hidden_states)
#         q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
#         q, k = self.rotary_emb(positions, q, k)
#         o = self.attn(q, k, v)
#         output = self.o_proj(o)
#         return output


# class LlamaMLP(nn.Module):
#     def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, bias: bool = False, config=None) -> None:
#         super().__init__()

#         quant_config = getattr(config, "quantization_config", {})
#         quant_method = quant_config.get("quant_method", "")
#         is_fp8 = quant_method in ["fp8", "modelopt", "fp8_static"]
        
#         if is_fp8:
#             self.gate_up_proj = FP8MergedColumnParallelLinear(input_size=hidden_size, output_sizes=[intermediate_size] * 2, bias=bias)
#             self.down_proj = FP8RowParallelLinear(input_size=intermediate_size, output_size=hidden_size, bias=bias)
#         else:
#             self.gate_up_proj = MergedColumnParallelLinear(input_size=hidden_size, output_sizes=[intermediate_size] * 2, bias=bias)
#             self.down_proj = RowParallelLinear(input_size=intermediate_size, output_size=hidden_size, bias=bias)
            
#         self.act_fn = SiluAndMul()

#     def forward(self, x):
#         x = self.gate_up_proj(x)
#         x = self.act_fn(x)
#         x = self.down_proj(x)
#         return x


# class LlamaDecoderLayer(nn.Module):
#     def __init__(self, config: LlamaConfig) -> None:
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)
        
#         self.self_attn = LlamaAttention(config=config, hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads), rope_theta=getattr(config, "rope_theta", 10000), max_position_embeddings=getattr(config, "max_position_embeddings", 8192), bias=attention_bias, bias_o_proj=attention_bias)
#         self.mlp = LlamaMLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, bias=getattr(config, "mlp_bias", False), config=config)
#         self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
#         if residual is None:
#             residual = hidden_states
#             hidden_states = self.input_layernorm(hidden_states)
#         else:
#             hidden_states, residual = self.input_layernorm(hidden_states, residual)
#         hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

#         hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
#         hidden_states = self.mlp(hidden_states)
#         return hidden_states, residual


# class LlamaModel(nn.Module):
#     def __init__(self, config: LlamaConfig) -> None:
#         super().__init__()
#         self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
#         self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
#         self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.embed_tokens(input_ids)
#         residual = None
#         for layer in self.layers:
#             hidden_states, residual = layer(positions, hidden_states, residual)
#         hidden_states, _ = self.norm(hidden_states, residual)
#         return hidden_states


# class LlamaForCausalLM(nn.Module):
#     # 打包映射表保持不变，用于正确路由权重到各个层
#     packed_modules_mapping = {
#         "q_proj": ("qkv_proj", "q"),
#         "k_proj": ("qkv_proj", "k"),
#         "v_proj": ("qkv_proj", "v"),
#         "gate_proj": ("gate_up_proj", 0),
#         "up_proj": ("gate_up_proj", 1),
#     }

#     def __init__(self, config: LlamaConfig, tp_group=None, ep_group=None) -> None:
#         super().__init__()
#         self.model = LlamaModel(config)
#         self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
#         if config.tie_word_embeddings:
#             self.lm_head.weight.data = self.model.embed_tokens.weight.data

#     def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
#         return self.model(input_ids, positions)

#     def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         return self.lm_head(hidden_states)