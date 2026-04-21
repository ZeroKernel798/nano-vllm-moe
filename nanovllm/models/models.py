from .llama import LlamaForCausalLM
from .qwen2 import Qwen2ForCausalLM
from .qwen2_fp8 import Qwen2ForCausalLMFP8
from .qwen2_int8 import Qwen2ForCausalLMInt8
from .qwen3 import Qwen3ForCausalLM
from .qwen2_moe import Qwen2MoeForCausalLM
from .qwen3_moe import Qwen3MoeForCausalLM

model_dict = {
    "llama": LlamaForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    # FP8 variants
    "qwen2_fp8_w8a16": Qwen2ForCausalLMFP8,
    "qwen2_fp8_w8a8_static": Qwen2ForCausalLMFP8,   # scheme detected from config
    # INT8 variants (scheme detected from quantization_type in config)
    "qwen2_int8_w8a16": Qwen2ForCausalLMInt8,
    "qwen2_int8_w8a8": Qwen2ForCausalLMInt8,
    "qwen2_int8_w8a8_static": Qwen2ForCausalLMInt8,
    "qwen2_moe": Qwen2MoeForCausalLM,
    "qwen3": Qwen3ForCausalLM,
    "qwen3_moe": Qwen3MoeForCausalLM,
}
