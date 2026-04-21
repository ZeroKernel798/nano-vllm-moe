import os
from dataclasses import dataclass

from transformers import AutoConfig

from nanovllm.utils.kv_cache import assert_kv_cache_runtime_supported, normalize_kv_cache_dtype


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.7
    tp_size: int = 1
    ep_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    #: Logical KV storage dtype: ``"bf16"`` (default) or ``"fp8_e4m3"`` (planned; see ``utils/kv_cache``).
    kv_cache_dtype: str = "bf16"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tp_size <= 8
        assert 1 <= self.ep_size <= 8
        self.kv_cache_dtype = normalize_kv_cache_dtype(self.kv_cache_dtype)
        assert_kv_cache_runtime_supported(self.kv_cache_dtype)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len
