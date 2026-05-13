import os
from dataclasses import dataclass

from transformers import AutoConfig

from nanovllm.utils.kv_cache import assert_kv_cache_runtime_supported, normalize_kv_cache_dtype, normalize_kv_cache_scale_dtype


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
    experimental_kv_cache_fp8: bool = False
    kv_cache_scale_dtype: str = "float16"
    moe_backend: str = "fused"
    moe_ep_backend: str = "torch"
    chunked_prefill_policy: str = "prefill_first"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tp_size <= 8
        assert 1 <= self.ep_size <= 8
        if self.moe_ep_backend not in {"torch"}:
            raise ValueError("moe_ep_backend must be 'torch'")
        if self.chunked_prefill_policy not in {"prefill_first", "decode_first"}:
            raise ValueError("chunked_prefill_policy must be one of {'prefill_first', 'decode_first'}")
        if self.ep_size > 1 and not self.enforce_eager:
            print(
                "[nanovllm] EP dynamic all-to-all does not support CUDA graph capture yet; "
                "forcing enforce_eager=True."
            )
            self.enforce_eager = True
        self.kv_cache_dtype = normalize_kv_cache_dtype(self.kv_cache_dtype)
        self.kv_cache_scale_dtype = normalize_kv_cache_scale_dtype(self.kv_cache_scale_dtype)
        assert_kv_cache_runtime_supported(self.kv_cache_dtype, self.experimental_kv_cache_fp8)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.max_model_len <= 0:
            self.max_model_len = self.hf_config.max_position_embeddings
        else:
            self.max_model_len = min(
                self.max_model_len, self.hf_config.max_position_embeddings
            )
