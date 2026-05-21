import pickle
import inspect
import os
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.utils.kv_cache import (
    k_int8_group_size_from_env,
    k_int8_recent_bf16_window_from_env,
    kv_cache_scale_dtype_bytes_per_element,
    normalize_kv_cache_dtype,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.models import model_dict
from nanovllm.executor.moe.profile import get_moe_profile, reset_moe_profile
from nanovllm.quantization import get_quant_config, process_weights_after_loading
from nanovllm.utils.context import get_context, reset_context, set_context
from nanovllm.utils.kv_cache_profile import get_kv_cache_profile, reset_kv_cache_profile
from nanovllm.utils.loader import load_model


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        
        self.tp_size = config.tp_size
        self.ep_size = config.ep_size
        self.world_size = self.tp_size * self.ep_size
        
        self.rank = rank
        self.event = event

        dist.init_process_group(
            "nccl", "tcp://localhost:2335", world_size=self.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        self.device = torch.device(f"cuda:{rank}")
        
        self.tp_group = None
        self.ep_group = None
        
        for i in range(self.world_size // self.tp_size):
            ranks = list(range(i * self.tp_size, (i + 1) * self.tp_size))
            group = dist.new_group(ranks)
            if rank in ranks: 
                self.tp_group = group
                
        for i in range(self.tp_size):
            ranks = list(range(i, self.world_size, self.tp_size))
            group = dist.new_group(ranks)
            if rank in ranks: 
                self.ep_group = group

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        quant_type = getattr(hf_config, "quantization_type", None)
        model_key = f"{hf_config.model_type}_{quant_type}" if quant_type else hf_config.model_type
        model_cls = model_dict[model_key]
        model_init_params = inspect.signature(model_cls.__init__).parameters
        model_kwargs = {}
        if "tp_group" in model_init_params:
            model_kwargs["tp_group"] = self.tp_group
        if "ep_group" in model_init_params:
            model_kwargs["ep_group"] = self.ep_group
        if "moe_backend" in model_init_params:
            model_kwargs["moe_backend"] = config.moe_backend
        if "moe_ep_backend" in model_init_params:
            model_kwargs["moe_ep_backend"] = config.moe_ep_backend
        quant_config = get_quant_config(hf_config)
        if "quant_config" in model_init_params:
            model_kwargs["quant_config"] = quant_config
        self.model = model_cls(hf_config, **model_kwargs)
        load_model(self.model, config.model)
        process_weights_after_loading(self.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier(device_ids=[rank])
            else:
                dist.barrier(device_ids=[rank])
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier(device_ids=[self.rank])
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def reset_moe_profile(self):
        reset_moe_profile()

    def get_moe_profile(self):
        return get_moe_profile()

    def reset_kv_cache_profile(self):
        reset_kv_cache_profile()

    def get_kv_cache_profile(self):
        return get_kv_cache_profile()

    def warmup_model(self):
        if os.environ.get("NANOVLLM_SKIP_MODEL_WARMUP", "0") == "1":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            return
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        warmup_tokens = int(os.environ.get("NANOVLLM_MODEL_WARMUP_TOKENS", "0"))
        is_moe_model = (
            hasattr(self.config.hf_config, "moe_intermediate_size")
            or "moe" in str(getattr(self.config.hf_config, "model_type", "")).lower()
        )
        if warmup_tokens <= 0 and not self.enforce_eager and self.ep_size == 1 and is_moe_model:
            warmup_tokens = min(max_num_batched_tokens, 64)
        if warmup_tokens > 0:
            max_num_batched_tokens = min(max_num_batched_tokens, warmup_tokens)
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(
            max_num_batched_tokens // seq_len, self.config.max_num_seqs
        )
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        used = total - free
        
        tp_size = dist.get_world_size(self.tp_group) if hasattr(self, "tp_group") and self.tp_group is not None else 1
        num_kv_heads = hf_config.num_key_value_heads // tp_size
        
        assert hf_config.hidden_size % hf_config.num_attention_heads == 0
        head_dim = (
            hf_config.head_dim
            if hasattr(hf_config, "head_dim")
            else hf_config.hidden_size // hf_config.num_attention_heads
        )
        kv_cache_dtype_name = normalize_kv_cache_dtype(config.kv_cache_dtype)
        if kv_cache_dtype_name == "bf16":
            kv_data_bpe = 2 * hf_config.torch_dtype.itemsize
        elif kv_cache_dtype_name == "k_int8_v_fp8":
            kv_data_bpe = 2
            recent_k_window = k_int8_recent_bf16_window_from_env()
            scale_bpe = kv_cache_scale_dtype_bytes_per_element(config.kv_cache_scale_dtype)
            k_group_size = k_int8_group_size_from_env(head_dim)
            k_groups = head_dim // k_group_size
            scale_block_bytes = (
                hf_config.num_hidden_layers
                * self.block_size
                * num_kv_heads
                * (k_groups + 1)
                * scale_bpe
            )
            shadow_k_block_bytes = (
                hf_config.num_hidden_layers
                * self.block_size
                * num_kv_heads
                * head_dim
                * hf_config.torch_dtype.itemsize
                if recent_k_window > 0
                else 0
            )
        elif kv_cache_dtype_name == "fp8":
            kv_data_bpe = 2
            scale_bpe = kv_cache_scale_dtype_bytes_per_element(config.kv_cache_scale_dtype)
            scale_block_bytes = (
                hf_config.num_hidden_layers
                * self.block_size
                * num_kv_heads
                * 2
                * scale_bpe
            )
            shadow_k_block_bytes = 0
        else:
            raise AssertionError(f"Unhandled kv_cache_dtype={config.kv_cache_dtype!r}")
        kv_data_block_bytes = (
            hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * kv_data_bpe
        )
        if kv_cache_dtype_name == "bf16":
            scale_block_bytes = 0
            shadow_k_block_bytes = 0
        block_bytes = kv_data_block_bytes + scale_block_bytes + shadow_k_block_bytes
        requested_num_blocks = int(config.num_kvcache_blocks)
        workspace_reserve_bytes = 1024 * 1024 * 1024
        budget_bytes = int(free * config.gpu_memory_utilization) - workspace_reserve_bytes
        if requested_num_blocks > 0:
            config.num_kvcache_blocks = requested_num_blocks
        else:
            config.num_kvcache_blocks = budget_bytes // block_bytes
        if config.num_kvcache_blocks <= 0:
            raise RuntimeError(
                "KV cache block budget is non-positive: "
                f"num_kvcache_blocks={config.num_kvcache_blocks}, "
                f"budget_bytes={budget_bytes}, block_bytes={block_bytes}, "
                f"free_bytes={free}, used_bytes={used}, total_bytes={total}, "
                f"gpu_memory_utilization={config.gpu_memory_utilization}, "
                f"workspace_reserve_bytes={workspace_reserve_bytes}."
            )
        cache_shape = (
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        scale_cache_shape = (hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads)
        if kv_cache_dtype_name == "k_int8_v_fp8":
            recent_k_window = k_int8_recent_bf16_window_from_env()
            self.k_cache_storage = torch.empty(cache_shape, dtype=torch.int8, device="cuda")
            self.v_cache_storage = torch.empty(cache_shape, dtype=torch.float8_e4m3fn, device="cuda")
            self.kv_cache = torch.tensor([], device="cuda")
            self.k_bf16_shadow_cache = (
                torch.empty(cache_shape, dtype=hf_config.torch_dtype, device="cuda")
                if recent_k_window > 0
                else torch.tensor([], device="cuda")
            )
            scale_dtype = getattr(torch, config.kv_cache_scale_dtype)
            k_group_size = k_int8_group_size_from_env(head_dim)
            self.k_scale_cache = torch.empty(
                (*scale_cache_shape, head_dim // k_group_size), dtype=scale_dtype, device="cuda"
            )
            self.v_scale_cache = torch.empty(scale_cache_shape, dtype=scale_dtype, device="cuda")
        elif kv_cache_dtype_name == "fp8":
            self.k_cache_storage = torch.empty(cache_shape, dtype=torch.float8_e4m3fn, device="cuda")
            self.v_cache_storage = torch.empty(cache_shape, dtype=torch.float8_e4m3fn, device="cuda")
            self.kv_cache = torch.tensor([], device="cuda")
            self.k_bf16_shadow_cache = torch.tensor([], device="cuda")
            scale_dtype = getattr(torch, config.kv_cache_scale_dtype)
            self.k_scale_cache = torch.empty(scale_cache_shape, dtype=scale_dtype, device="cuda")
            self.v_scale_cache = torch.empty(scale_cache_shape, dtype=scale_dtype, device="cuda")
        else:
            kv_cache_dtype = hf_config.torch_dtype
            kv_cache_shape = (2, *cache_shape)
            self.kv_cache = torch.zeros(kv_cache_shape, dtype=kv_cache_dtype, device="cuda")
            self.k_cache_storage = self.kv_cache[0]
            self.v_cache_storage = self.kv_cache[1]
            self.k_bf16_shadow_cache = torch.tensor([], device="cuda")
            self.k_scale_cache = self.v_scale_cache = torch.tensor([], device="cuda")
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.k_cache_storage[layer_id]
                module.v_cache = self.v_cache_storage[layer_id]
                if hasattr(module, "k_bf16_shadow_cache"):
                    module.k_bf16_shadow_cache = (
                        self.k_bf16_shadow_cache[layer_id]
                        if self.k_bf16_shadow_cache.numel()
                        else self.k_bf16_shadow_cache
                    )
                if self.k_scale_cache.numel() and hasattr(module, "k_scale_cache"):
                    module.k_scale_cache = self.k_scale_cache[layer_id]
                if self.v_scale_cache.numel() and hasattr(module, "v_scale_cache"):
                    module.v_scale_cache = self.v_scale_cache[layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        reset_context()
        return token_ids

    def run_with_logits(self, seqs: list[Sequence], is_prefill: bool):
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        if self.rank == 0:
            token_ids = self.sampler(logits, temperatures).tolist()
            logits_cpu = logits.detach().float().cpu()
        else:
            token_ids = None
            logits_cpu = None
        reset_context()
        return token_ids, logits_cpu

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            old_moe_graph_align = os.environ.get("NANOVLLM_MOE_GRAPH_ALIGN")
            os.environ["NANOVLLM_MOE_GRAPH_ALIGN"] = "1"
            try:
                # Two warmup passes BEFORE capture: the first triggers Triton
                # JIT / autotune for this bs, the second confirms the autotune
                # cache hit so capture sees no host-side compile work. The
                # explicit synchronize drains any deferred host kernels.
                for _ in range(2):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                torch.cuda.synchronize()
                with torch.cuda.graph(graph, self.graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            finally:
                if old_moe_graph_align is None:
                    os.environ.pop("NANOVLLM_MOE_GRAPH_ALIGN", None)
                else:
                    os.environ["NANOVLLM_MOE_GRAPH_ALIGN"] = old_moe_graph_align
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
