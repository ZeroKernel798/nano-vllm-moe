"""Block manager for KV cache with prefix caching support.

支持两种 prefix cache backend:
- "hash": 使用 hash map (原始实现)
- "radix": 使用 radix tree (HiRadix L1)
"""

from collections import deque
import os
import time
from typing import Literal

import numpy as np
import xxhash

from nanovllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManagerStats:
    """Prefix cache statistics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.num_queries = 0
        self.total_lookup_time_ms = 0.0
        self.total_hit_blocks = 0
        self.total_hit_tokens = 0
        self.total_logical_tokens = 0

    def record_lookup(self, hit_blocks: int, hit_tokens: int, logical_tokens: int, lookup_time_ms: float):
        self.num_queries += 1
        self.total_lookup_time_ms += lookup_time_ms
        self.total_hit_blocks += hit_blocks
        self.total_hit_tokens += hit_tokens
        self.total_logical_tokens += logical_tokens

    def to_dict(self) -> dict:
        return {
            "num_queries": self.num_queries,
            "total_lookup_time_ms": round(self.total_lookup_time_ms, 3),
            "avg_lookup_time_ms": round(self.total_lookup_time_ms / max(1, self.num_queries), 3),
            "total_hit_blocks": self.total_hit_blocks,
            "total_hit_tokens": self.total_hit_tokens,
            "total_logical_tokens": self.total_logical_tokens,
            "block_hit_rate": round(self.total_hit_blocks / max(1, self.total_logical_tokens // 256), 3),
            "token_hit_rate": round(self.total_hit_tokens / max(1, self.total_logical_tokens), 3),
        }


def get_prefix_cache_backend() -> Literal["hash", "radix"]:
    """Get prefix cache backend from environment variable."""
    return os.environ.get("NANOVLLM_PREFIX_CACHE_BACKEND", "hash").lower()


class BlockManager:
    """Block manager with pluggable prefix cache backend."""

    def __init__(self, num_blocks: int, block_size: int, prefix_cache_backend: str | None = None):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.stats = BlockManagerStats()

        # Select prefix cache backend
        self.prefix_cache_backend = prefix_cache_backend or get_prefix_cache_backend()

        # Initialize prefix cache
        if self.prefix_cache_backend == "radix":
            from nanovllm.engine.radix_tree import RadixTreePrefixCache
            self._prefix_cache = RadixTreePrefixCache(block_size)
        else:
            # Default: hash-based prefix cache
            self._hash_to_block_id: dict[int, int] = dict()
            self._prefix_cache = None

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0

        # Remove from prefix cache
        if self.prefix_cache_backend == "radix":
            self._prefix_cache.remove_block(block_id)
        else:
            if block.hash != -1 and self._hash_to_block_id.get(block.hash) == block_id:
                del self._hash_to_block_id[block.hash]

        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        start_time = time.perf_counter()
        num_cached_blocks = 0
        cached_used_blocks = 0
        logical_tokens = (seq.num_blocks - 1) * self.block_size

        if self.prefix_cache_backend == "radix":
            # Radix tree based lookup
            token_blocks = [seq.block(i) for i in range(seq.num_blocks - 1)]
            num_cached_blocks, matched_block_ids = self._prefix_cache.match_prefix(token_blocks)
            cached_used_blocks = sum(1 for block_id in matched_block_ids if block_id in self.used_block_ids)
        else:
            # Hash based lookup
            h = -1
            for i in range(seq.num_blocks - 1):
                token_ids = seq.block(i)
                h = self.compute_hash(token_ids, h)
                block_id = self._hash_to_block_id.get(h, -1)
                if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                    break
                num_cached_blocks += 1
                if block_id in self.used_block_ids:
                    cached_used_blocks += 1

        lookup_time_ms = (time.perf_counter() - start_time) * 1000
        hit_tokens = num_cached_blocks * self.block_size
        self.stats.record_lookup(num_cached_blocks, hit_tokens, logical_tokens, lookup_time_ms)

        num_new_blocks = seq.num_blocks - cached_used_blocks
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table

        if self.prefix_cache_backend == "radix":
            # Radix tree based allocation
            matched_blocks, matched_block_ids = self._prefix_cache.match_prefix(
                [seq.block(i) for i in range(num_cached_blocks)]
            )
            for block_id in matched_block_ids:
                block = self.blocks[block_id]
                if block_id in self.used_block_ids:
                    block.ref_count += 1
                else:
                    block.ref_count = 1
                    self.free_block_ids.remove(block_id)
                    self.used_block_ids.add(block_id)
                seq.block_table.append(block_id)

            for _ in range(num_cached_blocks, seq.num_blocks):
                seq.block_table.append(self._allocate_block())
        else:
            # Hash based allocation
            h = -1
            for i in range(num_cached_blocks):
                token_ids = seq.block(i)
                h = self.compute_hash(token_ids, h)
                block_id = self._hash_to_block_id[h]
                block = self.blocks[block_id]
                if block_id in self.used_block_ids:
                    block.ref_count += 1
                else:
                    block.ref_count = 1
                    self.free_block_ids.remove(block_id)
                    self.used_block_ids.add(block_id)
                seq.block_table.append(block_id)
            for _ in range(num_cached_blocks, seq.num_blocks):
                seq.block_table.append(self._allocate_block())

        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def allocate_and_hash(self, seq: Sequence, num_cached_blocks: int):
        """Allocate blocks and immediately hash them for batch-level reuse."""
        self.allocate(seq, num_cached_blocks)

        # Hash all complete blocks immediately
        if self.prefix_cache_backend == "radix":
            token_blocks = [seq.block(i) for i in range(seq.num_blocks - 1)]
            block_ids = seq.block_table[:len(token_blocks)]
            self._prefix_cache.insert(token_blocks, block_ids)
        else:
            h = -1
            for i in range(seq.num_blocks - 1):
                token_ids = seq.block(i)
                h = self.compute_hash(token_ids, h)
                block_id = seq.block_table[i]
                block = self.blocks[block_id]
                block.update(h, token_ids)
                self._hash_to_block_id[h] = block_id

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return

        if self.prefix_cache_backend == "radix":
            token_blocks = [seq.block(i) for i in range(start, end)]
            block_ids = seq.block_table[start:end]
            self._prefix_cache.insert(token_blocks, block_ids)
        else:
            h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
            for i in range(start, end):
                block = self.blocks[seq.block_table[i]]
                token_ids = seq.block(i)
                h = self.compute_hash(token_ids, h)
                block.update(h, token_ids)
                self._hash_to_block_id[h] = block.block_id
