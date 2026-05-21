"""Radix tree for prefix cache metadata management.

HiRadix L1: 使用 radix tree 替代 hash map，提供更好的 prefix 匹配效率。
当前只保留已经接入 BlockManager 并有 benchmark 证据的 block-level prefix match/insert。
"""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional


@dataclass
class RadixNode:
    """Radix tree node representing a KV block."""
    node_id: int
    token_ids: list[int] = field(default_factory=list)
    block_id: Optional[int] = None
    parent: Optional["RadixNode"] = None
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    last_access_time: float = 0.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None


class RadixTreePrefixCache:
    """Block-level radix prefix cache.

    与 HashPrefixCache 并存，通过配置选择使用哪种实现。
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.root = RadixNode(node_id=0)
        self._next_node_id = 1
        self._node_count = 1
        self._block_to_node: dict[int, RadixNode] = {}

    def _create_node(self, token_ids: list[int], parent: Optional[RadixNode] = None) -> RadixNode:
        node = RadixNode(
            node_id=self._next_node_id,
            token_ids=token_ids,
            parent=parent,
            last_access_time=perf_counter(),
        )
        self._next_node_id += 1
        self._node_count += 1
        return node

    def match_prefix(self, token_blocks: list[list[int]]) -> tuple[int, list[int]]:
        """Match prefix in the radix tree.

        Returns:
            Tuple of (matched_block_count, matched_block_ids).
        """
        current = self.root
        matched_blocks = 0
        matched_block_ids: list[int] = []

        for block_tokens in token_blocks:
            if not block_tokens:
                break
            key = block_tokens[0]

            if key not in current.children:
                break

            child = current.children[key]
            if child.token_ids != block_tokens:
                break

            matched_blocks += 1
            if child.block_id is not None:
                matched_block_ids.append(child.block_id)
            current = child
            current.last_access_time = perf_counter()

        return matched_blocks, matched_block_ids

    def insert(self, token_blocks: list[list[int]], block_ids: list[int]):
        """Insert blocks into the radix tree."""
        matched_blocks, _ = self.match_prefix(token_blocks)

        current = self.root
        for i in range(matched_blocks):
            block_tokens = token_blocks[i]
            key = block_tokens[0]
            current = current.children[key]

        for i in range(matched_blocks, len(token_blocks)):
            block_tokens = token_blocks[i]
            block_id = block_ids[i] if i < len(block_ids) else None

            key = block_tokens[0]
            new_node = self._create_node(block_tokens, parent=current)
            new_node.block_id = block_id

            if block_id is not None:
                self._block_to_node[block_id] = new_node

            current.children[key] = new_node
            current = new_node

        current.last_access_time = perf_counter()

    def remove_block(self, block_id: int):
        if block_id in self._block_to_node:
            node = self._block_to_node[block_id]
            if node.parent:
                key = node.token_ids[0] if node.token_ids else 0
                if key in node.parent.children:
                    del node.parent.children[key]
            del self._block_to_node[block_id]
            self._node_count -= 1

    def get_stats(self) -> dict:
        leaf_count = 0
        max_depth = 0

        def traverse(node: RadixNode, depth: int):
            nonlocal leaf_count, max_depth
            max_depth = max(max_depth, depth)
            if node.is_leaf() and not node.is_root():
                leaf_count += 1
            for child in node.children.values():
                traverse(child, depth + 1)

        traverse(self.root, 0)

        return {
            "node_count": self._node_count,
            "leaf_count": leaf_count,
            "max_depth": max_depth,
        }
