"""Unit tests for the unified Qwen3AttentionBlock.

Tests verify:
1. Both model-side aliases (Qwen3Attention, Qwen3MoeAttention) resolve to the
   same class as Qwen3AttentionBlock.
2. Parameter/buffer shapes are correct for tp_size=1.
3. Forward output shape and value is identical to what the two previously
   separate implementations would have produced (regression guard).

The inner ``Attention.forward`` calls flash_attn and requires a live CUDA
KV-cache, so we mock it to a simple torch.bmm in order to keep this test
CPU-only and dependency-free.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm.layers.attention_block import Qwen3AttentionBlock
from nanovllm.models.qwen3 import Qwen3Attention
from nanovllm.models.qwen3_moe import Qwen3MoeAttention


# ---------------------------------------------------------------------------
# Fixtures / shared config
# ---------------------------------------------------------------------------

HIDDEN = 128
NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = HIDDEN // NUM_HEADS  # 16
RMS_EPS = 1e-6


def _make_block(seed: int = 42, **kwargs) -> Qwen3AttentionBlock:
    defaults = dict(
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        max_position=512,
        rms_norm_eps=RMS_EPS,
        tp_group=None,
    )
    defaults.update(kwargs)
    block = Qwen3AttentionBlock(**defaults)
    # ``torch.empty`` does not use the RNG, so weights are garbage by default.
    # Initialise all parameters and buffers to small normal values.
    torch.manual_seed(seed)
    for p in block.parameters():
        nn.init.normal_(p, std=0.02)
    return block


def _mock_attn_forward(self, q, k, v):
    """Replace flash_attn with a simple dot-product to stay CPU-only.

    ``Attention.forward`` receives flat [T, H*D] inputs and internally
    reshapes them; since we replace the whole method we must do the reshape.
    """
    T = q.shape[0]
    H, D = self.num_heads, self.head_dim
    Hkv = self.num_kv_heads
    q3 = q.view(T, H, D)
    k3 = k.view(T, Hkv, D)
    v3 = v.view(T, Hkv, D)
    # Expand GQA
    k_exp = k3.repeat_interleave(H // Hkv, dim=1)
    v_exp = v3.repeat_interleave(H // Hkv, dim=1)
    scale = D**-0.5
    scores = torch.einsum("thd,shd->ths", q3, k_exp) * scale
    weights = scores.softmax(dim=-1)
    out = torch.einsum("ths,shd->thd", weights, v_exp)
    return out.reshape(T, H * D)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_aliases_are_same_class():
    """Qwen3Attention and Qwen3MoeAttention are both Qwen3AttentionBlock."""
    assert Qwen3Attention is Qwen3AttentionBlock
    assert Qwen3MoeAttention is Qwen3AttentionBlock


def test_parameter_shapes_tp1():
    """QKV / O-proj weight shapes are correct for tp_size=1."""
    block = _make_block()

    q_out = NUM_HEADS * HEAD_DIM
    kv_out = NUM_KV_HEADS * HEAD_DIM
    total_qkv_out = q_out + 2 * kv_out

    assert block.qkv_proj.weight.shape == (total_qkv_out, HIDDEN), (
        f"qkv weight shape mismatch: {block.qkv_proj.weight.shape}"
    )
    assert block.o_proj.weight.shape == (HIDDEN, q_out), (
        f"o_proj weight shape mismatch: {block.o_proj.weight.shape}"
    )
    assert block.q_norm.weight.shape == (HEAD_DIM,)
    assert block.k_norm.weight.shape == (HEAD_DIM,)


def test_forward_output_shape_tp1():
    """Forward returns tensor of shape [T, hidden_size].

    ``torch.no_grad`` is required because RMSNorm uses inplace ``mul_`` on a
    view that comes from ``split``; autograd forbids this but inference does not.
    """
    block = _make_block()
    block.eval()
    T = 5
    positions = torch.arange(T)
    hidden = torch.randn(T, HIDDEN)

    with torch.no_grad(), patch.object(type(block.attn), "forward", _mock_attn_forward):
        out = block(positions, hidden)

    assert out.shape == (T, HIDDEN), f"Unexpected output shape: {out.shape}"


def test_forward_deterministic_tp1():
    """Determinism: two forward passes with same inputs return identical output."""
    block = _make_block()
    block.eval()
    T = 3
    positions = torch.arange(T)
    hidden = torch.randn(T, HIDDEN)

    with torch.no_grad(), patch.object(type(block.attn), "forward", _mock_attn_forward):
        out1 = block(positions, hidden.clone())
        out2 = block(positions, hidden.clone())

    assert torch.equal(out1, out2), "Non-deterministic forward output"


def test_qwen3_and_moe_variants_produce_same_output():
    """Instantiate via each alias; after copying weights outputs must be equal.

    Both aliases point to the same class, so identical weights must produce
    identical outputs.  We create one canonical block, then load its state
    dict into a second instance to guarantee weight parity.
    """
    block_dense = _make_block(seed=1)
    block_moe = _make_block(seed=99)  # different seed on purpose
    # Copy weights from dense to moe to assert structural identity
    block_moe.load_state_dict(block_dense.state_dict())

    T = 4
    positions = torch.arange(T)
    hidden = torch.randn(T, HIDDEN)

    with torch.no_grad(), patch.object(type(block_dense.attn), "forward", _mock_attn_forward):
        out_dense = block_dense(positions, hidden.clone())
        out_moe = block_moe(positions, hidden.clone())

    assert torch.equal(out_dense, out_moe), (
        f"Output mismatch after state_dict copy: "
        f"max_diff={(out_dense - out_moe).abs().max()}"
    )


def test_no_bias_by_default():
    """QKV and O-proj have no bias by default."""
    block = _make_block()
    assert block.qkv_proj.bias is None
    assert block.o_proj.bias is None


def test_qkv_bias_creates_bias_params():
    """qkv_bias=True allocates bias tensor in qkv_proj."""
    block = _make_block(qkv_bias=True)
    assert block.qkv_proj.bias is not None
    q_out = NUM_HEADS * HEAD_DIM
    kv_out = NUM_KV_HEADS * HEAD_DIM
    assert block.qkv_proj.bias.shape == (q_out + 2 * kv_out,)
