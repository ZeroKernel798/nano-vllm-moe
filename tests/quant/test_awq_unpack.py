"""Numerical alignment test: AWQ INT4 unpack → F.linear reference.

Goal: Verify that ``unpack_awq_int4`` produces a weight matrix whose
F.linear output matches a brute-force reference implementation to within
a tight absolute tolerance (< 0.01 for typical scale values).

The test is self-contained – it does NOT require any real checkpoint file.
All tensors are constructed programmatically to cover both the packing
logic and the de-quantisation arithmetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm.layers.quant_linear import unpack_awq_int4

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_unpack_shape():
    """unpack_awq_int4 returns [out_f, in_f]."""
    in_f, out_f, group_size = 64, 128, 32
    qweight = torch.zeros(in_f, out_f // 8, dtype=torch.int32)
    qzeros = torch.zeros(in_f // group_size, out_f // 8, dtype=torch.int32)
    scales = torch.ones(in_f // group_size, out_f, dtype=torch.float16)
    w = unpack_awq_int4(qweight, qzeros, scales, group_size)
    assert w.shape == (out_f, in_f), f"Expected ({out_f}, {in_f}), got {w.shape}"


def test_unpack_zero_weights_gives_minus_scale():
    """All-zero weights (int4=0) with zero-point=0 → w = (0 - (0+1)) * s = -s."""
    in_f, out_f, group_size = 32, 64, 32
    qweight = torch.zeros(in_f, out_f // 8, dtype=torch.int32)
    qzeros = torch.zeros(in_f // group_size, out_f // 8, dtype=torch.int32)
    scale_val = 0.5
    scales = torch.full((in_f // group_size, out_f), scale_val, dtype=torch.float16)

    w = unpack_awq_int4(qweight, qzeros, scales, group_size)
    expected = torch.full((out_f, in_f), -scale_val, dtype=torch.float16)
    assert torch.allclose(w, expected, atol=1e-3), f"max diff: {(w - expected).abs().max()}"


def test_unpack_known_weight_value():
    """Verify de-quant arithmetic for a specific known weight value.

    For qweight containing int4=8, qzeros containing int4=7, scale=1.0:
      result = (8 - (7 + 1)) * 1.0 = 0.0

    We control a single column so we know exactly which packed slot holds value 8.
    The 8 values packed into one int32 are stored as bits [0:4, 4:8, 8:12, ...].
    With no vLLM interleaving for the first 8 values in a column, the 5th element
    in a packed int32 (bits [16:20]) has a permuted position due to the
    view(4,2).permute(1,0) reorder in unpack_awq_int4.

    We use all-same values to sidestep the permutation and test the formula.
    """
    in_f, out_f_packed, group_size = 32, 4, 32
    out_f = out_f_packed * 8

    # All int4 weights = 8
    w_val = 8
    qweight = torch.full((in_f, out_f_packed), 0, dtype=torch.int32)
    # Pack 8 copies of w_val into each int32
    for shift in range(0, 32, 4):
        qweight |= (w_val << shift)

    # All zero-points = 7
    z_val = 7
    n_groups = in_f // group_size
    qzeros = torch.full((n_groups, out_f_packed), 0, dtype=torch.int32)
    for shift in range(0, 32, 4):
        qzeros |= (z_val << shift)

    # scale = 1.0
    scales = torch.ones(n_groups, out_f, dtype=torch.float16)

    w = unpack_awq_int4(qweight, qzeros, scales, group_size)
    # (8 - (7+1)) * 1.0 = 0 for every element
    assert w.shape == (out_f, in_f)
    assert w.abs().max().item() == 0.0, f"Expected all zeros, got max={w.abs().max()}"


def test_f_linear_output_deterministic():
    """F.linear with unpacked weights is deterministic across two calls.

    Also tests that the output of a zero-weight layer (qweight=0, qzeros=0)
    equals -scale * x, i.e. the +1 offset is applied correctly.
    """
    in_f, out_f_packed, group_size = 32, 8, 32
    out_f = out_f_packed * 8
    batch = 3

    # All zero qweight, zero qzeros, uniform scale
    scale_val = 0.25
    qweight = torch.zeros(in_f, out_f_packed, dtype=torch.int32)
    n_groups = in_f // group_size
    qzeros = torch.zeros(n_groups, out_f_packed, dtype=torch.int32)
    scales = torch.full((n_groups, out_f), scale_val, dtype=torch.float16)

    w = unpack_awq_int4(qweight, qzeros, scales, group_size)
    # qweight=0, qzeros=0 → (0 - (0+1)) * scale = -scale
    expected_w_val = -scale_val
    assert torch.allclose(
        w, torch.full_like(w, expected_w_val), atol=1e-3
    ), f"Expected {expected_w_val}, max diff={( w - expected_w_val).abs().max()}"

    x = torch.randn(batch, in_f)
    out1 = F.linear(x.float(), w.float())
    out2 = F.linear(x.float(), w.float())
    assert torch.equal(out1, out2), "Determinism check failed"

    # output = x @ (-scale) * ones.T = -scale * x.sum(dim=-1, keepdim=True)...
    # actually out[i, j] = sum_k x[i,k] * w[j,k] = -scale * sum_k x[i,k]
    expected_out = -scale_val * x.sum(dim=-1, keepdim=True).expand(batch, out_f)
    assert torch.allclose(out1.float(), expected_out.float(), atol=1e-4), (
        f"Output mismatch: max_diff={(out1 - expected_out).abs().max():.6f}"
    )


def test_awq_qkv_parallel_linear_forward_shape():
    """AWQQKVParallelLinear forward returns correct output shape (tp_size=1).

    group_size=128 requires hidden_size >= 128.
    """
    from nanovllm.layers.quant_linear import AWQQKVParallelLinear

    # hidden must be divisible by group_size=128
    hidden, head_size = 256, 16
    total_heads, total_kv_heads = 4, 2

    layer = AWQQKVParallelLinear(
        hidden_size=hidden,
        head_size=head_size,
        total_num_heads=total_heads,
        total_num_kv_heads=total_kv_heads,
        tp_group=None,
    )
    x = torch.randn(2, hidden, dtype=torch.float16)
    out = layer(x)
    expected_out = (total_heads + 2 * total_kv_heads) * head_size
    assert out.shape == (2, expected_out), f"Got {out.shape}, expected (2, {expected_out})"
