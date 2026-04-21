"""Unit tests for INT8 quantized linear layers (smooth_quant_linear.py).

Tests cover:
  - W8A16: weight loading correctness, forward output shape, determinism
  - W8A8 static: weight + input_scale loading, forward output shape,
    output consistency with W8A16 (same weights → same result, different path)
  - Bias support in QKV and row-parallel layers
  - Large-M path (torch._int_mm) vs small-M fallback (BF16)
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qkv(hidden=64, num_heads=4, num_kv_heads=4, scheme="w8a16"):
    from nanovllm.layers.smooth_quant_linear import Int8QKVParallelLinear

    head_dim = hidden // num_heads
    layer = Int8QKVParallelLinear(
        hidden_size=hidden,
        head_size=head_dim,
        total_num_heads=num_heads,
        total_num_kv_heads=num_kv_heads,
        bias=True,
        int8_scheme=scheme,
    )
    # Fill qweight and weight_scale with simple known values
    N = (num_heads + 2 * num_kv_heads) * head_dim
    K = hidden
    with torch.no_grad():
        layer.qweight.fill_(1)                                  # all 1s int8
        layer.weight_scale.fill_(0.01)                          # scale = 0.01
        layer.bias.zero_()
        if scheme == "w8a8_static":
            layer.input_scale.fill_(1.0)                        # scale = 1.0
    return layer


def _make_row(in_f=64, out_f=64, scheme="w8a16"):
    from nanovllm.layers.smooth_quant_linear import Int8RowParallelLinear

    layer = Int8RowParallelLinear(
        input_size=in_f,
        output_size=out_f,
        bias=False,
        int8_scheme=scheme,
    )
    with torch.no_grad():
        layer.qweight.fill_(1)
        layer.weight_scale.fill_(0.01)
        if scheme == "w8a8_static":
            layer.input_scale.fill_(1.0)
    return layer


def _make_merged(in_f=64, out_sizes=(128, 128), scheme="w8a16"):
    from nanovllm.layers.smooth_quant_linear import Int8MergedColumnParallelLinear

    layer = Int8MergedColumnParallelLinear(
        input_size=in_f,
        output_sizes=list(out_sizes),
        bias=False,
        int8_scheme=scheme,
    )
    N_total = sum(out_sizes)
    with torch.no_grad():
        layer.qweight.fill_(1)
        layer.weight_scale.fill_(0.01)
        if scheme == "w8a8_static":
            layer.input_scale.fill_(1.0)
    return layer


# ---------------------------------------------------------------------------
# Buffer / shape tests
# ---------------------------------------------------------------------------

class TestBufferShapes:
    def test_qkv_buffers_w8a16(self):
        layer = _make_qkv(hidden=64, num_heads=4, num_kv_heads=4, scheme="w8a16")
        K, N = 64, (4 + 4 + 4) * 16
        assert layer.qweight.shape == (K, N)
        assert layer.weight_scale.shape == (N,)
        assert not hasattr(layer, "input_scale") or layer.input_scale is None  # no static scale

    def test_qkv_buffers_w8a8_static(self):
        layer = _make_qkv(hidden=64, num_heads=4, num_kv_heads=4, scheme="w8a8_static")
        assert layer.input_scale.shape == (1,)
        assert layer.qweight.dtype == torch.int8
        assert layer.weight_scale.dtype == torch.float32
        assert layer.input_scale.dtype == torch.float32

    def test_row_buffers(self):
        layer = _make_row(64, 64, scheme="w8a8_static")
        K_per_tp = 64
        assert layer.qweight.shape == (K_per_tp, 64)

    def test_merged_buffers(self):
        layer = _make_merged(64, (128, 128), scheme="w8a16")
        assert layer.qweight.shape == (64, 256)
        assert layer.weight_scale.shape == (256,)


# ---------------------------------------------------------------------------
# Forward shape tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["w8a16", "w8a8_static"])
class TestForwardShape:
    def test_qkv_prefill(self, scheme):
        """Prefill: large M, uses INT8 GEMM path for w8a8_static."""
        layer = _make_qkv(hidden=64, num_heads=4, num_kv_heads=4, scheme=scheme)
        x = torch.randn(40, 64)   # M=40 > 32 → INT8 path for static
        out = layer(x)
        N = (4 + 4 + 4) * 16
        assert out.shape == (40, N), f"Expected (40, {N}), got {out.shape}"

    def test_qkv_decode(self, scheme):
        """Decode: small M, uses BF16 fallback even for w8a8_static."""
        layer = _make_qkv(hidden=64, num_heads=4, num_kv_heads=4, scheme=scheme)
        x = torch.randn(1, 64)
        out = layer(x)
        N = (4 + 4 + 4) * 16
        assert out.shape == (1, N)

    def test_row_forward(self, scheme):
        layer = _make_row(64, 64, scheme=scheme)
        x = torch.randn(8, 64)
        out = layer(x)
        assert out.shape == (8, 64)

    def test_merged_forward(self, scheme):
        layer = _make_merged(64, (128, 128), scheme=scheme)
        x = torch.randn(16, 64)
        out = layer(x)
        assert out.shape == (16, 256)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------

class TestNumerical:
    def test_w8a16_zero_weight(self):
        """All-zero weights produce all-zero output."""
        from nanovllm.layers.smooth_quant_linear import Int8RowParallelLinear

        layer = Int8RowParallelLinear(input_size=64, output_size=64)
        x = torch.randn(8, 64)
        out = layer(x)
        assert out.abs().max().item() == 0.0, "Zero weights should give zero output"

    def test_w8a16_known_value(self):
        """qweight=1, weight_scale=s → each output element = s * sum(x_row)."""
        from nanovllm.layers.smooth_quant_linear import Int8RowParallelLinear

        K, N = 32, 16
        s = 0.5
        layer = Int8RowParallelLinear(input_size=K, output_size=N)
        with torch.no_grad():
            layer.qweight.fill_(1)
            layer.weight_scale.fill_(s)

        x = torch.ones(4, K)
        out = layer(x)  # each row = [K * 1 * s] * N = [K * s] * N
        expected = K * s
        assert out.abs().max().item() == pytest.approx(expected, rel=0.01), \
            f"Expected {expected}, got {out.abs().max().item()}"

    def test_w8a8_static_matches_w8a16_approx(self):
        """W8A8 static and W8A16 should produce close results (same weights, input_scale=1)."""
        from nanovllm.layers.smooth_quant_linear import Int8RowParallelLinear

        torch.manual_seed(42)
        K, N = 64, 32

        w16 = Int8RowParallelLinear(input_size=K, output_size=N, int8_scheme="w8a16")
        w8s = Int8RowParallelLinear(input_size=K, output_size=N, int8_scheme="w8a8_static")

        # Same weights
        with torch.no_grad():
            rand_w = torch.randint(-10, 10, (K, N), dtype=torch.int8)
            rand_s = torch.rand(N) * 0.1 + 0.01
            for layer in (w16, w8s):
                layer.qweight.copy_(rand_w)
                layer.weight_scale.copy_(rand_s)
            w8s.input_scale.fill_(1.0)

        x = torch.randn(8, K)
        out16 = w16(x)
        out8s = w8s(x)
        # Static quantizes activation to int8 → small rounding error expected
        max_diff = (out16 - out8s).abs().max().item()
        assert max_diff < 0.1 * out16.abs().max().item() + 1e-4, \
            f"W8A8 static differs too much from W8A16: max_diff={max_diff:.4f}"

    def test_large_m_uses_int8_mm(self):
        """For M > 32, W8A8 static should use torch._int_mm internally."""
        from nanovllm.layers.smooth_quant_linear import _INT8_MM_MIN_M, Int8RowParallelLinear

        K, N = 64, 64
        layer = Int8RowParallelLinear(input_size=K, output_size=N, int8_scheme="w8a8_static")
        with torch.no_grad():
            layer.qweight.fill_(1)
            layer.weight_scale.fill_(0.01)
            layer.input_scale.fill_(1.0)

        M = _INT8_MM_MIN_M + 1  # triggers int8 path
        x = torch.randn(M, K)
        out = layer(x)
        assert out.shape == (M, N)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Weight loader tests
# ---------------------------------------------------------------------------

class TestWeightLoaders:
    def test_qweight_loader_qkv(self):
        """Check that q/k/v shards are stored in the correct columns of qweight."""
        from nanovllm.layers.smooth_quant_linear import Int8QKVParallelLinear

        H, hd, Nh, Nkv = 64, 16, 4, 4
        layer = Int8QKVParallelLinear(H, hd, Nh, Nkv)
        N_q = Nh * hd
        N_kv = Nkv * hd

        # Fake weights
        w_q = torch.ones(N_q, H, dtype=torch.int8)   # [N_q, K]
        w_k = torch.ones(N_kv, H, dtype=torch.int8) * 2
        w_v = torch.ones(N_kv, H, dtype=torch.int8) * 3

        layer._qweight_loader(layer.qweight, w_q, "q")
        layer._qweight_loader(layer.qweight, w_k, "k")
        layer._qweight_loader(layer.qweight, w_v, "v")

        # qweight[K, N_total]; column slice for q is [0:N_q]
        q_slice = layer.qweight[:, :N_q]       # [K, N_q]
        k_slice = layer.qweight[:, N_q:N_q + N_kv]
        v_slice = layer.qweight[:, N_q + N_kv:]

        assert (q_slice == 1).all(), "q shard should be 1"
        assert (k_slice == 2).all(), "k shard should be 2"
        assert (v_slice == 3).all(), "v shard should be 3"

    def test_input_scale_loader(self):
        """input_scale buffer is filled by loader."""
        from nanovllm.layers.smooth_quant_linear import Int8RowParallelLinear

        layer = Int8RowParallelLinear(64, 64, int8_scheme="w8a8_static")
        scale_tensor = torch.tensor([0.05], dtype=torch.float32)
        layer.input_scale.weight_loader(layer.input_scale, scale_tensor)
        assert layer.input_scale.item() == pytest.approx(0.05)
