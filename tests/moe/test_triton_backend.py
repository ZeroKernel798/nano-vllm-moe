from __future__ import annotations

import pytest
import torch

from nanovllm.executor.moe.backends.triton import TritonMoEBackend


def test_dispatch_ep1_topk1_keeps_token_order():
    backend = TritonMoEBackend(tp_group=None, ep_group=None)
    x = torch.arange(6, dtype=torch.float32).view(2, 3)
    topk_ids = torch.tensor([[0], [1]])
    topk_weights = torch.tensor([[0.25], [0.75]], dtype=torch.float32)

    state = backend.dispatch(
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        local_num_experts=2,
        top_k=1,
    )

    assert torch.equal(state["recv_x"], x)
    assert torch.equal(state["recv_local_ids"], torch.tensor([0, 1]))
    assert torch.equal(state["recv_weights"], torch.tensor([0.25, 0.75]))
    assert state["s_list"] == [2]
    assert state["r_list"] == [2]


def test_combine_ep1_sums_topk_without_permutation_buffer():
    backend = TritonMoEBackend(tp_group=None, ep_group=None)
    local_out = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        dtype=torch.float32,
    )

    out = backend.combine(
        local_out_fp32=local_out,
        model_dtype=torch.float32,
        m_tokens=2,
        hidden_size=2,
        top_k=2,
        permute_indices=torch.arange(4),
        s_list=[4],
        r_list=[4],
    )

    expected = torch.tensor([[4.0, 6.0], [40.0, 60.0]])
    assert torch.equal(out, expected)


def test_fp8_moe_compute_requires_weight_scales_before_kernel_launch():
    backend = TritonMoEBackend(tp_group=None, ep_group=None)

    with pytest.raises(ValueError, match="w13_weight_scale"):
        backend.compute(
            recv_x=torch.zeros(1, 4, dtype=torch.bfloat16),
            recv_local_ids=torch.zeros(1, dtype=torch.long),
            recv_weights=torch.ones(1, dtype=torch.float32),
            w13_stacked=torch.zeros(1, 8, 4, dtype=torch.uint8),
            w2_stacked=torch.zeros(1, 4, 4, dtype=torch.bfloat16),
            local_num_experts=1,
            local_inter_size=4,
            hidden_size=4,
            model_dtype=torch.bfloat16,
        )
