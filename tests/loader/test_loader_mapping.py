from __future__ import annotations

import warnings
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm.utils.loader import load_model, parse_moe_expert_weight_name


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, param, loaded_weight, *args):
        self.calls.append((param.shape, loaded_weight.shape, args))
        param.data.copy_(loaded_weight)


class _PackedProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("qweight", torch.zeros(2, 2, dtype=torch.int32))
        self.register_buffer("weight_scale", torch.zeros(2, dtype=torch.float32))
        recorder = _Recorder()
        self.qweight.weight_loader = recorder
        self.weight_scale.weight_loader = recorder
        self.recorder = recorder


class _SelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = _PackedProj()


class _Mlp(nn.Module):
    def __init__(self, register_weight: bool = True):
        super().__init__()
        # used for expert interception path
        if register_weight:
            self.w13_stacked = nn.Parameter(torch.zeros(1, 1))
        else:
            # keep attribute but avoid parameter/buffer registration
            self.w13_stacked = object()


class _Layer(nn.Module):
    def __init__(self, register_moe_weight: bool = True):
        super().__init__()
        self.self_attn = _SelfAttn()
        self.mlp = _Mlp(register_weight=register_moe_weight)


class _DummyModel(nn.Module):
    packed_modules_mapping = {"q_proj": ("qkv_proj", "q")}

    def __init__(self, register_moe_weight: bool = True):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(register_moe_weight=register_moe_weight)])


def _write_ckpt(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> str:
    ckpt = tmp_path / "model.safetensors"
    save_file(tensors, str(ckpt))
    return str(tmp_path)


def test_parse_moe_expert_weight_name():
    parsed = parse_moe_expert_weight_name("model.layers.1.mlp.experts.7.gate_proj.weight")
    assert parsed == {"layer_idx": "1", "expert_id": 7, "proj_name": "gate_proj"}
    assert parse_moe_expert_weight_name("model.layers.1.mlp.gate_proj.weight") is None


def test_loader_maps_packed_qweight_and_fp8_suffix(tmp_path: Path):
    model = _DummyModel(register_moe_weight=False)
    model_dir = _write_ckpt(
        tmp_path,
        {
            "model.layers.0.self_attn.q_proj.qweight": torch.ones(2, 2, dtype=torch.int32),
            "model.layers.0.self_attn.q_proj.weight_scale": torch.ones(2, dtype=torch.float32),
        },
    )

    load_model(model, model_dir)

    recorder = model.model.layers[0].self_attn.qkv_proj.recorder
    assert len(recorder.calls) == 2
    assert torch.equal(model.model.layers[0].self_attn.qkv_proj.qweight, torch.ones(2, 2, dtype=torch.int32))
    assert torch.equal(model.model.layers[0].self_attn.qkv_proj.weight_scale, torch.ones(2, dtype=torch.float32))


def test_loader_warns_on_unmatched_keys(tmp_path: Path):
    model = _DummyModel(register_moe_weight=False)
    model_dir = _write_ckpt(
        tmp_path,
        {
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.ones(1, 1),
            "model.layers.0.unknown.weight": torch.ones(1, 1),
        },
    )

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        load_model(model, model_dir)

    warn_msgs = [str(r.message) for r in records]
    assert any("unmatched checkpoint keys" in msg for msg in warn_msgs)
    assert any("mlp.experts.0.gate_proj.weight" in msg for msg in warn_msgs)
