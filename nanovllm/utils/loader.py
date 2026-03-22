import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)
                
                # 拦截 MoE 专家权重的逻辑 
                if "mlp.experts" in weight_name:
                    parts = weight_name.split(".")
                    layer_idx = parts[2]
                    expert_id = int(parts[5])
                    proj_name = parts[6]
                    
                    prefix = f"model.layers.{layer_idx}.mlp."
                    
                    mlp_module = model.get_submodule(f"model.layers.{layer_idx}.mlp")

                    if hasattr(mlp_module, "w13_stacked"):
                        if proj_name in ["gate_proj", "up_proj"]:
                            param_name = prefix + "w13_stacked"
                            shard_id = 0 if proj_name == "gate_proj" else 1
                            param = model.get_parameter(param_name)
                            param.weight_loader(param, loaded_weight, expert_id, shard_id)
                        elif proj_name == "down_proj":
                            param_name = prefix + "w2_stacked"
                            param = model.get_parameter(param_name)
                            param.weight_loader(param, loaded_weight, expert_id)
                        
                        print(f"Loaded {weight_name} into {param_name}[{expert_id}]")
                        continue 

                found_packed = False
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, loaded_weight, shard_id)
                        found_packed = True
                        break
                
                if not found_packed:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader 
                    )
                    weight_loader(param, loaded_weight)


def print_model(path: str):
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                print(f"{weight_name} {f.get_tensor(weight_name).shape}")


if __name__ == "__main__":
    import argparse

    argparse = argparse.ArgumentParser(description="nano vllm")
    argparse.add_argument(
        "--model-path", type=str, default="/home/zerokernel_ac/huggingface/qwen/Qwen1.5-MoE-A2.7B-Chat"
    )
    args = argparse.parse_args()
    print_model(args.model_path)
