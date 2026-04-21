# import os
# from glob import glob

# import torch
# from safetensors import safe_open
# from torch import nn


# def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
#     param.data.copy_(loaded_weight)

# def load_model(model: nn.Module, path: str):
#     packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
#     for file in glob(os.path.join(path, "*.safetensors")):
#         with safe_open(file, "pt", "cpu") as f:
#             for weight_name in f.keys():
#                 loaded_weight = f.get_tensor(weight_name)
                
#                 # 拦截 MoE 专家权重的逻辑 
#                 if "mlp.experts" in weight_name:
#                     parts = weight_name.split(".")
#                     layer_idx = parts[2]
#                     expert_id = int(parts[5])
#                     proj_name = parts[6]
                    
#                     prefix = f"model.layers.{layer_idx}.mlp."
                    
#                     mlp_module = model.get_submodule(f"model.layers.{layer_idx}.mlp")

#                     if hasattr(mlp_module, "w13_stacked"):
#                         if proj_name in ["gate_proj", "up_proj"]:
#                             param_name = prefix + "w13_stacked"
#                             shard_id = 0 if proj_name == "gate_proj" else 1
#                             param = model.get_parameter(param_name)
#                             param.weight_loader(param, loaded_weight, expert_id, shard_id)
#                         elif proj_name == "down_proj":
#                             param_name = prefix + "w2_stacked"
#                             param = model.get_parameter(param_name)
#                             param.weight_loader(param, loaded_weight, expert_id)
                        
#                         print(f"Loaded {weight_name} into {param_name}[{expert_id}]")
#                         continue 

#                 found_packed = False
#                 for k in packed_modules_mapping:
#                     if k in weight_name:
#                         v, shard_id = packed_modules_mapping[k]
#                         param_name = weight_name.replace(k, v)
#                         param = model.get_parameter(param_name)
#                         weight_loader = getattr(param, "weight_loader")
#                         weight_loader(param, loaded_weight, shard_id)
#                         found_packed = True
#                         break
                
#                 if not found_packed:
#                     param = model.get_parameter(weight_name)
#                     weight_loader = getattr(
#                         param, "weight_loader", default_weight_loader 
#                     )
#                     weight_loader(param, loaded_weight)


# def print_model(path: str):
#     for file in glob(os.path.join(path, "*.safetensors")):
#         with safe_open(file, "pt", "cpu") as f:
#             for weight_name in f.keys():
#                 print(f"{weight_name} {f.get_tensor(weight_name).shape}")


# if __name__ == "__main__":
#     import argparse

#     argparse = argparse.ArgumentParser(description="nano vllm")
#     argparse.add_argument(
#         "--model-path", type=str, default="/home/zerokernel_ac/huggingface/qwen/Qwen1.5-MoE-A2.7B-Chat"
#     )
#     args = argparse.parse_args()
#     print_model(args.model_path)


import os
import warnings
from glob import glob

import torch
from safetensors import safe_open
from torch import nn

def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
    # 注意：这里的 param 可能是 nn.Parameter 也可能是 Buffer(Tensor)
    param.data.copy_(loaded_weight)

def get_param_or_buffer(model: nn.Module, tensor_name: str):
    """【新增】安全获取器：PyTorch中量化权重通常存为Buffer，get_parameter会找不到"""
    try:
        return model.get_parameter(tensor_name)
    except AttributeError:
        try:
            return model.get_buffer(tensor_name)
        except AttributeError:
            return None

def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    unmatched_keys: list[str] = []
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)
                
                # 拦截 MoE 专家权重的逻辑 (完全保留你的原有逻辑)
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
                            param = get_param_or_buffer(model, param_name)
                            if param is not None:
                                param.weight_loader(param, loaded_weight, expert_id, shard_id)
                        elif proj_name == "down_proj":
                            param_name = prefix + "w2_stacked"
                            param = get_param_or_buffer(model, param_name)
                            if param is not None:
                                param.weight_loader(param, loaded_weight, expert_id)
                        
                        print(f"Loaded {weight_name} into {param_name}[{expert_id}]")
                        continue 

                # --- 【核心新增逻辑：分离基础名称和量化后缀】 ---
                # 例如：将 'model.layers.0.self_attn.q_proj.qweight' 
                # 拆分为 base_name='...q_proj', suffix='.qweight'
                parts = weight_name.rsplit(".", 1)
                base_name = weight_name
                suffix = ""
                # 兼容普通权重(.weight)和AWQ权重
                if len(parts) == 2 and parts[1] in [
                    "qweight",
                    "qzeros",
                    "scales",
                    "weight",
                    "weight_scale",
                    "input_scale",
                ]:
                    base_name = parts[0]
                    suffix = "." + parts[1]

                # --- 组装与映射逻辑 ---
                found_packed = False
                for k in packed_modules_mapping:
                    if k in base_name:  # 重点：用去掉后缀的 base_name 来匹配 (比如 q_proj)
                        v, shard_id = packed_modules_mapping[k]
                        # 重新拼接名称：替换基础名字，然后加上后缀 (比如拼成 qkv_proj.qweight)
                        param_name = base_name.replace(k, v) + suffix
                        
                        param = get_param_or_buffer(model, param_name)
                        if param is not None:
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, loaded_weight, shard_id)
                        else:
                            unmatched_keys.append(weight_name)
                        found_packed = True
                        break
                
                if not found_packed:
                    # 如果不是 packed 模块，也可能是带着 .qweight 后缀的普通层 (如 o_proj.qweight)
                    param = get_param_or_buffer(model, weight_name)
                    if param is not None:
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader 
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        unmatched_keys.append(weight_name)

    if unmatched_keys:
        preview = ", ".join(unmatched_keys[:8])
        suffix = " ..." if len(unmatched_keys) > 8 else ""
        warnings.warn(
            f"{len(unmatched_keys)} unmatched checkpoint keys while loading {path}: {preview}{suffix}",
            stacklevel=2,
        )


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