# Nano-vLLM-MOE

本项目是以Nano-vLLM项目为基底，添加了对Qwen2、Qwen3系列的Moe模型支持，并提供了EP、TP以及EP、TP混合的调度实现。用于在现有的Nano-vLLM的基础上，学习和分析LLM的分布式并行策略。原项目地址为：https://github.com/GeeeekExplorer/nano-vllm.git

## 主要特点

* 支持多种并行策略：支持单机多卡下的 TP、EP 以及 TP+EP 混合调度，可应对更复杂的分布式推理场景。
* 高性能Triton算子：使用 Triton 实现了 Group-Gemm 等核心算子，显著降低 MoE 层通信延迟。
* 特性继承：完整保留了原项目的 Prefix caching, Torch compilation 以及 CUDA graph 等生产级优化。

## 安装

git clone https://github.com/ZeroKernel798/nano-vllm-moe.git

cd nano-vllm-moe

pip install -e .

## 模型下载

推荐使用内置的下载脚本从 ModelScope 极速获取模型

python download_model.py --model qwen/Qwen1.5-MoE-A2.7B-Chat --path ./my_models


## Benchmark

可使用ep_tp_bench.py进行生产环境下的随机输入序列模拟测试

**测试配置**
- Hardware: NVIDIA A100-SXM4-80GB (HBM2e / NVLink)
- Model: Qwen1.5-MoE-A2.7B-Chat
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 128–512 tokens
- Output Length: Randomly sampled between 128–512 tokens

**测试结果:**
| 并行配置 | 显卡数 | 解码吞吐（tokens/s） | 总吞吐（tokens/s） |测试耗时|
|---------|--------|--------------------|--------------------|-----|
|TP=1, EP=1|Cards=1|Out_TP=2240.64|Total_TP=4609.59|Time=34.77s|
|TP=1, EP=2|Cards=2|Out_TP=1937.79|Total_TP=3986.54|Time=40.20s|
|TP=2, EP=1|Cards=2|Out_TP=2455.39|Total_TP=5051.39|Time=31.72s|
|TP=2, EP=2|Cards=4|Out_TP=1821.96|Total_TP=3748.26|Time=42.75s|
|TP=4, EP=1|Cards=4|Out_TP=2430.07|Total_TP=4999.32|Time=32.06s|


