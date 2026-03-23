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

使用 ep_bench.py 模拟生产环境下随机输入序列，测试Group-Gemm相关实现的性能

**测试配置**
- Hardware: NVIDIA A100-SXM4-80GB (HBM2e / NVLink)
- Model: Qwen1.5-MoE-A2.7B-Chat
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 128–1024 tokens
- Output Length: Randomly sampled between 128–1024 tokens
- Parallelism Strategy: TP = 1, EP = 1

**测试结果:**
|          实现方案          | 总输入 tokens | 总输出 tokens | 解码吞吐（tokens/s）| 总吞吐（tokens/s）|
|---------------------------|---------------|--------------|--------------------|------------------|
|        Python Loop        |     155216    |   153174     |      156.67        |      315.42      |      
|     Triton Group-GEMM     |     155216    |   153174     |      1286.5        |      2590.1      |        
|Triton Group-GEMM w Overlap|     155216    |   153174     |      1328.5        |      2674.7      |   



可使用 ep_tp_bench.py 模拟生产环境下随机输入序列多卡推理测试，横向对比各种并行策略，一键生成测试结果

**测试配置**
- Hardware: NVIDIA A100-SXM4-80GB (HBM2e / NVLink)
- Model: Qwen1.5-MoE-A2.7B-Chat
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 128–512 tokens
- Output Length: Randomly sampled between 128–512 tokens
- Expert Execution Logic: Group-Gemm + Overlap

**测试结果:**
| 并行配置  | 显卡数 | 解码吞吐（tokens/s） | 总吞吐（tokens/s） |测试耗时|
|----------|-------|--------------------|--------------------|-----|
|TP=1, EP=1|   1   |      2240.64       |     4609.59        |34.77s|
|TP=1, EP=2|   2   |      1937.79       |     3986.54        |40.20s|
|TP=2, EP=1|   2   |      2455.39       |     5051.39        |31.72s|
|TP=2, EP=2|   4   |      1821.96       |     3748.26        |42.75s|
|TP=4, EP=1|   4   |      2430.07       |     4999.32        |32.06s|



