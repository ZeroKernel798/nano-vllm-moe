import atexit
from dataclasses import fields
from time import perf_counter

import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.world_size = config.tp_size * config.ep_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, self.world_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    # def generate(
    #     self,
    #     prompts: list[str] | list[list[int]],
    #     sampling_params: SamplingParams | list[SamplingParams],
    #     use_tqdm: bool = True,
    # ) -> list[str]:
    #     if use_tqdm:
    #         pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
    #     if not isinstance(sampling_params, list):
    #         sampling_params = [sampling_params] * len(prompts)
    #     for prompt, sp in zip(prompts, sampling_params):
    #         self.add_request(prompt, sp)
    #     outputs = {}
    #     prefill_throughput = decode_throughput = 0.0
    #     while not self.is_finished():
    #         t = perf_counter()
    #         output, num_tokens = self.step()
    #         if use_tqdm:
    #             if num_tokens > 0:
    #                 prefill_throughput = num_tokens / (perf_counter() - t)
    #             else:
    #                 decode_throughput = -num_tokens / (perf_counter() - t)
    #             pbar.set_postfix(
    #                 {
    #                     "Prefill": f"{int(prefill_throughput)}tok/s",
    #                     "Decode": f"{int(decode_throughput)}tok/s",
    #                 }
    #             )
    #         for seq_id, token_ids in output:
    #             outputs[seq_id] = token_ids
    #             if use_tqdm:
    #                 pbar.update(1)
    #     outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
    #     outputs = [
    #         {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
    #         for token_ids in outputs
    #     ]
    #     if use_tqdm:
    #         pbar.close()
    #     return outputs
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> dict: # 修改返回类型为 dict，带上统计数据
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        # --- 新增统计变量 ---
        stats = {
            "prefill_tokens": 0,
            "prefill_time": 0.0,
            "decode_tokens": 0,
            "decode_time": 0.0,
            "ttft_sum": 0.0, # 用于计算平均首字延迟
        }
        start_time = perf_counter()
        
        while not self.is_finished():
            t_step_start = perf_counter()
            output, num_tokens = self.step()
            t_step_end = perf_counter()
            step_latency = t_step_end - t_step_start

            if num_tokens > 0: # Prefill 阶段
                stats["prefill_tokens"] += num_tokens
                stats["prefill_time"] += step_latency
                # 粗略估计 TTFT：prefill 完成的时间点减去开始时间
                stats["ttft_sum"] += (t_step_end - start_time) * len(output) 
            else: # Decode 阶段
                stats["decode_tokens"] += abs(num_tokens)
                stats["decode_time"] += step_latency

            if use_tqdm:
                p_tp = stats["prefill_tokens"] / max(stats["prefill_time"], 1e-6)
                d_tp = stats["decode_tokens"] / max(stats["decode_time"], 1e-6)
                pbar.set_postfix({"Prefill": f"{int(p_tp)}tok/s", "Decode": f"{int(d_tp)}tok/s"})
            
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # 整理返回结果
        final_outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        if use_tqdm: pbar.close()

        # 返回包含统计信息的字典
        return {
            "results": final_outputs,
            "stats": {
                "prefill_tps": stats["prefill_tokens"] / max(stats["prefill_time"], 1e-6),
                "decode_tps": stats["decode_tokens"] / max(stats["decode_time"], 1e-6),
                "avg_ttft_ms": (stats["ttft_sum"] / len(prompts)) * 1000,
                "total_time": perf_counter() - start_time
            }
        }