# Chunked Prefill Optimization

## 目标

验证长 prefill 请求插入到短 decode 流时，chunked prefill 对短请求 inter-token latency 的影响。

## Benchmark 结果

远端证据：`.remote-logs/prefix_chunked_20260518/`。

矩阵：

- 模型：Qwen2.5-3B-Instruct、Qwen2.5-7B-Instruct。
- 请求 A：短 prompt `32` tokens，输出 `128` tokens。
- 请求 B：长 prompt `2K/4K/8K/16K/32K`，输出 `1` token，在 A 生成 8 个 token 后插入。
- 模式：
  - `no_chunk`: 不拆 B prefill。
  - `chunk_prefill_first_128`: 拆成 chunk `128`，但调度仍然 `prefill_first`。
  - `chunk_decode_first_{128,256,512,1024}`: 拆 chunk，并用 `decode_first`。

核心指标：A 的最大 inter-token latency（越低越好）和 B 的 TTFT（越低越好）。

### 3B 结果

| B prompt | no chunk A max ITL | prefill_first 128 A max ITL | decode_first best A max ITL | decode_first 512 A max / B TTFT | decode_first 1024 A max / B TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | `2640.9 ms` | `572.2 ms` | `61.8 ms` | `61.8 / 245.3 ms` | `77.6 / 153.6 ms` |
| 4K | `200.3 ms` | `1098.6 ms` | `62.1 ms` | `62.1 / 491.3 ms` | `82.3 / 309.7 ms` |
| 8K | `402.7 ms` | `2165.2 ms` | `62.5 ms` | `62.9 / 995.2 ms` | `86.7 / 633.2 ms` |
| 16K | `888.0 ms` | `4419.1 ms` | `65.4 ms` | `69.9 / 1999.9 ms` | `103.3 / 1418.1 ms` |
| 32K | `2273.4 ms` | `9069.3 ms` | `63.6 ms` | `83.6 / 4420.9 ms` | `136.6 / 3344.0 ms` |

### 7B 结果

| B prompt | no chunk A max ITL | prefill_first 128 A max ITL | decode_first best A max ITL | decode_first 512 A max / B TTFT | decode_first 1024 A max / B TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | `2255.0 ms` | `473.2 ms` | `52.7 ms` | `72.8 / 286.1 ms` | `120.4 / 238.7 ms` |
| 4K | `387.8 ms` | `891.3 ms` | `52.7 ms` | `75.7 / 585.5 ms` | `127.0 / 490.8 ms` |
| 8K | `778.2 ms` | `1774.7 ms` | `52.0 ms` | `82.3 / 1228.1 ms` | `140.3 / 1032.5 ms` |
| 16K | `1739.6 ms` | `3868.8 ms` | `57.1 ms` | `95.4 / 2649.1 ms` | `165.9 / 2268.1 ms` |
| 32K | `4197.3 ms` | `9221.8 ms` | `57.2 ms` | `121.8 / 6137.8 ms` | `219.2 / 5375.0 ms` |

## 结论

- `decode_first` 是降低短请求 decode 卡顿的关键。chunk `128` 在 3B/7B 全部 B 长度上给出最低 A 最大 ITL，32K 时分别约 `63.6 ms` 和 `57.2 ms`。
- `prefill_first` 即使拆 chunk，也会连续服务 B 的 prefill，长 B 下反而会把 A 最大 ITL 拉到秒级；它不是短 decode 流的延迟优化。
- chunk 越大，B 的 TTFT 越低，但 A 的最大 ITL 越高。`512`/`1024` 是更偏吞吐/TTFT 的折中，`128` 是更偏 decode 响应性的配置。
