# InferX P/D 分离测试报告 4090 llama3.1 8b

| 阶段 | 测试配置 | 吞吐量 (Throughput) | 延迟 (Latency) |
| :--- | :--- | :--- | :--- |
| Prefill | TP=1,EP=1,BS=32,L=512 | 11209.68 tok/s | 1464.45 ms |
| Prefill | TP=1,EP=1,BS=32,L=1024 | 11109.80 tok/s | 2214.53 ms |
| Prefill | TP=1,EP=1,BS=64,L=512 | 11215.26 tok/s | 2195.61 ms |
| Prefill | TP=1,EP=1,BS=64,L=1024 | 11097.44 tok/s | 3694.66 ms |
| Prefill | TP=1,EP=1,BS=128,L=512 | 11199.99 tok/s | 3664.55 ms |
| Prefill | TP=1,EP=1,BS=128,L=1024 | 11079.48 tok/s | 6660.69 ms |
| Prefill | TP=1,EP=1,BS=256,L=512 | 11179.05 tok/s | 6607.77 ms |
| Prefill | TP=1,EP=1,BS=256,L=1024 | 11055.62 tok/s | 12608.54 ms |
| Prefill | TP=1,EP=1,BS=32,L=4096 | 10452.48 tok/s | 7053.88 ms |
| Prefill | TP=1,EP=1,BS=64,L=4096 | 10427.11 tok/s | 13357.75 ms |
| Prefill | TP=1,EP=1,BS=128,L=4096 | 10400.08 tok/s | 26000.77 ms |
| Prefill | TP=1,EP=1,BS=256,L=4096 | 10396.32 tok/s | 51231.13 ms |
| Decode | TP=1,EP=1,BS=1,L=4 | 79.53 tok/s | 13.00 ms |
| Decode | TP=1,EP=1,BS=32,L=4 | 2303.03 tok/s | 14.31 ms |
| Decode | TP=1,EP=1,BS=128,L=4 | 6810.84 tok/s | 19.18 ms |
| Decode | TP=1,EP=1,BS=256,L=4 | 9404.59 tok/s | 27.78 ms |
