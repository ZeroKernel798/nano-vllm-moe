# InferX P/D 分离测试报告 4090 llama3.1 8b

| 阶段 | 测试配置 | 吞吐量 (Throughput) | 延迟 (Latency) |
| :--- | :--- | :--- | :--- |
| Prefill | TP=1,EP=1,BS=32,L=512 | 9712.15 tok/s | 1690.26 ms |
| Prefill | TP=1,EP=1,BS=32,L=1024 | 9571.38 tok/s | 2568.51 ms |
| Prefill | TP=1,EP=1,BS=64,L=512 | 9640.74 tok/s | 2551.64 ms |
| Prefill | TP=1,EP=1,BS=64,L=1024 | 9513.60 tok/s | 4307.78 ms |
| Prefill | TP=1,EP=1,BS=128,L=512 | 9578.22 tok/s | 4283.73 ms |
| Prefill | TP=1,EP=1,BS=128,L=1024 | 9470.78 tok/s | 7788.54 ms |
| Prefill | TP=1,EP=1,BS=256,L=512 | 9558.32 tok/s | 7724.58 ms |
| Prefill | TP=1,EP=1,BS=256,L=1024 | 9465.10 tok/s | 14723.61 ms |
| Prefill | TP=1,EP=1,BS=32,L=4096 | 9027.82 tok/s | 8165.76 ms |
| Prefill | TP=1,EP=1,BS=64,L=4096 | 9002.01 tok/s | 15470.00 ms |
| Prefill | TP=1,EP=1,BS=128,L=4096 | 9002.81 tok/s | 30033.86 ms |
| Prefill | TP=1,EP=1,BS=256,L=4096 | 8993.83 tok/s | 59222.55 ms |
| Decode | TP=1,EP=1,BS=1,L=4 | 80.59 tok/s | 12.69 ms |
| Decode | TP=1,EP=1,BS=32,L=4 | 2908.12 tok/s | 11.32 ms |
| Decode | TP=1,EP=1,BS=128,L=4 | 5591.11 tok/s | 23.27 ms |
| Decode | TP=1,EP=1,BS=256,L=4 | 7010.33 tok/s | 56.81 ms |
