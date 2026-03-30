# InferX P/D 分离测试报告 4090 llama3.1 8b

| 阶段 | 测试配置 | 吞吐量 (Throughput) | 延迟 (Latency) |
| :--- | :--- | :--- | :--- |
| Prefill | TP=1,EP=1,BS=32,L=512 | 10994.44 tok/s | 1493.12 ms |
| Prefill | TP=1,EP=1,BS=32,L=1024 | 10840.78 tok/s | 2265.92 ms |
| Prefill | TP=1,EP=1,BS=64,L=512 | 10924.44 tok/s | 2251.42 ms |
| Prefill | TP=1,EP=1,BS=64,L=1024 | 10800.84 tok/s | 3791.86 ms |
| Prefill | TP=1,EP=1,BS=128,L=512 | 10882.32 tok/s | 3767.76 ms |
| Prefill | TP=1,EP=1,BS=128,L=1024 | 10738.85 tok/s | 6863.47 ms |
| Prefill | TP=1,EP=1,BS=256,L=512 | 10818.95 tok/s | 6818.34 ms |
| Prefill | TP=1,EP=1,BS=256,L=1024 | 10650.16 tok/s | 13073.48 ms |
| Prefill | TP=1,EP=1,BS=32,L=4096 | 10081.92 tok/s | 7309.96 ms |
| Prefill | TP=1,EP=1,BS=64,L=4096 | 10048.54 tok/s | 13851.22 ms |
| Prefill | TP=1,EP=1,BS=128,L=4096 | 10005.16 tok/s | 27020.60 ms |
| Prefill | TP=1,EP=1,BS=256,L=4096 | 9991.53 tok/s | 53298.34 ms |
| Decode | TP=1,EP=1,BS=1,L=4 | 60.40 tok/s | 16.69 ms |
| Decode | TP=1,EP=1,BS=32,L=4 | 1735.19 tok/s | 18.57 ms |
| Decode | TP=1,EP=1,BS=128,L=4 | 5513.71 tok/s | 23.57 ms |
| Decode | TP=1,EP=1,BS=256,L=4 | 7686.51 tok/s | 33.99 ms |
