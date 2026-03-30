 4090 llama3.1 8b
   --num-seqs 256 \
    --input-len 1024 \
    --output-len 1024 \
    --max-model-len 4096 \
    --seed 0 \
    --random-lens
w8a16  wall_time=99.906s  prefill_tps=4182.87  decode_tps=2858.90  avg_ttft_ms=0.00  total_gen_tokens=133966
w8a8 wall_time=69.216s  prefill_tps=12908.28  decode_tps=2620.51  avg_ttft_ms=0.00  total_gen_tokens=133966
wall_time=74.900s  prefill_tps=11070.90  decode_tps=2454.68  avg_ttft_ms=0.00  total_gen_tokens=133966