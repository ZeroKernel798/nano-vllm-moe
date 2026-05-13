# Optimization Notes

Only current refactor notes are kept here. Historical experiment logs were removed to keep the repository readable; the legacy README is archived at `docs/README_legacy.md`.

| File | Purpose |
| --- | --- |
| `4090_quant_stack_opt.md` | Active RTX 4090 / 7B quantization refactor notes |

Current order: MoE stage-stable -> chunked prefill stage-stable -> quantization active -> EP baseline only.
