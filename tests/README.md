# Tests

The test surface is intentionally minimal after the refactor cleanup.

Current policy:

- Add tests only for stable contracts on the active tracks: MoE, chunked prefill, quantization.
- Keep performance validation in remote scripts, not unit tests.
- EP tests should only cover the retained baseline `torch` prepare/finalize path.
- Do not reintroduce old exploratory tests unless the corresponding feature becomes part of the current roadmap.
