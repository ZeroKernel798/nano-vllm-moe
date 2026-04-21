# Tests Layout

This directory contains tests and validation scripts.

## quant

Quantization-related tests and debug checks.

- `python tests/quant/test_awqint4.py`

Planned next steps:

- Add `tests/loader/` for checkpoint key mapping and unmatched-key assertions.
- Add `tests/moe/` for dispatch/combine correctness.
- Add `tests/models/` for end-to-end regression checks.
