# Find Q8_K activation peaks with AVX2

Keeps the first maximum on ties, the original scale, rounding and block sums. Other ISA builds and blocks with nonfinite or exclusively tiny values keep the scalar maximum. Inspired by an upstream NEON repacking proposal.

[Source](https://github.com/ggml-org/llama.cpp/pull/32554)
