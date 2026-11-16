# Keep AVX2 Gated Delta Net state rows hot

Processes each independent row through the existing multiply, dot and update kernels before advancing. Preserves dot accumulation and snapshots. Enables measured widths 192 and 256, plus width 64 with scalar decay.

[Source](https://github.com/ggml-org/llama.cpp/pull/32028)
