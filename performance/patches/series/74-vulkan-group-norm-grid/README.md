# Extend Vulkan GroupNorm dispatch beyond X

Uses the physical workgroup limits and a bounded linear index for larger batch/group counts. The normalization math is unchanged. Invalid, oversized and unevenly partitioned inputs use the CPU route.

[Source](https://github.com/ggml-org/llama.cpp/pull/31930)
