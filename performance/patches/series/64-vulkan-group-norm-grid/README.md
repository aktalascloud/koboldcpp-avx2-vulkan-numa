# Validate Vulkan GroupNorm dimensions

Uses the upstream multi-axis workgroup grid when available. Validates nonempty F32 tensors, even group partitioning and bounded shader indexing before Vulkan dispatch.

[Source](https://github.com/ggml-org/llama.cpp/pull/31930)
