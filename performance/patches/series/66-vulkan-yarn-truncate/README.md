# Respect model-selected YaRN truncation on Vulkan

Uses the existing CPU correction-dimension helper with the tensor truncate bit and preserves it in debug clones. Enables nontruncated F32 only after operator qualification;F16 nontruncated paths stay on CPU because the strict numerical control failed. No RoPE setting is changed and unknown remote RPC capabilities stay guarded.

[Source](https://github.com/ggml-org/llama.cpp/pull/30802)
