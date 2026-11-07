# Reserve the small Vulkan BAR for transfers

Detects whether the main device-local heap is host-visible and keeps the existing UMA and explicit host-memory paths.

[Source](https://github.com/ggml-org/llama.cpp/pull/31658)
