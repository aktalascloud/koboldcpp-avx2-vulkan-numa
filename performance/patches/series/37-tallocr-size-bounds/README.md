# Check tensor allocator size before padding

Prevents size and offset wrap while preserving zero-size allocations and existing failure behavior.

[Source](https://github.com/ggml-org/llama.cpp/pull/26318)
