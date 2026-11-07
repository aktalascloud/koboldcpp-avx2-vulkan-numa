# Duplicate-safe Vulkan TopK masks without half arithmetic

Adapts the native mask op using an8KiB shared atomic bitset and one writer per output word. Preserves mask bits including F16 payloads and rejects unsupported odd/misaligned F16 layouts or oversized/empty input buffers. No duplicate scatter races,half arithmetic or feature spoofing.

[Source](https://github.com/ggml-org/llama.cpp/pull/31544)
