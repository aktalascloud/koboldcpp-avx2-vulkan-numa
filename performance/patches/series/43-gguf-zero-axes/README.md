# Preserve zero-sized GGUF tensor axes

Trims only trailing ones when serializing tensor metadata, so empty tensors retain their shape and cannot consume a following tensor on reload.

[Source](https://github.com/ggml-org/llama.cpp/pull/31845)
