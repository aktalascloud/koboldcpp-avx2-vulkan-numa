# Reject zero ranks in recurrent model metadata

Fails during model loading for zero SSM time-step rank in Mamba2, Granite Hybrid and Nemotron-H, and zero RWKV6 head size. Other GGUF shape rules are unchanged.

[Source](https://github.com/ggml-org/llama.cpp/pull/31948)
