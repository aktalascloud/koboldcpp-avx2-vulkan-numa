# Retain scheduler buffers across graph reservations

Recreates the scheduler only for initial allocation or increased node capacity; explicit pipeline-parallel fallback still recreates it. No model or sampling setting changes.

[Source](https://github.com/ggml-org/llama.cpp/pull/31841)
