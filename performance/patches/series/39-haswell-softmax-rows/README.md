# Spread CPU Softmax rows across heads and batches

Distributes rows when the leading row dimension would leave workers idle, preserving row math and existing paths for full worker use.

[Source](https://github.com/ggml-org/llama.cpp/pull/30923)
