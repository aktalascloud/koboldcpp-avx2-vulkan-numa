# CPU Hadamard12/20 with typed strided input

Adapts the Kronecker FWHT path to existing F32/F16 dispatch and four-dimensional byte strides instead of the proposed flat F32 cast. Power-of-two transforms keep their existing numerical algorithm.

[Source](https://github.com/ggml-org/llama.cpp/pull/28490)
