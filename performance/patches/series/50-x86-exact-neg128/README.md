# Exact x86 int8 dot products at minus 128

Places full-range activations on the unsigned-magnitude side for bounded weights; Q8 by Q8 alone uses a guarded widening path. No clipping or quantizer changes. Other excluded IQ2/IQ3/repack kernels are not claimed fixed.

[Source](https://github.com/ggml-org/llama.cpp/pull/29359)
