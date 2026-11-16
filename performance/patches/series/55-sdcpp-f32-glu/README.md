# F32 native GLU for packed activations and LLM MLP

Adopts native packed/split SwiGLU only for the tested F32 contiguous-row layouts. Preserves GELU and unusual-layout/type paths.224 real CPU/Vulkan graph comparisons are bit-exact; timing evidence is operator-level only.

[Source](https://github.com/leejet/stable-diffusion.cpp/pull/2250)
