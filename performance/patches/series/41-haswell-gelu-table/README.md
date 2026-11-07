# AVX2 GELU and GEGLU table lookup

Uses AVX2/F16C on the existing FP16 table and retains the scalar NaN path.

[Source](https://github.com/ggml-org/llama.cpp/pull/30232)
