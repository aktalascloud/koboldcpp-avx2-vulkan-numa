# Read integer LoRA scalars as FP32

Adds the missing I32 to F32 conversion used by integer LoRA metadata tensors. Existing float and quantized conversions keep their original paths.

[Source](https://github.com/leejet/stable-diffusion.cpp/pull/2263)
