# Vulkan RMSNorm beyond the X workgroup limit

Uses a separate row-stride shader for long RMSNorm and RMSNorm+MUL, with ordinary pipelines for smaller tensors.

[Source](https://github.com/0xShug0/audio.cpp/pull/1120)
