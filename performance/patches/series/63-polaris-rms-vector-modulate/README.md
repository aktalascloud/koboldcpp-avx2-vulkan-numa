# Polaris F32 RMS weight and vector modulation fusion

Adapts the MOSS RMS-affine idea to native Vulkan graph matching, preserving the current wave64 reduction and stored MUL/ADD rounding. Explicit F32/vector/device/width/alignment, external-use and overlap gates retain original paths; allocator dependencies keep vector operands alive. No alternate extension ABI or precision setting.

[Source](https://github.com/leejet/stable-diffusion.cpp/pull/2273)
