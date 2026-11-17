# Native F32 SiTU with preserved stage rounding

Adapts MOSS SiTU to the existing GGML GLU dispatch for local CPU/Vulkan only. Adds no foreign extension ABI and changes no existing enum values. F32 layout/capability gates retain the original composite fallback. Precise staged Vulkan arithmetic replaces the rejected non-bit-exact prototype.

[Source](https://github.com/leejet/stable-diffusion.cpp/pull/2210)
