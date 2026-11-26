# Honor the final reclaimed-memory capacity check

After ordinary eviction attempts, synchronize pending Vulkan frees once when the budget permits and honor the detailed final query only when no reclamation remains. No sleep, memory-budget or residency-setting change.

[Source](https://github.com/leejet/stable-diffusion.cpp/pull/2256)
