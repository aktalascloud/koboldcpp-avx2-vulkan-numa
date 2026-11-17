# KoboldCpp · Haswell AVX2 / Polaris Vulkan

[Русский](README.ru.md) · [Windows release](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases/latest) · [Build](performance/BUILD.md) · [Patch series](performance/patches/series/) · [Media API](https://github.com/aktalascloud/koboldcpp-media-pipeline)

Windows x64 fork of [LostRuins/KoboldCpp](https://github.com/LostRuins/koboldcpp), branch `concedo_experimental`. Targets Haswell AVX2 and AMD RX 580 Polaris. Run `koboldcpp-nocuda.exe`. Weights are external.

**1.125 · 2026-11-17 · Native d9b2e6b5** · [Upstream and startup evidence](results/mainstream-2026-11-17.json).

Use `py -3.12 performance/setup.py --build` for O3/LTO builds and [RUN.txt](performance/RUN.txt) for the tested configuration.

The November 11 build reduced ABDM Normal time from 151.281 to 129.601 s and YuE2 Vulkan request time from 43.729 to 40.155 s. The records include repeats, exact executable hashes and CPU results. [Dated measurements](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/v1.125-20261117-d9b2e6b5/results/performance-2026-11-11-kernels.json)

[License and corresponding source notices](performance/NOTICE.md).
