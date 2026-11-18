# KoboldCpp · Haswell AVX2 and Polaris Vulkan

[Русский](README.ru.md) · [Release / Скачать](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases/tag/v1.125-20261118-ec425dc0) · [Guide / Документация](performance/RUN.txt) · [Evidence / Данные](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/v1.125-20261117-d9b2e6b5/results/performance-2026-11-11-kernels.json) · [Upstream](https://github.com/LostRuins/koboldcpp)

Portable Windows x64 inference for Intel Haswell and AMD Polaris / RX 580. This experimental KoboldCpp fork carries compiler, kernel and NUMA patches for local GGUF text and media workloads.

**Quick start.** Extract the Windows ZIP and run `koboldcpp-nocuda.exe`. Supply your own GGUF weights. For an O3/LTO build, run `py -3.12 performance/setup.py --build` and start with the tested configuration in [RUN.txt](performance/RUN.txt).

**Evidence.** In the November 11, 2026 RX 580 comparison, ABDM Normal took 151.281 → 129.601 s and YuE2 Vulkan took 43.729 → 40.155 s. The test record lists the exact binaries and repeats. On Native `ec425dc0` (November 18), the separate ABDM Normal comparison measured warm unprofiled medians of 117.04 and 116.91 s.

[November 18 Vulkan runs / Замеры Vulkan 18 ноября](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/v1.125-20261118-ec425dc0/results/vulkan-rx580-20261118.json) · [Patch series / Патчи](performance/patches/series/) · [Licenses and corresponding source / Лицензии и исходники](performance/NOTICE.md) · [Media pipeline / Медиа](https://github.com/aktalascloud/koboldcpp-media-pipeline)

[All projects / Все проекты](https://github.com/aktalascloud)
