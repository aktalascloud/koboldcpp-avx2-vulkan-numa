# KoboldCpp · Haswell AVX2 / Polaris Vulkan / NUMA
[Русский](README.ru.md) · [Windows EXE](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases/latest) · [Build](performance/BUILD.md) · [Patches](performance/PATCHES.md) · [Media pipeline](https://github.com/aktalascloud/koboldcpp-media-pipeline)

Fork of [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp), branch **concedo_experimental**. Windows x64; target Xeon Haswell/AVX2 and RX 580 Polaris/gfx803. Direct GUI/CLI launch: **koboldcpp-nocuda.exe**. No model weights included.

Goal: maximize correct AVX2/Polaris generation throughput while continuously merging official experimental commits and adapting useful security, local-model and fork features. The integrated [media product](https://github.com/aktalascloud/koboldcpp-media-pipeline) provides one portable KoboldCpp.exe, one runtime and a unified on-demand photo/audio/video/effects API; models and settings remain configurable.

**1.125 · 2026-11-07 · diff follow-up · Native `da9c0018`.** Official `6ae75742`, O3/Haswell + LTO. [43 patch diffs](performance/patches/series/) · [Diff review](performance/review-2026-11-07-r1.md) · [New measurement data](results/performance-2026-11-07-r1.json).

Matched warm Normal from `05e9dc7a` to `da9c0018` was 146.334 → 146.483 s in the engine and 155.312 → 155.430 s through the public API. Separate 4× was 5.499 → 5.498 s. All output RGB hashes matched. CPU GELU/GEGLU gains are isolated operation results. [All repeats](results/performance-2026-11-07-r1.json).

## Manual Normal and 4× · 2026-11-06

Same current models, settings, prompt and seed, previous installed fork `2df2c89f` → `dc51f393`. Warm Normal medians use two repeats per arm. Separate 4× medians use two repeats per arm.

| Measure | Before → after |
|---|---:|
| Normal, Native | 146.905 s → 146.391 s |
| Normal, public API | 155.512 s → 155.473 s |
| 4×, public API | 5.715 s → 5.482 s |

All raw, finished and upscaled RGB samples matched exactly. On operation tests, Haswell Q8_0 GEMM took 8–13.5% less time, Vulkan F16 value packing 26.7–28.1% less on the three large shapes, and aligned Polaris Q4_K matmul 1.5–3.2% less. [All repeats, CPU quant, KV/UMT5 checks and rejected trials](results/performance-2026-11-06.json) · [Historical official stock comparison](results/photo-latest.json).

## Last LLM token comparison · 2026-10-25
Date **2026-10-25**, same Qwen3-4B-Instruct-2507-heretic-uncens.i1-IQ4_XS.gguf, CPU 4 / Vulkan 2 repeats per arm. Tokens/s means:

| Stage | Original stock | Last measured | Change |
|---|---:|---:|---:|
| CPU prompt | 39.468 | 85.153 | +115.8% |
| CPU output | 16.415 | 17.043 | +3.8% |
| Vulkan prompt | 411.010 | 712.810 | +73.4% |
| Vulkan output | 17.035 | 23.655 | +38.9% |

Full settings, SHA, repeats/ranges: [measurements](results/measurements.json). CPU ranges overlap; an operation-level gain is not a whole-generation gain. **The current1.125 / 6ae75742 build has no new token benchmark; these historical speeds are not its benchmark.** HIP/ROCm was not selected: media crashed. Physical dual-socket NUMA is unmeasured.

Vulkan starting point: threads 6 / BLAS 4, batch/ubatch 1024, GPU layers 99, NUMA auto. CPU-only 16/20; media encoder 20 separately. Graphics queue 1, batched submits 1, node 100 and gated Polaris controls are described in [RUN.txt](performance/RUN.txt).

Build from a clean checkout: `py -3.12 performance/setup.py --build`. Original pinned toolchains download locally; max 4 compilers, sequential LTO links. [Local-model rules](AGENTS.md). Models, drivers, BIOS and clocks are not changed.

[AGPL/MIT and third-party source/credit](performance/NOTICE.md). Source/toolchain/license files are necessary build inputs, not cleanup targets. Results live only in `results/`.
