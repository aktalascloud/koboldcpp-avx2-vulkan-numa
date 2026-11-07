# KoboldCpp · Haswell AVX2 / Polaris Vulkan / NUMA
[Русский](README.ru.md) · [Windows EXE](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases/latest) · [Build](performance/BUILD.md) · [Patches](performance/PATCHES.md) · [Media pipeline](https://github.com/aktalascloud/koboldcpp-media-pipeline)

Fork of [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp), branch **concedo_experimental**. Windows x64; target Xeon Haswell/AVX2 and RX 580 Polaris/gfx803. Direct GUI/CLI launch: **koboldcpp-nocuda.exe**. No model weights included.

Goal: maximize correct AVX2/Polaris generation throughput while continuously merging official experimental commits and adapting useful security, local-model and fork features. The integrated [media product](https://github.com/aktalascloud/koboldcpp-media-pipeline) provides one portable KoboldCpp.exe, one runtime and a unified on-demand photo/audio/video/effects API; models and settings remain configurable.

**1.125 · 2026-11-07 · Native `5e7c8de5`.** Official `0041730`, O3/Haswell + LTO. [46 separate patches](performance/patches/series/) · [Completed remaining-source review](performance/review-2026-11-07-r2.md) · [Every measurement](results/performance-2026-11-07-r2.json).

## Manual Normal and 4× · 2026-11-07

Same current weights/settings, prompt and seed: prior fork `da9c0018` → `5e7c8de5`. Three Normal calls and two separate upscales per arm; warm Normal excludes the first call.

| Measure | Before → after |
|---|---:|
| Normal, Native warm | 146.483 s → 146.645 s |
| Normal, public API warm | 155.430 s → 155.695 s |
| 4×, public API | 5.498 s → 5.475 s |

All raw, finished and upscaled RGB hashes match. Native warm is0.11% longer in these controls; no new photo acceleration is claimed. API/SR ranges overlap. [All repeats and limits](results/performance-2026-11-07-r2.json) · [Historical original-stock comparison, 2 November](results/photo-latest.json).

Isolated CPU examples: standard/common greedy chain, vocabulary151936, **302.11 → 286.27 µs**; RMSNorm4096×1×16×2 with16 workers, **141.17 → 15.23 µs**. The original row math is preserved. Windows mapped-file control releases64MiB to8KiB resident with unchanged bytes. These are operation/working-set results, not tokens/s or whole-pipeline gains. [Controls, production ISA proof and preliminary/rejected trials](results/operations-2026-11-07-r2.json).

## Last LLM token comparison · 2026-10-25
Date **2026-10-25**, same Qwen3-4B-Instruct-2507-heretic-uncens.i1-IQ4_XS.gguf, CPU 4 / Vulkan 2 repeats per arm. Tokens/s means:

| Stage | Original stock | Last measured | Change |
|---|---:|---:|---:|
| CPU prompt | 39.468 | 85.153 | +115.8% |
| CPU output | 16.415 | 17.043 | +3.8% |
| Vulkan prompt | 411.010 | 712.810 | +73.4% |
| Vulkan output | 17.035 | 23.655 | +38.9% |

Full settings, SHA, repeats/ranges: [measurements](results/measurements.json). CPU ranges overlap; an operation-level gain is not a whole-generation gain. **The current1.125 / 0041730 build has no new token benchmark; these historical speeds are not its benchmark.** HIP/ROCm was not selected: media crashed. Physical dual-socket NUMA is unmeasured.

Vulkan starting point: threads 6 / BLAS 4, batch/ubatch 1024, GPU layers 99, NUMA auto. CPU-only 16/20; media encoder 20 separately. Graphics queue 1, batched submits 1, node 100 and gated Polaris controls are described in [RUN.txt](performance/RUN.txt).

Build from a clean checkout: `py -3.12 performance/setup.py --build`. Original pinned toolchains download locally; max 4 compilers, sequential LTO links. [Local-model rules](AGENTS.md). Models, drivers, BIOS and clocks are not changed.

[AGPL/MIT and third-party source/credit](performance/NOTICE.md). Source/toolchain/license files are necessary build inputs, not cleanup targets. Results live only in `results/`.
