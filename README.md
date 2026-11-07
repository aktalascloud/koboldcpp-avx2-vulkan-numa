# KoboldCpp · Haswell AVX2 / Polaris Vulkan / NUMA
[Русский](README.ru.md) · [Windows EXE](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases/latest) · [Build](performance/BUILD.md) · [Patches](performance/PATCHES.md) · [Media pipeline](https://github.com/aktalascloud/koboldcpp-media-pipeline)

Fork of [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp), branch **concedo_experimental**. Windows x64; target Xeon Haswell/AVX2 and RX 580 Polaris/gfx803. Direct GUI/CLI launch: **koboldcpp-nocuda.exe**. No model weights included.

Goal: maximize correct AVX2/Polaris generation throughput while continuously merging official experimental commits and adapting useful security, local-model and fork features. The integrated [media product](https://github.com/aktalascloud/koboldcpp-media-pipeline) provides one portable KoboldCpp.exe, one runtime and a unified on-demand photo/audio/video/effects API; models and settings remain configurable.

**1.125 · 2026-11-07 · Native `0519dcbf`.** Official base `9958bb2`, O3/Haswell+LTO. [58 separate patches](performance/patches/series/) · [PR/SDCPP/security](performance/review-2026-11-07-r4.md).

## Fresh manual Normal+4× · 2026-11-07

Same current models, settings, prompt and seed 424242; no High. Latest available official rolling 1.125 `db00fc38`, updated November3 → `0519dcbf`.3 Normal calls per arm, first separate, warm median of two;2 separate4× calls per arm with photo resident before every call.

| Stage | Official 1.125 | Fork 0519dcbf | Less time |
|---|---:|---:|---:|
| Native | 176.692 s | 146.621 s | 30.071 s / 17.02% |
| Normal ready | 184.247 s | 155.334 s | 28.913 s / 15.69% |
| Separate 4× | 2.860 s | 2.740 s | ranges overlap |
| Normal+4×* | 187.107 s | 158.073 s | 29.034 s / 15.52% |

*Sum of independently measured Normal and4× medians. Production Pipeline/private Native HTTP includes PNG/base64/JSON; public5001 queue/transport, model loading and photo return are excluded. All raw, finished and upscaled RGB match. SR shares the same media code; overlapping ranges are not an upscaler gain. [Every repeat, first call, resident-SR controls, hashes and limits](results/performance-2026-11-07-r3.json). Older results retain their own date/SHA. [Isolated CPU operations](results/operations-2026-11-07-r4.json) · [Security/Pillow13](results/security-2026-11-07.json).

## Last LLM token comparison · 2026-10-25
Date **2026-10-25**, same Qwen3-4B-Instruct-2507-heretic-uncens.i1-IQ4_XS.gguf, CPU 4 / Vulkan 2 repeats per arm. Tokens/s means:

| Stage | Original stock | Last measured | Change |
|---|---:|---:|---:|
| CPU prompt | 39.468 | 85.153 | +115.8% |
| CPU output | 16.415 | 17.043 | +3.8% |
| Vulkan prompt | 411.010 | 712.810 | +73.4% |
| Vulkan output | 17.035 | 23.655 | +38.9% |

Full settings, SHA, repeats/ranges: [measurements](results/measurements.json). CPU ranges overlap; an operation-level gain is not a whole-generation gain. **The current1.125 / 9958bb2 build has no new token benchmark; these historical speeds are not its benchmark.** HIP/ROCm was not selected: media crashed. Physical dual-socket NUMA is unmeasured.

Vulkan starting point: threads 6 / BLAS 4, batch/ubatch 1024, GPU layers 99, NUMA auto. CPU-only 16/20; media encoder 20 separately. Graphics queue 1, batched submits 1, node 100 and gated Polaris controls are described in [RUN.txt](performance/RUN.txt).

Build from a clean checkout: `py -3.12 performance/setup.py --build`. Original pinned toolchains download locally; max 4 compilers, sequential LTO links. [Local-model rules](AGENTS.md). Models, drivers, BIOS and clocks are not changed.

[AGPL/MIT and third-party source/credit](performance/NOTICE.md). Source/toolchain/license files are necessary build inputs, not cleanup targets. Results live only in `results/`.
