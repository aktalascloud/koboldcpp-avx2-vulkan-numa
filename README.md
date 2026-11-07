# KoboldCpp · Haswell AVX2 / Polaris Vulkan / NUMA
[Русский](README.ru.md) · [Windows EXE](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases/latest) · [Build](performance/BUILD.md) · [Patches](performance/PATCHES.md) · [Media pipeline](https://github.com/aktalascloud/koboldcpp-media-pipeline)

Fork of [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp), branch **concedo_experimental**. Windows x64; target Xeon Haswell/AVX2 and RX 580 Polaris/gfx803. Direct GUI/CLI launch: **koboldcpp-nocuda.exe**. No model weights included.

Goal: maximize correct AVX2/Polaris generation throughput while continuously merging official experimental commits and adapting useful security, local-model and fork features. The integrated [media product](https://github.com/aktalascloud/koboldcpp-media-pipeline) provides one portable KoboldCpp.exe, one runtime and a unified on-demand photo/audio/video/effects API; models and settings remain configurable.

**1.125 · 2026-11-07 · Native `efddce0e`.** Official `2d46974`,O3/Haswell+LTO,no PGO. [65 separate patches](performance/patches/series/) · [332 PR decisions and review depth](performance/review-2026-11-07-r5.md).

## Manual Normal+4× · 2026-11-07

Latest available official rolling1.125 `db00fc38`,updated November3 → `efddce0e`. Three arms:stock,previous0519dcbf,new EXE;the same current models,settings,prompt and seed424242,no High.3 Normal/arm,first separate,warm median of two;2 separate4×/arm with photo resident before every call.

| Stage | Official1.125 | Fork efddce0e | Less time |
|---|---:|---:|---:|
| Native | 176.810 s | 143.500 s | 33.310 s / 18.84% |
| Normal ready | 185.187 s | 152.333 s | 32.854 s / 17.74% |
| Separate4× | 2.838 s | 2.772 s | 0.066 s / 2.33% |
| Normal+4×* | 188.025 s | 155.105 s | 32.920 s / 17.51% |

*Sum of independently measured phase medians,not one joint API request. Production Pipeline/private Native HTTP includes PNG/base64/JSON;public5001 queue/transport,model loading and photo return are excluded. All9 raw/finished RGB and6 SR outputs match. Against previous0519dcbf,the sum saves **2.745s / 1.74%**;separate4× takes **33.5ms longer**. SR shares identical media code;this is not an upscaler-kernel gain. [Every repeat,range,load,hash and limit](results/performance-2026-11-07-r4.json). Arm order is fixed,sample size small.

## Last LLM token comparison · 2026-10-25
Date **2026-10-25**, same Qwen3-4B-Instruct-2507-heretic-uncens.i1-IQ4_XS.gguf, CPU 4 / Vulkan 2 repeats per arm. Tokens/s means:

| Stage | Original stock | Last measured | Change |
|---|---:|---:|---:|
| CPU prompt | 39.468 | 85.153 | +115.8% |
| CPU output | 16.415 | 17.043 | +3.8% |
| Vulkan prompt | 411.010 | 712.810 | +73.4% |
| Vulkan output | 17.035 | 23.655 | +38.9% |

Full settings, SHA, repeats/ranges: [measurements](results/measurements.json). CPU ranges overlap; an operation-level gain is not a whole-generation gain. **The current1.125 / 2d46974 build has no new token benchmark; these historical speeds are not its benchmark.** HIP/ROCm was not selected: media crashed. Physical dual-socket NUMA is unmeasured.

Vulkan starting point: threads 6 / BLAS 4, batch/ubatch 1024, GPU layers 99, NUMA auto. CPU-only 16/20; media encoder 20 separately. Graphics queue 1, batched submits 1, node 100 and gated Polaris controls are described in [RUN.txt](performance/RUN.txt).

Build from a clean checkout: `py -3.12 performance/setup.py --build`. Original pinned toolchains download locally; max 4 compilers, sequential LTO links. [Local-model rules](AGENTS.md). Models, drivers, BIOS and clocks are not changed.

[AGPL/MIT and third-party source/credit](performance/NOTICE.md). Source/toolchain/license files are necessary build inputs, not cleanup targets. Results live only in `results/`.
