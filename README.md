# KoboldCpp · Haswell AVX2 / Polaris Vulkan / NUMA
[Русский](README.ru.md) · [Windows EXE](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases/latest) · [Build](performance/BUILD.md) · [Patches](performance/PATCHES.md) · [Media pipeline](https://github.com/aktalascloud/koboldcpp-media-pipeline)

Fork of [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp), branch **concedo_experimental**. Windows x64; target Xeon Haswell/AVX2 and RX 580 Polaris/gfx803. Direct GUI/CLI launch: **koboldcpp-nocuda.exe**. No model weights included.

Goal: maximize correct AVX2/Polaris generation throughput while continuously merging official experimental commits and adapting useful security, local-model and fork features. The integrated [media product](https://github.com/aktalascloud/koboldcpp-media-pipeline) provides one portable KoboldCpp.exe, one runtime and a unified on-demand photo/audio/video/effects API; models and settings remain configurable.

O3/Haswell + LTO, no PGO/fast-math. **Current1.125 / ab00ee52,2026-11-04: build/source/hash, cold startup, visible window and GET acceptance only; no new generation/speed tests.** Official KV/FA, Vulkan safety, model/SDCPP and YuE2 fixes included; [PR2581 Vulkan build fix](https://github.com/LostRuins/koboldcpp/pull/2581), [review/limits](performance/daily-review.json). Retained [ports and attribution](performance/PATCHES.md), monolithic `KCPP_COMMONIZE=0` and explicit C ABI.

## Manual photo comparison · 2026-11-02

Current Ray Q8 + Afterlight0.25/RealismEnhancer1, unchanged encoder/VAE/settings: Normal736×1104,8 Euler/simple, Cache-DiT0.08. Official rolling1.124.2 (`c66550f4`) versus the last measured fork (`cb260832`). Warm model-ready medians,2 repeats per final arm:

| Time | Official stock | Measured cb260832 | Less time |
|---|---:|---:|---:|
| Native photo |176.670s|146.815s|16.90%|
| Native + faces/Gater |184.423s|154.698s|16.12%|

The new Polaris row-SCALE/MOSS FFN-view patch adds **0.47% Native /0.37% pipeline** versus the previous fork, separately from the whole-fork gains. It removes504 materializations in this workload;184 CPU/Vulkan cases and17 unprofiled real generations retained identical RGB. [All repeats, SHA, models, profiler data and limits](results/photo-latest.json) · [Patch](performance/PATCHES.md#manual-photo-optimization--2-november-2026).

The matched table uses common PNG6. The [media backend](https://github.com/aktalascloud/koboldcpp-media-pipeline) separately adds lossless PNG3:4x encoding1.566→1.052s, payload+1.4%, exact RGB/text. Whole SR HTTP ranges overlap; this is not a whole-request gain claim. First calls and return controls are retained, including the trial's slower first Native151.229s versus149.658/149.732s. Model loading/return, public HTTP queue and diagnostic runs are excluded from the table. One prompt/seed and this hardware/model set; no new LLM/video benchmark.

## Last LLM token comparison · 2026-10-25
Date **2026-10-25**, same Qwen3-4B-Instruct-2507-heretic-uncens.i1-IQ4_XS.gguf, CPU 4 / Vulkan 2 repeats per arm. Tokens/s means:

| Stage | Original stock | Last measured | Change |
|---|---:|---:|---:|
| CPU prompt | 39.468 | 85.153 | +115.8% |
| CPU output | 16.415 | 17.043 | +3.8% |
| Vulkan prompt | 411.010 | 712.810 | +73.4% |
| Vulkan output | 17.035 | 23.655 | +38.9% |

Full settings, SHA, repeats/ranges: [measurements](results/measurements.json). CPU ranges overlap; an operation-level gain is not a whole-generation gain. **The current1.125 / ab00ee52 build has no new token benchmark; these historical speeds are not its benchmark.** HIP/ROCm was not selected: media crashed. Physical dual-socket NUMA is unmeasured.

Vulkan starting point: threads 6 / BLAS 4, batch/ubatch 1024, GPU layers 99, NUMA auto. CPU-only 16/20; media encoder 20 separately. Graphics queue 1, batched submits 1, node 100 and gated Polaris controls are described in [RUN.txt](performance/RUN.txt).

Build from a clean checkout: `py -3.12 performance/setup.py --build`. Original pinned toolchains download locally; max 4 compilers, sequential LTO links. [Local-model rules](AGENTS.md). Models, drivers, BIOS and clocks are not changed.

[AGPL/MIT and third-party source/credit](performance/NOTICE.md). Source/toolchain/license files are necessary build inputs, not cleanup targets. Results live only in `results/`.
