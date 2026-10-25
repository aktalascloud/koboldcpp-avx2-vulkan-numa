# KoboldCpp: Haswell AVX2 / RX 580 Polaris

[Русский](README.ru.md) · **[Windows release](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases/latest)** · [Media pipeline](https://github.com/aktalascloud/koboldcpp-media-pipeline) · [Patches](performance/PATCHES.md) · [Build](performance/BUILD.md)

Performance fork of **[LostRuins/KoboldCpp](https://github.com/LostRuins/koboldcpp)**, strictly **concedo_experimental**. Windows x64, Haswell/AVX2, AMD RX 580 Polaris/gfx803, Vulkan and optional NUMA. Direct EXE launch; no models bundled.

## Stock versus latest measured result

Xeon E5-2696v3 / RX 580 2048SP 8 GB; identical Qwen3-4B IQ4_XS weights. CPU 16/20, Vulkan 6/4; context 512, batch/ubatch 256, 64 output tokens. Both GPU controls use the same upstream graphics-queue option. Values are tokens/s; all repeats, ranges, settings and SHA are in [the proof](performance/benchmarks/latest.json).

| Stage | Original stock | Latest | Change |
|---|---:|---:|---:|
| CPU: Prompt | 39.468 | 85.153 | +115.75% |
| CPU: Output | 16.415 | 17.043 | +3.82% |
| Vulkan: Prompt | 411.010 | 712.810 | +73.43% |
| Vulkan: Output | 17.035 | 23.655 | +38.86% |

Native prompt+decode time, stock/latest, excluding model loading: CPU: **15.287 s / 9.020 s**, 41.00% less time; Vulkan: **4.847 s / 3.334 s**, 31.21% less time.

Measured 2026-10-25, EXE **fb4a6ad1**, source **e68c9b10**. Prompt and output speeds are separate; percentages are not additive. Small overlapping differences are reported without a statistical-significance claim. All repeats, including the first cold control, are retained; CPU ranges overlap. Daily builds are not automatically benchmarked.

Saved photo control, same576×864 / three LoRAs / faces / Gater / no SR: **163.15s stock,138.35s first,131.83s warm**. Warm elapsed time is19.20% lower. This is a dated media measurement, not a new photo test of every native release. [Photos, eyes and complete pipeline](https://github.com/aktalascloud/koboldcpp-media-pipeline).

## Run

Extract the release and double-click **koboldcpp-nocuda.exe**. Select your GGUF in the standard GUI, or run:

```text
koboldcpp-nocuda.exe --model model.gguf --usevulkan 0 --gpulayers 99 --threads 6 --blasthreads 4 --batchsize 1024 --ubatchsize 1024 --numa auto
```

CPU only: replace Vulkan/GPU arguments with **--usecpu --threads 16 --blasthreads 20 --numa auto** (separate option/value with spaces). SD encoder uses 20 independently.

Measured Vulkan environment: GGML_VK_ALLOW_GRAPHICS_QUEUE=1; GGML_VK_BATCHED_SUBMITS=1; GGML_VK_MAX_NODES_PER_SUBMIT=100; GGML_VK_GCN_TILE64=1; GGML_VK_GCN_TILE_VARIANT=k16; GGML_VK_IQ4_XS_ROWS=4; GGML_VK_IQ4_XS_WG=64. Configure through Windows environment settings; no launcher script required.

## What is retained

- O3/Haswell/AVX2, mandatory LTO, **no PGO**, strict math; selected tiled compiler flags.
- Polaris IQ4_XS rows4/WG64, bounded batched submits, FP32 GCN64/K16 and measured GCN driver/LDS tuning.
- Allocation-free small-CDF/reusable large-CDF token sampling; seeded answers and RNG state checked.
- Large 32-bit transposed concat: padded shared-memory tile, measured on RX580; small/general views retain the original path.
- Windows ABI/stack repairs, embedded shader-object invalidation and grammar-cache lifetime fix.
- Optional Windows CPU Sets/groups NUMA and HybridMoE/SLRU. Dense media leaves expert cache off. Physical dual-socket gains are unmeasured.

[Research decisions](performance/RESEARCH.md) distinguish adopted, already-present and inapplicable ideas. GPU sampling, MoE epilogue fusion, draft and Native HIP are not production defaults.

## Updates and licenses

Daily 08:00 Asia/Bishkek: merge official experimental, build, install, verify source/SHA/startup and publish. **Benchmarks are manual only.** The visible window includes the build day. Local maintenance requires the computer/Codex running.

No model weights, credentials, cache, logs or PID files in the core package. [AGPL-3.0](LICENSE.md), [upstream MIT notices](MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md), [third-party notices and corresponding source](performance/NOTICE.md). Exact source is tagged with each release.
