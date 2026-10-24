# KoboldCpp performance patches: AVX2 and RX 580 Polaris

**[Русская версия](README.ru.md)**

This is a fork of **[the original KoboldCpp by LostRuins](https://github.com/LostRuins/koboldcpp)**, based strictly on **concedo_experimental**. It contains measured speed improvements for Haswell/AVX2 CPUs and **AMD RX 580 Polaris (gfx803)**, plus optional Windows NUMA and HybridMoE support.

**[Download the Windows build](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases/latest)** · **[Photo/music media pipeline](https://github.com/aktalascloud/koboldcpp-media-pipeline)** · [Patch details](performance/PATCHES.md) · [Build instructions](performance/BUILD.md) · [Original README](README.upstream.md)

Topics: KoboldCpp, llama.cpp, AVX2, Haswell, RX580, Polaris, gfx803, Vulkan, NUMA, speed, performance, patches, PGO and LTO.

<!-- daily-check:start -->
## Current build: CPU compiler tuning, 2026-10-24

Official experimental **6475b2f**, EXE **75b0ed03**. **O3 / Haswell / AVX2 + LTO, without PGO**. Eight compiler flag recipes were compared manually. The selected recipe enables Haswell AVX2 in **only `ggml-tiled.o` and `ggml-tiled-kernel.o`**, which previously used generic C++ flags. It preserves baseline flags for other units and the LTO link, register renaming, alignment 32, hot no-clone and the no-AVX FFI exception. No fast-math or precision change was introduced.

Same Qwen3 4B IQ4_XS, CPU 16/20, context 512, batch/ubatch 256, 64 output tokens. Four runs per arm, alternating forward/reverse order. Values are tokens/s:

| CPU stage | Stock 1.123.1 | Previous 6546333c | New 75b0ed03 |
|---|---:|---:|---:|
| Prompt processing | 39.060 | 41.042 | 89.145 |
| Output generation | 17.460 | 17.470 | 17.405 |

Measured change versus stock: **+128.23% prompt / -0.32% output**. Output ranges: stock [17.36, 17.54]; working [17.34, 17.61]; candidate [17.3, 17.52]. Overlapping ranges do not establish statistical significance. The main gain is prompt processing; do not describe it as a doubling of output-token speed.

Full-20-unit Haswell, function cloning, vector preference, loop unrolling, compact alignment and hot O2 alternatives, including every losing run, are retained in [the complete manual comparison](performance/benchmarks/cpu-compiler-flags.json). Real CPU arithmetic, English, Russian and prompt-director API answers passed for this EXE.

The **Vulkan DLL is byte-identical** to the previous accepted backend; it was reused explicitly rather than claimed as freshly benchmarked. Its last matched manual control remains **17.040 stock / 21.290 optimized output tokens/s**, measured on 6546333c, not a new GPU run. Image/music generation was not repeated. Existing photo timings below retain their original SHA and scope.

The terminal and GUI now show the **embedded build day**, for example `KoboldCpp 1.123.1 [2026-10-24] Terminal`. It is the build date, not the launch date. Daily automation builds, installs, checks startup and publishes; **all performance and generation tests remain manual only**.
<!-- daily-check:end -->

## CPU and GPU: before versus after

Measured on **Xeon E5-2696 v3, RX 580 2048SP 8 GB, Windows**, using the same Qwen3 4B IQ4_XS weights. The short benchmark uses context 512, batch 256 and 64 output tokens. The following table shows **output generation speed**.

| Configuration | Stock KoboldCpp | This build | Speed increase |
|---|---:|---:|---:|
| CPU, identical 16/20 thread settings | 17.34 tokens/s | **17.50 tokens/s** | **+0.92%** |
| Vulkan, identical graphics queue and 6/4 thread settings | 17.09 tokens/s | **21.33 tokens/s** | **+24.81%** |
| Vulkan, stock defaults versus the complete tuned configuration | 12.125 tokens/s | **21.33 tokens/s** | **+75.92%** |

The last row includes both build changes and configuration tuning. The graphics queue option already exists upstream; its entire benefit must not be attributed to these patches.

Input prompt processing also improved:

| Configuration | Before | After |
|---|---:|---:|
| CPU, identical thread settings | 40.90 tokens/s | **42.10 tokens/s** |
| Vulkan, identical graphics queue and thread settings | 410.45 tokens/s | **429.74 tokens/s** |

A real API request with **675 input tokens and 191 output tokens** improved from **10.77 to 8.62 seconds**, taking **19.96% less time** with an identical complete answer.

All four matched CPU controls are included: stock range 17.12–17.49, new range 17.08–17.70 tokens/s. The ranges overlap; the small measured mean gain is retained without claiming statistical certainty. Each GPU configuration has two repetitions. [Recorded results](performance/benchmarks/results.json).

## Photo generation, in seconds

Same scene, seed and **576×864** resolution, with three LoRAs, face refinement and Gater Restore, without upscaling:

| Build / state | Complete image pipeline |
|---|---:|
| Stock KoboldCpp | **163.15 s** |
| This build, first request | **138.35 s** |
| This build, repeated request | **131.83 s** |

The warm pipeline takes **19.20% less time** than stock. The previous optimized build recorded 136.02 s, so the new build's first request was slightly slower than that previous optimized result. This regression is preserved in the evidence.

The current media profile uses a larger **592×896** image: **145.34 s on the first request and 140.09 s on the repeat**, without upscaling. These timings are listed separately because the resolution differs.

**[Photo examples, close-ups of eyes and faces, LoRA settings and Helaman 4× upscaling are in the media pipeline repository](https://github.com/aktalascloud/koboldcpp-media-pipeline).** Publication reused existing comparisons; no new images were generated.

## Run the EXE directly

Extract the Windows release and **double-click koboldcpp-nocuda.exe** to open the standard KoboldCpp GUI. Select your own GGUF model and configure the settings. No PowerShell launcher is required.

Recommended settings for the measured machine:

| Setting | RX 580 / Vulkan | CPU only |
|---|---:|---:|
| Threads | 6 | 16 |
| BLAS threads | 4 | 20 |
| Batch / ubatch | 1024 / 1024 | Tune for the workload |
| GPU layers | 99, when the model fits | 0 |
| NUMA | auto | auto |
| Process priority | Normal | Normal |

Direct Windows command line:

```text
koboldcpp-nocuda.exe --model model.gguf --usevulkan 0 --gpulayers 99 --threads 6 --blasthreads 4 --batchsize 1024 --ubatchsize 1024 --numa auto
```

CPU only:

```text
koboldcpp-nocuda.exe --model model.gguf --usecpu --threads 16 --blasthreads 20 --numa auto
```

For the full measured Polaris configuration, set these Windows environment variables before launching the EXE. The Windows environment editor is sufficient; no script file is needed.

| Variable | Value |
|---|---|
| GGML_VK_ALLOW_GRAPHICS_QUEUE | 1 |
| GGML_VK_BATCHED_SUBMITS | 1 |
| GGML_VK_MAX_NODES_PER_SUBMIT | 100 |
| GGML_VK_GCN_TILE64 | 1 |
| GGML_VK_GCN_TILE_VARIANT | k16 |
| GGML_VK_IQ4_XS_ROWS | 4 |
| GGML_VK_IQ4_XS_WG | 64 |

These values were measured on this RX 580. Tune threads, offload and batch for other hardware. No main model weights are included in Git or the Windows release.

## Optimizations retained

* **O3 / Haswell / AVX2 / FMA / F16C, LTO without PGO (current recipe)**, register renaming, function/loop alignment 32 and hot `-fno-ipa-cp-clone`; CPU-only tiled AVX2/Haswell. Individual compiler flags do not have independently measured percentages.
* **IQ4_XS rows4 / WG64** for single-column Vulkan operations. Within the same candidate: 17.355 versus 21.345 tokens/s with the previous rows8 setting. Weights and precision are unchanged.
* **Batched Vulkan submissions and FP32 GCN64/K16**, retaining the computation watchdog limit. vkblas ideas were adapted to GGML. Warm raw photo generation with K32/K16 took 129.9435 / 129.7727 s; RGB pixels matched across all six checks.
* **Opt-in HybridMoE / SLRU** for MoE models. On the tested OLMoE, one upload reduced the API cycle from 14.5312 to 13.3383 s. Dense Qwen/media keep expert caching disabled. Full GPU offload is faster when the whole model fits.
* **Windows NUMA, CPU Sets and processor groups**, restoring caller affinity. Auto was selected on this single-node host. Physical dual-socket performance was not measured; memory page allocation is unchanged.
* Python/C ABI, stack alignment, AVX exp/silu and shader dependency fixes. These are correctness/build fixes without a separate throughput claim.

**Rejected:** GPU sampling, which was 1–2.4% slower; active MoE/SwiGLU fusion; a separate draft model; and losing CPU variants. Native HIP/ROCm passed restricted text tests, but photo generation crashed in the system AMD HIP library. HIP is excluded from the production release. [Detailed patch results](performance/PATCHES.md).

## Maintenance and licenses

Official **concedo_experimental** commits are merged into this fork. Local maintenance runs daily at **08:00 Asia/Bishkek**, builds a candidate and publishes Windows assets after source, build, hash and startup checks; speed/generation tests are manual only. The computer and Codex must be running for the local schedule.

Runtime caches, logs, PID files, PGO counters, credentials and personal settings are excluded. [sources.json](performance/sources.json) lists all 23 investigated repositories. History review was bounded; a line-by-line audit of every repository is not claimed.

KoboldCpp is licensed under [AGPL-3.0](LICENSE.md). GGML/llama.cpp/stable-diffusion.cpp retain their [upstream MIT notices](MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md). Corresponding source is available at each release tag. [Third-party notices](performance/NOTICE.md).
