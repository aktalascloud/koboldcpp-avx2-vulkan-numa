# KoboldCpp AVX2 / Vulkan / NUMA

Measured Windows fork of [LostRuins/KoboldCpp](https://github.com/LostRuins/koboldcpp), based strictly on **concedo_experimental**. Haswell AVX2 CPU kernels, PGO + LTO, RX580 Vulkan tuning, optional HybridMoE and Windows processor-group/NUMA support. Original upstream README: [README.upstream.md](README.upstream.md).

**[Download Windows CPU/Vulkan releases](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases)** · [Media integration](https://github.com/aktalascloud/koboldcpp-media-pipeline) · [Patch list](performance/PATCHES.md) · [Build](performance/BUILD.md) · [Machine-readable results](performance/benchmarks/results.json)

## Actual measurements

Xeon E5-2696v3, RX580 2048SP 8GB, Windows. Same Qwen3 4B IQ4_XS weights; short controlled token benchmark, context512, batch256, output64. All four matched CPU controls are included. These are recorded tests, not universal hardware promises.

| Backend / control | Stock prompt tok/s | New prompt tok/s | Stock decode tok/s | New decode tok/s | Decode change |
|---|---:|---:|---:|---:|---:|
| CPU, same16/20 threads, four runs |40.900|42.100|17.340|17.500|+0.92%|
| Vulkan, stock defaults10/10 vs tuned6/4 |362.315|429.740|12.125|21.330|+75.92%|
| Vulkan, same graphics queue and6/4 |410.445|429.740|17.090|21.330|+24.81%|
| Native HIP/ROCm, historical diagnostic |not measured|140.57|not measured|14.75|no matched stock comparison|

CPU decode ranges overlap: stock17.12–17.49, new17.08–17.70. The small measured mean gain is retained, without claiming statistical certainty. Vulkan controls have two repeats per case. Graphics queue is an **upstream option**; the +75.92% includes runtime configuration and is not solely a patch effect. Combined percentages must not be added.

The real675-input/191-output-token director request improved **10.7706 to8.6207s** (19.96% less time), with identical complete answers. Russian, English, arithmetic, JSON grammar and1023/1024/1030/2049 batch boundaries passed. Production LLM settings:6/4 threads, batch/ubatch1024/1024. SD text encoding uses its separate20-thread setting.

HIP passed restricted text/API tests once on an earlier source revision, but photo generation crashed in the unchanged system AMD HIP library. **HIP is not shipped or selected.** CUDA/Pascal and a physical dual-socket machine were not measured here. This release contains CPU/Vulkan, not a CUDA-optimized Pascal binary; Pascal users may evaluate Vulkan, without a promised gain.

## What remains enabled

* O3/Haswell/AVX2/FMA/F16C, fresh20-unit PGO, LTO4, register renaming, function/loop alignment32, hot `-fno-ipa-cp-clone`.
* Safe Python C ABI: no AVX in expose.o, inline AVX exp/silu helpers, stack realignment, explicit C exports, static compiler runtimes. No weight/quantization change or blanket fast-math.
* RX580 IQ4_XS single-column rows4/WG64; same-build17.355 to21.345tok/s (+22.99%). Other quantizations/columns retain defaults.
* Batched Vulkan submissions with upstream watchdog limits retained; FP32 GCN64 tile and K16 specialization inspired by vkblas. This is an adapted GGML path, **not a linked vkblas library**. Warm raw photo K32/K16:129.9435/129.7727s; all six outputs have identical RGB. The tiny measured gain is retained.
* Opt-in HybridMoE CPU misses/GPU hot-expert cache adapted to upstream SLRU, short-batch fix and native environment bridge. On temporary OLMoE, one decode upload improved director14.5312 to13.3383s (8.21% less time). Full GPU is faster when the whole model fits. Dense Qwen/media keep this disabled.
* Windows NUMA CPU Sets/groups and affinity restoration; auto selected. This changes thread placement, not page allocation. On this one-node host auto17.185, distribute16.025, isolate16.560tok/s. No physical two-socket speed claim.

## Rejected experiments

GPU sampling: matching answers but1–2.4% slower, including WG1024/trace-off. Active Vulkan MoE SwiGLU fusion:79.060 to78.385tok/s full GPU,25.845 to24.300 hybrid. Both removed from working code. Compact CPU loops, vec/dot PGO exclusions and separate speculative Qwen0.6B were also rejected. Negative results remain in the compact evidence; failed experimental ports are not shipped as working patches.

## Run

Extract the Windows release, download your own GGUF, then run `Launch-Vulkan.ps1 -Model <your-model.gguf>` or `Launch-CPU.ps1 -Model <your-model.gguf>`. No model, token, API credential, cache, log or PID state is distributed. Settings are recommendations for this measured host; tune threads/offload for your machine.

Upstream experimental commits are merged into this fork. Daily local08:00 Asia/Bishkek maintenance publishes a Windows release only after a successful build and SHA-bound correctness/performance acceptance. Failed builds are not promoted. [Reproduction and maintenance rules](performance/BUILD.md).

License: [AGPL-3.0](LICENSE.md); upstream GGML/llama/SD components retain their [MIT notices](MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md). Corresponding source is included at each release tag. See [third-party notices](performance/NOTICE.md).
