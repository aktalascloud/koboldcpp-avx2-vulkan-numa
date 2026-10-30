# KoboldCpp: Haswell AVX2 / RX580 Polaris

[Р СѓСЃСЃРєРёР№](README.ru.md) В· [Windows download](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases/latest) В· [All patches and sources](performance/PATCHES.md) В· [Media pipeline](https://github.com/aktalascloud/koboldcpp-media-pipeline)

Performance fork of [LostRuins/KoboldCpp](https://github.com/LostRuins/koboldcpp), based on **concedo_experimental**. Windows x64, Haswell/AVX2, RX580 Polaris/gfx803, Vulkan and optional Windows NUMA. **O3 + LTO, no PGO.** No main model weights included.

30October: official experimental `f7eb772d`, Qwen3-TTS LTO thunk fix and optional **Media** launcher (`--media`). [Daily decisions](performance/daily-review.json). No new speed tests.

30October update: official experimental `f7eb772d`, Qwen3-TTS LTO thunk fix and optional **Media** launcher tab (`--media`). [Daily review and source links](performance/daily-review.json). The portable media kit supplies unified APIs, faces/upscale/video/FPS/audio; no new speed tests.

Daily28October: CPU row-copy parallelism and bounded Vulkan RMS cache/prefetch are enabled by user request; speed impact is unmeasured. [Checks/scope](performance/benchmarks/compute-correctness.json).


## Stock versus latest measured result

Xeon E5-2696v3 / RX580 2048SP 8GB; identical Qwen3-4B IQ4_XS. Tokens/s, measured 2026-10-25, EXE `fb4a6ad1`.

| Stage | Original stock | Last measured | Change |
|---|---:|---:|---:|
| CPU prompt | 39.468 | 85.153 | +115.75% |
| CPU output | 16.415 | 17.043 | +3.82% |
| Vulkan prompt | 411.010 | 712.810 | +73.43% |
| Vulkan output | 17.035 | 23.655 | +38.86% |

CPU threads 16/20, Vulkan 6/4; context 512, batch/ubatch 256, 64 output tokens. CPU 4 / Vulkan 2 repeats per arm; first cold controls and every range retained. CPU ranges overlap. [Settings, all repeats and hashes](performance/benchmarks/latest.json). These totals include earlier retained optimizations; they are not the gain of the newest patches alone.

## Retained improvements

- Haswell/AVX2 compiler tuning; Win64 ABI/stack repairs; Polaris matvec, FP32 tiles and bounded submissions.
- Windows CPU Sets/groups; optional HybridMoE/SLRU; grammar-cache safety; shader rebuild checks.
- [CDF sampling](https://github.com/ikawrakow/ik_llama.cpp/pull/1187), independently adapted: **3.17x** faster 40-candidate selection; RNG and seeded answers checked.
- [Transposed concat](https://github.com/ggml-org/llama.cpp/pull/30591): **3.04вЂ“4.30x** faster tested large 32-bit operations, bit-exact; small/general shapes retain the original path.

The last two figures describe isolated operations, not whole-generation multipliers. [Operation proof, including rejected variants](performance/benchmarks/operations.json) В· [Complete patch list with source links](performance/PATCHES.md).

## Run

Extract the Windows release and double-click **koboldcpp-nocuda.exe**, or run:

```text
koboldcpp-nocuda.exe --model model.gguf --usevulkan 0 --gpulayers 99 --threads 6 --blasthreads 4 --batchsize 1024 --ubatchsize 1024 --numa auto
```

CPU only: `--usecpu --threads 16 --blasthreads 20 --numa auto`. [Environment and settings](performance/RUN.txt) В· [Build from source](performance/BUILD.md). No PowerShell launcher required.

Saved photo control: 163.15s stock; 138.35s cold / 131.83s warm, 576Г—864 with the same historical three-LoRA pipeline, 23 October. This is not a new MOSS/photo benchmark. [Current media pipeline and examples](https://github.com/aktalascloud/koboldcpp-media-pipeline).

Daily 08:00 Asia/Bishkek maintenance checks PRs/issues independently of upstream changes, builds and checks startup. Benchmarks are manual only; physical dual-socket gains remain unmeasured.

[AGPL-3.0](LICENSE.md) В· Audio.cpp Apache-2.0 В· [MIT notices](MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md) В· [Dependencies, attribution and corresponding source](performance/NOTICE.md). Upstream tiny decoder assets retain their MIT notices. No keys, private state, logs or cache in releases.

<!-- native-video-ports -->

Native video: JPEG95, optional Super-TinyVAE and full-encoder/Tiny-decoder split. Approximate decoding may soften details; [ports and attribution](performance/PATCHES.md#native-video-ports). No new token-throughput claim.

Optional media integration now uses the original Image Gen and Audio tabs plus Video; Extra has RU/EN and logon startup. Load/Save uses user-selected .kcpps files. The standalone executable retains the original model selectors and icon; media dependencies are in the separate [media pipeline](https://github.com/aktalascloud/koboldcpp-media-pipeline). No new CPU/Vulkan speed measurement accompanies this GUI update.
