# Retained patches and sources

The checkout already contains these changes. Do not reapply [runtime.patch](patches/runtime.patch). [Exact source/EXE hashes and flags](provenance.json).

| Patch | Source / implementation | Scope |
|---|---|---|
| O3 / Haswell / AVX2 / LTO, no PGO | [Build recipe](build.py), [CPU policy](cpu-policy.json) | Strict math; CPU and Vulkan recipes kept separate. |
| Win64 C ABI, stack alignment, no-AVX FFI, inline AVX exp/silu | [ABI/stack diff](patches/runtime.patch), [exports](export_abi.py) | Correctness and portable Haswell execution. |
| Polaris queue/device initialization | [Polaris research](https://github.com/andrewdhannah/vulkan-polaris-llama), [implementation](../ggml/src/ggml-vulkan/ggml-vulkan.cpp) | Feature checks retained; graphics queue itself is an upstream option. |
| IQ4_XS rows4 / WG64 | [Vulkan implementation](../ggml/src/ggml-vulkan/ggml-vulkan.cpp), [shader generator](../ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp) | Single-column Polaris matvec; unchanged weights and precision. |
| GCN32KiB / proprietary-driver gates | [GCN source ideas](https://github.com/doteyeso-ops/rx580-vulkan-fa-lds) | Real capabilities retained; no FP16 spoofing. |
| FP32 GCN64 / K16 tiles | [vkblas inspiration](https://github.com/lakitu12/vkblas), [GGML implementation](../ggml/src/ggml-vulkan/ggml-vulkan.cpp) | GGML's own LDS layout; no vkblas runtime linked. |
| Batched submits / bounded node limit | [HybridMoE patch collection](https://github.com/MohammadR3za/HybridMoE), [implementation](../ggml/src/ggml-vulkan/ggml-vulkan.cpp) | Reduced submission overhead; computation watchdog retained. |
| HybridMoE / SLRU with CPU expert misses | [MIT implementation](https://github.com/MohammadR3za/llama.cpp-hybrid-moe), [adaptation](../src/llama-moe-cache.cpp) | Opt-in; small-batch, norm/rescale, MTP/guidance and environment-bridge guards. Off for dense media. |
| CPU MoE next-token prefetch | [ponyol ideas](https://github.com/ponyol/llama.cpp.patch.speed), [adaptation](../ggml/src/ggml-cpu/ggml-cpu.c) | Read hint only; inactive on dense Qwen. |
| MoE prefill routing-readback bypass | [Performance notebook](https://github.com/pauldmartinphd/llm-performance-engineering-notebook), [adaptation](../src/llama-moe-cache.cpp) | Separate opt-in; not a default dense-model speedup. |
| Windows NUMA CPU Sets / groups | [PR6832](https://github.com/ggml-org/llama.cpp/pull/6832), [adaptation](../ggml/src/ggml-cpu/ggml-cpu.c) | Restores affinity; does not change page placement. Physical dual-socket gain unmeasured. |
| Grammar rejection-cache lifetime / lookup | [PR2434](https://github.com/LostRuins/koboldcpp/pull/2434), [adaptation](../src/llama-grammar.cpp) | Candidate indices instead of stale pointers; dense lookup / sparse restoration. |
| Reusable sampling CDF | [PR1187 idea](https://github.com/ikawrakow/ik_llama.cpp/pull/1187), [independent adaptation](../src/llama-sampler.cpp) | Same GCC 13 normalization/RNG; stack or per-thread storage avoids per-token allocations. |
| Large transposed concat | [PR30591](https://github.com/ggml-org/llama.cpp/pull/30591), [shader](../ggml/src/ggml-vulkan/vulkan-shaders/concat.comp) | Bit-exact 32-bit shared tile; strict large 2D gate and original fallback. |
| Shader invalidation / embedded-object rebuild / 4-worker cap | [Build](build.py), [Makefile](../Makefile), [generator](../ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp) | Avoid stale SPIR-V; compile/link stages bounded separately. |
| Build day in GUI/window title | [Packaging](package.py), [Native](../koboldcpp.py) | Embedded build date; API version unchanged. |

Already in upstream and retained: [Soprano include fix PR2557](https://github.com/LostRuins/koboldcpp/pull/2557), [AVX2 reduction PR30296](https://github.com/ggml-org/llama.cpp/pull/30296), [F16C lookup fix PR30343](https://github.com/ggml-org/llama.cpp/pull/30343), [tokenization/logprob stability](https://github.com/henk717/koboldcpp/commit/874f833a54). These are not presented as new local inventions.

[Stock versus latest proof](benchmarks/latest.json) · [Operation A/B tests and rejected probe](benchmarks/operations.json) · [Research decisions](RESEARCH.md) · [Licenses](NOTICE.md). Individual patch multipliers are not added or attributed to full-model throughput.

## 28 October2026 additions — speed impact unmeasured

|Port|Source|Scope/checks|
|---|---|---|
|CPU F32 repeat row parallelism|[ik PR2757](https://github.com/ikawrakow/ik_llama.cpp/pull/2757), [local ops](../ggml/src/ggml-cpu/ops.cpp)|Large contiguous destinations only; shared planner/kernel gate, strided sources preserved;120 byte-equivalence cases. Different from the original single-thread IQK tree.|
|Vulkan RMS4 cache + partial prefetch|[ROCmFPX commit](https://github.com/HarshaVardhan3002/ROCmFPX/commit/1fe09ec4e3), [shader](../ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp)|At most4 values per lane, unchanged workgroup/reduction order and large-row fallback. Compile/startup checked; no measured speed claim.|
|Startup credential masking|[PR2566](https://github.com/LostRuins/koboldcpp/pull/2566), [source](../koboldcpp.py)|Masks4 fields in a copied Namespace; live credentials unchanged. Synthetic check passed.|

User explicitly authorized these compute ports without a prior speed gate. [Correctness scope](benchmarks/compute-correctness.json). Historic stock/latest speed tables do not measure these additions.

## Native video ports

### 29 October 2026 fixes (KoboldCpp 1.124)

Current additions:

- [Wan5B latent-only export](../otherarch/sdcpp/src/pipeline/diffusion_engine.cpp): full encoder/diffusion, atomic bounded48-channel posterior capture, explicit `wan_latents` operation and empty media. No placeholder RGB, Tiny decoder or C ABI change. The external Pro decoder is part of the separate media pipeline.
- [Wan5B cache runtime gates](../otherarch/sdcpp/src/runtime/cache_dit.hpp): enforce warmup, computation-mask and cache-count restrictions before reusing a condition. Warmup still records full computations for later reuse. Enabled only for Wan5B; photo behavior is unchanged. Real-header C++ controls cover warmup, mask, max-cache and legacy-photo behavior.
- Local Pro+user-adapter/720p/face/RIFE/YuE2 complete API calls finished in190.76s and258.05s for17native frames, not17seconds of footage. These are media-pipeline measurements, not LLM/token or kernel gains. The second call includes82.55s queue/profile readiness; do not attribute that difference to the scheduler.

The initial cache-forwarding comparison below predates the runtime-gate repair. Retain it as historical evidence: it changed pose/appearance and did not establish lossless acceleration. With repaired gates, the tested light0.04/0.08 configurations produced the same latent SHA as the control in one scene; observed native times55.55/64.27s vs57.23s do not prove a consistent cache benefit.

| Fix | Source | Evidence and scope |
|---|---|---|
| Forward video cache settings | [Video adapter](../otherarch/sdcpp/sdtype_adapter.cpp) | Previously parsed cache settings were not copied into video parameters. In one 17-frame Wan5B trial, Cache-DiT0.08 skipped1/4 steps: sampling34.98 to25.85s; Native56.62 to46.62s. The output changes. This is not a lossless or general LLM speed claim. |
| GCC13/MinGW LTO audio destructors | [OmniVoice](../otherarch/audiocpp/upstream/src/models/omnivoice/session.cpp), [VibeVoice](../otherarch/audiocpp/upstream/src/models/vibevoice/session.cpp) | Emit virtual destructors out of line to avoid duplicate COMDAT thunk definitions. CPU and Vulkan DLLs link with LTO; no class layout or virtual-slot changes. No performance gain claimed. |

The new upstream grammar cache already uses candidate indices; its implementation supersedes the earlier local lifetime repair. Upstream startup credential redaction is preserved without a duplicate patch.

[Video cache measurements, exact SHA and all tested configurations](benchmarks/video-cache-20261029.json). These video tests were requested manually; daily builds still perform no generation or speed benchmarks.

|Port|Source|Scope|
|---|---|---|
|JPEG95 Native AVI output|[Adapter](../otherarch/sdcpp/sdtype_adapter.cpp)|Replaces the previous hardcoded JPEG40. Actual AVI quantizer checked; MP4 cannot restore details already lost.|
|Super-Tiny Wan5B decoder|[MIT reference](https://github.com/madebyollin/taehv/tree/011dfc2112197741c540e0bdd5b7b67bcc930771), [native port](../otherarch/sdcpp/src/model/vae/tae.hpp)|ConvNeXt memory blocks, PixelShuffle and checkpoint detection. Structural tests and actual 17/49-frame generation; full numerical parity with original Torch was not tested. Approximate VAE can soften detail.|
|Full encoder / independent Tiny decoder|[Workflow idea](https://github.com/advanced-lvl-up/Rx470-Vega10-Rx580-gfx803-gfx900-fix-AMD-GPU/blob/6547664db8732305530b969b54c9b7d3806c02e1/New-Wan5B-image2vid-workflow-VAE-FIX.json), [local code](../otherarch/sdcpp/src/pipeline/diffusion_engine.cpp)|Opt-in KCPP_WAN_TAE_DECODER; the video Worker validates path/SHA. Normal Wan encoder remains unchanged. Local semantic implementation, no vendor binaries.|
|Optional Polaris convolution WG512|[Local Vulkan tuning](../ggml/src/ggml-vulkan/ggml-vulkan.cpp)|Preserves the 128×128 tile, with AMD GCN/subgroup64/known Polaris/device-limit gates. GGML_VK_GCN_CONV_WG=512; default 256. A small 17-frame gain did not repeat on 49 frames, so it is not globally enabled. AVI was bit-identical in both comparisons.|
|Bounded Wan5B latent snapshot|[Local pipeline](../otherarch/sdcpp/src/pipeline/diffusion_engine.cpp)|Opt-in KCPP_WAN_LATENT_SNAPSHOT captures raw posterior means for decoder comparisons. Worker restricts this to owned video jobs and unique portable paths. Versioned binary,48channels,batch1,at most128MiB. No C ABI change;off by default.|

The separate 64-tile convolution candidate was slower and removed. ROCm/Tensile patches and coopmat2-only RTX optimizations are not presented as RX580 Vulkan gains. Video pipeline measurements and codec quality limits belong to the [media pipeline](https://github.com/aktalascloud/koboldcpp-media-pipeline/blob/main/benchmarks/video.json). The historical token table does not measure these ports.

Additional numerical check: Native Super-Tiny and the pinned MIT PyTorch reference decoded the same generated latent. The last PNG of the17-frame run differed by at most1 RGB level (mean0.0000543);99% of values matched. This is an RGB8 comparison,not full floating-point tensor parity. The snapshot-enabled candidate also retained the prior AVI bytes for the same control. No new Vulkan-kernel speedup is claimed.

##30October2026

- Fresh official experimental `f7eb772d`: YuE2 estimates, portable BF16 audio casts, Qwen3-TTS/Breeze/Ming/NeuTTS integration and Anthropic tool_choice fix.
- GCC13/MinGW LTO destructor port extended to new Qwen3TTSSession after an actual duplicate-thunk link failure. LTO remains enabled; no layout or speed claim.
- Optional Media GUI launcher and `--media` entry open the portable workbench. Core stays directly runnable; model switching/postprocessing live in the separate media kit.
- [PR/issue/fork decisions](daily-review.json):24 configured sources,21 selected forks, no audit errors. No scheduled inference/benchmarks.
