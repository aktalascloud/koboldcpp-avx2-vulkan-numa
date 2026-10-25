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
