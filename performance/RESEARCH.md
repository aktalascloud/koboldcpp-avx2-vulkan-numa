# Research decisions
Reviewed24October2026:100 updated KoboldCpp PRs, all6 open PRs,20 selected active/starred forks (up to30 PRs and20 default-branch commits each), and13 global GitHub searches. This is bounded discovery, not every fork/file. Existing source pins are in sources.json.

| Source | Decision for Haswell / Windows / RX580 |
|---|---|
| [GCN FA/LDS gates](https://github.com/doteyeso-ops/rx580-vulkan-fa-lds) | Adapted and measured. Enable the GCN proprietary-driver path; accept32KiB LDS on GCN while retaining capability checks. No FP16 spoofing or driver replacement. |
| [Grammar cache PR2434](https://github.com/LostRuins/koboldcpp/pull/2434) | Cache reject indices instead of pointers to caller-owned buffers. Reconstruct candidates from the current call. |
| [Soprano PR2557](https://github.com/LostRuins/koboldcpp/pull/2557) | Fresh-upstream build repair: include implementation from mtmd, not llama-model. No Soprano model bundled or benchmark claimed. |
| [AVX2 reduction PR30296](https://github.com/ggml-org/llama.cpp/pull/30296) | Already present in the base. Retained; not claimed as our new patch. |
| [henk717 stability fixes](https://github.com/henk717/koboldcpp/commit/874f833a54) | Relevant tokenization/logprob lifetime and locking repairs already present. |
| [F16C PR30343](https://github.com/ggml-org/llama.cpp/pull/30343) | Reported Clang/Q4_0/Q8_0 regression does not establish a GCC/IQ4_XS gain. Not imported blindly. |
| [CPU Sets PR6832](https://github.com/ggml-org/llama.cpp/pull/6832) | Windows CPU Sets/groups ideas are already adapted; affinity restored after work. |
| [NUMA mirror PR27986](https://github.com/ggml-org/llama.cpp/pull/27986), [ik PR2396](https://github.com/ikawrakow/ik_llama.cpp/pull/2396) | Linux weight replication/mbind implementations. Not advertised as a Windows dual-socket optimization; physical destination testing remains necessary. |
| [P100 patch collection](https://github.com/shinbunbun/llama-cpp-p100-patches) | Reviewed host/scheduler ideas separately from CUDA kernels. Kobold has its own mark-based penalty sampler. CUDA-specific gains are not RX580 results. |
| [FSR4 gfx803](https://github.com/Schaka/fsr4-gfx803) | Linux RADV/FSR4 driver-layer work, not a drop-in Windows LLM patch. System driver untouched. |
| [HybridMoE](https://github.com/MohammadR3za/HybridMoE) | Opt-in CPU misses/GPU hot experts, adapted to SLRU and new MoE graph guards. Dense media leaves it off. |
| [vkblas](https://github.com/lakitu12/vkblas) | GCN FP32 tile ideas adapted to GGML's own LDS layout. |
| Global alternative engines | Different quantization/model/OS assumptions do not justify replacing the required Kobold backend or protected model. |

Strict-math compiler screening covers vectorization, scheduling, alignment/unrolling and Vulkan host code. Losing variants remain local. Only measured retained policy enters production. SPIR-V already uses upstream optimization except intentionally excluded shader families; unsafe optimization flags were not forced.

GPU sampling, active MoE epilogue fusion, speculative draft and Native HIP remain rejected production options. NUMA changes worker placement, not page allocation; dual-socket throughput is unmeasured.

Performance proof compares only [original stock and latest measured result](benchmarks/latest.json), with all repeated values and matching source/binary identities. No percentages from another repository are presented as our measurements.
