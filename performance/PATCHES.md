# Working changes and attribution

The exact native diff against experimental99bfa84 is [patches/runtime.patch](patches/runtime.patch), SHA256 e2164b83350bd9dd99836f3fed00905254c441bccabc5b9ddda821728772209b. The full fork source already includes it: **do not apply twice**. [provenance.json](provenance.json) identifies the unchanged released EXE.

| Change | Scope / measured contribution |
|---|---|
| O3/Haswell, PGO20 + LTO, register renaming, align32, hot no-clone |Combined matched CPU17.34 to17.50tok/s; no independent additive percentage for each compiler flag.|
| Win64 ABI/stack/AVX-helper repairs |Correctness fixes; no separate speed claim.|
| IQ4_XS rows4/WG64 |Same candidate17.355 to21.345tok/s; single-column only.|
| Graphics queue |Existing upstream option; runtime recommendation, not original patch.|
| Batched submits/node100 |Adapted from HybridMoE; combined Vulkan result only, no isolated attribution. FLOP watchdog retained.|
| FP32 GCN64/K16 |vkblas-inspired geometry; initial raw142.62 to139.52s, later matched warm K32/K16129.9435/129.7727s. Different experiments, not additive.|
| HybridMoE/SLRU + short-batch fix |Optional MoE only;15000 cache invariants and actual API/ubatch16/32 checks; uploads1 vs0 gives14.5312 to13.3383s on tested OLMoE.|
| MoE next-token CPU prefetch |ponyol/cubetitled idea; dense model does not execute it. No independent speed claim.|
| MoE prefill routing readback bypass |Notebook-inspired optional path; off by default. No independent universal gain.|
| Windows NUMA/groups, CPU Sets, affinity restore, CRT environment bridge |Auto on this single-node host. Dual-socket implementation is not a measured dual-socket gain.|
| Shader dependencies /4-worker cap /object-flag fingerprints |Build reliability and stale-SPIR-V prevention; not runtime token throughput.|

All23 reviewed source repositories and their pins are in [sources.json](sources.json). Review covered bounded history (up to40 commits/100 issue or PR entries per source) and17 selected ik_llama PR diffs, not every line of every project. Linux ROCm, CUDA/Metal and incompatible quantizations were not blindly imported into Windows Vulkan. GPU sampling and MoE fusion are deliberately absent after negative local measurements.
