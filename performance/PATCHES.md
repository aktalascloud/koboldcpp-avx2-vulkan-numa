# Working changes and attribution

The exact native diff against experimental6475b2f is [patches/runtime.patch](patches/runtime.patch), SHA256 9c694a90e7243e9f534f582da591ef7b6fa72afd55727e7167a2b8a43e75a51f. The full fork source already includes it: **do not apply twice**. [provenance.json](provenance.json) identifies the current released EXE.

| Change | Scope / measured contribution |
|---|---|
| Historical23October: O3/Haswell, PGO20 + LTO, register renaming, align32, hot no-clone |Combined matched CPU17.34 to 17.50 tok/s; no independent additive percentage for each compiler flag.|
| Win64 ABI/stack/AVX-helper repairs |Correctness fixes; no separate speed claim.|
| IQ4_XS rows4/WG64 |Same candidate17.355 to 21.345 tok/s; single-column only.|
| Graphics queue |Existing upstream option; runtime recommendation, not original patch.|
| Batched submits/node100 |Adapted from HybridMoE; combined Vulkan result only, no isolated attribution. FLOP watchdog retained.|
| FP32 GCN64/K16 |vkblas-inspired geometry; initial raw142.62 to 139.52 s, later matched warm K32/K16129.9435/129.7727 s. Different experiments, not additive.|
| HybridMoE/SLRU + short-batch fix |Optional MoE only;15000 cache invariants and actual API/ubatch16/32 checks; uploads1 vs0 gives14.5312 to 13.3383 s on tested OLMoE.|
| MoE next-token CPU prefetch |ponyol/cubetitled idea; dense model does not execute it. No independent speed claim.|
| MoE prefill routing readback bypass |Notebook-inspired optional path; off by default. No independent universal gain.|
| Windows NUMA/groups, CPU Sets, affinity restore, CRT environment bridge |Auto on this single-node host. Dual-socket implementation is not a measured dual-socket gain.|
| Shader dependencies /4-worker cap /object-flag fingerprints |Build reliability and stale-SPIR-V prevention; not runtime token throughput.|

All23 reviewed source repositories and their pins are in [sources.json](sources.json). Review covered bounded history (up to 40 commits/100 issue or PR entries per source) and17 selected ik_llama PR diffs, not every line of every project. Linux ROCm, CUDA/Metal and incompatible quantizations were not blindly imported into Windows Vulkan. GPU sampling and MoE fusion are deliberately absent after negative local measurements.

Current24October recipe: O3/Haswell/AVX2+LTO without PGO. Fresh manually requested comparisons are in benchmarks/current.json. Historical per-patch figures above are not new measurements of the current EXE. Daily builds perform build/startup checks; speed and generation tests require a manual request.

CPU compiler update 2026-10-24: selected tiled-only Haswell/vector256 with unchanged strict math, no PGO and mandatory LTO. See [all eight recipes and complete manual runs](benchmarks/cpu-compiler-flags.json); prompt/output changes versus stock are +128.23% / -0.32%. Other CPU units retain baseline flags; Vulkan DLL reused byte-for-byte. Window titles display the embedded build day.
