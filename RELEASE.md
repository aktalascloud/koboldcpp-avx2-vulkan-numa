Haswell / Polaris maintenance · 2026-11-04
- Official experimental1.125: `SOURCE_COMMIT`; EXE SHA256 `NATIVE_SHA256`.82 new official commits/206 files; ancestry and local ports preserved. O3/Haswell/AVX2 + LTO, no PGO, max4 workers.
- Upstream KV/FlashAttention, Vulkan view-buffer/NVFP4 bounds, model/SDCPP and YuE2 fixes. Include [PR2581 Vulkan comm fix](https://github.com/LostRuins/koboldcpp/pull/2581); adapt batched-submit and opt-in HybridMoE bindings. [Review and limits](performance/daily-review.json).
- Source/hash, cold startup, dated visible window and GET APIs verified; user settings and exact profiles preserved. No inference benchmarks. [November2 photo evidence](results/photo-latest.json) retains its own cb260832 SHA and is not a benchmark of this release.
- [All retained patches/licenses](performance/PATCHES.md) · [Media](https://github.com/aktalascloud/koboldcpp-media-pipeline). No weights or private owner bundle mirrored.
