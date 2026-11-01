Haswell / Polaris maintenance update
- Official experimental source: `SOURCE_COMMIT`. EXE SHA256: `NATIVE_SHA256`.
- Selected MOSS QKV/slice/position-cache optimizations ([PR2576](https://github.com/LostRuins/koboldcpp/pull/2576)); independent legacy AMD readback fallback ([PR2564](https://github.com/LostRuins/koboldcpp/pull/2564)). AVX2 + O3 + LTO, no PGO.
- [All retained patches and licenses](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/concedo_experimental/performance/PATCHES.md). No new token-speed claim; dated stock/latest evidence stays in README/results.
- Run the EXE directly. Models are not included. [Media pipeline](https://github.com/aktalascloud/koboldcpp-media-pipeline).
