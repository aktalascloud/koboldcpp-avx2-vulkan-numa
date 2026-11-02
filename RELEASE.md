Haswell / Polaris maintenance update · 2026-11-02
- Official experimental source: `SOURCE_COMMIT`. EXE SHA256: `NATIVE_SHA256`.
- Nine new official commits: YuE2 VRAM lifetime/cover fixes, SheetSage2 engine support, WebP input and grammar/reasoning-budget repair. Retained Haswell/Polaris/MOSS/readback ports; AVX2 + O3 + LTO, no PGO.
- Startup, visible dated window, GET APIs and unchanged user profiles verified. [PR/fork review and limits](performance/daily-review.json): 24 configured sources, 13 active forks; one unavailable fork returned 404. No inference benchmarks.
- [All retained patches and licenses](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/concedo_experimental/performance/PATCHES.md). No new token-speed claim; dated stock/latest evidence stays in README/results.
- Run the EXE directly. Models are not included. [Media pipeline](https://github.com/aktalascloud/koboldcpp-media-pipeline).
