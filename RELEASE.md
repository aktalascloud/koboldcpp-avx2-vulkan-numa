Haswell / Polaris maintenance · 2026-11-07

- Fresh official experimental source SOURCE_COMMIT. Native SHA-256 NATIVE_SHA256. O3/Haswell + LTO, monolithic build and preserved ancestry.
- Vulkan main-VRAM allocation policy for non-ReBAR devices, parallel CPU Softmax rows and checked allocation arithmetic. [40 patch descriptions](performance/patches/series/README.md) · [Two-month review](performance/review-2026-11-07.md).
- 20 bit-exact Softmax cases and 10 allocator cases. RX 580 confirmed separate 256 MiB BAR and the new startup policy. [Operation data](results/operations-2026-11-07.json).
- Cold startup, visible window, GET endpoints and exact derived profile bytes verified. [November 6 photo timings](results/performance-2026-11-06.json) retain their original SHA.
