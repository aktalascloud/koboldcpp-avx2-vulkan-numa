Haswell / Polaris diff follow-up · 2026-11-07

- Official experimental SOURCE_COMMIT. Native SHA-256 NATIVE_SHA256. O3/Haswell + LTO.
- AVX2 GELU and GEGLU, Vulkan RMSNorm beyond the X workgroup limit, and preserved buffer-split view tails. [43 separate patches](performance/patches/series/README.md).
- 3,469 additional PR diffs retrieved from the earlier metadata queue. [Decisions and coverage](performance/review-2026-11-07-r1.md).
- Warm Normal 146.334 → 146.483 s. Separate 4× 5.499 → 5.498 s. All raw, finished and upscaled RGB hashes matched. [Measurements and scope](results/performance-2026-11-07-r1.json).
- User settings, exact derived profile bytes, visible startup and API readiness verified. [Licenses](performance/NOTICE.md).
