Haswell / Polaris completed-source review · 2026-11-07

- Official SOURCE_COMMIT; Native SHA-256 NATIVE_SHA256. O3/Haswell + LTO, no PGO.
- [46 separate described patches](performance/patches/series/README.md); corrected baseline-x64 sampler activation, exact CPU RMSNorm row scheduling and Windows working-set cleanup.
- [29736 remaining diffs read, 76 unavailable, zero pending](performance/review-2026-11-07-r2.md). Full-text screen plus scoped semantic reviews.
- Normal warm Native 146.483 s → 146.645 s; separate4× API 5.498 s → 5.475 s. RGB exact. [All measurements/limits](results/performance-2026-11-07-r2.json) · [Operation results and excluded trials](results/operations-2026-11-07-r2.json).
- Exact profiles/settings, dated visible startup and API readiness verified. [Licenses](performance/NOTICE.md).
