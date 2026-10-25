# Working patches

Exact diff: patches/runtime.patch. Current source/binary identities and flags: provenance.json. The checkout already includes the patch; do not apply it twice.

| Change | Purpose |
|---|---|
| O3/Haswell/AVX2 + LTO, no PGO | Strict-math build; tiled AVX2 enabled, selected backend flags recorded in policy. |
| Win64 C ABI/stack fixes | Explicit exports, static runtimes, no-AVX FFI boundary, always-inline AVX exp/silu. |
| IQ4_XS rows4/WG64 | Faster single-column Polaris matvec without changing weights/precision. |
| GCN32KiB/proprietary-driver gates | Enable the measured GCN path while retaining real device limits. |
| Batched submits, FP32 GCN64/K16 | Bound submission overhead; retain computation watchdog. Graphics queue is an upstream option. |
| Grammar-cache lifetime | Store indices, not caller-owned pointers; dense direct lookup/linear sparse fallback. |
| Shader invalidation and four-worker cap | Avoid stale SPIR-V and excessive build load. |
| Windows NUMA / CPU Sets / groups | Worker placement and restored affinity; no claim of physical dual-socket gain or changed page allocation. |
| Opt-in HybridMoE / SLRU | CPU misses/GPU hot experts; small-batch fix, new norm/rescale guards, off for dense media. |
| Soprano include placement | Build repair for fresh upstream; no Soprano quality/speed claim. |
| Reusable token-sampling CDF | Avoid per-token allocations; retain GCC 13 RNG draws and probability normalization. |
| Large transposed concat | Bit-preserving32-bit shared-tile path; small/general views retain upstream fallback. |
| Build day in window title | Embedded day, not launch day; API version unchanged. |

[Original stock versus latest measured result](benchmarks/latest.json): all repeats, ranges, settings and SHA. Individual compiler/source percentages are not additive and are not invented.

[Research and attribution](RESEARCH.md) · [Pinned sources](sources.json) · [Licenses](NOTICE.md).
