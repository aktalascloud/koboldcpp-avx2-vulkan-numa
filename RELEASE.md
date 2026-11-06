Haswell / Polaris maintenance · 2026-11-06

- Official KoboldCpp experimental 1.125 at `SOURCE_COMMIT`. Native SHA-256 `NATIVE_SHA256`. Ancestry preserved across 79 official commits.
- Bit-exact operation tests retained Haswell Q8_0 GEMM and AVX2 Q5_K/Q6_K/IQ3, indexed KV rollback, buffered GGUF metadata, Polaris Q4_K prefill and Vulkan F16 value staging. [Individual patches](performance/patches/series/README.md) · [Two-month source review](performance/review-2026-11-06.md).
- Warm Native Normal 146.905 s → 146.391 s. 4× public API 5.715 s → 5.482 s. All raw, finished and upscaled RGB samples matched exactly. [All repeats and scope](results/performance-2026-11-06.json).
- Windows startup, visible window, profile hashes and API verified. [Licenses and corresponding source](performance/NOTICE.md).
