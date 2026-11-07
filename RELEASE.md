Haswell/Polaris deferred-PR update ·2026-11-07

- Official SOURCE_COMMIT;Native SHA256 NATIVE_SHA256.70 separate patches,5 new adaptations/own optimizations;O3/Haswell+LTO,no PGO.
- Normal+separate4× phase medians **185.657→153.743s** vs official rolling1.125. Observed time difference vs preceding efddce0e **-0.119s**,within overlapping phase ranges;no incremental Normal speedup demonstrated. All9 Normal and6 SR RGB outputs match;settings preserved. [All repeats,ranges,hashes and limits](results/performance-2026-11-07-r5.json).
- Phase sum is not one API request;model load/return/public queue/transport excluded. Fixed order,small warm sample;shared SR code. [Operator gains and rejected FP16/Q5 experiments](results/operations-2026-11-07-r6.json) are separate from full generation.
- [12 deeper deferred-PR follow-ups and remaining validation limits](results/pr-review-2026-11-07-r5.json);not a whole-repository line-by-line audit. F32 YaRN,duplicate-safe packed TopK,typed Hadamard12/20,own AVX2 low stages and wide RMS cache. [70-patch catalog](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/tree/concedo_experimental/performance/patches/series).
- Existing Pillow13/runtime/model assets and licensing retained;no main or non-commercial weights mirrored. No new High/full-LLM token benchmark.
