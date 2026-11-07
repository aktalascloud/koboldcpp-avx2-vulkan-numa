# Fuse the first three Haswell FWHT stages

Own AVX2 lane-shuffle butterflies reduce three independent low-stage memory passes to one. Preserves add/sub dependency order,normalization and larger SIMD stages;small or extreme/nonfinite rows keep the original scalar-stage path. No new ISA or settings.
