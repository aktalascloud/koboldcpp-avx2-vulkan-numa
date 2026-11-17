# Scale and copy wide F32 rows in one AVX2 pass

Uses one load, one multiplication and one store for wide vector-aligned, out-of-place, zero-bias F32 Scale. Short or unaligned widths, in-place, biased and other backend paths keep their existing kernels.
