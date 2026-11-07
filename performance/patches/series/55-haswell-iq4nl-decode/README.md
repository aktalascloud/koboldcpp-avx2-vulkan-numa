# Exact AVX2 IQ4_NL row decoding

Adapts Strata split-half table shuffles to eight Haswell lanes in the shared GGML row decoder. Preserves the scale multiply and scalar fallback; operator-only evidence, not a whole-model claim.

[Source](https://github.com/aktalascloud/strata-legacy-gpu/tree/157f5810bf0458b940521baf59f32fd2f71345c2/p40/patches/avx1-ple-iq4nl-shuffle)
