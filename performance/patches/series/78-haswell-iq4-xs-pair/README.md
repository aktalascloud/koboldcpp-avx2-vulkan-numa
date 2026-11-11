# Decode two IQ4_XS subblocks together with AVX2

Uses one 256-bit load and two byte table shuffles per pair. Restores the original nibble and subblock lane ordering before the same integer and floating-point reductions.
