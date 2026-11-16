# Shorten the AVX2 Q8_0 dot guard

Uses a byte movemask and byte absolute value on the regular Q8_0 path. The widening path for the full int8 range and the existing float accumulation order are preserved.
