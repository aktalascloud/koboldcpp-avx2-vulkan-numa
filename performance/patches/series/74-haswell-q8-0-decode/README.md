# Decode Q8_0 in eight-value AVX2 groups

Uses direct signed byte widening for CPU dequantization with the original F16 scale and F32 multiply. Maintains the existing scalar path on other targets.
