# Vectorize aligned FP32 Scale on Polaris

Loads and stores four adjacent F32 values per invocation. Keeps the tested scale and bias expression, uses real Polaris IDs and alignment, and leaves strided, small, odd, unaligned and nonfinite-parameter tensors on the scalar path.
