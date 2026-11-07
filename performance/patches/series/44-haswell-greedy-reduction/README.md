# Haswell greedy selection and candidate-copy bypass

Uses independent maxima with first-index ties and the existing NaN behavior; bare greedy retains compact backend token IDs and one accept callback. Measured sampler-operation benefit only.

[Source](https://github.com/ggml-org/llama.cpp/pull/30659)

Validation · 2026-11-07, source for Native `5e7c8de54808cef1173cb4f0fef25ad6024b42ecefdee7e51ed2339732f69736`:898 contract cases pass, including ties/NaNs, compact IDs and one accept callback. The production `llama.o` recipe is baseline x64 with Haswell tuning; the gate is x64, not AVX2-only. Size0/1 preserves the stock no-read return. Minimal one-child chain,151936 raw candidates: 302.11→286.27µs. This is not the complete request/filter chain or model tokens/s. The singleton wrapper retains about1.6ns dispatch overhead; all controls and the superseded AVX2-only trial are preserved.

[All repeats, flags and limits](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/v1.125-20261107-5e7c8de5/results/operations-2026-11-07-r2.json)
