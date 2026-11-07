# Spread short CPU RMSNorm rows across heads and batches

Adapts the few-row scheduling idea while retaining the original double sum, scaling, fusion and strides. The ordinary full-worker row assignment is unchanged.

[Source](https://github.com/ikawrakow/ik_llama.cpp/pull/2457)

Validation · 2026-11-07, source for Native `5e7c8de54808cef1173cb4f0fef25ad6024b42ecefdee7e51ed2339732f69736`:144 real-function cases pass bit for bit, including padding, fusion, broadcast weights and1/4/16 workers. Example4096×1×16×2,16 workers: 141.17→15.23µs. Full-worker controls keep their row assignment and retain their observed small negative/positive timing variation. No whole-model speed claim follows from this operation test.

[All repeats, flags and limits](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/v1.125-20261107-5e7c8de5/results/operations-2026-11-07-r2.json)
