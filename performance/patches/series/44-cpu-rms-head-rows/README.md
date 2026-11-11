# Spread short CPU RMSNorm rows across heads and batches

Adapts the few-row scheduling idea while retaining the original double sum, scaling, fusion and strides. The ordinary full-worker row assignment is unchanged.

[Source](https://github.com/ikawrakow/ik_llama.cpp/pull/2457)
