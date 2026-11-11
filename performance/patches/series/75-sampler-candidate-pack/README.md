# Pack four sampling candidates per host iteration

Uses SSE2 to write the existing token ID, F32 logit and zero probability layout. Retains scalar tails, a non-x86 fallback, and the original sampler order and RNG.
