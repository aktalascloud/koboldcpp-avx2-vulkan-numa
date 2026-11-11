# Allocate graph metadata without eagerly zeroing unused pages

Reuses the allocation at the same capacity. GGML still initializes each tensor and graph object as it is allocated.

[Source](https://github.com/ggml-org/llama.cpp/pull/32043)
