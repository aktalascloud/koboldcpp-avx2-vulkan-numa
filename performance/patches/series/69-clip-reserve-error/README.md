# Propagate CLIP compute allocation failure

Raises the existing model initialization error when reservation fails, before reading sizes or accessing buffers. Covers automatic flash attention, the fallback attempt and explicit modes.

[Source](https://github.com/ggml-org/llama.cpp/pull/31988)
