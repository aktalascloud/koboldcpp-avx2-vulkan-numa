# Materialize deep RPC graphs iteratively

Uses two passes and the existing tensor map as the visited set, removing recursion without a redundant hash set. Shared nodes, views and existing memo entries keep their semantics.

[Source](https://github.com/ggml-org/llama.cpp/pull/31843)
