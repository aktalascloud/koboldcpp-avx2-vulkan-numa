# Reject GGUF shapes before constructor overflow asserts

Matches the constructor product bound even after zero axes and returns a parse error for malformed dimensions.

[Source](https://github.com/ggml-org/ggml/pull/1673)
