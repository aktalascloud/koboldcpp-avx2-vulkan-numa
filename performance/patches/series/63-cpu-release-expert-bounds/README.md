# Check CPU expert indices in release builds

Adapts the untrusted expert-index boundary principle to the generic GGML CPU MoE workspace. Replaces an NDEBUG-disabled assert with GGML_ASSERT before indexing, preserving valid expert IDs and the existing minus-one skip. Repack already has the permanent check.

[Source](https://github.com/vllm-project/vllm/pull/65929)
