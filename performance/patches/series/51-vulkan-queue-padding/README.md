# Deterministic Vulkan queue descriptors

Zeroes the full flagged queue descriptor including padding and uses the ordinary queue API when no queue flags are needed. Preserves queue selection and synchronization policy.

[Source](https://github.com/ggml-org/ggml/pull/1672)
