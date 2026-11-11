# Reuse batched YuE2 logits across CFG decode steps

Reads both guidance rows from one reusable batched result and retains the same floating-point guidance expression, input tensors, graph, cache update and sampler.
