# Reuse YuE2 NAR midpoint and velocity buffers

Reuses host state, midpoint and readback allocations across fixed-shape ODE steps. The synchronous graph call already waits for completion. The second synchronization is omitted and the solver expressions are unchanged.
