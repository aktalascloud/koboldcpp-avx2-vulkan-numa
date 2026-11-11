# Cache matching fallback workspace measurements

Keeps the existing direct-allocation fast path and caches only repeated fallback scheduler measurements using full graph, topology, use-count and external-placement metadata. Multi-device runners stay uncached; release invalidates the cache. No arithmetic changes.

[Source](https://github.com/leejet/stable-diffusion.cpp/pull/2251)
