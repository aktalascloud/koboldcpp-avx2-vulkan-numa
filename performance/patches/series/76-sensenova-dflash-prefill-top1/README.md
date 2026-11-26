# DFlash prefill and Vulkan top-one

Avoids unused target LM-head outputs during DFlash prompt evaluation. Ports the single-candidate part of the author's patch 0009 at zero confidence cutoff. Vulkan reduces one candidate through existing TopK buffers, using float comparisons on the non-subgroup path.

[Source](https://huggingface.co/satellitedown/SenseNova-Loop-9B-abliterated-GGUF/blob/2644d93673251404271a0bdf22a4a4436c0d0a54/patches/0009-ggml-metal-top_k-of-one-value-as-an-argmax-speculati.patch)
