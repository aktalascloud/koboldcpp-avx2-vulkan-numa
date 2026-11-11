# Preserve the first assigned backend for shared graph nodes

Uses map insertion so a later segment cannot overwrite an existing shared-node assignment. Bounds/null checks and configured device lists remain intact. Physical multi-GPU performance is unmeasured.

[Source](https://github.com/leejet/stable-diffusion.cpp/pull/2262)
