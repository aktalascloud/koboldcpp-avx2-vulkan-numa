# Qualification sources ·2026-11-07

Link deferred-check.cpp or rms-native-check.cpp against the actual CPU/Vulkan GGML objects with Haswell O3/LTO and static runtimes. Include ggml/include,ggml/src,ggml/src/ggml-cpu and Vulkan headers;link vulkan-1,ws2_32,thread runtimes. Run only on an idle owned device,with the unchanged configured Vulkan environment and no compilers. Operator fixtures are synthetic,not full-model tests. Hashes,run receipts,all timings and initial rejected FP16 evidence are in results/operations-2026-11-07-r6.json. No fixture executables/caches are mirrored.

old-fwht.inc comes from the recorded official Kobold/GGML base (MIT components);H12/H20 use llama.cpp28490 with MIT attribution. q5-widen-contract.cpp adapts Manogna-Sree/llama.cpp19707 at its pinned original head,under upstream GGML MIT. Its private widening experiment is not part of Native production dispatch. Core source/third-party notices retain the required licenses.
