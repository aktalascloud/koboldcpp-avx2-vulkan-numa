# Build rules
1. Work in this checkout. Read README.md and performance/BUILD.md. Required host: Windows x64, Python3.12 x64, Haswell/AVX2.
2. One command: py -3.12 performance/setup.py --build. Do not invent paths or copy another runtime. Compiler pins are in setup.py.
3. Official upstream: LostRuins/koboldcpp concedo_experimental only. performance/merge_upstream.py preserves history; no force push/stable replacement. Resolve conflicts semantically.
4. O3/Haswell/AVX2/FMA/F16C/BMI and LTO required. NO PGO/GCOV/fast-math/AVX512/VNNI. Max4 compilers; shader stage separately, LTO links sequentially.
5. Keep KCPP_COMMONIZE=0, explicit C ABI, static runtimes, -mstackrealign, no-AVX expose.o and inline AVX exp/silu. Changed source/headers/shaders/flags invalidate cached objects/SPIR-V.
6. CPU tiled-haswell and Vulkan baseline object policies must not mix. Do not mask missing symbols or silently change precision.
7. NUMAauto uses Windows CPU Sets/groups and restores affinity, not page allocation. Dual-socket is unmeasured. HybridMoE/SLRU is opt-in, dense off; keep routing/normalization/MTP/guidance exclusions and CRT bridge.
8. GPU sampling/fusion/draft/HIP were rejected; do not reenable without a human-requested study. Keep bounds and fallback on new ports.
9. No generation/speed benchmarks during routine builds. Basic checks: source/hash, LTO/noPGO, DLL symbols, startup/window/GET APIs. Historical results keep their own date/SHA.
10. Preserve licenses/credits/corresponding source. No models, credentials, personal addresses, cache/log/state in Git or releases. Only original stock and last measured performance on the front page.
