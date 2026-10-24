# Reproduce the Windows build

Use the full tagged source on a Haswell-compatible AVX2/FMA/F16C CPU. Install:
- WinLibs GCC13.2 MSVCRT POSIX SEH with LTO: performance/toolchain/winlibs/mingw64.
- Make/shell from w64devkit: performance/toolchain/stable/w64devkit.
- Python dependencies from requirements-build.txt. Upstream Vulkan glslc/headers are included.

Run: python performance/build.py --jobs 4.

The policy in cpu-policy.json selects O3/Haswell/AVX2 + LTO, without PGO. CPU tiled flags and Vulkan host flags are separate. Shaders compile separately with at most four workers; backend LTO links run sequentially. Objects rebuild so no instrumented/stale flags leak into the link.

Preserve explicit C ABI exports, static runtimes, -mstackrealign, no-AVX expose.o, always-inline AVX exp/silu and shader dependency invalidation. No blanket fast-math, reduced precision, AVX512/VNNI assumptions or driver/BIOS/clock changes.

merge_upstream.py merges only official LostRuins/koboldcpp concedo_experimental into patched history. Resolve conflicts by meaning; never force-push or replace with stable.

Daily maintenance builds, deploys, checks source/SHA/registered startup/visible window/GET APIs, then publishes. Benchmarks and media generation are manual only. New unmeasured releases do not inherit old numerical claims; the last measured proof keeps its SHA/date. Public comparisons show only original stock and latest achieved result.

The EXE launches directly through GUI/CLI. No models are bundled. Window titles show the packaged build day; API version is unchanged.
