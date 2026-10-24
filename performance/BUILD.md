# Reproduce and maintain

The current recipe is **O3 / Haswell / AVX2 + LTO, without PGO**. It retains register renaming, 32-byte function/loop alignment, hot `-fno-ipa-cp-clone`, static runtimes and the Windows ABI fixes. A training model is not required.

Use the full tagged source on Windows x64 with a Haswell-compatible AVX2/FMA/F16C CPU. Install WinLibs GCC 13.2 MSVCRT POSIX SEH **with LTO** under `performance/toolchain/winlibs/mingw64`, and GNU Make/shell from w64devkit under `performance/toolchain/stable/w64devkit`. The latter's own GCC disables LTO and is not the selected compiler. Vulkan glslc and headers are the upstream pinned ones. Install the Python build dependencies from `performance/requirements-build.txt`.

Run:

```text
python performance/build.py --jobs 4
```

The build reads `performance/cpu-policy.json`, regenerates shaders with at most four workers, and links Vulkan and CPU backends sequentially with LTO. It rebuilds native objects so old instrumented/profile-guided objects cannot enter the new link. Current compile/link commands must contain no `-fprofile-*` or GCOV runtime. Historical PGO experiments and their dated results are retained for comparison; they are not the release recipe.

CPU-only hot-unit flags are selected by `cpu_hot_variant` in that policy and defined in `cpu_flag_options.py`. The Vulkan backend retains its own baseline recipe. Shared object files are rebuilt between backend links to prevent one backend's flags leaking into the other.

Packaging embeds the actual build day in `embd_res/kcpp_build.json`. The GUI and terminal titles display it as `[YYYY-MM-DD]`; the API version remains unchanged. Launching an older build on a later day does not change its displayed build date.

Preserve explicit C ABI exports, static runtimes, stack realignment, no-AVX `expose.o`, always-inline AVX exp/silu helpers and shader dependency invalidation. Do not enable blanket fast-math, lower precision or change drivers, BIOS, TDR, voltage or clocks. Do not benchmark inference while compiling.

`python performance/merge_upstream.py` fetches only LostRuins/koboldcpp **concedo_experimental**, then merges the exact official commit into the patched branch. Resolve conflicts by meaning, preserve upstream fixes and local invariants, and never force-push or substitute the stable branch.

Only when the user explicitly requests manual benchmarks, compare stock, the previous working EXE and the new EXE with the same model and thread settings. Retain every run and range, including negative results. For a manual validation session, check real Russian/English/arithmetic answers, JSON grammar, batch boundaries and the same fixed-seed photo pipeline. Keep cold/warm and raw/full-pipeline timings separate. Verify startup, the visible Native window, APIs and model switching after deployment.

The maintainer's current policy prioritizes a fresh functional experimental build even when measured throughput is lower. Report that difference; never relabel a negative performance verdict as a pass. Correctness, source/SHA, secret and license checks remain mandatory. A new EXE must have its own matching evidence.

Publish the matching source, exact patch, EXE and checksums together. End users run `koboldcpp-nocuda.exe` directly through its GUI or CLI. The separate media repository supplies photo/music routing, face processing and upscaling.

Daily automation builds, installs, verifies startup/API availability and publishes. It does not generate photos/music or run performance benchmarks. New SHA-bound startup evidence is required; previous manual measurements retain their original date and executable SHA.
