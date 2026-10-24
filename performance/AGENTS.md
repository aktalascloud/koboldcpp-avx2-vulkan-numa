# Maintainer and agent rules

Read BUILD.md, PATCHES.md, provenance.json and benchmarks/current.json before changing code. Merge official **concedo_experimental**, preserving patch history. Public publishing is authorized after validation, secret and license checks.

The current policy, updated on 24 October 2026, requires **O3/Haswell/AVX2 + LTO without PGO**. Do not train or apply profiles for the release recipe. Keep register renaming, alignment 32 and hot no-clone according to cpu-policy.json. Rebuild objects after a flag change; no `-fprofile-*` or GCOV may enter the release link. Old PGO results are historical evidence only.

Use at most four compile workers, isolate shader compilation and run LTO links sequentially. Do not benchmark while compiling. Preserve C ABI/stack/inline AVX fixes and shader invalidation. Do not change drivers, BIOS, TDR, clocks, voltages, model precision or weights.

Run speed or generation tests only on explicit manual user request. For those tests, compare stock/previous/new with identical settings; keep all controls, negative runs and ranges. The maintainer requires a fresh functional build to be installed and published even if throughput is lower. Store the actual performance verdict and the authorized release policy separately. Never bypass correctness, API, media, ownership or source/SHA checks. New experimental optimizations still need demonstrated benefit before enabling them.

Graphics queue is an upstream setting. GPU sampling, MoE fusion and a separate draft model lost locally; do not restore automatically. HybridMoE remains opt-in, off for dense media. NUMA auto is selected on this single-node host; physical dual-socket performance is unmeasured. Native HIP photo failed and is not a production backend.

Never publish main model weights, credentials, private addresses/user paths, runtime cache/log/PID/state, PGO counters, toolchains or private diagnostic dumps. Preserve licenses and attribution, and distribute corresponding source for the binary. Bind each release to its source, patch, EXE/DLL hashes and actual acceptance evidence.

Native users launch the EXE directly through the normal GUI or CLI. Do not add PowerShell launchers. Document threads, batch/ubatch and environment settings. Use RX 580 Polaris/gfx803 and Haswell/AVX2 keywords, English README by default, Russian as the second version, links to the original KoboldCpp/media pipeline, and honest CPU/GPU/photo comparisons.

Daily automation builds, installs, verifies startup/API availability and publishes. It does not generate photos/music or run performance benchmarks. New SHA-bound startup evidence is required; previous manual measurements retain their original date and executable SHA.
