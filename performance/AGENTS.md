# Maintainer and agent rules

Read BUILD.md, PATCHES.md, provenance.json, benchmarks/results.json before changing code. Merge the official concedo_experimental branch, never silently replace it with stable. Public publishing is authorized for this project, after validation, secret scan and license checks.

Keep O3/Haswell/AVX2, PGO+LTO and the measured hot flags without lowering precision. Max4 compile workers, shader stage isolated, sequential LTO links. No inference benchmarks while compiling. Do not change drivers/BIOS/TDR/clocks/voltages. Preserve C ABI/stack/inline AVX fixes and shader dependency invalidation. Fresh source/root/compiler requires fresh PGO, including vec source fingerprints; stale counters and suppressed missing-profile failures are forbidden.

Only enable changes that produce correct answers and measured gains on the same control. Retain small measured gains and negative controls; report ranges. Existing upstream graphics queue is not a local patch. GPU sampling/fusion and separate draft model lost locally; do not restore automatically. HybridMoE remains opt-in; no expert cache on dense Qwen. NUMA auto; physical dual-socket unmeasured. Native HIP photo failed; do not ship as production.

Never publish models, credentials, local IP addresses, user paths, runtime cache/log/PID/state, PGO counters, toolchains or raw private diagnostic dumps. Preserve upstream licenses/attribution; distribute matching source for binaries. Every release binds source SHA, exact native patch, EXE/DLL hashes and validation. Publishing a rebuilt binary under old measurements as if newly tested is forbidden.

Native releases launch koboldcpp-nocuda.exe directly through the standard GUI or CLI. Do not ship PowerShell launchers; document threads/batch/ubatch and environment recommendations in README/RUN.txt. Hardware keywords are AMD RX580 Polaris/gfx803 and Haswell/AVX2. Preserve links to the original KoboldCpp and the media pipeline, clear CPU/GPU before-after tables and photo seconds.
