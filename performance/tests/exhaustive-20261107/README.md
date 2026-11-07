# Manual isolated controls · 7 November 2026

These fixed before/after functions come from the recorded official source plus the retained patch stack. Run from the repository root with the pinned MinGW g++ toolchain. No models are downloaded and no model inference occurs. Test-process affinity uses physical cores first, then SMT siblings; runtime/user settings are untouched.

The standard sampler belongs to baseline-x64 `llama.o`. **Do not add `-march=haswell` to its fixture**: an explicit compiler check prevents accidentally testing a different ISA. RMSNorm belongs to a Haswell/AVX2 compute object. All builds use O3, strict math and LTO, never PGO/fast-math.

```powershell
g++ -std=c++17 -O3 -fno-finite-math-only -mtune=haswell -mstackrealign -fno-math-errno -flto=4 -static performance/tests/exhaustive-20261107/greedy-check.cpp -o greedy-check.exe
./greedy-check.exe --bench
./greedy-check.exe --small

g++ -std=c++17 -O3 -fno-finite-math-only -march=haswell -mtune=haswell -mstackrealign -fno-math-errno -flto=4 -static -Iggml/include -Iggml/src -Iggml/src/ggml-cpu performance/tests/exhaustive-20261107/cpu-rms-check.cpp -o cpu-rms-check.exe
./cpu-rms-check.exe --bench

g++ -std=c++17 -O3 -mtune=haswell -mstackrealign -flto=4 -static performance/tests/exhaustive-20261107/windows-release-check.cpp -lpsapi -o windows-release-check.exe
./windows-release-check.exe
```

The complete local recipe also retains `-frename-registers -falign-functions=32 -falign-loops=32 -fno-ipa-cp-clone`; use those flags for reproducing the recorded timings. At most4 compiler/LTO workers and no inference benchmark alongside compilation.

The greedy chain fixture has one greedy child, with no temperature/top-k/grammar filters. It does not simulate the complete request chain. The mapper/accept and NaN/tie controls pass898 cases. RMSNorm covers144 cases with padding, fusion, broadcast weights and1/4/16 workers; its double sum is unchanged. The Win32 fixture creates its own64MiB file with CREATE_NEW and DELETE_ON_CLOSE, checks resident pages and all mapped bytes, and preserves boundary pages. An existing file with the fixture name is never overwritten.

[Measured samples, ordinary full-worker controls and rejected/preliminary trials](../../../results/operations-2026-11-07-r2.json). Operation latency and working-set reductions are not model tokens/s or whole-photo gains.
