# Windows build
Prerequisites: Git checkout of this fork, Windows64, Python3.12 x64 (`py -3.12`), internet for first setup, Haswell/AVX2 CPU. No models required.

```powershell
py -3.12 performance/setup.py --build
```

setup.py downloads original WinLibsGCC13.2r8/MSVCRT and w64devkit1.22.0 after SHA256 verification, creates performance/.venv, installs requirements-build.txt and runs build.py --jobs 4. Subsequent offline builds: performance/.venv/Scripts/python.exe performance/build.py --jobs 4.

Output: performance/candidate/koboldcpp-nocuda.exe. Build receipts: performance/results (local, ignored). Do not commit toolchains/build caches. CPU/Vulkan both use O3/Haswell/LTO withoutPGO; full backend LTO requires upstream KCPP_COMMONIZE=0. Explicit exports and static runtimes stay enabled. Shader changes invalidate embedded objects. Max4 workers, links sequential. Full guardrails: ../AGENTS.md. Successful compilation is not a speed/quality benchmark.
